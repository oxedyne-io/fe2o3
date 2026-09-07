//! `austenite` -- compile a Typst document to a set of pages.
//!
//! Reads a Typst root, follows its `#include` chain through the [`book`](oxedyne_fe2o3_austenite::book)
//! assembler (or, for a lone file, straight through the [`lang`](oxedyne_fe2o3_austenite::lang) reader),
//! authors the block stream through the block layer, runs the two-pass driver to a fixed point, decorates
//! each page with a running head and a folio, and writes every page as SVG alongside the resolved ledger
//! and a single PDF of the whole run.
//!
//! A construct the reader cannot yet set -- a `#show` rule, a `#columns` wrapper, an unknown `#func` --
//! is passed over rather than failing the compile, and the lone-file path reports the tally on one terse
//! line so a dropped construct is visible.
//!
//! Usage: `austenite <SOURCE.typ> [OUTPUT_DIR]` (default output `austenite-out`), or
//! `austenite --watch <SOURCE.typ> [OUTPUT_DIR]` to recompile on every change to the root, its includes,
//! its `config.typ`, or its assets.

use oxedyne_fe2o3_austenite::{
	book,
	doc::{
		self,
		Heading,
		Style,
	},
	driver::{
		self,
		Config,
	},
	emit::{
		self,
		svg,
	},
	font::FontMetrics,
	ir::DrawOp,
	ledger::{
		AnchorId,
		AnchorKind,
		Ledger,
	},
	lang,
	page::{
		Frame,
		Page,
		PageGeometry,
		PlacedKind,
	},
	watch,
};

use oxedyne_fe2o3_core::prelude::*;
use oxedyne_fe2o3_font::{
	face::Role,
	shape::Dir,
};
use oxedyne_fe2o3_graphics::pdf::{
	OutlineItem,
	PdfPage,
};

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// An estimate of the extra memory rendering this page holds in flight, in bytes -- dominated by the
/// figure rasters, which [`emit::pdf::render_page`] copies out of the shared frame into the page's own
/// straight-RGB (and, when translucent, grey) buffers. A text page estimates near zero; a full-page
/// illustration estimates several megabytes. The chunker sums this across a forming chunk and closes it
/// before the sum would breach the memory budget, so an illustration-dense book self-limits its window
/// while a text book packs a chunk full. The glyph outlines are not counted: they were shown to stay
/// flat to a wide window, and are shed the moment the content stream is serialised.
fn page_hold_estimate(page: &Page) -> usize {
	// Rough bytes-per-unit for the SVG text and PDF content a page's ink expands to while it is in flight.
	// A glyph becomes an outline of a couple of dozen path operators in each of the two serialisations; a
	// figure's own path is written op for op; a raster is copied sample by sample. The constants are
	// deliberately generous -- the estimate gates concurrency, so over-counting only narrows a chunk.
	const PER_GLYPH:	usize = 800;	// one glyph's outline, in both serialisations (measured)
	const PER_SEG:		usize = 320;	// one figure path segment, across every live buffer (measured)
	const PER_SAMPLE:	usize = 8;		// RGB copy plus soft mask, with headroom

	let mut bytes = 0usize;
	for placed in &page.frame.placed {
		match &placed.kind {
			PlacedKind::Text(shaped) => {
				bytes += shaped.run().glyphs.len() * PER_GLYPH;
			},
			PlacedKind::Graphic(g) => {
				for op in &g.ops {
					match op {
						DrawOp::Fill { path, .. }	=> bytes += path.segs().len() * PER_SEG,
						DrawOp::Stroke { path, .. }	=> bytes += path.segs().len() * PER_SEG,
						DrawOp::Image { image, .. }	=> bytes += image.width * image.height * PER_SAMPLE,
					}
				}
			},
			_ => {},
		}
	}
	bytes
}

/// One page's PDF, rendered off the writer's thread: its draw list (images only, once the outlines are
/// serialised) and that list already serialised to content-stream bytes. The costly transforms and
/// serialisation are done; the sequential writer only frames these and folds them into the file in page
/// order. The page's SVG is written straight to its own file by the worker, since an SVG page is an
/// independent file that owes nothing to page order, so its string never travels back or accumulates.
struct Prepared {
	pdf:		PdfPage,
	content:	Vec<u8>,
}

/// The result of a compile, for the caller to report: the page count, the number of driver passes to
/// the fixed point, the count of anchors in the resolved ledger, and the terse skip line (or `None`
/// when nothing was skipped).
struct CompileStats {
	pages:		usize,
	passes:		u32,
	anchors:	usize,
	skip_line:	Option<String>,
}

/// Renders one page to both artefacts, the pure work a chunk runs across the cores. The SVG is written
/// to its file here and dropped; the PDF content stream is serialised here too -- the bulk of the cost --
/// and returned for the ordered writer to frame.
fn render_page_pair(page: &Page, out_dir: &str) -> Outcome<Prepared> {
	let svg			= res!(svg::render_page(page));
	let path		= fmt!("{}/page-{:03}.svg", out_dir, page.number);
	res!(std::fs::write(&path, &svg));
	drop(svg);

	let mut pdf		= res!(emit::pdf::render_page(page));
	let content		= pdf.content_bytes();
	// The glyph outlines are now serialised into `content`; free them so a chunk rendered ahead holds
	// only its images and content bytes, not every page's paths at once. Peak memory stays flat.
	pdf.shed_serialised_draws();
	Ok(Prepared { pdf, content })
}

/// The one terse skip line -- `skipped: #show ×2, #columns ×1` -- built from the summary's per-name
/// counts, or `None` when the reader set everything it met. Ordered by the summary (descending count,
/// then name), so the line leads with the construct that cost the most.
fn terse_skip_line(skips: &lang::SkipSummary) -> Option<String> {
	if skips.is_empty() {
		return None;
	}
	let parts: Vec<String> = skips.entries().into_iter()
		.map(|(n, c)| fmt!("{} ×{}", n, c))
		.collect();
	Some(fmt!("skipped: {}", parts.join(", ")))
}

/// Builds the PDF document outline (the viewer's bookmark side panel) from the resolved ledger: the
/// three front-matter leaves first -- title page, meta (imprint) page and contents -- then every body
/// heading in reading order. The front-matter pages carry no heading of their own, so the block layer
/// records a `Label` anchor at the top of each (`frontmatter:title`, `frontmatter:meta`,
/// `frontmatter:contents`); this reads their page back from the ledger. A leaf the book omits sets no
/// anchor, so its entry is simply absent. Body headings resolve their page through the heading anchor,
/// and their depth matches the contents list -- a chapter or a part at the top, deeper headings nested
/// under it. Pages are zero-based, as [`OutlineItem`] wants; the ledger stores them one-based.
fn build_outline(heads: &[Heading], ledger: &Ledger) -> Vec<OutlineItem> {
	let mut items: Vec<OutlineItem> = Vec::new();

	// The front matter, at the top and at depth zero, so it stands as a sibling of the first body level.
	let front = [
		("frontmatter:title",		"Title"),
		("frontmatter:meta",		"Meta"),
		("frontmatter:contents",	"Contents"),
	];
	for (key, label) in front {
		let id = AnchorId::new(AnchorKind::Label, key);
		if let Some(page) = ledger.page_of(&id) {
			items.push(OutlineItem { title: label.to_string(), page: (page - 1) as usize, level: 0 });
		}
	}

	// Every body heading, its depth the contents indent: a chapter or a part at depth zero, a `==`
	// section at one, and so on. A heading the ledger has not fixed is skipped rather than guessed.
	for h in heads {
		if let Some(page) = ledger.page_of(&h.id) {
			let level = (h.level.max(1) - 1) as u8;
			items.push(OutlineItem { title: h.title.clone(), page: (page - 1) as usize, level });
		}
	}
	items
}

/// Compiles the Typst root at `source` into `out_dir`, writing every page's SVG, the resolved ledger,
/// and one PDF of the whole run. Returns the counts and the terse skip line for the caller to report;
/// prints nothing itself save the phase profile when `AUS_PROFILE` is set.
fn compile(source: &str, out_dir: &str) -> Outcome<CompileStats> {
	// Phase timing, gated on AUS_PROFILE so a normal run is untouched. Each phase reports its wall time
	// to stderr, leaving stdout (and every emitted byte) exactly as it was.
	let prof = std::env::var("AUS_PROFILE").is_ok();
	let mark = |label: &str, t: std::time::Instant| {
		if prof {
			eprintln!("[profile] {:<22} {:>8.1} ms", label, t.elapsed().as_secs_f64() * 1000.0);
		}
	};
	let t_all = std::time::Instant::now();

	let src = match std::fs::read_to_string(source) {
		Ok(s)	=> s,
		Err(e)	=> return Err(err!(e,
			"Could not read the source file {:?}.", source; File, Read)),
	};

	// A figure's `/assets/...` image path is root-relative in Typst, not filesystem-absolute; the image
	// loader resolves it against this directory and, failing that, its ancestors, so a chapter compiled
	// on its own finds the shared assets through the book's `assets` entry just as a whole book does.
	if let Some(dir) = std::path::Path::new(source).parent() {
		res!(oxedyne_fe2o3_austenite::image::set_base_dir(dir.to_path_buf()));
	}

	// A book root assembles chapters and carries its own geometry, fonts and type; a lone file sets on
	// A4 with the embedded Libertinus, as before. The block stream, geometry, style and faces come from
	// one place or the other, and the rest of the run is identical.
	let t_parse = std::time::Instant::now();
	// The terse skip line, set from whichever path assembles the source: a book or doc root through
	// `book::load`'s merged tally, a lone file through its own reader summary.
	let skip_line: Option<String>;
	let (blocks, fonts, geom, style, title, heading, front, bib) = if book::is_book_root(&src) {
		// A book or doc root assembles its chapters through the reader and merges each chapter's skip tally
		// into one summary, so a whole-book or whole-doc compile reports its skipped constructs on the same
		// terse line the lone-file path prints.
		let spec = res!(book::load(std::path::Path::new(source)));
		skip_line = terse_skip_line(&spec.skips);
		(spec.blocks, spec.fonts, spec.geom, spec.style, spec.title, spec.heading, Some(spec.front), spec.bib)
	} else {
		// A lone chapter installs the shared `term-dict` from a `terms.typ` beside or above it, so its
		// `#t`/`#g` term calls resolve to their values just as in a whole-book compile.
		if let Some(dir) = std::path::Path::new(source).parent() {
			res!(book::install_term_dict(dir));
		}
		let (mut blocks, skips)	= res!(lang::to_blocks_with_skips(&src));
		skip_line = terse_skip_line(&skips);
		// Resolve citations against a `refs.bib` found beside or above the chapter, so a lone-file compile
		// sets Chicago author-year in text and a reference list at the end rather than the raw cite key.
		let bib		= res!(book::load_lone_bibliography(std::path::Path::new(source), &mut blocks));
		let fonts	= Arc::new(res!(oxedyne_fe2o3_austenite::fonts::libertinus()));
		(blocks, fonts, PageGeometry::a4(), Style::default(), String::new(), None, None, bib)
	};
	mark("parse+lower+fonts", t_parse);

	let t_author			= std::time::Instant::now();
	let (document, heads)	= res!(doc::author(fonts.clone(), geom, style, heading, &blocks, front.as_ref(), bib.as_ref()));
	mark("author(shape+break)", t_author);
	let metrics				= FontMetrics::new(fonts.clone(), Role::Body, Dir::Ltr, style.body_size);
	let t_run				= std::time::Instant::now();
	let mut out				= res!(driver::run(&document, &metrics, Config::default()));
	mark("driver::run", t_run);
	let t_decorate			= std::time::Instant::now();
	let footer_logo			= front.as_ref().and_then(|f| f.footer_logo.as_deref());
	res!(doc::decorate(&mut out.pages, &out.ledger, &heads, &fonts, style, geom, &title, footer_logo));
	mark("decorate", t_decorate);

	// Mirror the margins: the driver laid every page at the recto split (binding on the left). A verso
	// page -- an even folio -- is that whole frame shifted to the fore-edge, so the binding margin sits
	// at the spine on both sides of the leaf. Uniform margins give a zero shift, so a non-book run is
	// untouched.
	let shift = geom.mirror_shift();
	if shift.raw() != 0 {
		for page in &mut out.pages {
			if page.number % 2 == 0 {
				for placed in &mut page.frame.placed {
					placed.x = placed.x + shift;
				}
			}
		}
	}

	res!(std::fs::create_dir_all(out_dir));

	// The ledger is small and independent of the pages, so it is written first and out of the way.
	let ledger_path = fmt!("{}/ledger.jdat", out_dir);
	res!(out.ledger.to_file(&ledger_path));

	// Emit each page and drop its frame before the next. Both writers are streaming: the SVG is one file
	// per page, and the PDF is written object by object into the file as each page is composed, never
	// accumulated. Holding a bounded window of pages' glyph outlines -- rather than every page's at once,
	// as a buffered whole-document PDF would -- is what keeps a whole-book compile flat in memory.
	//
	// Almost the whole cost of a compile is here: turning each glyph into a filled outline and serialising
	// it, page after page. That work is a pure function of the placed frame and is independent between
	// pages, so a chunk of pages is rendered across the cores at once. The order is preserved exactly: a
	// chunk's results are written to the SVG files and folded into the single PDF stream in page order,
	// so the bytes are identical to a sequential emit -- only the wall time differs. The PDF's object
	// numbering and its running `/ID` hash stay strictly sequential in the writer, on this thread.
	let t_emit	= std::time::Instant::now();
	let mut t_render_ms	= 0.0f64;	// wall spent in the parallel render stage
	let mut t_write_ms	= 0.0f64;	// wall spent writing results out in order
	let pdf_file	= res!(File::create(fmt!("{}/document.pdf", out_dir)));
	let outline		= build_outline(&heads, &out.ledger);
	let mut pdf		= res!(emit::pdf::open_document_with_outline(
		BufWriter::new(pdf_file), out.pages.len(), outline));

	// Emit is by far the costliest phase and is embarrassingly parallel: each page's outline transforms
	// and serialisation are a pure function of its frame, independent of every other page. But a rendered
	// page is large -- its glyph outlines and figures expand to megabytes of SVG and PDF operators -- so
	// holding several at once regresses peak memory, which the engine keeps to a page-at-a-time budget.
	// For an illustration-dense book the per-page ink is heavy enough that even a pair of pages can breach
	// that budget, so parallelism there is not free.
	//
	// The default therefore opens the window to eight pages (capped at the core count), which brings a
	// text book to roughly Typst's own wall time while peak memory stays a few hundred megabytes -- far
	// under Typst's gigabytes -- and the shared glyph-outline cache speeds every path besides. The ink
	// budget below keeps the window honest: AUS_EMIT_BUDGET_MB caps the estimated page ink in flight (see
	// [`page_hold_estimate`]), so a run of heavy figure pages closes its chunk early and never all
	// coincide; a chunk always holds at least one page, so a page heavier than the budget still renders --
	// alone. A caller wanting the strict page-at-a-time floor sets AUS_EMIT_WINDOW=1.
	let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
	let width = std::env::var("AUS_EMIT_WINDOW")
		.ok()
		.and_then(|s| s.parse::<usize>().ok())
		.filter(|n| *n >= 1)
		.unwrap_or(8)
		.min(cores);
	let budget = std::env::var("AUS_EMIT_BUDGET_MB")
		.ok()
		.and_then(|s| s.parse::<usize>().ok())
		.unwrap_or(8)
		.saturating_mul(1024 * 1024);

	if prof {
		let ests: Vec<usize> = out.pages.iter().map(page_hold_estimate).collect();
		let sum: usize = ests.iter().sum();
		let max = ests.iter().copied().max().unwrap_or(0);
		eprintln!("[profile]   est/page max {:.2} MB, mean {:.2} MB",
			max as f64 / 1048576.0, sum as f64 / 1048576.0 / out.pages.len().max(1) as f64);
	}

	let total = out.pages.len();
	let mut start = 0usize;
	while start < total {
		// Grow the chunk to the page-count width, but stop early once the page ink in flight would exceed
		// the memory budget -- keeping at least the one page so a heavy page still renders.
		let mut end		= start;
		let mut held	= 0usize;
		while end < total && end - start < width {
			let cost = page_hold_estimate(&out.pages[end]);
			if end > start && held + cost > budget {
				break;
			}
			held += cost;
			end += 1;
		}
		let slice	= &out.pages[start..end];

		// Render this chunk's pages in parallel: each worker builds its page's SVG string, its PDF draw
		// list, and that list serialised to content-stream bytes -- all pure, all independent.
		let tr = std::time::Instant::now();
		let out_ref = out_dir;
		let rendered: Vec<Outcome<Prepared>> = std::thread::scope(|scope| {
			let handles: Vec<_> = slice.iter()
				.map(|page| scope.spawn(move || render_page_pair(page, out_ref)))
				.collect();
			handles.into_iter()
				.map(|h| match h.join() {
					Ok(r)	=> r,
					Err(_)	=> Err(err!("A page-render worker thread panicked."; Bug, Thread)),
				})
				.collect()
		});
		if prof { t_render_ms += tr.elapsed().as_secs_f64() * 1000.0; }

		// Fold the chunk into the PDF stream in page order (its `/ID` hashes page by page). The SVG files
		// were already written by the workers. Then free each page's frame, holding no chunk beyond this.
		let tw = std::time::Instant::now();
		for prep in rendered {
			let prep	= res!(prep);
			res!(emit::pdf::write_page_prepared(&mut pdf, &prep.pdf, &prep.content));
		}
		for page in &mut out.pages[start..end] {
			page.frame = Frame::new();
		}
		if prof { t_write_ms += tw.elapsed().as_secs_f64() * 1000.0; }

		start = end;
	}
	res!(pdf.finish());
	mark("emit(svg+pdf)", t_emit);
	if prof {
		eprintln!("[profile]   render (parallel){:>8.1} ms", t_render_ms);
		eprintln!("[profile]   write (in order) {:>8.1} ms", t_write_ms);
		eprintln!("[profile]   width/budgetMB    {:>8}", width);
		eprintln!("[profile] {:<22} {:>8.1} ms", "TOTAL", t_all.elapsed().as_secs_f64() * 1000.0);
	}

	Ok(CompileStats {
		pages:		out.pages.len(),
		passes:		out.passes,
		anchors:	out.ledger.len(),
		skip_line,
	})
}

/// The first double-quoted run in a slice, its contents without the quotes. Mirrors the book
/// assembler's include parsing so the watch set follows exactly the files a compile reads.
fn first_quoted(s: &str) -> Option<String> {
	let open	= match s.find('"') {
		Some(i)	=> i,
		None	=> return None,
	};
	let rest	= &s[open + 1..];
	let close	= match rest.find('"') {
		Some(i)	=> i,
		None	=> return None,
	};
	Some(rest[..close].to_string())
}

/// Adds every file under `dir`, recursively, to `set`, plus `dir` itself so an asset added or removed
/// changes the watched snapshot. Bounded in depth and count so a large font tree cannot make a tick
/// expensive; the cap is generous for a book's assets.
fn collect_files(dir: &std::path::Path, set: &mut Vec<PathBuf>, depth: usize) {
	const MAX_DEPTH:	usize = 6;
	const MAX_FILES:	usize = 4000;

	if depth > MAX_DEPTH || set.len() > MAX_FILES {
		return;
	}
	set.push(dir.to_path_buf());
	let entries = match std::fs::read_dir(dir) {
		Ok(e)	=> e,
		Err(_)	=> return,
	};
	for entry in entries.flatten() {
		let path = entry.path();
		if path.is_dir() {
			collect_files(&path, set, depth + 1);
		} else {
			set.push(path);
		}
		if set.len() > MAX_FILES {
			return;
		}
	}
}

/// The set of files a compile of `source` depends on, for the watch to poll: the root itself, its
/// `config.typ`, each file it `#include`s, and the assets trees a book resolves against (beside the
/// root and one level up, per the book assembler). Recomputed each tick, so a newly added include or
/// asset is watched without a restart.
fn watch_set(source: &str) -> Vec<PathBuf> {
	let mut set: Vec<PathBuf> = Vec::new();
	let src_path = PathBuf::from(source);
	set.push(src_path.clone());

	let root_dir = src_path.parent()
		.map(|p| p.to_path_buf())
		.unwrap_or_else(|| PathBuf::from("."));
	set.push(root_dir.join("config.typ"));

	// Read the root fresh so an include added mid-session joins the watch; a read failure just leaves the
	// include set as it was on the previous tick.
	if let Ok(src) = std::fs::read_to_string(&src_path) {
		for line in src.lines() {
			let t = line.trim_start();
			if let Some(rest) = t.strip_prefix("#include") {
				if let Some(rel) = first_quoted(rest) {
					set.push(root_dir.join(rel));
				}
			}
		}
	}

	// The assets tree sits beside the root and, for a book, one level up at the project root. Watch both,
	// recursively, so an edited figure or image triggers a rebuild, not only an added or removed file.
	collect_files(&root_dir.join("assets"), &mut set, 0);
	if let Some(project_dir) = root_dir.parent() {
		collect_files(&project_dir.join("assets"), &mut set, 0);
	}
	set
}

/// Prints one terse status line for a compile that produced `stats` of `source` into `out_dir`, taking
/// `elapsed` wall: the source, the page count, the wall in seconds, and the skip line folded on where
/// there is one.
fn print_status(source: &str, out_dir: &str, stats: &CompileStats, elapsed: Duration) {
	let mut line = fmt!("[austenite] {} -> {} page(s), {:.2}s -> {}/",
		source, stats.pages, elapsed.as_secs_f64(), out_dir);
	if let Some(skip) = &stats.skip_line {
		line.push_str("; ");
		line.push_str(skip);
	}
	println!("{}", line);
}

fn main() -> Outcome<()> {
	// Flags may precede or follow the paths; only `--watch` (`-w`) is recognised, everything else is a
	// positional argument in order: the source root, then the optional output directory.
	let mut watching	= false;
	let mut pos:	Vec<String>	= Vec::new();
	for a in std::env::args().skip(1) {
		match a.as_str() {
			"--watch" | "-w"	=> watching = true,
			_					=> pos.push(a),
		}
	}
	let source = match pos.first() {
		Some(s)	=> s.clone(),
		None	=> return Err(err!(
			"Usage: austenite [--watch] <SOURCE.typ> [OUTPUT_DIR]"; Input, Invalid, Missing)),
	};
	let out_dir = match pos.get(1) {
		Some(s)	=> s.clone(),
		None	=> "austenite-out".to_string(),
	};

	if watching {
		// Poll interval: brisk enough to feel live, cheap enough to leave the cores to the compile.
		let interval	= Duration::from_millis(400);
		let src_files	= source.clone();		// the file-set closure borrows this
		let src_build	= source.clone();		// the build closure owns this
		let out			= out_dir.clone();
		println!("[austenite] watching {} -> {}/ (Ctrl-C to stop)", source, out_dir);
		return watch::run(
			move || watch_set(&src_files),
			move || {
				let t = std::time::Instant::now();
				match compile(&src_build, &out) {
					Ok(stats)	=> {
						// The skip line is folded into the status line, so the rebuild is one line.
						print_status(&src_build, &out, &stats, t.elapsed());
						Ok(())
					},
					Err(e)		=> Err(e),
				}
			},
			interval,
		);
	}

	let t = std::time::Instant::now();
	let stats = res!(compile(&source, &out_dir));
	if let Some(skip) = &stats.skip_line {
		eprintln!("[austenite] {}", skip);
	}
	println!(
		"austenite: {} -> {} page(s) in {} pass(es); {} anchor(s) in the ledger; {:.2}s; written to {}/",
		source, stats.pages, stats.passes, stats.anchors, t.elapsed().as_secs_f64(), out_dir);
	Ok(())
}
