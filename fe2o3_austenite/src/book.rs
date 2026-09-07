//! Whole-book assembly: follow a Typst root's `#include` chain into one ordered block stream, and read
//! the concrete production values the book's `config.typ` selects.
//!
//! A book is not one source file. Its root (`oxpecker.typ`, `lucronics.typ`) sets the page through a
//! template call and then pulls each chapter in with `#include "chap.typ"`; the geometry and type are
//! chosen in `config.typ` by a `format` switch. The reader ([`lang::to_blocks`](crate::lang)) sets one
//! file and skips code lines, so the include-following and the config-reading live here, above it: this
//! module resolves the includes in document order, feeds each chapter through the reader, and reads the
//! one branch of `config.typ` the book's `format` selects into a [`PageGeometry`] and [`Style`].
//!
//! This is targeted extraction, not a Typst evaluator. It reads the concrete fields these books define
//! -- page size, mirror margins, body and heading type -- from the arm the `format` string picks, and
//! nothing more. A field a book does not set keeps the engine default.
//!
//! Two root idioms are recognised. A *book* root (the elearnity manuscripts) carries a `config.typ`
//! beside it and selects its geometry and type from a `format` switch there. A *doc-template* root (the
//! oxedyne documentation trees -- the Hematite guide, the Austenite design) has no `config.typ`: it takes
//! its A4 page and 2.5 cm margins from the shared `template.typ` and its body size from the `#show:
//! doc.with(...)` call. The presence of `config.typ` picks the path; both follow the root's includes the
//! same way, and both degrade to a readable A4 default rather than failing on a field they cannot find.

use crate::bib::{
	Bibliography,
	RefStyle,
};
use crate::doc::{
	Block,
	FrontMatter,
	HeadingStyle,
	Segment,
	Style,
};
use crate::fonts;
use crate::ir::Sp;
use crate::lang::parse::flatten_markup;
use crate::lang;
use crate::page::PageGeometry;

use oxedyne_fe2o3_core::prelude::*;
use oxedyne_fe2o3_font::{
	font::Font,
	set::FontSet,
};

use std::collections::HashMap;
use std::path::{
	Path,
	PathBuf,
};
use std::sync::Arc;

const MM_PER_PT: f64 = 72.0 / 25.4;	// points in one millimetre

/// A whole book, assembled and ready to author: the block stream in document order, the page geometry
/// and type style read from its config, and the faces loaded by path from its assets.
pub struct BookSpec {
	pub geom:	PageGeometry,
	pub style:	Style,
	pub fonts:	Arc<FontSet>,
	pub blocks:	Vec<Block>,
	pub title:	String,	// the book title, for the verso running head
	// The display face for chapter and level-2 headings (Radley), loaded by path from the book's assets;
	// None when the book ships no such face, so headings fall back to the body bold.
	pub heading:	Option<Arc<Font>>,
	pub front:		FrontMatter,	// the title page, cover and imprint, read from the root's template call
	// The bibliography, parsed from the file the root names and marked with the keys the body cited, so
	// the block layer resolves an in-text `#cite` against it. None when the book names no bibliography.
	pub bib:		Option<Bibliography>,
	// The constructs the reader skipped across every chapter, merged into one tally, so the binary reports
	// a book or doc compile's skipped constructs on the same terse line a lone file already prints.
	pub skips:		lang::SkipSummary,
}

/// Does this source read as a book root -- a Typst file that assembles chapters through `#include`?
/// A single manuscript has none, so the binary can tell a book from a lone file by the source itself.
pub fn is_book_root(src: &str) -> bool {
	src.lines().any(|l| l.trim_start().starts_with("#include"))
}

/// Assembles the document rooted at `root_path` into a [`BookSpec`], recognising both root idioms the
/// house Typst trees use. A *book* root sets its geometry and type through a `config.typ` beside it,
/// chosen by a `format` switch (the elearnity manuscripts); a *doc-template* root has no `config.typ`
/// and takes its A4 geometry from the shared `template.typ` and its body size from the `#show:
/// doc.with(...)` call (the oxedyne documentation trees -- the Hematite guide, the Austenite design).
/// The presence of `config.typ` beside the root selects the path: the book reader is unchanged, and a
/// root without a config falls to the doc reader rather than failing on the missing file.
pub fn load(root_path: &Path) -> Outcome<BookSpec> {
	// Canonicalise the root so its parent and grandparent are real absolute directories. A root given
	// relatively (`lucronics.typ`) otherwise has an empty parent, and the project directory the shared
	// `refs.bib` and assets sit in cannot be found -- a symlinked `assets` masks this for fonts, but the
	// bibliography one level up is missed.
	let root_path = std::fs::canonicalize(root_path).unwrap_or_else(|_| root_path.to_path_buf());
	let root_path = root_path.as_path();
	let root_dir = match root_path.parent() {
		Some(d)	=> d.to_path_buf(),
		None	=> return Err(err!("The book root {:?} has no parent directory.", root_path; Input, Invalid)),
	};
	let root_src = match std::fs::read_to_string(root_path) {
		Ok(s)	=> s,
		Err(e)	=> return Err(err!(e, "Could not read the book root {:?}.", root_path; File, Read)),
	};

	// Install the book's `term-dict` from a `terms.typ` beside or above the root, so the term-dictionary
	// glossary family resolves each key to its value as the chapters are read below.
	res!(install_term_dict(&root_dir));

	// A `config.typ` beside the root marks the book (`format`-switch) idiom; without it, the root sets its
	// page through the shared `template.typ` and the `doc.with` call, which is the documentation idiom.
	let config_path = root_dir.join("config.typ");
	if !config_path.exists() {
		return load_doc(&root_dir, &root_src);
	}
	load_book(&root_dir, &root_src)
}

/// The book (`format`-switch) path: reads the `config.typ` beside the root, loads the shared Libertinus
/// faces by path from the project assets tree, and follows the root's includes into one block stream.
fn load_book(root_dir: &Path, root_src: &str) -> Outcome<BookSpec> {
	// The config sits beside the root; the assets tree is one level up (the project root), holding the
	// Libertinus directory both books share.
	let config_path	= root_dir.join("config.typ");
	let config_src	= match std::fs::read_to_string(&config_path) {
		Ok(s)	=> s,
		Err(e)	=> return Err(err!(e, "Could not read the book config {:?}.", config_path; File, Read)),
	};
	let project_dir = match root_dir.parent() {
		Some(d)	=> d.to_path_buf(),
		None	=> root_dir.to_path_buf(),
	};
	let libertinus_dir = project_dir.join("assets").join("fonts").join("libertinus");
	let fonts = Arc::new(res!(fonts::libertinus_from_dir(&libertinus_dir)));
	// The heading display face (Radley) sits beside Libertinus in the shared assets tree. A book without
	// it still sets, with headings in the body bold, so a failed load is a fall-back rather than an error.
	let radley_path	= project_dir.join("assets").join("fonts").join("Radley-Regular.ttf");
	let heading		= fonts::font_from_file(&radley_path).ok();

	let (geom, raw) = res!(read_config(&config_src));
	let style		= build_style(&raw);
	let (mut blocks, skips)	= res!(assemble(root_src, root_dir));
	let title		= content_field(root_src, "title").unwrap_or_default();
	let front		= read_front_matter(root_src, &config_src, &title);

	// The bibliography the root names, if any: parse it, mark every key the body cited, and append the
	// Chicago reference list as back matter. The marked bibliography then resolves each in-text `#cite`.
	let bib = res!(load_bibliography(root_src, &project_dir, &mut blocks));

	Ok(BookSpec { geom, style, fonts, blocks, title, heading, front, bib, skips })
}

// ┌───────────────────────────────────────────────────────────────────────────┐
// │ DOCUMENTATION IDIOM                                                        │
// └───────────────────────────────────────────────────────────────────────────┘

/// The template's `avg_reading_speed`, in words per minute, used to turn the word count into the reading
/// time the meta page shows.
const AVG_READING_SPEED: usize = 230;

/// The acknowledgement paragraph the documentation template seats at the foot of its meta page, a fixed
/// constant of the doc idiom rather than a `meta-data` field (`template.typ`'s `meta-page`).
const DOC_ACKNOWLEDGEMENT: &str = "We acknowledge the Indigenous peoples who have been traditional \
custodians of lands and waters around the world, past and present. Through their languages, stories, and \
practices, they have sought to maintain living connections to humanity's deepest roots and the ancient \
wisdom of sustainable coexistence with Earth's ecosystems.";

/// Maps an AI-declaration slug to its mark image path and caption, mirroring the template's
/// `ai-declarations` dictionary and its `doc` medium. The path is image-base-relative, resolved the same
/// way as the title logos; an unknown slug yields no mark, so the author cell sets the name alone.
fn ai_declaration_mark(slug: &str) -> Option<(String, String)> {
	let (file, words) = match slug {
		"no-ai"			=> ("doc_made_with_no_ai_opt.svg",			"Made with no AI"),
		"some-ai"		=> ("doc_made_with_some_ai_opt.svg",		"Made with some AI"),
		"with-ai"		=> ("doc_made_with_ai_opt.svg",				"Made with AI"),
		"mostly-ai"		=> ("doc_made_with_ai_mostly_opt.svg",		"Made mostly with AI"),
		"entirely-ai"	=> ("doc_made_with_ai_entirely_opt.svg",	"Made with AI entirely"),
		_				=> return None,
	};
	Some((fmt!("assets/svg/{}", file), words.to_string()))
}

/// The documentation (`doc.with`) path: the oxedyne doc trees (Hematite, Austenite) carry no
/// `config.typ`. Their A4 page and 2.5 cm margins live in the shared `template.typ`, and the body size
/// is the `text-size:` argument of the root's `#show: doc.with(...)` call. Geometry and type are read
/// from those two sources; the body font is the embedded Libertinus, which is the doc body and heading
/// family both (a doc heading is Libertinus bold, so no separate display face is loaded); and the
/// includes are followed exactly as for a book. A field the tree omits keeps a readable default.
fn load_doc(root_dir: &Path, root_src: &str) -> Outcome<BookSpec> {
	let (geom, raw)	= res!(read_doc_config(root_dir, root_src));
	let mut style	= build_style(&raw);
	let title		= content_field(root_src, "title").unwrap_or_default();
	// A doc tree sets `numbering: none`; its top-level headings open with the template's grey banner bar
	// unless the tree draws its own per-section `#section-banner` logo bars, in which case each level-1
	// heading is set inline beneath the section's banner. This mirrors the template's `chapter-banners`
	// argument: an explicit `true`/`false` decides, and its `auto` default turns the chapter banners off
	// only for the Hematite guide, whose sections carry logo banners instead.
	let want_banners = tri_bool(root_src, "chapter-banners").unwrap_or(title != "Hematite");
	style.heading_style = if want_banners {
		HeadingStyle::DocBanner
	} else {
		HeadingStyle::DocInline
	};
	let fonts		= Arc::new(res!(fonts::libertinus()));
	// A doc heading is set in Libertinus bold -- the body family -- so no display face is supplied, and
	// the heading path falls back to the body bold, which is exactly what the template's show rule sets.
	let heading:	Option<Arc<Font>>	= None;
	let (blocks, skips)	= res!(assemble(root_src, root_dir));
	let mut front	= read_doc_front_matter(root_dir, root_src, &raw, &title);

	// The reading time the meta page appends to its notes cell: the whole-document word count over the
	// template's 230 words/min, rounded up, matching its `calc.ceil(words.final() / avg_reading_speed)`.
	let words		= crate::doc::count_words(&blocks);
	front.reading_min	= Some(((words + AVG_READING_SPEED - 1) / AVG_READING_SPEED) as u32);

	// A doc tree names its bibliography, glossary and index through raw Typst calls the reader skips, not
	// the book's `meta-data.bibliography` field, so no reference back matter is assembled here.
	Ok(BookSpec { geom, style, fonts, blocks, title, heading, front, bib: None, skips })
}

/// Reads a doc-template root's geometry and type: the paper and margins from the shared `template.typ`
/// beside the root, and the body size from the root's `doc.with(text-size: ..)` argument. The template
/// fixes uniform margins with a slightly deeper foot (`margins.a4 + 0.25cm`), matching its `set page`.
/// Everything the tree does not state -- leading, paragraph spacing, heading sizes -- takes the Typst
/// default the template inherits, so an unfamiliar doc root still assembles onto a readable A4 page.
fn read_doc_config(root_dir: &Path, root_src: &str) -> Outcome<(PageGeometry, RawStyle)> {
	// The template is symlinked in beside the root; a tree without it falls back to A4 at 2.5 cm.
	let template = std::fs::read_to_string(root_dir.join("template.typ")).unwrap_or_default();

	let paper_name	= first_quoted_after(&template, "paper:").unwrap_or_else(|| "a4".to_string());
	let (pw_mm, ph_mm)	= paper_dims_mm(&paper_name);

	// The uniform margin: the `a4:` entry of the template's `#let margins = (...)` dictionary, a length.
	let margin_pt	= let_dict_field(&template, "margins", "a4")
		.and_then(|v| parse_len_pt(&v))
		.unwrap_or(2.5 * 10.0 * MM_PER_PT);	// 2.5 cm default
	let foot_extra	= 0.25 * 10.0 * MM_PER_PT;	// the template's `bottom: margins.a4 + 0.25cm`

	let geom = PageGeometry::with_margins(
		Sp::from_pt(pw_mm * MM_PER_PT),
		Sp::from_pt(ph_mm * MM_PER_PT),
		Sp::from_pt(margin_pt),
		Sp::from_pt(margin_pt),
		Sp::from_pt(margin_pt),
		Sp::from_pt(margin_pt + foot_extra),
	);

	// The body size the doc.with call sets, else the template's own `text-size: 11pt` default.
	let body_pt	= first_len_after(root_src, "text-size:")
		.or_else(|| first_len_after(&template, "text-size:"))
		.unwrap_or(11.0);

	// The doc template leaves leading and paragraph spacing at the Typst defaults it inherits (0.65 em
	// leading; a paragraph gap of the same order with no first-line indent), and sizes its headings in the
	// show rule: a level-1 heading at 14 pt small-caps, level 2 at 12 pt, level 3 at 13 pt, level 4 at 12 pt.
	let raw = RawStyle {
		body_pt,
		leading_em:		0.65,
		par_skip_em:	0.65,
		indent_em:		0.0,
		chap_num_pt:	54.0,
		chap_grid:		[72.0, 8.0, 36.0, 20.0],
		h1_pt:			14.0,
		h2_pt:			12.0,
		h3_pt:			13.0,
		h4_pt:			12.0,
	};
	Ok((geom, raw))
}

/// The front matter a doc root states: its title and subtitle, the author from the first `meta-data`
/// entry, and the documentation template's two-column title-page furniture -- the sidebar width and
/// colour, the two sidebar logos with their declared widths, the small-caps flag, and the footer logo --
/// read from the `#show: doc.with(...)` call and the shared `template.typ`. A doc tree carries no imprint
/// (no ISBN, publisher or copyright tuple), so only a title page and the contents are composed from this.
fn read_doc_front_matter(root_dir: &Path, root_src: &str, raw: &RawStyle, title: &str) -> FrontMatter {
	let subtitle	= content_field(root_src, "subtitle");
	let meta		= meta_block(root_src).unwrap_or_default();
	let author		= string_field(&meta, "authors").unwrap_or_default();

	// The revision rows the template's meta/colophon page draws: each row's version, date, notes, and the
	// AI declaration whose slug picks the mark image and its caption (a `declaration-words` field rescopes
	// the caption without changing the mark). A doc tree may state several rows, newest first.
	let meta_rows: Vec<crate::doc::MetaRow> = meta_rows(&meta).iter().map(|row| {
		let (ai_mark_path, ai_mark_words) = match string_field(row, "declaration") {
			Some(slug)	=> match ai_declaration_mark(&slug) {
				Some((path, words))	=> {
					let words = string_field(row, "declaration-words").unwrap_or(words);
					(Some(path), Some(words))
				},
				None				=> (None, None),
			},
			None		=> (None, None),
		};
		crate::doc::MetaRow {
			version:	string_field(row, "version"),
			date:		string_field(row, "date"),
			authors:	string_field(row, "authors").unwrap_or_default(),
			notes:		string_field(row, "notes"),
			ai_mark_path,
			ai_mark_words,
		}
	}).collect();
	// The colophon furniture the template fixes for the doc idiom: the acknowledgement paragraph, and the
	// copyright line composed from the organisation the term dictionary names (`#t("org")`), falling back
	// to Oxedyne. Both are template constants rather than `meta-data`, so they are set here for every doc.
	let acknowledgement	= Some(DOC_ACKNOWLEDGEMENT.to_string());
	let org				= crate::lang::parse::term_value("org").unwrap_or_else(|| "Oxedyne".to_string());
	let copyright		= Some(fmt!("Copyright © 12025 {}. All rights reserved.", org));

	// The sidebar width is `margins.title_page` in the shared template (a percentage of the page); the fill
	// is the `title-colour` the call names, resolved to a grey level. A doc tree always draws the sidebar,
	// so `sidebar_grey` is set here (marking the two-column idiom) even when the call omits its colour.
	let template	= std::fs::read_to_string(root_dir.join("template.typ")).unwrap_or_default();
	let sidebar_frac	= let_dict_field(&template, "margins", "title_page")
		.and_then(|v| parse_percent(&v))
		.unwrap_or(0.45);
	let colour_name	= string_field(root_src, "title-colour").unwrap_or_default();
	let sidebar_grey	= Some(grey_luma(&colour_name));

	let non_empty	= |s: Option<String>| s.filter(|p| !p.is_empty());
	let top_logo	= non_empty(string_field(root_src, "title-top-logo-path"));
	let bottom_logo	= non_empty(string_field(root_src, "title-bottom-logo-path"));
	let footer_logo	= non_empty(string_field(root_src, "footer-left-logo-path"));
	let top_w		= first_len_after(root_src, "title-top-logo-width:").unwrap_or(80.0);
	let bottom_w	= first_len_after(root_src, "title-bottom-logo-width:").unwrap_or(120.0);
	let smallcaps	= bool_field(root_src, "title-smallcaps");

	FrontMatter {
		title:			title.to_string(),
		subtitle,
		author,
		cover_image:	None,
		logo_image:		None,
		publisher:		None,
		edition:		None,
		isbn:			None,
		copyright,
		rights:			None,
		ai_declaration:	None,
		website:		None,
		toolchain:		false,
		dedication:		None,
		about_author:	None,
		title_size:		Sp::from_pt(28.0),
		subtitle_size:	Sp::from_pt(16.0),
		author_size:	Sp::from_pt(17.0),
		back_title_size:	Sp::from_pt(raw.h1_pt),
		sidebar_grey,
		sidebar_frac,
		title_smallcaps:	smallcaps,
		top_logo,
		top_logo_width:		Sp::from_pt(top_w),
		bottom_logo,
		bottom_logo_width:	Sp::from_pt(bottom_w),
		footer_logo,
		meta_rows,
		reading_min:	None,	// set by `load_doc`, which has the body blocks to count
		acknowledgement,
	}
}

/// Resolves a `template.typ` colour name to a grey level. Only the greys the doc trees reach for are
/// mapped by name; every other name falls to the template's `colours.light` (luma 240), a light sidebar.
fn grey_luma(name: &str) -> u8 {
	match name {
		"white"			=> 255,
		"lightgrey"		=> 240,
		"light"			=> 240,
		_				=> 240,
	}
}

/// Reads a Typst percentage literal (`45%`) as a fraction (`0.45`). `None` when no number leads it.
fn parse_percent(s: &str) -> Option<f64> {
	first_num(s).map(|n| n / 100.0)
}

// ┌───────────────────────────────────────────────────────────────────────────┐
// │ BIBLIOGRAPHY                                                               │
// └───────────────────────────────────────────────────────────────────────────┘

/// Parses the bibliography the root's `meta-data.bibliography` names, marks every key the body cited,
/// and appends the Bibliography back matter (a heading and the sorted, cited-only reference list) to the
/// block stream. Returns the marked bibliography for the in-text citation formatter, or `None` when the
/// book names no bibliography or the file cannot be read.
fn load_bibliography(root_src: &str, project_dir: &Path, blocks: &mut Vec<Block>) -> Outcome<Option<Bibliography>> {
	let meta = match meta_block(root_src) {
		Some(m)	=> m,
		None	=> return Ok(None),
	};
	let path_str = match string_field(&meta, "bibliography") {
		Some(p)	=> p,
		None	=> return Ok(None),	// no bibliography named
	};

	// The path is Typst-root-relative (`/refs.bib`); resolve it against the project directory.
	let rel		= path_str.trim_start_matches('/');
	let bib_path	= project_dir.join(rel);
	let src = match std::fs::read_to_string(&bib_path) {
		Ok(s)	=> s,
		Err(_)	=> return Ok(None),	// a named bibliography that will not read is a reported gap, not a failure
	};
	let bib = res!(Bibliography::parse(&src));
	Ok(Some(append_bibliography(bib, blocks)))
}

/// Marks every key the body cited on `bib`, then appends the Bibliography back matter -- the section
/// heading and one Reference block per sorted, cited reference -- to the block stream, returning the
/// marked bibliography for the in-text citation formatter. Shared by the whole-book path and the lone
/// chapter path, so a chapter compiled on its own resolves its citations exactly as the book does.
fn append_bibliography(mut bib: Bibliography, blocks: &mut Vec<Block>) -> Bibliography {
	// Mark every key the body cited, so the reference list holds exactly the cited works.
	for keys in collect_cite_keys(blocks) {
		for k in keys {
			bib.mark_cited(&k);
		}
	}

	// Append the back matter: the section heading, then one Reference block per sorted, cited reference.
	blocks.push(Block::back_matter_heading("Bibliography"));
	for reference in bib.reference_list() {
		let runs: Vec<(String, bool)> = reference.runs.iter()
			.map(|r| (r.text.clone(), r.style == RefStyle::Italic))
			.collect();
		blocks.push(Block::reference(runs));
	}
	bib
}

/// Locates a `refs.bib` beside a lone chapter or in an ancestor directory, parses it, marks the keys the
/// chapter cited, appends the reference list as back matter, and returns the marked bibliography so the
/// block layer resolves each in-text `#cite` to Chicago author-year -- as a whole-book compile does.
/// `None` when no `refs.bib` is found or it will not read, in which case the raw cite key stands as before.
pub fn load_lone_bibliography(source: &Path, blocks: &mut Vec<Block>) -> Outcome<Option<Bibliography>> {
	let start = match source.parent() {
		Some(d)	=> d,
		None	=> return Ok(None),
	};
	let bib_path = match find_up(start, "refs.bib") {
		Some(p)	=> p,
		None	=> return Ok(None),
	};
	let src = match std::fs::read_to_string(&bib_path) {
		Ok(s)	=> s,
		Err(_)	=> return Ok(None),	// a bibliography found but unreadable is a reported gap, not a failure
	};
	let bib = res!(Bibliography::parse(&src));
	Ok(Some(append_bibliography(bib, blocks)))
}

// ┌───────────────────────────────────────────────────────────────────────────┐
// │ TERM DICTIONARY                                                            │
// └───────────────────────────────────────────────────────────────────────────┘

/// Reads the book's `term-dict` from a `terms.typ` beside or above `start_dir` and installs it, so the
/// term-dictionary glossary family (`t`, `tcap`, `graw`, `g`, `gi`, `gcap`, `gcapi`) resolves each key to
/// its value while the chapters are read. An absent or `term-dict`-less `terms.typ` installs an empty
/// map, under which every key falls back to its own text.
pub fn install_term_dict(start_dir: &Path) -> Outcome<()> {
	let src = match find_up(start_dir, "terms.typ") {
		Some(p)	=> std::fs::read_to_string(&p).unwrap_or_default(),
		None	=> String::new(),
	};
	res!(crate::lang::parse::set_term_dict(parse_term_dict(&src)));
	Ok(())
}

/// Parses the `#let term-dict = ( "key": "value", ... )` block from a `terms.typ` source into a key→value
/// map. `terms.typ` is a pure data file (no imports, state or side effects), so the literal is read
/// directly rather than evaluated: the quoted strings inside the dictionary's balanced parentheses come in
/// key, value order, and are paired off. An empty map when the source names no `term-dict`.
fn parse_term_dict(src: &str) -> HashMap<String, String> {
	let mut map = HashMap::new();
	// Find the assignment `term-dict =`, not a mention in a comment: the name must be followed, after only
	// whitespace, by `=`. The file opens with a `// term-dict: ...` comment whose own parentheses would
	// otherwise be read as the literal, so the first bare occurrence is not enough.
	let at = match assignment_offset(src, "term-dict") {
		Some(a)	=> a,
		None	=> return map,
	};
	let chars: Vec<char> = src[at..].chars().collect();

	// Advance to the opening parenthesis of the dictionary literal.
	let mut i = 0;
	while i < chars.len() && chars[i] != '(' {
		i += 1;
	}
	if i >= chars.len() {
		return map;
	}

	// Walk the balanced group, collecting each `"..."` string; string escapes are honoured so a quote or
	// backslash inside a value does not end it early. The strings alternate key, value, key, value.
	let mut depth	= 0i32;
	let mut strings:	Vec<String>	= Vec::new();
	while i < chars.len() {
		match chars[i] {
			'('	=> depth += 1,
			')'	=> {
				depth -= 1;
				if depth == 0 {
					break;
				}
			},
			'"'	=> {
				let mut s	= String::new();
				let mut esc	= false;
				i += 1;
				while i < chars.len() {
					let c = chars[i];
					if esc			{ s.push(c); esc = false; }
					else if c == '\\'	{ esc = true; }
					else if c == '"'	{ break; }
					else			{ s.push(c); }
					i += 1;
				}
				strings.push(s);
			},
			_	=> {},
		}
		i += 1;
	}

	let mut k = 0;
	while k + 1 < strings.len() {
		map.insert(strings[k].clone(), strings[k + 1].clone());
		k += 2;
	}
	map
}

/// The byte offset of a `name =` assignment in `src` -- the position of `name` where the next
/// non-whitespace character after it is `=`. Skips a mention of the name in a comment or another context
/// (say `// name: ...`), returning the first true assignment, or `None` when there is none.
fn assignment_offset(src: &str, name: &str) -> Option<usize> {
	let mut from = 0;
	while let Some(rel) = src[from..].find(name) {
		let at		= from + rel;
		let after	= at + name.len();
		let rest	= src[after..].trim_start();
		if rest.starts_with('=') {
			return Some(at);
		}
		from = after;
	}
	None
}

/// Searches `start` and up to a few ancestor directories for a file named `name`, returning the first
/// that exists. The bound keeps a lone-file compile from walking to the filesystem root: a book's shared
/// `terms.typ` or `refs.bib` sits at most a couple of levels above a chapter.
fn find_up(start: &Path, name: &str) -> Option<PathBuf> {
	const MAX_HOPS: usize = 6;
	let mut dir		= Some(start);
	let mut hops	= 0usize;
	while let Some(d) = dir {
		let cand = d.join(name);
		if cand.exists() {
			return Some(cand);
		}
		if hops >= MAX_HOPS {
			break;
		}
		hops += 1;
		dir = d.parent();
	}
	None
}

/// Gathers the citation keys the body's blocks carry, in document order, so each can be marked cited.
fn collect_cite_keys(blocks: &[Block]) -> Vec<Vec<String>> {
	let mut out = Vec::new();
	for block in blocks {
		match block {
			Block::RichParagraph { segments }	=> collect_cite_segments(segments, &mut out),
			Block::List { items, .. }			=> for item in items {
				collect_cite_segments(item, &mut out);
			},
			_									=> {},
		}
	}
	out
}

/// Pushes the keys of every citation segment in `segments` onto `out`.
fn collect_cite_segments(segments: &[Segment], out: &mut Vec<Vec<String>>) {
	for seg in segments {
		if let Segment::Cite(keys) = seg {
			out.push(keys.clone());
		}
	}
}

// ┌───────────────────────────────────────────────────────────────────────────┐
// │ FRONT MATTER EXTRACTION                                                    │
// └───────────────────────────────────────────────────────────────────────────┘

/// Reads the front matter the root's `#show: doc.with(...)` sets: the title and subtitle, the author and
/// imprint from `meta-data`, the cover the config selects for a development build, and the display sizes
/// from the config's type scale. A field the book omits is left `None`, and its page or line is not set.
fn read_front_matter(root_src: &str, config_src: &str, title: &str) -> FrontMatter {
	let subtitle	= content_field(root_src, "subtitle");
	let meta		= meta_block(root_src).unwrap_or_default();

	let author		= string_field(&meta, "authors").unwrap_or_default();
	let publisher	= content_field(&meta, "publisher").map(|s| clean_content(&s));
	let edition		= string_field(&meta, "edition");
	let isbn		= string_field(&meta, "isbn");
	let copyright	= copyright_line(&meta);
	let rights		= content_field(&meta, "rights").map(|s| clean_content(&s));
	let ai_decl		= content_field(&meta, "ai-declaration").map(|s| clean_content(&s));
	let website		= content_field(&meta, "website").map(|s| clean_content(&s)).filter(|s| !s.is_empty());
	let toolchain	= bool_field(&meta, "show-toolchain");
	let dedication	= string_field(&meta, "dedication").filter(|s| s != "none" && !s.is_empty());
	let about		= content_field(&meta, "bio").map(|s| clean_content(&s)).filter(|s| !s.is_empty());
	let logo		= string_field(root_src, "title-logo-path");

	// The cover the config picks: none in an interior build, the format's raster in a development one.
	let format		= read_let_string(config_src, "format").unwrap_or_default();
	let mode		= read_let_string(config_src, "mode").unwrap_or_default();
	let cover		= if mode == "interior" {
		None
	} else {
		arm(config_src, "cover-image-path", &format).as_deref().and_then(first_quoted)
	};

	// The display sizes from the type scale the format selects.
	let scale		= arm(config_src, "type-scale", &format);
	let sz = |key: &str, default: f64| -> Sp {
		Sp::from_pt(scale.as_deref().and_then(|a| num_after(a, key)).unwrap_or(default))
	};

	FrontMatter {
		title:			title.to_string(),
		subtitle,
		author,
		cover_image:	cover,
		logo_image:		logo,
		publisher,
		edition,
		isbn,
		copyright,
		rights,
		ai_declaration:	ai_decl,
		website,
		toolchain,
		dedication,
		about_author:	about,
		title_size:		sz("title:", 28.0),
		subtitle_size:	sz("subtitle:", 16.0),
		author_size:	sz("author:", 17.0),
		back_title_size:	sz("back-matter-title:", 17.0),
		// A book draws its own plain centred title page, not the doc template's two-column one, so the
		// documentation sidebar and logos are left unset (`sidebar_grey: None` keeps the plain title page).
		sidebar_grey:		None,
		sidebar_frac:		0.0,
		title_smallcaps:	false,
		top_logo:			None,
		top_logo_width:		Sp::ZERO,
		bottom_logo:		None,
		bottom_logo_width:	Sp::ZERO,
		footer_logo:		None,
		// The doc template's meta/colophon fields; a book draws its own imprint page, not the doc colophon.
		meta_rows:			Vec::new(),
		reading_min:		None,
		acknowledgement:	None,
	}
}

/// The inner text of the root's `meta-data: ( ... )` argument, balanced across nested groups and
/// strings, or `None` when the root sets no `meta-data`.
fn meta_block(src: &str) -> Option<String> {
	let at		= src.find("meta-data:")?;
	let rest	= &src[at + "meta-data:".len()..];
	let open	= rest.find('(')?;
	let bytes	= rest.as_bytes();
	let mut depth	= 0i32;
	let mut in_str	= false;
	let mut esc		= false;
	let mut i		= open;
	while i < bytes.len() {
		let c = bytes[i] as char;
		if in_str {
			if esc			{ esc = false; }
			else if c == '\\'	{ esc = true; }
			else if c == '"'	{ in_str = false; }
			i += 1;
			continue;
		}
		match c {
			'"'	=> in_str = true,
			'('	=> depth += 1,
			')'	=> {
				depth -= 1;
				if depth == 0 {
					return Some(rest[open + 1..i].to_string());
				}
			},
			_	=> {},
		}
		i += 1;
	}
	None
}

/// Splits a `meta-data` block into its revision rows: the text inside each top-level parenthesised tuple,
/// in source order. Nested parentheses and strings are respected, so a row whose value carries a comma or
/// a bracket is not split early. A block with no nested tuple (a bare single row) yields no rows.
fn meta_rows(block: &str) -> Vec<String> {
	let bytes	= block.as_bytes();
	let mut rows:	Vec<String>	= Vec::new();
	let mut depth	= 0i32;
	let mut in_str	= false;
	let mut esc		= false;
	let mut start	= 0usize;
	let mut i		= 0usize;
	while i < bytes.len() {
		let c = bytes[i] as char;
		if in_str {
			if esc			{ esc = false; }
			else if c == '\\'	{ esc = true; }
			else if c == '"'	{ in_str = false; }
			i += 1;
			continue;
		}
		match c {
			'"'	=> in_str = true,
			'('	=> {
				if depth == 0 { start = i + 1; }
				depth += 1;
			},
			')'	=> {
				depth -= 1;
				if depth == 0 {
					rows.push(block[start..i].to_string());
				}
			},
			_	=> {},
		}
		i += 1;
	}
	rows
}

/// The string a `name: "..."` field binds: the first `"..."` in the field's value, which runs to the
/// next top-level comma (a comma inside the string does not end it). `None` when the value holds no
/// string literal -- a `name: none` reads as absent -- so a later field's value is never read by mistake.
fn string_field(src: &str, name: &str) -> Option<String> {
	let needle	= fmt!("{}:", name);
	let at		= src.find(&needle)?;
	let rest	= &src[at + needle.len()..];
	// Bound the value at the next depth-zero comma, respecting strings, so the search stays in this field.
	let bytes	= rest.as_bytes();
	let mut depth	= 0i32;
	let mut in_str	= false;
	let mut esc		= false;
	let mut end		= rest.len();
	let mut i		= 0usize;
	while i < bytes.len() {
		let c = bytes[i] as char;
		if in_str {
			if esc			{ esc = false; }
			else if c == '\\'	{ esc = true; }
			else if c == '"'	{ in_str = false; }
			i += 1;
			continue;
		}
		match c {
			'"'					=> in_str = true,
			'(' | '[' | '{'		=> depth += 1,
			')' | ']' | '}'		=> depth -= 1,
			',' if depth == 0	=> { end = i; break; },
			_					=> {},
		}
		i += 1;
	}
	first_quoted(&rest[..end])
}

/// The `Copyright © YEAR HOLDER. NOTICE` line the template composes from the `copyright: (year, [holder],
/// notice)` tuple, or `None` when the book sets no copyright tuple.
fn copyright_line(meta: &str) -> Option<String> {
	let at		= meta.find("copyright:")?;
	let rest	= &meta[at + "copyright:".len()..];
	let open	= rest.find('(')?;
	// The tuple's three parts: a year string, a `[holder]` content, and a notice string.
	let inner	= balanced_parens(&rest[open..])?;
	let parts	= split_top(&inner);
	if parts.is_empty() {
		return None;
	}
	let year	= parts.first().map(|s| unquote_or_content(s)).unwrap_or_default();
	let holder	= parts.get(1).map(|s| unquote_or_content(s)).unwrap_or_default();
	let notice	= parts.get(2).map(|s| unquote_or_content(s)).unwrap_or_default();
	Some(fmt!("Copyright © {} {}. {}", year.trim(), holder.trim(), notice.trim()))
}

/// The contents of a `(...)` at the start of `s`, balanced across nesting and strings.
fn balanced_parens(s: &str) -> Option<String> {
	let bytes	= s.as_bytes();
	let mut depth	= 0i32;
	let mut in_str	= false;
	let mut esc		= false;
	let mut i		= 0usize;
	while i < bytes.len() {
		let c = bytes[i] as char;
		if in_str {
			if esc			{ esc = false; }
			else if c == '\\'	{ esc = true; }
			else if c == '"'	{ in_str = false; }
			i += 1;
			continue;
		}
		match c {
			'"'	=> in_str = true,
			'('	=> depth += 1,
			')'	=> {
				depth -= 1;
				if depth == 0 {
					return Some(s[1..i].to_string());
				}
			},
			_	=> {},
		}
		i += 1;
	}
	None
}

/// Splits `s` at its top-level commas, respecting nesting and strings.
fn split_top(s: &str) -> Vec<String> {
	let mut out:	Vec<String>	= Vec::new();
	let mut cur					= String::new();
	let mut depth				= 0i32;
	let mut in_str				= false;
	let mut esc					= false;
	for c in s.chars() {
		if in_str {
			cur.push(c);
			if esc			{ esc = false; }
			else if c == '\\'	{ esc = true; }
			else if c == '"'	{ in_str = false; }
			continue;
		}
		match c {
			'"'					=> { in_str = true; cur.push(c); },
			'(' | '[' | '{'		=> { depth += 1; cur.push(c); },
			')' | ']' | '}'		=> { depth -= 1; cur.push(c); },
			',' if depth == 0	=> out.push(std::mem::take(&mut cur)),
			_					=> cur.push(c),
		}
	}
	if !cur.trim().is_empty() {
		out.push(cur);
	}
	out
}

/// Reads a tuple part as a plain string: a `"..."` literal unquoted, or a `[...]` content flattened.
fn unquote_or_content(part: &str) -> String {
	let t = part.trim();
	if t.starts_with('"') && t.ends_with('"') && t.len() >= 2 {
		return t[1..t.len() - 1].to_string();
	}
	if t.starts_with('[') && t.ends_with(']') && t.len() >= 2 {
		return flatten_markup(&t[1..t.len() - 1]);
	}
	t.to_string()
}

/// Whether a `name: true` boolean field is set true.
fn bool_field(src: &str, name: &str) -> bool {
	let needle	= fmt!("{}:", name);
	match src.find(&needle) {
		Some(at)	=> {
			let rest	= &src[at + needle.len()..];
			let end		= rest.find(',').unwrap_or(rest.len());
			rest[..end].trim().starts_with("true")
		},
		None		=> false,
	}
}

/// A three-way boolean argument in the root's template call: `Some(true)`/`Some(false)` when the field is
/// set to `true`/`false`, and `None` when it is absent or left `auto`, so the caller can supply its own
/// default for the `auto` case.
fn tri_bool(src: &str, name: &str) -> Option<bool> {
	let needle	= fmt!("{}:", name);
	let at		= src.find(&needle)?;
	let rest	= &src[at + needle.len()..];
	let end		= rest.find(',').unwrap_or(rest.len());
	let val		= rest[..end].trim();
	if val.starts_with("true") {
		Some(true)
	} else if val.starts_with("false") {
		Some(false)
	} else {
		None
	}
}

/// Reduces a content field to a single line of display text: markup flattened, Typst line breaks (`\`)
/// turned to spaces, and any leftover `#name[...]` term call stripped, so an imprint or biography line
/// reads as plain prose.
fn clean_content(s: &str) -> String {
	let flat	= flatten_markup(s);
	let mut out	= flat.replace('\\', " ");
	// Drop a `#ident[...]` or `#ident("...")` term call left after flattening (e.g. `#t[website]`).
	while let Some(h) = out.find('#') {
		let tail	= &out[h + 1..];
		let name_len	= tail.chars().take_while(|c| c.is_alphanumeric() || *c == '-' || *c == '_').count();
		let after	= &tail[name_len..];
		let end = if after.starts_with('[') {
			after.find(']').map(|e| h + 1 + name_len + e + 1)
		} else if after.starts_with('(') {
			after.find(')').map(|e| h + 1 + name_len + e + 1)
		} else {
			Some(h + 1 + name_len)
		};
		match end {
			Some(e)	=> { out.replace_range(h..e, ""); },
			None	=> break,
		}
	}
	out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The text of a `name: [ ... ]` content field in the root's template call -- the book title, say --
/// with the surrounding brackets dropped and inner whitespace trimmed. Bracket-balanced, so a nested
/// group does not close it early.
fn content_field(src: &str, name: &str) -> Option<String> {
	let needle	= fmt!("{}:", name);
	let at		= src.find(&needle)?;
	let rest	= &src[at + needle.len()..];
	let open	= rest.find('[')?;
	let bytes	= rest.as_bytes();
	let mut depth	= 0i32;
	let mut i	= open;
	while i < bytes.len() {
		match bytes[i] {
			b'['	=> depth += 1,
			b']'	=> {
				depth -= 1;
				if depth == 0 {
					return Some(rest[open + 1..i].trim().to_string());
				}
			},
			_	=> {},
		}
		i += 1;
	}
	None
}

// ┌───────────────────────────────────────────────────────────────────────────┐
// │ INCLUDE FOLLOWING                                                          │
// └───────────────────────────────────────────────────────────────────────────┘

/// Follows a root's `#include "..."` lines in order, reading each chapter and setting it through the
/// reader, and lifts each `#part-page[...]` divider to a level-1 heading so the part titles keep their
/// place in the flow. The root's own inline markup between the code lines is read too, in document order:
/// a doc root opens with a section (`= Purpose ...`) written straight in the root before its includes,
/// where a book root carries only the template call. Non-code lines are accumulated and flushed through
/// the reader at each include or part boundary, so the reader sees whole markup runs -- a heading and its
/// paragraphs together -- and the root's opening section keeps its place ahead of the first chapter. The
/// template call itself (`#show: doc.with(...)`, `#import`, `#pagebreak`) is code the reader skips and
/// tallies, so it never leaks into the flow.
pub fn assemble(root_src: &str, root_dir: &Path) -> Outcome<(Vec<Block>, lang::SkipSummary)> {
	let mut blocks: Vec<Block> = Vec::new();
	let mut skips = lang::SkipSummary::default();
	let mut buf = String::new();	// the root's inline markup gathered since the last boundary
	for line in root_src.lines() {
		let t = line.trim_start();
		if let Some(rest) = t.strip_prefix("#include") {
			res!(flush_inline(&mut buf, &mut blocks, &mut skips));
			if let Some(rel) = first_quoted(rest) {
				let path	= root_dir.join(&rel);
				let src		= match std::fs::read_to_string(&path) {
					Ok(s)	=> s,
					Err(e)	=> return Err(err!(e,
						"Could not read the included chapter {:?}.", path; File, Read)),
				};
				let (chap, chap_skips) = res!(lang::to_blocks_with_skips(&src));
				blocks.extend(chap);
				skips.merge(&chap_skips);
			}
		} else if t.starts_with("#part-page") {
			res!(flush_inline(&mut buf, &mut blocks, &mut skips));
			// A part divider: its title is the last bracket group on the line. A part is a level-0 heading
			// -- unnumbered and centred on its own page, outside the chapter numbering -- so a chapter keeps
			// its number across a part boundary and a part never appears in a running head.
			if let Some(title) = bracket_body(t) {
				blocks.push(Block::heading(0, title));
			}
		} else {
			buf.push_str(line);
			buf.push('\n');
		}
	}
	// The tail after the last include: back-matter markup a doc root closes with, if any.
	res!(flush_inline(&mut buf, &mut blocks, &mut skips));
	Ok((blocks, skips))
}

/// Reads the accumulated inline markup through the reader, appending its blocks and merging its skips,
/// then clears the buffer. A buffer holding only code and whitespace yields no blocks -- a book root's
/// template call reduces to nothing, so the book path is unchanged.
fn flush_inline(buf: &mut String, blocks: &mut Vec<Block>, skips: &mut lang::SkipSummary) -> Outcome<()> {
	if !buf.trim().is_empty() {
		let (b, s) = res!(lang::to_blocks_with_skips(buf));
		blocks.extend(b);
		skips.merge(&s);
	}
	buf.clear();
	Ok(())
}

/// The first double-quoted run in a slice, its contents without the quotes.
fn first_quoted(s: &str) -> Option<String> {
	let open	= s.find('"')?;
	let rest	= &s[open + 1..];
	let close	= rest.find('"')?;
	Some(rest[..close].to_string())
}

/// The contents of the first `[...]` group in a line, balanced so a nested bracket does not close it
/// early. Used to lift a `#part-page[Title]` divider's title.
fn bracket_body(s: &str) -> Option<String> {
	let open	= s.find('[')?;
	let bytes	= s.as_bytes();
	let mut depth	= 0i32;
	let mut i	= open;
	while i < bytes.len() {
		match bytes[i] {
			b'['	=> depth += 1,
			b']'	=> {
				depth -= 1;
				if depth == 0 {
					return Some(s[open + 1..i].trim().to_string());
				}
			},
			_	=> {},
		}
		i += 1;
	}
	None
}

// ┌───────────────────────────────────────────────────────────────────────────┐
// │ CONFIG EXTRACTION                                                          │
// └───────────────────────────────────────────────────────────────────────────┘

/// The raw type values read from a config arm, before they are turned into a [`Style`]. Kept apart from
/// the geometry because the geometry is complete on its own, while the style also needs the loaded font
/// metrics to turn an em-leading into a baseline distance.
struct RawStyle {
	body_pt:	f64,
	leading_em:	f64,	// par leading, a multiple of the em
	par_skip_em:	f64,	// space between paragraphs, a multiple of the em
	indent_em:	f64,	// first-line indent, a multiple of the em
	chap_num_pt:	f64,	// the giant chapter-opener number size
	chap_grid:	[f64; 4],	// chapter-opener grid rows: number band, gap, title band, gap-to-body, in points
	h1_pt:		f64,	// chapter-title size
	h2_pt:		f64,	// level-2 sub-heading size
	h3_pt:		f64,	// level-3 sub-heading size
	h4_pt:		f64,	// level-4 sub-heading size
}

/// Reads the branch of a config the `format` switch selects into a geometry and the raw type values.
/// A book that omits a field falls back to a readable default rather than failing, so an unfamiliar
/// config still assembles.
fn read_config(src: &str) -> Outcome<(PageGeometry, RawStyle)> {
	let format = match read_let_string(src, "format") {
		Some(f)	=> f,
		None	=> return Err(err!(
			"The config sets no `#let format = \"...\"`, so no page branch can be chosen."; Input, Missing)),
	};

	let dims	= arm(src, "page-dims", &format);
	let margins	= arm(src, "page-margins", &format);
	let scale	= arm(src, "type-scale", &format);

	let width	= dims.as_deref().and_then(|a| num_after(a, "width:")).unwrap_or(148.0);
	let height	= dims.as_deref().and_then(|a| num_after(a, "height:")).unwrap_or(210.0);
	let inside	= margins.as_deref().and_then(|a| num_after(a, "inside:")).unwrap_or(19.0);
	let outside	= margins.as_deref().and_then(|a| num_after(a, "outside:")).unwrap_or(17.0);
	let top		= margins.as_deref().and_then(|a| num_after(a, "top:")).unwrap_or(19.0);
	let bottom	= margins.as_deref().and_then(|a| num_after(a, "bottom:")).unwrap_or(21.0);

	let geom = PageGeometry::with_margins(
		Sp::from_pt(width  * MM_PER_PT),
		Sp::from_pt(height * MM_PER_PT),
		Sp::from_pt(inside  * MM_PER_PT),
		Sp::from_pt(outside * MM_PER_PT),
		Sp::from_pt(top    * MM_PER_PT),
		Sp::from_pt(bottom * MM_PER_PT),
	);

	let body_pt		= arm(src, "body-text-size", &format).as_deref().and_then(first_num).unwrap_or(11.0);
	let leading_em	= arm(src, "body-line-spacing", &format).as_deref().and_then(first_num).unwrap_or(0.75);
	let par_skip_em	= arm(src, "body-par-spacing", &format).as_deref().and_then(first_num).unwrap_or(0.75);
	let indent_em	= arm(src, "body-par-indent", &format).as_deref().and_then(first_num).unwrap_or(0.0);
	let chap_num_pt	= scale.as_deref().and_then(|a| num_after(a, "chapter-num:")).unwrap_or(54.0);
	// The chapter-opener grid rows: the number band, the gap below it, the title band, and the gap down to
	// the body. Absent, the opener falls back to spacers roughly matching a 20 pt body scale.
	let grid		= scale.as_deref().and_then(|a| tuple_after(a, "chapter-grid-rows:")).unwrap_or_default();
	let chap_grid	= [
		grid.first().copied().unwrap_or(72.0),
		grid.get(1).copied().unwrap_or(8.0),
		grid.get(2).copied().unwrap_or(36.0),
		grid.get(3).copied().unwrap_or(20.0),
	];
	let h1_pt		= scale.as_deref().and_then(|a| num_after(a, "chapter-title:")).unwrap_or(20.0);
	// The template sizes a sub-heading by `sub-headings.at(level - 1)`: level 2 takes the second entry,
	// level 3 the third, level 4 the fourth. The first entry is the section-title reserve, unused by the
	// show rule, so the level-2 heading is only a step above the body.
	let subs		= scale.as_deref().and_then(|a| tuple_after(a, "sub-headings:")).unwrap_or_default();
	let h2_pt		= subs.get(1).copied().unwrap_or(12.5);
	let h3_pt		= subs.get(2).copied().unwrap_or(11.5);
	let h4_pt		= subs.get(3).copied().unwrap_or(11.0);

	Ok((geom, RawStyle { body_pt, leading_em, par_skip_em, indent_em, chap_num_pt, chap_grid, h1_pt, h2_pt, h3_pt, h4_pt }))
}

// The Libertinus line box Typst sets, as a fraction of the em, measured from the oracle. Typst's config
// leading is the gap ADDED between line boxes; the baseline-to-baseline skip is that gap plus the box.
// The box is not the face's nominal ascender + descender (fe2o3_font reports ~1.14 em for Libertinus,
// which sets ~30% too loose); Typst's rendered Libertinus line box measures ~0.66 em -- for an 11 pt
// body at 0.78 em leading that gives (0.66 + 0.78) x 11 = 15.84 pt baseline-to-baseline, matching the
// Lucronics oracle measured at 300 DPI (66 px). An earlier 0.682 set the pitch 0.24 pt too loose,
// losing ~1 line per page and driving the whole-book pagination drift. The driver takes a baseline
// distance, so the style carries box + leading, and the flow then lands on Typst's grid.
const LIBERTINUS_LINE_BOX_EM: f64 = 0.660;

/// Turns the raw config values into a [`Style`]. The leading is the one derived value: the config sets
/// a gap in ems, and the driver wants a baseline-to-baseline distance, so the Libertinus line box (see
/// [`LIBERTINUS_LINE_BOX_EM`]) is added to it -- the calibration that puts the line grid on the oracle's.
fn build_style(raw: &RawStyle) -> Style {
	let baseline = (LIBERTINUS_LINE_BOX_EM + raw.leading_em) * raw.body_pt;

	let mut style = Style::default();
	style.body_size	= Sp::from_pt(raw.body_pt);
	style.leading	= Sp::from_pt(baseline);
	style.para_skip	= Sp::from_pt(raw.par_skip_em * raw.body_pt);
	style.indent	= Sp::from_pt(raw.indent_em * raw.body_pt);
	style.chap_num_size	= Sp::from_pt(raw.chap_num_pt);
	style.chap_grid		= [
		Sp::from_pt(raw.chap_grid[0]),
		Sp::from_pt(raw.chap_grid[1]),
		Sp::from_pt(raw.chap_grid[2]),
		Sp::from_pt(raw.chap_grid[3]),
	];
	style.h1_size	= Sp::from_pt(raw.h1_pt);
	style.h2_size	= Sp::from_pt(raw.h2_pt);
	style.h3_size	= Sp::from_pt(raw.h3_pt);
	style.h4_size	= Sp::from_pt(raw.h4_pt);
	style
}

/// The string a `#let <name> = "..."` binds, if the config sets one as a plain literal.
fn read_let_string(src: &str, name: &str) -> Option<String> {
	let needle	= fmt!("#let {} =", name);
	let at		= src.find(&needle)?;
	let rest	= &src[at + needle.len()..];
	first_quoted(rest)
}

/// The body of the `if`/`else if` arm a `#let <name> = if format == "<fmt>" {...}` chain selects for
/// `fmt`. Bounds the search to the one `#let` so a later binding's arms are not read by mistake, finds
/// the arm whose condition tests this format, and returns its balanced `{...}` body.
fn arm(src: &str, name: &str, fmt: &str) -> Option<String> {
	let needle	= fmt!("#let {} =", name);
	let start	= src.find(&needle)?;
	let tail	= &src[start + needle.len()..];
	// The binding ends at the next top-level `#let`, or the end of the file.
	let end		= tail.find("\n#let ").unwrap_or(tail.len());
	let block	= &tail[..end];

	let cond	= fmt!("== \"{}\"", fmt);
	let at		= block.find(&cond)?;
	let after	= &block[at..];
	let brace	= after.find('{')?;
	balanced_braces(&after[brace..])
}

/// The contents of a `{...}` at the start of `s`, matched by brace depth so a nested record does not
/// close it early.
fn balanced_braces(s: &str) -> Option<String> {
	let bytes	= s.as_bytes();
	let mut depth	= 0i32;
	let mut i	= 0usize;
	while i < bytes.len() {
		match bytes[i] {
			b'{'	=> depth += 1,
			b'}'	=> {
				depth -= 1;
				if depth == 0 {
					return Some(s[1..i].to_string());
				}
			},
			_	=> {},
		}
		i += 1;
	}
	None
}

/// The first number after `key` in `s` -- the digits and one decimal point that follow the key. The
/// unit (`mm`, `pt`, `em`) is known from the key, so it is read off and dropped.
fn num_after(s: &str, key: &str) -> Option<f64> {
	let at	= s.find(key)?;
	first_num(&s[at + key.len()..])
}

/// The first number appearing anywhere in `s`, as an `f64` -- the leading numeric run after any
/// non-numeric lead-in. `11pt` and `0.75em` both read as their number.
fn first_num(s: &str) -> Option<f64> {
	let bytes	= s.as_bytes();
	let mut i	= 0usize;
	// Skip to the first digit or a decimal point that starts a number.
	while i < bytes.len() && !(bytes[i].is_ascii_digit() || bytes[i] == b'.') {
		i += 1;
	}
	let begin = i;
	while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
		i += 1;
	}
	if i == begin {
		return None;
	}
	s[begin..i].parse::<f64>().ok()
}

/// The numbers of the first `( ... )` tuple after `key` -- `sub-headings: (15pt, 12.5pt, ...)` reads as
/// `[15.0, 12.5, ...]`.
fn tuple_after(s: &str, key: &str) -> Option<Vec<f64>> {
	let at		= s.find(key)?;
	let after	= &s[at + key.len()..];
	let open	= after.find('(')?;
	let close	= after[open..].find(')')?;
	let inner	= &after[open + 1..open + close];
	let nums: Vec<f64> = inner.split(',').filter_map(first_num).collect();
	Some(nums)
}

// ┌───────────────────────────────────────────────────────────────────────────┐
// │ DOC TEMPLATE EXTRACTION HELPERS                                            │
// └───────────────────────────────────────────────────────────────────────────┘

/// The trim of a named paper, in millimetres. Only the sizes the doc trees reach for are tabulated;
/// an unknown name falls to A4, so a doc that sets an exotic paper still lands on a readable page.
fn paper_dims_mm(name: &str) -> (f64, f64) {
	match name {
		"a3"		=> (297.0, 420.0),
		"a4"		=> (210.0, 297.0),
		"a5"		=> (148.0, 210.0),
		"us-letter"	=> (215.9, 279.4),
		"us-legal"	=> (215.9, 355.6),
		_			=> (210.0, 297.0),
	}
}

/// The first `"..."` string after `key` anywhere in `src` -- `paper: "a4"` reads as `a4`. Used to read a
/// bare `name: "value"` setting that is not bounded by the field machinery the book path needs.
fn first_quoted_after(src: &str, key: &str) -> Option<String> {
	let at = src.find(key)?;
	first_quoted(&src[at + key.len()..])
}

/// The value bound to `field` inside a top-level `#let <dict> = ( ... )` dictionary -- the `a4:` entry of
/// the template's `#let margins = (a4: 2.5cm, ...)`, say. The dictionary is matched by paren depth from
/// the `#let`, and the field's value runs to the next depth-zero comma, so a nested group does not end it.
fn let_dict_field(src: &str, dict: &str, field: &str) -> Option<String> {
	let needle	= fmt!("#let {} =", dict);
	let start	= src.find(&needle)?;
	let tail	= &src[start + needle.len()..];
	let open	= tail.find('(')?;
	let body	= balanced_parens(&tail[open..])?;
	// Within the dictionary body, find `field:` and take its value up to the next top-level comma.
	let key		= fmt!("{}:", field);
	let at		= body.find(&key)?;
	let rest	= &body[at + key.len()..];
	let bytes	= rest.as_bytes();
	let mut depth	= 0i32;
	let mut end		= rest.len();
	let mut i		= 0usize;
	while i < bytes.len() {
		match bytes[i] as char {
			'(' | '[' | '{'		=> depth += 1,
			')' | ']' | '}'		=> depth -= 1,
			',' if depth == 0	=> { end = i; break; },
			_					=> {},
		}
		i += 1;
	}
	Some(rest[..end].trim().to_string())
}

/// A Typst length token as points: the leading number scaled by its unit (`cm`, `mm`, `in`, `pt`). A
/// bare number with no unit reads as points. `None` when no number leads the slice.
fn parse_len_pt(s: &str) -> Option<f64> {
	let n = first_num(s)?;
	let unit = s.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == '-' || c.is_whitespace());
	let per_pt = if unit.starts_with("cm") {
		10.0 * MM_PER_PT
	} else if unit.starts_with("mm") {
		MM_PER_PT
	} else if unit.starts_with("in") {
		72.0
	} else {
		1.0	// `pt` or unitless
	};
	Some(n * per_pt)
}

/// The first length after `key` in `src`, in points -- `text-size: 11pt` reads as `11.0`.
fn first_len_after(src: &str, key: &str) -> Option<f64> {
	let at = src.find(key)?;
	parse_len_pt(&src[at + key.len()..])
}

#[cfg(test)]
mod tests {
	use super::*;

	// A miniature two-format config with the shape the real books use: a `format` switch and a chain of
	// `if format == "..." {...}` arms per setting.
	const CFG: &str = r#"
#let format = "ingram-5x8"
#let page-dims = if format == "ingram-5x8" {
  (width: 127mm, height: 203mm)
} else {
  (width: 148mm, height: 210mm)
}
#let page-margins = if format == "ingram-5x8" {
  (inside: 17mm, outside: 15mm, top: 18mm, bottom: 18mm)
} else {
  (inside: 19mm, outside: 17mm, top: 19mm, bottom: 21mm)
}
#let body-text-size = if format == "ingram-5x8" { 11pt } else { 12pt }
#let body-line-spacing = if format == "ingram-5x8" { 0.75em } else { 0.75em }
#let body-par-spacing = if format == "ingram-5x8" { 0.75em } else { 0.75em }
#let type-scale = if format == "ingram-5x8" {
  ( title: 24pt, chapter-title: 20pt, sub-headings: (15pt, 12.5pt, 11.5pt, 11pt) )
} else {
  ( title: 27pt, chapter-title: 23pt, sub-headings: (17pt, 14.5pt, 13pt, 12.5pt) )
}
"#;

	#[test]
	fn test_the_selected_format_arm_is_read_00() -> Outcome<()> {
		let (geom, raw) = res!(read_config(CFG));
		// 127 mm and 203 mm in points, not the a5 fallback branch.
		assert_eq!(geom.width.to_pt().round() as i64, 360, "width should be 127 mm = 360 pt");
		assert_eq!(geom.height.to_pt().round() as i64, 575, "height should be 203 mm = 575 pt");
		// Mirror margins: inside binds wider than the fore-edge.
		assert_eq!(geom.inside.to_pt().round() as i64, 48, "inside 17 mm = 48 pt");
		assert_eq!(geom.outside.to_pt().round() as i64, 43, "outside 15 mm = 43 pt");
		assert!(geom.inside > geom.outside, "the binding margin is the wider of the two");
		assert!((raw.body_pt - 11.0).abs() < 1e-9, "body 11 pt, found {}", raw.body_pt);
		assert!((raw.h1_pt - 20.0).abs() < 1e-9, "h1 = chapter-title 20 pt, found {}", raw.h1_pt);
		// The level-2 heading takes sub-headings[1], not the first (section-title) entry.
		assert!((raw.h2_pt - 12.5).abs() < 1e-9, "h2 = sub-heading[1] 12.5 pt, found {}", raw.h2_pt);
		assert!((raw.h3_pt - 11.5).abs() < 1e-9, "h3 = sub-heading[2] 11.5 pt, found {}", raw.h3_pt);
		assert!((raw.h4_pt - 11.0).abs() < 1e-9, "h4 = sub-heading[3] 11 pt, found {}", raw.h4_pt);
		Ok(())
	}

	#[test]
	fn test_the_mirror_shift_moves_a_verso_to_the_fore_edge_01() -> Outcome<()> {
		let (geom, _) = res!(read_config(CFG));
		// Recto content starts at the inside margin; the verso shift lands it at the outside one.
		let verso_left = geom.content_left() + geom.mirror_shift();
		assert_eq!(verso_left, geom.outside, "a shifted verso page's left edge is the fore-edge margin");
		Ok(())
	}

	#[test]
	fn test_a_root_with_includes_reads_as_a_book_02() {
		assert!(is_book_root("#show: doc.with()\n#include \"chap_01.typ\"\n"));
		assert!(!is_book_root("= A lone heading\n\nSome prose.\n"));
	}

	#[test]
	fn test_a_typst_length_reads_in_points_04() -> Outcome<()> {
		assert!((res!(parse_len_pt("11pt").ok_or_else(|| err!("no number"; Test, Bug))) - 11.0).abs() < 1e-9);
		// 2.5 cm = 25 mm = 25 * 72 / 25.4 = 70.866 pt.
		let cm = res!(parse_len_pt("2.5cm").ok_or_else(|| err!("no number"; Test, Bug)));
		assert!((cm - 70.866).abs() < 1e-2, "2.5 cm should be ~70.87 pt, found {}", cm);
		let mm = res!(parse_len_pt("18mm").ok_or_else(|| err!("no number"; Test, Bug)));
		assert!((mm - 51.024).abs() < 1e-2, "18 mm should be ~51.02 pt, found {}", mm);
		// A bare number reads as points.
		assert!((res!(parse_len_pt("150").ok_or_else(|| err!("no number"; Test, Bug))) - 150.0).abs() < 1e-9);
		Ok(())
	}

	#[test]
	fn test_the_margin_dict_field_and_paper_read_05() -> Outcome<()> {
		let tmpl = r#"
#let margins = (
  a4: 2.5cm,
  title_page: 45%,
  section_header: 150pt,
)
#let doc(body) = {
  set page(paper: "a4", margin: (top: 0pt))
  body
}
"#;
		let a4 = res!(let_dict_field(tmpl, "margins", "a4").ok_or_else(|| err!("no a4 field"; Test, Bug)));
		assert_eq!(a4, "2.5cm", "the margins.a4 entry is the 2.5cm length");
		let paper = res!(first_quoted_after(tmpl, "paper:").ok_or_else(|| err!("no paper"; Test, Bug)));
		assert_eq!(paper, "a4", "the page paper is a4");
		let (w, h) = paper_dims_mm(&paper);
		assert_eq!((w as i64, h as i64), (210, 297), "a4 is 210 by 297 mm");
		Ok(())
	}

	#[test]
	fn test_doc_front_matter_reads_title_subtitle_author_06() {
		let root = r#"
#show: doc.with(
  title: [Austenite],
  subtitle: [Design Document],
  text-size: 11pt,
  meta-data: (
    ( version: "0.1.0", authors: "J. D. Hoogland", notes: "Initial." ),
  ),
)
= Purpose
"#;
		let raw = RawStyle {
			body_pt: 11.0, leading_em: 0.65, par_skip_em: 0.65, indent_em: 0.0,
			chap_num_pt: 54.0, chap_grid: [72.0, 8.0, 36.0, 20.0],
			h1_pt: 14.0, h2_pt: 12.0, h3_pt: 13.0, h4_pt: 12.0,
		};
		let fm = read_doc_front_matter(std::path::Path::new("/nonexistent"), root, &raw, "Austenite");
		assert_eq!(fm.title, "Austenite");
		assert_eq!(fm.subtitle.as_deref(), Some("Design Document"));
		assert_eq!(fm.author, "J. D. Hoogland");
		// A doc tree carries no book imprint (no ISBN), but it does compose the template's colophon: the
		// revision fields, the copyright line and the acknowledgement paragraph the meta page seats.
		assert!(fm.isbn.is_none(), "a doc tree carries no book imprint");
		assert_eq!(fm.meta_rows.len(), 1, "one revision row");
		assert_eq!(fm.meta_rows[0].version.as_deref(), Some("0.1.0"));
		assert_eq!(fm.meta_rows[0].notes.as_deref(), Some("Initial."));
		assert!(fm.copyright.as_deref().unwrap_or_default().contains("All rights reserved."),
			"a doc meta page carries a copyright line");
		assert!(fm.acknowledgement.is_some(), "a doc meta page carries an acknowledgement");
		// A doc tree always draws the template's two-column title page, so the sidebar is marked even when
		// the miniature root names no colour; the fraction falls to the template default with no template file.
		assert!(fm.sidebar_grey.is_some(), "a doc title page draws the sidebar");
		assert!((fm.sidebar_frac - 0.45).abs() < 1e-9, "the sidebar default fraction is 0.45");
	}

	#[test]
	fn test_doc_front_matter_reads_declaration_mark_07() {
		let root = r#"
#show: doc.with(
  title: [Austenite],
  footer-left-logo-path: "assets/svg/fe2o3_logo_text_right.svg",
  meta-data: (
    ( version: "0.1.0", date: "12026-08-08", authors: "J. D. Hoogland", declaration: "with-ai", notes: "Created." ),
  ),
)
= Purpose
"#;
		let raw = RawStyle {
			body_pt: 11.0, leading_em: 0.65, par_skip_em: 0.65, indent_em: 0.0,
			chap_num_pt: 54.0, chap_grid: [72.0, 8.0, 36.0, 20.0],
			h1_pt: 14.0, h2_pt: 12.0, h3_pt: 13.0, h4_pt: 12.0,
		};
		let fm = read_doc_front_matter(std::path::Path::new("/nonexistent"), root, &raw, "Austenite");
		assert_eq!(fm.meta_rows.len(), 1, "one revision row");
		let mr = &fm.meta_rows[0];
		assert_eq!(mr.date.as_deref(), Some("12026-08-08"));
		assert_eq!(mr.ai_mark_words.as_deref(), Some("Made with AI"));
		assert!(mr.ai_mark_path.as_deref().unwrap_or_default().ends_with("doc_made_with_ai_opt.svg"),
			"the with-ai slug picks the doc mark image");
		assert_eq!(fm.footer_logo.as_deref(), Some("assets/svg/fe2o3_logo_text_right.svg"));
	}

	#[test]
	fn test_doc_front_matter_reads_multiple_rows_and_custom_words_09() {
		let root = r#"
#show: doc.with(
  title: [Hematite],
  meta-data: (
    ( version: "2.0.0", date: "12026-04-11", authors: "J. D. Hoogland", declaration: "entirely-ai", declaration-words: "Additions made entirely with AI", notes: "Restructured." ),
    ( version: "1.0.0", date: "12023-10-01", authors: "J. D. Hoogland", declaration: "some-ai", notes: "Initial release." ),
  ),
)
= Purpose
"#;
		let raw = RawStyle {
			body_pt: 11.0, leading_em: 0.65, par_skip_em: 0.65, indent_em: 0.0,
			chap_num_pt: 54.0, chap_grid: [72.0, 8.0, 36.0, 20.0],
			h1_pt: 14.0, h2_pt: 12.0, h3_pt: 13.0, h4_pt: 12.0,
		};
		let fm = read_doc_front_matter(std::path::Path::new("/nonexistent"), root, &raw, "Hematite");
		assert_eq!(fm.meta_rows.len(), 2, "both revision rows are read");
		assert_eq!(fm.meta_rows[0].version.as_deref(), Some("2.0.0"));
		// A `declaration-words` rescopes the caption without changing the mark image.
		assert_eq!(fm.meta_rows[0].ai_mark_words.as_deref(), Some("Additions made entirely with AI"));
		assert!(fm.meta_rows[0].ai_mark_path.as_deref().unwrap_or_default().ends_with("doc_made_with_ai_entirely_opt.svg"));
		assert_eq!(fm.meta_rows[1].version.as_deref(), Some("1.0.0"));
		assert_eq!(fm.meta_rows[1].ai_mark_words.as_deref(), Some("Made with some AI"));
	}

	#[test]
	fn test_doc_front_matter_reads_title_page_logos_08() {
		let root = r#"
#show: doc.with(
  title: [Austenite],
  subtitle: [Design Document],
  title-colour: "lightgrey",
  title-smallcaps: true,
  title-top-logo-path: "assets/svg/austenite_logo_text_right.svg",
  title-top-logo-width: 150pt,
  title-bottom-logo-path: "assets/svg/oxedyne_logo_dark_text_below_opt.svg",
  title-bottom-logo-width: 120pt,
  footer-left-logo-path: "assets/svg/fe2o3_logo_text_right.svg",
  meta-data: (
    ( version: "0.1.0", authors: "J. D. Hoogland", notes: "Initial." ),
  ),
)
= Purpose
"#;
		let raw = RawStyle {
			body_pt: 11.0, leading_em: 0.65, par_skip_em: 0.65, indent_em: 0.0,
			chap_num_pt: 54.0, chap_grid: [72.0, 8.0, 36.0, 20.0],
			h1_pt: 14.0, h2_pt: 12.0, h3_pt: 13.0, h4_pt: 12.0,
		};
		let fm = read_doc_front_matter(std::path::Path::new("/nonexistent"), root, &raw, "Austenite");
		assert_eq!(fm.sidebar_grey, Some(240), "lightgrey resolves to luma 240");
		assert!(fm.title_smallcaps, "the title sets in small caps");
		assert_eq!(fm.top_logo.as_deref(), Some("assets/svg/austenite_logo_text_right.svg"));
		assert_eq!(fm.bottom_logo.as_deref(), Some("assets/svg/oxedyne_logo_dark_text_below_opt.svg"));
		assert_eq!(fm.footer_logo.as_deref(), Some("assets/svg/fe2o3_logo_text_right.svg"));
		assert!((fm.top_logo_width.to_pt() - 150.0).abs() < 1e-6, "top logo width 150 pt");
		assert!((fm.bottom_logo_width.to_pt() - 120.0).abs() < 1e-6, "bottom logo width 120 pt");
	}

	#[test]
	fn test_root_inline_markup_is_read_before_includes_07() -> Outcome<()> {
		let dir = std::path::Path::new("/nonexistent");
		// A doc root opens with an inline section written straight in the root, ahead of its includes (the
		// Austenite design's `= Purpose`). With no include present, only that inline markup is read; its
		// heading and paragraph must both survive, and the template call above them must be skipped, not set.
		let root = "#import \"template.typ\": *\n#show: doc.with(title: [X])\n\n= Purpose\n\nAustenite is an engine.\n";
		let (blocks, _skips) = res!(assemble(root, dir));
		assert!(
			blocks.iter().any(|b| matches!(b, Block::Heading { level: 1, .. })),
			"the root's inline level-1 heading is read into the flow");
		assert!(
			blocks.iter().any(|b| matches!(b, Block::Paragraph { .. } | Block::RichParagraph { .. })),
			"the inline paragraph beneath the heading is read too");
		Ok(())
	}

	#[test]
	fn test_a_part_page_divider_lifts_to_a_heading_03() -> Outcome<()> {
		let dir = std::path::Path::new("/nonexistent");
		let (blocks, _skips) = res!(assemble("#part-page(label: \"Part\")[The Pattern]\n", dir));
		assert_eq!(blocks.len(), 1, "one divider, one heading");
		match &blocks[0] {
			Block::Heading { level, segments, .. } => {
				assert_eq!(*level, 0, "a part divider is a level-0 heading, outside the chapter numbering");
				let title = segments.iter().map(|s| match s {
					Segment::Text(t)	=> t.clone(),
					_					=> String::new(),
				}).collect::<String>();
				assert_eq!(title, "The Pattern", "the title is the bracket body");
			},
			other => return Err(err!("expected a heading, found {:?}", other; Test, Bug)),
		}
		Ok(())
	}

	/// The term-dict reader picks the `#let term-dict = (...)` assignment, not the `// term-dict: ...`
	/// comment above it whose own parentheses would otherwise be read as the literal.
	#[test]
	fn test_term_dict_reader_skips_the_comment_04() {
		let src = r#"
// term-dict:  key -> display value (plain strings)
#let term-dict = (
  "org": "Elearnity Pty Ltd",
  "website": "elearnity.oxegen.io",
  "iniverse": "iniverse",
)
"#;
		let map = parse_term_dict(src);
		assert_eq!(map.get("org").map(String::as_str), Some("Elearnity Pty Ltd"));
		assert_eq!(map.get("website").map(String::as_str), Some("elearnity.oxegen.io"));
		assert_eq!(map.get("iniverse").map(String::as_str), Some("iniverse"));
		assert_eq!(map.len(), 3, "unexpected entries: {:?}", map);
	}

	/// A lone chapter finds a `refs.bib` in an ancestor directory, marks the key it cited, and returns a
	/// bibliography that resolves that key to an author-year citation rather than the raw key.
	#[test]
	fn test_lone_chapter_resolves_its_citation_05() -> Outcome<()> {
		// A unique scratch tree: refs.bib at the top, the chapter one level down, so the walk-up finds it.
		let base = std::env::temp_dir().join(fmt!("austenite-bibtest-{}",
			std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
				.map(|d| d.as_nanos()).unwrap_or(0)));
		let sub = base.join("chapters");
		res!(std::fs::create_dir_all(&sub));
		res!(std::fs::write(base.join("refs.bib"),
			"@article{smith2020, author = {Smith, John}, title = {A Title}, year = {2020}, journal = {J}}\n"));
		let chapter = sub.join("chap.typ");
		res!(std::fs::write(&chapter, "cited here"));

		let mut blocks = vec![Block::RichParagraph { segments: vec![Segment::Cite(vec!["smith2020".to_string()])] }];
		let bib = res!(load_lone_bibliography(&chapter, &mut blocks));

		// Clean up before asserting, so a failed assertion still leaves no scratch behind.
		let _ = std::fs::remove_dir_all(&base);

		let bib = res!(bib.ok_or_else(|| err!("no bibliography was found beside the chapter"; Test, Missing)));
		let cite = res!(bib.format_citation(&["smith2020"]));
		assert!(cite.contains("Smith") && cite.contains("2020"),
			"citation did not resolve to author-year: {:?}", cite);
		assert!(!cite.contains("smith2020"), "the raw cite key leaked: {:?}", cite);
		Ok(())
	}
}

