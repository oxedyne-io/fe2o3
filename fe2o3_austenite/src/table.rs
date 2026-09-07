//! Tables: a ruled grid of cells, laid out as one keep box in the vertical flow.
//!
//! A table is authored as [`Table`] -- rows of [`Cell`]s -- and [`lower`] turns it into a single
//! [`Node::VBox`], so the driver's greedy page breaker moves the whole table to the next page when it
//! will not fit where it stands. Column widths are measured from the cell text and, when the natural
//! widths overrun the measure, shrunk proportionally; a cell too wide for its column is wrapped with
//! [`break_paragraph`](crate::linebreak::break_paragraph) at the column width, exactly as a paragraph
//! is wrapped at the measure.
//!
//! One fact a reader could not derive, and the reason the layout looks the way it does. The driver
//! renders a box *nested inside an HBox* as a placeholder rectangle -- only leaves (glyph runs and
//! rules) draw as ink there -- so a cell's glyphs cannot be a nested box. A row is therefore
//! decomposed into horizontal *bands*: each band is one HBox holding, positioned by glue, the
//! vertical rules at the column boundaries and one wrapped line from each cell. Bands stack with no
//! gap, so the per-band rule segments tile into continuous column rules, and a cell that wraps to
//! several lines simply contributes to several bands. The whole grid is thus leaves in HBoxes stacked
//! in one VBox, which is the only shape the driver draws as real glyphs throughout.

use crate::doc::{
	Segment,
	Style,
	superscript,
};
use crate::font::ShapedText;
use crate::ir::{
	BoxNode,
	Dims,
	DrawOp,
	Glue,
	Graphic,
	Leaf,
	Node,
	Sp,
};
use crate::linebreak::{
	Piece,
	break_paragraph_pieces,
};
use crate::math;

use oxedyne_fe2o3_core::prelude::*;
use oxedyne_fe2o3_font::{
	face::Role,
	set::FontSet,
	shape::Dir,
};
use oxedyne_fe2o3_graphics::{
	colour::Rgba,
	path::{
		Bounds,
		Path,
	},
};

use std::collections::HashMap;
use std::sync::Arc;

/// How a cell's text sits within its column width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
	Left,
	Centre,
	Right,
}

/// An image mark stacked beneath a cell's text: the meta page's "Made with AI" chip and its caption. The
/// chip is a clickable link to the declaration's scheme page (`url`); the PDF writer draws a link
/// annotation over it, while the SVG writer sets it as a plain image. Only the image is the link, the
/// caption beside it plain, matching the template. The mark occupies its own band under the cell's last
/// text line, the image seated left with the caption beside it.
#[derive(Clone, Debug)]
pub struct CellMark {
	pub path:	String,			// the mark image, resolved against the image base directory
	pub height:	Sp,				// the drawn image height
	pub words:	String,			// the caption set beside the image
	pub url:	Option<String>,	// the declaration scheme page the chip links to
}

/// One cell: its inline content, and how that content aligns in the column. A cell carries a run of
/// [`Segment`]s -- a bold header, an italic word, a superscript dagger or an in-cell maths span each set
/// with its own face -- broken to the column width exactly as a rich paragraph is broken to the measure.
#[derive(Clone, Debug)]
pub struct Cell {
	pub content:	Vec<Segment>,
	pub align:		Align,
	pub mark:		Option<CellMark>,	// an image mark stacked beneath the text, the meta page's AI chip
}

impl Cell {
	pub fn new<S: Into<String>>(text: S) -> Self {
		Self { content: vec![Segment::text(text)], align: Align::Left, mark: None }
	}

	pub fn aligned<S: Into<String>>(text: S, align: Align) -> Self {
		Self { content: vec![Segment::text(text)], align, mark: None }
	}

	/// A cell carrying a run of rich segments -- the form the reader builds from a Typst cell's markup.
	pub fn rich(content: Vec<Segment>, align: Align) -> Self {
		Self { content, align, mark: None }
	}

	/// A cell carrying a run of rich segments and an image mark stacked beneath them -- the meta page's
	/// author cell, whose name is followed by the "Made with AI" chip and its caption.
	pub fn rich_with_mark(content: Vec<Segment>, align: Align, mark: CellMark) -> Self {
		Self { content, align, mark: Some(mark) }
	}
}

/// One row of cells. A row shorter than the widest row is padded with empty cells when the table is
/// laid out, so a ragged authoring is legal.
#[derive(Clone, Debug)]
pub struct Row {
	pub cells:	Vec<Cell>,
}

impl Row {
	pub fn new(cells: Vec<Cell>) -> Self {
		Self { cells }
	}
}

/// A table: its rows, and whether the first row is a header (set bold, with a heavier rule beneath
/// it). Spanning cells and a caption with a "Table N" number are later additions.
#[derive(Clone, Debug)]
pub struct Table {
	pub rows:		Vec<Row>,
	pub header:		bool,
	pub weights:	Vec<f64>,		// per-column fractional (`fr`) weight; 0.0 for a content-sized column, empty for none
	pub text_size:	Option<Sp>,		// the `text(size: Npt)` wrapper's size, so a small-set table sets small; body size when none
	pub inset:		Option<Sp>,		// the `inset:` cell padding, overriding the style default when set
}

impl Table {
	pub fn new(header: bool, rows: Vec<Row>) -> Self {
		Self { rows, header, weights: Vec::new(), text_size: None, inset: None }
	}

	/// A table with declared fractional column weights, reproducing Typst's `columns: (2fr, 5fr, ...)`
	/// sizing; a `0.0` weight leaves that column sized to its content.
	pub fn with_weights(header: bool, rows: Vec<Row>, weights: Vec<f64>) -> Self {
		Self { rows, header, weights, text_size: None, inset: None }
	}

	/// A grid of string rows, every cell left-aligned; the first row a header when `header`.
	pub fn grid(header: bool, rows: Vec<Vec<&str>>) -> Self {
		let rows = rows.into_iter()
			.map(|r| Row::new(r.into_iter().map(Cell::new).collect()))
			.collect();
		Self { rows, header, weights: Vec::new(), text_size: None, inset: None }
	}
}

/// One wrapped line of a cell: the shaped leaves and justifying glue as `break_paragraph` set them,
/// with the natural extent kept so a band can align and stack them.
struct CellLine {
	children:	Vec<Node>,
	width:		Sp,
	height:		Sp,
	depth:		Sp,
}

/// Lowers a table to one keep box. The measure is the width a table spanning the full text block may
/// use; a table whose natural columns are narrower than the measure is set narrower, flush left.
pub fn lower(
	fonts:		Arc<FontSet>,
	style:		Style,
	measure:	Sp,
	table:		&Table,
	refs:		&HashMap<String, String>,
)
	-> Outcome<Node>
{
	// A `text(size: Npt)` wrapper sets the whole table at its reduced size (the books' claim tables set at
	// 7 pt), and an explicit `inset:` overrides the cell padding on both axes; the interline gap within a
	// cell scales with the text so a small table sets tight. The rules and header wash keep the style's.
	let size		= table.text_size.unwrap_or(style.body_size);
	let scale		= size.raw() as f64 / style.body_size.raw().max(1) as f64;
	let pad_x		= table.inset.unwrap_or(style.cell_pad_x);
	let pad_y		= table.inset.unwrap_or(style.cell_pad_y);
	let line_gap	= if table.text_size.is_some() {
		Sp::from_pt(style.line_gap.to_pt() * scale)
	} else {
		style.line_gap
	};
	let cell_leading	= Sp::from_pt(size.to_pt() * 1.2);	// dropped by `break_cell`, but a sane interline base
	let rows	= &table.rows;
	if rows.is_empty() {
		return Err(err!("A table needs at least one row."; Input, Invalid, Missing));
	}
	let ncols = rows.iter().map(|r| r.cells.len()).max().unwrap_or(0);
	if ncols == 0 {
		return Err(err!("A table needs at least one column."; Input, Invalid, Missing));
	}

	// The vertical rules: a heavier pen frames the grid, a lighter one divides the columns. Their
	// widths take real horizontal space, so they are budgeted before the columns are sized.
	let mut tv = vec![style.rule_thin; ncols + 1];
	tv[0]		= style.rule_thick;
	tv[ncols]	= style.rule_thick;
	let vrule_total: i32 = tv.iter().map(|s| s.raw()).sum();

	// The text width left for the columns after the padding either side of every cell and the rules
	// between them are taken out of the measure.
	let pad2		= pad_x.raw() * 2 * ncols as i32;
	let available	= measure.raw() - pad2 - vrule_total;
	if available <= 0 {
		return Err(err!(
			"A table of {} columns leaves no width for text within the measure of {} sp; \
			reduce the columns or the padding.", ncols, measure.raw(); Input, Invalid, TooBig));
	}

	// Each cell's inline content is built once into the pieces the line breaker weaves -- a run of shaped
	// text in its face, a maths cluster, a superscript mark -- and reused for measuring the columns and for
	// the final wrap, so a cell shapes its faces only once. The base role per row is bold in a header row,
	// so a plain header label still sets bold, and body elsewhere.
	let (piece_grid, bases) = res!(build_grid(fonts.clone(), style, table, ncols, refs));
	let colwidth = res!(size_columns(
		fonts.clone(), size, &piece_grid, &bases, ncols, available, &table.weights));

	// Column boundary positions, the running x of each vertical rule and each cell's text left.
	let mut vrule_left	= vec![Sp::ZERO; ncols + 1];
	let mut cx			= Sp::ZERO;
	for b in 0..=ncols {
		vrule_left[b]	= cx;
		cx				= cx + tv[b];
		if b < ncols {
			cx = cx + Sp(pad_x.raw() * 2) + colwidth[b];
		}
	}
	let table_width = cx;
	let mut content_left = vec![Sp::ZERO; ncols];
	for c in 0..ncols {
		content_left[c] = vrule_left[c] + tv[c] + pad_x;
	}

	// Wrap every cell to its column, and note the tallest stack of lines in each row.
	let mut grid:	Vec<Vec<Vec<CellLine>>>	= Vec::with_capacity(rows.len());
	let mut nbands:	Vec<usize>				= Vec::with_capacity(rows.len());
	for r in 0..rows.len() {
		let mut cells = Vec::with_capacity(ncols);
		let mut bands = 0usize;
		for c in 0..ncols {
			let mut lines = res!(break_cell(
				fonts.clone(), bases[r], size, &piece_grid[r][c], colwidth[c], cell_leading));
			// An image mark stacked beneath the cell's text takes its own band under the last line. A mark
			// whose image will not load leaves the cell to its text alone, as a title logo degrades.
			if let Some(mark) = rows[r].cells.get(c).and_then(|cell| cell.mark.as_ref()) {
				if let Ok(line) = mark_line(fonts.clone(), size, mark) {
					lines.push(line);
				}
			}
			bands = bands.max(lines.len());
			cells.push(lines);
		}
		grid.push(cells);
		nbands.push(bands.max(1));	// an all-empty row still occupies one line band
	}

	// A fallback line extent for a band no cell fills, so an empty row keeps a sensible height.
	let sample	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, size, "Ag"));
	let default_v	= sample.dims().height + sample.dims().depth;

	let mut children:	Vec<Node> = Vec::new();
	let mut total_h		= Sp::ZERO;

	// The top frame.
	push_hrule(&mut children, &mut total_h, table_width, style.rule_thick);

	for r in 0..rows.len() {
		// A header row carries a grey wash behind every one of its bands, drawn before the rules and text
		// so they sit over it; a body row has none.
		let fill = if table.header && r == 0 { Some(style.header_fill) } else { None };

		// The top padding of the row: a rules-only band, so the first line's baseline clears the rule.
		let empty:	Vec<Option<&CellLine>>	= vec![None; ncols];
		let flush:	Vec<Align>				= vec![Align::Left; ncols];
		let pad_band = build_band(
			ncols, &vrule_left, &tv, &content_left, &colwidth,
			pad_y, table_width, &empty, &flush, fill);
		children.push(pad_band);
		total_h += pad_y;

		let bands = nbands[r];
		for k in 0..bands {
			// The band height is the tallest line at this level, plus a trailing gap -- interline
			// leading between lines, the bottom cell padding after the last.
			let mut lh = Sp::ZERO;
			for c in 0..ncols {
				if let Some(cl) = grid[r][c].get(k) {
					let v = cl.height + cl.depth;
					if v > lh { lh = v; }
				}
			}
			if lh.raw() == 0 { lh = default_v; }
			let trailing	= if k + 1 < bands { line_gap } else { pad_y };
			let bh			= lh + trailing;

			let mut opt:	Vec<Option<&CellLine>>	= Vec::with_capacity(ncols);
			let mut aligns:	Vec<Align>				= Vec::with_capacity(ncols);
			for c in 0..ncols {
				opt.push(grid[r][c].get(k));
				aligns.push(rows[r].cells.get(c).map_or(Align::Left, |cell| cell.align));
			}

			let hb = build_band(
				ncols, &vrule_left, &tv, &content_left, &colwidth,
				bh, table_width, &opt, &aligns, fill);
			children.push(hb);
			total_h += bh;
		}

		// The rule under the row: heavy beneath a header and at the very foot, light between body rows.
		let th = if table.header && r == 0 {
			style.rule_thick
		} else if r + 1 == rows.len() {
			style.rule_thick
		} else {
			style.rule_thin
		};
		push_hrule(&mut children, &mut total_h, table_width, th);
	}

	let dims = Dims::new(table_width, total_h, Sp::ZERO);
	Ok(Node::VBox(BoxNode::new(children, dims)))
}

/// Builds a cell's image mark as one line: the image seated at the left, then a gap, then the caption
/// beside it. The image is drawn at the mark's fixed height and the caption set a little smaller; the
/// caption is lowered so it sits about the image's vertical middle rather than clinging to its top, since
/// a band tops-aligns its leaves. The line's extent is the image height, so its band clears the chip.
fn mark_line(
	fonts:	Arc<FontSet>,
	size:	Sp,
	mark:	&CellMark,
)
	-> Outcome<CellLine>
{
	let mut graphic	= res!(crate::doc::image_at_height(&fonts, &mark.path, mark.height.to_pt()));
	// The chip carries the declaration's scheme link; the PDF writer draws an annotation over its box.
	if let Some(url) = &mark.url {
		graphic = graphic.with_link(url.clone());
	}
	let img		= Leaf::graphic(graphic);
	let img_w	= img.dims.width;
	let img_h	= img.dims.height;

	let cap_size	= Sp(size.raw() * 82 / 100);
	let shaped		= res!(ShapedText::new(fonts, Role::Body, Dir::Ltr, cap_size, &mark.words));
	let cd			= shaped.dims();
	let mut words	= Leaf::text(shaped);
	// Lower the caption to about the image's vertical middle; the band tops-aligns its leaves, so without
	// the shift the words would cling to the chip's top edge.
	let drop		= (img_h.raw() - cd.height.raw()).max(0) / 2;
	words.shift		= Sp(drop);

	let gap		= Sp::from_pt(6.0);
	let children	= vec![Node::Leaf(img), Node::Glue(Glue::fixed(gap)), Node::Leaf(words)];
	let width	= img_w + gap + cd.width;
	Ok(CellLine { children, width, height: img_h, depth: Sp::ZERO })
}

/// The base role a row's plain text sets in: bold for a header row, the body face otherwise. Authored
/// emphasis within a cell keeps its own face over this base.
fn base_role(table: &Table, r: usize) -> Role {
	if table.header && r == 0 { Role::Bold } else { Role::Body }
}

/// Builds every cell's pieces once, and the base role of each row. A missing cell (a ragged row shorter
/// than the widest) gives an empty piece list, so the column simply carries nothing there.
fn build_grid(
	fonts:	Arc<FontSet>,
	style:	Style,
	table:	&Table,
	ncols:	usize,
	refs:	&HashMap<String, String>,
)
	-> Outcome<(Vec<Vec<Vec<Piece>>>, Vec<Role>)>
{
	let mut grid:	Vec<Vec<Vec<Piece>>>	= Vec::with_capacity(table.rows.len());
	let mut bases:	Vec<Role>				= Vec::with_capacity(table.rows.len());
	for (r, row) in table.rows.iter().enumerate() {
		let base = base_role(table, r);
		bases.push(base);
		let mut cols = Vec::with_capacity(ncols);
		for c in 0..ncols {
			let pieces = match row.cells.get(c) {
				Some(cell)	=> res!(cell_pieces(fonts.clone(), style, &cell.content, base, refs)),
				None		=> Vec::new(),
			};
			cols.push(pieces);
		}
		grid.push(cols);
	}
	Ok((grid, bases))
}

/// Turns a cell's rich segments into the pieces the line breaker weaves. Plain text takes the row's base
/// role -- a header cell's bold, a body cell's body; `*strong*`, `_emph_`, a `#super[...]`, inline code
/// and an in-cell maths span keep their own faces, so a cell sets exactly as a run of prose would. A
/// footnote or a cross-reference in a cell -- rare -- is not set here; a citation falls back to its keys.
fn cell_pieces(
	fonts:		Arc<FontSet>,
	style:		Style,
	segments:	&[Segment],
	base:		Role,
	refs:		&HashMap<String, String>,
)
	-> Outcome<Vec<Piece>>
{
	let size = style.body_size;
	let mut pieces = Vec::with_capacity(segments.len());
	for seg in segments {
		match seg {
			Segment::Text(t)		=> pieces.push(Piece::Text { text: t.clone(), role: base }),
			Segment::Strong(t)		=> pieces.push(Piece::Text { text: t.clone(), role: Role::Bold }),
			Segment::Emph(t)		=> pieces.push(Piece::Text { text: t.clone(), role: emph_role(base) }),
			Segment::BoldItalic(t)	=> pieces.push(Piece::Text { text: t.clone(), role: Role::BoldItalic }),
			Segment::Code(t)		=> pieces.push(Piece::Text { text: t.clone(), role: Role::Mono }),
			Segment::Glossary { display, .. }
									=> pieces.push(Piece::Text { text: display.clone(), role: base }),
			Segment::Cite(keys)		=> pieces.push(Piece::Text { text: fmt!("({})", keys.join("; ")), role: base }),
			Segment::PageRef(label) => {
				// A cross-reference resolves to Typst's own supplement-and-number text -- "Chapter 13",
				// "Table 1" -- fixed by the document-order pre-pass and set as the cell's base face. A label
				// the pre-pass did not record sets nothing; a cell has no reserved page-number slot the driver
				// could later fill, so an unresolved reference is dropped rather than left as a gap.
				if let Some(text) = refs.get(label) {
					pieces.push(Piece::Text { text: text.clone(), role: base });
				}
			},
			Segment::Footnote { .. }	=> {},	// a footnote in a cell is not set at this increment
			Segment::Super(t) => {
				let (shaped, dims) = res!(superscript(fonts.clone(), base, size, t));
				pieces.push(Piece::Mark(Leaf::text_dims(shaped, dims)));
			},
			Segment::Math(expr) => {
				// The inline box is flattened to leaves and glue by the maths layout; its children weave into
				// the line as real glyphs, its baseline seated on the text baseline.
				let node = res!(math::layout(fonts.clone(), &style, expr, false));
				if let Node::HBox(b) = node {
					let ascent	= res!(ShapedText::new(
						fonts.clone(), base, Dir::Ltr, size, "0")).dims().height;
					let over	= if b.dims.height > ascent { b.dims.height - ascent } else { Sp::ZERO };
					pieces.push(Piece::Math {
						nodes:	b.list,
						width:	b.dims.width,
						height:	ascent,
						depth:	b.dims.depth,
						over,
					});
				}
			},
		}
	}
	Ok(pieces)
}

/// The face an emphasised run takes over a base: bold-italic within a header (whose base is bold), plain
/// italic elsewhere.
fn emph_role(base: Role) -> Role {
	if base == Role::Bold { Role::BoldItalic } else { Role::Italic }
}

/// Assigns each column a text width. Each column asks for its widest cell's natural width; when the
/// columns together fit the available width they keep it (the table sets narrower than the measure),
/// and when they overrun it they are shrunk. The shrink holds every column at no less than its widest
/// single word and shares the remaining width in proportion to how much each wanted above that
/// minimum; if even the minimums overrun, the columns shrink in proportion to their natural widths
/// and an over-long word is left to run under a rule. It does not know which column would most repay
/// extra width, and it cannot span a cell across columns -- both later refinements.
fn size_columns(
	fonts:		Arc<FontSet>,
	size:		Sp,
	grid:		&[Vec<Vec<Piece>>],
	bases:		&[Role],
	ncols:		usize,
	available:	i32,
	weights:	&[f64],
)
	-> Outcome<Vec<Sp>>
{
	let mut natural	= vec![0i64; ncols];
	let mut minw	= vec![0i64; ncols];
	for (r, row) in grid.iter().enumerate() {
		let base = bases[r];
		for c in 0..ncols {
			let pieces = &row[c];
			if pieces.is_empty() {
				continue;
			}
			// The natural width is the cell set on one line; the minimum is the widest line once the cell is
			// broken as hard as it can be, the least the column can shrink to before a word must protrude.
			let nat = res!(measure_cell(fonts.clone(), base, size, pieces, Sp::from_pt(100_000.0)));
			natural[c] = natural[c].max(nat.raw() as i64);
			let lw = res!(measure_cell(fonts.clone(), base, size, pieces, Sp(1)));
			minw[c] = minw[c].max(lw.raw() as i64);
		}
	}

	let avail			= available as i64;

	// A declared `columns: (2fr, 5fr, ...)` track list sizes the columns fractionally, as Typst does: a
	// content-sized (`auto`, weight 0) column takes its natural one-line width first, and the `fr` columns
	// share the width left over in proportion to their weights. This is what a table authored with an `fr`
	// spec expects, and it keeps every column on the oracle's proportions rather than sizing each from its
	// own widest cell -- which drifted a column a hair too narrow and left a rich cell overrunning its rule.
	if weights.iter().any(|&w| w > 0.0) {
		let mut colwidth	= vec![Sp::ZERO; ncols];
		let mut fixed		= 0i64;
		for c in 0..ncols {
			if weights.get(c).copied().unwrap_or(0.0) <= 0.0 {
				colwidth[c]	= Sp(natural[c] as i32);
				fixed		+= natural[c];
			}
		}
		let remaining	= (avail - fixed).max(0);
		let wsum:	f64	= weights.iter().filter(|&&w| w > 0.0).sum();
		let mut acc		= 0i64;
		let mut last_fr	= 0usize;
		if wsum > 0.0 {
			for c in 0..ncols {
				let w = weights.get(c).copied().unwrap_or(0.0);
				if w > 0.0 {
					let cw = (remaining as f64 * w / wsum) as i64;
					colwidth[c]	= Sp(cw as i32);
					acc			+= cw;
					last_fr		= c;
				}
			}
			// The rounding remainder lands on the last fractional column, so the widths sum to the budget.
			colwidth[last_fr] = Sp(colwidth[last_fr].raw() + (remaining - acc) as i32);
		}
		return Ok(colwidth);
	}

	let total_natural:	i64 = natural.iter().sum();
	let mut colwidth	= vec![Sp::ZERO; ncols];

	if total_natural == 0 || total_natural <= avail {
		for c in 0..ncols {
			colwidth[c] = Sp(natural[c] as i32);
		}
		return Ok(colwidth);
	}

	let sum_min: i64 = minw.iter().sum();
	let mut acc = 0i64;
	if sum_min < avail {
		let extra	= avail - sum_min;
		let span	= total_natural - sum_min;
		for c in 0..ncols {
			let give = if span > 0 {
				((natural[c] - minw[c]) as i128 * extra as i128 / span as i128) as i64
			} else {
				0
			};
			let w = minw[c] + give;
			colwidth[c] = Sp(w as i32);
			acc += w;
		}
	} else {
		// Even the minimums overrun the measure: proportional to natural width, words may protrude.
		for c in 0..ncols {
			let w = natural[c] * avail / total_natural;
			colwidth[c] = Sp(w as i32);
			acc += w;
		}
	}
	// The rounding remainder lands on the last column, so the widths sum to the budget exactly.
	let last = ncols - 1;
	colwidth[last] = Sp(colwidth[last].raw() + (avail - acc) as i32);
	Ok(colwidth)
}

/// The widest line a cell's pieces set to when broken at `measure`: at a large measure this is the cell's
/// natural one-line width, at a tiny one the least it can shrink to. Zero for an empty cell.
fn measure_cell(
	fonts:		Arc<FontSet>,
	base:		Role,
	size:		Sp,
	pieces:		&[Piece],
	measure:	Sp,
)
	-> Outcome<Sp>
{
	let mut m = Sp::ZERO;
	for line in res!(break_cell(fonts, base, size, pieces, measure, size)) {
		if line.width > m {
			m = line.width;
		}
	}
	Ok(m)
}

/// Breaks a cell's pieces to its column, reusing the rich-paragraph line breaker at the column width, so a
/// cell's faces, superscripts and maths flow exactly as a paragraph's do. Each returned HBox is one line;
/// its leaves and glue are kept, with the natural width summed for alignment. The interline glue the
/// breaker inserts is dropped -- a band supplies its own vertical spacing. An empty cell yields no lines.
fn break_cell(
	fonts:		Arc<FontSet>,
	base:		Role,
	size:		Sp,
	pieces:		&[Piece],
	colwidth:	Sp,
	leading:	Sp,
)
	-> Outcome<Vec<CellLine>>
{
	if pieces.is_empty() {
		return Ok(Vec::new());
	}
	// A cell is set ragged (`justify = false`): every space keeps its natural width, so the band's own
	// justification to the table width -- for which the cells would otherwise hold the only stretchable
	// glue -- cannot stretch or collapse the words within a cell. Typst sets table cells left-aligned.
	let nodes = res!(break_paragraph_pieces(fonts.clone(), base, Dir::Ltr, size, pieces, colwidth, leading, false));
	let mut out = Vec::new();
	for n in nodes {
		if let Node::HBox(b) = n {
			let mut w = Sp::ZERO;
			for ch in &b.list {
				match ch {
					Node::Leaf(l)	=> w = w + l.dims.width,
					Node::Glue(g)	=> w = w + g.natural,
					_				=> (),
				}
			}
			out.push(CellLine { children: b.list, width: w, height: b.dims.height, depth: b.dims.depth });
		}
	}
	// The ragged line breaker rejects a short interior line whose slack exceeds its stretch tolerance, so
	// a narrow column of few-word cells can leave a cell it cannot break within tolerance set overfull --
	// one line that overruns the column rule into the next cell. When that happens, re-wrap the cell
	// greedily instead: pack words up to the column width, breaking before any word that would overrun.
	// The greedy pass handles only plain-text pieces (a superscript mark included); a cell carrying an
	// in-cell maths cluster keeps the breaker's result, since a maths box must be woven into the line as
	// leaves rather than nested as one box (which the driver would draw as a placeholder rectangle).
	let overfull = out.iter().any(|l| l.width > colwidth);
	let has_math = pieces.iter().any(|p| matches!(p, Piece::Math { .. }));
	if overfull && !has_math {
		return greedy_break_cell(fonts, base, size, pieces, colwidth);
	}
	Ok(out)
}

/// One shaped word (or a superscript mark) with the breakable space that precedes it, ready to be packed
/// into greedy ragged lines.
struct Unit {
	node:			Node,
	width:			Sp,
	height:			Sp,
	depth:			Sp,
	space_before:	Sp,	// the natural width of the breakable space before this unit, zero when it is glued to the previous
}

/// Wraps a plain-text cell greedily: each word is placed on the current line while it fits the column,
/// and a word that would overrun starts a fresh line. This is the ragged-right fallback for a cell the
/// optimal breaker leaves overfull -- a narrow column whose slack outruns the breaker's stretch tolerance
/// -- and it never sets a line wider than the column unless a single word is itself wider than the column.
fn greedy_break_cell(
	fonts:		Arc<FontSet>,
	base:		Role,
	size:		Sp,
	pieces:		&[Piece],
	colwidth:	Sp,
)
	-> Outcome<Vec<CellLine>>
{
	let space	= res!(ShapedText::new(fonts.clone(), base, Dir::Ltr, size, " "));
	let sp_w	= space.dims().width;

	// Flatten the pieces into shaped word units, carrying the breakable space that precedes each. A run of
	// spaces between two words is one breakable space; a mark clings to the word before it with no space.
	let mut units:		Vec<Unit>	= Vec::new();
	let mut pending:	i32			= 0;	// spaces seen but not yet attached to the next word
	for piece in pieces {
		match piece {
			Piece::Text { text, role } => {
				let mut word = String::new();
				for ch in text.chars() {
					if ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' {
						if !word.is_empty() {
							res!(push_word_unit(&mut units, fonts.clone(), *role, size, &word, Sp(sp_w.raw() * pending)));
							word.clear();
							pending = 0;
						}
						if ch == ' ' {
							pending += 1;
						}
					} else {
						word.push(ch);
					}
				}
				if !word.is_empty() {
					res!(push_word_unit(&mut units, fonts.clone(), *role, size, &word, Sp(sp_w.raw() * pending)));
					pending = 0;
				}
			},
			Piece::Mark(leaf) => {
				units.push(Unit {
					node:			Node::Leaf(leaf.clone()),
					width:			leaf.dims.width,
					height:			leaf.dims.height,
					depth:			leaf.dims.depth,
					space_before:	Sp::ZERO,	// a mark clings to the word before it
				});
				pending = 0;
			},
			// A maths piece never reaches here: `break_cell` keeps the breaker's result for a cell with maths.
			Piece::Math { .. } => {},
		}
	}

	let mut out:		Vec<CellLine>	= Vec::new();
	let mut children:	Vec<Node>		= Vec::new();
	let mut cur_w		= Sp::ZERO;
	let mut cur_h		= Sp::ZERO;
	let mut cur_d		= Sp::ZERO;
	for unit in units {
		let non_empty	= !children.is_empty();
		let sb			= if non_empty { unit.space_before } else { Sp::ZERO };
		// Break before this word when the line already holds something and the word (with its space) would
		// overrun the column. A word wider than the column on its own line is left to overrun, as there is
		// nothing narrower to set it to.
		if non_empty && cur_w + sb + unit.width > colwidth {
			out.push(CellLine { children: std::mem::take(&mut children), width: cur_w, height: cur_h, depth: cur_d });
			cur_w = Sp::ZERO;
			cur_h = Sp::ZERO;
			cur_d = Sp::ZERO;
		}
		if !children.is_empty() && sb.raw() > 0 {
			children.push(Node::Glue(Glue::fixed(sb)));
			cur_w = cur_w + sb;
		}
		children.push(unit.node);
		cur_w = cur_w + unit.width;
		if unit.height > cur_h { cur_h = unit.height; }
		if unit.depth  > cur_d { cur_d = unit.depth; }
	}
	if !children.is_empty() {
		out.push(CellLine { children, width: cur_w, height: cur_h, depth: cur_d });
	}
	Ok(out)
}

/// Shapes one word in its face and appends it as a [`Unit`] with the breakable space that precedes it.
fn push_word_unit(
	units:			&mut Vec<Unit>,
	fonts:			Arc<FontSet>,
	role:			Role,
	size:			Sp,
	word:			&str,
	space_before:	Sp,
)
	-> Outcome<()>
{
	let shaped	= res!(ShapedText::new(fonts, role, Dir::Ltr, size, word));
	let d		= shaped.dims();
	units.push(Unit {
		node:	Node::Leaf(Leaf::text(shaped)),
		width:	d.width,
		height:	d.height,
		depth:	d.depth,
		space_before,
	});
	Ok(())
}

/// Builds one band: an HBox carrying the vertical rules at every column boundary, each the band's own
/// height, and one line from each cell placed at its column and alignment. The x cursor is tracked as
/// the driver will track it, so every rule lands on its fixed boundary whatever the lines do -- an
/// over-long line runs under the next rule rather than displacing it, keeping the columns straight
/// from row to row.
#[allow(clippy::too_many_arguments)]
fn build_band(
	ncols:			usize,
	vrule_left:		&[Sp],
	tv:				&[Sp],
	content_left:	&[Sp],
	colwidth:		&[Sp],
	band_height:	Sp,
	table_width:	Sp,
	lines:			&[Option<&CellLine>],
	aligns:			&[Align],
	fill:			Option<Rgba>,
)
	-> Node
{
	let mut kids:	Vec<Node>	= Vec::new();
	let mut cursor				= Sp::ZERO;

	// The row wash sits behind the band: a zero-width graphic leaf drawn first, so it does not advance the
	// horizontal cursor the rules and lines are positioned against, yet its path spans the whole band and
	// paints under them (the writer draws in frame order, so the rules and glyphs that follow sit over it).
	if let Some(colour) = fill {
		if let Some(g) = fill_band(table_width, band_height, colour) {
			kids.push(Node::Leaf(g));
		}
	}

	for b in 0..=ncols {
		// The vertical rule at this boundary, seated on its fixed x.
		kids.push(Node::Glue(Glue::fixed(vrule_left[b] - cursor)));
		kids.push(Node::Leaf(Leaf::rule(Dims::new(tv[b], band_height, Sp::ZERO))));
		cursor = vrule_left[b] + tv[b];

		if b < ncols {
			if let Some(line) = lines[b] {
				let slack	= (colwidth[b].raw() - line.width.raw()).max(0);
				let off		= match aligns[b] {
					Align::Left		=> 0,
					Align::Centre	=> slack / 2,
					Align::Right	=> slack,
				};
				let target = content_left[b] + Sp(off);
				kids.push(Node::Glue(Glue::fixed(target - cursor)));
				for ch in &line.children {
					kids.push(ch.clone());
				}
				cursor = target + line.width;
			}
			// An empty cell adds nothing; the next boundary's glue jumps the column's width.
		}
	}
	Node::HBox(BoxNode::new(kids, Dims::new(table_width, band_height, Sp::ZERO)))
}

/// Builds the background wash of one band: a graphic leaf whose single fill covers the band rectangle in
/// `colour`. The leaf declares zero width so it does not move the band's horizontal cursor -- the path is
/// drawn from the leaf's placement whatever the declared advance -- and its height matches the band, so
/// the wash tiles seamlessly with the band above and below. `None` for a degenerate (zero-area) band, so
/// `Path::rect` is never handed an empty rectangle.
fn fill_band(width: Sp, height: Sp, colour: Rgba) -> Option<Leaf> {
	let w = width.to_pt() as f32;
	let h = height.to_pt() as f32;
	if w <= 0.0 || h <= 0.0 {
		return None;
	}
	let rect	= Path::rect(Bounds::new(0.0, 0.0, w, h)).ok()?;
	let graphic	= Graphic::new(vec![DrawOp::Fill { path: rect, colour }], Dims::new(Sp::ZERO, height, Sp::ZERO));
	Some(Leaf::graphic(graphic))
}

/// Pushes a full-width horizontal rule and advances the running height by its thickness.
fn push_hrule(children: &mut Vec<Node>, total_h: &mut Sp, width: Sp, thick: Sp) {
	children.push(Node::Leaf(Leaf::rule(Dims::new(width, thick, Sp::ZERO))));
	*total_h += thick;
}
