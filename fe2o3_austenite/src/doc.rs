//! The authoring layer: blocks of prose above the box-glue-penalty stream.
//!
//! [`driver::Document`](crate::driver::Document) is the composed form -- a flat vertical stream the
//! two-pass driver paginates. This module sits above it. An author writes a [`Block`] list --
//! headings and paragraphs -- and [`author`] turns each block into the stream: a heading is shaped
//! bold and larger, its identity recorded as a [`Heading`](crate::ledger::AnchorKind::Heading)
//! anchor so a running head or a table of contents can later find its page; a paragraph is set into
//! justified lines by [`break_paragraph`](crate::linebreak::break_paragraph).
//!
//! Two facts a reader could not derive. A heading is kept with the first line of its paragraph by
//! setting the two inside one unbreakable box, so the driver's greedy page breaker never leaves a
//! heading stranded at a page foot (the widow guard). And the page furniture -- the running head and
//! the folio -- is added by [`decorate`] after the document has converged, because it lives in the
//! margins, outside the text block, and so cannot disturb the pagination it describes. The running
//! head is TeX's `\mark` reimplemented through the ledger: the section current at the top of a page
//! is the most recent heading the ledger resolved to an earlier page.

use crate::bib::Bibliography;
use crate::driver::{
	Document,
	FootStyle,
};
use crate::font::ShapedText;
use crate::ir::{
	BoxNode,
	Dims,
	DrawOp,
	Footnote,
	Glue,
	Graphic,
	Leaf,
	LeafKind,
	Length,
	Node,
	Penalty,
	RasterImage,
	Sp,
};
use crate::ledger::{
	AnchorId,
	AnchorKind,
	Ledger,
	Ref,
};
use crate::linebreak::{
	break_paragraph,
	break_paragraph_pieces,
	Piece,
};
use crate::math::{
	self,
	Atom,
};
use crate::table::{
	self,
	Align,
	Cell,
	Row,
	Table,
};
use crate::page::{
	Frame,
	Page,
	PageGeometry,
	Placed,
	PlacedKind,
};

use oxedyne_fe2o3_core::prelude::*;
use oxedyne_fe2o3_font::{
	face::Role,
	font::Font,
	set::FontSet,
	shape::Dir,
};
use oxedyne_fe2o3_graphics::{
	colour::Rgba,
	path::{
		Bounds,
		Path,
		PathBuilder,
		Pt,
	},
	svg_doc::{
		Anchor,
		SvgOp,
		SvgPicture,
	},
	transform::Transform,
};

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

/// One run of a rich paragraph: a stretch of body text, a strongly emphasised run (`*strong*`, set
/// bold), an emphasised run (`/emph/`, set italic), or a footnote whose mark falls after the run before
/// it. The note text is set at the foot of the page the mark lands on, and numbered in document order.
#[derive(Clone, Debug)]
pub enum Segment {
	Text(String),
	Strong(String),	// set in the bold face
	Emph(String),	// set in the italic face
	BoldItalic(String),	// `*_x_*`/`_*x*_`, set in the bold-italic face
	Super(String),	// #super[...], set raised and smaller, its baseline lifted above the line's
	Footnote { note: Vec<Segment> },
	Math(Atom),	// an inline maths expression, set within the running line
	PageRef(String),	// a cross-reference to a labelled anchor, resolving to its page number
	Code(String),	// an inline code span, set in the mono face
	Glossary { term: String, display: String },	// a glossary term: bold-italic on its first document use, plain after
	Cite(Vec<String>),	// a citation, resolved to "(Author Year)" against the bibliography
}

impl Segment {
	pub fn text<S: Into<String>>(text: S) -> Self {
		Self::Text(text.into())
	}

	pub fn strong<S: Into<String>>(text: S) -> Self {
		Self::Strong(text.into())
	}

	pub fn emph<S: Into<String>>(text: S) -> Self {
		Self::Emph(text.into())
	}

	pub fn bold_italic<S: Into<String>>(text: S) -> Self {
		Self::BoldItalic(text.into())
	}

	pub fn superscript<S: Into<String>>(text: S) -> Self {
		Self::Super(text.into())
	}

	pub fn footnote(note: Vec<Segment>) -> Self {
		Self::Footnote { note }
	}

	pub fn math(expr: Atom) -> Self {
		Self::Math(expr)
	}

	pub fn page_ref<S: Into<String>>(label: S) -> Self {
		Self::PageRef(label.into())
	}

	pub fn code<S: Into<String>>(text: S) -> Self {
		Self::Code(text.into())
	}

	pub fn glossary<T: Into<String>, D: Into<String>>(term: T, display: D) -> Self {
		Self::Glossary { term: term.into(), display: display.into() }
	}

	pub fn cite(keys: Vec<String>) -> Self {
		Self::Cite(keys)
	}
}

/// One block of the authored document. The closed vocabulary the block layer sets; richer blocks
/// (lists, quotes, figures) are later variants here.
#[derive(Clone, Debug)]
pub enum Block {
	Heading { level: u8, segments: Vec<Segment>, label: Option<String> },	// segments: the title's rich runs; label: an author anchor a `#ref` resolves to
	Paragraph { text: String },
	RichParagraph { segments: Vec<Segment> },	// a paragraph carrying footnote marks
	List { ordered: bool, items: Vec<Vec<Segment>> },	// a bullet or numbered list, each item a run sequence
	Code { lines: Vec<String> },	// a verbatim code block, set in the mono face, whitespace preserved
	Table(Table),
	Equation { expr: Atom, numbered: bool, label: Option<String> },	// a display equation on its own centred line; label anchors an @-reference
	Figure { graphic: Graphic, caption: Option<String> },	// a drawn figure, centred, numbered, captioned
	// A `#figure(...)` wrapping a `#table(...)`: the ruled table, then a numbered caption beneath. The
	// supplement is the caption's leading word ("Table"/"Figure"); the label anchors a cross-reference.
	TableFigure { table: Table, caption: Option<Vec<Segment>>, supplement: String, label: Option<String> },
	// A `#figure(...)` wrapping an image: the loaded raster centred in the measure with the numbered
	// caption beneath, or -- when the path resolves to nothing or is a vector SVG with no raster beside
	// it -- a sized placeholder box in its place. The sizing hints size the drawn image.
	ImageFigure {
		path:		String,
		width:		Option<Length>,
		height:		Option<Length>,
		scale:		Option<f64>,
		caption:	Option<Vec<Segment>>,
		supplement:	String,
		label:		Option<String>,
	},
	// A `#figure(...)` whose body is drawn by code -- a CeTZ/Fletcher diagram, a bar chart or a line plot.
	// The graphic is built at render time from the document's font set and placed like an image figure,
	// with the numbered caption beneath.
	CodeFigure {
		figure:		crate::lang::codefig::CodeFigure,
		caption:	Option<Vec<Segment>>,
		supplement:	String,
		label:		Option<String>,
	},
	// A back-matter section title (the Bibliography) on its own page, set left in the display face and
	// unnumbered. It records a heading anchor so the contents lists it, and a back-matter marker so the
	// running head is dropped and the folio centres from here on.
	BackMatterHeading { title: String },
	// One bibliography reference: its styled runs, each carrying whether it sets in italic. Set small,
	// as a paragraph the reader reads as one entry.
	Reference { runs: Vec<(String, bool)> },
	// A standalone `#line(...)` horizontal divider: a stroked rule of the given width (a fraction of the
	// measure or an absolute length), thickness in points, and grey level, with a paragraph skip either side.
	Rule { width: Length, thickness: f64, grey: u8 },
	// A line-leading `#padded-image(...)`/`#image(...)`: the loaded image centred in the measure with a
	// little space either side, carrying no figure number or caption -- a section opener's logo, not a float.
	Image { path: String, width: Option<Length>, height: Option<Length>, scale: Option<f64> },
	// A line-leading `#section-banner("logo")`: a fresh page, then the template's full-width grey bar hanging
	// into the top and side margins, carrying the section's logo right-aligned on the band's vertical middle.
	SectionBanner { path: String },
}

impl Block {
	pub fn heading<S: Into<String>>(level: u8, text: S) -> Self {
		Self::Heading { level, segments: vec![Segment::text(text.into())], label: None }
	}

	/// A heading carrying an author label, so a `#ref(<label>)` elsewhere resolves to its page.
	pub fn heading_labelled<S: Into<String>>(level: u8, text: S, label: Option<String>) -> Self {
		Self::Heading { level, segments: vec![Segment::text(text.into())], label }
	}

	/// A heading whose title carries rich inline runs -- emphasis, a glossary term, an index call or a
	/// maths span -- so each sets its display text in the head and the table of contents rather than
	/// leaking its raw source.
	pub fn heading_rich(level: u8, segments: Vec<Segment>, label: Option<String>) -> Self {
		Self::Heading { level, segments, label }
	}

	pub fn paragraph<S: Into<String>>(text: S) -> Self {
		Self::Paragraph { text: text.into() }
	}

	pub fn rich(segments: Vec<Segment>) -> Self {
		Self::RichParagraph { segments }
	}

	/// A bullet (`ordered` false) or numbered (`ordered` true) list. Each item is a run sequence, so an
	/// item may carry emphasis, a footnote or inline maths exactly as a rich paragraph does.
	pub fn list(ordered: bool, items: Vec<Vec<Segment>>) -> Self {
		Self::List { ordered, items }
	}

	/// A verbatim code block: each line set in the mono face with its whitespace preserved and no
	/// justification, the way source is shown.
	pub fn code(lines: Vec<String>) -> Self {
		Self::Code { lines }
	}

	pub fn table(table: Table) -> Self {
		Self::Table(table)
	}

	pub fn rule(width: Length, thickness: f64, grey: u8) -> Self {
		Self::Rule { width, thickness, grey }
	}

	/// A display equation set centred on its own line. A numbered one takes the next equation number at
	/// the right margin and records an [`Equation`](crate::ledger::AnchorKind::Equation) anchor; a trailing
	/// `<label>` lets an `@`-reference resolve to "Equation N".
	pub fn equation(expr: Atom, numbered: bool, label: Option<String>) -> Self {
		Self::Equation { expr, numbered, label }
	}

	/// A drawn figure, centred on its own line and captioned "Figure N" beneath, its identity recorded
	/// as a [`Float`](crate::ledger::AnchorKind::Float) anchor so a cross-reference resolves its page.
	pub fn figure(graphic: Graphic, caption: Option<String>) -> Self {
		Self::Figure { graphic, caption }
	}

	/// A table wrapped in a figure: the ruled grid, then a "{supplement} N: {caption}" line beneath,
	/// numbered per supplement so tables and figures carry independent counts.
	pub fn table_figure(
		table:		Table,
		caption:	Option<Vec<Segment>>,
		supplement:	String,
		label:		Option<String>,
	)
		-> Self
	{
		Self::TableFigure { table, caption, supplement, label }
	}

	/// An image wrapped in a figure: the raster at `path`, sized by the declared hints, centred in the
	/// measure with its numbered caption beneath. A path that resolves to nothing, or a vector SVG with no
	/// raster beside it, falls back to a placeholder box at render time.
	#[allow(clippy::too_many_arguments)]
	pub fn image_figure(
		path:		String,
		width:		Option<Length>,
		height:		Option<Length>,
		scale:		Option<f64>,
		caption:	Option<Vec<Segment>>,
		supplement:	String,
		label:		Option<String>,
	)
		-> Self
	{
		Self::ImageFigure { path, width, height, scale, caption, supplement, label }
	}

	/// A figure drawn by code (a diagram, bar chart or line plot): its builder, numbered caption, and the
	/// label a cross-reference resolves to. The graphic is built at render time from the font set.
	pub fn code_figure(
		figure:		crate::lang::codefig::CodeFigure,
		caption:	Option<Vec<Segment>>,
		supplement:	String,
		label:		Option<String>,
	)
		-> Self
	{
		Self::CodeFigure { figure, caption, supplement, label }
	}

	/// A back-matter section heading (the Bibliography), on its own page, unnumbered.
	pub fn back_matter_heading<S: Into<String>>(title: S) -> Self {
		Self::BackMatterHeading { title: title.into() }
	}

	/// One bibliography reference, a sequence of runs each flagged for italic.
	pub fn reference(runs: Vec<(String, bool)>) -> Self {
		Self::Reference { runs }
	}

	/// A plain centred image (a `#padded-image`/`#image` section logo), with any declared sizing.
	pub fn image(path: String, width: Option<Length>, height: Option<Length>, scale: Option<f64>) -> Self {
		Self::Image { path, width, height, scale }
	}

	/// A documentation section's opening banner: a fresh page carrying the template's full-width grey bar
	/// with the logo at `path` right-aligned on it.
	pub fn section_banner(path: String) -> Self {
		Self::SectionBanner { path }
	}
}

/// The point sizes and vertical spaces the block layer sets to. Every length is scaled points, so
/// How a top-level heading opens. A book chapter opens with the giant grey number and dotted numbering
/// the manuscripts use ([`BookOpener`](HeadingStyle::BookOpener)); a documentation tree that opens each
/// chapter with the template's full-width grey banner bar and no numbering takes
/// ([`DocBanner`](HeadingStyle::DocBanner)); a documentation tree whose sections carry their own
/// `#section-banner` logo bar (the Hematite guide) sets its level-1 headings inline instead, with no
/// banner and no numbering ([`DocInline`](HeadingStyle::DocInline)) -- the template's `chapter-banners:
/// false`. The block layer reads this to pick the opener and to decide whether a heading carries a
/// number, so one authoring path serves every idiom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadingStyle {
	BookOpener,
	DocBanner,
	DocInline,
}

/// the styling never leaves the integer domain the driver breaks on.
#[derive(Clone, Copy, Debug)]
pub struct Style {
	pub heading_style:	HeadingStyle,	// which top-level opener and numbering the headings take
	pub body_size:		Sp,
	pub leading:		Sp,
	pub para_skip:		Sp,	// extra space between one paragraph and the next
	pub indent:		Sp,	// first-line indent of a paragraph following another paragraph
	pub h1_size:		Sp,	// the chapter title, set beneath the chapter number
	pub h2_size:		Sp,
	pub h3_size:		Sp,
	pub h4_size:		Sp,
	pub chap_num_size:	Sp,	// the giant chapter number on a chapter-opening page
	pub chap_num_grey:	Rgba,	// the fill of that number, a light grey
	pub chap_grid:		[Sp; 4],	// chapter-opener grid rows: number band, gap, title band, gap-to-body
	pub header_size:	Sp,	// the running head's size
	pub folio_size:		Sp,
	pub foot_size:		Sp,	// the footnote text's size, a touch below the body
	pub foot_leading:	Sp,	// leading between the wrapped lines of one footnote
	pub list_marker_gap:	Sp,	// space between a list marker and the item text it introduces
	pub list_item_skip:		Sp,	// vertical space set between one list item and the next
	pub table_skip:		Sp,	// space set above and below a table
	pub cell_pad_x:		Sp,	// horizontal padding between a cell's text and its column rules
	pub cell_pad_y:		Sp,	// vertical padding above and below a cell's lines
	pub line_gap:		Sp,	// leading between the wrapped lines within one cell
	pub rule_thin:		Sp,	// an interior grid rule
	pub rule_thick:		Sp,	// the frame and the rule beneath a header
	pub header_fill:	Rgba,	// the wash behind a header row, the books' light.lighten(10%)
}

impl Default for Style {
	fn default() -> Self {
		Self {
			heading_style:	HeadingStyle::BookOpener,
			body_size:		Sp::from_pt(11.0),
			leading:		Sp::from_pt(13.2),	// 1.2x the body
			para_skip:		Sp::from_pt(6.0),
			indent:			Sp::ZERO,	// no first-line indent unless a book config sets one
			h1_size:		Sp::from_pt(16.0),
			h2_size:		Sp::from_pt(13.0),
			h3_size:		Sp::from_pt(12.0),
			h4_size:		Sp::from_pt(11.0),
			chap_num_size:	Sp::from_pt(54.0),
			chap_num_grey:	Rgba::opaque(200, 200, 200),	// Typst's luma(200)
			chap_grid:		[Sp::from_pt(72.0), Sp::from_pt(8.0), Sp::from_pt(36.0), Sp::from_pt(20.0)],
			header_size:	Sp::from_pt(9.5),
			folio_size:		Sp::from_pt(10.0),
			foot_size:		Sp::from_pt(9.0),
			foot_leading:	Sp::from_pt(10.8),	// 1.2x the footnote size
				list_marker_gap:	Sp::from_pt(6.0),
				list_item_skip:		Sp::from_pt(3.0),
			table_skip:		Sp::from_pt(10.0),
			cell_pad_x:		Sp::from_pt(5.0),
			cell_pad_y:		Sp::from_pt(3.0),
			line_gap:		Sp::from_pt(3.0),
			rule_thin:		Sp::from_pt(0.4),
			rule_thick:		Sp::from_pt(0.8),
			header_fill:	Rgba::opaque(235, 238, 241),	// #E9ECEF lightened 10%, the template's header1
		}
	}
}

impl Style {
	fn heading_size(&self, level: u8) -> Sp {
		match level {
			0 => self.h1_size,	// a part-divider title, set at the chapter-title size
			1 => self.h1_size,
			2 => self.h2_size,
			3 => self.h3_size,
			_ => self.h4_size,
		}
	}

	/// The space set above a heading of this level, always greater than the space below it, so a
	/// heading binds visually to the text it introduces rather than to the text it follows.
	fn space_above(&self, level: u8) -> Sp {
		match level {
			1 => Sp::from_pt(20.0),
			2 => Sp::from_pt(15.0),
			_ => Sp::from_pt(12.0),
		}
	}

	fn space_below(&self, level: u8) -> Sp {
		match level {
			1 => Sp::from_pt(8.0),
			2 => Sp::from_pt(6.0),
			_ => Sp::from_pt(5.0),
		}
	}
}

/// A recorded heading: the anchor identity the ledger resolves to a page, its level, and its display
/// title. The block layer keeps this table beside the composed stream so [`decorate`] can read a
/// title back from an anchor -- the ledger stores only the identity, not the words.
#[derive(Clone, Debug)]
pub struct Heading {
	pub id:			AnchorId,
	pub level:		u8,
	pub title:		String,	// the display words, markup removed, for the anchor slug and a plain fallback
	pub segments:	Vec<Segment>,	// the title's rich runs, so a running head or contents entry renders its maths and emphasis
	pub number:		String,	// the dotted number a numbered heading shows ("2.3.1"); empty for a part divider
	pub banner:		bool,	// set inline beneath a `#section-banner`, so the page suppresses its running head like a chapter opener
}

/// The book's front matter, read from the root's template call: the title, subtitle and author the
/// title page sets, the cover raster a development build carries, and the imprint the meta page prints.
/// A field a book omits is `None` and its line is not set. The whole struct is `None` for a lone
/// manuscript, which carries no front matter at all.
#[derive(Clone, Debug, Default)]
pub struct FrontMatter {
	pub title:			String,
	pub subtitle:		Option<String>,
	pub author:			String,
	pub cover_image:	Option<String>,	// a `/assets/...` raster path, set only in a development build
	pub logo_image:		Option<String>,	// the publisher logo under the title, often an SVG (then not set)
	pub publisher:		Option<String>,
	pub edition:		Option<String>,
	pub isbn:			Option<String>,
	pub copyright:		Option<String>,	// the already-composed "Copyright © 2026 ..." line
	pub rights:			Option<String>,
	pub ai_declaration:	Option<String>,
	pub website:		Option<String>,
	pub toolchain:		bool,			// whether to print the "Created using ..." toolchain line
	pub dedication:		Option<String>,
	pub about_author:	Option<String>,	// the author biography, set on its own page
	// The display sizes the title page and back-matter titles set, read from the config's type scale.
	pub title_size:		Sp,
	pub subtitle_size:	Sp,
	pub author_size:	Sp,
	pub back_title_size:	Sp,	// the "About the Author"/"Bibliography" heading size
	// The documentation template's two-column title page (`template.typ`'s `title-page`): a full-height
	// coloured sidebar down the left carrying a logo near its top and one near its foot, with the title and
	// subtitle centred on the white right. `sidebar_grey` marks this idiom -- `Some(luma)` draws it and
	// `None` keeps the book's plain centred title page. The rest are read from the root's `doc.with` call.
	pub sidebar_grey:		Option<u8>,	// the sidebar fill as a grey level; None keeps the plain title page
	pub sidebar_frac:		f64,		// the sidebar width as a fraction of the page width (`margins.title_page`)
	pub title_smallcaps:	bool,		// whether the title sets in small caps rather than italic
	pub top_logo:			Option<String>,	// the logo near the sidebar's top
	pub top_logo_width:		Sp,
	pub bottom_logo:		Option<String>,	// the logo near the sidebar's foot
	pub bottom_logo_width:	Sp,
	pub footer_logo:		Option<String>,	// the logo the template seats at the left of the page footer
	// The documentation template's meta/colophon page (`template.typ`'s `meta-page`): a bordered
	// Ver/Date/Author(s)/Notes table over an acknowledgement, a copyright line and a toolchain line at the
	// page foot. `meta_rows` carries the revision rows, newest first; a non-empty list (or a named author)
	// marks the idiom, so the doc meta page is composed only for a doc tree, never over the book imprint.
	pub meta_rows:			Vec<MetaRow>,	// the revision rows the version table sets, in source order
	pub reading_min:		Option<u32>,	// the whole-document reading time in minutes, appended to the last row's notes
	pub acknowledgement:	Option<String>,	// the acknowledgement paragraph set near the page foot
}

/// One revision row of the documentation meta/colophon table: its version, date, author(s), notes, and
/// the AI-declaration mark the row carries beneath its author. A field the row omits is `None`, and its
/// column is left out of the table when every row omits it (matching the template's `filled` test).
#[derive(Clone, Debug)]
pub struct MetaRow {
	pub version:		Option<String>,
	pub date:			Option<String>,
	pub authors:		String,
	pub notes:			Option<String>,
	pub ai_mark_path:	Option<String>,	// the declaration mark image, resolved from the row's slug
	pub ai_mark_words:	Option<String>,	// the mark's caption, the row's own words when it rescopes them
	pub ai_mark_url:	Option<String>,	// the scheme page the mark links to, <scheme>/<slug>/<medium>
}

/// Turns an authored block list into the composed document, and the heading table the running heads
/// resolve against. The geometry fixes the measure every paragraph is set to.
///
/// When `front` is set the front matter -- cover, title page, imprint, dedication and author note --
/// is composed ahead of the body, so the body's first heading fixes where the printed folio restarts
/// at one; a lone manuscript passes `None` and carries no front matter.
pub fn author(
	fonts:		Arc<FontSet>,
	geom:		PageGeometry,
	style:		Style,
	heading:	Option<Arc<Font>>,
	blocks:		&[Block],
	front:		Option<&FrontMatter>,
	bib:		Option<&Bibliography>,
)
	-> Outcome<(Document, Vec<Heading>)>
{
	let measure	= geom.content_width();
	let mut nodes:	Vec<Node>		= Vec::new();
	let mut heads:	Vec<Heading>	= Vec::new();

	let mut i		= 0usize;
	let mut first	= true;
	// The chapter and section counters, a document-order fold. A level-L heading (L>=1) steps counter
	// L and clears the deeper ones; a level-0 part divider steps none, so it stays outside the numbering.
	let mut sec:	[u32; 6]	= [0; 6];
	// Whether the block just emitted was a paragraph. A paragraph following another paragraph takes the
	// first-line indent; one opening a section (after a heading, list, figure or the document start) does
	// not -- Typst's `first-line-indent` with `all: false`, and what the oracle sets.
	let mut prev_para	= false;
	// Set when a `#section-banner` was the block just emitted, so the level-1 heading that follows it is
	// marked as opening beneath a banner. It is consumed (and cleared) by that heading, the block the
	// source always places immediately after the banner.
	let mut pending_banner	= false;
	let mut part_no	= 0u32;	// the part-divider ordinal, a document-order fold, shown as a Roman numeral
	let mut foot_no	= 0u32;	// the footnote number, a document-order fold over the marks
	let mut ref_no	= 0u32;	// a running counter giving each inline cross-reference its own anchor id
	let mut eq_no	= 0u32;	// the equation number, a document-order fold over the numbered displays
	let mut fig_no	= 0u32;	// the figure number, a document-order fold over the drawn figures
	// The number per figure supplement ("Figure", "Table"): a document-order fold, so tables and figures
	// carry independent counts, matching Typst's per-kind numbering.
	let mut counters:	HashMap<String, u32>	= HashMap::new();
	// Glossary terms already set once, in document order. The first mention of a term is set bold-italic
	// and every later mention plain; author walks the blocks in order, so the set decides first-use with
	// no second pass. Keyed by the term as written, matching the template's case-sensitive tracking.
	let mut seen:	HashSet<String>	= HashSet::new();
	// The text every labelled cross-reference resolves to, settled once from document order so a forward
	// reference reads its referent's supplement and number without a layout round-trip.
	let refs = ref_targets(blocks);
	while i < blocks.len() {
		match &blocks[i] {
			Block::Heading { level, segments, label } => {
				// Step the counters for a numbered level (1..); a part divider (level 0) steps none.
				if *level >= 1 {
					let l = (*level as usize).min(6);
					sec[l - 1] += 1;
					for k in l..6 { sec[k] = 0; }
				}
				// A documentation tree sets `numbering: none`: its headings carry no dotted number, on the
				// heading line, in the contents, or before a sub-heading. A book keeps the document-order number.
				let number = match style.heading_style {
					HeadingStyle::DocBanner | HeadingStyle::DocInline	=> String::new(),
					HeadingStyle::BookOpener							=> heading_number(*level, &sec),
				};

				// The rendered title, its markup reduced to display words: it keys the anchor slug and is the
				// title the contents list and the running head read back. The heading itself is set from the
				// rich runs below, so a glossary term or emphasis in a heading renders rather than leaking.
				let title = flatten_segments(segments);
				let id = AnchorId::new(AnchorKind::Heading, fmt!("{:02}-{}", heads.len() + 1, slug(&title)));
				// A level-1 heading that follows a `#section-banner` opens its section beneath the banner, so
				// its page suppresses the running head like a chapter opener; the flag is one-shot.
				let banner = pending_banner && *level == 1;
				pending_banner = false;
				heads.push(Heading {
					id:			id.clone(),
					level:		*level,
					title:		title.clone(),
					segments:	segments.clone(),
					number:		number.clone(),
					banner,
				});

				// A chapter (level 1) or a part divider (level 0) opens a fresh page and stands alone; a
				// deeper heading binds to the first line of the paragraph it introduces, so the greedy page
				// breaker never strands it at a page foot. A level-1 heading that carries its own
				// `#section-banner` is the exception: it is set inline beneath the banner the section drew, so
				// it takes the sub-heading path with no page break of its own -- the banner already turned the
				// page. This holds whether the tree sets every section that way (`DocInline`, the Hematite
				// guide) or opts one chapter in with an explicit `#section-banner` while defaulting to the
				// grey title bar (`DocBanner`): an explicit banner always owns its chapter's header, so the
				// duplicate title bar is suppressed regardless of the doc's default mode.
				let opens = *level == 0
					|| (*level == 1 && style.heading_style != HeadingStyle::DocInline && !banner);
				if opens {
					if !first {
						nodes.push(Node::Penalty(Penalty::eject()));
					}
					// A part divider (level 0) carries a "Part N" run-in label above its title; a chapter carries
					// none. The ordinal is a Roman numeral, the template's `smallcaps(part-counter.display("I"))`.
					let part_label = if *level == 0 {
						part_no += 1;
						fmt!("Part {}", roman(part_no))
					} else {
						String::new()
					};
					res!(chapter_opener(
						&mut nodes, &fonts, heading.as_ref(), style, geom, measure, *level, &number, &title,
						&part_label, &id, label.as_deref()));
					i += 1;
					first = false;
					prev_para = false;	// the opener is not a paragraph, so the first body line takes no indent
					continue;
				}

				// Space above the heading. At a page top the driver discards it, so the first heading on a
				// page still sits flush to the text block.
				if !first {
					nodes.push(Node::Glue(Glue::fixed(style.space_above(*level))));
				}

				let hbox = res!(subheading_hbox(
					fonts.clone(), heading.as_ref(), style, *level, &number, segments, &mut seen));

				let mut keep:	Vec<Node> = vec![Node::Anchor(id)];
				if let Some(l) = label {
					keep.push(Node::Anchor(AnchorId::new(AnchorKind::Label, l.clone())));
				}
				keep.push(hbox);
				keep.push(Node::Glue(Glue::fixed(style.space_below(*level))));
				let mut rest:	Vec<Node> = Vec::new();
				let mut consumed_para = false;
				if let Some(Block::Paragraph { text: para }) = blocks.get(i + 1) {
					// The first paragraph after a heading opens the section, so it takes no first-line indent.
					let mut lines = res!(break_paragraph(
						fonts.clone(), Role::Body, Dir::Ltr, style.body_size, para, measure, style.leading));
					if !lines.is_empty() {
						keep.push(lines.remove(0));	// the first line joins the heading
						rest = lines;				// its leading glue and the remaining lines follow
					}
					consumed_para = true;
					i += 2;
				} else {
					i += 1;
				}

				nodes.push(vbox(keep, measure));
				nodes.extend(rest);
				first = false;
				// A heading opens a section: the paragraph it swallowed took no indent, but the NEXT paragraph
				// follows a paragraph and so is indented.
				prev_para = consumed_para;
			},
			Block::Paragraph { text } => {
				if !first {
					nodes.push(Node::Glue(Glue::fixed(style.para_skip)));
				}
				// A plain paragraph is set through the piece breaker so a leading indent box can ride the
				// front of its first line; without an indent it produces exactly what `break_paragraph` does.
				let mut pieces = Vec::new();
				if prev_para && style.indent.raw() > 0 {
					pieces.push(indent_piece(style.indent));
				}
				pieces.push(Piece::Text { text: text.clone(), role: Role::Body });
				let lines = res!(break_paragraph_pieces(
					fonts.clone(), Role::Body, Dir::Ltr, style.body_size, &pieces, measure, style.leading, true));
				nodes.extend(lines);
				i += 1;
				first = false;
				prev_para = true;
			},
			Block::RichParagraph { segments } => {
				if !first {
					nodes.push(Node::Glue(Glue::fixed(style.para_skip)));
				}
				let mut pieces = Vec::new();
				if prev_para && style.indent.raw() > 0 {
					pieces.push(indent_piece(style.indent));
				}
				pieces.extend(res!(build_pieces(
					fonts.clone(), geom, style, segments, &mut foot_no, &mut ref_no, &mut seen, bib, &refs)));
				let lines = res!(break_paragraph_pieces(
					fonts.clone(), Role::Body, Dir::Ltr, style.body_size, &pieces, measure, style.leading, true));
				nodes.extend(lines);
				i += 1;
				first = false;
				prev_para = true;
			},
			Block::List { ordered, items } => {
				if !first {
					nodes.push(Node::Glue(Glue::fixed(style.para_skip)));
				}
				res!(list(&mut nodes, fonts.clone(), geom, style, measure, *ordered, items, &mut foot_no, &mut ref_no, &mut seen, bib, &refs));
				i += 1;
				first = false;
				prev_para = false;
			},
			Block::Code { lines: src } => {
				if !first {
					nodes.push(Node::Glue(Glue::fixed(style.para_skip)));
				}
				res!(code_block(&mut nodes, fonts.clone(), style, src));
				nodes.push(Node::Glue(Glue::fixed(style.para_skip)));
				i += 1;
				first = false;
				prev_para = false;
			},
			Block::Table(t) => {
				// Space above the table, discarded at a page top like any other leading. The table lowers
				// to one keep box, so the driver moves it whole to the next page when it will not fit.
				if !first {
					nodes.push(Node::Glue(Glue::fixed(style.table_skip)));
				}
				nodes.push(res!(table::lower(fonts.clone(), style, measure, t, &refs)));
				nodes.push(Node::Glue(Glue::fixed(style.table_skip)));
				i += 1;
				first = false;
				prev_para = false;
			},
			Block::Equation { expr, numbered, .. } => {
				if !first {
					nodes.push(Node::Glue(Glue::fixed(style.para_skip)));
				}
				let number = if *numbered { eq_no += 1; Some(eq_no) } else { None };
				res!(equation(&mut nodes, fonts.clone(), style, measure, expr, number));
				nodes.push(Node::Glue(Glue::fixed(style.para_skip)));
				i += 1;
				first = false;
				prev_para = false;
			},
			Block::Figure { graphic, caption } => {
				// Space above the figure, discarded at a page top like any other leading. The figure is
				// one keep box, so the breaker moves it whole to the next page when it will not fit.
				if !first {
					nodes.push(Node::Glue(Glue::fixed(style.table_skip)));
				}
				fig_no += 1;
				res!(figure(&mut nodes, fonts.clone(), style, measure, graphic.clone(), caption.as_deref(), fig_no));
				nodes.push(Node::Glue(Glue::fixed(style.table_skip)));
				i += 1;
				first = false;
			},
			Block::TableFigure { table, caption, supplement, label } => {
				if !first {
					nodes.push(Node::Glue(Glue::fixed(style.table_skip)));
				}
				let number = next_number(&mut counters, supplement);
				res!(table_figure(
					&mut nodes, fonts.clone(), style, measure, table,
					caption.as_deref(), supplement, number, label.as_deref(), &refs));
				nodes.push(Node::Glue(Glue::fixed(style.table_skip)));
				i += 1;
				first = false;
			},
			Block::ImageFigure { path, width, height, scale, caption, supplement, label } => {
				if !first {
					nodes.push(Node::Glue(Glue::fixed(style.table_skip)));
				}
				let number = next_number(&mut counters, supplement);
				res!(image_figure(
					&mut nodes, fonts.clone(), style, measure, path, *width, *height, *scale,
					caption.as_deref(), supplement, number, label.as_deref()));
				nodes.push(Node::Glue(Glue::fixed(style.table_skip)));
				i += 1;
				first = false;
			},
			Block::CodeFigure { figure, caption, supplement, label } => {
				if !first {
					nodes.push(Node::Glue(Glue::fixed(style.table_skip)));
				}
				let number = next_number(&mut counters, supplement);
				res!(code_figure(
					&mut nodes, fonts.clone(), style, measure, figure,
					caption.as_deref(), supplement, number, label.as_deref()));
				nodes.push(Node::Glue(Glue::fixed(style.table_skip)));
				i += 1;
				first = false;
			},
			Block::BackMatterHeading { title } => {
				if !first {
					nodes.push(Node::Penalty(Penalty::eject()));
				}
				// The back-matter marker (a Citation anchor) fixes where the running head drops and the
				// folio centres; a heading anchor lists it in the contents. Both sit at the page top.
				nodes.push(Node::Anchor(AnchorId::new(AnchorKind::Citation, slug(title))));
				let id = AnchorId::new(AnchorKind::Heading, fmt!("{:02}-{}", heads.len() + 1, slug(title)));
				heads.push(Heading {
				id:			id.clone(),
				level:		0,
				title:		title.clone(),
				segments:	vec![Segment::text(title.clone())],
				number:		String::new(),
				banner:		false,
			});
				nodes.push(Node::Anchor(id));
				// The title left in the display face at the chapter-title size (the template's
				// glossary-index-title size, equal to it in these books' scales).
				let sh	= res!(head_shape(&fonts, &head_face(1, heading.as_ref()), style.h1_size, title));
				let d	= sh.dims();
				nodes.push(Node::HBox(BoxNode::new(
					vec![Node::Leaf(Leaf::text(sh))], Dims::new(measure, d.height, d.depth))));
				nodes.push(Node::Glue(Glue::fixed(Sp::from_pt(20.0))));
				i += 1;
				first = false;
				prev_para = false;
			},
			Block::Reference { runs } => {
				// The reference list is tight, entry under entry, so entries part by the interline leading
				// rather than the paragraph skip.
				if !first {
					let gap = if style.foot_leading > style.foot_size { style.foot_leading - style.foot_size } else { style.line_gap };
					nodes.push(Node::Glue(Glue::fixed(gap)));
				}
				res!(reference_block(&mut nodes, fonts.clone(), style, measure, runs));
				i += 1;
				first = false;
				prev_para = false;
			},
			Block::Rule { width, thickness, grey } => {
				if !first {
					nodes.push(Node::Glue(Glue::fixed(style.para_skip)));
				}
				rule_divider(&mut nodes, measure, *width, *thickness, *grey);
				nodes.push(Node::Glue(Glue::fixed(style.para_skip)));
				i += 1;
				first = false;
				prev_para = false;
			},
			Block::Image { path, width, height, scale } => {
				res!(plain_image(&mut nodes, fonts.clone(), measure, path, *width, *height, *scale));
				i += 1;
				first = false;
				prev_para = false;
			},
			Block::SectionBanner { path } => {
				// The template's `#section-banner` turns the page first (`pagebreak(weak: true)`); a forced
				// eject the driver drops when the page is already fresh, so it never opens a blank one.
				nodes.push(Node::Penalty(Penalty::eject()));
				res!(section_banner(&mut nodes, fonts.clone(), geom, measure, path));
				pending_banner = true;	// the section's level-1 heading follows and opens beneath this banner
				i += 1;
				first = false;
				prev_para = false;
			},
		}
	}

	// The front matter is composed ahead of the body so its cover, title, imprint and note leaves take
	// the physical pages before the body opens; the body then carries no heading anchor of the front
	// matter's, so the driver fixes the folio restart at the first body heading.
	let mut stream: Vec<Node> = Vec::new();
	if let Some(fm) = front {
		res!(front_matter(&mut stream, &fonts, heading.as_ref(), geom, style, fm));
		// The contents follows the front matter and precedes the body, resolving each entry's folio as a
		// forward reference into the body the driver has not composed yet.
		stream.extend(res!(contents(
			fonts.clone(), heading.as_ref(), geom, style, fm.back_title_size, &heads)));
	}
	stream.extend(nodes);

	let mut document = Document::new(stream, geom);
	document.foot = foot_style(style);
	Ok((document, heads))
}

/// The foot spacing derived from the block style, so the separator rule and the gaps around the notes
/// match the document's other furniture. The rule runs a third of the measure, a conventional short
/// footnote rule.
fn foot_style(style: Style) -> FootStyle {
	FootStyle {
		gap_above_rule:	style.para_skip,
		rule_thick:		style.rule_thin,
		rule_width:		Sp(style.body_size.raw() * 12),
		gap_below_rule:	Sp::from_pt(4.0),
		gap_between:	Sp::from_pt(3.0),
	}
}

/// A first-line indent as a rigid leading piece: an empty box of the indent width that the optimiser
/// counts against the first line and that never breaks, so the first word sits one indent in and the
/// line still fills the measure. Modelled as a maths piece of zero height carrying a single fixed glue,
/// which is how the piece breaker already threads a pre-built inline cluster into the line.
fn indent_piece(indent: Sp) -> Piece {
	Piece::Math {
		nodes:	vec![Node::Glue(Glue::fixed(indent))],
		width:	indent,
		height:	Sp::ZERO,
		depth:	Sp::ZERO,
		over:	Sp::ZERO,
	}
}

/// The text each labelled cross-reference resolves to, fixed in a document-order pre-pass. A reference's
/// supplement word and number depend only on document order, not on layout, so they are settled once here
/// and set as static text -- Typst's own "Chapter 4" for a chapter, "Section 7.7" for a section, and
/// "{supplement} {number}" for a figure or table -- matching the oracle's own output. The heading and
/// figure and equation counters are stepped exactly as [`author`] steps them, so a label's number here is
/// the number the block itself sets -- a chapter, section, figure, table or "Equation N". A label the
/// pre-pass never records is left for the caller's page-number fallback.
fn ref_targets(blocks: &[Block]) -> HashMap<String, String> {
	let mut out:		HashMap<String, String>	= HashMap::new();
	let mut sec:		[u32; 6]				= [0; 6];
	let mut counters:	HashMap<String, u32>	= HashMap::new();
	let mut eq_no		= 0u32;	// the equation counter, stepped exactly as `author` steps it
	for block in blocks {
		match block {
			Block::Heading { level, label, .. } => {
				if *level >= 1 {
					let l = (*level as usize).min(6);
					sec[l - 1] += 1;
					for k in l..6 { sec[k] = 0; }
				}
				if let Some(l) = label {
					let number = heading_number(*level, &sec);
					// A chapter (level 1) takes the "Chapter" supplement the template sets; a deeper heading
					// takes "Section" with its full dotted number, as Typst's default heading reference does. A
					// part divider (level 0) carries no number and is no reference target.
					let text = match *level {
						0			=> continue,
						1			=> fmt!("Chapter {}", number),
						_			=> fmt!("Section {}", number),
					};
					out.insert(l.clone(), text);
				}
			},
			Block::TableFigure { supplement, label, .. }
			| Block::ImageFigure { supplement, label, .. }
			| Block::CodeFigure { supplement, label, .. } => {
				let n = next_number(&mut counters, supplement);
				if let Some(l) = label {
					out.insert(l.clone(), fmt!("{} {}", supplement, n));
				}
			},
			Block::Equation { numbered, label, .. } => {
				// A numbered equation steps the counter; a labelled one anchors "Equation N", Typst's
				// default equation reference. An unnumbered equation carries no number, so its label is
				// left to the caller's page-number fallback.
				if *numbered {
					eq_no += 1;
					if let Some(l) = label {
						out.insert(l.clone(), fmt!("Equation {}", eq_no));
					}
				}
			},
			_ => {},
		}
	}
	out
}

/// Turns a rich paragraph's segments into the pieces the line breaker weaves, assigning each footnote
/// its number from the running fold and setting its note as a small paragraph at the foot measure, and
/// each cross-reference a reserved inline slot the driver resolves in pass B. A text segment is a piece
/// as it stands; a footnote becomes a superscript mark piece carrying the set note; a page reference or
/// a total-pages call becomes a shrink-to-fit reserved leaf, unique by the running `ref_no`.
#[allow(clippy::too_many_arguments)]
fn build_pieces(
	fonts:		Arc<FontSet>,
	geom:		PageGeometry,
	style:		Style,
	segments:	&[Segment],
	foot_no:	&mut u32,
	ref_no:		&mut u32,
	seen:		&mut HashSet<String>,
	bib:		Option<&Bibliography>,
	refs:		&HashMap<String, String>,
)
	-> Outcome<Vec<Piece>>
{
	let measure			= geom.content_width();
	let mut pieces		= Vec::with_capacity(segments.len());
	for seg in segments {
		match seg {
			Segment::Text(text) => {
				pieces.push(Piece::Text { text: text.clone(), role: Role::Body });
			},
			Segment::Strong(text) => {
				pieces.push(Piece::Text { text: text.clone(), role: Role::Bold });
			},
			Segment::Emph(text) => {
				pieces.push(Piece::Text { text: text.clone(), role: Role::Italic });
			},
			Segment::BoldItalic(text) => {
				pieces.push(Piece::Text { text: text.clone(), role: Role::BoldItalic });
			},
			Segment::Super(text) => {
				// The same raise the footnote mark rides: a run shaped at 0.7x, its box shortened so the
				// emitter seats its baseline above the line's. It is rigid and never breaks -- the space
				// after it may -- exactly as a mark piece behaves.
				let (shaped, dims)	= res!(superscript(fonts.clone(), Role::Body, style.body_size, text));
				pieces.push(Piece::Mark(Leaf::text_dims(shaped, dims)));
			},
			Segment::Footnote { note } => {
				*foot_no += 1;
				let label			= fmt!("{}", *foot_no);
				let (mark, dims)	= res!(superscript(fonts.clone(), Role::Body, style.body_size, &label));
				let footnote		= res!(build_footnote(fonts.clone(), style, measure, *foot_no, note, mark));
				pieces.push(Piece::Mark(Leaf::mark(footnote, dims)));
			},
			Segment::Math(expr) => {
				// The inline box is flattened to leaves and glue by the maths layout; unwrap the HBox it
				// returns and weave its children into the line, so they draw as real glyphs rather than as
				// a nested rectangle. The box seats its baseline on the text baseline -- a body ascent
				// below the line top -- so the line asks for that ascent as its height; anything the maths
				// reaches above it is the overshoot the line above must open for.
				let node = res!(math::layout(fonts.clone(), &style, expr, false));
				if let Node::HBox(b) = node {
					let ascent	= res!(ShapedText::new(
						fonts.clone(), Role::Body, Dir::Ltr, style.body_size, "0")).dims().height;
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
			Segment::PageRef(label) => {
				// A cross-reference resolves to Typst's own supplement-and-number text -- "Chapter 4",
				// "Section 7.7", "Figure 2", "Table 1", "Equation 9" -- fixed by the document-order pre-pass
				// and set as body text. A label the pre-pass did not record falls back to the reserved
				// page-number slot the driver resolves in pass B, so the reference still reads rather than
				// vanishing.
				match refs.get(label) {
					Some(text)	=> pieces.push(Piece::Text { text: text.clone(), role: Role::Body }),
					None		=> pieces.push(Piece::Mark(res!(ref_slot(
						fonts.clone(), style, ref_no,
						Ref::PageOf(AnchorId::new(AnchorKind::Label, label.clone())))))),
				}
			},
			Segment::Code(text) => {
				pieces.push(Piece::Text { text: text.clone(), role: Role::Mono });
			},
			Segment::Glossary { term, display } => {
				// The first mention of a term is set bold-italic, matching the template's `*_term_*`;
				// every later mention is plain body text. Document order is the traversal order, so the
				// set alone decides, with no second pass.
				let role = if seen.insert(term.clone()) { Role::BoldItalic } else { Role::Body };
				pieces.push(Piece::Text { text: display.clone(), role });
			},
			Segment::Cite(keys) => {
				// Resolve the citation to "(Author Year)" against the bibliography, set as body text. A
				// key the bibliography does not hold, or a run with no bibliography loaded, falls back to
				// the bracketed keys so the citation still reads rather than vanishing.
				let text = match bib {
					Some(b) => {
						let refs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
						b.format_citation(&refs).unwrap_or_else(|_| fmt!("({})", keys.join("; ")))
					},
					None => fmt!("({})", keys.join("; ")),
				};
				pieces.push(Piece::Text { text, role: Role::Body });
			},
		}
	}
	Ok(pieces)
}

/// Builds one inline cross-reference: a reserved leaf, unique by the running `ref_no`, that reserves a
/// three-digit slot and shrinks to the value the driver resolves for `refr` in pass B. It seats on the
/// body baseline, taking a body digit's height and depth so it aligns with the prose around it.
fn ref_slot(
	fonts:	Arc<FontSet>,
	style:	Style,
	ref_no:	&mut u32,
	refr:	Ref,
)
	-> Outcome<Leaf>
{
	*ref_no += 1;
	let own		= AnchorId::new(AnchorKind::Label, fmt!("ref-{}", *ref_no));
	let slot	= res!(ShapedText::new(fonts, Role::Body, Dir::Ltr, style.body_size, "000"));
	let sd		= slot.dims();
	Ok(Leaf::reserved_inline(own, refr, Dims::new(sd.width, sd.height, sd.depth)))
}

/// Sets a bullet or numbered list into the vertical list. Each item is broken at a measure reduced by
/// the marker column and then hung under its marker: the first line carries the marker leaf and a gap
/// that together fill the indent, the rest are shifted right by it, so every line's right edge still
/// lands on the measure. The marker column is the widest marker the list uses plus
/// [`list_marker_gap`](Style), so a bullet list and a numbered list of ten items align their text
/// alike. Items are parted by [`list_item_skip`](Style); the list's space from its neighbours is the
/// caller's. Each item is a segment run, so it breaks through the same path a rich paragraph does and
/// may carry emphasis, a footnote or inline maths.
#[allow(clippy::too_many_arguments)]
fn list(
	nodes:		&mut Vec<Node>,
	fonts:		Arc<FontSet>,
	geom:		PageGeometry,
	style:		Style,
	measure:	Sp,
	ordered:	bool,
	items:		&[Vec<Segment>],
	foot_no:	&mut u32,
	ref_no:		&mut u32,
	seen:		&mut HashSet<String>,
	bib:		Option<&Bibliography>,
	refs:		&HashMap<String, String>,
)
	-> Outcome<()>
{
	// Shape every marker once and keep the widest, so each item's text starts at the one indent.
	let mut markers:	Vec<ShapedText>	= Vec::with_capacity(items.len());
	let mut marker_w					= Sp::ZERO;
	for idx in 0..items.len() {
		let label	= if ordered { fmt!("{}.", idx + 1) } else { "\u{2022}".to_string() };	// U+2022 bullet
		let shaped	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, style.body_size, &label));
		if shaped.dims().width > marker_w { marker_w = shaped.dims().width; }
		markers.push(shaped);
	}
	let indent	= marker_w + style.list_marker_gap;
	let inner	= if measure > indent { measure - indent } else { measure };

	for (idx, item) in items.iter().enumerate() {
		if idx > 0 {
			nodes.push(Node::Glue(Glue::fixed(style.list_item_skip)));
		}
		let pieces		= res!(build_pieces(fonts.clone(), geom, style, item, foot_no, ref_no, seen, bib, refs));
		let mut lines	= res!(break_paragraph_pieces(
			fonts.clone(), Role::Body, Dir::Ltr, style.body_size, &pieces, inner, style.leading, true));
		indent_item(&mut lines, Leaf::text(markers[idx].clone()), indent);
		nodes.extend(lines);
	}
	Ok(())
}

/// Sets a verbatim code block: each source line in the mono face, its leading whitespace preserved by
/// shaping the whole line, given a one-em hanging indent, and never justified or wrapped. A blank line
/// keeps the mono line's height so the block's vertical rhythm holds. The block's space from its
/// neighbours is the caller's. A long line overflows the measure rather than wrapping -- code is not
/// reflowed; a scrolling or wrapping treatment is a later refinement, as is keeping the block whole
/// across a page break.
fn code_block(
	nodes:	&mut Vec<Node>,
	fonts:	Arc<FontSet>,
	style:	Style,
	lines:	&[String],
)
	-> Outcome<()>
{
	// Code is set a touch smaller than the body, as most templates do, so more of a wide line fits the
	// measure before it overflows.
	let size	= style.foot_size;
	let indent	= style.body_size;	// a one-em hang, so the block sits off the left margin
	let sample	= res!(ShapedText::new(fonts.clone(), Role::Mono, Dir::Ltr, size, "0"));
	let sh		= sample.dims().height;	// a mono digit fixes the height of a blank line
	let sd		= sample.dims().depth;
	for (i, line) in lines.iter().enumerate() {
		let shaped	= res!(ShapedText::new(
			fonts.clone(), Role::Mono, Dir::Ltr, size,
			if line.is_empty() { " " } else { line }));
		let d		= shaped.dims();
		let h		= if d.height > Sp::ZERO { d.height } else { sh };
		let dep		= if d.depth > Sp::ZERO { d.depth } else { sd };
		let children = vec![Node::Glue(Glue::fixed(indent)), Node::Leaf(Leaf::text(shaped))];
		nodes.push(Node::HBox(BoxNode::new(children, Dims::new(indent + d.width, h, dep))));
		if i + 1 < lines.len() {
			let gap = if style.leading > h + dep { style.leading - h - dep } else { style.line_gap };
			nodes.push(Node::Glue(Glue::fixed(gap)));
		}
	}
	Ok(())
}

/// Hangs a broken item under its marker. The first line takes the marker leaf and a gap filling the
/// rest of the indent; every line takes a leading glue that shifts it right by the indent; each line's
/// box grows to the full measure. The item was broken at `measure - indent`, so the right edge lands on
/// the measure. Only [`Node::HBox`] lines are shifted -- the interline glue between them is left alone.
fn indent_item(lines: &mut [Node], marker: Leaf, indent: Sp) {
	let mut first = true;
	for line in lines.iter_mut() {
		if let Node::HBox(b) = line {
			if first {
				let gap = if indent > marker.dims.width { indent - marker.dims.width } else { Sp::ZERO };
				b.list.insert(0, Node::Glue(Glue::fixed(gap)));
				b.list.insert(0, Node::Leaf(marker.clone()));
				first = false;
			} else {
				b.list.insert(0, Node::Glue(Glue::fixed(indent)));
			}
			b.dims = Dims::new(b.dims.width + indent, b.dims.height, b.dims.depth);
		}
	}
}

/// Builds a footnote from its already-shaped body mark and its note text. The note is set as a small
/// paragraph at the foot measure, prefixed by the number as a hanging superscript, and its stacked
/// height noted so the page breaker can reserve it.
fn build_footnote(
	fonts:		Arc<FontSet>,
	style:		Style,
	measure:	Sp,
	number:		u32,
	note:		&[Segment],
	mark:		ShapedText,
)
	-> Outcome<Footnote>
{
	// The note's own inline runs, so a `*strong*` or `_emph_` term in the note sets with its own face
	// rather than flattening to upright text. A nested footnote or a cross-reference in a note -- rare --
	// sets nothing here, as a footnote carries no counter or reserved slot of its own.
	let pieces = res!(footnote_pieces(fonts.clone(), style, note));

	// The number sets as a small superscript that hangs to the left of the note: the note breaks at a
	// measure reduced by the mark's hang, its first line carries the mark and a gap that together fill the
	// hang, and every continuation line is shifted right by it, so the note's text block sits proud of its
	// mark exactly as Typst hangs a footnote.
	let (pre_shaped, pre_dims)	= res!(superscript(fonts.clone(), Role::Body, style.foot_size, &fmt!("{}", number)));
	let gap		= Sp(style.foot_size.raw() / 4);
	let hang	= pre_dims.width + gap;
	let inner	= if measure > hang { measure - hang } else { measure };

	let mut lines = res!(break_paragraph_pieces(
		fonts.clone(), Role::Body, Dir::Ltr, style.foot_size, &pieces, inner, style.foot_leading, true));
	indent_item(&mut lines, Leaf::text_dims(pre_shaped, pre_dims), hang);

	let mut height = Sp::ZERO;
	for n in &lines {
		height += n.vextent();
	}

	Ok(Footnote { number, mark, note: lines, height })
}

/// Turns a footnote's inline runs into the pieces the line breaker weaves: a text run keeps its face, a
/// `*strong*` sets bold, an `_emph_` italic, a superscript rides raised, a code span sets mono, an in-note
/// maths span is flattened to leaves, and a glossary term sets its display text. A nested footnote, a
/// cross-reference and a citation are set as plain text or dropped, since a footnote carries no counter,
/// reserved page slot or bibliography of its own at this increment.
fn footnote_pieces(
	fonts:		Arc<FontSet>,
	style:		Style,
	segments:	&[Segment],
)
	-> Outcome<Vec<Piece>>
{
	let size = style.foot_size;
	let mut pieces = Vec::with_capacity(segments.len());
	for seg in segments {
		match seg {
			Segment::Text(t)		=> pieces.push(Piece::Text { text: t.clone(), role: Role::Body }),
			Segment::Strong(t)		=> pieces.push(Piece::Text { text: t.clone(), role: Role::Bold }),
			Segment::Emph(t)		=> pieces.push(Piece::Text { text: t.clone(), role: Role::Italic }),
			Segment::BoldItalic(t)	=> pieces.push(Piece::Text { text: t.clone(), role: Role::BoldItalic }),
			Segment::Code(t)		=> pieces.push(Piece::Text { text: t.clone(), role: Role::Mono }),
			Segment::Glossary { display, .. }
									=> pieces.push(Piece::Text { text: display.clone(), role: Role::Body }),
			Segment::Cite(keys)		=> pieces.push(Piece::Text { text: fmt!("({})", keys.join("; ")), role: Role::Body }),
			Segment::PageRef(_)		=> {},	// a cross-reference in a note carries no reserved slot here
			Segment::Footnote { .. }	=> {},	// a nested footnote is not set within a footnote
			Segment::Super(t) => {
				let (shaped, dims) = res!(superscript(fonts.clone(), Role::Body, size, t));
				pieces.push(Piece::Mark(Leaf::text_dims(shaped, dims)));
			},
			Segment::Math(expr) => {
				let node = res!(math::layout(fonts.clone(), &style, expr, false));
				if let Node::HBox(b) = node {
					let ascent	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, size, "0")).dims().height;
					let over	= if b.dims.height > ascent { b.dims.height - ascent } else { Sp::ZERO };
					pieces.push(Piece::Math { nodes: b.list, width: b.dims.width, height: ascent, depth: b.dims.depth, over });
				}
			},
		}
	}
	Ok(pieces)
}

/// Shapes a short run at `0.7x` the surrounding size and returns it with the box that raises its
/// baseline. The box height is the surrounding ascent less a raise of a third of that ascent; the
/// emitter draws a run's baseline at `y + height`, so a shorter box lifts the run above the line's
/// baseline. The width and depth are the small run's own, keeping the mark narrow.
pub(crate) fn superscript(
	fonts:	Arc<FontSet>,
	role:	Role,
	base:	Sp,
	text:	&str,
)
	-> Outcome<(ShapedText, Dims)>
{
	let small	= Sp(base.raw() * 7 / 10);
	let shaped	= res!(ShapedText::new(fonts.clone(), role, Dir::Ltr, small, text));
	let sd		= shaped.dims();

	// The surrounding line's ascent, taken from a body-size digit, and the raise off its baseline.
	let sample	= res!(ShapedText::new(fonts, role, Dir::Ltr, base, "0"));
	let ascent	= sample.dims().height;
	let raise	= Sp(ascent.raw() * 35 / 100);
	let height	= if ascent > raise { ascent - raise } else { ascent };

	Ok((shaped, Dims::new(sd.width, height, sd.depth)))
}

/// Sets a display equation as a centred line, appended to the vertical list. The maths box is laid
/// out, its returned HBox unwrapped, and its leaves centred in the measure; a numbered equation gets
/// its number flush at the right margin and an [`Equation`](crate::ledger::AnchorKind::Equation) anchor
/// recorded just before the line, so the ledger can later resolve a reference to it. The line's height
/// and depth take the greater of the maths extent and a body digit, so a short equation still leaves
/// room for its number.
fn equation(
	nodes:		&mut Vec<Node>,
	fonts:		Arc<FontSet>,
	style:		Style,
	measure:	Sp,
	expr:		&Atom,
	number:		Option<u32>,
)
	-> Outcome<()>
{
	let node = res!(math::layout(fonts.clone(), &style, expr, true));
	let (list, dims) = match node {
		Node::HBox(b)	=> (b.list, b.dims),
		_				=> return Err(err!(
			"Maths layout returned a non-HBox node for a display equation."; Bug)),
	};

	let w		= dims.width;
	let centre	= if measure > w { Sp((measure.raw() - w.raw()) / 2) } else { Sp::ZERO };
	let baseline	= dims.height;	// the maths baseline's distance below the line top

	// A body digit fixes the line's minimum height and depth, so the number is never clipped when the
	// maths sits shallow.
	let sample	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, style.body_size, "0"));
	let height	= if baseline > sample.dims().height { baseline } else { sample.dims().height };
	let depth	= if dims.depth > sample.dims().depth { dims.depth } else { sample.dims().depth };

	let mut children:	Vec<Node> = Vec::new();
	if centre.raw() > 0 {
		children.push(Node::Glue(Glue::fixed(centre)));
	}
	for n in list {
		children.push(n);
	}
	let cursor = centre + w;	// where the maths ends, from the line's left

	if let Some(num) = number {
		let label	= fmt!("({})", num);
		let shaped	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, style.body_size, &label));
		let nw		= shaped.dims().width;
		let target	= if measure > nw { measure - nw } else { cursor };
		if target > cursor {
			children.push(Node::Glue(Glue::fixed(target - cursor)));
		}
		// The number sits on the maths baseline; a zero-height leaf plus the baseline shift seats it there.
		let leaf = Leaf::text_dims(shaped, Dims::new(nw, Sp::ZERO, Sp::ZERO)).with_shift(baseline);
		children.push(Node::Leaf(leaf));

		let id = AnchorId::new(AnchorKind::Equation, fmt!("eq-{}", num));
		nodes.push(Node::Anchor(id));
	}

	nodes.push(Node::HBox(BoxNode::new(children, Dims::new(measure, height, depth))));
	Ok(())
}

/// Sets a figure: its identity as a [`Float`](crate::ledger::AnchorKind::Float) anchor, the graphic
/// centred on its own line, and a caption centred beneath. The graphic's dimensions are its bounding
/// box, `height` the whole visual extent and `depth` zero, so the line advances by the figure's height
/// and the greedy breaker moves it whole. The anchor is recorded before the ink so a reference to the
/// figure resolves the page it lands on.
fn figure(
	nodes:		&mut Vec<Node>,
	fonts:		Arc<FontSet>,
	style:		Style,
	measure:	Sp,
	graphic:	Graphic,
	caption:	Option<&str>,
	number:		u32,
)
	-> Outcome<()>
{
	let id = AnchorId::new(AnchorKind::Float, fmt!("fig-{}", number));
	nodes.push(Node::Anchor(id));

	// The graphic centred: a fixed box with glue to its left, on a line whose height is the figure's.
	let leaf	= Leaf::graphic(graphic);
	let gw		= leaf.dims.width;
	let gh		= leaf.dims.height + leaf.dims.depth;
	let pad		= if measure > gw { Sp((measure.raw() - gw.raw()) / 2) } else { Sp::ZERO };
	let mut row:	Vec<Node> = Vec::new();
	if pad.raw() > 0 {
		row.push(Node::Glue(Glue::fixed(pad)));
	}
	row.push(Node::Leaf(leaf));
	nodes.push(Node::HBox(BoxNode::new(row, Dims::new(measure, gh, Sp::ZERO))));

	// The caption, centred beneath the figure, set in the italic at the footnote size.
	let text = match caption {
		Some(c)	=> fmt!("Figure {}.  {}", number, c),
		None	=> fmt!("Figure {}.", number),
	};
	nodes.push(Node::Glue(Glue::fixed(Sp::from_pt(5.0))));
	let shaped	= res!(ShapedText::new(fonts, Role::Italic, Dir::Ltr, style.foot_size, &text));
	let cd		= shaped.dims();
	let cpad	= if measure > cd.width { Sp((measure.raw() - cd.width.raw()) / 2) } else { Sp::ZERO };
	let mut crow:	Vec<Node> = Vec::new();
	if cpad.raw() > 0 {
		crow.push(Node::Glue(Glue::fixed(cpad)));
	}
	crow.push(Node::Leaf(Leaf::text(shaped)));
	nodes.push(Node::HBox(BoxNode::new(crow, Dims::new(measure, cd.height, cd.depth))));
	Ok(())
}

/// The next number for a figure supplement, incrementing its running count so tables and figures carry
/// independent sequences.
fn next_number(counters: &mut HashMap<String, u32>, supplement: &str) -> u32 {
	let n = counters.entry(supplement.to_string()).or_insert(0);
	*n += 1;
	*n
}

/// Sets a table wrapped in a figure: the figure's anchors, the ruled table as one keep box, then a
/// numbered caption beneath. The table lowers exactly as a bare [`Block::Table`] does, so it moves whole
/// to the next page when it will not fit where it stands.
#[allow(clippy::too_many_arguments)]
fn table_figure(
	nodes:		&mut Vec<Node>,
	fonts:		Arc<FontSet>,
	style:		Style,
	measure:	Sp,
	table:		&Table,
	caption:	Option<&[Segment]>,
	supplement:	&str,
	number:		u32,
	label:		Option<&str>,
	refs:		&HashMap<String, String>,
)
	-> Outcome<()>
{
	figure_anchors(nodes, supplement, number, label);
	nodes.push(res!(table::lower(fonts.clone(), style, measure, table, refs)));
	nodes.push(Node::Glue(Glue::fixed(Sp::from_pt(5.0))));
	res!(captioned(nodes, fonts, style, measure, supplement, number, caption));
	Ok(())
}

/// Sets an image wrapped in a figure: the figure's anchors, the loaded raster centred in the measure,
/// then a numbered caption beneath. A path that resolves to nothing, or a vector SVG with no raster
/// beside it, falls back to the placeholder box, which holds the same space so pagination is unchanged.
#[allow(clippy::too_many_arguments)]
fn image_figure(
	nodes:		&mut Vec<Node>,
	fonts:		Arc<FontSet>,
	style:		Style,
	measure:	Sp,
	path:		&str,
	width:		Option<Length>,
	height:		Option<Length>,
	scale:		Option<f64>,
	caption:	Option<&[Segment]>,
	supplement:	&str,
	number:		u32,
	label:		Option<&str>,
)
	-> Outcome<()>
{
	figure_anchors(nodes, supplement, number, label);

	// The loaded figure sized to the measure, or the placeholder box when nothing loads. A load failure
	// is not fatal: the figure keeps its space and its caption, and the missing ink is a reported gap. A
	// raster fills a rectangle; an SVG is drawn as its own scaled paths.
	let graphic = match crate::image::load_figure(path) {
		Ok(crate::image::Figure::Raster(img))	=> res!(image_graphic(measure, img, width, height, scale)),
		Ok(crate::image::Figure::Vector(pic))	=> res!(svg_graphic(fonts.clone(), measure, pic, width, height, scale)),
		Err(_)									=> res!(placeholder(measure)),
	};
	let leaf	= Leaf::graphic(graphic);
	let gw		= leaf.dims.width;
	let gh		= leaf.dims.height + leaf.dims.depth;
	let pad		= if measure > gw { Sp((measure.raw() - gw.raw()) / 2) } else { Sp::ZERO };
	let mut row:	Vec<Node> = Vec::new();
	if pad.raw() > 0 {
		row.push(Node::Glue(Glue::fixed(pad)));
	}
	row.push(Node::Leaf(leaf));
	nodes.push(Node::HBox(BoxNode::new(row, Dims::new(measure, gh, Sp::ZERO))));
	nodes.push(Node::Glue(Glue::fixed(Sp::from_pt(5.0))));
	res!(captioned(nodes, fonts, style, measure, supplement, number, caption));
	Ok(())
}

/// Sets a plain centred image with no figure number or caption -- a `#padded-image`/`#image` section
/// opener's logo. The image is sized and loaded exactly as a figure's is (an SVG drawn as its own scaled
/// paths, a raster to fill its box, a failed load standing in with the placeholder), then centred in the
/// measure with the template's 10 pt of padding above and below, so the words after it keep their air.
fn plain_image(
	nodes:		&mut Vec<Node>,
	fonts:		Arc<FontSet>,
	measure:	Sp,
	path:		&str,
	width:		Option<Length>,
	height:		Option<Length>,
	scale:		Option<f64>,
)
	-> Outcome<()>
{
	let graphic = match crate::image::load_figure(path) {
		Ok(crate::image::Figure::Raster(img))	=> res!(image_graphic(measure, img, width, height, scale)),
		Ok(crate::image::Figure::Vector(pic))	=> res!(svg_graphic(fonts.clone(), measure, pic, width, height, scale)),
		Err(_)									=> res!(placeholder(measure)),
	};
	let pad = Sp::from_pt(10.0);	// the template's `padded-image` padding, above and below
	nodes.push(Node::Glue(Glue::fixed(pad)));
	let leaf	= Leaf::graphic(graphic);
	let gw		= leaf.dims.width;
	let gh		= leaf.dims.height + leaf.dims.depth;
	let lpad	= if measure > gw { Sp((measure.raw() - gw.raw()) / 2) } else { Sp::ZERO };
	let mut row:	Vec<Node> = Vec::new();
	if lpad.raw() > 0 {
		row.push(Node::Glue(Glue::fixed(lpad)));
	}
	row.push(Node::Leaf(leaf));
	nodes.push(Node::HBox(BoxNode::new(row, Dims::new(measure, gh, Sp::ZERO))));
	nodes.push(Node::Glue(Glue::fixed(pad)));
	Ok(())
}

/// Sets a figure drawn by code: the figure's anchors, the built graphic centred in the measure (scaled
/// down uniformly if it is wider than the measure), then a numbered caption beneath. Building can fail --
/// a malformed diagram -- in which case the placeholder holds the space so pagination is unchanged.
#[allow(clippy::too_many_arguments)]
fn code_figure(
	nodes:		&mut Vec<Node>,
	fonts:		Arc<FontSet>,
	style:		Style,
	measure:	Sp,
	figure:		&crate::lang::codefig::CodeFigure,
	caption:	Option<&[Segment]>,
	supplement:	&str,
	number:		u32,
	label:		Option<&str>,
)
	-> Outcome<()>
{
	figure_anchors(nodes, supplement, number, label);

	let graphic = match figure.build(fonts.clone()) {
		Ok(g)	=> res!(fit_graphic(g, measure)),
		Err(_)	=> res!(placeholder(measure)),
	};
	let leaf	= Leaf::graphic(graphic);
	let gw		= leaf.dims.width;
	let gh		= leaf.dims.height + leaf.dims.depth;
	let pad		= if measure > gw { Sp((measure.raw() - gw.raw()) / 2) } else { Sp::ZERO };
	let mut row:	Vec<Node> = Vec::new();
	if pad.raw() > 0 {
		row.push(Node::Glue(Glue::fixed(pad)));
	}
	row.push(Node::Leaf(leaf));
	nodes.push(Node::HBox(BoxNode::new(row, Dims::new(measure, gh, Sp::ZERO))));
	nodes.push(Node::Glue(Glue::fixed(Sp::from_pt(5.0))));
	res!(captioned(nodes, fonts, style, measure, supplement, number, caption));
	Ok(())
}

/// Scales a built graphic down uniformly if it is wider than the measure, so a wide diagram fits the text
/// block; a graphic already within the measure is returned unchanged. Every path is carried through the
/// same factor, and the dimensions follow.
fn fit_graphic(g: Graphic, measure: Sp) -> Outcome<Graphic> {
	let w = g.dims.width;
	if w <= measure || w.raw() <= 0 {
		return Ok(g);
	}
	let s = measure.to_pt() as f32 / w.to_pt() as f32;
	let t = Transform::scale(s, s);
	let mut ops: Vec<DrawOp> = Vec::with_capacity(g.ops.len());
	for op in g.ops {
		ops.push(match op {
			DrawOp::Fill { path, colour } => DrawOp::Fill { path: res!(path.transform(&t)), colour },
			DrawOp::Stroke { path, colour, width } => DrawOp::Stroke {
				path:	res!(path.transform(&t)),
				colour,
				width:	width * s,
			},
			DrawOp::Image { image, x, y, w, h } => DrawOp::Image {
				image, x: x * s, y: y * s, w: w * s, h: h * s,
			},
		});
	}
	let dims = Dims::new(
		Sp::from_pt(g.dims.width.to_pt() * s as f64),
		Sp::from_pt(g.dims.height.to_pt() * s as f64),
		Sp::from_pt(g.dims.depth.to_pt() * s as f64),
	);
	Ok(Graphic::new(ops, dims))
}

/// Builds a graphic that draws a loaded raster to fill a box sized from the declared hints and the
/// image's own aspect. With no hint the image fills the measure; a `width`/`height` in the source sets
/// that axis and the other follows the aspect; a hint that would overflow the measure is clamped to it.
/// A single [`DrawOp::Image`] carries the pixels, so the emitters place one raster per figure.
fn image_graphic(
	measure:	Sp,
	img:		RasterImage,
	width:		Option<Length>,
	height:		Option<Length>,
	scale:		Option<f64>,
)
	-> Outcome<Graphic>
{
	let m		= measure.to_pt();
	let iw		= img.width.max(1) as f64;
	let ih		= img.height.max(1) as f64;
	let aspect	= ih / iw;

	// Resolve the declared width and height to points; a percentage is of the measure, a length absolute.
	let resolve = |len: Length| -> f64 {
		match len {
			Length::Rel(f)	=> m * f,
			Length::Abs(pt)	=> pt,
		}
	};

	// A width wins the sizing; else a height sets it through the aspect; else the image fills the
	// measure. `scale` on a `padded-image` multiplies a filled measure, so a 100% scale is the measure.
	let mut w = match (width, height) {
		(Some(wl), _)		=> resolve(wl),
		(None, Some(hl))	=> resolve(hl) / aspect,
		(None, None)		=> m * scale.unwrap_or(1.0),
	};
	if w > m || w <= 0.0 {
		w = m;
	}
	let h = match height {
		Some(hl) if width.is_none() && scale.is_none()	=> resolve(hl),
		_												=> w * aspect,
	};

	let wf	= w as f32;
	let hf	= h as f32;
	let ops	= vec![DrawOp::Image { image: Arc::new(img), x: 0.0, y: 0.0, w: wf, h: hf }];
	Ok(Graphic::new(ops, Dims::new(Sp::from_pt(w), Sp::from_pt(h), Sp::ZERO)))
}

/// Builds a graphic from a read SVG, scaled to fit the box the sizing hints and the picture's own aspect
/// ask for -- the same sizing a raster gets -- and its paths mapped to fill and stroke ops.
///
/// The picture comes out of the reader in its viewBox units, which for a typesetter's SVG are points, so
/// the intrinsic size stands in for a raster's pixel dimensions. One uniform factor scales every path;
/// a dashed or a capped stroke is baked to a filled outline first, since a plain [`DrawOp::Stroke`]
/// carries only a width, and the emitter would otherwise draw it solid. An illustrator's live `<text>`
/// arrives unshaped, so it is shaped here with the book's font set and baked to glyph outlines, and an
/// embedded raster is placed as a scaled [`DrawOp::Image`].
fn svg_graphic(
	fonts:		Arc<FontSet>,
	measure:	Sp,
	pic:		SvgPicture,
	width:		Option<Length>,
	height:		Option<Length>,
	scale:		Option<f64>,
)
	-> Outcome<Graphic>
{
	let m		= measure.to_pt();
	let iw		= (pic.width as f64).max(1.0);
	let ih		= (pic.height as f64).max(1.0);
	let aspect	= ih / iw;

	let resolve = |len: Length| -> f64 {
		match len {
			Length::Rel(f)	=> m * f,
			Length::Abs(pt)	=> pt,
		}
	};
	let mut w = match (width, height) {
		(Some(wl), _)		=> resolve(wl),
		(None, Some(hl))	=> resolve(hl) / aspect,
		(None, None)		=> m * scale.unwrap_or(1.0),
	};
	if w > m || w <= 0.0 {
		w = m;
	}
	let h = match height {
		Some(hl) if width.is_none() && scale.is_none()	=> resolve(hl),
		_												=> w * aspect,
	};

	// A uniform factor from the picture's intrinsic width to the drawn width; the height follows the
	// same factor, since the aspect was preserved above.
	let s	= (w / iw) as f32;
	let t	= Transform::scale(s, s);
	let mut ops: Vec<DrawOp> = Vec::with_capacity(pic.ops.len());
	for op in pic.ops {
		match op {
			SvgOp::Fill { path, colour } => {
				ops.push(DrawOp::Fill { path: res!(path.transform(&t)), colour });
			},
			SvgOp::Stroke { path, colour, stroke } => {
				if stroke.dash.is_some() {
					// Bake the dashes into an outline in the picture's frame, then scale that with the rest.
					let outline = res!(path.stroke(&stroke));
					ops.push(DrawOp::Fill { path: res!(outline.transform(&t)), colour });
				} else {
					ops.push(DrawOp::Stroke {
						path:	res!(path.transform(&t)),
						colour,
						width:	stroke.width * s,
					});
				}
			},
			SvgOp::Text { text, local, x, y, size, anchor, italic, bold, colour } => {
				res!(bake_svg_text(
					&mut ops, fonts.clone(), &text, &local, &t, x, y, size, anchor, italic, bold, colour));
			},
			SvgOp::Image { rgba, iw, ih, x, y, w: iwd, h: ihd } => {
				// The raster's placement rectangle is in the picture frame; the same factor scales it.
				let img = RasterImage { width: iw, height: ih, rgba };
				ops.push(DrawOp::Image {
					image:	Arc::new(img),
					x:		x * s,
					y:		y * s,
					w:		iwd * s,
					h:		ihd * s,
				});
			},
		}
	}
	Ok(Graphic::new(ops, Dims::new(Sp::from_pt(w), Sp::from_pt(h), Sp::ZERO)))
}

/// Shapes one live SVG text run with the book's font set and bakes it to filled glyph outlines. The run
/// is shaped at its own font-size in the picture's units; `local` maps that frame to the picture frame
/// and `t` the picture frame to the drawn frame. The anchor slides the pen from the run's start once the
/// advance is known, and each glyph's y-up outline is flipped onto the SVG's y-down baseline before the
/// two frame transforms carry it home -- the same bake the diagram and plot labels use.
#[allow(clippy::too_many_arguments)]
fn bake_svg_text(
	ops:	&mut Vec<DrawOp>,
	fonts:	Arc<FontSet>,
	text:	&str,
	local:	&Transform,
	t:		&Transform,
	x:		f32,
	y:		f32,
	size:	f32,
	anchor:	Anchor,
	italic:	bool,
	bold:	bool,
	colour:	Rgba,
)
	-> Outcome<()>
{
	if size <= 0.0 {
		return Ok(());
	}
	let role = match (bold, italic) {
		(true, true)	=> Role::BoldItalic,
		(true, false)	=> Role::Bold,
		(false, true)	=> Role::Italic,
		(false, false)	=> Role::Body,
	};
	let shaped	= res!(ShapedText::new(fonts, role, Dir::Ltr, Sp::from_pt(size as f64), text));
	let advance	= shaped.dims().width.to_pt() as f32;
	let pen_x	= match anchor {
		Anchor::Start	=> x,
		Anchor::Middle	=> x - advance / 2.0,
		Anchor::End		=> x - advance,
	};
	for glyph in &shaped.run().glyphs {
		let outline = res!(shaped.outline(glyph));
		if outline.is_empty() {
			continue;	// a space carries an advance but no ink
		}
		let place = Transform::scale(1.0, -1.0)
			.then(&Transform::translate(pen_x + glyph.x, y - glyph.y))
			.then(local)
			.then(t);
		ops.push(DrawOp::Fill { path: res!(outline.transform(&place)), colour });
	}
	Ok(())
}

/// Records a figure's anchors: an author label (when the source labelled it) so a cross-reference
/// resolves the figure's page, and a [`Float`](crate::ledger::AnchorKind::Float) anchor keyed by
/// supplement and number for the figure's own identity.
fn figure_anchors(nodes: &mut Vec<Node>, supplement: &str, number: u32, label: Option<&str>) {
	if let Some(l) = label {
		nodes.push(Node::Anchor(AnchorId::new(AnchorKind::Label, l.to_string())));
	}
	nodes.push(Node::Anchor(AnchorId::new(
		AnchorKind::Float, fmt!("{}-{}", supplement.to_lowercase(), number))));
}

/// One typeset unit of a caption: an unbreakable cluster of one or more boxes (a word, or a word with an
/// attached superscript, or a maths cluster) with its extent, or a breakable interword space.
enum CapTok {
	Unit { nodes: Vec<Node>, width: Sp, height: Sp, depth: Sp },
	Space,
}

/// Sets a figure caption -- "{supplement} {number}: {caption}" -- centred beneath the figure, wrapped
/// greedily into ragged centred lines at the body size. The caption's own runs are set with their faces,
/// so an emphasised word, a superscript or an in-caption maths span renders rather than flattening to
/// upright text or vanishing. A caption with no text sets just its number.
fn captioned(
	nodes:		&mut Vec<Node>,
	fonts:		Arc<FontSet>,
	style:		Style,
	measure:	Sp,
	supplement:	&str,
	number:		u32,
	caption:	Option<&[Segment]>,
)
	-> Outcome<()>
{
	let size	= style.body_size;
	let space_w	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, size, " ")).dims().width;

	// The leading "{supplement} {number}: " (or just the number when the caption has no text), then the
	// caption's segments, tokenised into words, spaces, superscripts and maths clusters. A shared
	// pending-space flag carries a run's trailing space to the next token, so spacing follows the source.
	let has_text	= caption.map(|c| segments_have_text(c)).unwrap_or(false);
	let prefix		= if has_text { fmt!("{} {}: ", supplement, number) } else { fmt!("{} {}", supplement, number) };

	let mut toks:	Vec<CapTok>	= Vec::new();
	let mut pending				= false;
	res!(push_caption_text(&mut toks, &mut pending, fonts.clone(), Role::Body, size, &prefix));
	if let Some(segs) = caption {
		for seg in segs {
			match seg {
				Segment::Text(t)	=> res!(push_caption_text(&mut toks, &mut pending, fonts.clone(), Role::Body, size, t)),
				Segment::Strong(t)		=> res!(push_caption_text(&mut toks, &mut pending, fonts.clone(), Role::Bold, size, t)),
				Segment::Emph(t)		=> res!(push_caption_text(&mut toks, &mut pending, fonts.clone(), Role::Italic, size, t)),
				Segment::BoldItalic(t)	=> res!(push_caption_text(&mut toks, &mut pending, fonts.clone(), Role::BoldItalic, size, t)),
				Segment::Code(t)	=> res!(push_caption_text(&mut toks, &mut pending, fonts.clone(), Role::Mono, size, t)),
				Segment::Glossary { display, .. }
									=> res!(push_caption_text(&mut toks, &mut pending, fonts.clone(), Role::Body, size, display)),
				Segment::Cite(keys)	=> res!(push_caption_text(
										&mut toks, &mut pending, fonts.clone(), Role::Body, size, &fmt!("({})", keys.join("; ")))),
				Segment::PageRef(_)		=> {},	// a cross-reference in a caption is not resolved here
				Segment::Footnote { .. }	=> {},	// a footnote in a caption is not set here
				Segment::Super(t) => {
					let (shaped, dims) = res!(superscript(fonts.clone(), Role::Body, size, t));
					push_caption_box(&mut toks, &mut pending,
						vec![Node::Leaf(Leaf::text_dims(shaped, dims))], dims.width, dims.height, dims.depth);
				},
				Segment::Math(expr) => {
					let node = res!(math::layout(fonts.clone(), &style, expr, false));
					if let Node::HBox(b) = node {
						push_caption_box(&mut toks, &mut pending, b.list, b.dims.width, b.dims.height, b.dims.depth);
					}
				},
			}
		}
	}

	// Greedy line fill: units joined by single spaces, broken before the unit that would overrun the
	// measure. Each finished line is centred by a left glue of half its slack.
	let mut line:	Vec<&CapTok>	= Vec::new();
	let mut line_w					= Sp::ZERO;
	let mut first					= true;
	for tok in &toks {
		if let CapTok::Unit { width, .. } = tok {
			let add = if line.is_empty() { *width } else { space_w + *width };
			if !line.is_empty() && line_w + add > measure {
				res!(emit_caption_units(nodes, style, measure, space_w, &line, line_w, &mut first));
				line.clear();
				line_w = Sp::ZERO;
			}
			line_w += if line.is_empty() { *width } else { space_w + *width };
			line.push(tok);
		}
	}
	if !line.is_empty() {
		res!(emit_caption_units(nodes, style, measure, space_w, &line, line_w, &mut first));
	}
	Ok(())
}

/// Whether any caption segment carries visible text, so the colon prefix is set only for a real caption.
fn segments_have_text(segs: &[Segment]) -> bool {
	segs.iter().any(|s| match s {
		Segment::Text(t) | Segment::Strong(t) | Segment::Emph(t) | Segment::BoldItalic(t) | Segment::Code(t) | Segment::Super(t)
							=> !t.trim().is_empty(),
		Segment::Glossary { display, .. }	=> !display.trim().is_empty(),
		Segment::Math(_) | Segment::Cite(_)	=> true,
		_									=> false,
	})
}

/// Tokenises a text run into word units and interword spaces, in the given face, appending to `toks`. A
/// leading or run-crossing space is carried in `pending` and emitted only before the next word, so the
/// source's spacing survives and a trailing space attaches to whatever segment follows.
fn push_caption_text(
	toks:		&mut Vec<CapTok>,
	pending:	&mut bool,
	fonts:		Arc<FontSet>,
	role:		Role,
	size:		Sp,
	text:		&str,
)
	-> Outcome<()>
{
	let mut word = String::new();
	for c in text.chars() {
		if c.is_whitespace() {
			if !word.is_empty() {
				res!(flush_caption_word(toks, pending, fonts.clone(), role, size, &mut word));
			}
			*pending = true;
		} else {
			word.push(c);
		}
	}
	if !word.is_empty() {
		res!(flush_caption_word(toks, pending, fonts.clone(), role, size, &mut word));
	}
	Ok(())
}

/// Shapes one word and pushes it as a unit, emitting a pending space before it when one is due.
fn flush_caption_word(
	toks:		&mut Vec<CapTok>,
	pending:	&mut bool,
	fonts:		Arc<FontSet>,
	role:		Role,
	size:		Sp,
	word:		&mut String,
)
	-> Outcome<()>
{
	let shaped	= res!(ShapedText::new(fonts, role, Dir::Ltr, size, word));
	let d		= shaped.dims();
	push_caption_box(toks, pending, vec![Node::Leaf(Leaf::text(shaped))], d.width, d.height, d.depth);
	word.clear();
	Ok(())
}

/// Pushes a pre-built box as a caption unit, emitting a pending interword space before it first. Adjacent
/// boxes with no pending space between them (a word and its attached superscript) become one unit.
fn push_caption_box(
	toks:		&mut Vec<CapTok>,
	pending:	&mut bool,
	mut boxes:	Vec<Node>,
	width:		Sp,
	height:		Sp,
	depth:		Sp,
)
{
	if *pending {
		toks.push(CapTok::Space);
		*pending = false;
	} else if let Some(CapTok::Unit { nodes, width: w, height: h, depth: dp }) = toks.last_mut() {
		// No space since the previous unit: attach to it, so a word and its superscript stay unbreakable.
		nodes.append(&mut boxes);
		*w		= *w + width;
		*h		= (*h).max(height);
		*dp		= (*dp).max(depth);
		return;
	}
	toks.push(CapTok::Unit { nodes: boxes, width, height, depth });
}

/// Sets one centred caption line from its units, with interline leading before every line but the first.
fn emit_caption_units(
	nodes:		&mut Vec<Node>,
	style:		Style,
	measure:	Sp,
	space_w:	Sp,
	line:		&[&CapTok],
	line_w:		Sp,
	first:		&mut bool,
)
	-> Outcome<()>
{
	let mut height	= Sp::ZERO;
	let mut depth	= Sp::ZERO;
	for tok in line {
		if let CapTok::Unit { height: h, depth: d, .. } = tok {
			height	= height.max(*h);
			depth	= depth.max(*d);
		}
	}
	if !*first {
		let vext	= height + depth;
		let gap		= if style.leading > vext { style.leading - vext } else { style.line_gap };
		nodes.push(Node::Glue(Glue::fixed(gap)));
	}
	*first = false;

	let pad = if measure > line_w { Sp((measure.raw() - line_w.raw()) / 2) } else { Sp::ZERO };
	let mut row:	Vec<Node> = Vec::new();
	if pad.raw() > 0 {
		row.push(Node::Glue(Glue::fixed(pad)));
	}
	for (k, tok) in line.iter().enumerate() {
		if let CapTok::Unit { nodes: ns, .. } = tok {
			if k > 0 {
				row.push(Node::Glue(Glue::fixed(space_w)));	// the single interword space between units
			}
			for n in ns { row.push(n.clone()); }
		}
	}
	nodes.push(Node::HBox(BoxNode::new(row, Dims::new(measure, height, depth))));
	Ok(())
}

/// Builds the placeholder box that stands in for an image this increment does not load: a light-filled,
/// lightly-stroked rectangle the width of the measure and half as tall, capped so a wide page does not
/// leave a giant void. The caption beneath still names the figure.
fn placeholder(measure: Sp) -> Outcome<Graphic> {
	let w	= measure.to_pt() as f32;
	let h	= (w * 0.5).clamp(120.0, 360.0);
	let mut pb = PathBuilder::new();
	pb.move_to(Pt::new(0.0, 0.0));
	pb.line_to(Pt::new(w, 0.0));
	pb.line_to(Pt::new(w, h));
	pb.line_to(Pt::new(0.0, h));
	pb.close();
	let path	= res!(pb.finish());
	let ops		= vec![
		DrawOp::Fill { path: path.clone(), colour: Rgba::opaque(238, 238, 240) },
		DrawOp::Stroke { path, colour: Rgba::opaque(150, 150, 150), width: 0.8 },
	];
	Ok(Graphic::new(ops, Dims::new(Sp::from_pt(w as f64), Sp::from_pt(h as f64), Sp::ZERO)))
}

// ┌───────────────────────────────────────────────────────────────────────────┐
// │ FRONT MATTER                                                               │
// └───────────────────────────────────────────────────────────────────────────┘

/// Composes the front matter ahead of the body: the cover raster (a development build only), the title
/// page, the imprint, an optional dedication, and an optional author biography, each on its own page
/// closed by a forced break. None of these leaves sets a heading anchor, so the body's first heading
/// still fixes where the printed folio restarts at one.
fn front_matter(
	nodes:		&mut Vec<Node>,
	fonts:		&Arc<FontSet>,
	display:	Option<&Arc<Font>>,
	geom:		PageGeometry,
	style:		Style,
	fm:			&FrontMatter,
)
	-> Outcome<()>
{
	// Cover: the raster filling the content box, a development build only. A path that will not load
	// (an SVG, or a missing file) sets no cover page rather than a placeholder.
	if let Some(path) = &fm.cover_image {
		if let Ok(node) = fm_cover_node(geom, path) {
			nodes.push(node);
			nodes.push(Node::Penalty(Penalty::eject()));
		}
	}

	// A documentation tree draws the template's two-column title page (a coloured sidebar with its logos and
	// the title on the right); a book draws its plain centred title page. The sidebar grey marks the idiom.
	// A `Label` anchor at the leaf's top records its page for the PDF outline; it sets no heading, so it
	// stays out of the running heads and the contents.
	nodes.push(Node::Anchor(AnchorId::new(AnchorKind::Label, "frontmatter:title")));
	if fm.sidebar_grey.is_some() {
		res!(fm_doc_title_page(nodes, fonts, geom, fm));
	} else {
		res!(fm_title_page(nodes, fonts, geom, style, fm));
	}
	nodes.push(Node::Penalty(Penalty::eject()));

	// The meta page: a doc tree draws the template's Ver/Date/Author(s)/Notes colophon, a book its plain
	// imprint page. Both push the `frontmatter:meta` anchor so the outline lists a Meta entry at this leaf.
	if fm.sidebar_grey.is_some() {
		if fm_has_doc_meta(fm) {
			nodes.push(Node::Anchor(AnchorId::new(AnchorKind::Label, "frontmatter:meta")));
			res!(fm_doc_meta_page(nodes, fonts, geom, style, fm));
			nodes.push(Node::Penalty(Penalty::eject()));
		}
	} else if fm_has_imprint(fm) {
		nodes.push(Node::Anchor(AnchorId::new(AnchorKind::Label, "frontmatter:meta")));
		res!(fm_meta_page(nodes, fonts, geom, style, fm));
		nodes.push(Node::Penalty(Penalty::eject()));
	}

	if let Some(ded) = &fm.dedication {
		res!(fm_dedication_page(nodes, fonts, geom, style, ded));
		nodes.push(Node::Penalty(Penalty::eject()));
	}

	if let Some(bio) = &fm.about_author {
		res!(fm_about_author_page(nodes, fonts, display, geom, style, fm.back_title_size, bio));
		nodes.push(Node::Penalty(Penalty::eject()));
	}

	Ok(())
}

/// Does the book set any imprint field, so a meta page is worth composing?
fn fm_has_imprint(fm: &FrontMatter) -> bool {
	fm.publisher.is_some() || fm.edition.is_some() || fm.isbn.is_some() || fm.copyright.is_some()
		|| fm.rights.is_some() || fm.ai_declaration.is_some() || fm.website.is_some() || fm.toolchain
}

/// Does the doc tree state a revision, so the template's meta/colophon page is worth composing? A doc
/// root always sets `meta-data` with at least one row, or names an author, so this holds for every doc.
fn fm_has_doc_meta(fm: &FrontMatter) -> bool {
	!fm.meta_rows.is_empty() || !fm.author.is_empty()
}

/// A rigid vertical spacer that a page top does not discard, so front-matter elements sit at fixed
/// fractions of the page down from the top. Modelled as an empty horizontal box of the wanted height,
/// which the greedy breaker advances the cursor by without placing any ink.
fn fm_spacer(height: Sp) -> Node {
	Node::HBox(BoxNode::new(Vec::new(), Dims::new(Sp::ZERO, height, Sp::ZERO)))
}

/// Shapes one line and pushes it centred in the measure, returning its vertical extent so the caller can
/// track the cursor down the page.
fn fm_centred_line(
	nodes:		&mut Vec<Node>,
	fonts:		&Arc<FontSet>,
	display:	Option<&Arc<Font>>,
	role:		Role,
	size:		Sp,
	text:		&str,
	measure:	Sp,
)
	-> Outcome<Sp>
{
	let shaped = match display {
		Some(f)	=> res!(ShapedText::new_with_font((*f).clone(), Dir::Ltr, size, text)),
		None	=> res!(ShapedText::new(fonts.clone(), role, Dir::Ltr, size, text)),
	};
	let d	= shaped.dims();
	let pad	= if measure > d.width { Sp((measure.raw() - d.width.raw()) / 2) } else { Sp::ZERO };
	let mut row: Vec<Node> = Vec::new();
	if pad.raw() > 0 {
		row.push(Node::Glue(Glue::fixed(pad)));
	}
	row.push(Node::Leaf(Leaf::text(shaped)));
	nodes.push(Node::HBox(BoxNode::new(row, Dims::new(measure, d.height, d.depth))));
	Ok(d.height + d.depth)
}

/// Sets a run of text as greedily-wrapped centred lines (the title, which may not fit one line),
/// returning the total vertical extent set.
fn fm_centred_wrap(
	nodes:		&mut Vec<Node>,
	fonts:		&Arc<FontSet>,
	role:		Role,
	size:		Sp,
	text:		&str,
	measure:	Sp,
	leading:	Sp,
)
	-> Outcome<Sp>
{
	let mut line	= String::new();
	let mut total	= Sp::ZERO;
	let mut first	= true;
	let mut flush = |nodes: &mut Vec<Node>, line: &str, first: &mut bool, total: &mut Sp| -> Outcome<()> {
		let shaped	= res!(ShapedText::new(fonts.clone(), role, Dir::Ltr, size, line));
		let d		= shaped.dims();
		if !*first {
			let vext	= d.height + d.depth;
			let gap		= if leading > vext { leading - vext } else { Sp::ZERO };
			nodes.push(Node::Glue(Glue::fixed(gap)));
			*total += gap;
		}
		*first = false;
		let pad	= if measure > d.width { Sp((measure.raw() - d.width.raw()) / 2) } else { Sp::ZERO };
		let mut row: Vec<Node> = Vec::new();
		if pad.raw() > 0 {
			row.push(Node::Glue(Glue::fixed(pad)));
		}
		row.push(Node::Leaf(Leaf::text(shaped)));
		nodes.push(Node::HBox(BoxNode::new(row, Dims::new(measure, d.height, d.depth))));
		*total += d.height + d.depth;
		Ok(())
	};
	for word in text.split_whitespace() {
		let trial	= if line.is_empty() { word.to_string() } else { fmt!("{} {}", line, word) };
		let shaped	= res!(ShapedText::new(fonts.clone(), role, Dir::Ltr, size, &trial));
		if shaped.dims().width > measure && !line.is_empty() {
			res!(flush(nodes, &line, &mut first, &mut total));
			line = word.to_string();
		} else {
			line = trial;
		}
	}
	if !line.is_empty() {
		res!(flush(nodes, &line, &mut first, &mut total));
	}
	Ok(total)
}

/// Builds the cover page: the raster at `path` bleeding to the trim edge on all four sides. The box the
/// breaker measures is the content area, so the cover paginates as a single leaf closed by the caller's
/// eject; the image inside is offset back to the physical page origin and sized to the whole trim, so it
/// paints under the margins to the paper edge. The emitter does not clip a graphic to its box, so the
/// overpaint lands. Page one is a recto -- the inside margin is the left one and no mirror shift applies
/// -- so the offset is simply the top and inside margins.
fn fm_cover_node(geom: PageGeometry, path: &str) -> Outcome<Node> {
	let img	= res!(crate::image::load(path));
	let cw	= geom.content_width();
	let ch	= geom.content_height();
	let ox	= -(geom.content_left().to_pt() as f32);
	let oy	= -(geom.content_top().to_pt() as f32);
	let pw	= geom.width.to_pt() as f32;
	let ph	= geom.height.to_pt() as f32;
	let ops	= vec![DrawOp::Image { image: Arc::new(img), x: ox, y: oy, w: pw, h: ph }];
	let graphic	= Graphic::new(ops, Dims::new(cw, ch, Sp::ZERO));
	Ok(Node::HBox(BoxNode::new(vec![Node::Leaf(Leaf::graphic(graphic))], Dims::new(cw, ch, Sp::ZERO))))
}

/// Sets the title page: the author name in the upper band, the title and subtitle about the centre, and
/// the publisher logo near the foot -- the template's three-band grid, approximated with fixed fractions
/// of the page height.
fn fm_title_page(
	nodes:	&mut Vec<Node>,
	fonts:	&Arc<FontSet>,
	geom:	PageGeometry,
	style:	Style,
	fm:		&FrontMatter,
)
	-> Outcome<()>
{
	let measure	= geom.content_width();
	let h		= geom.content_height();
	let mut y	= Sp::ZERO;

	// The author name, in the upper fifth.
	nodes.push(fm_spacer(Sp(h.raw() * 17 / 100)));
	y += Sp(h.raw() * 17 / 100);
	y += res!(fm_centred_line(nodes, fonts, None, Role::Body, fm.author_size, &fm.author, measure));

	// The title about the vertical centre, wrapped when it will not fit one line, then the subtitle.
	let target = Sp(h.raw() * 38 / 100);
	if target > y {
		nodes.push(fm_spacer(target - y));
		y = target;
	}
	let title_lead = Sp(fm.title_size.raw() * 6 / 5);
	y += res!(fm_centred_wrap(nodes, fonts, Role::Bold, fm.title_size, &fm.title, measure, title_lead));
	if let Some(sub) = &fm.subtitle {
		let gap = Sp(fm.title_size.raw() * 3 / 5);
		nodes.push(fm_spacer(gap));
		y += gap;
		y += res!(fm_centred_line(nodes, fonts, None, Role::Italic, fm.subtitle_size, sub, measure));
	}

	// The publisher logo near the foot, when it loads (an SVG logo does not, and is simply omitted).
	if let Some(logo) = &fm.logo_image {
		if let Ok(node) = fm_logo_node(fonts, geom, style, logo) {
			let target = Sp(h.raw() * 84 / 100);
			if target > y {
				nodes.push(fm_spacer(target - y));
			}
			nodes.push(node);
		}
	}
	Ok(())
}

/// Builds the logo line: the raster at `path` centred at a modest width. An SVG or missing file errors,
/// and the title page omits the logo.
fn fm_logo_node(_fonts: &Arc<FontSet>, geom: PageGeometry, _style: Style, path: &str) -> Outcome<Node> {
	let img	= res!(crate::image::load(path));
	let measure	= geom.content_width();
	let w	= Sp::from_pt(110.0);	// the type scale's logo width, about 110 pt
	let aspect	= (img.height.max(1) as f64) / (img.width.max(1) as f64);
	let hh	= Sp::from_pt(110.0 * aspect);
	let ops	= vec![DrawOp::Image {
		image: Arc::new(img), x: 0.0, y: 0.0, w: w.to_pt() as f32, h: hh.to_pt() as f32 }];
	let graphic	= Graphic::new(ops, Dims::new(w, hh, Sp::ZERO));
	let pad	= if measure > w { Sp((measure.raw() - w.raw()) / 2) } else { Sp::ZERO };
	let mut row: Vec<Node> = Vec::new();
	if pad.raw() > 0 {
		row.push(Node::Glue(Glue::fixed(pad)));
	}
	row.push(Node::Leaf(Leaf::graphic(graphic)));
	Ok(Node::HBox(BoxNode::new(row, Dims::new(measure, hh, Sp::ZERO))))
}

/// Sets the documentation template's two-column title page (`template.typ`'s `title-page`): a full-height
/// coloured sidebar down the left carrying a logo near its top and one near its foot, and the title (large,
/// in small caps or italic) with its subtitle centred on the white right. The whole page is one box whose
/// graphic ops bleed past the box bounds to the paper edges -- the emitter clips nothing -- exactly as
/// `doc_banner` draws its full-bleed bar. Its box origin is the content top-left (y down), so the page
/// origin is `(-inside, -top)` and the paper corner `(page_w - inside, page_h - top)`.
fn fm_doc_title_page(
	nodes:	&mut Vec<Node>,
	fonts:	&Arc<FontSet>,
	geom:	PageGeometry,
	fm:		&FrontMatter,
)
	-> Outcome<()>
{
	let measure	= geom.content_width();
	let box_h	= geom.content_height();
	let il		= geom.content_left().to_pt() as f32;	// left margin, and the sidebar logos' `margins.a4` pad
	let it		= geom.content_top().to_pt() as f32;	// top margin, equal to the template's `margins.a4`
	let pw		= geom.width.to_pt() as f32;
	let ph		= geom.height.to_pt() as f32;
	let frac	= fm.sidebar_frac as f32;
	let side_w	= frac * pw;	// the sidebar width, `margins.title_page` of the page

	let mut ops:	Vec<DrawOp> = Vec::new();

	// The sidebar: a solid rectangle from the page's top-left corner, `side_w` wide and the full page tall.
	let grey	= fm.sidebar_grey.unwrap_or(240);
	let fill	= Rgba::opaque(grey, grey, grey);
	ops.push(DrawOp::Fill {
		path:	res!(Path::rect(Bounds::new(-il, -it, -il + side_w, -it + ph))),
		colour:	fill,
	});

	// The top logo, centred across the sidebar, its top edge one `margins.a4` down from the page top -- which
	// equals the top margin, so its box-frame top is zero. The bottom logo sits one `margins.a4` up from the
	// page foot. Both are drawn at the width the `doc.with` call declared; a logo that will not load is left
	// out, as the template's own missing-image path would leave a gap.
	let side_mid_box	= -il + side_w / 2.0;	// the sidebar's horizontal centre, in the box frame
	if let Some(path) = &fm.top_logo {
		let w = fm.top_logo_width.to_pt() as f32;
		if let Ok((logo, _)) = logo_ops(fonts, path, w, side_mid_box - w / 2.0, 0.0) {
			ops.extend(logo);
		}
	}
	if let Some(path) = &fm.bottom_logo {
		let w = fm.bottom_logo_width.to_pt() as f32;
		if let Ok((logo, lh)) = logo_ops(fonts, path, w, 0.0, 0.0) {
			// Re-place now the height is known: bottom edge one `margins.a4` up from the page foot.
			let dy = (ph - it) - it - lh;
			let placed = res!(translate_ops(logo, side_mid_box - w / 2.0, dy));
			ops.extend(placed);
		}
	}

	// The title and subtitle centred on the right column: from the sidebar's right edge plus the template's
	// 20 pt, running to the page's right margin less 20 pt. The title rides the column's vertical centre
	// (the template's 40%/10%/50% grid seats it at the half), the subtitle two lines below.
	let col_l		= side_w + 20.0;
	let col_w		= pw - side_w - 40.0;
	let centre_box	= -il + col_l + col_w / 2.0;
	let title_size	= 40.0f32;	// the template's fixed title size, independent of the config type scale
	let sub_size	= 20.0f32;
	let title_top	= ph / 2.0 - it;	// the column centre, in the box frame
	let sample		= res!(head_shape(fonts, &HeadFace::Role(Role::Body), Sp::from_pt(title_size as f64), "Ag"));
	let asc			= sample.dims().height.to_pt() as f32;
	let dep			= sample.dims().depth.to_pt() as f32;
	let title_base	= title_top + asc;
	res!(title_run_ops(&mut ops, fonts, &fm.title, title_size, centre_box, title_base, fm.title_smallcaps));
	if let Some(sub) = &fm.subtitle {
		// Two blank lines below the title (the template's `\ \`), then the subtitle in italic.
		let sub_base = title_base + dep + 28.0 + sub_size;
		res!(title_run_ops(&mut ops, fonts, sub, sub_size, centre_box, sub_base, false));
	}

	let graphic = Graphic::new(ops, Dims::new(measure, box_h, Sp::ZERO));
	nodes.push(Node::HBox(BoxNode::new(
		vec![Node::Leaf(Leaf::graphic(graphic))], Dims::new(measure, box_h, Sp::ZERO))));
	Ok(())
}

/// Loads a logo (an SVG drawn as its own scaled paths, or a raster) at the drawn width `w`, translates its
/// ops to `(dx, dy)` in the caller's frame, and returns them with the drawn height. The picture comes out
/// sized to `w` with its aspect kept, so the height stands for where a bottom-aligned logo's top sits.
fn logo_ops(
	fonts:	&Arc<FontSet>,
	path:	&str,
	w:		f32,
	dx:		f32,
	dy:		f32,
)
	-> Outcome<(Vec<DrawOp>, f32)>
{
	let width	= Some(Length::Abs(w as f64));
	let graphic	= match crate::image::load_figure(path) {
		Ok(crate::image::Figure::Raster(img))	=> res!(image_graphic(Sp::from_pt(w as f64), img, width, None, None)),
		Ok(crate::image::Figure::Vector(pic))	=> res!(svg_graphic(fonts.clone(), Sp::from_pt(w as f64), pic, width, None, None)),
		Err(e)									=> return Err(e),
	};
	let h	= (graphic.dims.height + graphic.dims.depth).to_pt() as f32;
	let ops	= res!(translate_ops(graphic.ops, dx, dy));
	Ok((ops, h))
}

/// Translates every op of a graphic by `(dx, dy)` -- the fill and stroke paths through a translation, an
/// embedded raster by shifting its placement corner. Used to seat a logo built at the origin where it belongs.
fn translate_ops(src: Vec<DrawOp>, dx: f32, dy: f32) -> Outcome<Vec<DrawOp>> {
	let t = Transform::translate(dx, dy);
	let mut out: Vec<DrawOp> = Vec::with_capacity(src.len());
	for op in src {
		out.push(match op {
			DrawOp::Fill { path, colour }			=> DrawOp::Fill { path: res!(path.transform(&t)), colour },
			DrawOp::Stroke { path, colour, width }	=> DrawOp::Stroke { path: res!(path.transform(&t)), colour, width },
			DrawOp::Image { image, x, y, w, h }		=> DrawOp::Image { image, x: x + dx, y: y + dy, w, h },
		});
	}
	Ok(out)
}

/// Bakes a title or subtitle run to filled glyph outlines centred on `centre_x` at baseline `base_y`, in
/// the box frame (y down). Small caps are synthesised run by run as the banner sets them (was-lowercase
/// letters uppercased at 0.75 of the size); a plain run sets italic, matching the template's `emph`. The
/// advance is measured first so the run seats on its centre, then each glyph's y-up outline is flipped
/// onto the y-down baseline.
fn title_run_ops(
	ops:		&mut Vec<DrawOp>,
	fonts:		&Arc<FontSet>,
	text:		&str,
	size_pt:	f32,
	centre_x:	f32,
	base_y:		f32,
	smallcaps:	bool,
)
	-> Outcome<()>
{
	let size		= Sp::from_pt(size_pt as f64);
	let small_size	= Sp(size.raw() * 3 / 4);
	// Small caps sets upright (the template's `smallcaps`); a plain title sets italic (its `emph`).
	let face		= if smallcaps { HeadFace::Role(Role::Body) } else { HeadFace::Role(Role::Italic) };
	let runs		= if smallcaps { smallcaps_runs(text) } else { vec![(text.to_string(), false)] };

	// Total advance, so the run seats centred on `centre_x`.
	let mut total = 0.0f32;
	for (run, is_small) in &runs {
		let rs		= if *is_small { small_size } else { size };
		let shaped	= res!(head_shape(fonts, &face, rs, run));
		total += shaped.dims().width.to_pt() as f32;
	}

	let mut x = centre_x - total / 2.0;
	for (run, is_small) in &runs {
		let rs		= if *is_small { small_size } else { size };
		let shaped	= res!(head_shape(fonts, &face, rs, run));
		for glyph in &shaped.run().glyphs {
			let outline = res!(shaped.outline(glyph));
			if outline.is_empty() {
				continue;	// a space carries an advance but no ink
			}
			let t = Transform::scale(1.0, -1.0)
				.then(&Transform::translate(x + glyph.x, base_y - glyph.y));
			ops.push(DrawOp::Fill { path: res!(outline.transform(&t)), colour: Rgba::BLACK });
		}
		x += shaped.dims().width.to_pt() as f32;
	}
	Ok(())
}

/// Sets the imprint (meta) page: the publisher, edition, copyright, rights, AI declaration, website and
/// toolchain lines, set small in the lower half of the page as the template bottom-aligns them.
fn fm_meta_page(
	nodes:	&mut Vec<Node>,
	fonts:	&Arc<FontSet>,
	geom:	PageGeometry,
	style:	Style,
	fm:		&FrontMatter,
)
	-> Outcome<()>
{
	let measure	= geom.content_width();
	let h		= geom.content_height();
	let size	= Sp(style.body_size.raw() * 4 / 5);	// the template's 0.8em imprint

	// Drop to the lower part of the page; the template bottom-aligns, approximated here by a top spacer.
	nodes.push(fm_spacer(Sp(h.raw() * 48 / 100)));

	let mut lines: Vec<String> = Vec::new();
	if let Some(p) = &fm.publisher		{ lines.push(p.clone()); }
	if let Some(e) = &fm.edition		{ lines.push(e.clone()); }
	if let Some(i) = &fm.isbn			{ lines.push(fmt!("ISBN {}", i)); }
	if let Some(c) = &fm.copyright		{ lines.push(c.clone()); }
	if let Some(r) = &fm.rights			{ lines.push(r.clone()); }
	if let Some(a) = &fm.ai_declaration	{ lines.push(a.clone()); }
	if let Some(w) = &fm.website		{ lines.push(w.clone()); }
	if fm.toolchain {
		lines.push("Created using Austenite (built using Rust) and Inkscape.".to_string());
	}

	let mut first = true;
	for line in &lines {
		if !first {
			nodes.push(Node::Glue(Glue::fixed(style.para_skip)));
		}
		first = false;
		let broken = res!(break_paragraph(fonts.clone(), Role::Body, Dir::Ltr, size, line, measure, Sp(size.raw() * 6 / 5)));
		nodes.extend(broken);
	}
	Ok(())
}

/// Sets the documentation template's meta/colophon page (`template.typ`'s `meta-page`): a bordered
/// Ver/Date/Author(s)/Notes table at the top carrying the one revision row -- its version, date, author
/// with the "Made with AI" declaration mark beneath the name, and its notes with the reading time
/// appended -- then, seated at the page foot, the acknowledgement paragraph, the copyright line, the
/// "created using" line and the footer logo. The template `place`s the foot block against the page
/// bottom; here the foot block is measured and a rigid spacer drops it there, so the whole page sets as
/// one flow without a second leaf. The footer logo is drawn into the page here rather than by `decorate`,
/// which seats the folio footer on body pages only and leaves the front matter clean.
fn fm_doc_meta_page(
	nodes:	&mut Vec<Node>,
	fonts:	&Arc<FontSet>,
	geom:	PageGeometry,
	style:	Style,
	fm:		&FrontMatter,
)
	-> Outcome<()>
{
	let measure	= geom.content_width();
	let h		= geom.content_height();

	// The version table, at the very top of the content box, exactly as the template sets it flush under
	// the top margin.
	let table	= res!(build_meta_table(fm));
	let refs:	HashMap<String, String>	= HashMap::new();
	let tnode	= res!(table::lower(fonts.clone(), style, measure, &table, &refs));
	let table_h	= node_vext(&tnode);
	nodes.push(tnode);

	// The foot block: the acknowledgement, the copyright line, the toolchain line and the footer logo,
	// built into a buffer so its height is known and a spacer can drop it to the page foot. The gaps
	// between the four elements approximate the template's `place(bottom, dy: ..)` offsets.
	let mut foot:	Vec<Node>	= Vec::new();
	let mut foot_h				= Sp::ZERO;
	let gap						= Sp(style.body_size.raw() * 3 / 4);

	if let Some(ack) = &fm.acknowledgement {
		let size	= Sp(style.body_size.raw() * 85 / 100);
		let broken	= res!(break_paragraph(fonts.clone(), Role::Body, Dir::Ltr, size, ack, measure, Sp(size.raw() * 6 / 5)));
		for n in &broken { foot_h += node_vext(n); }
		foot.extend(broken);
	}
	if let Some(cr) = &fm.copyright {
		foot.push(Node::Glue(Glue::fixed(gap)));
		foot_h += gap;
		let size	= style.body_size;
		let broken	= res!(break_paragraph(fonts.clone(), Role::Body, Dir::Ltr, size, cr, measure, Sp(size.raw() * 6 / 5)));
		for n in &broken { foot_h += node_vext(n); }
		foot.extend(broken);
	}
	// The toolchain line, the template's fixed "created using" credit for the doc idiom.
	{
		foot.push(Node::Glue(Glue::fixed(gap)));
		foot_h += gap;
		let size	= Sp(style.body_size.raw() * 3 / 4);
		let line	= "This document was created using Austenite (built using Rust).";
		let broken	= res!(break_paragraph(fonts.clone(), Role::Body, Dir::Ltr, size, line, measure, Sp(size.raw() * 6 / 5)));
		for n in &broken { foot_h += node_vext(n); }
		foot.extend(broken);
	}
	if let Some(path) = &fm.footer_logo {
		if let Ok(graphic) = image_at_height(fonts, path, 18.0) {
			let logo = Leaf::graphic(graphic);
			let lh	 = logo.dims.height + logo.dims.depth;
			let big	 = Sp(style.body_size.raw() * 3 / 2);	// a little more air above the logo
			foot.push(Node::Glue(Glue::fixed(big)));
			foot_h += big + lh;
			foot.push(Node::HBox(BoxNode::new(vec![Node::Leaf(logo)], Dims::new(measure, lh, Sp::ZERO))));
		}
	}

	// Drop the foot block to the page bottom: a rigid spacer taking up the slack between the table and the
	// foot. A page too short for both simply sets them adjacent rather than overflowing to a second leaf.
	let used = table_h + foot_h;
	if h > used {
		nodes.push(fm_spacer(h - used));
	}
	nodes.extend(foot);
	Ok(())
}

/// Builds the meta page's Ver/Date/Author(s)/Notes table from the doc's revision rows. A column every row
/// leaves blank is dropped (the template's `filled` test): Author and Notes always stand, Ver and Date
/// only when some row sets them. Each row's author cell carries the declaration mark stacked beneath the
/// name, and the last row's notes take the reading time appended -- matching the template's `meta-page`
/// table with its `2fr, 2fr, 4fr, 6fr` columns.
fn build_meta_table(fm: &FrontMatter) -> Outcome<Table> {
	let has_ver	= fm.meta_rows.iter().any(|r| r.version.as_deref().unwrap_or("") != "");
	let has_date	= fm.meta_rows.iter().any(|r| r.date.as_deref().unwrap_or("") != "");

	let mut weights:	Vec<f64>		= Vec::new();
	let mut header:		Vec<Cell>	= Vec::new();
	if has_ver {
		weights.push(2.0);
		header.push(Cell::rich(vec![Segment::strong("Ver")], Align::Centre));
	}
	if has_date {
		weights.push(2.0);
		header.push(Cell::rich(vec![Segment::strong("Date")], Align::Centre));
	}
	weights.push(4.0);
	header.push(Cell::rich(vec![Segment::strong("Author(s)")], Align::Left));
	weights.push(6.0);
	header.push(Cell::rich(vec![Segment::strong("Notes")], Align::Left));

	let mut rows = vec![Row::new(header)];
	let last = fm.meta_rows.len().saturating_sub(1);
	for (i, mr) in fm.meta_rows.iter().enumerate() {
		let mut cells: Vec<Cell> = Vec::new();
		if has_ver {
			cells.push(Cell::rich(vec![Segment::text(mr.version.clone().unwrap_or_default())], Align::Centre));
		}
		if has_date {
			cells.push(Cell::rich(vec![Segment::text(mr.date.clone().unwrap_or_default())], Align::Centre));
		}
		// The author cell carries the name and, where the row declares one, the AI mark beneath it.
		let author_cell = match (&mr.ai_mark_path, &mr.ai_mark_words) {
			(Some(path), Some(words)) => {
				let mark = crate::table::CellMark {
					path:	path.clone(),
					height:	Sp::from_pt(36.0),	// the template's `image(.., height: 36pt)`
					words:	words.clone(),
					url:	mr.ai_mark_url.clone(),
				};
				Cell::rich_with_mark(vec![Segment::text(mr.authors.clone())], Align::Left, mark)
			},
			_ => Cell::rich(vec![Segment::text(mr.authors.clone())], Align::Left),
		};
		cells.push(author_cell);
		// The reading time is appended to the last row's notes only, as the template does.
		let notes = mr.notes.clone().unwrap_or_default();
		let notes = match (i == last, fm.reading_min) {
			(true, Some(m)) => if notes.is_empty() {
				fmt!("Reading time: {} [min]", m)
			} else {
				fmt!("{} Reading time: {} [min]", notes, m)
			},
			_ => notes,
		};
		cells.push(Cell::rich(vec![Segment::text(notes)], Align::Left));
		rows.push(Row::new(cells));
	}

	Ok(Table::with_weights(true, rows, weights))
}

/// Counts the words in a block stream, matching the template's reading-time counter, which steps once per
/// maximal run of letters (`\p{L}+`) as the body renders. Every text-bearing block contributes -- prose,
/// headings, list items, table cells, figure captions, code and references -- so the tally tracks Typst's
/// own `words.final()` closely; the reading time is that count over the average reading speed.
pub(crate) fn count_words(blocks: &[Block]) -> usize {
	fn count_str(s: &str, n: &mut usize) {
		let mut in_word = false;
		for ch in s.chars() {
			if ch.is_alphabetic() {
				if !in_word { *n += 1; in_word = true; }
			} else {
				in_word = false;
			}
		}
	}
	fn count_segs(segs: &[Segment], n: &mut usize) {
		for seg in segs {
			match seg {
				Segment::Text(t) | Segment::Strong(t) | Segment::Emph(t) | Segment::BoldItalic(t)
				| Segment::Super(t) | Segment::Code(t)	=> count_str(t, n),
				Segment::Glossary { display, .. }		=> count_str(display, n),
				Segment::Footnote { note }				=> count_segs(note, n),
				Segment::Cite(keys)						=> for k in keys { count_str(k, n); },
				Segment::PageRef(_) | Segment::Math(_)	=> {},
			}
		}
	}
	fn count_cells(table: &Table, n: &mut usize) {
		for row in &table.rows {
			for cell in &row.cells {
				count_segs(&cell.content, n);
			}
		}
	}
	let mut n = 0usize;
	for b in blocks {
		match b {
			Block::Heading { segments, .. }		=> count_segs(segments, &mut n),
			Block::Paragraph { text }			=> count_str(text, &mut n),
			Block::RichParagraph { segments }	=> count_segs(segments, &mut n),
			Block::List { items, .. }			=> for it in items { count_segs(it, &mut n); },
			Block::Code { lines }				=> for l in lines { count_str(l, &mut n); },
			Block::Table(t)						=> count_cells(t, &mut n),
			Block::Figure { caption, .. }		=> if let Some(c) = caption { count_str(c, &mut n); },
			Block::TableFigure { table, caption, .. } => {
				count_cells(table, &mut n);
				if let Some(c) = caption { count_segs(c, &mut n); }
			},
			Block::ImageFigure { caption, .. } | Block::CodeFigure { caption, .. }
												=> if let Some(c) = caption { count_segs(c, &mut n); },
			Block::BackMatterHeading { title }	=> count_str(title, &mut n),
			Block::Reference { runs }			=> for (t, _) in runs { count_str(t, &mut n); },
			Block::Equation { .. } | Block::Rule { .. } | Block::Image { .. }
			| Block::SectionBanner { .. }		=> {},
		}
	}
	n
}

/// The vertical extent a node occupies in a flow: a box's height plus depth, a glue's natural size, a
/// leaf's height plus depth. Anchors and penalties take no space.
fn node_vext(n: &Node) -> Sp {
	match n {
		Node::HBox(b) | Node::VBox(b)	=> b.dims.height + b.dims.depth,
		Node::Leaf(l)					=> l.dims.height + l.dims.depth,
		Node::Glue(g)					=> g.natural,
		_								=> Sp::ZERO,
	}
}

/// Sets the dedication page: the dedication centred, in italic, about the vertical centre.
fn fm_dedication_page(
	nodes:	&mut Vec<Node>,
	fonts:	&Arc<FontSet>,
	geom:	PageGeometry,
	style:	Style,
	text:	&str,
)
	-> Outcome<()>
{
	let measure	= geom.content_width();
	let h		= geom.content_height();
	nodes.push(fm_spacer(Sp(h.raw() * 40 / 100)));
	let size	= Sp(style.body_size.raw() * 11 / 10);
	res!(fm_centred_wrap(nodes, fonts, Role::Italic, size, text, measure, Sp(size.raw() * 6 / 5)));
	Ok(())
}

/// Sets the "About the Author" page: the title in the display face, then the biography justified below.
fn fm_about_author_page(
	nodes:		&mut Vec<Node>,
	fonts:		&Arc<FontSet>,
	display:	Option<&Arc<Font>>,
	geom:		PageGeometry,
	style:		Style,
	title_size:	Sp,
	bio:		&str,
)
	-> Outcome<()>
{
	let measure	= geom.content_width();
	nodes.push(fm_spacer(Sp::from_pt(24.0)));
	let title = res!(head_shape(fonts, &head_face(1, display), title_size, "About the Author"));
	let td = title.dims();
	nodes.push(Node::HBox(BoxNode::new(vec![Node::Leaf(Leaf::text(title))], td)));
	nodes.push(Node::Glue(Glue::fixed(Sp::from_pt(18.0))));
	let size	= Sp(style.body_size.raw() * 9 / 10);
	let broken	= res!(break_paragraph(fonts.clone(), Role::Body, Dir::Ltr, size, bio, measure, Sp(size.raw() * 7 / 5)));
	nodes.extend(broken);
	Ok(())
}

/// The deepest heading level the contents lists, matching the template's `outline(depth: 3)`.
const TOC_DEPTH: u8 = 3;

/// Sets a table of contents from the heading table: the "Contents" title in the display face, then one
/// entry per heading -- its number in a column indented by level, its title, a dotted leader, and its
/// printed folio flush at the right. The folio is a forward reference resolved with [`Ref::FolioOf`]
/// against the incoming ledger, so it reads the body folio (which restarts at one) rather than the
/// physical page, reusing the same reserve-then-resolve slot the driver runs for any forward reference.
/// The caller prepends these nodes after the front matter; a trailing forced break opens the body.
///
/// A fact a reader could not derive. Each entry reserves a fixed slot for its folio -- three digits
/// wide, so a resolved number never outgrows it -- and its line height is the entry's, whatever the
/// folio turns out to be, and the dotted leader takes the width left over. The contents block therefore
/// has a constant vertical extent from the first pass, so the body it displaces settles once and the
/// forward references converge in the usual two passes, with no special case in the driver.
pub fn contents(
	fonts:		Arc<FontSet>,
	display:	Option<&Arc<Font>>,
	geom:		PageGeometry,
	style:		Style,
	title_size:	Sp,
	heads:		&[Heading],
)
	-> Outcome<Vec<Node>>
{
	let measure			= geom.content_width();
	let mut nodes:	Vec<Node> = Vec::new();

	// A `Label` anchor at the top of the contents leaf records its page for the PDF outline. It sets no
	// heading, so the contents is neither a running-head section nor an entry in its own list.
	nodes.push(Node::Anchor(AnchorId::new(AnchorKind::Label, "frontmatter:contents")));

	// The block's own heading, in the display face at the back-matter title size, recorded as no anchor --
	// so it is neither a running-head section nor an entry in its own list.
	let title	= res!(head_shape(&fonts, &head_face(1, display), title_size, "Contents"));
	let td		= title.dims();
	nodes.push(Node::HBox(BoxNode::new(vec![Node::Leaf(Leaf::text(title))], td)));
	nodes.push(Node::Glue(Glue::fixed(style.space_below(1))));

	// A fixed slot wide enough for a three-digit folio, so a resolved number never overflows its
	// reservation and every entry keeps a constant height across passes.
	let slot	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, style.body_size, "000"));
	let slot_w	= slot.dims().width;
	// A dot-and-space leader unit, measured once, so a leader is filled with a whole number of dots.
	let dot		= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, style.body_size, ". "));
	let dot_w	= dot.dims().width.raw().max(1);
	// The step a level indents the number column by, and the gap between a number and its title.
	let step	= Sp(style.body_size.raw() * 3 / 2);
	let gap		= Sp(style.body_size.raw() * 3 / 5);

	for (i, h) in heads.iter().enumerate() {
		// The template sets `outline(depth: 3)`, so the contents stops at level 3 (a `===` subsection,
		// dotted number x.y.z); a level-4 `====` heading is listed in no contents and is skipped here.
		if h.level > TOC_DEPTH {
			continue;
		}
		// The number column is indented per level: a part (level 0) and a chapter (level 1) sit at the
		// margin, deeper levels step right. The number is empty for a part, which then shows title alone.
		let depth	= (h.level.max(1) - 1) as i32;
		let indent	= step * depth;
		let numw	= if h.number.is_empty() {
			Sp::ZERO
		} else {
			let n = res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, style.body_size, &h.number));
			n.dims().width
		};
		// The entry's title set from its rich runs, so a maths span or emphasis in a heading renders here
		// rather than dropping to a gap. The height is the sample's, constant across passes.
		let (entry, ed)	= res!(inline_segments(&fonts, style, &h.segments, Role::Body, style.body_size));

		// The leader span from the title's end to the folio slot; a title too wide to leave a one-em
		// minimum keeps that minimum and runs under its folio -- the over-wide case, left as it falls.
		let num_col	= if numw.raw() > 0 { numw + gap } else { Sp::ZERO };
		let taken	= indent + num_col + ed.width + slot_w;
		let min_lead	= style.body_size;
		let leader_w	= if measure > taken + min_lead { measure - taken } else { min_lead };

		// Fill the leader with a whole number of dots, padded to the folio slot on the right so the slot's
		// right edge falls on the measure.
		let lead_margin	= Sp(style.body_size.raw() / 2);
		let usable		= (leader_w.raw() - lead_margin.raw()).max(0);
		let n_dots		= (usable / dot_w).max(0) as usize;
		let dots		= res!(ShapedText::new(
			fonts.clone(), Role::Body, Dir::Ltr, style.body_size, &". ".repeat(n_dots)));
		let dots_w		= dots.dims().width;
		let trailing	= if leader_w > lead_margin + dots_w { leader_w - lead_margin - dots_w } else { Sp::ZERO };

		// The entry's own identity, distinct from the heading it points at, so recording the slot never
		// overwrites the heading's ledger row. Its reference resolves the heading's folio.
		let toc_id		= AnchorId::new(AnchorKind::Label, fmt!("toc-{}", h.id.key));
		let slot_dims	= Dims::new(slot_w, ed.height, ed.depth);

		let mut children:	Vec<Node> = Vec::new();
		if indent.raw() > 0 {
			children.push(Node::Glue(Glue::fixed(indent)));
		}
		if numw.raw() > 0 {
			children.push(Node::Leaf(Leaf::text(res!(
				ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, style.body_size, &h.number)))));
			children.push(Node::Glue(Glue::fixed(gap)));
		}
		children.extend(entry);
		children.push(Node::Glue(Glue::fixed(lead_margin)));
		children.push(Node::Leaf(Leaf::text(dots)));
		if trailing.raw() > 0 {
			children.push(Node::Glue(Glue::fixed(trailing)));
		}
		children.push(Node::Leaf(Leaf::reserved(toc_id, Ref::FolioOf(h.id.clone()), slot_dims)));

		let line_dims = Dims::new(measure, ed.height, ed.depth);
		nodes.push(Node::HBox(BoxNode::new(children, line_dims)));

		// Leading between entries, but not after the last.
		if i + 1 < heads.len() {
			let vextent	= ed.height + ed.depth;
			let lead	= if style.leading > vextent { style.leading - vextent } else { Sp::ZERO };
			nodes.push(Node::Glue(Glue::fixed(lead)));
		}
	}

	// The contents stands alone at the front; the body opens on a fresh page.
	nodes.push(Node::Penalty(Penalty::eject()));
	Ok(nodes)
}

/// Sets one bibliography reference: its runs woven into justified lines at the footnote size, with a
/// hanging indent -- the first line flush left, every continuation line indented, as a Chicago
/// reference list sets. The runs' italic flag chooses the face, so a book or journal title sets italic.
fn reference_block(
	nodes:		&mut Vec<Node>,
	fonts:		Arc<FontSet>,
	style:		Style,
	measure:	Sp,
	runs:		&[(String, bool)],
)
	-> Outcome<()>
{
	let hang	= Sp(style.body_size.raw() * 3 / 2);	// the 1.5 em hang the continuation lines take
	let inner	= if measure > hang { measure - hang } else { measure };

	let mut pieces: Vec<Piece> = Vec::with_capacity(runs.len());
	for (text, italic) in runs {
		let role = if *italic { Role::Italic } else { Role::Body };
		pieces.push(Piece::Text { text: text.clone(), role });
	}
	let mut lines = res!(break_paragraph_pieces(
		fonts.clone(), Role::Body, Dir::Ltr, style.foot_size, &pieces, inner, style.foot_leading, true));

	// Indent every line but the first by the hang, so the entry hangs under its first line.
	let mut first = true;
	for line in lines.iter_mut() {
		if let Node::HBox(b) = line {
			if first {
				first = false;
			} else {
				b.list.insert(0, Node::Glue(Glue::fixed(hang)));
				b.dims = Dims::new(b.dims.width + hang, b.dims.height, b.dims.depth);
			}
		}
	}
	nodes.extend(lines);
	Ok(())
}

/// Wraps a vertical run of nodes as a keep box, its extent the sum of its children's, so the driver
/// places it whole or moves it whole. The whole extent is carried as height; a block has no baseline
/// the page cares about, so the depth is zero.
fn vbox(list: Vec<Node>, width: Sp) -> Node {
	let mut ext = Sp::ZERO;
	for n in &list {
		ext += n.vextent();
	}
	Node::VBox(BoxNode::new(list, Dims::new(width, ext, Sp::ZERO)))
}

/// The plain display words of a heading's rich runs: the text a reader sees with the markup removed, so
/// the anchor slug, the table-of-contents entry and the running head read the rendered title rather than
/// its raw source. A glossary term contributes its display, emphasis and code their inner words; a maths
/// span, a cross-reference, a footnote and a citation have no plain form here and contribute nothing.
fn flatten_segments(segments: &[Segment]) -> String {
	let mut out = String::new();
	for seg in segments {
		match seg {
			Segment::Text(t)				=> out.push_str(t),
			Segment::Strong(t)				=> out.push_str(t),
			Segment::Emph(t)				=> out.push_str(t),
			Segment::BoldItalic(t)			=> out.push_str(t),
			Segment::Super(t)				=> out.push_str(t),
			Segment::Code(t)				=> out.push_str(t),
			Segment::Glossary { display, .. }	=> out.push_str(display),
			Segment::Math(_)				=> {},
			Segment::PageRef(_)				=> {},
			Segment::Footnote { .. }		=> {},
			Segment::Cite(_)				=> {},
		}
	}
	out
}

/// Sets a title's rich runs into one horizontal line at `size` in `role`, so a running head or a
/// table-of-contents entry renders the title's maths, emphasis and glossary term rather than flattening
/// them to plain words or dropping the maths to a gap. Every run seats on a common baseline taken from a
/// full-size sample; a maths span is set at `size` and its glyphs woven into the line as the body sets
/// inline maths. A cross-reference, footnote or citation in a title has no form here and is dropped. The
/// returned dims carry the line's total width and the sample's ascent and depth, so a caller lays it out
/// with a constant height whatever the runs turn out to be.
fn inline_segments(
	fonts:		&Arc<FontSet>,
	style:		Style,
	segments:	&[Segment],
	role:		Role,
	size:		Sp,
)
	-> Outcome<(Vec<Node>, Dims)>
{
	let sample	= res!(ShapedText::new(fonts.clone(), role, Dir::Ltr, size, "Ag"));
	let asc		= sample.dims().height;
	let dep		= sample.dims().depth;
	let italic	= role == Role::Italic;

	let mut children:	Vec<Node> = Vec::new();
	let mut width		= Sp::ZERO;
	for seg in segments {
		// A maths span is unwrapped and its leaves woven straight into the line; every other run resolves to
		// a text string set in a face chosen against the base role, so an emphasis in an italic running head
		// toggles upright as Typst sets it.
		let (text, r): (&str, Role) = match seg {
			Segment::Text(t)		=> (t, role),
			Segment::Strong(t)		=> (t, if italic { Role::BoldItalic } else { Role::Bold }),
			Segment::Emph(t)		=> (t, if italic { Role::Body } else { Role::Italic }),
			Segment::BoldItalic(t)	=> (t, if italic { Role::Bold } else { Role::BoldItalic }),
			Segment::Super(t)		=> (t, role),
			Segment::Code(t)		=> (t, Role::Mono),
			Segment::Glossary { display, .. }	=> (display, role),
			Segment::Math(atom)	=> {
				let mut hs = style;
				hs.body_size = size;
				if let Node::HBox(b) = res!(math::layout(fonts.clone(), &hs, atom, false)) {
					width += b.dims.width;
					children.extend(b.list);
				}
				continue;
			},
			Segment::PageRef(_) | Segment::Footnote { .. } | Segment::Cite(_)	=> continue,
		};
		let sh	= res!(ShapedText::new(fonts.clone(), r, Dir::Ltr, size, text));
		let w	= sh.dims().width;
		children.push(Node::Leaf(Leaf::text_dims(sh, Dims::new(w, asc, dep))));
		width += w;
	}
	Ok((children, Dims::new(width, asc, dep)))
}

/// Places a horizontal run of leaves (a rich running head from [`inline_segments`]) into a page frame,
/// starting at `x0` with `top` the run's box top. It mirrors the driver's own line placement: a text or
/// graphic leaf lands at the running x plus its own shift, glue advances the cursor, and a reserved or
/// rule leaf -- neither of which a heading run holds -- is skipped.
fn place_run(frame: &mut Frame, nodes: &[Node], x0: Sp, top: Sp) {
	let mut x = x0;
	for n in nodes {
		match n {
			Node::Leaf(l) => {
				let y = top + l.shift;
				match &l.kind {
					LeafKind::Text(sh)		=> frame.push(Placed::new(x, y, l.dims, PlacedKind::Text(sh.clone()))),
					LeafKind::Graphic(g)	=> frame.push(Placed::new(x, y, l.dims, PlacedKind::Graphic(g.clone()))),
					_						=> {},
				}
				x += l.dims.width;
			},
			Node::Glue(g)	=> x += g.natural,
			_				=> {},
		}
	}
}

/// A filesystem-safe key from a heading's words: lowercase, runs of non-alphanumerics collapsed to a
/// single dash. Prefixed with an ordinal by the caller, so two headings of the same words stay
/// distinct identities.
fn slug(text: &str) -> String {
	let mut out		= String::new();
	let mut dash	= false;
	for c in text.chars() {
		if c.is_ascii_alphanumeric() {
			out.push(c.to_ascii_lowercase());
			dash = false;
		} else if !dash && !out.is_empty() {
			out.push('-');
			dash = true;
		}
	}
	while out.ends_with('-') {
		out.pop();
	}
	if out.is_empty() { "heading".to_string() } else { out }
}

// ┌───────────────────────────────────────────────────────────────────────────┐
// │ HEADINGS                                                                   │
// └───────────────────────────────────────────────────────────────────────────┘

/// The number shown before a heading: the chapter number alone for a chapter (level 1), the dotted path
/// for a deeper level (`2.3.1`), and nothing for a part divider (level 0).
fn heading_number(level: u8, sec: &[u32; 6]) -> String {
	match level {
		0 => String::new(),
		1 => fmt!("{}", sec[0]),
		_ => {
			let l = (level as usize).min(6);
			let parts: Vec<String> = sec[..l].iter().map(|n| fmt!("{}", n)).collect();
			parts.join(".")
		},
	}
}

/// A heading run's face: the display face (Radley) a book supplies for its chapters and level-2
/// sections, or a reading-set role for the finer levels.
#[derive(Clone, Copy)]
enum HeadFace<'a> {
	Solo(&'a Arc<Font>),
	Role(Role),
}

/// The face a heading level sets in. Levels 0-2 take the display face when the book supplies one, else
/// the body bold; level 3 is Libertinus italic and level 4+ Libertinus upright -- the template's
/// `if it.level <= 2 { "Radley" } else { "Libertinus Serif" }` with its level-3 italic.
fn head_face(level: u8, display: Option<&Arc<Font>>) -> HeadFace<'_> {
	match display {
		Some(f) if level <= 2	=> HeadFace::Solo(f),
		_ if level == 3			=> HeadFace::Role(Role::Italic),
		_ if level <= 2			=> HeadFace::Role(Role::Bold),	// no display face: the body bold stands in
		_						=> HeadFace::Role(Role::Body),
	}
}

/// Shapes one heading run in its face.
fn head_shape(
	fonts:	&Arc<FontSet>,
	face:	&HeadFace,
	size:	Sp,
	text:	&str,
)
	-> Outcome<ShapedText>
{
	match face {
		HeadFace::Solo(f)	=> ShapedText::new_with_font((*f).clone(), Dir::Ltr, size, text),
		HeadFace::Role(r)	=> ShapedText::new(fonts.clone(), *r, Dir::Ltr, size, text),
	}
}

/// Splits a title into runs for synthetic small caps: a run of originally-lowercase letters, uppercased
/// and to be set at the small size, alternates with runs of everything else (capitals, digits, spaces,
/// punctuation) kept at the full size. The bool is true for the small (was-lowercase) runs. Synthetic
/// because the shaper applies no OpenType `smcp`; used only where the template's face (Libertinus, levels
/// 3-4) really carries small caps -- Radley does not, so the level-1/2 titles keep their case.
fn smallcaps_runs(text: &str) -> Vec<(String, bool)> {
	let mut runs:	Vec<(String, bool)> = Vec::new();
	let mut cur		= String::new();
	let mut small	= false;
	for ch in text.chars() {
		let is_small = ch.is_lowercase();
		if !cur.is_empty() && is_small != small {
			runs.push((std::mem::take(&mut cur), small));
		}
		small = is_small;
		if is_small {
			for u in ch.to_uppercase() { cur.push(u); }
		} else {
			cur.push(ch);
		}
	}
	if !cur.is_empty() {
		runs.push((cur, small));
	}
	runs
}

/// Builds a sub-heading line (levels 2-4): the number in the heading face, a thin gap, then the title
/// set from its rich runs, small-capped from level 3 down. Runs of differing size seat on one baseline
/// by taking a common ascent and depth from a full-size sample, so the small caps and the full caps sit
/// level. A glossary term keeps its own first-use bold-italic (recorded in `seen`, shared with the body
/// so document order decides), emphasis its face, and a maths span is set at the heading size and its
/// glyphs woven into the line -- so a call in a heading renders rather than leaking its raw source.
fn subheading_hbox(
	fonts:		Arc<FontSet>,
	display:	Option<&Arc<Font>>,
	style:		Style,
	level:		u8,
	number:		&str,
	segments:	&[Segment],
	seen:		&mut HashSet<String>,
)
	-> Outcome<Node>
{
	// A documentation tree sets its sub-headings from the template's show rule -- an inline level-1 heading
	// (a `DocInline` tree) bold small-caps, level 2 bold-italic, level 3 italic, deeper levels upright; a
	// book takes the display face (or body bold) and small-caps the finer levels.
	let doc			= matches!(style.heading_style, HeadingStyle::DocBanner | HeadingStyle::DocInline);
	let face		= if doc { doc_head_face(level) } else { head_face(level, display) };
	let size		= style.heading_size(level);
	let small_size	= Sp(size.raw() * 3 / 4);	// small caps at 0.75 of the heading size
	let smallcaps	= if doc { level == 1 } else { level >= 3 };
	let sample		= res!(head_shape(&fonts, &face, size, "Ag"));
	let asc			= sample.dims().height;
	let dep			= sample.dims().depth;

	let mut children:	Vec<Node> = Vec::new();
	let mut width		= Sp::ZERO;

	if !number.is_empty() {
		let sh	= res!(head_shape(&fonts, &face, size, number));
		let w	= sh.dims().width;
		children.push(Node::Leaf(Leaf::text_dims(sh, Dims::new(w, asc, dep))));
		width += w;
		let gap = Sp(size.raw() / 5);	// ~0.2 em, the template's `h(0.2em)`
		children.push(Node::Glue(Glue::fixed(gap)));
		width += gap;
	}

	for seg in segments {
		match seg {
			Segment::Text(t)	=> res!(push_head_text(
				&mut children, &mut width, &fonts, &face, size, small_size, smallcaps, t, asc, dep)),
			Segment::Strong(t)	=> res!(push_head_text(
				&mut children, &mut width, &fonts, &head_run_face(&face, HeadRun::Strong), size, small_size, smallcaps, t, asc, dep)),
			Segment::Emph(t)	=> res!(push_head_text(
				&mut children, &mut width, &fonts, &head_run_face(&face, HeadRun::Emph), size, small_size, smallcaps, t, asc, dep)),
			Segment::BoldItalic(t)	=> res!(push_head_text(
				&mut children, &mut width, &fonts, &head_run_face(&face, HeadRun::BoldItalic), size, small_size, smallcaps, t, asc, dep)),
			// A superscript in a heading is vanishingly rare; set its text in the heading face rather than
			// raising it, so the words are kept without a scripted run in display type.
			Segment::Super(t)	=> res!(push_head_text(
				&mut children, &mut width, &fonts, &face, size, small_size, smallcaps, t, asc, dep)),
			Segment::Code(t)	=> res!(push_head_text(
				&mut children, &mut width, &fonts, &face, size, small_size, smallcaps, t, asc, dep)),
			Segment::Glossary { term, display: disp }	=> {
				// First use is set bold-italic, matching the template's `*_term_*`; a later use takes the
				// heading's own face. The set is the body's, so a term first seen in a heading is plain in
				// the prose after it, exactly as document order dictates.
				let f = if seen.insert(term.clone()) { head_run_face(&face, HeadRun::Gloss) } else { face };
				res!(push_head_text(
					&mut children, &mut width, &fonts, &f, size, small_size, smallcaps, disp, asc, dep));
			},
			Segment::Math(atom)	=> {
				// The span is set at the heading size and unwrapped, its leaves woven into the line as the
				// body sets inline maths, so a subscripted variable in a heading draws as real glyphs.
				let mut hs = style;
				hs.body_size = size;
				if let Node::HBox(b) = res!(math::layout(fonts.clone(), &hs, atom, false)) {
					width += b.dims.width;
					children.extend(b.list);
				}
			},
			// A footnote, cross-reference or citation in a heading is vanishingly rare and has no display
			// form here; it is dropped rather than set, leaving the heading its words.
			Segment::Footnote { .. }	=> {},
			Segment::PageRef(_)			=> {},
			Segment::Cite(_)			=> {},
		}
	}

	Ok(Node::HBox(BoxNode::new(children, Dims::new(width, asc, dep))))
}

/// The face a documentation heading level sets in, matching `template.typ`'s heading show rule: an inline
/// level-1 heading bold (and small-capped by its caller), level 2 bold-italic, level 3 italic, level 4 and
/// deeper upright. All in the body family (Libertinus), which is the doc heading family too, so no display
/// face is consulted.
fn doc_head_face(level: u8) -> HeadFace<'static> {
	match level {
		1	=> HeadFace::Role(Role::Bold),
		2	=> HeadFace::Role(Role::BoldItalic),
		3	=> HeadFace::Role(Role::Italic),
		_	=> HeadFace::Role(Role::Body),
	}
}

/// A heading run marked for emphasis: strong (`*..*`), emph (`_.._`), or a glossary term's first-use
/// bold-italic.
enum HeadRun {
	Strong,
	Emph,
	BoldItalic,
	Gloss,
}

/// The face one emphasised heading run sets in. A display face (Radley, levels 1-2) has no role variants
/// loaded, so every run keeps it; a role face (levels 3-4) takes the run's own role, an emphasis inside
/// an italic heading toggling upright as Typst sets it, a strong one going bold-italic.
fn head_run_face<'a>(base: &HeadFace<'a>, run: HeadRun) -> HeadFace<'a> {
	match base {
		HeadFace::Solo(f)	=> HeadFace::Solo(f),
		HeadFace::Role(r)	=> {
			let italic = *r == Role::Italic;
			let role = match run {
				HeadRun::Strong		=> if italic { Role::BoldItalic } else { Role::Bold },
				HeadRun::BoldItalic	=> if italic { Role::Bold } else { Role::BoldItalic },	// nested emphasis toggles against an italic heading
				HeadRun::Gloss		=> Role::BoldItalic,
				HeadRun::Emph		=> if italic { Role::Body } else { Role::Italic },	// emph toggles against an italic heading
			};
			HeadFace::Role(role)
		},
	}
}

/// Sets one heading text run into the line, small-capping it (levels 3-4) run by run so a was-lowercase
/// stretch sets uppercase at the small size while capitals keep the full size, both seated on the common
/// baseline. A run with no small caps sets whole at the full size.
#[allow(clippy::too_many_arguments)]
fn push_head_text(
	children:	&mut Vec<Node>,
	width:		&mut Sp,
	fonts:		&Arc<FontSet>,
	face:		&HeadFace,
	size:		Sp,
	small_size:	Sp,
	smallcaps:	bool,
	text:		&str,
	asc:		Sp,
	dep:		Sp,
)
	-> Outcome<()>
{
	if smallcaps {
		for (run, is_small) in smallcaps_runs(text) {
			let rs	= if is_small { small_size } else { size };
			let sh	= res!(head_shape(fonts, face, rs, &run));
			let w	= sh.dims().width;
			children.push(Node::Leaf(Leaf::text_dims(sh, Dims::new(w, asc, dep))));
			*width += w;
		}
	} else {
		let sh	= res!(head_shape(fonts, face, size, text));
		let w	= sh.dims().width;
		children.push(Node::Leaf(Leaf::text_dims(sh, Dims::new(w, asc, dep))));
		*width += w;
	}
	Ok(())
}

/// Renders a shaped run as a coloured graphic: each glyph outline filled in `colour`, so a heading can
/// take a fill the text emitter (which draws every run black) does not carry. The outline is font-frame,
/// y up; it is flipped and seated on the run's baseline, `height` below the box top.
fn coloured_run(shaped: &ShapedText, colour: Rgba) -> Outcome<Graphic> {
	let base_y = shaped.dims().height.to_pt() as f32;
	let mut ops = Vec::new();
	for glyph in &shaped.run().glyphs {
		let path = res!(shaped.outline(glyph));
		if path.is_empty() {
			continue;
		}
		let t = Transform::scale(1.0, -1.0)
			.then(&Transform::translate(glyph.x, base_y - glyph.y));
		ops.push(DrawOp::Fill { path: res!(path.transform(&t)), colour });
	}
	Ok(Graphic::new(ops, shaped.dims()))
}

/// Pushes a shaped run centred within `measure` on its own line.
fn push_centred_shape(nodes: &mut Vec<Node>, sh: ShapedText, measure: Sp) -> Outcome<()> {
	let d	= sh.dims();
	let pad	= if measure > d.width { Sp((measure.raw() - d.width.raw()) / 2) } else { Sp::ZERO };
	let mut row: Vec<Node> = Vec::new();
	if pad.raw() > 0 {
		row.push(Node::Glue(Glue::fixed(pad)));
	}
	row.push(Node::Leaf(Leaf::text(sh)));
	nodes.push(Node::HBox(BoxNode::new(row, Dims::new(measure, d.height, d.depth))));
	Ok(())
}

/// The upper-case Roman numeral for `n`, covering the part range a book uses.
fn roman(mut n: u32) -> String {
	let table = [
		(100u32, "C"), (90, "XC"), (50, "L"), (40, "XL"),
		(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
	];
	let mut out = String::new();
	for (v, s) in table {
		while n >= v {
			out.push_str(s);
			n -= v;
		}
	}
	out
}

/// Sets a chapter opener (level 1) or a part divider (level 0) on a fresh page. A chapter shows its
/// number as a giant grey display numeral centred near the page top, then its title beneath in the
/// display face at the chapter-title size. A part divider carries `part_label` ("Part I") in the
/// display face above its title, both centred and set about the vertical middle of the page, matching
/// the template's `align(center + horizon)` part page. The anchor (and any label) is recorded at the
/// opener, so a running head or a cross-reference finds its page.
#[allow(clippy::too_many_arguments)]
fn chapter_opener(
	nodes:		&mut Vec<Node>,
	fonts:		&Arc<FontSet>,
	display:	Option<&Arc<Font>>,
	style:		Style,
	geom:		PageGeometry,
	measure:	Sp,
	level:		u8,
	number:		&str,
	title:		&str,
	part_label:	&str,
	id:			&AnchorId,
	label:		Option<&str>,
)
	-> Outcome<()>
{
	nodes.push(Node::Anchor(id.clone()));
	if let Some(l) = label {
		nodes.push(Node::Anchor(AnchorId::new(AnchorKind::Label, l.to_string())));
	}

	let face = head_face(level, display);

	// A part divider fills its own page: a small display label, a gap, then the title, the block set
	// about the vertical centre. The label is upper-cased, the template's tracked small caps rendered as
	// plain caps here (the shaper carries no tracking); the gap is the template's `#v(2em)`, two body em.
	if level == 0 {
		let label_size	= Sp::from_pt(style.h1_size.to_pt() * 0.55);	// the template's part-label, ~13/24 of the title
		let lab			= res!(head_shape(fonts, &face, label_size, &part_label.to_uppercase()));
		let ttl			= res!(head_shape(fonts, &face, style.h1_size, title));
		let gap			= Sp::from_pt(style.body_size.to_pt() * 2.0);	// #v(2em)
		let lab_v		= lab.dims().height + lab.dims().depth;
		let ttl_v		= ttl.dims().height + ttl.dims().depth;
		let block_v		= lab_v + gap + ttl_v;
		let content_h	= geom.content_height();
		// Drop the block so its middle sits at the page's vertical centre; a box spacer, which a page top
		// keeps where glue would be discarded.
		if content_h > block_v {
			let top = Sp((content_h.raw() - block_v.raw()) / 2);
			nodes.push(Node::HBox(BoxNode::new(vec![], Dims::new(Sp::ZERO, top, Sp::ZERO))));
		}
		res!(push_centred_shape(nodes, lab, measure));
		nodes.push(Node::Glue(Glue::fixed(gap)));
		res!(push_centred_shape(nodes, ttl, measure));
		return Ok(());
	}

	// A documentation tree opens a level-1 heading with the template's full-width grey banner bar carrying
	// the title in small caps, rather than a numbered chapter opener.
	if level == 1 && style.heading_style == HeadingStyle::DocBanner {
		res!(doc_banner(nodes, fonts, geom, measure, title));
		return Ok(());
	}

	if level == 1 && !number.is_empty() {
		// The opener reproduces the template's four-row grid (`chapter-grid-rows`): a tall band holding the
		// number centred on its middle, a gap, a shorter band holding the title on its foot, and a gap down
		// to the body. Every row is a box, not glue -- a page top discards leading glue, and the opener sits
		// at the page top -- so the bands hold their heights and the body lands on the grid's foot.
		let sh		= res!(head_shape(fonts, &face, style.chap_num_size, number));
		let d		= sh.dims();
		let num_v	= d.height + d.depth;
		let band	= style.chap_grid[0];
		// The number rides the middle of its band (Typst's `center + horizon`): the slack splits above and
		// below. A band shorter than the number leaves no slack and the number simply fills it.
		let above	= if band > num_v { Sp((band.raw() - num_v.raw()) / 2) } else { Sp::ZERO };
		let below	= if band > num_v + above { band - num_v - above } else { Sp::ZERO };
		nodes.push(vspacer(above));

		let graphic	= res!(coloured_run(&sh, style.chap_num_grey));
		let pad		= if measure > d.width { Sp((measure.raw() - d.width.raw()) / 2) } else { Sp::ZERO };
		let mut row:	Vec<Node> = Vec::new();
		if pad.raw() > 0 {
			row.push(Node::Glue(Glue::fixed(pad)));
		}
		row.push(Node::Leaf(Leaf::graphic(graphic)));
		nodes.push(Node::HBox(BoxNode::new(row, Dims::new(measure, num_v, Sp::ZERO))));
		nodes.push(vspacer(below));
		nodes.push(vspacer(style.chap_grid[1]));	// the gap row between number and title

		// The title rides the foot of its band (Typst's `left + bottom`): all the slack sits above it.
		let sh_t	= res!(head_shape(fonts, &face, style.h1_size, title));
		let dt		= sh_t.dims();
		let title_v	= dt.height + dt.depth;
		let band2	= style.chap_grid[2];
		let top2	= if band2 > title_v { band2 - title_v } else { Sp::ZERO };
		nodes.push(vspacer(top2));
		nodes.push(Node::HBox(BoxNode::new(vec![Node::Leaf(Leaf::text(sh_t))], Dims::new(measure, dt.height, dt.depth))));
		nodes.push(vspacer(style.chap_grid[3]));	// the gap row down to the body
		return Ok(());
	}

	// An unnumbered level-1 opener (no grid number): the title set left in the display face, then a gap.
	let sh	= res!(head_shape(fonts, &face, style.h1_size, title));
	let d	= sh.dims();
	nodes.push(Node::HBox(BoxNode::new(vec![Node::Leaf(Leaf::text(sh))], Dims::new(measure, d.height, d.depth))));
	nodes.push(Node::Glue(Glue::fixed(Sp::from_pt(20.0))));
	Ok(())
}

/// Draws the documentation template's chapter banner: a full-width grey bar hanging into the page's top
/// and side margins (`template.typ`'s `chapter-banner`, a `place`d rect 150 pt tall from the page top,
/// `100% + 2*margin` wide), carrying the title left-aligned in small-caps bold, seated on the band's
/// vertical middle. The bar is drawn from one box whose ops bleed past its bounds -- the emitter clips
/// nothing -- and the box holds the template's `#v(95pt)` of following space, so the body lands where the
/// oracle sets it. No number is drawn: a doc tree sets `numbering: none`.
fn doc_banner(
	nodes:		&mut Vec<Node>,
	fonts:		&Arc<FontSet>,
	geom:		PageGeometry,
	measure:	Sp,
	title:		&str,
)
	-> Outcome<()>
{
	let grey		= Rgba::opaque(240, 240, 240);	// the template's `colours.lightgrey`, luma(240)
	let banner_h	= 150.0f32;						// the template's rect height
	let follow		= Sp::from_pt(95.0);			// the template's `#v(95pt)` down to the body
	// The box origin is the content top-left; the graphic's ops are in that frame, y down. The bar reaches
	// the page's left edge (x = -inside) and top edge (y = -top), and runs the full page width and 150 pt
	// deep, so it hangs into both margins exactly as the placed rect does.
	let inside_pt	= geom.content_left().to_pt() as f32;
	let top_pt		= geom.content_top().to_pt() as f32;
	let page_w_pt	= geom.width.to_pt() as f32;
	let x0			= -inside_pt;
	let y0			= -top_pt;
	let x1			= page_w_pt - inside_pt;
	let y1			= banner_h - top_pt;

	let mut ops:	Vec<DrawOp>	= Vec::new();
	ops.push(DrawOp::Fill { path: res!(Path::rect(Bounds::new(x0, y0, x1, y1))), colour: grey });

	// The title in the heading face's bold, at the template's 26 pt, small-capped run by run (the shaper
	// carries no `smcp`, so the case is synthesised: was-lowercase letters uppercased at 0.75 of the size).
	let face		= HeadFace::Role(Role::Bold);
	let size		= Sp::from_pt(26.0);
	let small_size	= Sp(size.raw() * 3 / 4);
	let sample		= res!(head_shape(fonts, &face, size, "Ag"));
	let asc			= sample.dims().height.to_pt() as f32;
	let dep			= sample.dims().depth.to_pt() as f32;
	// The band's vertical middle in the box frame, then the baseline that centres the run's box on it.
	let band_mid	= (banner_h / 2.0) - top_pt;
	let base_y		= band_mid + (asc - dep) / 2.0;

	let mut x_off	= 0.0f32;	// the title's left edge sits at the content left (box origin)
	for (run, is_small) in smallcaps_runs(title) {
		let rs		= if is_small { small_size } else { size };
		let shaped	= res!(head_shape(fonts, &face, rs, &run));
		for glyph in &shaped.run().glyphs {
			let path = res!(shaped.outline(glyph));
			if path.is_empty() {
				continue;
			}
			let t = Transform::scale(1.0, -1.0)
				.then(&Transform::translate(x_off + glyph.x, base_y - glyph.y));
			ops.push(DrawOp::Fill { path: res!(path.transform(&t)), colour: Rgba::BLACK });
		}
		x_off += shaped.dims().width.to_pt() as f32;
	}

	let graphic = Graphic::new(ops, Dims::new(measure, follow, Sp::ZERO));
	// The box holds the `#v(95pt)` of flow space; the bar draws past its top edge into the margins.
	nodes.push(Node::HBox(BoxNode::new(vec![Node::Leaf(Leaf::graphic(graphic))], Dims::new(measure, follow, Sp::ZERO))));
	Ok(())
}

/// Draws the documentation template's section banner (`template.typ`'s `section-banner`): the same
/// full-width grey bar `doc_banner` hangs into the page's top and side margins, but carrying the section's
/// logo right-aligned on the band's vertical middle rather than a title -- the Hematite guide's per-section
/// mark. The logo is loaded at the template's 30 pt height and its right edge seated one page margin
/// (2.5 cm) in from the page's right edge (the template's `pad(right: 2.5cm)`), which lands on the content's
/// right edge. The bar is drawn from one box whose ops bleed past its bounds -- the emitter clips nothing --
/// and the box holds the template's `#v(95pt)` of following space, so the inline heading beneath lands where
/// the oracle sets it. A logo that will not load leaves the bar alone.
fn section_banner(
	nodes:		&mut Vec<Node>,
	fonts:		Arc<FontSet>,
	geom:		PageGeometry,
	measure:	Sp,
	path:		&str,
)
	-> Outcome<()>
{
	let grey		= Rgba::opaque(240, 240, 240);	// the template's `colours.lightgrey`, luma(240)
	let banner_h	= 150.0f32;						// the template's rect height
	let follow		= Sp::from_pt(95.0);			// the template's `#v(95pt)` down to the heading
	let logo_h		= 30.0f32;						// the template's `image(logo_path, height: 30pt)`
	// The box origin is the content top-left, y down; the bar reaches the page's left edge (x = -inside) and
	// top edge (y = -top), runs the full page width and 150 pt deep, so it hangs into both margins.
	let inside_pt	= geom.content_left().to_pt() as f32;
	let top_pt		= geom.content_top().to_pt() as f32;
	let page_w_pt	= geom.width.to_pt() as f32;
	let x0			= -inside_pt;
	let y0			= -top_pt;
	let x1			= page_w_pt - inside_pt;
	let y1			= banner_h - top_pt;

	let mut ops:	Vec<DrawOp>	= Vec::new();
	ops.push(DrawOp::Fill { path: res!(Path::rect(Bounds::new(x0, y0, x1, y1))), colour: grey });

	// The logo, loaded 30 pt tall, its right edge one page margin in from the page's right edge (the content
	// right edge) and its box centred on the band's vertical middle. Its own ops are in a top-left frame,
	// y down; a plain translation seats them. A logo that will not load draws the bar alone.
	if let Ok(logo) = image_at_height(&fonts, path, logo_h as f64) {
		let lw			= logo.dims.width.to_pt() as f32;
		let lh			= (logo.dims.height + logo.dims.depth).to_pt() as f32;
		let right		= x1 - inside_pt;			// 2.5 cm in from the page right edge = the content right edge
		let band_mid	= (banner_h / 2.0) - top_pt;	// the band's vertical middle, box frame, y down
		let tx			= right - lw;
		let ty			= band_mid - lh / 2.0;
		let t			= Transform::translate(tx, ty);
		for op in logo.ops {
			ops.push(match op {
				DrawOp::Fill { path, colour }			=> DrawOp::Fill { path: res!(path.transform(&t)), colour },
				DrawOp::Stroke { path, colour, width }	=> DrawOp::Stroke { path: res!(path.transform(&t)), colour, width },
				DrawOp::Image { image, x, y, w, h }		=> DrawOp::Image { image, x: x + tx, y: y + ty, w, h },
			});
		}
	}

	let graphic = Graphic::new(ops, Dims::new(measure, follow, Sp::ZERO));
	// The box holds the `#v(95pt)` of flow space; the bar draws past its top edge into the margins.
	nodes.push(Node::HBox(BoxNode::new(vec![Node::Leaf(Leaf::graphic(graphic))], Dims::new(measure, follow, Sp::ZERO))));
	Ok(())
}

/// A rigid vertical spacer: a zero-width box of the given height, so it holds its space at a page top
/// where leading glue would be discarded.
fn vspacer(height: Sp) -> Node {
	Node::HBox(BoxNode::new(vec![], Dims::new(Sp::ZERO, height, Sp::ZERO)))
}

/// Appends a horizontal rule -- a standalone `#line(...)` divider -- as a filled grey bar of the given
/// width (a fraction of the measure or an absolute length), thickness and grey level, seated flush left.
/// A degenerate rule (zero width or thickness) adds nothing rather than an empty box.
fn rule_divider(nodes: &mut Vec<Node>, measure: Sp, width: Length, thickness: f64, grey: u8) {
	let w = match width {
		Length::Rel(f)	=> Sp::from_pt(measure.to_pt() * f),
		Length::Abs(pt)	=> Sp::from_pt(pt),
	};
	let wf = w.to_pt() as f32;
	let hf = thickness as f32;
	if wf <= 0.0 || hf <= 0.0 {
		return;
	}
	let h		= Sp::from_pt(thickness);
	let colour	= Rgba::opaque(grey, grey, grey);
	let rect = match Path::rect(Bounds::new(0.0, 0.0, wf, hf)) {
		Ok(r)	=> r,
		Err(_)	=> return,
	};
	let graphic	= Graphic::new(vec![DrawOp::Fill { path: rect, colour }], Dims::new(w, h, Sp::ZERO));
	nodes.push(Node::HBox(BoxNode::new(vec![Node::Leaf(Leaf::graphic(graphic))], Dims::new(measure, h, Sp::ZERO))));
}

/// Loads an image at a fixed drawn height: the image at `path` read (an SVG as its own scaled paths, a
/// raster to fill its box) at height `h`, its ops seated at the origin so a caller can place it. The
/// height fixes the size and the width follows the aspect, matching the template's `image(.., height: Npt)`.
/// Used for the page footer logo, and for the meta page's declaration mark within a table cell.
pub(crate) fn image_at_height(fonts: &Arc<FontSet>, path: &str, h: f64) -> Outcome<Graphic> {
	let height = Some(Length::Abs(h));
	// A wide box so the height hint, not the measure, governs the size; the picture keeps its aspect.
	let box_w	= Sp::from_pt(1000.0);
	match crate::image::load_figure(path) {
		Ok(crate::image::Figure::Raster(img))	=> image_graphic(box_w, img, None, height, None),
		Ok(crate::image::Figure::Vector(pic))	=> svg_graphic(fonts.clone(), box_w, pic, None, height, None),
		Err(e)									=> Err(e),
	}
}

/// Draws the page furniture -- a running head in the top margin and a folio -- onto every composed
/// page. Called after the driver has converged: the furniture sits outside the text block, so adding
/// it moves nothing and cannot reopen the fixed point.
///
/// The running head follows the book's own scheme, the even/odd split the template sets. A verso (even)
/// page carries the folio at the outer edge and the book title, in italic, at the inner; a recto (odd)
/// page carries the current chapter title, in italic, at the inner edge and the folio at the outer. The
/// current chapter is the most recent level-1 heading the ledger resolved to an earlier page. A page a
/// chapter opens at its very top -- and the first page, before any chapter runs -- omits the running
/// head and sets a centred folio at the foot instead, the usual chapter-opening treatment. The frame is
/// laid at the recto (binding-left) split; `ingot` mirrors a verso page's whole frame to the fore-edge
/// afterwards, so placing the folio at the block's left on a verso page lands it at the outer margin.
/// Both the head and the folio are shaped through the same path as the body and drawn as glyph outlines.
pub fn decorate(
	pages:			&mut [Page],
	ledger:			&Ledger,
	heads:			&[Heading],
	fonts:			&Arc<FontSet>,
	style:			Style,
	geom:			PageGeometry,
	book_title:		&str,
	footer_logo:	Option<&str>,
)
	-> Outcome<()>
{
	let content_top		= geom.content_top();
	let content_left	= geom.content_left();
	let content_width	= geom.content_width();
	// The documentation template seats a logo at the left of every page footer. It is loaded once and
	// placed on each body page; a logo that will not load leaves the footer to the folio alone.
	let footer = footer_logo.and_then(|p| image_at_height(fonts, p, 18.0).ok().map(Arc::new));
	// The body opens on this physical page; the printed folio restarts at one here, so a body page's
	// folio is its physical page less the front matter before it. A run with no headings (a lone
	// manuscript) leaves `body_start_page` zero, so the whole document is body and the folio is physical.
	let body_start		= ledger.body_start_page.max(1);
	for page in pages.iter_mut() {
		// Front matter -- the cover, title, imprint and contents leaves before the body -- carries no
		// running head and no folio, exactly as the template sets `numbering: none` there.
		if page.number < body_start {
			continue;
		}
		let folio = page.number - (body_start - 1);

		// The footer logo, seated at the left of the foot on every body page, its top at the folio's foot line.
		if let Some(g) = &footer {
			let foot_top = content_top + geom.content_height() + Sp::from_pt(14.0);
			page.frame.push(Placed::new(content_left, foot_top, g.dims, PlacedKind::Graphic(g.clone())));
		}

		// The back matter -- the bibliography and beyond -- drops the running head and centres the folio at
		// the foot, as the template sets it.
		let back_start = ledger.back_matter_start_page;
		if back_start != 0 && page.number >= back_start {
			let foot_top	= content_top + geom.content_height() + Sp::from_pt(14.0);
			let shaped		= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, style.folio_size, &fmt!("{}", folio)));
			let d			= shaped.dims();
			let x			= centre_x(geom, d.width);
			page.frame.push(Placed::new(x, foot_top, d, PlacedKind::Text(shaped)));
			continue;
		}

		// The chapter running at the top of this page (the most recent level-1 heading resolved to an
		// earlier page), whether a chapter opens at the very top of this one, and whether a part divider
		// does -- a part page carries no folio at all.
		let mut chapter:	Option<&Heading>	= None;
		let mut opens					= false;
		let mut opens_part				= false;
		for h in heads {
			if let Some(a) = ledger.get(&h.id) {
				if a.pos.page < page.number {
					if h.level == 1 { chapter = Some(h); }
				} else if a.pos.page == page.number {
					// A chapter opens the page either at its very top (a numbered or banner-bar opener) or
					// beneath a `#section-banner` this page carries: both suppress the running head and seat the
					// folio at the foot, so the head never lands on the grey bar.
					if h.level == 1 && (a.pos.y == content_top || h.banner) { opens = true; }
					if h.level == 0 && a.pos.y == content_top { opens_part = true; }	// a part divider opens at the very top
				} else {
					break;	// headings are in document order, so the rest resolve to later pages
				}
			}
		}

		// A part divider stands alone with no folio and no head, as the template's part page sets.
		if opens_part {
			continue;
		}

		// The head baseline sits a fixed step above the text block; a folio at the foot sits a step below.
		let head_base	= content_top - Sp::from_pt(8.0);
		let foot_top	= content_top + geom.content_height() + Sp::from_pt(14.0);
		let num			= fmt!("{}", folio);

		if opens || chapter.is_none() {
			// A chapter-opening page: no running head, a centred folio at the foot.
			let shaped	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, style.folio_size, &num));
			let d		= shaped.dims();
			let x		= centre_x(geom, d.width);
			page.frame.push(Placed::new(x, foot_top, d, PlacedKind::Text(shaped)));
			continue;
		}

		// The folio, at the outer margin of the running head. On a recto (odd) page the outer edge is the
		// block's right; on a verso (even) page it is the block's left, which the mirror shift carries to
		// the fore-edge.
		let folio	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, style.folio_size, &num));
		let fd		= folio.dims();
		let folio_x	= if page.number % 2 == 0 {
			content_left
		} else {
			content_left + content_width - fd.width
		};
		page.frame.push(Placed::new(folio_x, head_base - fd.height, fd, PlacedKind::Text(folio)));

		// The title side: the book title on a verso page, set against the folio at the inner edge; the
		// chapter title on a recto, set from its rich runs so a maths span or emphasis in the title renders
		// rather than dropping to a gap. Both italic.
		if page.number % 2 == 0 {
			if !book_title.is_empty() {
				let shaped	= res!(ShapedText::new(fonts.clone(), Role::Italic, Dir::Ltr, style.header_size, book_title));
				let d		= shaped.dims();
				let x		= content_left + content_width - d.width;	// verso: title at the inner (spine) edge
				page.frame.push(Placed::new(x, head_base - d.height, d, PlacedKind::Text(shaped)));
			}
		} else if let Some(ch) = chapter {
			let (rnodes, rd) = res!(inline_segments(fonts, style, &ch.segments, Role::Italic, style.header_size));
			// Recto: title at the inner (spine) edge, its box top a full ascent above the head baseline.
			place_run(&mut page.frame, &rnodes, content_left, head_base - rd.height);
		}
	}
	Ok(())
}

/// The x that centres a box of width `w` in the text block. A box wider than the measure starts at
/// the left edge rather than hanging off it.
fn centre_x(geom: PageGeometry, w: Sp) -> Sp {
	let slack = (geom.content_width().raw() - w.raw()).max(0) / 2;
	geom.content_left() + Sp(slack)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn count_words_counts_letter_runs_across_blocks() {
		// Letter runs, as the template's `\p{L}+` counter steps: "don't" is two runs, a bare number none.
		let blocks = vec![
			Block::Heading { level: 1, segments: vec![Segment::text("The Purpose")], label: None },
			Block::Paragraph { text: "It reads a document and writes 42 pages.".to_string() },
			Block::List { ordered: false, items: vec![vec![Segment::strong("one two")]] },
		];
		// Heading: 2; paragraph: "It reads a document and writes pages" = 7 (the "42" counts none);
		// list item: 2. Total 11.
		assert_eq!(count_words(&blocks), 11);
	}

	#[test]
	fn build_meta_table_appends_reading_time_and_mark() {
		let fm = FrontMatter {
			title:			"Austenite".to_string(),
			subtitle:		None,
			author:			"J. D. Hoogland".to_string(),
			cover_image:	None,
			logo_image:		None,
			publisher:		None,
			edition:		None,
			isbn:			None,
			copyright:		Some("Copyright © 12025 Oxedyne. All rights reserved.".to_string()),
			rights:			None,
			ai_declaration:	None,
			website:		None,
			toolchain:		false,
			dedication:		None,
			about_author:	None,
			title_size:		Sp::from_pt(28.0),
			subtitle_size:	Sp::from_pt(16.0),
			author_size:	Sp::from_pt(17.0),
			back_title_size:	Sp::from_pt(14.0),
			sidebar_grey:	Some(240),
			sidebar_frac:	0.45,
			title_smallcaps:	true,
			top_logo:		None,
			top_logo_width:		Sp::ZERO,
			bottom_logo:	None,
			bottom_logo_width:	Sp::ZERO,
			footer_logo:	None,
			meta_rows:		vec![MetaRow {
				version:		Some("0.1.0".to_string()),
				date:			Some("12026-08-08".to_string()),
				authors:		"J. D. Hoogland".to_string(),
				notes:			Some("Created.".to_string()),
				ai_mark_path:	Some("assets/svg/doc_made_with_ai_opt.svg".to_string()),
				ai_mark_words:	Some("Made with AI".to_string()),
				ai_mark_url:	Some("https://need2know.ai/with-ai/doc".to_string()),
			}],
			reading_min:	Some(51),
			acknowledgement:	Some("We acknowledge...".to_string()),
		};
		let table = build_meta_table(&fm).expect("meta table builds");
		assert_eq!(table.rows.len(), 2, "one header row and one revision row");
		assert!(table.header, "the first row is the header");
		// The author cell carries the declaration mark stacked beneath the name (column index 2: Ver, Date, Author).
		assert!(table.rows[1].cells[2].mark.is_some(), "the author cell carries the AI mark");
		// The notes cell has the reading time appended to the authored notes.
		let notes = match &table.rows[1].cells[3].content[0] {
			Segment::Text(t)	=> t.clone(),
			_					=> String::new(),
		};
		assert!(notes.contains("Created.") && notes.contains("Reading time: 51 [min]"),
			"the notes cell appends the reading time: {:?}", notes);
	}

	#[test]
	fn docbanner_section_banner_sets_its_heading_inline_not_a_second_opener() -> Outcome<()> {
		// A `DocBanner` tree -- its default chapter opener is the grey title bar -- that opts one chapter in
		// with an explicit `#section-banner` must set that chapter's title inline beneath the banner, not
		// open a second grey bar of its own. The section banner forces one page eject; the heading that
		// follows it must add none. Before the fix the heading opened as a chapter and the forced-eject
		// count was two, which is the duplicate bar this guards against.
		let fonts	= Arc::new(res!(crate::fonts::libertinus()));
		let geom	= PageGeometry::a4();
		let mut style	= Style::default();
		style.heading_style = HeadingStyle::DocBanner;
		let blocks = vec![
			Block::Paragraph { text: "Intro before the section.".to_string() },
			Block::section_banner("assets/svg/pearlite_logo_text_right.svg".to_string()),
			Block::Heading { level: 1, segments: vec![Segment::text("Pearlite")], label: None },
			Block::Paragraph { text: "Pearlite is the format.".to_string() },
		];
		let (doc, heads) = res!(author(fonts, geom, style, None, &blocks, None, None));
		assert!(heads.iter().any(|h| h.level == 1 && h.title == "Pearlite" && h.banner),
			"the level-1 heading after a #section-banner carries the banner flag");
		let forced = doc.nodes.iter()
			.filter(|n| matches!(n, Node::Penalty(p) if p.is_forced()))
			.count();
		assert_eq!(forced, 1,
			"only the section banner forces a page break; its heading opens inline, adding none (got {})", forced);
		Ok(())
	}
}
