//! The surface syntax tree of a Typst source file, before lowering to [`doc::Block`](crate::doc::Block).
//!
//! Increment 2 adds inline emphasis to the markup spine: a document is a sequence of headings and
//! paragraphs, and a paragraph is a sequence of [`Inline`] runs -- plain text, `*strong*`, `_emph_`.
//! The `#` code mode and the declared-query references of `sec_language` are later increments -- the
//! tree names only what the engine can already set.

use crate::ir::Length;
use crate::ir::Span;
use crate::lang::codefig::CodeFigure;
use crate::math::Atom;
use crate::table::Align;

/// One block of Ingot markup. The byte span is carried for a future diagnostic caret; the driver's
/// `Span` model already reserves it, so the front end records it from the first increment.
#[derive(Clone, Debug)]
pub enum Item {
	Heading { level: u8, runs: Vec<Inline>, label: Option<String>, span: Span },	// label: a trailing <name>, runs: the title's inline markup
	Paragraph { runs: Vec<Inline>, label: Option<String>, span: Span },	// label: a trailing <name>, anchoring a display equation for cross-reference
	List { ordered: bool, items: Vec<Vec<Inline>>, span: Span },	// `-` bullets or `+` numbered
	Code { lines: Vec<String>, span: Span },	// a ```-fenced block, set verbatim in the mono face
	Table { spec: TableSpec, span: Span },	// a bare `#table(...)`, not wrapped in a figure
	Figure { body: FigureBody, caption: Option<Vec<Inline>>, supplement: String, label: Option<String>, span: Span },	// caption: the caption's inline markup
	Image { path: String, width: Option<Length>, height: Option<Length>, scale: Option<f64>, span: Span },	// a line-leading `#padded-image(...)`/`#image(...)`, set centred without a figure number
	SectionBanner { path: String, span: Span },	// a line-leading `#section-banner("logo")`, a full-width grey bar carrying a right-aligned section logo
	Rule { width: Length, thickness: f64, grey: u8, span: Span },	// a standalone `#line(length:.., stroke:..)` horizontal divider
}

/// What a `#figure(...)` wraps: a `#table(...)` this reader sets in full, or an image call whose ink is
/// deferred to a later increment and stood in for by a sized placeholder box.
#[derive(Clone, Debug)]
pub enum FigureBody {
	Table(TableSpec),
	// The image path and any sizing the call declared: `width`/`height` from `image(...)`, `scale` from
	// `padded-image(...)`. A hint the call omits is `None`, and the figure fills the measure.
	Image { path: String, width: Option<Length>, height: Option<Length>, scale: Option<f64> },
	// A figure the document draws by code -- a CeTZ/Fletcher diagram, a bar chart or a line plot -- read
	// from the `#figure` body's source into a ready builder that draws real vector ink.
	Code(CodeFigure),
}

/// A parsed Typst `#table(...)` call, before it is built into a [`table::Table`](crate::table::Table).
/// Each cell keeps its inline runs, row-major, so a bold header, an italic word, a superscript or an
/// in-cell maths span sets with its own face rather than flattening to upright text; `header` is set when
/// a `fill:` keys the first row; `align` records the column alignment the call declared.
#[derive(Clone, Debug)]
pub struct TableSpec {
	pub ncols:		usize,
	pub header:		bool,
	pub align:		AlignSpec,
	pub weights:	Vec<f64>,		// the `Nfr` weight per column; 0.0 for an `auto`/fixed track, empty for a bare `columns: N`
	pub text_pt:	Option<f64>,	// a `text(size: Npt)[...]` wrapper's size, so a small table sets small
	pub inset_pt:	Option<f64>,	// the `inset:` cell padding in points, overriding the default
	pub cells:		Vec<Vec<Inline>>,	// flat, row-major; each cell a run of inline markup
}

/// How a table's cells align. `Uniform` sets every cell alike; `PerColumn` gives each column its own
/// alignment (a header row still sets centred); `Closure` carries the `(col, row) => ...` idiom these
/// books use, evaluated per cell so the source's own row/column logic is honoured rather than guessed.
#[derive(Clone, Debug)]
pub enum AlignSpec {
	Uniform(Align),
	PerColumn(Vec<Align>),
	Closure(ClosureAlign),
}

/// A table `align:` closure captured for evaluation at each cell. Typst passes the closure the cell's
/// 0-based `(col, row)` and expects an alignment back; the parameter names and body are kept verbatim so
/// a cell's alignment is computed from the expression the source declared. The supported body is the
/// idiom these books use: an `if`/`else if`/`else` chain over `col`/`row` comparisons (`==`, `!=`, `<`,
/// `<=`, `>`, `>=`, combined with `or`/`and`), and alignment expressions that are a word (`left`,
/// `center`, `right`, with any `+ horizon`/`+ top` ignored) or a `(a, b, ...).at(col)` tuple pick.
#[derive(Clone, Debug)]
pub struct ClosureAlign {
	pub col_var:	String,
	pub row_var:	String,
	pub body:		String,
}

impl ClosureAlign {
	/// The alignment the closure yields for the cell at 0-based `col`, `row`. An expression the grammar
	/// does not cover falls back to left, the Typst default for an unrecognised alignment.
	pub fn align_at(&self, col: usize, row: usize) -> Align {
		self.eval_body(self.body.trim(), col, row)
	}

	fn eval_body(&self, s: &str, col: usize, row: usize) -> Align {
		let s = strip_braces(s.trim()).trim();
		if let Some(rest) = s.strip_prefix("if ").or_else(|| s.strip_prefix("if(")) {
			// The condition runs to the top-level `{` that opens the then-block.
			let chars: Vec<char> = rest.chars().collect();
			if let Some(open) = find_top_brace(&chars) {
				let cond: String = chars[..open].iter().collect();
				if let Some((block, after)) = read_brace_group(&chars, open) {
					if self.eval_cond(&cond, col, row) {
						return self.eval_body(&block, col, row);
					}
					let tail: String = chars[after..].iter().collect();
					let tail = tail.trim();
					if let Some(e) = tail.strip_prefix("else") {
						return self.eval_body(e.trim(), col, row);
					}
					return Align::Left;	// an `if` with no `else` leaves the rest flush left
				}
			}
			return Align::Left;
		}
		self.eval_expr(s, col, row)
	}

	fn eval_expr(&self, s: &str, col: usize, row: usize) -> Align {
		let s = s.trim();
		// A `(a, b, ...).at(index)` tuple pick: split the tuple, index it by the resolved argument.
		if let Some(pos) = s.find(".at(") {
			let tuple = s[..pos].trim();
			let idx_src: String = s[pos + 4..].chars().take_while(|&c| c != ')').collect();
			let items = split_top_commas(strip_parens(tuple));
			let idx = self.eval_index(&idx_src, col, row);
			return items.get(idx).map_or(Align::Left, |it| word_to_align(it));
		}
		word_to_align(s)
	}

	fn eval_index(&self, s: &str, col: usize, row: usize) -> usize {
		let s = s.trim();
		if s == self.col_var { return col; }
		if s == self.row_var { return row; }
		s.parse::<usize>().unwrap_or(0)
	}

	fn eval_cond(&self, s: &str, col: usize, row: usize) -> bool {
		// `or` is the loosest binding, then `and`, then a single comparison.
		if s.contains(" or ") {
			return s.split(" or ").any(|p| self.eval_cond(p, col, row));
		}
		if s.contains(" and ") {
			return s.split(" and ").all(|p| self.eval_cond(p, col, row));
		}
		self.eval_atom(s.trim(), col, row)
	}

	fn eval_atom(&self, s: &str, col: usize, row: usize) -> bool {
		for op in ["==", "!=", "<=", ">=", "<", ">"] {
			if let Some((lhs, rhs)) = s.split_once(op) {
				let lv = self.eval_index(lhs.trim(), col, row);
				let rv = match rhs.trim().parse::<usize>() {
					Ok(n)	=> n,
					Err(_)	=> return false,
				};
				return match op {
					"=="	=> lv == rv,
					"!="	=> lv != rv,
					"<="	=> lv <= rv,
					">="	=> lv >= rv,
					"<"		=> lv < rv,
					">"		=> lv > rv,
					_		=> false,
				};
			}
		}
		false
	}
}

/// Maps a Typst alignment word to an [`Align`], ignoring a `+ horizon`/`+ top` vertical component and
/// treating `start`/`end` as left/right. An unknown word is left-aligned, the Typst default.
fn word_to_align(s: &str) -> Align {
	let first = s.trim().split(|c: char| c.is_whitespace() || c == '+').next().unwrap_or("").trim();
	match first {
		"center" | "centre"	=> Align::Centre,
		"right" | "end"		=> Align::Right,
		_					=> Align::Left,
	}
}

/// Strips one matching outer `{ ... }` layer, leaving other text untouched.
fn strip_braces(s: &str) -> &str {
	let t = s.trim();
	match (t.strip_prefix('{'), t.strip_suffix('}')) {
		(Some(_), Some(_))	=> &t[1..t.len() - 1],
		_					=> t,
	}
}

/// Strips one matching outer `( ... )` layer.
fn strip_parens(s: &str) -> &str {
	let t = s.trim();
	match (t.strip_prefix('('), t.strip_suffix(')')) {
		(Some(_), Some(_))	=> &t[1..t.len() - 1],
		_					=> t,
	}
}

/// The index of the first `{` not nested inside parentheses, or `None`.
fn find_top_brace(chars: &[char]) -> Option<usize> {
	let mut depth = 0i32;
	for (i, &c) in chars.iter().enumerate() {
		match c {
			'(' | '['	=> depth += 1,
			')' | ']'	=> depth -= 1,
			'{' if depth == 0	=> return Some(i),
			_			=> {},
		}
	}
	None
}

/// Reads the `{ ... }` group whose opening brace sits at `open`, returning its inner text and the index
/// just past the closing brace.
fn read_brace_group(chars: &[char], open: usize) -> Option<(String, usize)> {
	let mut depth = 0i32;
	for i in open..chars.len() {
		match chars[i] {
			'{'	=> depth += 1,
			'}'	=> {
				depth -= 1;
				if depth == 0 {
					let inner: String = chars[open + 1..i].iter().collect();
					return Some((inner, i + 1));
				}
			},
			_	=> {},
		}
	}
	None
}

/// Splits a tuple's inner text on top-level commas, ignoring commas nested in brackets.
fn split_top_commas(s: &str) -> Vec<String> {
	let mut out		= Vec::new();
	let mut depth	= 0i32;
	let mut cur		= String::new();
	for c in s.chars() {
		match c {
			'(' | '[' | '{'	=> { depth += 1; cur.push(c); },
			')' | ']' | '}'	=> { depth -= 1; cur.push(c); },
			',' if depth == 0	=> out.push(std::mem::take(&mut cur)),
			_				=> cur.push(c),
		}
	}
	if !cur.trim().is_empty() {
		out.push(cur);
	}
	out
}

/// One inline run of a paragraph: ordinary prose, a run marked for emphasis, a cross-reference, or an
/// inline code span. A run's text is flat; the one nesting the vocabulary carries is strong-within-emph
/// or emph-within-strong (`*_x_*`, `_*x*_`), which collapses to a single [`Inline::BoldItalic`] run.
#[derive(Clone, Debug)]
pub enum Inline {
	Text(String),
	Strong(String),	// *strong*, lowered to a bold segment
	Emph(String),	// _emph_ or #emph[...], lowered to an italic segment
	BoldItalic(String),	// *_x_* or _*x*_, the two faces nested, lowered to a bold-italic segment
	Super(String),	// #super[...], lowered to a raised, smaller segment
	PageRef(String),	// @label, resolving to the labelled anchor's page number
	Code(String),	// `raw` or #raw("..."), set in the mono face
	Math(Atom),		// $...$, parsed to the engine's maths tree
	Glossary { term: String, display: String },	// a glossary term: bold-italic on its first document use
	Footnote(Vec<Inline>),	// #footnote[...], its note markup set at the foot of the page its mark lands on
	Cite(Vec<String>),	// #cite(<key>) or #cite(<a>, <b>), resolved to (Author Year) against the bibliography
}
