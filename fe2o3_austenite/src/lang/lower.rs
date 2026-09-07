//! Lowering from the Ingot surface tree to the block layer.
//!
//! The seam between front end and engine: a surface [`Item`](super::ast::Item) becomes a
//! [`doc::Block`](crate::doc::Block) the two-pass driver already knows how to set. A heading lowers to
//! a heading; a paragraph of plain prose to a plain [`Block::Paragraph`], and a paragraph carrying any
//! emphasis to a [`Block::RichParagraph`] of [`Segment`]s -- keeping the fast single-role break path
//! for the common case. The source spans are dropped here, since the block layer does not yet carry
//! them. As the surface language grows, this is where a richer item collapses to what the engine sets.

use crate::doc::{
	Block,
	Segment,
};
use crate::ir::Sp;
use crate::table::{
	Align,
	Cell,
	Row,
	Table,
};

use super::ast::{
	AlignSpec,
	FigureBody,
	Inline,
	Item,
	TableSpec,
};

/// Lowers a surface item list to the block list the driver authors from.
pub fn blocks(items: &[Item]) -> Vec<Block> {
	let mut out = Vec::with_capacity(items.len());
	for item in items {
		match item {
			Item::Heading { level, runs, label, .. }	=> out.push(
				Block::heading_rich(*level, lower_runs(runs), label.clone())),
			Item::Paragraph { runs, label, .. }	=> out.push(lower_paragraph(runs, label.clone())),
			Item::List { ordered, items, .. }	=> out.push(Block::list(
				*ordered,
				items.iter().map(|item| lower_runs(item)).collect())),
			Item::Code { lines, .. }			=> out.push(Block::code(lines.clone())),
			Item::Table { spec, .. }			=> out.push(Block::table(build_table(spec))),
			Item::Rule { width, thickness, grey, .. }	=> out.push(Block::rule(*width, *thickness, *grey)),
			Item::Figure { body, caption, supplement, label, .. }	=> {
				let caption = caption.as_ref().map(|runs| lower_runs(runs));
				out.push(match body {
					FigureBody::Table(spec)	=> Block::table_figure(
						build_table(spec), caption, supplement.clone(), label.clone()),
					FigureBody::Image { path, width, height, scale }	=> Block::image_figure(
						path.clone(), *width, *height, *scale,
						caption, supplement.clone(), label.clone()),
					FigureBody::Code(figure)	=> Block::code_figure(
						figure.clone(), caption, supplement.clone(), label.clone()),
				});
			},
			Item::Image { path, width, height, scale, .. }	=> out.push(
				Block::image(path.clone(), *width, *height, *scale)),
		}
	}
	out
}

/// Builds a [`Table`] from the parsed spec: the flat cells are chunked into rows of `ncols`, each cell
/// carrying its inline runs lowered to segments and its alignment from the [`AlignSpec`]. A header row's
/// cells set centred under the fixed forms; a closure is evaluated per cell, so a `(col, row) => ...`
/// spec sets each cell exactly as its own row/column logic dictates.
fn build_table(spec: &TableSpec) -> Table {
	let ncols = spec.ncols.max(1);
	let mut rows:	Vec<Row>	= Vec::new();
	for (r, chunk) in spec.cells.chunks(ncols).enumerate() {
		let mut cells = Vec::with_capacity(ncols);
		for (c, runs) in chunk.iter().enumerate() {
			let content = lower_runs(runs);
			cells.push(Cell::rich(content, cell_align(&spec.align, spec.header, r, c)));
		}
		rows.push(Row::new(cells));
	}
	// A `columns: (2fr, 5fr, ...)` track list sizes the columns fractionally, as Typst does; a bare
	// `columns: N` carries no weights and the columns fall back to content sizing. A weight list shorter
	// than the columns is padded with content-sized zeros so every column has an entry.
	let mut table = if spec.weights.iter().any(|&w| w > 0.0) {
		let mut weights = spec.weights.clone();
		weights.resize(ncols, 0.0);
		Table::with_weights(spec.header, rows, weights)
	} else {
		Table::new(spec.header, rows)
	};
	// A `text(size: Npt)` wrapper (the books set their claim tables at 7 pt) and an explicit `inset:` cell
	// padding carry through, so the table sets at the oracle's reduced size rather than the body size.
	table.text_size	= spec.text_pt.map(Sp::from_pt);
	table.inset		= spec.inset_pt.map(Sp::from_pt);
	table
}

/// The alignment of one cell at row `r`, column `c`, given the table's declared [`AlignSpec`]. A closure
/// carries its own row/column logic and is evaluated for every cell, header row included; the fixed
/// forms have no row dependence, so a header row centres its labels as Typst's book style does.
fn cell_align(spec: &AlignSpec, header: bool, r: usize, c: usize) -> Align {
	if let AlignSpec::Closure(cl) = spec {
		return cl.align_at(c, r);
	}
	if header && r == 0 {
		return Align::Centre;	// a header row centres its labels
	}
	match spec {
		AlignSpec::Uniform(a)		=> *a,
		AlignSpec::PerColumn(cols)	=> cols.get(c).copied().unwrap_or(Align::Left),
		AlignSpec::Closure(cl)		=> cl.align_at(c, r),
	}
}

/// Lowers a paragraph's inline runs. A paragraph of one plain text run keeps the plain-paragraph path
/// (a single-role Knuth-Plass break); the moment it carries an emphasis run it becomes a rich paragraph
/// of segments, which the driver breaks with a face per run. A `label` is the paragraph's trailing
/// `<name>`, carried onto a display equation so an `@`-reference can resolve to it.
fn lower_paragraph(runs: &[Inline], label: Option<String>) -> Block {
	match runs {
		[Inline::Text(text)]	=> Block::paragraph(text.clone()),
		// A paragraph that is nothing but one maths span is a display equation on its own line. The
		// template sets `math.equation(numbering: "(1)")`, so every display equation takes the next
		// number; inline maths, a run among others, never does.
		[Inline::Math(atom)]	=> Block::equation(atom.clone(), true, label),
		_						=> Block::rich(lower_runs(runs)),
	}
}

/// Lowers a run of inlines to segments, then groups adjacent citations, so a `#cite ... #cite`
/// sequence parted by nothing but whitespace sets as one parenthesis.
fn lower_runs(runs: &[Inline]) -> Vec<Segment> {
	group_adjacent_cites(runs.iter().map(lower_inline).collect())
}

/// Groups adjacent citations into one, as Typst does: two or more `#cite` calls separated by nothing
/// but whitespace set as a single parenthesis with a semicolon separator (`(A Year; B Year)`), while a
/// citation parted from the next by any prose keeps its own parenthesis. The whitespace between grouped
/// citations is dropped, matching the oracle.
fn group_adjacent_cites(segments: Vec<Segment>) -> Vec<Segment> {
	// Grouping can only change a run holding two or more citations.
	if segments.iter().filter(|s| matches!(s, Segment::Cite(_))).count() < 2 {
		return segments;
	}
	let n = segments.len();
	let mut out: Vec<Segment> = Vec::with_capacity(n);
	let mut i = 0usize;
	while i < n {
		if let Segment::Cite(keys) = &segments[i] {
			let mut group = keys.clone();
			let mut j = i + 1;
			// Absorb each following citation reachable across whitespace-only text alone.
			loop {
				let mut k = j;
				while k < n && is_whitespace_text(&segments[k]) {
					k += 1;
				}
				match segments.get(k) {
					Some(Segment::Cite(more))	=> {
						group.extend(more.iter().cloned());
						j = k + 1;
					},
					_							=> break,
				}
			}
			out.push(Segment::Cite(group));
			i = j;
		} else {
			out.push(segments[i].clone());
			i += 1;
		}
	}
	out
}

/// Is the segment a text run of nothing but whitespace?
fn is_whitespace_text(seg: &Segment) -> bool {
	matches!(seg, Segment::Text(t) if t.chars().all(char::is_whitespace))
}

fn lower_inline(run: &Inline) -> Segment {
	match run {
		Inline::Text(text)		=> Segment::text(text.clone()),
		Inline::Strong(text)	=> Segment::strong(text.clone()),
		Inline::Emph(text)		=> Segment::emph(text.clone()),
		Inline::BoldItalic(text)	=> Segment::bold_italic(text.clone()),
		Inline::Super(text)		=> Segment::superscript(text.clone()),
		Inline::PageRef(label)	=> Segment::page_ref(label.clone()),
		Inline::Code(text)		=> Segment::code(text.clone()),
		Inline::Math(atom)		=> Segment::math(atom.clone()),
		Inline::Glossary { term, display }
								=> Segment::glossary(term.clone(), display.clone()),
		Inline::Footnote(note)	=> Segment::footnote(lower_runs(note)),
		Inline::Cite(keys)		=> Segment::cite(keys.clone()),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn cite(k: &str) -> Inline { Inline::Cite(vec![k.to_string()]) }

	/// Adjacent citations parted by nothing but whitespace collapse to one citation segment carrying
	/// every key in source order, the whitespace between them dropped.
	#[test]
	fn adjacent_cites_group() {
		let runs = vec![
			Inline::Text("A ".to_string()),
			cite("a"),
			Inline::Text(" ".to_string()),
			cite("b"),
			Inline::Text(" end.".to_string()),
		];
		let segs = lower_runs(&runs);
		assert_eq!(segs.len(), 3, "got: {:?}", segs);
		assert!(matches!(&segs[0], Segment::Text(t) if t == "A "));
		match &segs[1] {
			Segment::Cite(keys)	=> assert_eq!(keys, &vec!["a".to_string(), "b".to_string()]),
			other				=> panic!("expected one grouped cite, got {:?}", other),
		}
		assert!(matches!(&segs[2], Segment::Text(t) if t == " end."));
	}

	/// Three adjacent citations all fall into one group.
	#[test]
	fn three_adjacent_cites_group() {
		let runs = vec![
			cite("a"),
			Inline::Text(" ".to_string()),
			cite("b"),
			Inline::Text(" ".to_string()),
			cite("c"),
		];
		let segs = lower_runs(&runs);
		assert_eq!(segs.len(), 1, "got: {:?}", segs);
		match &segs[0] {
			Segment::Cite(keys)	=> assert_eq!(
				keys, &vec!["a".to_string(), "b".to_string(), "c".to_string()]),
			other				=> panic!("expected one grouped cite, got {:?}", other),
		}
	}

	/// A citation, then prose, then a citation stays two separate citation segments -- no over-grouping.
	#[test]
	fn cite_prose_cite_stays_separate() {
		let runs = vec![
			cite("a"),
			Inline::Text(" and then ".to_string()),
			cite("b"),
		];
		let segs = lower_runs(&runs);
		let cites: Vec<&Segment> = segs.iter()
			.filter(|s| matches!(s, Segment::Cite(_)))
			.collect();
		assert_eq!(cites.len(), 2, "expected two separate cites, got: {:?}", segs);
		assert!(matches!(&cites[0], Segment::Cite(k) if k == &vec!["a".to_string()]));
		assert!(matches!(&cites[1], Segment::Cite(k) if k == &vec!["b".to_string()]));
		// The intervening prose survives.
		assert!(segs.iter().any(|s| matches!(s, Segment::Text(t) if t == " and then ")));
	}

	/// A single citation is left untouched.
	#[test]
	fn lone_cite_untouched() {
		let runs = vec![
			Inline::Text("D ".to_string()),
			cite("a"),
			Inline::Text(" end.".to_string()),
		];
		let segs = lower_runs(&runs);
		assert_eq!(segs.len(), 3);
		assert!(matches!(&segs[1], Segment::Cite(k) if k == &vec!["a".to_string()]));
		// The trailing space after a lone cite is preserved.
		assert!(matches!(&segs[2], Segment::Text(t) if t == " end."));
	}

	/// A `#cite(<a>, <b>)` multi-key call already lowers to one segment; grouping leaves it whole.
	#[test]
	fn multikey_call_unchanged() {
		let runs = vec![Inline::Cite(vec!["a".to_string(), "b".to_string()])];
		let segs = lower_runs(&runs);
		assert_eq!(segs.len(), 1);
		assert!(matches!(&segs[0], Segment::Cite(k) if k == &vec!["a".to_string(), "b".to_string()]));
	}
}
