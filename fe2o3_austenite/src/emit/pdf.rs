//! The PDF page writer.
//!
//! The parallel of [`super::svg`], over the very same placed frames. Where the SVG writer renders each
//! box and glyph outline as a `<path>` element, this one hands the same [`Path`] and [`Rgba`] to
//! `fe2o3_graphics`'s [`PdfWriter`], which emits them as fill and stroke operators in a page's content
//! stream. No font is embedded: a glyph is a filled outline path, exactly as it is for SVG, which is
//! the whole reason a PDF writer here is short.
//!
//! A document is one file across all its pages, not a string per page, so this module's entry point is
//! [`render_document`] rather than the per-page `render_page` the [`super::Emitter`] enum uses for
//! SVG.

use crate::font::ShapedText;
use crate::ir::{
	DrawOp,
	Graphic,
	Sp,
};
use crate::page::{
	Page,
	PlacedKind,
};

use std::io::Write;

use oxedyne_fe2o3_core::prelude::*;
use oxedyne_fe2o3_graphics::{
	colour::Rgba,
	path::{
		Bounds,
		Path,
	},
	pdf::{
		OutlineItem,
		PdfPage,
		PdfStream,
		PdfWriter,
	},
	transform::Transform,
};

/// Renders a whole document -- every page -- as one PDF file, held in a buffer. A convenience for a
/// short run; a whole book streams to a file with [`stream_document`] instead, which never holds more
/// than one page's outlines. The bytes are the same either way.
pub fn render_document(pages: &[Page]) -> Outcome<Vec<u8>> {
	let mut writer = PdfWriter::new();
	for page in pages {
		writer.add_page(res!(render_page(page)));
	}
	writer.to_bytes()
}

/// Opens a page-at-a-time PDF stream over `out`, for a document of exactly `total` pages.
///
/// This is the streaming half of the emitter, and the reason a whole-book compile is flat in memory:
/// the caller composes one page, calls [`write_page`] to serialise it to `out`, then drops the page's
/// frame, so neither the engine nor the writer ever holds every page's glyph outlines at once. Close
/// the stream with [`PdfStream::finish`] once all `total` pages are written. Compression is off, as
/// [`render_document`] leaves it, so the two produce identical bytes.
pub fn open_document<W: Write>(out: W, total: usize) -> Outcome<PdfStream<W>> {
	PdfStream::new(out, total, false)
}

/// As [`open_document`], but the file also carries a document outline (the viewer's bookmark side
/// panel), built by the caller from the heading table and the front-matter anchors. An empty outline
/// yields a file byte-identical to [`open_document`]'s.
pub fn open_document_with_outline<W: Write>(
	out:		W,
	total:		usize,
	outline:	Vec<OutlineItem>,
)
	-> Outcome<PdfStream<W>>
{
	PdfStream::new_with_outline(out, total, false, outline)
}

/// Renders one page's frame to the open PDF stream. The page's outlines live only for this call: the
/// [`PdfPage`] built here is written and dropped before returning, so the caller may drop the page's
/// frame the moment this returns.
pub fn write_page<W: Write>(stream: &mut PdfStream<W>, page: &Page) -> Outcome<()> {
	stream.page(&res!(render_page(page)))
}

/// Writes a page whose draw list and content stream were built elsewhere -- on a worker thread, so the
/// costly path transforms and serialisation run off the writer's thread. The sequential framing and the
/// page-ordered `/ID` fold stay here, so the bytes are identical to [`write_page`]'s. `content` is the
/// [`PdfPage::content_bytes`] of the same `pdf_page`.
pub fn write_page_prepared<W: Write>(
	stream:		&mut PdfStream<W>,
	pdf_page:	&PdfPage,
	content:	&[u8],
)
	-> Outcome<()>
{
	stream.page_prepared(pdf_page, content)
}

/// Builds one page's draw list: a white ground, then each placed box as a fill or a stroke.
///
/// The coordinates are the engine's page frame -- top-left origin, y down -- and are handed on
/// unflipped, since `fe2o3_graphics::pdf` flips the whole page itself.
pub fn render_page(page: &Page) -> Outcome<PdfPage> {
	let w = page.geom.width.to_pt();
	let h = page.geom.height.to_pt();
	let mut out = PdfPage::new(w, h);

	// A white ground, matching the SVG writer's opaque background rectangle.
	out.fill(res!(Path::rect(Bounds::new(0.0, 0.0, w as f32, h as f32))), Rgba::WHITE);

	// A half-point grey pen outlines a reservation, so a proof shows where a resolved value will sit
	// without the box reading as content.
	let grey = Rgba::new(176, 176, 176, 255);

	for placed in &page.frame.placed {
		// Real text is drawn glyph by glyph as filled outlines; a rule or a reservation as one
		// rectangle.
		if let PlacedKind::Text(shaped) = &placed.kind {
			res!(draw_text(&mut out, placed.x, placed.y, placed.dims.height, shaped));
			continue;
		}
		if let PlacedKind::Graphic(g) = &placed.kind {
			res!(draw_graphic(&mut out, placed.x, placed.y, g));
			continue;
		}

		let x0 = placed.x.to_pt() as f32;
		let y0 = placed.y.to_pt() as f32;
		let x1 = (placed.x + placed.dims.width).to_pt() as f32;
		let y1 = (placed.y + placed.dims.height + placed.dims.depth).to_pt() as f32;

		// A zero-area box has nothing to draw, and `Path::rect` would reject it.
		if x1 <= x0 || y1 <= y0 {
			continue;
		}
		let path = res!(Path::rect(Bounds::new(x0, y0, x1, y1)));
		match &placed.kind {
			PlacedKind::Rule		=> out.fill(path, Rgba::BLACK),
			PlacedKind::Reserved	=> out.stroke(path, grey, 0.5),
			PlacedKind::Text(_)		=> continue,	// drawn above
			PlacedKind::Graphic(_)	=> continue,	// drawn above
		}
	}

	// The running head and folio arrive as `PlacedKind::Text` and are drawn as glyph outlines with the
	// body, above. This writer adds no page furniture of its own.
	Ok(out)
}

/// Draws a placed graphic: each op's path translated to where the graphic landed, then filled or
/// stroked. The paths are y down in points already, so only a translation is needed; the PDF writer
/// flips the whole page once, which leaves the graphic the right way up like the rest of the page.
fn draw_graphic(
	out:		&mut PdfPage,
	bx:			Sp,
	by:			Sp,
	graphic:	&Graphic,
)
	-> Outcome<()>
{
	let t = Transform::translate(bx.to_pt() as f32, by.to_pt() as f32);
	let ox = bx.to_pt();
	let oy = by.to_pt();
	for op in &graphic.ops {
		match op {
			DrawOp::Fill { path, colour }			=> out.fill(res!(path.transform(&t)), *colour),
			DrawOp::Stroke { path, colour, width }	=> out.stroke(res!(path.transform(&t)), *colour, (*width).into()),
			DrawOp::Image { image, x, y, w, h } => {
				// The raster fills its rectangle at the graphic's placement; the PDF writer embeds it as an
				// image XObject, straight RGB with a soft mask only when a sample is translucent.
				let (rgb, alpha) = crate::image::split_rgba(image);
				out.image(
					rgb, alpha, image.width, image.height,
					ox + *x as f64, oy + *y as f64, *w as f64, *h as f64);
			},
		}
	}
	Ok(())
}

/// Draws a placed run as filled glyph outlines. `height` is the face ascent, so `by + height` is the
/// baseline. `bx`/`by` are the box's top-left.
fn draw_text(
	out:	&mut PdfPage,
	bx:		Sp,
	by:		Sp,
	height:	Sp,
	shaped:	&ShapedText,
)
	-> Outcome<()>
{
	let base_x	= bx.to_pt() as f32;
	let base_y	= (by + height).to_pt() as f32;
	for glyph in &shaped.run().glyphs {
		let path = res!(shaped.outline(glyph));
		// A glyph with no ink -- a space -- carries an advance but nothing to fill.
		if path.is_empty() {
			continue;
		}
		// The outline is font-frame, y up; the page is y down. Flip in y, then move onto the baseline
		// at the glyph's own offset. The run is shaped in points, so no scale beyond the flip. The page
		// itself is flipped once more into PDF's y-up frame by the PDF writer, which leaves the glyph
		// the right way up on the page.
		let t = Transform::scale(1.0, -1.0)
			.then(&Transform::translate(base_x + glyph.x, base_y - glyph.y));
		let placed = res!(path.transform(&t));
		out.fill(placed, Rgba::BLACK);
	}
	Ok(())
}
