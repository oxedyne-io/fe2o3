//! A minimal PDF writer: pages of filled and stroked outline paths.
//!
//! The typesetter above this crate turns every glyph into a filled outline [`Path`] -- the Pearl
//! principle, that a font is a set of outlines and not a program -- so a page is a list of filled
//! paths and nothing else. This writer leans on that entirely: it embeds no font, no CMap and no font
//! program, and writes each glyph as ordinary path-construction and fill operators in a content
//! stream. That is the whole simplification, and it is why the file this module produces is small and
//! self-contained.
//!
//! The geometry and colour are the crate's own [`Path`], [`Pt`], [`Seg`] and [`Rgba`]; nothing here
//! defines a parallel type. A quadratic segment is elevated to a cubic on the way out, since PDF has
//! no quadratic operator, and the whole page is flipped in y so the engine's top-left, y-down frame
//! meets PDF's bottom-left, y-up one.
//!
//! The bytes are deterministic: no dates are written, the `/ID` is derived from the file's own
//! content rather than the clock, and no producer string leaks a version or a build. The same page
//! list yields the same bytes on every run, which is what a content-addressed pipeline needs.
//!
//! [Written with AI entirely](https://need2know.ai/entirely-ai/code)\
//! Anthropic Claude

use crate::colour::Rgba;
use crate::path::{
	Path,
	Pt,
	Seg,
};

use std::io::Write;

use oxedyne_fe2o3_core::prelude::*;

// The FNV-1a parameters the file's deterministic `/ID` is folded with: two bases give the two hash
// halves, and the one prime multiplies each step. Kept as constants so the streaming writer can fold
// the body as it goes rather than hashing a whole buffer at the end.
const FNV_BASIS_A:	u64 = 0xcbf2_9ce4_8422_2325;
const FNV_BASIS_B:	u64 = 0x8422_2325_cbf2_9ce4;
const FNV_PRIME:	u64 = 0x0000_0100_0000_01b3;

/// One drawn shape: a path and how it is painted, or a raster image placed in a rectangle. Fill and
/// stroke are the two the typesetter needs for vector ink -- glyphs and rules fill, a held-open
/// reservation strokes; `Image` embeds a decoded raster (a figure's photograph or diagram) as an image
/// XObject, its samples straight RGB with an optional grey soft mask for translucency.
#[derive(Clone, Debug)]
pub enum Draw {
	Fill {
		path:	Path,
		colour:	Rgba,
	},
	Stroke {
		path:	Path,
		colour:	Rgba,
		width:	f64,	// pen width, in points
	},
	Image {
		rgb:	Vec<u8>,			// packed RGB, iw*ih*3, row-major, top row first
		alpha:	Option<Vec<u8>>,	// packed grey soft mask, iw*ih, present only when a pixel is translucent
		iw:		usize,				// image width in samples
		ih:		usize,				// image height in samples
		x:		f64,				// placement rectangle, engine frame (top-left, y down), points
		y:		f64,
		w:		f64,
		h:		f64,
	},
}

impl Draw {

	/// The paint colour of a vector draw, opaque for an image (whose translucency rides its own soft
	/// mask, not the page's alpha graphics state).
	fn colour(&self) -> Rgba {
		match self {
			Draw::Fill { colour, .. }	=> *colour,
			Draw::Stroke { colour, .. }	=> *colour,
			Draw::Image { .. }			=> Rgba::new(0, 0, 0, 255),
		}
	}
}

/// One page: its size in points, and the shapes drawn on it, back to front.
///
/// The coordinates in the paths are the engine's page frame -- top-left origin, y increasing
/// downwards -- exactly as the SVG writer receives them. [`PdfWriter`] applies the flip to PDF's
/// y-up frame itself, so a caller hands the same paths to either writer.
#[derive(Clone, Debug)]
pub struct PdfPage {
	pub width:	f64,	// media box width, in points
	pub height:	f64,	// media box height, in points
	pub draws:	Vec<Draw>,
}

impl PdfPage {

	pub fn new(width: f64, height: f64) -> Self {
		Self { width, height, draws: Vec::new() }
	}

	pub fn fill(&mut self, path: Path, colour: Rgba) {
		self.draws.push(Draw::Fill { path, colour });
	}

	pub fn stroke(&mut self, path: Path, colour: Rgba, width: f64) {
		self.draws.push(Draw::Stroke { path, colour, width });
	}

	/// Places a decoded raster in the rectangle at top-left `(x, y)`, `w` wide and `h` tall, in the
	/// engine's y-down point frame. `rgb` is `iw * ih * 3` straight-RGB samples, top row first; `alpha`,
	/// when given, is the matching `iw * ih` grey soft mask that carries any translucency. The image is
	/// scaled to fill the rectangle, so the caller sizes the rectangle to the image's aspect if it wants
	/// no distortion.
	#[allow(clippy::too_many_arguments)]
	pub fn image(
		&mut self,
		rgb:	Vec<u8>,
		alpha:	Option<Vec<u8>>,
		iw:		usize,
		ih:		usize,
		x:		f64,
		y:		f64,
		w:		f64,
		h:		f64,
	) {
		self.draws.push(Draw::Image { rgb, alpha, iw, ih, x, y, w, h });
	}

	/// The page's content stream, serialised: the flip matrix and every shape's paint operators, the raw
	/// bytes before any `/Filter` compression. This is the costly half of writing a page -- thousands of
	/// glyph outlines turned into path operators -- and it is a pure function of the page, so a caller
	/// may compute it off the writer's thread and hand it back to [`PdfStream::page_prepared`], which does
	/// only the sequential framing and hashing the bytes then need.
	pub fn content_bytes(&self) -> Vec<u8> {
		content_stream(self).into_bytes()
	}

	/// Drops the draws no longer needed once [`content_bytes`](Self::content_bytes) has been taken: every
	/// opaque vector fill and stroke, whose only remaining use would have been the content stream that is
	/// now serialised. What is kept is exactly what [`PdfStream::page_prepared`] still reads -- the images
	/// (written as XObjects) and any translucent vector draw (whose alpha the page's `/ExtGState` names).
	/// An opaque draw contributes only the alpha 255 that resource set always carries, so dropping it
	/// leaves the written bytes identical; it only frees the glyph outlines a rendered-ahead page would
	/// otherwise hold. Call it after the content bytes are in hand, never before.
	pub fn shed_serialised_draws(&mut self) {
		self.draws.retain(|d| match d {
			Draw::Image { .. }			=> true,
			Draw::Fill { colour, .. }	=> colour.a != 255,
			Draw::Stroke { colour, .. }	=> colour.a != 255,
		});
	}
}

/// One entry in the document outline (the viewer's bookmark side panel): a title, the zero-based page
/// it jumps to, and its nesting depth. Depth zero is a top-level entry; a deeper entry nests under the
/// nearest preceding entry of a shallower depth, so a flat list in reading order builds the tree. The
/// destination is the top of the page fitted to the window, which every entry shares -- the outline
/// names pages, not positions within them.
#[derive(Clone, Debug)]
pub struct OutlineItem {
	pub title:	String,
	pub page:	usize,	// zero-based page index the entry jumps to
	pub level:	u8,		// nesting depth, zero at the top
}

/// The parent, sibling and child links one outline item needs, resolved from the flat level list.
/// Indices are into the item slice; `count` is the number of descendants, always shown open.
struct OutlineLinks {
	parent:	Option<usize>,
	prev:	Option<usize>,
	next:	Option<usize>,
	first:	Option<usize>,
	last:	Option<usize>,
	count:	usize,
}

/// Resolves the flat, reading-order outline list into a tree: each item's parent, siblings and
/// children, and the roots. A stack of open ancestors gives the nearest shallower item as parent, so
/// a level that skips a depth still nests sensibly.
fn build_outline_links(items: &[OutlineItem]) -> (Vec<OutlineLinks>, Vec<usize>) {
	let n = items.len();
	let mut children:	Vec<Vec<usize>>	= vec![Vec::new(); n];
	let mut parent:		Vec<Option<usize>> = vec![None; n];
	let mut roots:		Vec<usize>		= Vec::new();
	let mut stack:		Vec<usize>		= Vec::new();
	for i in 0..n {
		while let Some(&t) = stack.last() {
			if items[t].level >= items[i].level {
				stack.pop();
			} else {
				break;
			}
		}
		match stack.last() {
			Some(&p) => {
				parent[i] = Some(p);
				children[p].push(i);
			},
			None => roots.push(i),
		}
		stack.push(i);
	}

	// The descendant count of a pre-order list is the run of following items whose level stays deeper.
	let mut links = Vec::with_capacity(n);
	for i in 0..n {
		let siblings = match parent[i] {
			Some(p)	=> &children[p],
			None	=> &roots,
		};
		let at		= siblings.iter().position(|&j| j == i).unwrap_or(0);
		let prev	= if at > 0 { Some(siblings[at - 1]) } else { None };
		let next	= siblings.get(at + 1).copied();
		let first	= children[i].first().copied();
		let last	= children[i].last().copied();
		let mut count = 0usize;
		let mut j = i + 1;
		while j < n && items[j].level > items[i].level {
			count += 1;
			j += 1;
		}
		links.push(OutlineLinks { parent: parent[i], prev, next, first, last, count });
	}
	(links, roots)
}

/// A PDF text string for a title: a parenthesised literal with `(`, `)` and `\` escaped when the text
/// is printable ASCII, else a UTF-16BE hex string with a byte-order mark so any Unicode renders. Both
/// forms are deterministic, which the content-addressed file needs.
fn pdf_text_string(s: &str) -> String {
	if s.bytes().all(|b| (0x20..0x7f).contains(&b)) {
		let mut out = String::from("(");
		for c in s.chars() {
			match c {
				'('		=> out.push_str("\\("),
				')'		=> out.push_str("\\)"),
				'\\'	=> out.push_str("\\\\"),
				_		=> out.push(c),
			}
		}
		out.push(')');
		out
	} else {
		let mut out = String::from("<FEFF");
		for u in s.encode_utf16() {
			out.push_str(&fmt!("{:04X}", u));
		}
		out.push('>');
		out
	}
}

/// Accumulates pages and writes them out as one PDF file.
#[derive(Clone, Debug, Default)]
pub struct PdfWriter {
	pages:		Vec<PdfPage>,
	compress:	bool,
	outline:	Vec<OutlineItem>,
}

impl PdfWriter {

	pub fn new() -> Self {
		Self::default()
	}

	/// Compress each content stream with zlib and mark it `/FlateDecode`. Off by default: an
	/// uncompressed stream is trivially deterministic and easy to read while the writer is young.
	pub fn with_compression(mut self, on: bool) -> Self {
		self.compress = on;
		self
	}

	pub fn add_page(&mut self, page: PdfPage) {
		self.pages.push(page);
	}

	/// Sets the document outline (the viewer's bookmark side panel), replacing any earlier one. An empty
	/// list leaves the file with no outline, byte for byte as before the feature existed.
	pub fn set_outline(&mut self, outline: Vec<OutlineItem>) {
		self.outline = outline;
	}

	/// Renders the whole document to PDF bytes.
	///
	/// A convenience over [`PdfStream`]: it streams the accumulated pages into an in-memory buffer and
	/// returns it. The bytes are exactly those [`PdfStream`] writes page by page, so a caller that
	/// cannot hold the whole document keeps the identical file by streaming to a file handle instead.
	pub fn to_bytes(&self) -> Outcome<Vec<u8>> {
		let mut stream = res!(PdfStream::new_with_outline(
			Vec::new(), self.pages.len(), self.compress, self.outline.clone()));
		for page in &self.pages {
			res!(stream.page(page));
		}
		Ok(res!(stream.finish()))
	}
}

/// A PDF writer that emits one page at a time to any [`Write`] sink, holding no more than the page in
/// hand. Where [`PdfWriter`] accumulates every page's outline paths and then serialises them into one
/// buffer -- three live copies of the whole document at the peak -- this writes each page's objects
/// the moment it is handed over and lets the caller drop the page, so a book of any length costs one
/// page of memory. The bytes are identical to [`PdfWriter::to_bytes`]: the same object numbering, the
/// same body order, the same content-derived `/ID`.
///
/// The page count is fixed at construction because the page-tree object -- written first, before any
/// page -- names every page object and their count. The deterministic `/ID` is a hash of the body,
/// folded here as each byte is written rather than over a finished buffer, so no buffer is needed.
pub struct PdfStream<W: Write> {
	out:		W,
	compress:	bool,
	n:			usize,			// the fixed page count, named by the page tree
	offsets:	Vec<usize>,		// one-based object byte offsets; [0] is the free object
	pos:		usize,			// bytes of body written so far, the next object's offset
	added:		usize,			// pages handed over so far
	next_extra:	usize,			// next free object number past the fixed block and the outline, for image XObjects
	hash_a:		u64,			// running FNV-1a of the body, first `/ID` half
	hash_b:		u64,			// running FNV-1a of the body, second half
	outline:	Vec<OutlineItem>,	// document outline entries, empty for none
	outline_root:	usize,		// object number of the /Outlines dict, zero when there is no outline
}

impl<W: Write> PdfStream<W> {
	/// Opens a stream for a document of exactly `n` pages, writing the header, catalogue and page tree
	/// at once. `compress` zlib-compresses each content stream, matching [`PdfWriter::with_compression`].
	pub fn new(out: W, n: usize, compress: bool) -> Outcome<Self> {
		Self::new_with_outline(out, n, compress, Vec::new())
	}

	/// As [`new`](Self::new), but the file also carries a document outline (the viewer's bookmark side
	/// panel). Each entry's page must be one of the `n` promised, since its destination names that page
	/// object. An empty outline yields a file byte-identical to [`new`]'s.
	pub fn new_with_outline(out: W, n: usize, compress: bool, outline: Vec<OutlineItem>) -> Outcome<Self> {
		for it in &outline {
			if it.page >= n {
				return Err(err!(
					"An outline entry points at page {} (zero-based), but the document has only {} \
					page(s); its destination could not be named.", it.page, n; Input, Invalid, Range));
			}
		}
		let obj_count		= 2 + 2 * n;
		let has_outline		= !outline.is_empty();
		// The outline dict and one object per entry sit directly after the fixed page/content block; any
		// image XObject is numbered past them. With no outline the numbering is exactly the original.
		let outline_root	= if has_outline { obj_count + 1 } else { 0 };
		let next_extra		= if has_outline {
			outline_root + 1 + outline.len()
		} else {
			obj_count + 1
		};
		let mut s = Self {
			out,
			compress,
			n,
			offsets:	vec![0; obj_count + 1],
			pos:		0,
			added:		0,
			next_extra,
			hash_a:		FNV_BASIS_A,
			hash_b:		FNV_BASIS_B,
			outline,
			outline_root,
		};

		res!(s.body(b"%PDF-1.7\n"));
		// A comment of high bytes tells a naive tool the file is binary, so it is not mangled in
		// transit. Four bytes above 127, as the specification suggests.
		res!(s.body(b"%\xE2\xE3\xCF\xD3\n"));

		// The catalogue. When the file carries an outline the catalogue names it and asks the viewer to
		// open the bookmark panel; with none it is byte for byte the original.
		s.offsets[1] = s.pos;
		if has_outline {
			let cat = fmt!(
				"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /Outlines {} 0 R /PageMode /UseOutlines >>\nendobj\n",
				outline_root);
			res!(s.body(cat.as_bytes()));
		} else {
			res!(s.body(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"));
		}

		// The page tree, naming every page object up front, which is why `n` is fixed here.
		s.offsets[2] = s.pos;
		let mut kids = String::new();
		for i in 0..n {
			if i > 0 {
				kids.push(' ');
			}
			kids.push_str(&fmt!("{} 0 R", 3 + 2 * i));
		}
		let tree = fmt!("2 0 obj\n<< /Type /Pages /Kids [{}] /Count {} >>\nendobj\n", kids, n);
		res!(s.body(tree.as_bytes()));

		Ok(s)
	}

	/// Writes the next page -- its page object and its content stream -- then advances. The page must
	/// be the next of the `n` promised at construction; an extra page is a mismatch the file could not
	/// name, so it is refused rather than written past the page tree.
	pub fn page(&mut self, page: &PdfPage) -> Outcome<()> {
		let raw = content_stream(page).into_bytes();
		self.page_prepared(page, &raw)
	}

	/// Writes the next page from a content stream already serialised -- by [`PdfPage::content_bytes`],
	/// typically on another thread -- so this call does only the sequential work: assigning object
	/// numbers, framing the page and its stream, writing any images, and folding every byte into the
	/// running `/ID` in page order. The bytes are identical to [`page`](Self::page)'s; the only
	/// difference is where the content stream was built.
	pub fn page_prepared(&mut self, page: &PdfPage, raw: &[u8]) -> Outcome<()> {
		if self.added >= self.n {
			return Err(err!(
				"A PdfStream opened for {} page(s) was handed a further page; the page tree cannot \
				name it.", self.n; Input, Invalid, Excessive));
		}
		let i			= self.added;
		let page_obj	= 3 + 2 * i;
		let content_obj	= 4 + 2 * i;

		// Assign object numbers for every image on the page -- one for the image itself, and one more for
		// its soft mask when it carries translucency -- so the page's resource dictionary can name them
		// before the streams are written. The numbers run past the fixed page/content block, growing the
		// object count a no-image document never touches.
		let mut img_objs: Vec<(usize, Option<usize>)> = Vec::new();
		for d in &page.draws {
			if let Draw::Image { alpha, .. } = d {
				let image_obj = self.next_extra;
				self.next_extra += 1;
				let smask_obj = if alpha.is_some() {
					let m = self.next_extra;
					self.next_extra += 1;
					Some(m)
				} else {
					None
				};
				img_objs.push((image_obj, smask_obj));
			}
		}

		// The content stream was serialised by the caller so its `/Length` is known before the object that
		// wraps it; only the optional compression is left to do here.
		let bytes = if self.compress {
			res!(deflate(raw))
		} else {
			raw.to_vec()
		};

		self.offsets[page_obj] = self.pos;
		let head = fmt!(
			"{} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] {} \
				/Contents {} 0 R >>\nendobj\n",
			page_obj, numf(page.width), numf(page.height), resources(page, &img_objs), content_obj);
		res!(self.body(head.as_bytes()));

		self.offsets[content_obj] = self.pos;
		let filter = if self.compress { " /Filter /FlateDecode" } else { "" };
		let open = fmt!(
			"{} 0 obj\n<< /Length {}{} >>\nstream\n", content_obj, bytes.len(), filter);
		res!(self.body(open.as_bytes()));
		res!(self.body(&bytes));
		res!(self.body(b"\nendstream\nendobj\n"));

		// The image XObjects, in the order their numbers were assigned. The soft mask, when present, is
		// written straight after the image object that references it.
		let mut idx = 0;
		for d in &page.draws {
			if let Draw::Image { rgb, alpha, iw, ih, .. } = d {
				let (image_obj, smask_obj) = img_objs[idx];
				idx += 1;
				res!(self.write_image(image_obj, rgb, *iw, *ih, smask_obj));
				if let (Some(m), Some(a)) = (smask_obj, alpha) {
					res!(self.write_smask(m, a, *iw, *ih));
				}
			}
		}

		self.added += 1;
		Ok(())
	}

	/// Writes an image XObject: a straight-RGB, eight-bit `/DeviceRGB` sample stream, always
	/// zlib-compressed so a photograph does not bloat the file, and pointing at its soft mask when one
	/// was assigned. The samples are folded into the deterministic `/ID` like all body bytes.
	fn write_image(
		&mut self,
		obj:	usize,
		rgb:	&[u8],
		iw:		usize,
		ih:		usize,
		smask:	Option<usize>,
	)
		-> Outcome<()>
	{
		let data = res!(deflate(rgb));
		self.set_extra_offset(obj);
		let mask = match smask {
			Some(m)	=> fmt!(" /SMask {} 0 R", m),
			None	=> String::new(),
		};
		let head = fmt!(
			"{} 0 obj\n<< /Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace /DeviceRGB \
				/BitsPerComponent 8{} /Filter /FlateDecode /Length {} >>\nstream\n",
			obj, iw, ih, mask, data.len());
		res!(self.body(head.as_bytes()));
		res!(self.body(&data));
		res!(self.body(b"\nendstream\nendobj\n"));
		Ok(())
	}

	/// Writes a soft-mask XObject: a single-channel `/DeviceGray` image the same size as its owner, its
	/// samples the straight alpha, zlib-compressed and folded into the `/ID` like any body bytes.
	fn write_smask(&mut self, obj: usize, alpha: &[u8], iw: usize, ih: usize) -> Outcome<()> {
		let data = res!(deflate(alpha));
		self.set_extra_offset(obj);
		let head = fmt!(
			"{} 0 obj\n<< /Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace /DeviceGray \
				/BitsPerComponent 8 /Filter /FlateDecode /Length {} >>\nstream\n",
			obj, iw, ih, data.len());
		res!(self.body(head.as_bytes()));
		res!(self.body(&data));
		res!(self.body(b"\nendstream\nendobj\n"));
		Ok(())
	}

	/// Writes the document outline: the `/Outlines` dictionary, then one object per entry, each a title,
	/// its tree links, and a destination fitting the top of the page it names. The tree is built from the
	/// flat, reading-order entry list by nesting each entry under the nearest preceding shallower one; the
	/// counts are shown open, so a viewer opens the whole tree. The objects take the numbers reserved at
	/// construction, directly after the fixed page/content block.
	fn write_outline(&mut self) -> Outcome<()> {
		let root		= self.outline_root;
		let item_base	= root + 1;	// the first entry's object number
		// Taken out so the entry loop may borrow it while `self` is mutated for each object written.
		let outline		= std::mem::take(&mut self.outline);
		let (links, roots) = build_outline_links(&outline);

		// The `/Outlines` dict: its first and last top-level entries, and the total number of entries,
		// all open.
		self.set_extra_offset(root);
		let first_root	= roots.first().map(|&i| item_base + i).unwrap_or(0);
		let last_root	= roots.last().map(|&i| item_base + i).unwrap_or(0);
		let head = fmt!(
			"{} 0 obj\n<< /Type /Outlines /First {} 0 R /Last {} 0 R /Count {} >>\nendobj\n",
			root, first_root, last_root, outline.len());
		res!(self.body(head.as_bytes()));

		// One object per entry, in reading order so its reserved number matches its slice index.
		for (i, item) in outline.iter().enumerate() {
			let obj		= item_base + i;
			let link	= &links[i];
			let parent	= match link.parent {
				Some(p)	=> item_base + p,
				None	=> root,
			};
			let page_obj = 3 + 2 * item.page;

			let mut dict = fmt!("{} 0 obj\n<< /Title {} /Parent {} 0 R",
				obj, pdf_text_string(&item.title), parent);
			if let Some(p) = link.prev {
				dict.push_str(&fmt!(" /Prev {} 0 R", item_base + p));
			}
			if let Some(nx) = link.next {
				dict.push_str(&fmt!(" /Next {} 0 R", item_base + nx));
			}
			if let (Some(f), Some(l)) = (link.first, link.last) {
				dict.push_str(&fmt!(" /First {} 0 R /Last {} 0 R /Count {}",
					item_base + f, item_base + l, link.count));
			}
			dict.push_str(&fmt!(" /Dest [{} 0 R /Fit] >>\nendobj\n", page_obj));
			self.set_extra_offset(obj);
			res!(self.body(dict.as_bytes()));
		}
		Ok(())
	}

	/// Records the byte offset of an extra object -- an image or soft mask numbered past the fixed
	/// page/content block -- growing the offset table to reach it. The extras are assigned and written in
	/// increasing number order, so the table grows one slot at a time and stays indexed by object number.
	fn set_extra_offset(&mut self, obj: usize) {
		while self.offsets.len() <= obj {
			self.offsets.push(0);
		}
		self.offsets[obj] = self.pos;
	}

	/// Closes the file: writes the cross-reference table and the trailer, flushes, and returns the sink.
	/// The `/ID` is the body hash folded as the body was written, so the file matches
	/// [`PdfWriter::to_bytes`] to the byte.
	pub fn finish(mut self) -> Outcome<W> {
		// The outline objects -- the `/Outlines` dict and one object per entry -- are written after the
		// pages, on the numbers reserved for them at construction. A document with no outline writes none.
		if !self.outline.is_empty() {
			res!(self.write_outline());
		}

		// The fixed page/content block is `2 + 2n` objects; the outline and every image and soft mask took
		// further numbers past it, so the highest object written is one below the next free number. A
		// document with no outline and no image leaves `next_extra` at `2 + 2n + 1`, the original count.
		let obj_count = self.next_extra - 1;

		// The identifier is derived from the body already written, never from the clock. The two halves
		// were folded byte by byte as the body streamed out.
		let id = fmt!("{:016x}{:016x}", self.hash_a, self.hash_b);

		// The cross-reference table and trailer sit after the body and are not part of the hash, so they
		// are written straight to the sink without folding. Every entry is exactly twenty bytes: a
		// ten-digit offset, a five-digit generation, the type, and a two-byte end.
		let xref_off = self.pos;
		let mut tail = String::new();
		tail.push_str(&fmt!("xref\n0 {}\n", obj_count + 1));
		tail.push_str("0000000000 65535 f\r\n");
		for k in 1..=obj_count {
			tail.push_str(&fmt!("{:010} 00000 n\r\n", self.offsets[k]));
		}
		tail.push_str(&fmt!(
			"trailer\n<< /Size {} /Root 1 0 R /ID [<{}> <{}>] >>\nstartxref\n{}\n%%EOF\n",
			obj_count + 1, id, id, xref_off));
		res!(self.out.write_all(tail.as_bytes()));
		res!(self.out.flush());
		Ok(self.out)
	}

	/// Writes a run of body bytes: out to the sink, on to the running offset, and folded into both
	/// halves of the deterministic `/ID`. Only the body passes through here; the xref and trailer, which
	/// the hash excludes, are written directly.
	fn body(&mut self, bytes: &[u8]) -> Outcome<()> {
		res!(self.out.write_all(bytes));
		self.pos += bytes.len();
		for &b in bytes {
			self.hash_a ^= b as u64;
			self.hash_a = self.hash_a.wrapping_mul(FNV_PRIME);
			self.hash_b ^= b as u64;
			self.hash_b = self.hash_b.wrapping_mul(FNV_PRIME);
		}
		Ok(())
	}
}

/// Builds the content stream for one page: the flip, then every shape's colour, path and paint.
///
/// The first operator flips the frame. PDF has its origin at the bottom left with y increasing
/// upwards; the engine places from the top left with y increasing downwards. The matrix `1 0 0 -1 0
/// H` maps `(x, y)` to `(x, H - y)`, so a point at the top of the page (`y = 0`) lands at PDF's `H`
/// and one at the foot (`y = H`) lands at `0`. Applied once as the current transform, it carries the
/// whole page across, and the paths need no per-point flip.
fn content_stream(page: &PdfPage) -> String {
	let mut s = String::new();
	s.push_str(&fmt!("1 0 0 -1 0 {} cm\n", numf(page.height)));

	// A translucent shape needs its alpha set through a graphics state; an all-opaque page needs none,
	// and sets nothing.
	let translucent = page.draws.iter().any(|d| d.colour().a != 255);
	let mut cur_alpha: Option<u8> = None;
	let mut img_k = 0;	// the image index, naming each `/Im{k}` XObject in draw order

	for d in &page.draws {
		match d {
			Draw::Image { x, y, w, h, .. } => {
				// The page CTM already flips y into the engine's top-left frame. An image's sample space
				// paints the unit square with its top row at the square's top, so mapping it into the
				// rectangle at top-left (x, y) needs `w 0 0 -h x (y+h)`: the negative height and the raised
				// origin put the first row at y and the last at y+h. Bracketed in q/Q so it disturbs nothing
				// after it.
				s.push_str("q\n");
				s.push_str(&fmt!("{} 0 0 {} {} {} cm\n", numf(*w), numf(-*h), numf(*x), numf(*y + *h)));
				s.push_str(&fmt!("/Im{} Do\n", img_k));
				s.push_str("Q\n");
				img_k += 1;
			},
			Draw::Fill { path, colour } => {
				if translucent {
					set_alpha(&mut s, &mut cur_alpha, colour.a);
				}
				s.push_str(&fmt!("{} {} {} rg\n",
					chan(colour.r), chan(colour.g), chan(colour.b)));
				path_ops(&mut s, path);
				// Non-zero winding, to match the SVG writer, whose fill-rule defaults to nonzero.
				s.push_str("f\n");
			},
			Draw::Stroke { path, colour, width } => {
				if translucent {
					set_alpha(&mut s, &mut cur_alpha, colour.a);
				}
				s.push_str(&fmt!("{} {} {} RG\n",
					chan(colour.r), chan(colour.g), chan(colour.b)));
				s.push_str(&fmt!("{} w\n", numf(*width)));
				path_ops(&mut s, path);
				s.push_str("S\n");
			},
		}
	}
	s
}

/// Emits the path-construction operators for one path.
///
/// A move is `m`, a line `l`, a cubic `c`, a close `h`. A quadratic has no operator of its own and is
/// elevated to a cubic exactly: the cubic through the same ends whose two controls sit two-thirds of
/// the way from each end towards the quadratic's single control traces the identical curve. The
/// current point is tracked because the elevation needs the segment's start, and a close returns it
/// to where the contour began.
fn path_ops(s: &mut String, path: &Path) {
	let mut cur = Pt::default();
	let mut start = Pt::default();
	for seg in path.segs() {
		match *seg {
			Seg::MoveTo(p) => {
				s.push_str(&fmt!("{} {} m\n", numf32(p.x), numf32(p.y)));
				cur = p;
				start = p;
			},
			Seg::LineTo(p) => {
				s.push_str(&fmt!("{} {} l\n", numf32(p.x), numf32(p.y)));
				cur = p;
			},
			Seg::QuadTo(c, p) => {
				let two_thirds = 2.0 / 3.0;
				let c0 = Pt::new(
					cur.x + two_thirds * (c.x - cur.x),
					cur.y + two_thirds * (c.y - cur.y));
				let c1 = Pt::new(
					p.x + two_thirds * (c.x - p.x),
					p.y + two_thirds * (c.y - p.y));
				s.push_str(&fmt!("{} {} {} {} {} {} c\n",
					numf32(c0.x), numf32(c0.y), numf32(c1.x), numf32(c1.y),
					numf32(p.x), numf32(p.y)));
				cur = p;
			},
			Seg::CubicTo(c0, c1, p) => {
				s.push_str(&fmt!("{} {} {} {} {} {} c\n",
					numf32(c0.x), numf32(c0.y), numf32(c1.x), numf32(c1.y),
					numf32(p.x), numf32(p.y)));
				cur = p;
			},
			Seg::Close => {
				s.push_str("h\n");
				cur = start;
			},
		}
	}
}

/// The page's `/Resources`: an `/ExtGState` for each distinct alpha when a shape is translucent, and an
/// `/XObject` dict naming each image `/Im{k}` by the object number assigned in [`PdfStream::page`]. An
/// all-opaque page with no image carries an empty resource dictionary -- byte for byte the original.
fn resources(page: &PdfPage, img_objs: &[(usize, Option<usize>)]) -> String {
	let translucent = page.draws.iter().any(|d| d.colour().a != 255);

	// The image resource dict, `/Im{k}` in draw order to match the content stream's `Do` names.
	let xobjects = if img_objs.is_empty() {
		String::new()
	} else {
		let mut x = String::from(" /XObject << ");
		for (k, (obj, _)) in img_objs.iter().enumerate() {
			x.push_str(&fmt!("/Im{} {} 0 R ", k, obj));
		}
		x.push_str(">>");
		x
	};

	if !translucent {
		if xobjects.is_empty() {
			return fmt!("/Resources << >>");
		}
		return fmt!("/Resources <<{} >>", xobjects);
	}

	let mut alphas: Vec<u8> = Vec::new();
	for d in &page.draws {
		let a = d.colour().a;
		if !alphas.contains(&a) {
			alphas.push(a);
		}
	}
	// The opaque state is always present, so a shape after a translucent one can return to full
	// opacity.
	if !alphas.contains(&255) {
		alphas.push(255);
	}
	alphas.sort_unstable();
	let mut gs = String::new();
	for a in &alphas {
		let v = chan(*a);
		gs.push_str(&fmt!("/GS{} << /ca {} /CA {} >> ", a, v, v));
	}
	fmt!("/Resources << /ExtGState << {}>>{} >>", gs, xobjects)
}

/// Sets the alpha graphics state, but only when it changes, naming each state `/GSn` by its alpha
/// byte to match [`resources`].
fn set_alpha(s: &mut String, cur: &mut Option<u8>, a: u8) {
	if *cur != Some(a) {
		s.push_str(&fmt!("/GS{} gs\n", a));
		*cur = Some(a);
	}
}

/// One 8-bit channel as a PDF colour component from 0 to 1.
fn chan(c: u8) -> String {
	dec6((c as f64) / 255.0)
}

/// A number to at most six decimal places, trailing zeros trimmed, for a colour or an alpha.
fn dec6(v: f64) -> String {
	let s = fmt!("{:.6}", v);
	let t = s.trim_end_matches('0').trim_end_matches('.');
	if t.is_empty() { "0".to_string() } else { t.to_string() }
}

/// A length or coordinate in its shortest exact decimal. Rust's own float formatting already gives
/// the shortest form that reads back to the same value.
fn numf(v: f64) -> String {
	fmt!("{}", v)
}

fn numf32(v: f32) -> String {
	fmt!("{}", v)
}

/// Zlib-compresses a content stream, for `/FlateDecode`.
///
/// The level is fixed so the output is byte-deterministic for a given input and a given `flate2`
/// version.
fn deflate(raw: &[u8]) -> Outcome<Vec<u8>> {
	use flate2::write::ZlibEncoder;
	use flate2::Compression;
	use std::io::Write;
	let mut enc = ZlibEncoder::new(Vec::new(), Compression::new(6));
	res!(enc.write_all(raw));
	Ok(res!(enc.finish()))
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::path::{
		Bounds,
		PathBuilder,
	};

	#[test]
	fn test_a_quadratic_elevates_to_the_matching_cubic_00() -> Outcome<()> {
		// A quadratic with start (0,0), control (0,10), end (10,10) elevates to a cubic whose controls
		// sit two-thirds of the way from each end towards (0,10): (0, 6.6667) and (3.3333, 10).
		let mut pb = PathBuilder::new();
		pb.move_to(Pt::new(0.0, 0.0));
		pb.quad_to(Pt::new(0.0, 10.0), Pt::new(10.0, 10.0));
		let p = res!(pb.finish());
		let mut s = String::new();
		path_ops(&mut s, &p);
		// The move, then one cubic ending at the quadratic's endpoint.
		assert!(s.contains("0 0 m"), "the move, found: {}", s);
		assert!(s.contains(" c\n"), "a cubic operator, found: {}", s);
		assert!(s.contains("10 10 c"), "the cubic ends where the quadratic did, found: {}", s);
		Ok(())
	}

	#[test]
	fn test_the_file_has_a_header_xref_and_trailer_01() -> Outcome<()> {
		let mut w = PdfWriter::new();
		let mut page = PdfPage::new(100.0, 200.0);
		page.fill(res!(Path::rect(Bounds::new(10.0, 10.0, 90.0, 90.0))), Rgba::BLACK);
		w.add_page(page);
		let bytes = res!(w.to_bytes());
		let text = String::from_utf8_lossy(&bytes);
		assert!(text.starts_with("%PDF-1.7"), "the header");
		assert!(text.contains("/Type /Catalog"), "the catalogue");
		assert!(text.contains("/MediaBox [0 0 100 200]"), "the media box, found in: {}", text);
		assert!(text.contains("1 0 0 -1 0 200 cm"), "the y-flip for a 200pt page");
		assert!(text.contains("xref"), "the cross-reference table");
		assert!(text.contains("startxref"), "the startxref");
		assert!(text.trim_end().ends_with("%%EOF"), "the end-of-file marker");
		Ok(())
	}

	#[test]
	fn test_the_bytes_are_deterministic_02() -> Outcome<()> {
		// The same pages twice give the same bytes: no clock, no random source anywhere in the file.
		let build = || -> Outcome<Vec<u8>> {
			let mut w = PdfWriter::new();
			let mut page = PdfPage::new(100.0, 100.0);
			page.fill(res!(Path::rect(Bounds::new(1.0, 1.0, 9.0, 9.0))), Rgba::new(10, 20, 30, 255));
			w.add_page(page);
			w.to_bytes()
		};
		assert_eq!(res!(build()), res!(build()));
		Ok(())
	}

	#[test]
	fn test_the_outline_nests_by_level_04() -> Outcome<()> {
		// Two top-level entries, the second with a child and a grandchild, then a third top-level entry.
		let items = vec![
			OutlineItem { title: "Title".into(),	page: 0, level: 0 },
			OutlineItem { title: "One".into(),		page: 1, level: 0 },
			OutlineItem { title: "One.a".into(),	page: 2, level: 1 },
			OutlineItem { title: "One.a.i".into(),	page: 3, level: 2 },
			OutlineItem { title: "Two".into(),		page: 4, level: 0 },
		];
		let (links, roots) = build_outline_links(&items);
		assert_eq!(roots, vec![0, 1, 4], "the three top-level entries are the roots");
		// The first entry has no parent, no previous sibling, and "One" as its next.
		assert_eq!(links[0].parent, None);
		assert_eq!(links[0].prev, None);
		assert_eq!(links[0].next, Some(1));
		// "One" parents "One.a", and its descendant count includes the grandchild.
		assert_eq!(links[1].first, Some(2));
		assert_eq!(links[1].last, Some(2));
		assert_eq!(links[1].count, 2, "child and grandchild are both descendants");
		assert_eq!(links[1].next, Some(4), "Two is the next top-level sibling");
		// "One.a" nests under "One" and parents the grandchild.
		assert_eq!(links[2].parent, Some(1));
		assert_eq!(links[2].first, Some(3));
		assert_eq!(links[3].parent, Some(2));
		assert_eq!(links[3].count, 0, "the leaf has no descendants");
		Ok(())
	}

	#[test]
	fn test_the_outline_reaches_the_file_and_dest_pages_05() -> Outcome<()> {
		// A three-page document with a two-entry outline: the catalogue names the outline and the entries
		// carry destinations to their page objects (3 + 2*page).
		let mut w = PdfWriter::new();
		for _ in 0..3 {
			let mut page = PdfPage::new(50.0, 50.0);
			page.fill(res!(Path::rect(Bounds::new(1.0, 1.0, 9.0, 9.0))), Rgba::BLACK);
			w.add_page(page);
		}
		w.set_outline(vec![
			OutlineItem { title: "Title".into(),	page: 0, level: 0 },
			OutlineItem { title: "Body".into(),		page: 2, level: 0 },
		]);
		let bytes = res!(w.to_bytes());
		let text = String::from_utf8_lossy(&bytes);
		assert!(text.contains("/Outlines"), "the catalogue names an outline");
		assert!(text.contains("/Type /Outlines"), "the outline root dict is present");
		assert!(text.contains("/Title (Title)"), "the first entry's title");
		assert!(text.contains("/Title (Body)"), "the second entry's title");
		// Page 0 is object 3, page 2 is object 7.
		assert!(text.contains("/Dest [3 0 R /Fit]"), "the first entry jumps to page object 3");
		assert!(text.contains("/Dest [7 0 R /Fit]"), "the second entry jumps to page object 7");
		Ok(())
	}

	#[test]
	fn test_no_outline_leaves_the_catalogue_untouched_06() -> Outcome<()> {
		// A document with no outline set carries the bare catalogue, byte for byte as before the feature.
		let mut w = PdfWriter::new();
		let mut page = PdfPage::new(50.0, 50.0);
		page.fill(res!(Path::rect(Bounds::new(1.0, 1.0, 9.0, 9.0))), Rgba::BLACK);
		w.add_page(page);
		let bytes = res!(w.to_bytes());
		let text = String::from_utf8_lossy(&bytes);
		assert!(text.contains("<< /Type /Catalog /Pages 2 0 R >>"), "the bare catalogue");
		assert!(!text.contains("/Outlines"), "no outline object when none was set");
		Ok(())
	}

	#[test]
	fn test_a_non_ascii_title_encodes_as_utf16_07() -> Outcome<()> {
		// A title with an em dash cannot be a printable-ASCII literal, so it is a UTF-16BE hex string with
		// a byte-order mark.
		let s = pdf_text_string("A — B");
		assert!(s.starts_with("<FEFF"), "a UTF-16BE hex string, found: {}", s);
		assert!(s.ends_with('>'), "closed hex string");
		// A plain title stays a readable literal.
		assert_eq!(pdf_text_string("Contents"), "(Contents)");
		// Parentheses and backslashes in a literal are escaped.
		assert_eq!(pdf_text_string("a (b) \\ c"), "(a \\(b\\) \\\\ c)");
		Ok(())
	}

	#[test]
	fn test_the_xref_offsets_land_on_their_objects_03() -> Outcome<()> {
		// The heart of a valid PDF: every offset in the cross-reference table must point at the first
		// byte of the object it names. This reads each twenty-byte entry's offset back and confirms the
		// object at that offset opens with "N 0 obj", which catches an off-by-one in the byte
		// accounting. One page gives four objects: catalogue, page tree, page, content.
		let mut w = PdfWriter::new();
		let mut page = PdfPage::new(50.0, 50.0);
		page.fill(res!(Path::rect(Bounds::new(1.0, 1.0, 9.0, 9.0))), Rgba::BLACK);
		w.add_page(page);
		let bytes = res!(w.to_bytes());
		let obj_count = 4;

		// The entries begin after "xref\n" and the "0 M\n" subsection header. The free object is entry
		// zero; objects 1..=obj_count follow, twenty bytes each.
		// Search the raw bytes, not a lossy string: the header's binary-marker comment holds non-UTF-8
		// bytes, so a String index would not line up with the byte offsets the entries are read at.
		let needle = b"xref\n0 ";
		let marker = match bytes.windows(needle.len()).position(|w| w == needle) {
			Some(i) => i,
			None => return Err(err!("no xref section in the file"; Test)),
		};
		let nl1 = match bytes[marker..].iter().position(|&b| b == b'\n') {
			Some(i) => marker + i,
			None => return Err(err!("the xref header is malformed"; Test)),
		};
		let nl2 = match bytes[nl1 + 1..].iter().position(|&b| b == b'\n') {
			Some(i) => nl1 + 1 + i,
			None => return Err(err!("the xref header is malformed"; Test)),
		};
		let entries = &bytes[nl2 + 1..];
		for obj in 1..=obj_count {
			let field = res!(std::str::from_utf8(&entries[obj * 20..obj * 20 + 10]));
			let off: usize = res!(field.parse::<usize>());
			let want = fmt!("{} 0 obj", obj);
			assert!(bytes[off..].starts_with(want.as_bytes()),
				"object {} offset {} does not open with '{}'", obj, off, want);
		}
		Ok(())
	}
}
