//! The anchor ledger: the one channel through which a layout fact reaches anything.
//!
//! Stratification (see `sec_decisions.typ`) forbids user code from observing layout directly. What
//! it may see is here: a map from an anchor's identity to the page and position it resolved to. The
//! ledger is filled during composition, is content-addressed by anchor identity, serialises to jdat,
//! and -- in a later phase -- ships inside the Pearl file so the document is queryable without the
//! engine.
//!
//! Content addressing is what buys incremental compilation and what turns a convergence failure into
//! a report: two ledgers can be differenced, and the difference names the anchor that moved and the
//! pages it moved between.

use crate::ir::Sp;

use oxedyne_fe2o3_core::prelude::*;
use oxedyne_fe2o3_jdat::prelude::*;

use std::collections::BTreeMap;

/// The closed vocabulary of things a reference can resolve to. A kind the engine does not know is a
/// limit it declares, not a gap it hides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AnchorKind {
	Label,		// \label{...}, the general cross-reference target
	Heading,	// a section or chapter title, for running heads and the table of contents
	IndexEntry,	// an index term at the point it occurs
	Float,		// a figure or table, placed away from its anchor
	Citation,	// a bibliographic reference
	Equation,	// a numbered display equation
}

impl AnchorKind {
	fn tag(&self) -> u8 {
		match self {
			AnchorKind::Label		=> 0,
			AnchorKind::Heading		=> 1,
			AnchorKind::IndexEntry	=> 2,
			AnchorKind::Float		=> 3,
			AnchorKind::Citation	=> 4,
			AnchorKind::Equation	=> 5,
		}
	}

	fn from_tag(tag: u8) -> Outcome<Self> {
		match tag {
			0 => Ok(AnchorKind::Label),
			1 => Ok(AnchorKind::Heading),
			2 => Ok(AnchorKind::IndexEntry),
			3 => Ok(AnchorKind::Float),
			4 => Ok(AnchorKind::Citation),
			5 => Ok(AnchorKind::Equation),
			_ => Err(err!(
				"Anchor kind tag {} is not one of the six known kinds.", tag; Input, Invalid)),
		}
	}
}

/// An anchor's identity, its kind and a key unique within it. Content-addressed, not positional, so
/// a label keeps its identity when a paragraph moves it to another page.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AnchorId {
	pub kind:	AnchorKind,
	pub key:	String,
}

impl AnchorId {
	pub fn new<S: Into<String>>(kind: AnchorKind, key: S) -> Self {
		Self { kind, key: key.into() }
	}

	/// A stable 64-bit FNV-1a address over kind and key, for the incremental cache. The identity
	/// stays the map key, so a collision costs a comparison, not a wrong answer.
	pub fn address(&self) -> u64 {
		let mut h: u64 = 0xcbf2_9ce4_8422_2325;
		let mix = |h: &mut u64, b: u8| {
			*h ^= b as u64;
			*h = h.wrapping_mul(0x0000_0100_0000_01b3);
		};
		mix(&mut h, self.kind.tag());
		for b in self.key.as_bytes() {
			mix(&mut h, *b);
		}
		h
	}
}

impl ToDat for AnchorId {
	fn to_dat(&self) -> Outcome<Dat> {
		Ok(omapdat!{
			"kind"	=> dat!(self.kind.tag()),
			"key"	=> dat!(self.key.clone()),
		})
	}
}

impl FromDat for AnchorId {
	fn from_dat(mut dat: Dat) -> Outcome<Self> {
		let tag	= try_extract_dat!(res!(dat.map_remove_must(&dat!("kind"))), U8);
		let key	= try_extract_dat!(res!(dat.map_remove_must(&dat!("key"))), Str);
		Ok(Self { kind: res!(AnchorKind::from_tag(tag)), key })
	}
}

/// Where an anchor resolved: the one-based page, and the position of its top-left on that page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Position {
	pub page:	u32,
	pub x:		Sp,
	pub y:		Sp,
}

impl Position {
	pub fn new(page: u32, x: Sp, y: Sp) -> Self {
		Self { page, x, y }
	}
}

/// One resolved anchor. `reserved` is the width a forward reference held open before its value was
/// known, `realised` what the value took; `realised` over `reserved` overflowed the reservation and
/// owes the driver another pass.
#[derive(Clone, Debug)]
pub struct Anchor {
	pub id:			AnchorId,
	pub pos:		Position,
	pub reserved:	Sp,
	pub realised:	Sp,
}

impl Anchor {
	pub fn new(id: AnchorId, pos: Position) -> Self {
		Self { id, pos, reserved: Sp::ZERO, realised: Sp::ZERO }
	}

	/// Did the resolved value outgrow the width held open for it?
	pub fn overflowed(&self) -> bool {
		self.realised > self.reserved
	}
}

impl ToDat for Anchor {
	fn to_dat(&self) -> Outcome<Dat> {
		Ok(omapdat!{
			"id"		=> res!(self.id.to_dat()),
			"page"		=> dat!(self.pos.page),
			"x"			=> res!(self.pos.x.to_dat()),
			"y"			=> res!(self.pos.y.to_dat()),
			"reserved"	=> res!(self.reserved.to_dat()),
			"realised"	=> res!(self.realised.to_dat()),
		})
	}
}

impl FromDat for Anchor {
	fn from_dat(mut dat: Dat) -> Outcome<Self> {
		let id			= res!(AnchorId::from_dat(res!(dat.map_remove_must(&dat!("id")))));
		let page		= try_extract_dat!(res!(dat.map_remove_must(&dat!("page"))), U32);
		let x			= res!(Sp::from_dat(res!(dat.map_remove_must(&dat!("x")))));
		let y			= res!(Sp::from_dat(res!(dat.map_remove_must(&dat!("y")))));
		let reserved	= res!(Sp::from_dat(res!(dat.map_remove_must(&dat!("reserved")))));
		let realised	= res!(Sp::from_dat(res!(dat.map_remove_must(&dat!("realised")))));
		Ok(Self { id, pos: Position::new(page, x, y), reserved, realised })
	}
}

/// One anchor that moved between two ledgers: the identity, and the pages it left and arrived on.
/// A non-empty diff is exactly the convergence-failure report the architecture promises.
#[derive(Clone, Debug)]
pub struct Delta {
	pub id:		AnchorId,
	pub from:	u32,
	pub to:		u32,
}

/// What a forward reference resolves to: a value the previous pass fixed and this pass can read.
/// A closed vocabulary, so a kind the engine cannot resolve is a limit it declares, not a gap it
/// hides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ref {
	TotalPages,			// the document's own page count, the "of M" in "page N of M"
	PageOf(AnchorId),	// the physical page a named anchor resolved to, the general cross-reference
	// The printed folio a named anchor resolved to: its physical page less the front-matter offset, so
	// a table-of-contents entry reads the body folio (which restarts at 1 after the front matter) rather
	// than the physical page it shares with the cover, title and contents leaves.
	FolioOf(AnchorId),
}

impl Ref {
	/// The value this reference resolves to against `incoming`, or `None` when the previous pass has
	/// not fixed it yet -- the empty ledger of Pass A, or an anchor not yet recorded. A composed
	/// ledger always fixes at least one page, so a zero total is the Pass A tell.
	pub fn resolve(&self, incoming: &Ledger) -> Option<u32> {
		match self {
			Ref::TotalPages	=> if incoming.total_pages == 0 { None } else { Some(incoming.total_pages) },
			Ref::PageOf(id)	=> incoming.page_of(id),
			// The body has not been located until a pass has recorded its first heading, so a zero
			// `body_start_page` is the Pass A tell: reserve the slot and defer, exactly as an unresolved
			// page reference does.
			Ref::FolioOf(id) => if incoming.body_start_page == 0 {
				None
			} else {
				incoming.page_of(id).map(|p| p.saturating_sub(incoming.body_start_page - 1))
			},
		}
	}
}

/// The whole anchor table for one composition, plus the total page count the last page fixed.
#[derive(Clone, Debug, Default)]
pub struct Ledger {
	entries:			BTreeMap<AnchorId, Anchor>,
	pub total_pages:	u32,
	// The physical page the body opens on -- the page of the first heading recorded during composition,
	// after the cover, title, imprint and contents leaves the front matter sets. The printed folio
	// restarts at 1 there, so a body page's folio is its physical page less `body_start_page - 1`, and a
	// contents entry's folio is resolved through [`Ref::FolioOf`] the same way. Zero until a heading is
	// recorded, which is the Pass A tell a folio reference reads.
	pub body_start_page:	u32,
	// The physical page the back matter opens on -- the page carrying the bibliography marker. From there
	// the running head is dropped and the folio centres at the foot, as the template sets its back matter.
	// Zero when the document carries no back matter.
	pub back_matter_start_page:	u32,
}

impl Ledger {
	pub fn new() -> Self {
		Self { entries: BTreeMap::new(), total_pages: 0, body_start_page: 0, back_matter_start_page: 0 }
	}

	/// Records an anchor's placement, replacing any earlier record of the same identity within this
	/// pass. The last placement wins because a pass overwrites a stale one as it re-lays the stream.
	///
	/// The first heading recorded fixes where the body opens, and the bibliography marker (the only
	/// Citation-kind anchor) where the back matter does. The front matter sets only Label anchors, so the
	/// first heading to arrive is the body's opening chapter or section -- and it is caught here, as it is
	/// recorded, whether it reaches the driver as a top-level node (a chapter opener) or nested inside a
	/// keep box (a `#section-banner` section's inline level-1 heading, the `DocInline` idiom). Detecting it
	/// only among top-level nodes left the inline idiom with a zero `body_start_page`, so its front-matter
	/// pages were mistaken for body pages and stamped with a folio and footer logo.
	pub fn record(&mut self, anchor: Anchor) {
		if self.body_start_page == 0 && anchor.id.kind == AnchorKind::Heading {
			self.body_start_page = anchor.pos.page;
		}
		if self.back_matter_start_page == 0 && anchor.id.kind == AnchorKind::Citation {
			self.back_matter_start_page = anchor.pos.page;
		}
		self.entries.insert(anchor.id.clone(), anchor);
	}

	pub fn get(&self, id: &AnchorId) -> Option<&Anchor> {
		self.entries.get(id)
	}

	/// The page an anchor resolved to in this ledger, if it is known. A forward reference reads this
	/// from the previous pass's ledger; when it is absent (the first pass) the caller reserves a
	/// width and defers.
	pub fn page_of(&self, id: &AnchorId) -> Option<u32> {
		self.entries.get(id).map(|a| a.pos.page)
	}

	pub fn len(&self) -> usize {
		self.entries.len()
	}

	pub fn is_empty(&self) -> bool {
		self.entries.is_empty()
	}

	/// The anchors whose realised value overflowed its reservation. A non-empty result is why a third
	/// pass is needed.
	pub fn overflowed(&self) -> Vec<AnchorId> {
		self.entries.values().filter(|a| a.overflowed()).map(|a| a.id.clone()).collect()
	}

	/// Every anchor that sits on a different page than it did in `prev`. Ordering by identity makes
	/// the diff deterministic, so a report reads the same on every machine.
	pub fn diff(&self, prev: &Ledger) -> Vec<Delta> {
		let mut out = Vec::new();
		for (id, anchor) in &self.entries {
			if let Some(before) = prev.entries.get(id) {
				if before.pos.page != anchor.pos.page {
					out.push(Delta { id: id.clone(), from: before.pos.page, to: anchor.pos.page });
				}
			}
		}
		out
	}

	/// Has the ledger stopped moving? It is stable against `prev` when the total page count agrees
	/// and no anchor changed page. Position within a page may still differ without forcing another
	/// pass, because only a page change can move a forward reference's page number.
	pub fn is_stable_against(&self, prev: &Ledger) -> bool {
		self.total_pages == prev.total_pages
			&& self.body_start_page == prev.body_start_page
			&& self.back_matter_start_page == prev.back_matter_start_page
			&& self.diff(prev).is_empty()
	}

	/// Writes the ledger to a file as jdat text.
	pub fn to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Outcome<()> {
		let dat	= res!(self.to_dat());
		let cfg	= oxedyne_fe2o3_jdat::string::enc::EncoderConfig::<(), ()>::default();
		let s	= res!(dat.encode_string_with_config(&cfg));
		res!(std::fs::write(path, s));
		Ok(())
	}

	/// Reads a ledger back from a jdat file.
	pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Outcome<Self> {
		let s	= res!(std::fs::read_to_string(path));
		let dat	= res!(Dat::decode_string(s));
		Self::from_dat(dat)
	}
}

impl ToDat for Ledger {
	fn to_dat(&self) -> Outcome<Dat> {
		let mut anchors = Vec::with_capacity(self.entries.len());
		for a in self.entries.values() {
			anchors.push(res!(a.to_dat()));
		}
		Ok(omapdat!{
			"total_pages"			=> dat!(self.total_pages),
			"body_start_page"		=> dat!(self.body_start_page),
			"back_matter_start_page"	=> dat!(self.back_matter_start_page),
			"anchors"				=> Dat::List(anchors),
		})
	}
}

impl FromDat for Ledger {
	fn from_dat(mut dat: Dat) -> Outcome<Self> {
		if dat.kind() != Kind::OrdMap && dat.kind() != Kind::Map {
			return Err(err!(
				"A ledger must decode from a jdat map, found a {:?}.", dat.kind();
				Input, Invalid, Mismatch));
		}
		let total_pages	= try_extract_dat!(res!(dat.map_remove_must(&dat!("total_pages"))), U32);
		// A ledger written before the front-matter offset existed carries no `body_start_page`; default it
		// to zero so an older ledger still decodes.
		let body_start_page = match dat.map_remove(&dat!("body_start_page")) {
			Ok(Some(d))	=> try_extract_dat!(d, U32),
			_			=> 0,
		};
		let back_matter_start_page = match dat.map_remove(&dat!("back_matter_start_page")) {
			Ok(Some(d))	=> try_extract_dat!(d, U32),
			_			=> 0,
		};
		let anchors_dat	= try_extract_dat!(res!(dat.map_remove_must(&dat!("anchors"))), List);
		let mut entries = BTreeMap::new();
		for d in anchors_dat {
			let a = res!(Anchor::from_dat(d));
			entries.insert(a.id.clone(), a);
		}
		Ok(Self { entries, total_pages, body_start_page, back_matter_start_page })
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn record_sets_body_start_on_the_first_heading() {
		// The first heading recorded fixes the body start, whatever page it lands on and however it reached
		// the ledger -- this is what lets a heading nested in a keep box (the inline-heading idiom) fix the
		// body start, where detecting it only among top-level nodes left `body_start_page` zero and stamped
		// the front-matter pages with a folio. A front-matter Label recorded first must not fix it.
		let mut ledger = Ledger::new();
		ledger.record(Anchor::new(
			AnchorId::new(AnchorKind::Label, "frontmatter:contents"),
			Position::new(3, Sp::ZERO, Sp::ZERO)));
		assert_eq!(ledger.body_start_page, 0, "a front-matter Label does not open the body");

		ledger.record(Anchor::new(
			AnchorId::new(AnchorKind::Heading, "01-introduction"),
			Position::new(4, Sp::ZERO, Sp::ZERO)));
		assert_eq!(ledger.body_start_page, 4, "the first heading fixes the body start");

		// A later heading does not move it: the body opens once.
		ledger.record(Anchor::new(
			AnchorId::new(AnchorKind::Heading, "02-server"),
			Position::new(9, Sp::ZERO, Sp::ZERO)));
		assert_eq!(ledger.body_start_page, 4, "a later heading leaves the body start where it was");
	}

	#[test]
	fn record_sets_back_matter_start_on_the_citation_marker() {
		let mut ledger = Ledger::new();
		ledger.record(Anchor::new(
			AnchorId::new(AnchorKind::Citation, "bibliography"),
			Position::new(40, Sp::ZERO, Sp::ZERO)));
		assert_eq!(ledger.back_matter_start_page, 40, "the bibliography marker opens the back matter");
	}
}
