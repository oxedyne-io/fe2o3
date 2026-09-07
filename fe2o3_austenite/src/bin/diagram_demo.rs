//! `diagram_demo` -- draws the compile-loop flowchart and sets it in a real document.
//!
//! Builds the flowchart from the diagram DSL section of the Austenite language document -- Source
//! through Parse, a Types-OK? decision, Lay out and Write Pearl, with a Report-error box off the
//! decision and an orthogonal "fix" edge back to the parser -- wraps it as a captioned figure, and
//! authors a one-page document around it: a heading, a sentence of prose, then the figure. The whole
//! is run through the two-pass driver, decorated with a folio, and written as SVG pages and a PDF, so
//! the flowchart's boxes, routed edges and shaped labels land on a typeset page.
//!
//! Usage: `diagram_demo [OUTPUT_DIR]` (default `diagram-out`).

use oxedyne_fe2o3_austenite::{
	diagram::{
		layout::Route,
		shape::{
			Port,
			Shape,
		},
		Diagram,
		DiagramStyle,
		Endpoint,
	},
	doc::{
		self,
		Block,
		Style,
	},
	driver::{
		self,
		Config,
	},
	emit::Emitter,
	font::FontMetrics,
	ir::{
		Graphic,
		Sp,
	},
	page::PageGeometry,
};

use oxedyne_fe2o3_core::prelude::*;
use oxedyne_fe2o3_font::{
	face::Role,
	set::FontSet,
	shape::Dir,
};

use std::sync::Arc;

fn main() -> Outcome<()> {
	let args: Vec<String> = std::env::args().collect();
	let out_dir = match args.get(1) {
		Some(s)	=> s.clone(),
		None	=> "diagram-out".to_string(),
	};

	let fonts	= Arc::new(res!(oxedyne_fe2o3_austenite::fonts::libertinus()));
	let geom	= PageGeometry::a4();
	let style	= Style::default();

	let graphic	= res!(compile_loop(fonts.clone()));
	println!(
		"diagram_demo: flowchart is {:.1} x {:.1} pt.",
		graphic.dims.width.to_pt(), graphic.dims.height.to_pt());

	let blocks = vec![
		Block::heading(1, "A Flowchart Set on the Page"),
		Block::paragraph(
			"The figure below is drawn in Austenite's diagram sub-language: its boxes and edges are the \
			same outline paths the body text is, and its labels are shaped in the document's own font. \
			It shows the engine's own compile loop, the parser fed again whenever the type check fails."),
		Block::figure(graphic, Some("The compile loop".to_string())),
	];

	let (document, heads) = res!(doc::author(fonts.clone(), geom, style, None, &blocks, None, None));

	let metrics	= FontMetrics::new(fonts.clone(), Role::Body, Dir::Ltr, style.body_size);
	let mut out	= res!(driver::run(&document, &metrics, Config::default()));
	res!(doc::decorate(&mut out.pages, &out.ledger, &heads, &fonts, style, geom, "", None));

	res!(std::fs::create_dir_all(&out_dir));
	let emitter = Emitter::Svg;
	for page in &out.pages {
		let svg		= res!(emitter.render(page));
		let path	= fmt!("{}/page-{:03}.{}", out_dir, page.number, emitter.extension());
		res!(std::fs::write(&path, svg));
	}

	let pdf = res!(oxedyne_fe2o3_austenite::emit::pdf::render_document(&out.pages));
	res!(std::fs::write(fmt!("{}/document.pdf", out_dir), pdf));

	println!(
		"diagram_demo: composed {} page(s) in {} pass(es); wrote SVG page(s) and document.pdf to {}/",
		out.pages.len(), out.passes, out_dir);
	Ok(())
}

/// The compile-loop flowchart, exactly the graph the language document's DSL example draws.
fn compile_loop(fonts: Arc<FontSet>) -> Outcome<Graphic> {
	let down	= Sp::from_pt(30.0);
	let down_lo	= Sp::from_pt(34.0);	// a touch more room beneath the diamond
	let across	= Sp::from_pt(70.0);

	let mut d = Diagram::new();
	d.node_at("src", "Source", Sp::ZERO, Sp::ZERO, Shape::Stadium)
		.node_below("parse", "Parse", "src", down, Shape::Box)
		.node_below("check", "Types OK?", "parse", down, Shape::Diamond)
		.node_below("layout", "Lay out", "check", down_lo, Shape::Box)
		.node_below("pearl", "Write Pearl", "layout", down, Shape::Stadium)
		.node_right("err", "Report error", "check", across, Shape::Box);

	d.edge(Endpoint::node("src"), Endpoint::node("parse"), None, Route::Straight)
		.edge(Endpoint::node("parse"), Endpoint::node("check"), None, Route::Straight)
		.edge(Endpoint::port("check", Port::South), Endpoint::node("layout"), Some("yes"), Route::Straight)
		.edge(Endpoint::port("check", Port::East), Endpoint::port("err", Port::West), Some("no"), Route::Straight)
		.edge(Endpoint::port("err", Port::North), Endpoint::port("parse", Port::East), Some("fix"), Route::Orthogonal)
		.edge(Endpoint::node("layout"), Endpoint::node("pearl"), None, Route::Straight);

	d.build(fonts, &DiagramStyle::default())
}
