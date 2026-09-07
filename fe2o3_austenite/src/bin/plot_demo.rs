//! `plot_demo` -- plots functions as figures and sets them in a real document.
//!
//! Builds two plots with the plotting module -- the sine and cosine over two periods, and the standard
//! normal distribution -- wraps each as a captioned figure, and authors a one-page document around
//! them. The whole is run through the two-pass driver, decorated with a folio, and written as SVG pages
//! and a PDF, so the plotted axes, grid, curves and tick labels land on a typeset page as first-class
//! drawn content.
//!
//! Usage: `plot_demo [OUTPUT_DIR]` (default `plot-out`).

use oxedyne_fe2o3_austenite::{
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
	ir::Graphic,
	page::PageGeometry,
	plot::{
		Plot,
		Series,
	},
};

use oxedyne_fe2o3_core::prelude::*;
use oxedyne_fe2o3_font::{
	face::Role,
	set::FontSet,
	shape::Dir,
};
use oxedyne_fe2o3_graphics::colour::Rgba;

use std::f64::consts::PI;
use std::sync::Arc;

fn main() -> Outcome<()> {
	let args: Vec<String> = std::env::args().collect();
	let out_dir = match args.get(1) {
		Some(s)	=> s.clone(),
		None	=> "plot-out".to_string(),
	};

	let fonts	= Arc::new(res!(oxedyne_fe2o3_austenite::fonts::libertinus()));
	let geom	= PageGeometry::a4();
	let style	= Style::default();

	let waves	= res!(trig_plot(fonts.clone()));
	let bell	= res!(gaussian_plot(fonts.clone()));

	let blocks = vec![
		Block::heading(1, "Two Functions, Plotted"),
		Block::paragraph(
			"Each figure below is a plot built by Austenite's plotting module: the axes, the grid and \
			every curve are stroked outline paths, and the tick labels are shaped in the document's own \
			font, so a plot is drawn geometry a reader can inspect rather than a pasted picture."),
		Block::figure(waves, Some("The sine and cosine over two periods.".to_string())),
		Block::paragraph(
			"The standard normal distribution, the bell curve of the central limit theorem, is drawn the \
			same way from a sampling of its density."),
		Block::figure(bell, Some("The standard normal density.".to_string())),
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
		"plot_demo: composed {} page(s) in {} pass(es); wrote SVG page(s) and document.pdf to {}/",
		out.pages.len(), out.passes, out_dir);
	Ok(())
}

/// The sine and cosine over two full periods, sampled finely enough to read as smooth curves.
fn trig_plot(fonts: Arc<FontSet>) -> Outcome<Graphic> {
	let n		= 240;
	let (a, b)	= (-2.0 * PI, 2.0 * PI);
	let sample	= |f: fn(f64) -> f64| -> Vec<(f64, f64)> {
		(0..=n).map(|k| {
			let x = a + (b - a) * k as f64 / n as f64;
			(x, f(x))
		}).collect()
	};

	let plot = Plot {
		width:		380.0,
		height:		200.0,
		x_range:	(a, b),
		y_range:	(-1.25, 1.25),
		x_ticks:	vec![-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0],
		y_ticks:	vec![-1.0, -0.5, 0.0, 0.5, 1.0],
		series:		vec![
			Series { points: sample(f64::sin), colour: Rgba::opaque(30, 90, 200), width: 1.4, dashed: false, label: None },
			Series { points: sample(f64::cos), colour: Rgba::opaque(200, 60, 50), width: 1.4, dashed: false, label: None },
		],
		axis:		oxedyne_fe2o3_austenite::plot::AxisStyle::Framed,
		x_label:	None,
		y_label:	None,
		legend:		false,
	};
	plot.build(fonts)
}

/// The standard normal density, sampled over four standard deviations each side of the mean.
fn gaussian_plot(fonts: Arc<FontSet>) -> Outcome<Graphic> {
	let n		= 240;
	let (a, b)	= (-4.0, 4.0);
	let norm	= 1.0 / (2.0 * PI).sqrt();
	let points	= (0..=n).map(|k| {
		let x = a + (b - a) * k as f64 / n as f64;
		(x, norm * (-0.5 * x * x).exp())
	}).collect();

	let plot = Plot {
		width:		380.0,
		height:		190.0,
		x_range:	(a, b),
		y_range:	(0.0, 0.45),
		x_ticks:	vec![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
		y_ticks:	vec![0.0, 0.1, 0.2, 0.3, 0.4],
		series:		vec![
			Series { points, colour: Rgba::opaque(120, 40, 160), width: 1.6, dashed: false, label: None },
		],
		axis:		oxedyne_fe2o3_austenite::plot::AxisStyle::Framed,
		x_label:	None,
		y_label:	None,
		legend:		false,
	};
	plot.build(fonts)
}
