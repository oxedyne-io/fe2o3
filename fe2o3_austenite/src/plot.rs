//! Function plots and bar charts as a [`Graphic`](crate::ir::Graphic): axes, ticks, numbered labels and
//! one or more series, drawn in the same `fe2o3_graphics` paths the body text and diagrams use, so a
//! plot is first-class content -- stroked geometry and shaped labels, not a pasted raster. A caller
//! samples a function into a [`Series`] and places the result with [`Block::figure`](crate::doc).
//!
//! The frame is fixed points: a left and bottom margin hold the tick labels, the rest is the plot area.
//! Data coordinates map into that area with y flipped, since the page runs y downwards. Two axis styles
//! serve the books' figures: a framed plot with a light grid, and a bare left-and-bottom axis pair with
//! arrow tips and no grid, the way cetz-plot's `axis-style: "left"` draws. A [`BarChart`] draws
//! horizontal category bars against a bottom value axis, the way cetz-plot's `chart.barchart` does.

use crate::font::ShapedText;
use crate::ir::{
	Dims,
	DrawOp,
	Graphic,
	Sp,
};

use oxedyne_fe2o3_core::prelude::*;
use oxedyne_fe2o3_font::{
	face::Role,
	set::FontSet,
	shape::Dir,
};
use oxedyne_fe2o3_graphics::{
	colour::Rgba,
	path::{
		PathBuilder,
		Pt,
	},
	transform::Transform,
};

use std::sync::Arc;

/// How a plot's axes are drawn. `Framed` boxes the area with a light grid at every tick; `Left` draws
/// only the left and bottom axes as arrow-tipped lines with short tick marks and no grid, matching
/// cetz-plot's `axis-style: "left"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AxisStyle {
	Framed,
	Left,
}

/// One curve: its points in data coordinates, its pen colour and its stroke width in points. A dashed
/// curve is stroked as a run of short segments; a labelled curve appears in the legend.
#[derive(Clone, Debug)]
pub struct Series {
	pub points:	Vec<(f64, f64)>,
	pub colour:	Rgba,
	pub width:	f32,
	pub dashed:	bool,
	pub label:	Option<String>,
}

/// A plot to be built into a figure. The ranges fix the data window; the ticks are where a mark and a
/// numeric label are drawn on each axis (and, in the framed style, a grid line).
#[derive(Clone, Debug)]
pub struct Plot {
	pub width:		f32,		// overall figure width, points
	pub height:		f32,		// overall figure height, points
	pub x_range:	(f64, f64),
	pub y_range:	(f64, f64),
	pub x_ticks:	Vec<f64>,
	pub y_ticks:	Vec<f64>,
	pub series:		Vec<Series>,
	pub axis:		AxisStyle,
	pub x_label:	Option<String>,	// centred beneath the x axis
	pub y_label:	Option<String>,	// set at the top of the y axis
	pub legend:		bool,			// draw a legend from the series labels, inset top-left
}

impl Plot {
	/// Builds the plot into a [`Graphic`] sized to `width` x `height`, its ink drawn from the top-left of
	/// that box in page coordinates so the figure placement can treat it like any other drawn block.
	pub fn build(&self, fonts: Arc<FontSet>) -> Outcome<Graphic> {
		let left_axis	= self.axis == AxisStyle::Left;
		let ml = 34.0_f32;	// left margin, for the y tick labels
		let mb = 22.0_f32;	// bottom margin, for the x tick labels
		let mt = 12.0_f32;	// top margin, room for an axis arrow tip
		let mr = 14.0_f32;	// right margin, likewise
		let pw = self.width - ml - mr;
		let ph = self.height - mt - mb;

		let (x0, x1) = self.x_range;
		let (y0, y1) = self.y_range;
		let sx = |x: f64| -> f32 { ml + ((x - x0) / (x1 - x0)) as f32 * pw };
		let sy = |y: f64| -> f32 { mt + (1.0 - ((y - y0) / (y1 - y0)) as f32) * ph };

		let grid	= Rgba::opaque(225, 225, 225);
		let axis	= Rgba::opaque(120, 120, 120);
		let frame	= Rgba::opaque(90, 90, 90);
		let ink		= Rgba::opaque(30, 30, 30);

		let mut ops: Vec<DrawOp> = Vec::new();

		if left_axis {
			// The two axes as arrow-tipped lines: the y axis up the left, the x axis along the bottom, each
			// running a little past the plot area to carry its arrowhead, and no grid behind them.
			let ax = ml;			// the y axis sits at the left edge of the area
			let ay = mt + ph;		// the x axis sits at the bottom of the area
			let tip = 6.0_f32;
			ops.push(res!(seg(ax, ay, ax, mt - tip, ink, 1.0)));		// y axis, up
			ops.push(res!(seg(ax, ay, ml + pw + tip, ay, ink, 1.0)));	// x axis, right
			ops.push(res!(arrow_tip(ax, mt - tip, 0.0, -1.0, ink)));
			ops.push(res!(arrow_tip(ml + pw + tip, ay, 1.0, 0.0, ink)));
			// A short inward tick at each labelled value.
			for &xt in &self.x_ticks {
				let px = sx(xt);
				ops.push(res!(seg(px, ay, px, ay - 3.0, ink, 0.8)));
			}
			for &yt in &self.y_ticks {
				let py = sy(yt);
				ops.push(res!(seg(ax, py, ax + 3.0, py, ink, 0.8)));
			}
		} else {
			// The framed style: a faint grid at each tick, darker zero axes when zero is in range, and a
			// full frame around the area.
			for &xt in &self.x_ticks {
				let px = sx(xt);
				ops.push(res!(seg(px, mt, px, mt + ph, grid, 0.4)));
			}
			for &yt in &self.y_ticks {
				let py = sy(yt);
				ops.push(res!(seg(ml, py, ml + pw, py, grid, 0.4)));
			}
			if y0 < 0.0 && y1 > 0.0 {
				let py = sy(0.0);
				ops.push(res!(seg(ml, py, ml + pw, py, axis, 0.6)));
			}
			if x0 < 0.0 && x1 > 0.0 {
				let px = sx(0.0);
				ops.push(res!(seg(px, mt, px, mt + ph, axis, 0.6)));
			}
		}

		// The curves, each a stroked polyline through its mapped samples, dashed or solid.
		for s in &self.series {
			let pts: Vec<Pt> = s.points.iter().map(|(x, y)| Pt::new(sx(*x), sy(*y))).collect();
			if s.dashed {
				res!(dash_polyline(&mut ops, &pts, s.colour, s.width));
			} else {
				let mut pb = PathBuilder::new();
				for (k, p) in pts.iter().enumerate() {
					if k == 0 { pb.move_to(*p); } else { pb.line_to(*p); }
				}
				ops.push(DrawOp::Stroke { path: res!(pb.finish()), colour: s.colour, width: s.width });
			}
		}

		if !left_axis {
			ops.push(res!(seg(ml, mt, ml + pw, mt, frame, 0.6)));
			ops.push(res!(seg(ml, mt + ph, ml + pw, mt + ph, frame, 0.6)));
			ops.push(res!(seg(ml, mt, ml, mt + ph, frame, 0.6)));
			ops.push(res!(seg(ml + pw, mt, ml + pw, mt + ph, frame, 0.6)));
		}

		// The tick labels, shaped small and baked to glyph outlines like any other run.
		let size = Sp::from_pt(8.5);
		for &xt in &self.x_ticks {
			let shaped	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, size, &fmt_tick(xt)));
			let w		= shaped.dims().width.to_pt() as f32;
			let asc		= shaped.dims().height.to_pt() as f32;
			res!(bake(&mut ops, &shaped, sx(xt) - w / 2.0, mt + ph + 5.0 + asc));
		}
		for &yt in &self.y_ticks {
			let shaped	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, size, &fmt_tick(yt)));
			let w		= shaped.dims().width.to_pt() as f32;
			let asc		= shaped.dims().height.to_pt() as f32;
			res!(bake(&mut ops, &shaped, ml - 5.0 - w, sy(yt) + asc / 2.0));
		}

		// The axis labels.
		if let Some(xl) = &self.x_label {
			let shaped	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, size, xl));
			let w		= shaped.dims().width.to_pt() as f32;
			res!(bake(&mut ops, &shaped, ml + pw / 2.0 - w / 2.0, self.height - 2.0));
		}
		if let Some(yl) = &self.y_label {
			let shaped	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, size, yl));
			let w		= shaped.dims().width.to_pt() as f32;
			res!(bake(&mut ops, &shaped, ml - w / 2.0, mt - 3.0));
		}

		// The legend, inset a little from the top-left of the plot area: a short line sample of each
		// labelled series, its own dash, and its label to the right.
		if self.legend {
			let lsize	= Sp::from_pt(9.5);
			let lx		= ml + 14.0;
			let mut ly	= mt + 12.0;
			let sample	= 22.0_f32;
			for s in &self.series {
				let label = match &s.label {
					Some(l)	=> l,
					None	=> continue,
				};
				let a = Pt::new(lx, ly);
				let b = Pt::new(lx + sample, ly);
				if s.dashed {
					res!(dash_polyline(&mut ops, &[a, b], s.colour, s.width));
				} else {
					let mut pb = PathBuilder::new();
					pb.move_to(a);
					pb.line_to(b);
					ops.push(DrawOp::Stroke { path: res!(pb.finish()), colour: s.colour, width: s.width });
				}
				let shaped	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, lsize, label));
				let asc		= shaped.dims().height.to_pt() as f32;
				res!(bake(&mut ops, &shaped, lx + sample + 6.0, ly + asc / 2.0));
				ly += 16.0;
			}
		}

		Ok(Graphic {
			ops,
			dims: Dims::new(Sp::from_pt(self.width as f64), Sp::from_pt(self.height as f64), Sp::ZERO),
			link: None,
		})
	}
}

/// A horizontal bar chart: one bar per category, drawn top to bottom, against a bottom value axis with a
/// light dashed grid, matching cetz-plot's `chart.barchart(mode: "basic")`. The category label sits to
/// the left of each bar; the value axis carries numbered ticks and an optional label beneath.
#[derive(Clone, Debug)]
pub struct BarChart {
	pub width:		f32,			// value-axis (plot area) width, points
	pub height:		f32,			// category stack height, points
	pub bars:		Vec<(String, f64)>,	// category label and value, in draw order top to bottom
	pub x_max:		f64,			// the value axis maximum
	pub x_ticks:	Vec<f64>,
	pub x_label:	Option<String>,	// centred beneath the value axis
	pub bar_frac:	f64,			// bar thickness as a fraction of its row slot
	pub fills:		Vec<Rgba>,		// the fill cycled across bars
}

impl BarChart {
	/// Builds the bar chart into a [`Graphic`]. The left margin is sized to the widest category label so
	/// no label is clipped; the bottom margin holds the value ticks and the axis label.
	pub fn build(&self, fonts: Arc<FontSet>) -> Outcome<Graphic> {
		let lsize	= Sp::from_pt(9.5);
		let tsize	= Sp::from_pt(8.5);

		// The left margin follows the widest category label.
		let mut ml = 8.0_f32;
		for (label, _) in &self.bars {
			let shaped	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, lsize, label));
			let w		= shaped.dims().width.to_pt() as f32;
			if w + 10.0 > ml {
				ml = w + 10.0;
			}
		}
		let mb		= if self.x_label.is_some() { 30.0 } else { 20.0 };
		let mt		= 4.0_f32;
		let mr		= 6.0_f32;
		let pw		= self.width;
		let ph		= self.height;
		let total_w	= ml + pw + mr;
		let total_h	= mt + ph + mb;

		let x_max	= if self.x_max > 0.0 { self.x_max } else { 1.0 };
		let sx		= |v: f64| -> f32 { ml + (v / x_max) as f32 * pw };

		let grid	= Rgba::opaque(190, 190, 190);
		let ink		= Rgba::opaque(30, 30, 30);

		let mut ops: Vec<DrawOp> = Vec::new();

		// The dashed vertical grid and the value ticks along the bottom.
		let axis_y = mt + ph;
		for &xt in &self.x_ticks {
			let px = sx(xt);
			res!(dash_polyline(&mut ops, &[Pt::new(px, mt), Pt::new(px, axis_y)], grid, 0.5));
		}

		// Each bar in its row slot, filled from the cycle and outlined.
		let n = self.bars.len().max(1);
		let slot = ph / n as f32;
		let bar_h = (slot * self.bar_frac as f32).max(1.0);
		for (i, (label, value)) in self.bars.iter().enumerate() {
			let cy		= mt + slot * (i as f32 + 0.5);
			let top		= cy - bar_h / 2.0;
			let right	= sx(*value);
			let fill	= self.fills.get(i % self.fills.len().max(1)).copied().unwrap_or(Rgba::opaque(220, 90, 90));
			ops.push(res!(filled_rect(ml, top, right, top + bar_h, fill)));
			ops.push(res!(rect_outline(ml, top, right, top + bar_h, ink, 0.8)));
			// The category label, right-aligned into the left margin, vertically centred on the bar.
			let shaped	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, lsize, label));
			let w		= shaped.dims().width.to_pt() as f32;
			let asc		= shaped.dims().height.to_pt() as f32;
			let dep		= shaped.dims().depth.to_pt() as f32;
			res!(bake(&mut ops, &shaped, ml - 6.0 - w, cy + (asc - dep) / 2.0));
		}

		// The value tick labels beneath the axis.
		for &xt in &self.x_ticks {
			let shaped	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, tsize, &fmt_tick(xt)));
			let w		= shaped.dims().width.to_pt() as f32;
			let asc		= shaped.dims().height.to_pt() as f32;
			res!(bake(&mut ops, &shaped, sx(xt) - w / 2.0, axis_y + 4.0 + asc));
		}
		if let Some(xl) = &self.x_label {
			let shaped	= res!(ShapedText::new(fonts.clone(), Role::Body, Dir::Ltr, tsize, xl));
			let w		= shaped.dims().width.to_pt() as f32;
			res!(bake(&mut ops, &shaped, ml + pw / 2.0 - w / 2.0, total_h - 2.0));
		}

		Ok(Graphic {
			ops,
			dims: Dims::new(Sp::from_pt(total_w as f64), Sp::from_pt(total_h as f64), Sp::ZERO),
			link: None,
		})
	}
}

/// A nice value-axis maximum and tick step for a data maximum: the maximum is taken as the data max, and
/// the step is the "nice" value giving the interval count nearest ten, so a chart of values to 60 ticks
/// every 6 as cetz-plot does. Returns `(max, ticks)`.
pub fn nice_bar_axis(data_max: f64) -> (f64, Vec<f64>) {
	if data_max <= 0.0 {
		return (1.0, vec![0.0, 1.0]);
	}
	let mag		= 10f64.powf(data_max.log10().floor());
	// Candidate step multipliers within a decade; the one whose interval count is nearest ten wins.
	let cands	= [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
	let mut best_step	= mag;
	let mut best_err	= f64::INFINITY;
	for scale in [mag / 10.0, mag, mag * 10.0] {
		for c in cands {
			let step	= c * scale;
			if step <= 0.0 { continue; }
			let count	= data_max / step;
			if count < 4.0 || count > 14.0 { continue; }
			let err		= (count - 10.0).abs();
			if err < best_err {
				best_err = err;
				best_step = step;
			}
		}
	}
	let mut ticks	= Vec::new();
	let mut v		= 0.0;
	while v <= data_max + best_step * 0.001 {
		ticks.push((v * 1e6).round() / 1e6);
		v += best_step;
	}
	(data_max, ticks)
}

/// A straight stroked segment between two points in figure coordinates.
fn seg(x0: f32, y0: f32, x1: f32, y1: f32, colour: Rgba, width: f32) -> Outcome<DrawOp> {
	let mut pb = PathBuilder::new();
	pb.move_to(Pt::new(x0, y0));
	pb.line_to(Pt::new(x1, y1));
	Ok(DrawOp::Stroke { path: res!(pb.finish()), colour, width })
}

/// A filled axis-aligned rectangle.
fn filled_rect(x0: f32, y0: f32, x1: f32, y1: f32, colour: Rgba) -> Outcome<DrawOp> {
	let mut pb = PathBuilder::new();
	pb.move_to(Pt::new(x0, y0));
	pb.line_to(Pt::new(x1, y0));
	pb.line_to(Pt::new(x1, y1));
	pb.line_to(Pt::new(x0, y1));
	pb.close();
	Ok(DrawOp::Fill { path: res!(pb.finish()), colour })
}

/// A stroked axis-aligned rectangle outline.
fn rect_outline(x0: f32, y0: f32, x1: f32, y1: f32, colour: Rgba, width: f32) -> Outcome<DrawOp> {
	let mut pb = PathBuilder::new();
	pb.move_to(Pt::new(x0, y0));
	pb.line_to(Pt::new(x1, y0));
	pb.line_to(Pt::new(x1, y1));
	pb.line_to(Pt::new(x0, y1));
	pb.close();
	Ok(DrawOp::Stroke { path: res!(pb.finish()), colour, width })
}

/// A small filled triangle at `(x, y)` pointing along `(dx, dy)`, the tip an axis arrowhead takes.
fn arrow_tip(x: f32, y: f32, dx: f32, dy: f32, colour: Rgba) -> Outcome<DrawOp> {
	let len		= 6.0_f32;
	let half	= 2.6_f32;
	let (px, py) = (-dy, dx);	// unit perpendicular
	let bx = x - dx * len;
	let by = y - dy * len;
	let mut pb = PathBuilder::new();
	pb.move_to(Pt::new(x, y));
	pb.line_to(Pt::new(bx + px * half, by + py * half));
	pb.line_to(Pt::new(bx - px * half, by - py * half));
	pb.close();
	Ok(DrawOp::Fill { path: res!(pb.finish()), colour })
}

/// Strokes a polyline as a run of short dashes, so a dashed curve reads the same in SVG and PDF without
/// leaning on the emitter's dash support: it walks the line at a fixed dash-and-gap cadence.
fn dash_polyline(ops: &mut Vec<DrawOp>, pts: &[Pt], colour: Rgba, width: f32) -> Outcome<()> {
	let dash	= 4.0_f32;
	let gap		= 3.0_f32;
	let period	= dash + gap;
	let mut phase = 0.0_f32;	// distance into the current period, carried across segments
	for w in pts.windows(2) {
		let (a, b)	= (w[0], w[1]);
		let dx		= b.x - a.x;
		let dy		= b.y - a.y;
		let len		= (dx * dx + dy * dy).sqrt();
		if len <= f32::EPSILON {
			continue;
		}
		let (ux, uy) = (dx / len, dy / len);
		let mut d = 0.0_f32;
		while d < len {
			// Where in the period this point falls: ink while under `dash`, blank after.
			let into	= (phase + d) % period;
			if into < dash {
				let seg_end	= (d + (dash - into)).min(len);
				let mut pb = PathBuilder::new();
				pb.move_to(Pt::new(a.x + ux * d, a.y + uy * d));
				pb.line_to(Pt::new(a.x + ux * seg_end, a.y + uy * seg_end));
				ops.push(DrawOp::Stroke { path: res!(pb.finish()), colour, width });
				d = seg_end;
			} else {
				d += period - into;
			}
		}
		phase = (phase + len) % period;
	}
	Ok(())
}

/// Bakes a shaped run as filled glyph outlines at a baseline, flipping the font-frame y-up outline onto
/// the page's y-down frame, exactly as the diagram labels and the SVG writer do.
fn bake(ops: &mut Vec<DrawOp>, shaped: &ShapedText, base_x: f32, base_y: f32) -> Outcome<()> {
	for glyph in &shaped.run().glyphs {
		let path = res!(shaped.outline(glyph));
		if path.is_empty() {
			continue;
		}
		let t = Transform::scale(1.0, -1.0)
			.then(&Transform::translate(base_x + glyph.x, base_y - glyph.y));
		ops.push(DrawOp::Fill { path: res!(path.transform(&t)), colour: Rgba::BLACK });
	}
	Ok(())
}

/// Formats a tick value compactly: an integer without a decimal point, otherwise to two places with the
/// trailing zeros and any bare point trimmed, so 0.50 shows as 0.5 and 2.0 as 2.
fn fmt_tick(v: f64) -> String {
	if (v - v.round()).abs() < 1e-9 {
		return fmt!("{}", v.round() as i64);
	}
	let s = fmt!("{:.2}", v);
	let s = s.trim_end_matches('0');
	s.trim_end_matches('.').to_string()
}
