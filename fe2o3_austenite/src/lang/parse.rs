//! The Typst reader: line-oriented source into the surface tree of [`ast::Item`](super::ast::Item).
//!
//! The scan is deliberately simple, one pass over the lines. A line whose first non-blank character
//! is `=` is a heading, its level the run of leading `=`; a line opening with `-` or `+` (and a space)
//! is a list item, a run of them a bullet or numbered list; every other non-blank line accumulates into
//! a paragraph. A blank line, a heading, or the start of the other kind of block closes whatever is
//! open. Whitespace within a paragraph is made insignificant here, so the line breaker downstream owns
//! the measure. Byte offsets are tracked across the raw lines so each [`Item`] carries a true [`Span`].
//!
//! A closed paragraph's text is then scanned for inline emphasis by [`parse_inlines`]: `*strong*` and
//! `_emph_`. A delimiter pairs only when it flanks a word, so a stray asterisk, a date's slash, or
//! `and/or` is left as ordinary text rather than opening an emphasis that never closes.
//!
//! A Typst code statement (`#import`/`#let`/`#set`/`#show`) or a line-leading standalone template call
//! (`#name(...)` or `#name[...]`) is not set: it is skipped. When its delimiters do not balance on the
//! opening line -- a `#figure(...)`, `#table(...)`, `#aside-box[...]`, or a `#let x = (...)` data array
//! that spans many lines -- the reader consumes following lines, tracking nesting across `()`, `[]` and
//! `{}` and respecting string literals, until the delimiters balance, so the whole span renders nothing.
//! Every such skip is recorded by name into a [`SkipSummary`] the parse returns beside its items, so a
//! caller reports the constructs it dropped rather than losing them silently. A `#columns(n)[ ... ]`
//! wrapper is the exception the reader does not drop whole: its body is re-parsed and set single-column.
//!
//! Typst comments are stripped before a line is classified: a `//` runs to the line's end, and a
//! `/* ... */` spans lines, both dropped -- except within a `"..."` string or a `` `code` `` span, and a
//! `//` right after `:` is kept, so a bare URL survives. Inline glossary and index calls, defined in the
//! book template (`#gs`, `#gscap`, `#gsi`, `#gscapi`, `#glossind`, `#glossindcap`, the term-dictionary
//! family `#g`, `#gcap`, `#gi`, `#gcapi`, `#t`, `#tcap`, `#graw`, and `#idx`, `#idx-main`, `#idx-as`,
//! `#idx-main-as`, `#index`, `#index-main`, `#idx-nested`), plus a `#link(dest)[text]` hyperlink, are read
//! by [`parse_inlines`]: a glossary term sets its display text, bold-italic on its first document use; a
//! visible term or index call sets its display text plain; a link sets its text and drops the destination;
//! a pure index marker sets nothing. An inline `#func[...]` the reader does not know is consumed, recorded
//! in the summary, and its bracketed body folded in, so its words survive but its raw markup never leaks.

use crate::ir::Length;
use crate::ir::Span;
use crate::table::Align;

use super::ast::{AlignSpec, ClosureAlign, FigureBody, Inline, Item, TableSpec};
use super::mathparse;

use oxedyne_fe2o3_core::prelude::*;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::RwLock;

// The book's `term-dict`, set once by the loader before parsing so the term-dictionary glossary family
// (`t`, `tcap`, `graw`, `g`, `gi`, `gcap`, `gcapi`) resolves a key to its display value while the key
// identity is still known. A process-global rather than a threaded argument, mirroring the image base:
// the inline reader sets one run at a time and carries no book context of its own. `None` until the
// loader installs one, in which case every key falls back to its own text.
static TERM_DICT: RwLock<Option<HashMap<String, String>>> = RwLock::new(None);

/// Records the book's `term-dict`, read from a sibling `terms.typ`, so the term-dictionary glossary
/// family resolves each key to its value at parse time. Installing a fresh map replaces any prior one.
pub fn set_term_dict(dict: HashMap<String, String>) -> Outcome<()> {
	let mut guard = lock_write!(TERM_DICT, "While recording the term dictionary");
	*guard = Some(dict);
	Ok(())
}

/// The display value a `term-dict` key resolves to, or `None` when no map is installed or it holds no
/// such key. A poisoned lock reads as absent rather than failing the parse: a missing value falls back
/// to the key text, which is exactly the safe degradation here.
fn term_value(key: &str) -> Option<String> {
	match TERM_DICT.read() {
		Ok(guard)	=> guard.as_ref().and_then(|m| m.get(key).cloned()),
		Err(_)		=> None,
	}
}

/// A tally of the constructs the reader skipped rather than set, keyed by the source name each was
/// written with (`#show`, `#let`, `#columns`, an unknown `#func`) and counted. A caller prints it so a
/// dropped construct is a visible report rather than a silent gap. Empty when the reader set everything
/// it met. Names carry their leading `#`, so the report reads back as source.
#[derive(Clone, Debug, Default)]
pub struct SkipSummary {
	counts:	BTreeMap<String, usize>,
}

impl SkipSummary {
	/// Records one skipped construct by the source name it was written with (with its leading `#`).
	fn record(&mut self, name: &str) {
		*self.counts.entry(name.to_string()).or_insert(0) += 1;
	}

	pub fn is_empty(&self) -> bool { self.counts.is_empty() }

	/// The number of distinct construct names skipped.
	pub fn kinds(&self) -> usize { self.counts.len() }

	/// The total count of skipped constructs across every name.
	pub fn total(&self) -> usize { self.counts.values().sum() }

	/// Each skipped construct name with its count, ordered by descending count then name, so the report
	/// leads with the construct that cost the most.
	pub fn entries(&self) -> Vec<(String, usize)> {
		let mut v: Vec<(String, usize)> = self.counts.iter().map(|(k, &c)| (k.clone(), c)).collect();
		v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
		v
	}

	/// Folds another summary's counts into this one, so a caller assembling several chapters reports one
	/// total rather than a summary per file.
	pub fn merge(&mut self, other: &SkipSummary) {
		for (k, &c) in &other.counts {
			*self.counts.entry(k.clone()).or_insert(0) += c;
		}
	}

	/// A one-line report -- "skipped 3 unsupported constructs: #show (2), #columns (1)" -- or `None` when
	/// nothing was skipped, so a caller prints the line only when it has something to say.
	pub fn report(&self) -> Option<String> {
		if self.counts.is_empty() {
			return None;
		}
		let parts: Vec<String> = self.entries().into_iter()
			.map(|(n, c)| fmt!("{} ({})", n, c))
			.collect();
		let n = self.total();
		Some(fmt!("skipped {} unsupported construct{}: {}",
			n, if n == 1 { "" } else { "s" }, parts.join(", ")))
	}
}

/// Parses a whole Ingot source string into its surface items. The only error is an empty heading --
/// a `=` marker with no title -- which names the offending 1-based line.
pub fn document(src: &str) -> Outcome<Vec<Item>> {
	let (items, _) = res!(document_with_skips(src));
	Ok(items)
}

/// Parses a source string into its surface items and, alongside, the [`SkipSummary`] of every construct
/// the reader passed over rather than set -- a `#let`/`#set`/`#show`/`#import` code line, an unknown
/// line-leading `#func(...)` call, a `#columns` wrapper, and any unhandled inline `#func[...]`. The
/// caller prints the summary so a dropped construct is reported rather than lost silently.
pub fn document_with_skips(src: &str) -> Outcome<(Vec<Item>, SkipSummary)> {
	let mut skips:		SkipSummary	= SkipSummary::default();
	let mut items:		Vec<Item>	= Vec::new();
	let mut lines:		Vec<String>	= Vec::new();	// the current paragraph's constituent lines
	let mut para_start:	u32			= 0;			// byte offset of the paragraph's first line
	let mut para_end:	u32			= 0;			// byte offset just past its last line's content
	let mut offset:		u32			= 0;			// running byte offset of the current line's start
	let mut line_no					= 0usize;		// 1-based, for a diagnostic

	// The current list, if one is open: its kind, its items so far, and the source it spans. A list is a
	// run of consecutive marker lines; a blank line, a heading, a paragraph line, or a marker of the
	// other kind closes it.
	let mut list:		Vec<Vec<Inline>>	= Vec::new();
	let mut list_ord					= false;
	let mut list_start:	u32				= 0;
	let mut list_end:	u32				= 0;

	// A fenced code block, while one is open: the verbatim lines gathered so far and the byte offset it
	// began at. A ```-fence opens it, the next ```-fence closes it; between them every line is kept as it
	// stands, its indentation and markup untouched.
	let mut code:		Option<(Vec<String>, u32)>	= None;

	// A multi-line Typst code statement or standalone template call being skipped: the net bracket depth
	// still open across the lines consumed so far, and whether a string literal is currently open. `None`
	// when not skipping. While it is `Some`, every line is consumed and nothing is set until the delimiters
	// balance.
	let mut skip:		Option<SkipState>	= None;

	// A multi-line construct whose whole text is gathered so it can be parsed rather than skipped: a
	// `#figure(...)`, a bare `#table(...)`, or a `#let name = (...)` data array feeding a table. `None`
	// when none is open. The accumulated text is dispatched by its kind when the delimiters balance.
	let mut capture:	Option<Capture>		= None;

	// Data arrays declared by `#let name = (...)` and referenced by a table's `..name.flatten()` spread:
	// the name maps to the flat sequence of cells the array holds, each cell a run of inline markup.
	// Populated as the arrays are read, so a later figure resolves its cells against them.
	let mut arrays:		HashMap<String, Vec<Vec<Inline>>>	= HashMap::new();

	// Whether a `/* ... */` block comment is open across the line break. A `//` line comment never
	// straddles a line, so it needs no carried state.
	let mut comment	= CommentState { in_block: false };

	// `split_inclusive` keeps the trailing newline on each piece, so the running offset stays a true
	// byte position into the source rather than drifting by the count of stripped terminators.
	for raw in src.split_inclusive('\n') {
		line_no += 1;
		let start = offset;
		offset = offset.saturating_add(raw.len() as u32);

		// Strip the line terminator without consuming a real character: the final line may carry
		// neither a newline nor a carriage return.
		let mut line = raw;
		if let Some(s) = line.strip_suffix('\n') { line = s; }
		if let Some(s) = line.strip_suffix('\r') { line = s; }
		let end = start.saturating_add(line.len() as u32);

		// Strip Typst comments before classifying the line, but not while a fenced code block or a
		// multi-line call skip is open: inside a fence a `//` is verbatim, and a skipped span is dropped
		// whole regardless. The span above is computed from the raw line, so a diagnostic caret still
		// points into the source.
		let stripped;
		let line = if code.is_none() && skip.is_none() {
			stripped = strip_comments(line, &mut comment);
			stripped.as_str()
		} else {
			line
		};

		let trimmed = line.trim_start();

		// A multi-line code statement or standalone call is being skipped: keep consuming lines, tracking
		// bracket nesting across `()`, `[]` and `{}` and respecting string literals, until the delimiters
		// balance. Nothing between the opener and its close is set. This takes precedence over every other
		// rule, since the span is code, not markup.
		if let Some(state) = skip.as_mut() {
			scan_brackets(line, state);
			if state.depth <= 0 {
				skip = None;
			}
			continue;
		}

		// A multi-line construct is being gathered whole: keep appending its lines and tracking the bracket
		// balance until the delimiters close, then dispatch the accumulated text by its kind. Like the skip
		// above, this takes precedence over the markup rules, since the span is a code construct.
		if let Some(cap) = capture.as_mut() {
			cap.buf.push_str(line);
			cap.buf.push('\n');
			scan_brackets(line, &mut cap.state);
			if cap.state.depth <= 0 {
				let done = capture.take();
				if let Some(cap) = done {
					dispatch_capture(cap, &mut items, &mut arrays, &mut skips);
				}
			}
			continue;
		}

		// A fenced code block takes precedence over every other rule: inside it, only a closing fence is
		// special and every other line is verbatim, so its own `=`, `-` or `*` carry no markup meaning.
		if let Some((buf, cstart)) = code.as_mut() {
			if is_fence(trimmed) {
				items.push(Item::Code { lines: std::mem::take(buf), span: Span::new(*cstart, end) });
				code = None;
			} else {
				buf.push(line.to_string());
			}
			continue;
		}
		if is_fence(trimmed) {
			// An opening fence closes any paragraph or list, then begins a verbatim block. The fence line
			// itself (and any language tag on it) is not kept.
			flush_para(&mut items, &mut lines, para_start, para_end, &mut skips);
			flush_list(&mut items, &mut list, list_ord, list_start, list_end);
			code = Some((Vec::new(), start));
			continue;
		}

		if trimmed.is_empty() {
			// A blank line closes the paragraph it follows, but not an open list: Typst continues an enum
			// (or bullet list) across a blank line between items, restarting the numbering only when other
			// content intervenes. The list is therefore held open here; the marker branch joins a following
			// item of the same kind, while any other line -- a paragraph, heading, figure, fence or code
			// line -- flushes it first, so two lists parted by real content still restart.
			flush_para(&mut items, &mut lines, para_start, para_end, &mut skips);
		} else if let Some(kind) = capture_opener(trimmed) {
			// A multi-line construct the reader sets rather than skips -- a figure, a bare table, or a data
			// array feeding a table. It closes any open block, then its whole text is gathered by the check
			// at the top of the loop until the delimiters balance, and parsed by [`dispatch_capture`].
			flush_para(&mut items, &mut lines, para_start, para_end, &mut skips);
			flush_list(&mut items, &mut list, list_ord, list_start, list_end);
			let mut state	= SkipState { depth: 0, in_string: false };
			scan_brackets(line, &mut state);
			let mut buf		= String::new();
			buf.push_str(line);
			buf.push('\n');
			let cap = Capture { kind, buf, state };
			if cap.state.depth <= 0 {
				dispatch_capture(cap, &mut items, &mut arrays, &mut skips);	// the whole construct closed on one line
			} else {
				capture = Some(cap);
			}
		} else if trimmed.starts_with("#line(") && call_inner(trimmed, "line").is_some() {
			// A standalone `#line(length:.., stroke:..)` horizontal divider (the appendix brackets a note
			// with one above and below). It closes any open block and sets a stroked rule; a multi-line
			// `#line(` that does not close on this line falls through to the skip path below.
			flush_para(&mut items, &mut lines, para_start, para_end, &mut skips);
			flush_list(&mut items, &mut list, list_ord, list_start, list_end);
			if let Some(rule) = parse_line_rule(trimmed) {
				items.push(rule);
			}
		} else if let Some(decision) = code_skip(trimmed) {
			// A Typst code statement (`#import`, `#let`, `#set`, `#show`) or a line-leading standalone call
			// to a template function Austenite does not yet run: it closes any open block and is skipped.
			// The styling and computation layer is a later increment; the prose around it still sets. When
			// its delimiters do not balance on this line, the multi-line span is consumed by the check at the
			// top of the loop until they do.
			flush_para(&mut items, &mut lines, para_start, para_end, &mut skips);
			flush_list(&mut items, &mut list, list_ord, list_start, list_end);
			skips.record(&construct_name(trimmed));
			if let CodeSkip::Multi(state) = decision {
				skip = Some(state);
			}
		} else if trimmed.starts_with('=') {
			// A heading closes any paragraph or list above it, then stands on its own line.
			flush_para(&mut items, &mut lines, para_start, para_end, &mut skips);
			flush_list(&mut items, &mut list, list_ord, list_start, list_end);
			let level = trimmed.chars().take_while(|&c| c == '=').count();
			let raw = trimmed[level..].trim();	// '=' is ASCII, so a byte slice at the count is safe
			if raw.is_empty() {
				return Err(err!(
					"Empty heading on line {}: a `=` marker must be followed by a title.", line_no;
					Input, Invalid, Missing));
			}
			let (title, label) = split_label(raw);
			if title.is_empty() {
				return Err(err!(
					"Heading on line {} has a label but no title.", line_no; Input, Invalid, Missing));
			}
			// The title carries inline markup like any run, so a glossary term, an index call, emphasis or a
			// maths span in a heading sets its display text rather than leaking its raw source into the head
			// and the table of contents.
			items.push(Item::Heading {
				level:	level as u8,
				runs:	parse_inlines_in(&title, &mut skips),
				label,
				span:	Span::new(start, end),
			});
		} else if let Some((ord, text)) = marker(trimmed) {
			// A list item. It closes any open paragraph, and a list of the other kind, but joins a list of
			// its own kind. The item's text carries inline emphasis like any run.
			flush_para(&mut items, &mut lines, para_start, para_end, &mut skips);
			if !list.is_empty() && list_ord != ord {
				flush_list(&mut items, &mut list, list_ord, list_start, list_end);
			}
			if list.is_empty() {
				list_ord	= ord;
				list_start	= start;
			}
			list.push(parse_inlines_in(&text, &mut skips));
			list_end = end;
		} else {
			// Any other non-blank line joins the running paragraph, closing a list first; its own line
			// break and indentation carry no meaning, only its words.
			flush_list(&mut items, &mut list, list_ord, list_start, list_end);
			if lines.is_empty() {
				para_start = start;
			}
			lines.push(line.to_string());
			para_end = end;
		}
	}

	// A source that ends without a closing blank line still closes its last paragraph or list; an
	// unterminated code fence still yields the block it had gathered.
	flush_para(&mut items, &mut lines, para_start, para_end, &mut skips);
	flush_list(&mut items, &mut list, list_ord, list_start, list_end);
	if let Some((buf, cstart)) = code {
		items.push(Item::Code { lines: buf, span: Span::new(cstart, offset) });
	}
	// A construct left open at end of source is dispatched with what it gathered, so a missing closer
	// still yields its best-effort figure or table rather than swallowing the tail silently.
	if let Some(cap) = capture {
		dispatch_capture(cap, &mut items, &mut arrays, &mut skips);
	}
	Ok((items, skips))
}

/// Is this already-left-trimmed line a ```` ``` ```` code fence? An opening fence may carry a language
/// tag (```` ```rust ````); a closing fence is bare. Either way it opens with three backticks.
fn is_fence(trimmed: &str) -> bool {
	trimmed.starts_with("```")
}

/// Reads a list marker at the start of an already-left-trimmed line: `-` opens a bullet item, `+` a
/// numbered one. The marker must be the whole line or be followed by whitespace, so a dash inside a word
/// or a `+1` is ordinary prose, not a marker. Returns the item's kind and its text with the marker and
/// surrounding whitespace removed.
fn marker(trimmed: &str) -> Option<(bool, String)> {
	let first	= trimmed.chars().next()?;
	let ordered	= match first {
		'-'	=> false,
		'+'	=> true,
		_	=> return None,
	};
	let rest = &trimmed[first.len_utf8()..];
	if rest.is_empty() {
		return Some((ordered, String::new()));
	}
	if rest.starts_with(|c: char| c.is_whitespace()) {
		return Some((ordered, rest.trim().to_string()));
	}
	None
}

/// Closes the list being accumulated, if any, into one [`Item::List`]. An empty accumulator flushes
/// nothing, so a stray flush between two paragraphs costs nothing.
fn flush_list(
	items:		&mut Vec<Item>,
	list:		&mut Vec<Vec<Inline>>,
	ordered:	bool,
	start:		u32,
	end:		u32,
)
{
	if list.is_empty() {
		return;
	}
	items.push(Item::List { ordered, items: std::mem::take(list), span: Span::new(start, end) });
}

/// Closes the paragraph being accumulated, if any: its lines are joined, their whitespace collapsed,
/// and the result pushed as one [`Item::Paragraph`] spanning the source it came from. An empty
/// accumulator flushes nothing, so a run of blank lines closes a paragraph only once.
fn flush_para(
	items:	&mut Vec<Item>,
	lines:	&mut Vec<String>,
	start:	u32,
	end:	u32,
	skips:	&mut SkipSummary,
)
{
	if lines.is_empty() {
		return;
	}
	let text = normalise_ws(&lines.join(" "));
	// A trailing `<name>` labels the block -- in practice a display equation, `$ ... $ <eq_x>` -- and is
	// stripped before the runs are read, so the maths span stands alone and lowers to a numbered equation
	// rather than a rich paragraph. Ordinary prose ends in a full stop, so the conservative `split_label`
	// (a single whitespace-free token in angle brackets at the very end) does not fire on it.
	let (body, label) = split_label(&text);
	let runs = parse_inlines_in(&body, skips);
	items.push(Item::Paragraph { runs, label, span: Span::new(start, end) });
	lines.clear();
}

/// Collapses every run of whitespace to a single space and trims the ends, so a paragraph's set width
/// is left to the line breaker rather than to the source's own line breaks and indentation.
fn normalise_ws(s: &str) -> String {
	s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Splits a whitespace-collapsed paragraph into inline runs in Typst's markup: `*strong*`, `_emph_`, a
/// `@label` cross-reference, and `\`-escapes. An emphasis delimiter pairs only when it flanks a word --
/// whitespace or an opening bracket before it and a non-space after to open, the reverse to close -- so
/// `fe2o3_net`, `5 * 3` and a lone `_` are ordinary text. A backslash sets the next character literally,
/// so `\$`, `\#`, `\_` and `\@` appear as themselves. An unpaired delimiter, or an `@` with no label
/// after it, is ordinary text. Nesting is a later increment: the first valid closer ends a run.
fn parse_inlines(text: &str) -> Vec<Inline> {
	let mut skips = SkipSummary::default();
	parse_inlines_in(text, &mut skips)
}

/// The inline scanner proper, recording every unhandled inline call into `skips` so a `#func[...]` the
/// reader cannot set is reported rather than leaked into the running text. [`parse_inlines`] is the thin
/// wrapper for callers -- table cells, captions, flattening -- that do not surface the summary.
fn parse_inlines_in(text: &str, skips: &mut SkipSummary) -> Vec<Inline> {
	let chars:	Vec<char>	= text.chars().collect();
	let n					= chars.len();
	let mut runs:	Vec<Inline>	= Vec::new();
	let mut plain			= String::new();	// ordinary text gathered before the next run
	let mut i				= 0usize;
	while i < n {
		let c = chars[i];
		// A backslash escapes the next character, which is then set as itself.
		if c == '\\' && i + 1 < n {
			plain.push(chars[i + 1]);
			i += 2;
			continue;
		}
		// An inline maths span between dollars. A `\$` was already turned into a literal above, so a `$`
		// reaching here opens maths. If it parses, it is a maths run; if not, the literal `$...$` is kept.
		if c == '$' {
			if let Some(close) = (i + 1..n).find(|&j| chars[j] == '$') {
				let inner: String = chars[i + 1..close].iter().collect();
				if let Ok(atom) = mathparse::parse(&inner) {
					if !plain.is_empty() {
						runs.push(Inline::Text(std::mem::take(&mut plain)));
					}
					runs.push(Inline::Math(atom));
					i = close + 1;
					continue;
				}
			}
		}
		// An inline code span, `raw` between backticks: its content is verbatim, no markup within.
		if c == '`' {
			if let Some(close) = (i + 1..n).find(|&j| chars[j] == '`') {
				if !plain.is_empty() {
					runs.push(Inline::Text(std::mem::take(&mut plain)));
				}
				runs.push(Inline::Code(chars[i + 1..close].iter().collect()));
				i = close + 1;
				continue;
			}
		}
		// An inline footnote. Its bracketed content is markup, carried as inline runs so the note sets its
		// own emphasis at the foot of the page. The mark falls after the run before it.
		if c == '#' {
			if let Some((note, next)) = footnote_call(&chars, i, skips) {
				if !plain.is_empty() {
					runs.push(Inline::Text(std::mem::take(&mut plain)));
				}
				runs.push(Inline::Footnote(note));
				i = next;
				continue;
			}
		}
		// The function-call form of emphasis, `#emph[...]`, set exactly as `_..._`: its bracketed content is
		// markup, so it takes the same expansion, and a call nested within it renders its display text.
		if c == '#' {
			if let Some((inner, next)) = emph_call(&chars, i) {
				if !plain.is_empty() {
					runs.push(Inline::Text(std::mem::take(&mut plain)));
				}
				push_emphasis(&mut runs, false, &inner, skips);
				i = next;
				continue;
			}
		}
		// Typst's superscript, `#super[...]` or `#super("...")`. Its content is usually a short string or
		// number, reduced to display text here and set raised and smaller by the block layer.
		if c == '#' {
			if let Some((text, next)) = super_call(&chars, i) {
				if !plain.is_empty() {
					runs.push(Inline::Text(std::mem::take(&mut plain)));
				}
				runs.push(Inline::Super(text));
				i = next;
				continue;
			}
		}
		// An inline glossary or index call defined in the book template. A glossary term is a run of its
		// own, so [`doc::author`] can set it bold-italic on first use; a visible index call sets its
		// display text, which may itself carry markup, so it is parsed and folded in; a pure index marker
		// sets nothing.
		if c == '#' {
			if let Some((call, next)) = glossary_call(&chars, i, skips) {
				match call {
					Call::Glossary { term, display } => {
						if !plain.is_empty() {
							runs.push(Inline::Text(std::mem::take(&mut plain)));
						}
						runs.push(Inline::Glossary { term, display });
					},
					Call::Visible(display) => {
						let sub = parse_inlines_in(&display, skips);
						// A plain display folds back into the running text, keeping the fast single-run
						// path; a display carrying markup becomes its own runs.
						if let [Inline::Text(t)] = sub.as_slice() {
							plain.push_str(t);
						} else {
							if !plain.is_empty() {
								runs.push(Inline::Text(std::mem::take(&mut plain)));
							}
							runs.extend(sub);
						}
					},
					Call::Invisible => {},	// a pure index marker sets nothing
				}
				i = next;
				continue;
			}
		}
		// An inline citation, `#cite(<key>)` or `#cite(<a>, <b>)`. Its keys become a cite run the block
		// layer resolves to "(Author Year)" against the bibliography; a citation with no readable key is
		// dropped rather than left as raw source.
		if c == '#' {
			if let Some((keys, next)) = cite_call(&chars, i) {
				if !keys.is_empty() {
					if !plain.is_empty() {
						runs.push(Inline::Text(std::mem::take(&mut plain)));
					}
					runs.push(Inline::Cite(keys));
				}
				i = next;
				continue;
			}
		}
		// An inline claim marker, `#claim-label(...)` or `#claim-refs(...)`, from the book's claims
		// machinery. A `claim-label` registers invisible metadata and sets a small code in the outside
		// margin; a `claim-refs` registers metadata only. Neither places anything in the body text column,
		// so the marker is consumed and nothing set where it stood, matching Typst's body flow -- the
		// marginal code is not reproduced, the page having a single text column and no margin placement.
		if c == '#' {
			if let Some(next) = claim_call(&chars, i) {
				i = next;
				continue;
			}
		}
		// The `#raw("...")` call form of inline code.
		if c == '#' {
			if let Some((text, next)) = raw_call(&chars, i) {
				if !plain.is_empty() {
					runs.push(Inline::Text(std::mem::take(&mut plain)));
				}
				runs.push(Inline::Code(text));
				i = next;
				continue;
			}
		}
		// A Typst hyperlink, `#link("url")[text]` or `#link(<label>)[text]`. The link text is what a print
		// reader sees, so its markup is parsed and folded into the running line; the destination has no place
		// on a page with no clickable annotation and is dropped. A `#link("url")` with no bracket sets the
		// URL itself as its text, as Typst does.
		if c == '#' {
			if let Some((body, next)) = link_call(&chars, i, skips) {
				if let [Inline::Text(t)] = body.as_slice() {
					plain.push_str(t);
				} else {
					if !plain.is_empty() {
						runs.push(Inline::Text(std::mem::take(&mut plain)));
					}
					runs.extend(body);
				}
				i = next;
				continue;
			}
		}
		// A Typst cross-reference: `@` then a label.
		if c == '@' {
			if let Some((label, next)) = at_label(&chars, i) {
				if !plain.is_empty() {
					runs.push(Inline::Text(std::mem::take(&mut plain)));
				}
				runs.push(Inline::PageRef(label));
				i = next;
				continue;
			}
		}
		if (c == '*' || c == '_') && is_opener(&chars, i) {
			if let Some(close) = find_closer(&chars, i + 1, c) {
				if !plain.is_empty() {
					runs.push(Inline::Text(std::mem::take(&mut plain)));
				}
				let inner: String = chars[i + 1..close].iter().collect();
				push_emphasis(&mut runs, c == '*', &inner, skips);
				i = close + 1;
				continue;
			}
		}
		// An inline `#func(...)` or `#func[...]` call none of the handlers above claimed: a template function
		// the reader cannot yet run. It is recorded by name and consumed whole, so its raw markup no longer
		// leaks into the set text. When it wraps a single `[...]` content group -- the common shape of a Typst
		// content function -- that body is parsed and folded in, keeping its words rather than dropping them;
		// a call with only paren arguments (`#v(1em)`, `#colbreak()`) sets nothing where it stood.
		if c == '#' {
			if let Some((body, next, name)) = unknown_call(&chars, i, skips) {
				skips.record(&name);
				if let Some(body) = body {
					if let [Inline::Text(t)] = body.as_slice() {
						plain.push_str(t);
					} else {
						if !plain.is_empty() {
							runs.push(Inline::Text(std::mem::take(&mut plain)));
						}
						runs.extend(body);
					}
				}
				i = next;
				continue;
			}
		}
		plain.push(c);
		i += 1;
	}
	if !plain.is_empty() {
		runs.push(Inline::Text(plain));
	}
	if runs.is_empty() {
		runs.push(Inline::Text(String::new()));	// a paragraph of pure delimiters keeps one empty run
	}
	runs
}

/// Pushes an emphasised run (`*strong*` when `strong`, else `_emph_`) onto `runs`, reading its inner
/// markup. When the inner is plain text the run keeps the fast flat path -- one [`Inline::Strong`] or
/// [`Inline::Emph`]. When it carries a glossary term, an index call or a maths span -- as
/// `*The captation #gsi[attractor]*` does -- the emphasis is expanded: its plain stretches take the
/// emphasis face and the embedded calls become their own runs, so a call nested in emphasis renders its
/// display text rather than leaking its raw source. A glossary term keeps its own first-use bold-italic
/// (which subsumes the surrounding emphasis), so only the plain stretches carry the emphasis face.
fn push_emphasis(runs: &mut Vec<Inline>, strong: bool, inner: &str, skips: &mut SkipSummary) {
	let sub = parse_inlines_in(inner, skips);
	if let [Inline::Text(t)] = sub.as_slice() {
		runs.push(if strong { Inline::Strong(t.clone()) } else { Inline::Emph(t.clone()) });
		return;
	}
	for run in sub {
		match run {
			// A plain stretch takes the emphasis face.
			Inline::Text(t)					=> runs.push(if strong { Inline::Strong(t) } else { Inline::Emph(t) }),
			// An inner run of the opposite face (`*_word_*`, `_*word*_`) multiplies the two faces to a
			// bold-italic run rather than losing the outer; a run of the same face, a glossary term or a
			// maths span keeps its own face, one level of nesting being all the flat vocabulary carries.
			Inline::Emph(t) if strong		=> runs.push(Inline::BoldItalic(t)),
			Inline::Strong(t) if !strong	=> runs.push(Inline::BoldItalic(t)),
			other							=> runs.push(other),
		}
	}
}

/// Does the delimiter at `i` flank the left of a word? A non-space must follow it, and the start of the
/// paragraph, whitespace, or an opening bracket must precede it.
fn is_opener(chars: &[char], i: usize) -> bool {
	match chars.get(i + 1) {
		Some(c) if !c.is_whitespace()	=> {},
		_								=> return false,
	}
	match i.checked_sub(1).and_then(|p| chars.get(p)) {
		None		=> true,
		Some(&p)	=> p.is_whitespace() || matches!(p, '(' | '[' | '{' | '"' | '\''),
	}
}

/// Does the delimiter at `j` flank the right of a word? A non-space must precede it, and the end of the
/// paragraph, whitespace, or closing punctuation must follow.
fn is_closer(chars: &[char], j: usize) -> bool {
	match j.checked_sub(1).and_then(|p| chars.get(p)) {
		Some(p) if !p.is_whitespace()	=> {},
		_								=> return false,
	}
	match chars.get(j + 1) {
		None		=> true,
		Some(&c)	=> c.is_whitespace()
			|| matches!(c, ')' | ']' | '}' | '.' | ',' | ';' | ':' | '!' | '?' | '"' | '\''),
	}
}

/// The index of the first valid closing `delim` at or after `start`, or `None` when the run never
/// closes -- in which case the opener is ordinary text.
fn find_closer(chars: &[char], start: usize, delim: char) -> Option<usize> {
	(start..chars.len()).find(|&j| chars[j] == delim && is_closer(chars, j))
}

/// Reads a Typst cross-reference at `i` (an `@`): the label of letters, digits and `- _ :` that follows,
/// and the index just past it. `None` when no label char follows, so a bare or escaped `@` is ordinary
/// text. A trailing `.` is not a label character, so `@intro.` at the end of a sentence keeps its stop.
fn at_label(chars: &[char], i: usize) -> Option<(String, usize)> {
	let start	= i + 1;
	let mut j	= start;
	while j < chars.len() && is_label_char(chars[j]) {
		j += 1;
	}
	if j == start {
		return None;
	}
	Some((chars[start..j].iter().collect(), j))
}

/// A character legal within a Typst label. Deliberately excludes `.`, so a label does not swallow the
/// full stop that ends a sentence.
fn is_label_char(c: char) -> bool {
	c.is_alphanumeric() || matches!(c, '-' | '_' | ':')
}

/// Reads an inline `#raw("...")` at `i`, returning its literal content and the index past the closing
/// `")`. `None` when the shape does not match, so a `#raw` written any other way is left as ordinary
/// text. Escaped quotes inside the string are not handled -- a later refinement.
fn raw_call(chars: &[char], i: usize) -> Option<(String, usize)> {
	let open	= at_lit(chars, i, "#raw(\"")?;
	let close	= (open..chars.len()).find(|&j| chars[j] == '"')?;
	if chars.get(close + 1) != Some(&')') {
		return None;
	}
	Some((chars[open..close].iter().collect(), close + 2))
}

/// Reads an inline `#link(dest)[text]` (or a bare `#link(dest)`) at `i` (a `#`), returning the link's
/// display runs and the index just past it. The destination -- a `"url"` string or a `<label>` -- is read
/// and discarded, the page carrying no clickable annotation; the bracketed text is what the reader sees,
/// so it is parsed for its own markup. A `#link(dest)` with no following `[...]` sets the destination
/// string itself as its text, as Typst does. Any unhandled inline call within the text is recorded into
/// `skips`. `None` when the shape is not a link call or its arguments do not close.
fn link_call(chars: &[char], i: usize, skips: &mut SkipSummary) -> Option<(Vec<Inline>, usize)> {
	let Some(open) = at_lit(chars, i, "#link") else { return None; };
	if chars.get(open) != Some(&'(') {
		return None;
	}
	let Some((dest, after_dest)) = read_group(chars, open) else { return None; };
	// A following `[...]` group is the link text; without one, the destination stands as the text.
	if chars.get(after_dest) == Some(&'[') {
		let Some((body, next)) = read_group(chars, after_dest) else { return None; };
		return Some((parse_inlines_in(&body, skips), next));
	}
	let text = link_dest_text(&dest);
	Some((vec![Inline::Text(text)], after_dest))
}

/// The display text of a bare `#link(dest)` with no bracketed body: a `"url"` string loses its quotes, a
/// `<label>` its angle brackets, and anything else stands as written.
fn link_dest_text(dest: &str) -> String {
	let t = dest.trim();
	if let Some(inner) = t.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
		return inner.to_string();
	}
	unwrap_arg(t)
}

/// Reads an inline `#name(...)`/`#name[...]` call at `i` (a `#`) that no earlier handler claimed, so the
/// reader can consume it whole rather than leak its raw markup. Returns the bracketed body's runs (parsed,
/// so its own markup survives) when the call is a single `[...]` content group, `None` for the body when it
/// carries only paren arguments, together with the index just past the call and its `#name` for the skip
/// report. The final `None` is returned when `i` does not open a `#name(`/`#name[` call at all, so a bare
/// `#` or a `#variable` interpolation is left as ordinary text.
fn unknown_call(chars: &[char], i: usize, skips: &mut SkipSummary)
	-> Option<(Option<Vec<Inline>>, usize, String)>
{
	if chars.get(i) != Some(&'#') {
		return None;
	}
	let start	= i + 1;
	let mut j	= start;
	while j < chars.len() && (chars[j].is_ascii_alphanumeric() || chars[j] == '-' || chars[j] == '_' || chars[j] == '.') {
		j += 1;
	}
	if j == start {
		return None;
	}
	let name: String = chars[start..j].iter().collect();
	match chars.get(j) {
		// A `#name[body]`: the bracketed content is the call's displayable body.
		Some('[') => {
			let Some((body, next)) = read_group(chars, j) else { return None; };
			Some((Some(parse_inlines_in(&body, skips)), next, fmt!("#{}", name)))
		},
		// A `#name(args)` and any following `[body]`: read the arguments away, then fold a body if one trails.
		Some('(') => {
			let Some((_, after_args)) = read_group(chars, j) else { return None; };
			if chars.get(after_args) == Some(&'[') {
				let Some((body, next)) = read_group(chars, after_args) else { return None; };
				return Some((Some(parse_inlines_in(&body, skips)), next, fmt!("#{}", name)));
			}
			Some((None, after_args, fmt!("#{}", name)))
		},
		_ => None,
	}
}

/// The `#name` of a skipped line-leading code statement or standalone call, for the skip report: the
/// keyword itself for a block statement (`#let`, `#set`, `#show`, `#import`), or `#` and the identifier of
/// a standalone call. Falls back to the first whitespace-delimited token when neither shape reads, so the
/// tally always names something rather than nothing.
fn construct_name(trimmed: &str) -> String {
	for kw in ["#import", "#let", "#set", "#show"] {
		if trimmed.starts_with(kw) {
			return kw.to_string();
		}
	}
	let mut cs = trimmed.chars();
	if cs.next() == Some('#') {
		let mut ident = String::new();
		for c in cs {
			if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
				ident.push(c);
			} else {
				break;
			}
		}
		if !ident.is_empty() {
			return fmt!("#{}", ident);
		}
	}
	trimmed.split_whitespace().next().unwrap_or(trimmed).to_string()
}

/// If the literal `s` sits at `i` in `chars`, the index just past it; otherwise `None`.
fn at_lit(chars: &[char], i: usize, s: &str) -> Option<usize> {
	let mut k = i;
	for ch in s.chars() {
		if chars.get(k) != Some(&ch) {
			return None;
		}
		k += 1;
	}
	Some(k)
}

/// The running bracket balance while a multi-line code statement or template call is skipped: `depth`
/// is the net count of unclosed `()`, `[]` and `{}` openers, and `in_string` records whether a `"..."`
/// literal is currently open so a bracket inside it does not count. Both persist across the lines of a
/// span, since a string or a nesting may straddle the line break.
struct SkipState {
	depth:		i32,
	in_string:	bool,
}

/// What to do with a line-leading Typst code statement or standalone template call.
enum CodeSkip {
	Line,				// the call closes on this line; skip the one line, as before
	Multi(SkipState),	// the delimiters are still open; begin a multi-line skip carrying the depth
}

/// If this already-left-trimmed line begins a Typst code statement Austenite skips for now, decides how
/// much to skip: `Line` for a statement or standalone call that closes on this line, `Multi` for one
/// whose delimiters are still open at the end of it. `None` when the line is not code the reader skips,
/// so the caller sets it as prose.
///
/// The four block statements (`#import`, `#let`, `#set`, `#show`) are always code; a line-leading call
/// (`#name(` or `#name[`) is skipped only as a whole -- either it closes on the line, or it opens a
/// multi-line span. A balanced `#name[...]` with prose trailing it (`#index-main[x]More prose...`) is
/// left to set, since its content is a marker within a real paragraph, not a standalone call.
fn code_skip(trimmed: &str) -> Option<CodeSkip> {
	let keyword	= code_keyword(trimmed);
	if !keyword && !opens_standalone_call(trimmed) {
		return None;
	}
	let mut state = SkipState { depth: 0, in_string: false };
	scan_brackets(trimmed, &mut state);
	if state.depth > 0 {
		return Some(CodeSkip::Multi(state));
	}
	// The delimiters balance on this line. A block statement is skipped whatever trails it; a standalone
	// call is skipped only when it truly ends with its own closer, so a marker inside a paragraph sets.
	if keyword || trimmed.ends_with(')') || trimmed.ends_with(']') {
		return Some(CodeSkip::Line);
	}
	None
}

/// Does this already-left-trimmed line open one of the four Typst block statements the reader skips?
fn code_keyword(trimmed: &str) -> bool {
	for kw in ["#import ", "#import\"", "#let ", "#set ", "#show ", "#show:"] {
		if trimmed.starts_with(kw) {
			return true;
		}
	}
	false
}

/// Does this already-left-trimmed line open with a standalone call -- `#`, an identifier, then `(` or
/// `[`? A crude test, enough to recognise the opener of a call to an unrecognised template function
/// without inspecting where or whether it closes; the balance decides single- versus multi-line.
fn opens_standalone_call(trimmed: &str) -> bool {
	let mut cs = trimmed.chars();
	if cs.next() != Some('#') {
		return false;
	}
	let mut ident	= String::new();
	for c in cs {
		if c.is_alphanumeric() || c == '-' || c == '_' {
			ident.push(c);
			continue;
		}
		// The first non-identifier character must open the call. A line-leading inline glossary or
		// index call is content, not a skippable standalone call, even when it happens to close on its
		// own line, so [`parse_inlines`] sets its display text rather than the reader dropping it.
		return !ident.is_empty() && (c == '(' || c == '[') && !is_inline_call(&ident);
	}
	false
}

/// Is this identifier one of the book template's inline functions the reader sets in place -- a glossary
/// or index call, a term-dictionary lookup, a hyperlink, a citation or an emphasis call? These emit body
/// text (or an invisible marker) mid-paragraph, so a line that opens with one is prose the inline scanner
/// reads, never a standalone call the line scanner skips.
fn is_inline_call(name: &str) -> bool {
	matches!(name,
		// The simple string-keyed glossary/index family, keyed on their own display text.
		"gs" | "gscap" | "gsi" | "gscapi" | "glossind" | "glossindcap"
		// The term-dictionary family, keyed on a `term-dict` entry: `g`/`gcap`/`gi`/`gcapi` set the value
		// with first-use styling, `t`/`tcap` set it plain, `graw` sets it in the mono face.
		| "g" | "gcap" | "gi" | "gcapi" | "t" | "tcap" | "graw"
		| "idx" | "idx-main" | "idx-as" | "idx-main-as" | "idx-nested"
		| "index" | "index-main" | "cite" | "link"
		| "emph" | "super"
		| "claim-label" | "claim-refs")
}

/// Folds one line's `()[]{}` into the running [`SkipState`], updating the depth and the in-string flag.
/// A bracket inside a `"..."` literal is ignored, and a `\`-escaped character within a string is passed
/// over, so a quote or bracket written `\"` or `\(` does not miscount. The state carries into the next
/// line, so a string or a nesting that straddles the break is tracked correctly.
fn scan_brackets(line: &str, state: &mut SkipState) {
	let mut escaped = false;
	for c in line.chars() {
		if state.in_string {
			if escaped {
				escaped = false;
			} else if c == '\\' {
				escaped = true;
			} else if c == '"' {
				state.in_string = false;
			}
			continue;
		}
		match c {
			'"'					=> state.in_string = true,
			'(' | '[' | '{'		=> state.depth += 1,
			')' | ']' | '}'		=> state.depth -= 1,
			_					=> {},
		}
	}
}

/// Splits a trailing `<label>` off a heading title: a `<name>` with no inner whitespace at the very end
/// labels the heading and is removed from its text. A title that merely contains angle brackets, or a
/// `< >` with a space inside, keeps them as ordinary characters.
fn split_label(title: &str) -> (String, Option<String>) {
	let t = title.trim_end();
	if let Some(inner) = t.strip_suffix('>') {
		if let Some(p) = inner.rfind('<') {
			let label = &inner[p + 1..];
			if !label.is_empty() && !label.contains(char::is_whitespace) {
				return (inner[..p].trim_end().to_string(), Some(label.to_string()));
			}
		}
	}
	(t.to_string(), None)
}

/// Reads an inline `#footnote[...]` at `i` (a `#`), returning the note's inline markup -- parsed so a
/// `*strong*` or `_emph_` in the note sets with its own face -- and the index just past the closing `]`.
/// `None` when the shape is not a footnote call or its bracket does not close, so anything else is left as
/// ordinary text.
fn footnote_call(chars: &[char], i: usize, skips: &mut SkipSummary) -> Option<(Vec<Inline>, usize)> {
	let Some(open) = at_lit(chars, i, "#footnote") else { return None; };
	if chars.get(open) != Some(&'[') {
		return None;
	}
	let Some((inner, next)) = read_group(chars, open) else { return None; };
	Some((parse_inlines_in(&inner, skips), next))
}

/// Reads an inline `#emph[...]` at `i` (a `#`), returning its inner markup unreduced -- it is the call
/// form of `_..._` and the caller expands it the same way -- and the index just past the closing `]`.
/// `None` when the shape is not an emph call or its bracket does not close, so anything else is left as
/// ordinary text.
fn emph_call(chars: &[char], i: usize) -> Option<(String, usize)> {
	let open = at_lit(chars, i, "#emph")?;
	if chars.get(open) != Some(&'[') {
		return None;
	}
	read_group(chars, open)
}

/// Reads an inline `#super[...]` or `#super("...")` at `i` (a `#`), returning its content reduced to
/// display text by [`flatten_markup`] -- usually a short string or number -- and the index just past the
/// closing bracket. `None` when the shape is not a super call or its argument does not close.
fn super_call(chars: &[char], i: usize) -> Option<(String, usize)> {
	let open = at_lit(chars, i, "#super")?;
	match chars.get(open) {
		Some('[') | Some('(')	=> {},
		_						=> return None,
	}
	let (inner, next) = read_group(chars, open)?;
	Some((flatten_markup(&unwrap_arg(&inner)), next))
}

/// Reads an inline `#cite(...)` at `i` (a `#`), returning the citation keys and the index past the
/// closing `)`. Every `<label>` token inside the parentheses is a key; a named argument such as
/// `form: "prose"` carries no label and is ignored. `None` when the shape is not a cite call or its
/// parentheses do not close, so anything else is left as ordinary text.
fn cite_call(chars: &[char], i: usize) -> Option<(Vec<String>, usize)> {
	let open = at_lit(chars, i, "#cite")?;
	if chars.get(open) != Some(&'(') {
		return None;
	}
	let (inner, next) = read_group(chars, open)?;
	let keys = cite_keys(&inner);
	Some((keys, next))
}

/// Extracts the `<label>` citation keys from the inside of a `#cite(...)` call, in order. A `<` opens a
/// key and the next `>` closes it; anything outside a `<...>` pair (a named argument, a separating comma)
/// is skipped.
fn cite_keys(inner: &str) -> Vec<String> {
	let chars:	Vec<char>	= inner.chars().collect();
	let mut keys			= Vec::new();
	let mut i				= 0usize;
	while i < chars.len() {
		if chars[i] == '<' {
			if let Some(close) = (i + 1..chars.len()).find(|&j| chars[j] == '>') {
				let key: String = chars[i + 1..close].iter().collect();
				let key = key.trim().to_string();
				if !key.is_empty() {
					keys.push(key);
				}
				i = close + 1;
				continue;
			}
		}
		i += 1;
	}
	keys
}

/// Reads an inline `#claim-label(...)` or `#claim-refs(...)` at `i` (a `#`), returning the index just
/// past the closing `)`. Both are the book's claims plumbing: `claim-label` registers invisible metadata
/// and sets a compressed code string in the outside margin, `claim-refs` registers metadata only, and
/// neither sets anything in the body text column. The reader consumes the call and sets nothing where it
/// stood, so the raw markup no longer leaks and the surrounding prose closes over the gap as Typst's body
/// does. The marginal code annotation is not reproduced: the page carries a single text column with no
/// margin-placement facility. `None` when the shape is not a claim call or its parentheses do not close.
fn claim_call(chars: &[char], i: usize) -> Option<usize> {
	let open = at_lit(chars, i, "#claim-label")
		.or_else(|| at_lit(chars, i, "#claim-refs"))?;
	if chars.get(open) != Some(&'(') {
		return None;
	}
	let (_, next) = read_group(chars, open)?;
	Some(next)
}

/// Reduces a run of markup to plain display text: the words a reader sees, with the emphasis, code,
/// glossary and index delimiters removed. A glossary term contributes its display, a visible index call
/// its text, a pure index marker nothing; `*strong*` and `_emph_` contribute their inner words. Inline
/// maths and cross-references, which have no plain form here, contribute nothing. Used where the engine
/// takes a plain string -- a footnote's note, a table cell, a figure caption -- and cannot yet carry the
/// runs themselves.
pub fn flatten_markup(text: &str) -> String {
	let mut out = String::new();
	for run in parse_inlines(text) {
		match run {
			Inline::Text(t)					=> out.push_str(&t),
			// A `*strong*` or `_emph_` inner run may still carry markup -- `*_word_*` nests emphasis in
			// strong -- so it is flattened again to strip the inner delimiters. Parsing does not nest, so
			// each pass removes one layer and the recursion terminates on plain text.
			Inline::Strong(t)				=> out.push_str(&flatten_markup(&t)),
			Inline::Emph(t)					=> out.push_str(&flatten_markup(&t)),
			Inline::BoldItalic(t)			=> out.push_str(&t),	// already the flat inner of a nested run
			Inline::Super(t)				=> out.push_str(&t),	// a flattened string cannot raise; keep its text
			Inline::Code(t)					=> out.push_str(&t),
			Inline::Glossary { display, .. }	=> out.push_str(&display),
			Inline::PageRef(_)				=> {},	// a page number has no plain form before layout
			Inline::Math(_)					=> {},	// maths is dropped from a flattened string
			Inline::Footnote(_)				=> {},	// a nested footnote is not set within a flattened string
			Inline::Cite(_)					=> {},	// a citation has no plain form before the bibliography resolves it
		}
	}
	out
}

/// What an inline glossary or index call sets into the running text.
enum Call {
	Glossary { term: String, display: String },	// a glossary term, keyed by `term` for first-use styling
	Visible(String),	// display text set plain, its markup parsed by the caller
	Invisible,			// a pure index marker: nothing is set
}

/// Reads an inline glossary or index call at `i` (a `#`), returning what it sets and the index just past
/// it, or `None` when the `#name` is not one the reader knows or its argument brackets do not close.
///
/// The visible glossary functions set their bracket content as the term, capitalising the display for
/// the `-cap` variants; `idx`/`idx-main` set the content plain; `idx-as`/`idx-main-as` take a second
/// argument as the display and set that; `index`/`index-main`/`idx-nested` are pure markers and set
/// nothing. First use is keyed by the term as written, matching the template's own case-sensitive
/// `glossary-seen` set.
///
/// The term-dictionary family keys a `term-dict` entry rather than carrying its own display, and the
/// reader translates the key to that value at parse time (the loader installs the map from `terms.typ`
/// before parsing): `g`/`gi` set the value bold-italic on first use, `gcap`/`gcapi` capitalised, `t`/`tcap`
/// plain and `graw` plain (its mono face is not reproduced). A key with no `term-dict` entry falls back to
/// the key text and is recorded in `skips`, so an unknown key is visible on the terse skip line rather
/// than silently wrong -- the template panics on a miss, which the reader must not.
fn glossary_call(chars: &[char], i: usize, skips: &mut SkipSummary) -> Option<(Call, usize)> {
	if chars.get(i) != Some(&'#') {
		return None;
	}
	let start	= i + 1;
	let mut j	= start;
	while j < chars.len() && (chars[j].is_ascii_alphanumeric() || chars[j] == '-' || chars[j] == '_') {
		j += 1;
	}
	if j == start {
		return None;
	}
	let name: String = chars[start..j].iter().collect();
	if !is_inline_call(&name) {
		return None;
	}
	match chars.get(j) {
		Some('(') | Some('[')	=> {},
		_						=> return None,
	}
	let (a1, next1) = read_group(chars, j)?;

	// The two-argument display functions take the second argument as the visible text.
	if name == "idx-as" || name == "idx-main-as" {
		let (a2, next2) = read_group(chars, next1)?;
		return Some((Call::Visible(unwrap_arg(&a2)), next2));
	}
	// A nested index entry is a pure marker; consume an optional second argument.
	if name == "idx-nested" {
		let end = match read_group(chars, next1) {
			Some((_, n2))	=> n2,
			None			=> next1,
		};
		return Some((Call::Invisible, end));
	}

	let arg = unwrap_arg(&a1);
	let call = match name.as_str() {
		// The simple family keys its own display text (a `term-defs` entry), so no translation applies.
		"gs" | "gsi"							=> Call::Glossary { term: arg.clone(), display: arg },
		"gscap" | "gscapi"						=> Call::Glossary { term: arg.clone(), display: cap_first(&arg) },
		// `glossind`/`glossindcap` auto-detect: a key that is in `term-dict` sets its value, otherwise the
		// key stands as its own display, matching the template's `if key in term-dict` branch.
		"glossind"								=> {
			let display = term_value(&arg).unwrap_or_else(|| arg.clone());
			Call::Glossary { term: arg.clone(), display }
		},
		"glossindcap"							=> {
			let display = term_value(&arg).unwrap_or_else(|| arg.clone());
			Call::Glossary { term: arg.clone(), display: cap_first(&display) }
		},
		// The term-dictionary family translates the key to its value; first use is keyed by the key, as the
		// template keys `glossary-seen` by the key name rather than the value.
		"g" | "gi"								=> Call::Glossary { term: arg.clone(), display: resolve_term(&arg, &name, skips) },
		"gcap" | "gcapi"						=> Call::Glossary { term: arg.clone(), display: cap_first(&resolve_term(&arg, &name, skips)) },
		"t" | "graw"							=> Call::Visible(resolve_term(&arg, &name, skips)),
		"tcap"									=> Call::Visible(cap_first(&resolve_term(&arg, &name, skips))),
		"idx" | "idx-main"						=> Call::Visible(arg),
		"index" | "index-main"					=> Call::Invisible,
		_									=> return None,
	};
	Some((call, next1))
}

/// Resolves a term-dictionary key to its display value, or -- when no map is installed or it holds no
/// such key -- falls back to the key text and records the miss in `skips` under the calling function's
/// name, so an unknown key shows on the terse skip line rather than rendering silently as the raw key.
fn resolve_term(key: &str, func: &str, skips: &mut SkipSummary) -> String {
	match term_value(key) {
		Some(value)	=> value,
		None		=> {
			skips.record(&fmt!("#{} unknown term-dict key {:?}", func, key));
			key.to_string()
		},
	}
}

/// Reads a bracket or paren group whose opener sits at `i`, returning its inner content and the index
/// just past the matching closer. Nesting of the same delimiter and `"..."` strings are respected, so a
/// bracket inside a quoted argument or a nested group does not close the group early. `None` when the
/// group never closes, so a malformed call is left as ordinary text.
pub(crate) fn read_group(chars: &[char], i: usize) -> Option<(String, usize)> {
	let open	= *chars.get(i)?;
	let close	= match open {
		'['	=> ']',
		'('	=> ')',
		_	=> return None,
	};
	let mut depth	= 0i32;
	let mut in_str	= false;
	let mut escaped	= false;
	let mut inner	= String::new();
	let mut j		= i;
	while j < chars.len() {
		let c = chars[j];
		if in_str {
			inner.push(c);
			if escaped				{ escaped = false; }
			else if c == '\\'		{ escaped = true; }
			else if c == '"'		{ in_str = false; }
			j += 1;
			continue;
		}
		if c == '"' {
			in_str = true;
			inner.push(c);
		} else if c == open {
			depth += 1;
			if depth > 1 { inner.push(c); }	// keep a nested opener, drop the outer one
		} else if c == close {
			depth -= 1;
			if depth == 0 {
				return Some((inner, j + 1));
			}
			inner.push(c);
		} else {
			inner.push(c);
		}
		j += 1;
	}
	None
}

/// Strips a `"..."` wrapper from a paren-string argument, so `#gs("surplus")` reads the same term as
/// `#gs[surplus]`. A bracket argument has no quotes to strip and is returned unchanged.
fn unwrap_arg(inner: &str) -> String {
	let t = inner.trim();
	if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
		return t[1..t.len() - 1].to_string();
	}
	inner.to_string()
}

/// Capitalises the first character of a term for the `-cap` glossary variants, leaving the rest as it
/// stands. The template capitalises the first grapheme cluster; the first `char` matches it for every
/// term in these books.
fn cap_first(s: &str) -> String {
	let mut cs = s.chars();
	match cs.next() {
		Some(c)	=> c.to_uppercase().collect::<String>() + cs.as_str(),
		None	=> String::new(),
	}
}

/// Whether a `/* ... */` block comment is open across the line break.
struct CommentState {
	in_block:	bool,
}

/// Removes Typst comments from one line: a `//` to the line's end, and any `/* ... */` span, which may
/// have opened on an earlier line ([`CommentState::in_block`] carries that across). A `//` or `/*`
/// inside a `"..."` string or a `` `code` `` span is not a comment and is kept, and a `//` immediately
/// after `:` is kept so a bare URL survives. Quotes and backticks are treated as span delimiters here,
/// which is what the reader's markup needs; a real Typst code line with string literals is skipped whole
/// by the caller, so stripping it never reaches the output.
fn strip_comments(line: &str, st: &mut CommentState) -> String {
	let chars:	Vec<char>	= line.chars().collect();
	let mut out				= String::new();
	let mut in_str			= false;
	let mut in_raw			= false;
	let mut prev			= '\0';
	let mut i				= 0usize;
	while i < chars.len() {
		let c = chars[i];
		if st.in_block {
			if c == '*' && chars.get(i + 1) == Some(&'/') {
				st.in_block = false;
				i += 2;
				prev = '\0';
				continue;
			}
			i += 1;
			continue;
		}
		if in_str {
			out.push(c);
			if c == '"' { in_str = false; }
			prev = c;
			i += 1;
			continue;
		}
		if in_raw {
			out.push(c);
			if c == '`' { in_raw = false; }
			prev = c;
			i += 1;
			continue;
		}
		if c == '"' {
			in_str = true;
			out.push(c);
			prev = c;
			i += 1;
			continue;
		}
		if c == '`' {
			in_raw = true;
			out.push(c);
			prev = c;
			i += 1;
			continue;
		}
		if c == '/' && chars.get(i + 1) == Some(&'/') {
			if prev == ':' {
				out.push(c);	// a `://` is part of a URL, not a comment
				prev = c;
				i += 1;
				continue;
			}
			break;	// a line comment: drop the rest of the line
		}
		if c == '/' && chars.get(i + 1) == Some(&'*') {
			st.in_block = true;
			i += 2;
			prev = '\0';
			continue;
		}
		out.push(c);
		prev = c;
		i += 1;
	}
	out
}

// -- Multi-line figure, table and data-array capture ----------------------------------------------

/// A multi-line construct gathered whole so it can be parsed. The buffer accumulates its lines; the
/// bracket state closes it when the delimiters balance; the kind decides how the buffer is dispatched.
struct Capture {
	kind:	CaptureKind,
	buf:	String,
	state:	SkipState,
}

/// Which multi-line construct is being gathered.
enum CaptureKind {
	Figure,			// a `#figure(...)` call, possibly wrapping a table or an image
	Table,			// a bare `#table(...)` call
	Image,			// a line-leading `#padded-image(...)` or `#image(...)` set without a figure number
	Let(String),	// a `#let name = (...)` data array bound to this name
	Columns,		// a `#columns(n)[ ... ]` wrapper: its body is set single-column
}

/// Detects the opener of a multi-line construct the reader parses rather than skips: a `#figure(`, a
/// bare `#table(`, or a `#let name = (` data array. `None` for any other line, which the caller then
/// offers to [`code_skip`].
fn capture_opener(trimmed: &str) -> Option<CaptureKind> {
	if trimmed.starts_with("#figure(") {
		return Some(CaptureKind::Figure);
	}
	if trimmed.starts_with("#table(") {
		return Some(CaptureKind::Table);
	}
	if trimmed.starts_with("#columns(") {
		return Some(CaptureKind::Columns);
	}
	// A section opener draws its logo with a line-leading `#padded-image(...)` (the Pearl section's pearlite
	// mark), and a bare `#image(...)` places a graphic likewise. Both are set centred without a figure
	// number, so they are captured here rather than skipped. The hyphen keeps `#image(` from matching the
	// tail of `#padded-image(`, which is tried first.
	if trimmed.starts_with("#padded-image(") || trimmed.starts_with("#image(") {
		return Some(CaptureKind::Image);
	}
	let_array_name(trimmed).map(CaptureKind::Let)
}

/// If the line is a `#let name = (` binding whose value opens a paren group, its name; else `None`. Only
/// an array or tuple value is captured -- a scalar or a function `#let` (whose name carries `(`) is left
/// to [`code_skip`].
fn let_array_name(trimmed: &str) -> Option<String> {
	let rest	= trimmed.strip_prefix("#let ")?;
	let eq		= rest.find('=')?;
	let name	= rest[..eq].trim();
	if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
		return None;
	}
	let value = rest[eq + 1..].trim_start();
	if value.starts_with('(') {
		Some(name.to_string())
	} else {
		None
	}
}

/// Dispatches a completed capture: a data array is evaluated and stored under its name; a table or a
/// figure is parsed into an [`Item`]. A construct that does not parse -- an unresolved spread, an empty
/// table -- yields no item rather than an error, so a stray call never fails the whole document.
fn dispatch_capture(
	cap:	Capture,
	items:	&mut Vec<Item>,
	arrays:	&mut HashMap<String, Vec<Vec<Inline>>>,
	skips:	&mut SkipSummary,
)
{
	match cap.kind {
		CaptureKind::Let(name) => {
			arrays.insert(name, parse_let_array(&cap.buf));
		},
		CaptureKind::Table => {
			if let Some(inner) = call_inner(&cap.buf, "table") {
				if let Some(spec) = parse_table_spec(&inner, arrays, outer_text_size(&cap.buf)) {
					items.push(Item::Table { spec, span: Span::new(0, 0) });
				}
			}
		},
		CaptureKind::Figure => {
			if let Some(item) = parse_figure(&cap.buf, arrays) {
				items.push(item);
			}
		},
		CaptureKind::Image => {
			// A line-leading image call: its path and sizing are read the same way a figure's image body is,
			// then set as a plain centred image with no figure number. A call naming no path draws nothing.
			let (path, width, height, scale) = image_call(&cap.buf);
			if !path.is_empty() {
				items.push(Item::Image { path, width, height, scale, span: Span::new(0, 0) });
			}
		},
		CaptureKind::Columns => {
			// The reader has no column model: the `#columns(n)[ ... ]` wrapper is recorded as skipped and
			// its body set single-column, so the words survive even though the multi-column layout does not.
			// The body is a block sequence, so it is read through the document parser again and its items
			// spliced in; a nested skip (a `#colbreak()`, an unknown call) folds into the same summary.
			skips.record("#columns");
			if let Some(body) = columns_body(&cap.buf) {
				if let Ok((mut inner, sub)) = document_with_skips(&body) {
					skips.merge(&sub);
					items.append(&mut inner);
				}
			}
		},
	}
}

/// The `[ ... ]` body of a captured `#columns(n)[ ... ]` wrapper: the column count arguments are read and
/// dropped, and the bracketed block content returned for re-parsing. `None` when no `[...]` group follows
/// the arguments, so a malformed wrapper contributes no body.
fn columns_body(buf: &str) -> Option<String> {
	let chars:	Vec<char>	= buf.chars().collect();
	let Some(at) = find_lit(&chars, "#columns") else { return None; };
	let open	= at + "#columns".chars().count();
	if chars.get(open) != Some(&'(') {
		return None;
	}
	let Some((_, after_args)) = read_group(&chars, open) else { return None; };
	let mut j = after_args;
	while j < chars.len() && chars[j].is_whitespace() {
		j += 1;
	}
	if chars.get(j) != Some(&'[') {
		return None;
	}
	read_group(&chars, j).map(|(body, _)| body)
}

/// The index of the first occurrence of the literal `s` in `chars`, or `None`.
fn find_lit(chars: &[char], s: &str) -> Option<usize> {
	let pat:	Vec<char>	= s.chars().collect();
	if pat.is_empty() || chars.len() < pat.len() {
		return None;
	}
	(0..=chars.len() - pat.len()).find(|&start| chars[start..start + pat.len()] == pat[..])
}

/// Evaluates a `#let name = (...)` value into the flat sequence of cells it holds, each cell its inline
/// runs. The value is the paren group after the `=`; every `[...]` group within it, at any depth, is one
/// cell -- which is what `array.flatten()` yields for an array of content tuples.
fn parse_let_array(buf: &str) -> Vec<Vec<Inline>> {
	let chars:	Vec<char>	= buf.chars().collect();
	let eq = match chars.iter().position(|&c| c == '=') {
		Some(e)	=> e,
		None	=> return Vec::new(),
	};
	let open = match (eq + 1..chars.len()).find(|&j| !chars[j].is_whitespace()) {
		Some(v) if chars[v] == '('	=> v,
		_							=> return Vec::new(),
	};
	match read_group(&chars, open) {
		Some((inner, _))	=> collect_cells(&inner),
		None				=> Vec::new(),
	}
}

/// Collects every `[...]` group in `inner`, in order, each parsed into its inline runs. A `[` inside a
/// string is not a cell. Once a group opens, its whole content is one cell and is not descended into.
///
/// A `table.cell(colspan: n)[...]` wrapper is expanded to `n` grid cells: the bracketed content, then
/// `n - 1` empty span placeholders. A spanning cell consumes several columns in Typst, so without the
/// placeholders every cell after it slides one column to the left and the last column overruns the
/// table; the placeholders keep the flat cell stream aligned to the column grid. A bare `table.cell(...)`
/// (no `colspan`) is one cell, like a plain `[...]`.
fn collect_cells(inner: &str) -> Vec<Vec<Inline>> {
	let chars:	Vec<char>	= inner.chars().collect();
	let mut cells			= Vec::new();
	let mut in_str			= false;
	let mut esc				= false;
	let mut i				= 0usize;
	while i < chars.len() {
		let c = chars[i];
		if in_str {
			if esc			{ esc = false; }
			else if c == '\\'	{ esc = true; }
			else if c == '"'	{ in_str = false; }
			i += 1;
			continue;
		}
		if c == '"' {
			in_str = true;
			i += 1;
			continue;
		}
		// A `table.cell(args)[content]` wrapper: read the args for a `colspan`, then take the following
		// `[...]` as the content and emit it across that many columns.
		if let Some(after) = at_lit(&chars, i, "table.cell") {
			if chars.get(after) == Some(&'(') {
				if let Some((args, past_args)) = read_group(&chars, after) {
					let span	= cell_colspan(&args);
					let mut j	= past_args;
					while j < chars.len() && chars[j].is_whitespace() {
						j += 1;
					}
					if chars.get(j) == Some(&'[') {
						if let Some((content, next)) = read_group(&chars, j) {
							cells.push(parse_inlines(&content));
							for _ in 1..span {
								cells.push(vec![Inline::Text(String::new())]);
							}
							i = next;
							continue;
						}
					}
					// No content group followed the wrapper; step past its args and carry on.
					i = past_args;
					continue;
				}
			}
		}
		if c == '[' {
			if let Some((content, next)) = read_group(&chars, i) {
				cells.push(parse_inlines(&content));
				i = next;
				continue;
			}
		}
		i += 1;
	}
	cells
}

/// The `colspan:` of a `table.cell(...)` argument list, at least one. A missing or unreadable `colspan`
/// is a single column.
fn cell_colspan(args: &str) -> usize {
	for arg in split_top_args(args) {
		if let Some((key, val)) = named_arg(arg.trim()) {
			if key == "colspan" {
				if let Ok(n) = val.trim().parse::<usize>() {
					return n.max(1);
				}
			}
		}
	}
	1
}

/// Parses the inner text of a `#table(...)` call into a [`TableSpec`]. `columns:` fixes the column
/// count, `align:` the alignment, a `fill:` keyed on `row == 0` marks a header row; cells come from
/// inline `[...]` groups and from a `..name.flatten()` spread resolved against the data arrays. `None`
/// when no cells are found, so an empty or unresolved table sets nothing.
fn parse_table_spec(
	inner:			&str,
	arrays:			&HashMap<String, Vec<Vec<Inline>>>,
	outer_text_pt:	Option<f64>,
)
	-> Option<TableSpec>
{
	let mut ncols		= 1usize;
	let mut align		= AlignSpec::Uniform(Align::Left);
	let mut header		= false;
	let mut inset_pt:	Option<f64>			= None;
	let mut weights:	Vec<f64>			= Vec::new();
	let mut cells:		Vec<Vec<Inline>>	= Vec::new();
	for arg in split_top_args(inner) {
		let a = arg.trim();
		if a.is_empty() {
			continue;
		}
		if let Some((key, val)) = named_arg(a) {
			match key.as_str() {
				"columns"	=> { ncols = parse_columns(&val); weights = parse_column_weights(&val); },
				"align"		=> align = parse_align(&val),
				"fill"		=> if fill_marks_header(&val) { header = true; },
				"inset"		=> inset_pt = parse_length(&val).map(length_pt),
				_			=> {},	// stroke, gutter and the rest are not modelled
			}
			continue;
		}
		if let Some(name) = spread_name(a) {
			if let Some(v) = arrays.get(&name) {
				cells.extend(v.iter().cloned());
			}
			continue;
		}
		// An inline positional cell, or a `table.header(...)`/`table.cell(...)` wrapper whose bracketed
		// content is the cell -- each `[...]` group in the argument is one cell, in order.
		if a.contains('[') {
			cells.extend(collect_cells(a));
		}
	}
	if cells.is_empty() {
		return None;
	}
	Some(TableSpec { ncols: ncols.max(1), header, align, weights, text_pt: outer_text_pt, inset_pt, cells })
}

/// The absolute point value of a [`Length`], resolving a percentage against a nominal 100 pt so a
/// percentage inset still yields a sensible padding; a table's inset is in practice an absolute length.
pub(crate) fn length_pt(len: Length) -> f64 {
	match len {
		Length::Abs(pt)	=> pt,
		Length::Rel(f)	=> f * 100.0,
	}
}

/// The size of a `text(size: Npt)[...]` (or `#text(...)`) wrapper at the start of `text`, in points, or
/// `None` when the body is not wrapped in a sized `text` call. This lets a figure's small-set table --
/// `text(size: 7pt)[#table(...)]` -- carry its reduced size, which Typst applies to the whole table.
fn outer_text_size(text: &str) -> Option<f64> {
	let inner = call_inner(text, "text")?;
	for arg in split_top_args(&inner) {
		let a = arg.trim();
		if let Some((key, val)) = named_arg(a) {
			if key == "size" {
				return parse_length(&val).map(length_pt);
			}
		} else if a.ends_with("pt") || a.ends_with("em") {
			// The first positional length is the size, as in `text(7pt)[...]`.
			return parse_length(a).map(length_pt);
		}
	}
	None
}

/// Parses a `#figure(...)` call (its buffer, a trailing `<label>` and all) into an [`Item::Figure`]. The
/// positional argument is the body -- a wrapped `#table(...)` set in full, or an image call stood in for
/// by a placeholder; `caption:` sets the caption, `supplement:`/`kind:` the "Figure" or "Table" label.
fn parse_figure(buf: &str, arrays: &HashMap<String, Vec<Vec<Inline>>>) -> Option<Item> {
	let (body_src, label)	= strip_trailing_label(buf);
	let inner				= call_inner(&body_src, "figure")?;

	let mut caption:	Option<Vec<Inline>>	= None;
	let mut supplement:	Option<String>	= None;
	let mut kind:		Option<String>	= None;
	let mut positional:	Option<String>	= None;
	for arg in split_top_args(&inner) {
		let a = arg.trim();
		if a.is_empty() {
			continue;
		}
		if let Some((key, val)) = named_arg(a) {
			match key.as_str() {
				"caption"		=> caption = Some(caption_inlines(&val)),
				"supplement"	=> supplement = Some(unquote(&val)),
				"kind"			=> kind = Some(unquote(&val)),
				_				=> {},	// placement and the rest do not affect the set figure
			}
			continue;
		}
		if positional.is_none() {
			positional = Some(a.to_string());	// the first positional argument is the figure body
		}
	}

	let body_text	= positional.unwrap_or_default();
	let body		= figure_body(&body_text, arrays);
	let supplement	= supplement.unwrap_or_else(|| match kind.as_deref() {
		Some("table")	=> "Table".to_string(),
		_				=> "Figure".to_string(),
	});
	Some(Item::Figure { body, caption, supplement, label, span: Span::new(0, 0) })
}

/// Decides a figure's body from its positional text: a wrapped `#table(...)` if one is present and
/// parses, otherwise an image carrying the path and any declared sizing (empty path when none is found).
fn figure_body(text: &str, arrays: &HashMap<String, Vec<Vec<Inline>>>) -> FigureBody {
	if let Some(inner) = call_inner(text, "table") {
		if let Some(spec) = parse_table_spec(&inner, arrays, outer_text_size(text)) {
			return FigureBody::Table(spec);
		}
	}
	// A CeTZ/Fletcher diagram, bar chart or line plot drawn inline is read into a builder that draws it
	// for real; only when the body is none of these does it fall through to the image/placeholder path.
	if let Some(cf) = super::codefig::parse_code_figure(text) {
		return FigureBody::Code(cf);
	}
	let (path, width, height, scale) = image_call(text);
	FigureBody::Image { path, width, height, scale }
}

/// The path and sizing of a `padded-image("...")` or `image("...")` call in `text`. The custom wrapper is
/// tried first, since `image` is a word boundary within it only after the hyphen. The first positional
/// argument is the path; `width`/`height` size an `image(...)`, `scale` a `padded-image(...)`. A path
/// that is not found gives an empty string, which the block layer stands in for with a placeholder.
fn image_call(text: &str) -> (String, Option<Length>, Option<Length>, Option<f64>) {
	for name in ["padded-image", "image"] {
		if let Some(inner) = call_inner(text, name) {
			let mut path:	Option<String>	= None;
			let mut width:	Option<Length>	= None;
			let mut height:	Option<Length>	= None;
			let mut scale:	Option<f64>		= None;
			for arg in split_top_args(&inner) {
				let a = arg.trim();
				if a.is_empty() {
					continue;
				}
				if let Some((key, val)) = named_arg(a) {
					match key.as_str() {
						"width"		=> width = parse_length(&val),
						"height"	=> height = parse_length(&val),
						"scale"		=> scale = parse_percent(&val),
						_			=> {},	// padding and the rest do not size the set image
					}
					continue;
				}
				if path.is_none() {
					path = first_string(a);
				}
			}
			if let Some(p) = path {
				return (p, width, height, scale);
			}
		}
	}
	(String::new(), None, None, None)
}

/// Reads a Typst length argument into a [`Length`]: a percentage as a fraction of the measure, a `pt`,
/// `mm`, `cm` or `in` length as absolute points, a bare number as points. `auto` and anything unreadable
/// give `None`, so the figure falls back to filling the measure.
pub(crate) fn parse_length(val: &str) -> Option<Length> {
	let v = val.trim();
	if let Some(pct) = v.strip_suffix('%') {
		return pct.trim().parse::<f64>().ok().map(|n| Length::Rel(n / 100.0));
	}
	for (unit, per_pt) in [("pt", 1.0), ("mm", 72.0 / 25.4), ("cm", 72.0 / 2.54), ("in", 72.0)] {
		if let Some(num) = v.strip_suffix(unit) {
			return num.trim().parse::<f64>().ok().map(|n| Length::Abs(n * per_pt));
		}
	}
	v.parse::<f64>().ok().map(Length::Abs)
}

/// Parses a standalone `#line(length:.., stroke:..)` into an [`Item::Rule`]. The length is a fraction of
/// the measure (`100%`) or an absolute length; the stroke gives the rule's thickness (a `pt` length) and
/// its grey (a `luma(N)` component). A missing length fills the measure; a missing thickness is a hairline
/// half-point; a missing colour is black, Typst's default stroke.
fn parse_line_rule(trimmed: &str) -> Option<Item> {
	let inner		= call_inner(trimmed, "line")?;
	let mut width	= Length::Rel(1.0);
	let mut thickness	= 0.5;
	let mut grey	= 0u8;
	for arg in split_top_args(&inner) {
		let a = arg.trim();
		if let Some((key, val)) = named_arg(a) {
			match key.as_str() {
				"length"	=> if let Some(l) = parse_length(&val) { width = l; },
				"stroke"	=> {
					let (t, g) = parse_stroke(&val);
					if let Some(t) = t { thickness = t; }
					if let Some(g) = g { grey = g; }
				},
				_			=> {},	// start, end, angle and the rest do not affect a horizontal divider
			}
		}
	}
	Some(Item::Rule { width, thickness, grey, span: Span::new(0, 0) })
}

/// The thickness (a `pt` length) and grey (a `luma(N)` value, 0-255) of a `stroke:` value such as
/// `0.5pt + luma(180)`; either component may be absent. A `luma` is read as a grey level; a bare colour
/// name or an `rgb(...)` is not modelled and leaves the grey unset.
fn parse_stroke(val: &str) -> (Option<f64>, Option<u8>) {
	let mut thickness:	Option<f64>	= None;
	let mut grey:		Option<u8>	= None;
	for part in val.split('+') {
		let p = part.trim();
		if let Some(inner) = call_inner(p, "luma") {
			if let Ok(n) = inner.trim().trim_end_matches('%').trim().parse::<f64>() {
				grey = Some(n.clamp(0.0, 255.0) as u8);
			}
		} else if let Some(Length::Abs(pt)) = parse_length(p) {
			thickness = Some(pt);
		}
	}
	(thickness, grey)
}

/// Reads a percentage argument (`100%`) into a fraction (`1.0`), or `None` when it is not a percentage.
fn parse_percent(val: &str) -> Option<f64> {
	val.trim().strip_suffix('%').and_then(|p| p.trim().parse::<f64>().ok()).map(|n| n / 100.0)
}

/// The content of the first `name(...)` call in `text`, balanced across nesting and strings, or `None`.
/// `name` must sit at a word boundary, so a short name does not match inside a longer identifier.
pub(crate) fn call_inner(text: &str, name: &str) -> Option<String> {
	let chars:	Vec<char>	= text.chars().collect();
	let namev:	Vec<char>	= name.chars().collect();
	let paren				= find_call(&chars, &namev, 0)?;
	read_group(&chars, paren).map(|(inner, _)| inner)
}

/// The index of the `(` of the first `name(` at a word boundary at or after `from`, or `None`.
fn find_call(chars: &[char], name: &[char], from: usize) -> Option<usize> {
	if name.is_empty() {
		return None;
	}
	let mut i = from;
	while i + name.len() < chars.len() {
		if chars[i..].starts_with(name) && chars.get(i + name.len()) == Some(&'(') {
			let boundary = i == 0 || !is_call_ident(chars[i - 1]);
			if boundary {
				return Some(i + name.len());
			}
		}
		i += 1;
	}
	None
}

/// A character that continues a Typst identifier, for the word-boundary test in [`find_call`].
fn is_call_ident(c: char) -> bool {
	c.is_alphanumeric() || c == '-' || c == '_'
}

/// The first `"..."` string literal's content in `text`, or `None`.
pub(crate) fn first_string(text: &str) -> Option<String> {
	let chars:	Vec<char>	= text.chars().collect();
	let start				= chars.iter().position(|&c| c == '"')?;
	let end					= (start + 1..chars.len()).find(|&j| chars[j] == '"')?;
	Some(chars[start + 1..end].iter().collect())
}

/// Splits the inner text of a call by its top-level commas, respecting `()[]{}` nesting and `"..."`
/// strings, so a comma inside a nested group or a string does not part an argument.
pub(crate) fn split_top_args(inner: &str) -> Vec<String> {
	let mut args:	Vec<String>	= Vec::new();
	let mut cur					= String::new();
	let mut depth				= 0i32;
	let mut in_str				= false;
	let mut esc					= false;
	for c in inner.chars() {
		if in_str {
			cur.push(c);
			if esc			{ esc = false; }
			else if c == '\\'	{ esc = true; }
			else if c == '"'	{ in_str = false; }
			continue;
		}
		match c {
			'"'					=> { in_str = true; cur.push(c); },
			'(' | '[' | '{'		=> { depth += 1; cur.push(c); },
			')' | ']' | '}'		=> { depth -= 1; cur.push(c); },
			',' if depth == 0	=> { args.push(std::mem::take(&mut cur)); },
			_					=> cur.push(c),
		}
	}
	if !cur.trim().is_empty() {
		args.push(cur);
	}
	args
}

/// Splits a `key: value` argument at its top-level colon, returning the key and the trimmed value, or
/// `None` when there is no top-level colon or the key is not a bare identifier -- so a positional cell
/// or a spread is not mistaken for a named argument.
pub(crate) fn named_arg(arg: &str) -> Option<(String, String)> {
	let chars:	Vec<char>	= arg.chars().collect();
	let mut depth			= 0i32;
	let mut in_str			= false;
	let mut esc				= false;
	for (i, &c) in chars.iter().enumerate() {
		if in_str {
			if esc			{ esc = false; }
			else if c == '\\'	{ esc = true; }
			else if c == '"'	{ in_str = false; }
			continue;
		}
		match c {
			'"'				=> in_str = true,
			'(' | '[' | '{'	=> depth += 1,
			')' | ']' | '}'	=> depth -= 1,
			':' if depth == 0 => {
				let key: String = chars[..i].iter().collect();
				let key = key.trim().to_string();
				if !key.is_empty() && key.chars().all(is_call_ident) {
					let val: String = chars[i + 1..].iter().collect();
					return Some((key, val.trim().to_string()));
				}
				return None;
			},
			_				=> {},
		}
	}
	None
}

/// The array name of a `..name` or `..name.flatten()` spread argument, or `None`.
fn spread_name(arg: &str) -> Option<String> {
	let rest		= arg.trim().strip_prefix("..")?;
	let name: String	= rest.chars().take_while(|&c| is_call_ident(c)).collect();
	if name.is_empty() {
		None
	} else {
		Some(name)
	}
}

/// Parses a `columns:` value into a column count: an integer as itself, a track list `(a, b, c)` as its
/// entry count, anything else as one column.
fn parse_columns(val: &str) -> usize {
	let v = val.trim();
	if let Ok(n) = v.parse::<usize>() {
		return n.max(1);
	}
	if v.starts_with('(') {
		let ch: Vec<char> = v.chars().collect();
		if let Some((inner, _)) = read_group(&ch, 0) {
			let cnt = split_top_args(&inner).iter().filter(|p| !p.trim().is_empty()).count();
			return cnt.max(1);
		}
	}
	1
}

/// The per-column fractional weights of a `columns:` track list: a track `Nfr` (or a bare `fr`, weight 1)
/// contributes its weight, an `auto` or a fixed length contributes `0.0` so the column is sized to its
/// content. A bare `columns: N` gives no weights (an empty vector), leaving every column content-sized.
/// The weights let [`table::lower`](crate::table) reproduce Typst's fractional column sizing rather than
/// sizing every column from its widest cell.
fn parse_column_weights(val: &str) -> Vec<f64> {
	let v = val.trim();
	if !v.starts_with('(') {
		return Vec::new();
	}
	let ch: Vec<char> = v.chars().collect();
	let inner = match read_group(&ch, 0) {
		Some((inner, _))	=> inner,
		None				=> return Vec::new(),
	};
	let mut out = Vec::new();
	for track in split_top_args(&inner) {
		let t = track.trim();
		if t.is_empty() {
			continue;
		}
		out.push(track_weight(t));
	}
	out
}

/// The fractional weight of one `columns:` track: `Nfr` reads as `N`, a bare `fr` as `1`, and any other
/// track -- `auto`, `3cm`, `40pt`, `20%` -- as `0.0`, which marks the column content-sized.
fn track_weight(track: &str) -> f64 {
	match track.strip_suffix("fr") {
		Some(num) => {
			let n = num.trim();
			if n.is_empty() { 1.0 } else { n.parse::<f64>().unwrap_or(0.0) }
		},
		None => 0.0,
	}
}

/// Parses an `align:` value: a `(col, row) => ...` closure as [`AlignSpec::Closure`] (its parameter
/// names and body captured for per-cell evaluation), a tuple of column alignments as
/// [`AlignSpec::PerColumn`], a single alignment word as [`AlignSpec::Uniform`].
fn parse_align(val: &str) -> AlignSpec {
	let v = val.trim();
	if let Some(arrow) = v.find("=>") {
		let params	= v[..arrow].trim();
		let body	= v[arrow + 2..].trim().to_string();
		// The parameter list `(col, row)`; a bare single parameter has no parentheses. The first names the
		// column, the second the row, matching Typst's `(col, row)` order.
		let names: Vec<String> = {
			let pch: Vec<char> = params.chars().collect();
			match read_group(&pch, 0) {
				Some((inner, _))	=> split_top_args(&inner).iter().map(|s| s.trim().to_string()).collect(),
				None				=> vec![params.trim().to_string()],
			}
		};
		let col_var = names.first().cloned().filter(|s| !s.is_empty()).unwrap_or_else(|| "col".to_string());
		let row_var = names.get(1).cloned().filter(|s| !s.is_empty()).unwrap_or_else(|| "row".to_string());
		return AlignSpec::Closure(ClosureAlign { col_var, row_var, body });
	}
	if v.starts_with('(') {
		let ch: Vec<char> = v.chars().collect();
		if let Some((inner, _)) = read_group(&ch, 0) {
			let cols: Vec<Align> = split_top_args(&inner).iter().map(|p| word_align(p)).collect();
			if !cols.is_empty() {
				return AlignSpec::PerColumn(cols);
			}
		}
	}
	AlignSpec::Uniform(word_align(v))
}

/// Maps a Typst alignment word to an [`Align`], ignoring a `+ horizon`/`+ top` vertical component and
/// treating `start`/`end` as left/right. An unknown word is left-aligned.
fn word_align(s: &str) -> Align {
	let first = s.trim().split(|c: char| c.is_whitespace() || c == '+').next().unwrap_or("").trim();
	match first {
		"center" | "centre"	=> Align::Centre,
		"right" | "end"		=> Align::Right,
		_					=> Align::Left,
	}
}

/// Does a `fill:` value key on the first row, marking a header? A `fill: (col, row) => ...` closure whose
/// body tests the row index against zero (`row == 0` or the common `y == 0`) fills the first row, which is
/// the books' header idiom; a `y < n` band likewise begins at the first row. Written with or without
/// spaces, and matching either name the closure gives its second (row) parameter.
fn fill_marks_header(val: &str) -> bool {
	let compact: String = val.chars().filter(|c| !c.is_whitespace()).collect();
	compact.contains("row==0")
		|| compact.contains("y==0")
		|| compact.contains("row<")
		|| compact.contains("y<")
}

/// Parses a `caption: [...]` value into its inline runs: the bracket content scanned for markup, or the
/// whole value scanned when it is not a bracket group, so a caption's emphasis, superscript or in-caption
/// maths sets with its own face rather than flattening to upright text.
fn caption_inlines(val: &str) -> Vec<Inline> {
	let v = val.trim();
	let ch: Vec<char> = v.chars().collect();
	if ch.first() == Some(&'[') {
		if let Some((content, _)) = read_group(&ch, 0) {
			return parse_inlines(&content);
		}
	}
	parse_inlines(v)
}

/// Strips a trailing `<label>` from a captured call, returning the call text without it and the label.
/// A `<name>` with no inner whitespace at the very end labels the figure; anything else keeps the text.
fn strip_trailing_label(buf: &str) -> (String, Option<String>) {
	let t = buf.trim_end();
	if let Some(inner) = t.strip_suffix('>') {
		if let Some(p) = inner.rfind('<') {
			let label = &inner[p + 1..];
			if !label.is_empty() && !label.contains(char::is_whitespace) {
				return (inner[..p].to_string(), Some(label.to_string()));
			}
		}
	}
	(buf.to_string(), None)
}

/// Strips a surrounding `"..."` from a string-literal argument value, leaving anything else unchanged.
fn unquote(val: &str) -> String {
	let t = val.trim();
	if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
		return t[1..t.len() - 1].to_string();
	}
	t.to_string()
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A `#claim-label`/`#claim-refs` marker sets nothing in the body, and the prose on either side
	/// closes over the gap as one text run, so no raw markup leaks.
	#[test]
	fn claim_marker_sets_nothing_inline() {
		let runs = parse_inlines("clinical authority#claim-label(<LS8>) bites hardest.");
		assert_eq!(runs.len(), 1);
		match &runs[0] {
			Inline::Text(t) => assert_eq!(t, "clinical authority bites hardest."),
			other => panic!("expected one text run, got {:?}", other),
		}
		// The multi-code and metadata-only forms are consumed the same way.
		assert_eq!(
			flatten_markup("margins#claim-label(<CD14>, <CD15>, <CD4>) formalised#claim-refs(<A1>)."),
			"margins formalised.");
	}

	/// A line that opens with a claim marker is prose, not a standalone call the line scanner skips, so
	/// the sentence that follows the marker is set rather than dropped with it.
	#[test]
	fn line_leading_claim_marker_is_prose() {
		assert!(is_inline_call("claim-label"));
		assert!(is_inline_call("claim-refs"));
		assert!(code_skip("#claim-label(<CD18>). Equilibrium appropriation follows.").is_none());
	}

	/// A line-leading `#padded-image(...)` (a section opener's logo) is set as an [`Item::Image`] carrying
	/// its path and scale, not skipped as a template call and not wrapped in a numbered figure.
	#[test]
	fn line_leading_padded_image_reads_as_image() -> Outcome<()> {
		let (items, _skips) = res!(document_with_skips(
			"= Pearl\n\n#padded-image(\"assets/svg/pearlite_logo_text_right.svg\", scale: 45%)\n\nPearl is the format.\n"));
		let img = res!(items.iter().find_map(|it| match it {
			Item::Image { path, scale, .. }	=> Some((path.clone(), *scale)),
			_								=> None,
		}).ok_or_else(|| err!("no Item::Image was produced for the standalone padded-image"; Test, Bug)));
		assert_eq!(img.0, "assets/svg/pearlite_logo_text_right.svg", "the image path is read");
		assert_eq!(img.1, Some(0.45), "the padded-image scale is read as a fraction");
		// The line must not have been swallowed as a skipped construct, nor turned into a figure.
		assert!(!items.iter().any(|it| matches!(it, Item::Figure { .. })), "a section logo is not a figure");
		Ok(())
	}

	/// `#emph[...]` is the call form of `_..._`: it yields an [`Inline::Emph`] run with the same inner
	/// text, and nothing raw leaks.
	#[test]
	fn emph_call_reads_as_emphasis() {
		let runs = parse_inlines("The cube asks #emph[who] does the extracting.");
		assert!(runs.iter().any(|r| matches!(r, Inline::Emph(t) if t == "who")),
			"emph run missing: {:?}", runs);
		assert!(runs.iter().all(|r| !matches!(r, Inline::Text(t) if t.contains("#emph"))),
			"raw #emph leaked: {:?}", runs);
		// A call carrying its own markup expands the same way `_..._` does.
		let nested = parse_inlines("#emph[the #idx[Harvard Business Review] weekly]");
		assert!(nested.iter().all(|r| !matches!(r, Inline::Text(t) if t.contains("#emph") || t.contains("#idx"))),
			"raw markup leaked from nested emph: {:?}", nested);
	}

	/// Emphasis nested one level -- `*_x_*` or `_*x*_` -- collapses to a single [`Inline::BoldItalic`]
	/// run rather than dropping the outer face, so a bold-italic term sets in the bold-italic face in
	/// prose, a footnote or a table cell alike.
	#[test]
	fn nested_emphasis_reads_as_bold_italic() {
		for src in ["here *_both_* faces", "here _*both*_ faces"] {
			let runs = parse_inlines(src);
			assert!(runs.iter().any(|r| matches!(r, Inline::BoldItalic(t) if t == "both")),
				"expected a bold-italic run from {:?}, got {:?}", src, runs);
			assert!(runs.iter().all(|r| !matches!(r, Inline::Emph(t) if t == "both")),
				"outer face dropped to plain emph in {:?}: {:?}", src, runs);
		}
		// A lone `*strong*` or `_emph_` still yields its single face, unchanged.
		assert!(parse_inlines("just *strong* here").iter().any(|r| matches!(r, Inline::Strong(t) if t == "strong")));
		assert!(parse_inlines("just _emph_ here").iter().any(|r| matches!(r, Inline::Emph(t) if t == "emph")));
	}

	/// A `(col, row) => ...` alignment closure is evaluated per cell: a row-keyed closure centres the
	/// header and flushes the body left; a per-column closure honours each column's own alignment,
	/// including an `or` over several columns.
	#[test]
	fn align_closure_evaluates_per_cell() {
		let row_keyed = parse_align("(col, row) => { if row == 0 { center } else { left } }");
		match row_keyed {
			AlignSpec::Closure(cl) => {
				assert_eq!(cl.align_at(0, 0), Align::Centre);	// header, column 0
				assert_eq!(cl.align_at(0, 1), Align::Left);	// body, column 0 -- was wrongly centred before
				assert_eq!(cl.align_at(2, 3), Align::Left);
			},
			other => panic!("expected a closure, got {:?}", other),
		}
		let per_col = parse_align("(col, row) => { if row == 0 { center } else if col == 0 or col == 4 { left } else { center } }");
		match per_col {
			AlignSpec::Closure(cl) => {
				assert_eq!(cl.align_at(0, 1), Align::Left);
				assert_eq!(cl.align_at(4, 1), Align::Left);
				assert_eq!(cl.align_at(1, 1), Align::Centre);
				assert_eq!(cl.align_at(3, 2), Align::Centre);
			},
			other => panic!("expected a closure, got {:?}", other),
		}
		// A tuple pick indexed by the column parameter.
		let tuple = parse_align("(x, y) => (left, center, right).at(x)");
		match tuple {
			AlignSpec::Closure(cl) => {
				assert_eq!(cl.align_at(0, 5), Align::Left);
				assert_eq!(cl.align_at(1, 5), Align::Centre);
				assert_eq!(cl.align_at(2, 5), Align::Right);
			},
			other => panic!("expected a closure, got {:?}", other),
		}
	}

	/// `#super[...]` yields an [`Inline::Super`] run of its content, in both the bracket and the string
	/// argument forms, and never leaves raw source behind.
	#[test]
	fn super_call_reads_as_superscript() {
		let runs = parse_inlines("The area is 10#super[6] units, split#super[†] on the case.");
		let sups: Vec<&String> = runs.iter().filter_map(|r| match r {
			Inline::Super(t) => Some(t),
			_ => None,
		}).collect();
		assert_eq!(sups, vec!["6", "†"], "unexpected superscript runs: {:?}", runs);
		assert!(runs.iter().all(|r| !matches!(r, Inline::Text(t) if t.contains("#super"))),
			"raw #super leaked: {:?}", runs);
		// The string-argument form reads the same, its quotes stripped.
		let quoted = parse_inlines("x#super(\"2\")");
		assert!(quoted.iter().any(|r| matches!(r, Inline::Super(t) if t == "2")),
			"quoted super run missing: {:?}", quoted);
	}

	/// A citation nested in emphasis keeps its own [`Inline::Cite`] run rather than leaking its source,
	/// while the surrounding words take the emphasis face.
	#[test]
	fn cite_survives_inside_emphasis() {
		let runs = parse_inlines("the classic _early work #cite(<coase1937nature>) here_.");
		assert!(runs.iter().any(|r| matches!(r, Inline::Cite(k) if k == &vec!["coase1937nature".to_string()])),
			"cite run missing: {:?}", runs);
		assert!(runs.iter().all(|r| !matches!(r, Inline::Text(t) if t.contains("#cite"))),
			"raw #cite leaked: {:?}", runs);
	}

	/// `#link("url")[text]` sets the link text in the running line and drops the URL, so no raw `#link`
	/// leaks; a bare `#link("url")` with no bracket sets the URL as its own text.
	#[test]
	fn link_call_renders_text_not_markup() {
		let runs = parse_inlines("See #link(\"https://aistatement.com\")[Centre for AI Safety, May 2023] on risk.");
		assert!(runs.iter().all(|r| !matches!(r, Inline::Text(t) if t.contains("#link") || t.contains("http"))),
			"raw link markup leaked: {:?}", runs);
		let joined = flatten_markup("See #link(\"https://aistatement.com\")[Centre for AI Safety, May 2023] on risk.");
		assert_eq!(joined, "See Centre for AI Safety, May 2023 on risk.");
		// The label-destination form and the bare no-body form both set text without leaking.
		assert_eq!(flatten_markup("read #link(<intro>)[the opening]"), "read the opening");
		assert_eq!(flatten_markup("at #link(\"elearnity.io\")"), "at elearnity.io");
		// A line-leading link is prose the inline scanner reads, not a standalone call the line scanner drops.
		assert!(is_inline_call("link"));
		assert!(code_skip("#link(\"https://x.io\")[click]").is_none());
	}

	/// The term-dictionary aliases set their argument text with the styling of their `gs` siblings: `g`/`gi`
	/// a first-use glossary term, `t`/`tcap` plain text, and none of them leak raw markup. A line opening
	/// with one is prose, not a skipped standalone call.
	/// Installs a fixed term dictionary so the term-dictionary family resolves deterministically. The map
	/// is a process-global shared across the parallel tests, so every term-dependent test installs the
	/// same one and their order cannot matter.
	fn install_test_terms() {
		let mut m = HashMap::new();
		m.insert("org".to_string(),			"Elearnity Pty Ltd".to_string());
		m.insert("org_short".to_string(),	"Elearnity".to_string());
		m.insert("website".to_string(),		"elearnity.oxegen.io".to_string());
		m.insert("iniverse".to_string(),	"iniverse".to_string());
		set_term_dict(m).expect("install test term dict");
	}

	#[test]
	fn term_dict_aliases_render_without_leaking() {
		install_test_terms();
		// `iniverse` translates to itself, so the display equals the key here whether or not a map is set.
		let runs = parse_inlines("Call it the #g[iniverse], your inner universe.");
		assert!(runs.iter().any(|r| matches!(r, Inline::Glossary { term, display } if term == "iniverse" && display == "iniverse")),
			"glossary alias missing: {:?}", runs);
		// A key that differs from its value now sets the value, not the key.
		assert_eq!(flatten_markup("Visit #t[website] today"), "Visit elearnity.oxegen.io today");
		// A key absent from the dictionary falls back to its own text, capitalised for the `-cap` form.
		assert_eq!(flatten_markup("#tcap[donate] to help"), "Donate to help");
		assert!(is_inline_call("g") && is_inline_call("t") && is_inline_call("graw"));
		assert!(code_skip("#t[website]").is_none());
	}

	/// The term-dictionary family sets a key's value: `t`/`graw` plain, `tcap` capitalised, `g` bold-italic
	/// on first use keyed by the key; an unknown key falls back to the key text and is recorded, never a
	/// panic (the template panics on a miss, the reader must not).
	#[test]
	fn term_dict_resolves_key_to_value() {
		install_test_terms();
		assert_eq!(flatten_markup("Visit #t[website]"), "Visit elearnity.oxegen.io");
		assert_eq!(flatten_markup("#graw[org]"), "Elearnity Pty Ltd");
		let runs = parse_inlines("The #g[org] view.");
		assert!(runs.iter().any(|r| matches!(r, Inline::Glossary { term, display }
				if term == "org" && display == "Elearnity Pty Ltd")),
			"g did not translate the key to its value: {:?}", runs);
		// An unknown key: the key text stands and the miss is recorded on the skip tally.
		let mut skips = SkipSummary::default();
		let runs = parse_inlines_in("A #t[nonesuch] term.", &mut skips);
		assert!(runs.iter().any(|r| matches!(r, Inline::Text(t) if t.contains("nonesuch"))),
			"unknown term-dict key did not fall back to its text: {:?}", runs);
		assert_eq!(skips.total(), 1, "an unknown term-dict key was not recorded");
	}

	/// A blank line between numbered items does not restart the enum: Typst continues the numbering across
	/// the gap, so the items form one list. Real content between two lists still starts a fresh one.
	#[test]
	fn blank_line_between_enum_items_continues_one_list() {
		let src = "+ first item\n\n+ second item\n\n+ third item\n";
		let (items, _) = document_with_skips(src).expect("parse");
		let lists: Vec<&Item> = items.iter().filter(|it| matches!(it, Item::List { .. })).collect();
		assert_eq!(lists.len(), 1, "blank lines split the enum: {:?}", items);
		match lists[0] {
			Item::List { ordered, items, .. } => {
				assert!(*ordered, "the continued list lost its ordered kind");
				assert_eq!(items.len(), 3, "the enum dropped items across the blanks: {:?}", items);
			},
			_ => unreachable!(),
		}
		// A paragraph between two lists still restarts, so genuinely separate lists are not merged.
		let src2 = "+ a\n\n+ b\n\nA paragraph between.\n\n+ c\n";
		let (items2, _) = document_with_skips(src2).expect("parse");
		let lists2 = items2.iter().filter(|it| matches!(it, Item::List { .. })).count();
		assert_eq!(lists2, 2, "prose between two lists did not restart them: {:?}", items2);
	}

	/// An unhandled inline `#func[...]` is consumed and recorded rather than left as raw markup, its
	/// bracketed body folded in so its words survive; a paren-only call sets nothing where it stood.
	#[test]
	fn unknown_inline_call_is_recorded_not_leaked() {
		let mut skips = SkipSummary::default();
		let runs = parse_inlines_in("a #smallcaps[Nato] treaty and a #v(2pt) gap", &mut skips);
		assert!(runs.iter().all(|r| !matches!(r, Inline::Text(t) if t.contains("#smallcaps") || t.contains("#v("))),
			"raw unknown call leaked: {:?}", runs);
		assert!(runs.iter().any(|r| matches!(r, Inline::Text(t) if t.contains("Nato"))),
			"smallcaps body dropped: {:?}", runs);
		assert_eq!(skips.total(), 2);
		let names: Vec<String> = skips.entries().into_iter().map(|(n, _)| n).collect();
		assert!(names.contains(&"#smallcaps".to_string()) && names.contains(&"#v".to_string()),
			"unexpected skip names: {:?}", names);
	}

	/// The reader tallies the code lines and unknown calls it skips, and reports them one line, so a
	/// dropped construct is visible rather than silent. Handled inline calls do not appear in the tally.
	#[test]
	fn skip_summary_reports_skipped_constructs() {
		install_test_terms();	// so `#g[iniverse]` resolves and adds no term-dict miss to the tally
		let src = "#import \"x.typ\": *\n#set page(margin: 1cm)\n\nBody with #g[iniverse] and a #footnote[note].\n\n#show heading: it => it\n";
		let (_, skips) = document_with_skips(src).expect("parse");
		assert_eq!(skips.total(), 3);
		let report = skips.report().expect("a report");
		assert!(report.starts_with("skipped 3 unsupported constructs:"), "report was {:?}", report);
		for name in ["#import", "#set", "#show"] {
			assert!(report.contains(name), "{} missing from {:?}", name, report);
		}
		// A source the reader sets whole has nothing to report.
		let (_, clean) = document_with_skips("Just prose with #g[iniverse].\n").expect("parse");
		assert!(clean.is_empty() && clean.report().is_none());
	}

	/// A `#columns(n)[ ... ]` wrapper is recorded as skipped and its body set single-column, so the words
	/// survive and no raw wrapper leaks into the block stream.
	#[test]
	fn columns_wrapper_flattens_to_single_column() {
		let src = "#columns(2)[\nFirst paragraph here.\n\nSecond paragraph here.\n]\n";
		let (items, skips) = document_with_skips(src).expect("parse");
		let paras = items.iter().filter(|it| matches!(it, Item::Paragraph { .. })).count();
		assert_eq!(paras, 2, "column body not set as paragraphs: {:?}", items);
		assert_eq!(skips.entries(), vec![("#columns".to_string(), 1)]);
	}
}
