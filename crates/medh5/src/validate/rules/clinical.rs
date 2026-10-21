//! 1.1 §3--§8 --- the clinical profile (E8xx, W914).
//!
//! Three rules, one per level: `check_clinical` (structural) checks the
//! declaration, the descriptor and the column encoding and keeps the tables it
//! read; `check_clinical_records` (semantic) checks the rows against each other
//! and the sample; `check_clinical_digests` (integrity) requires the digest
//! every clinical dataset carries.  Actual digest *matches* are E701, checked
//! with every other dataset's.

use std::collections::{BTreeSet, HashMap};

use super::Context;
use crate::clinical::check::{check_descriptor, check_records_with, DocumentTexts, SampleContext};
use crate::clinical::columns::{
    read_table_deferring, Problem, RawTable, TextColumn, Values, DOCUMENT_COLUMNS, EVENT_COLUMNS, LINK_COLUMNS,
    SCAN_BYTES,
};
use crate::clinical::model::{
    ClinicalRecords, Descriptor, DESCRIPTOR, DOCUMENTS, EVENTS, GROUP, LINKS, MIN_VERSION, PROFILE,
};
use crate::clinical::schema::{validate_descriptor, validate_descriptor_projection};
use crate::clinical::table::{documents_of, events_of, links_of};
use crate::h5::attrs;
use crate::h5::data::{self, Kind};
use crate::h5::ops::{self, Node};
use crate::json::repr_str;
use crate::validate::Diagnostic;
use crate::Result;

/// The clinical tables as the structural rule read them: document text
/// deferred, and what its one scan found.
#[derive(Debug, Clone, Default)]
pub struct ClinicalTables {
    /// `clinical/meta`, when it parsed and its schema accepted it.
    pub descriptor: Option<Descriptor>,
    /// `clinical/events`, when it is a group.
    pub events: Option<RawTable>,
    /// `clinical/documents`, its `text` deferred, when present.
    pub documents: Option<RawTable>,
    /// `clinical/links`, when present.
    pub links: Option<RawTable>,
    /// What the span rule needs of the stored document text.
    pub texts: ScannedTexts,
}

/// What the span rule needs of the stored document text, from one scan.
#[derive(Debug, Clone, Default)]
pub struct ScannedTexts {
    /// Document id -> its byte range in the buffer (the first row of an id).
    spans: HashMap<String, (u64, u64)>,
    /// Probed buffer positions that fall inside a character.
    inside: BTreeSet<u64>,
}

impl DocumentTexts for ScannedTexts {
    fn n_bytes(&self, document_id: &str) -> Option<u64> {
        self.spans.get(document_id).map(|(a, b)| b - a)
    }

    fn starts_character(&self, document_id: &str, at: u64) -> bool {
        self.spans.get(document_id).is_some_and(|(a, _)| !self.inside.contains(&(a + at)))
    }
}

/// Scan the deferred `text` of a documents table: every row UTF-8 (E806,
/// recorded as the table's problem), and where the links' spans fall.  Memory
/// is one slab of the buffer, whatever its size.
fn scan_texts(group: &hdf5::Group, documents: &mut RawTable, links: Option<&RawTable>) -> Result<ScannedTexts> {
    let mut out = ScannedTexts::default();
    let Some(Values::Deferred { offsets }) = documents.column("text").map(|c| c.values.clone()) else {
        return Ok(out);
    };
    for i in 0..documents.rows {
        if let (Some(id), Some(span)) =
            (documents.text("document_id", i), documents.column("text").and_then(|c| c.span(i)))
        {
            out.spans.entry(id).or_insert(span);
        }
    }
    let mut probes = Vec::new();
    if let Some(links) = links {
        for i in 0..links.rows {
            if links.text("source_type", i).as_deref() != Some("document") {
                continue;
            }
            let (Some(id), Some(start), Some(end)) =
                (links.text("source_id", i), links.u64("source_start", i), links.u64("source_end", i))
            else {
                continue;
            };
            if let Some((a, b)) = out.spans.get(&id) {
                // `p < b - a`, not `a + p < b`: a span is any `uint64` the
                // file states, and the sum wrapped to a probe inside the text.
                probes.extend([start, end].into_iter().filter(|p| *p > 0 && *p < b - a).map(|p| a + p));
            }
        }
    }
    let column = TextColumn::open(group, "text", offsets, "/clinical/documents/text")?;
    let scan = column.scan(&probes, SCAN_BYTES)?;
    if let Some(i) = scan.invalid_row {
        documents.problems.push(Problem::new(
            "E806",
            format!("/clinical/documents/text#row={i}"),
            format!("row {i} of `text` is not valid UTF-8"),
        ));
    }
    out.inside = scan.inside_character;
    Ok(out)
}

impl ClinicalTables {
    /// Whether every table read without a structural problem.
    pub fn sound(&self) -> bool {
        [&self.events, &self.documents, &self.links].iter().all(|t| t.as_ref().is_none_or(RawTable::is_sound))
    }
}

/// The members `clinical/` defines.
const MEMBERS: [&str; 4] = [DESCRIPTOR, EVENTS, DOCUMENTS, LINKS];

/// Whether the sample is held to the clinical profile: declared, or required
/// by the caller.
fn held(ctx: &Context) -> bool {
    ctx.profiles.iter().any(|p| p == PROFILE)
}

/// Structural: the declaration, the descriptor, the members and the column
/// encoding (E009, E801--E808, E819, W913).
pub fn check_clinical(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let group = ops::child_group(&ctx.root, GROUP);
    let version = ctx.version.clone().unwrap_or_default();
    let modern = crate::version::at_least(&version, MIN_VERSION);
    let Some(group) = group else {
        if held(ctx) {
            out.push(ctx.err("E009", "/", "profile `clinical` requires the `clinical/` group, and it is absent"));
        } else if ops::exists(&ctx.root, GROUP) && modern {
            out.push(ctx.err("E803", "/clinical", "`clinical` is reserved by MEDH5 1.1 for the clinical profile"));
        }
        return Ok(out);
    };
    // E803 is about what the file declares, whatever a caller holds it to.  A
    // 1.0 file may hold an extension of that name; a 1.1 file reserves it, and
    // a recognised descriptor is the profile whatever the version.
    let claimed = attrs::get_strs(&ctx.root, "medh5_profiles")?.unwrap_or_default().iter().any(|p| p == PROFILE);
    if !claimed && (modern || crate::clinical::recognised(&ctx.root)) {
        out.push(ctx.err(
            "E803",
            "/clinical",
            "the sample holds clinical content but does not declare the `clinical` profile in `medh5_profiles`",
        ));
    }
    if !held(ctx) {
        return Ok(out);
    }
    if !modern {
        out.push(ctx.err(
            "E009",
            "/",
            format!("profile `clinical` is a MEDH5 1.1 profile, but the file declares MEDH5 {}", repr_str(&version)),
        ));
    }
    check_attributes(ctx, &group, &mut out)?;
    for name in ops::members(&group)? {
        if !MEMBERS.contains(&name.as_str()) {
            out.push(ctx.unknown(
                "E804",
                format!("/clinical/{name}"),
                format!("{} is not a member the clinical profile defines", repr_str(&name)),
            ));
        }
    }
    let mut tables =
        ClinicalTables { descriptor: check_descriptor_dataset(ctx, &group, &mut out)?, ..Default::default() };
    let table = |name: &str| -> Option<hdf5::Group> { ops::child_group(&group, name) };
    for (name, specs, required) in [
        (EVENTS, &EVENT_COLUMNS[..], true),
        (DOCUMENTS, &DOCUMENT_COLUMNS[..], false),
        (LINKS, &LINK_COLUMNS[..], false),
    ] {
        let location = format!("/clinical/{name}");
        let Some(g) = table(name) else {
            if ops::exists(&group, name) {
                out.push(ctx.err("E804", location, format!("`{name}` must be a group of columns")));
            } else if required {
                out.push(ctx.err("E804", location, format!("required table `{name}` is absent")));
            }
            continue;
        };
        // Text is checked below, in one bounded scan: never held whole.
        let deferred: &[&str] = if name == DOCUMENTS { &["text"] } else { &[] };
        let raw = read_table_deferring(&g, name, specs, ctx.projection, deferred)?;
        for p in &raw.problems {
            out.push(ctx.err(p.code, p.location.clone(), p.message.clone()));
        }
        if name == EVENTS && raw.is_sound() && raw.rows == 0 {
            out.push(ctx.err("E009", location, "the clinical profile needs at least one event"));
        }
        match name {
            EVENTS => tables.events = Some(raw),
            DOCUMENTS => tables.documents = Some(raw),
            _ => tables.links = Some(raw),
        }
    }
    if let (Some(documents), Some(g)) = (tables.documents.as_mut(), table(DOCUMENTS)) {
        let found = documents.problems.len();
        tables.texts = scan_texts(&g, documents, tables.links.as_ref())?;
        for p in &documents.problems[found..] {
            out.push(ctx.err(p.code, p.location.clone(), p.message.clone()));
        }
    }
    ctx.clinical = Some(tables);
    Ok(out)
}

/// E819: only `digest`, and only on datasets.  Clinical semantics live in the
/// columns and the descriptor, which `content_id` covers; an attribute would
/// carry meaning no digest attests (§3, §8).
fn check_attributes(ctx: &Context, group: &hdf5::Group, out: &mut Vec<Diagnostic>) -> Result<()> {
    let mut report = |path: &str, names: Vec<String>, is_dataset: bool| {
        for name in names {
            if is_dataset && name == "digest" {
                continue;
            }
            out.push(ctx.unknown(
                "E819",
                format!("/clinical{path}@{name}"),
                format!(
                    "attribute {} is not defined by the clinical profile, whose facts live in its columns",
                    repr_str(&name)
                ),
            ));
        }
    };
    report("", attrs::names(group)?, false);
    let mut found: Vec<(String, Vec<String>, bool)> = Vec::new();
    ops::visit(group, &mut |path, node| {
        match node {
            Node::Group(g) => found.push((format!("/{path}"), attrs::names(g)?, false)),
            Node::Dataset(d) => found.push((format!("/{path}"), attrs::names(d)?, true)),
        }
        Ok(true)
    })?;
    for (path, names, is_dataset) in found {
        report(&path, names, is_dataset);
    }
    Ok(())
}

/// E801/E802: `clinical/meta` is a scalar UTF-8 string of canonical JSON that
/// the descriptor schema accepts.
fn check_descriptor_dataset(
    ctx: &Context,
    group: &hdf5::Group,
    out: &mut Vec<Diagnostic>,
) -> Result<Option<Descriptor>> {
    let location = "/clinical/meta";
    let Some(meta) = ops::child_dataset(group, DESCRIPTOR) else {
        out.push(ctx.err("E801", location, "`clinical/meta` is absent"));
        return Ok(None);
    };
    if !matches!(data::kind(&meta), Ok(Kind::Strings)) || !meta.is_scalar() {
        out.push(ctx.err("E801", location, "`clinical/meta` is a scalar UTF-8 string dataset"));
        return Ok(None);
    }
    // Read as stored: a lossy read put U+FFFD in place of bytes that are not
    // UTF-8, and the descriptor parsed.
    let stored = data::read_stored_strings(&meta)?.into_iter().next().unwrap_or_default();
    let text = match String::from_utf8(stored) {
        Ok(text) => text,
        Err(e) => {
            out.push(ctx.err(
                "E801",
                location,
                format!("`clinical/meta` is not valid UTF-8 (byte {} is not)", e.utf8_error().valid_up_to()),
            ));
            return Ok(None);
        }
    };
    let value = match crate::json::loads(&text) {
        Ok(v) => v,
        Err(e) => {
            out.push(ctx.err("E801", location, format!("`clinical/meta` is not valid JSON: {e}")));
            return Ok(None);
        }
    };
    let canonical = crate::json::canonical(&value);
    if canonical != text {
        out.push(ctx.err(
            "E801",
            location,
            "`clinical/meta` is not canonical JSON (sorted keys, no whitespace; 1.0 §5.1), so equal descriptors \
             would digest differently",
        ));
    }
    let (errors, tolerated) =
        if ctx.projection { validate_descriptor_projection(&value) } else { (validate_descriptor(&value), Vec::new()) };
    let split = |m: &str| match m.split_once(": ") {
        Some((l, d)) => (format!("{location}#{l}"), d.to_string()),
        None => (location.to_string(), m.to_string()),
    };
    for message in &errors {
        let (l, d) = split(message);
        out.push(ctx.err("E802", l, d));
    }
    for message in &tolerated {
        let (l, d) = split(message);
        out.push(ctx.unknown("E802", l, d));
    }
    Ok(Descriptor::from_json(&value).ok().filter(|_| errors.is_empty()))
}

/// E811 at the column level: one bound of a pair valid and the other null
/// cannot reach a typed event at all, so it is checked on the stored columns.
fn half_null_bounds(ctx: &Context, events: &RawTable, out: &mut Vec<Diagnostic>) {
    for (lo, hi) in [
        ("effective_start_lo_us", "effective_start_hi_us"),
        ("effective_end_lo_us", "effective_end_hi_us"),
        ("available_lo_us", "available_hi_us"),
    ] {
        for i in 0..events.rows {
            if events.i64(lo, i).is_some() != events.i64(hi, i).is_some() {
                let id = events.text("event_id", i);
                out.push(ctx.err(
                    "E811",
                    crate::clinical::check::row_location("/clinical/events", id.as_deref(), i),
                    format!("`{lo}` and `{hi}` are both valid or both null"),
                ));
            }
        }
    }
}

/// A link's span is both endpoints or neither (1.1 §7.1, E814), checked on
/// the stored columns: a link record has a span or none, so a half-null pair
/// reached the span rules as no span at all, and an export dropped the
/// endpoint that was there.
fn half_null_spans(ctx: &Context, links: &RawTable, out: &mut Vec<Diagnostic>) {
    for i in 0..links.rows {
        let (start, end) = (links.u64("source_start", i), links.u64("source_end", i));
        if start.is_some() != end.is_some() {
            out.push(ctx.err(
                "E814",
                crate::clinical::check::row_location("/clinical/links", None, i),
                format!(
                    "`source_start` and `source_end` are both valid or both null; this link has only `{}`",
                    if start.is_some() { "source_start" } else { "source_end" }
                ),
            ));
        }
    }
}

/// The sample as the record rules see it.
///
/// Timepoints, provenance activities and the de-identification a shifted
/// clock rests on are declared in `/meta`.  When it could not be read (E004 or
/// E005, reported) there is nothing to check a reference into it against, so
/// those references are taken as resolving, rather than every one reported as
/// dangling, and the rules that need no document still run.
fn sample_context(ctx: &Context, records: &ClinicalRecords) -> Result<SampleContext> {
    let mut sample = SampleContext::from_root(&ctx.root, ctx.document.as_ref())?;
    if ctx.document.is_none() {
        let named = records.links.iter().flat_map(|l| [(&l.source_type, &l.source_id), (&l.target_type, &l.target_id)]);
        sample.timepoints.extend(named.filter(|(kind, _)| *kind == "timepoint").map(|(_, id)| id.clone()));
        sample.timepoints.extend(records.events.iter().filter_map(|e| e.timepoint_id.clone()));
        sample.activities.extend(records.events.iter().filter_map(|e| e.prov.clone()));
        sample.deidentified = true;
    }
    Ok(sample)
}

/// Semantic: the rows against each other and the sample (E808--E817, W914).
///
/// The tables are taken from the context for the check and put back after
/// it, not cloned: a copy of every clinical table was made for each pass.
pub fn check_clinical_records(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let Some(tables) = ctx.clinical.take() else { return Ok(Vec::new()) };
    let checked = check_records(ctx, &tables);
    ctx.clinical = Some(tables);
    checked
}

fn check_records(ctx: &Context, tables: &ClinicalTables) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let Some(events) = &tables.events else { return Ok(out) };
    if !tables.sound() {
        return Ok(out); // the structural findings come first
    }
    half_null_bounds(ctx, events, &mut out);
    if let Some(links) = &tables.links {
        half_null_spans(ctx, links, &mut out);
    }
    let records = ClinicalRecords {
        descriptor: tables.descriptor.clone().unwrap_or_else(|| {
            Descriptor::new(crate::clinical::Clock {
                id: String::new(),
                unit: "us".into(),
                reference: "utc".into(),
                origin_description: None,
            })
        }),
        events: events_of(events),
        documents: tables.documents.as_ref().map(documents_of).unwrap_or_default(),
        links: tables.links.as_ref().map(links_of).unwrap_or_default(),
    };
    let sample = sample_context(ctx, &records)?;
    if let Some(descriptor) = &tables.descriptor {
        for f in check_descriptor(descriptor, &sample, "/clinical/meta") {
            out.push(ctx.err(f.code, f.location, f.message));
        }
    }
    for f in check_records_with(&records, &tables.texts, &sample, "/clinical") {
        let d = if f.code == "E810" {
            ctx.unknown(f.code, f.location, f.message)
        } else {
            ctx.err(f.code, f.location, f.message)
        };
        out.push(d);
    }
    Ok(out)
}

/// Integrity: every dataset under `clinical/` carries its §13.1 digest (E818).
pub fn check_clinical_digests(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    if !held(ctx) {
        return Ok(out);
    }
    let Some(group) = ops::child_group(&ctx.root, GROUP) else { return Ok(out) };
    let mut missing = Vec::new();
    // Soft links followed, as the readers follow them: a column linked to
    // storage elsewhere is a column all the same.
    ops::visit_resolving(&group, &mut |path, node| {
        if let Node::Dataset(d) = node {
            if !attrs::has(d, "digest") {
                missing.push(format!("/clinical/{path}"));
            }
        }
        Ok(true)
    })?;
    for path in missing {
        out.push(ctx.err(
            "E818",
            path,
            "every dataset under `clinical/` carries its digest --- buffers, offsets, masks and the descriptor alike \
             --- so a change to any of them is detectable",
        ));
    }
    Ok(out)
}
