//! `clinical` (format 1.1), `task` and `cache` (the task-and-cache contract).
//!
//! Every answer is the engine's: what a cutoff admits is
//! `medh5::clinical::select`, what a task admits per row is
//! `medh5::companion::preflight`, and whether a cache may be used is
//! `medh5::companion::validate_cache` --- the same calls the Python package
//! makes, so the command line, Rust and Python agree by construction.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use clap::ArgMatches;
use serde_json::{json, Value};

use medh5::clinical::model::{Event, Link, HOUR};
use medh5::clinical::select::SelectionPolicy;
use medh5::clinical::ClinicalRecords;
use medh5::collection::{open_any, AnyFile};
use medh5::companion::cache::{validate_cache_in, CacheCheck, CacheEntry, FeatureCache};
use medh5::companion::task::notes_of;
use medh5::companion::view::preflight_subject_in;
use medh5::companion::{
    preflight_each_in, write_preflight_noted, Admitted, Finding, Note, RowView, SourceAccess, TaskManifest,
};
use medh5::h5::file::same_file;
use medh5::json::{pretty, pretty_at, repr_str};
use medh5::sample::Sample;
use medh5::Error;

use crate::common::*;
use crate::workers;

pub fn dispatch(argv: &[String], name: &str, m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let result = match (name, m.subcommand()) {
        ("clinical", Some(("show", sub))) => show(sub, ctx),
        ("clinical", Some(("select", sub))) => select_cmd(sub, ctx),
        ("clinical", Some(("export", sub))) => export(sub, ctx),
        ("clinical", Some(("augment", sub))) => augment(argv, sub, ctx),
        ("clinical", Some(("augment-cohort", sub))) => augment_cohort(argv, sub, ctx),
        ("clinical", Some(("strip", sub))) => strip(sub, ctx),
        ("task", Some(("new", sub))) => task_new(sub, ctx),
        ("task", Some(("validate", sub))) => task_validate(sub, ctx),
        ("task", Some(("preflight", sub))) => task_preflight(sub, ctx),
        ("task", Some(("reconcile", sub))) => task_reconcile(sub, ctx),
        ("task", Some(("repin", sub))) => task_repin(sub, ctx),
        ("cache", Some(("validate", sub))) => cache_validate(sub, ctx),
        ("cache", Some(("info", sub))) => cache_info(sub, ctx),
        ("cache", Some(("ls", sub))) => cache_ls(sub, ctx),
        ("clinical", _) => {
            return Ok(ctx.usage("usage: medh5 clinical {show|select|export|augment|augment-cohort|strip}"))
        }
        ("task", _) => return Ok(ctx.usage("usage: medh5 task {new|validate|preflight|reconcile|repin}")),
        _ => return Ok(ctx.usage("usage: medh5 cache {info|ls|validate}")),
    };
    match result {
        Err(e) if e.is_medh5() || matches!(e, Error::Io(_) | Error::Value(_) | Error::Key(_)) => {
            Ok(ctx.fail(top_level_message(&e)))
        }
        other => other,
    }
}

fn open_sample(m: &ArgMatches) -> medh5::Result<Sample> {
    let path = req_str(m, "path");
    match open_any(Path::new(path), get_str(m, "key"))? {
        AnyFile::Sample(s) => Ok(s),
        AnyFile::Collection(c) => {
            let mut keys = c.keys()?;
            keys.sort();
            Err(Error::invalid(format!(
                "{path} is a collection of {} sample(s); name one with --key (keys: {})",
                keys.len(),
                keys.into_iter().take(5).collect::<Vec<_>>().join(", ")
            )))
        }
    }
}

fn clinical_of(sample: &Sample, path: &str) -> medh5::Result<std::sync::Arc<medh5::clinical::Clinical>> {
    sample.clinical()?.cloned().ok_or_else(|| {
        Error::invalid(format!(
            "{path} does not declare the clinical profile (MEDH5 {})",
            sample.version().unwrap_or_default()
        ))
    })
}

/// `"-48 h"`, `"day 90 + 1 h"`: a time on the subject clock, for a table.
fn when(bounds: Option<medh5::clinical::model::Bounds>) -> String {
    let Some(b) = bounds else { return "?".into() };
    let one = |t: i64| -> String {
        let hours = t as f64 / HOUR as f64;
        if t % HOUR == 0 {
            format!("{} h", t / HOUR)
        } else {
            format!("{hours:.2} h")
        }
    };
    if b.lo == b.hi {
        one(b.lo)
    } else {
        format!("[{}, {}]", one(b.lo), one(b.hi))
    }
}

fn event_row(e: &Event) -> Vec<String> {
    let code = match (&e.code_system, &e.code) {
        (Some(s), Some(c)) => format!("{s}|{c}"),
        _ => "-".into(),
    };
    let value = match (e.value_num, &e.value_text) {
        (Some(v), _) => format!("{}{}", g(v), e.unit.as_deref().map(|u| format!(" {u}")).unwrap_or_default()),
        (None, Some(t)) => t.clone(),
        _ => "-".into(),
    };
    vec![
        e.event_id.clone(),
        e.record_id.clone(),
        e.kind.clone(),
        e.status.clone(),
        when(e.effective_start),
        when(e.available),
        code,
        value,
    ]
}

const EVENT_HEADERS: [&str; 8] = ["event", "record", "kind", "status", "effective", "available", "code", "value"];

fn show(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let sample = open_sample(m)?;
    let clinical = clinical_of(&sample, path)?;
    if flag(m, "json") {
        let mut out = clinical.summary();
        out["events_table"] = Value::Array(clinical.events.iter().map(Event::to_json).collect());
        out["links_table"] = Value::Array(clinical.links.iter().map(Link::to_json).collect());
        ctx.emit(&out, true);
        return Ok(EXIT_OK);
    }
    let summary = clinical.summary();
    let clock = &clinical.descriptor.clock;
    ctx.print(format!("{path}  (MEDH5 {}, clinical profile)", sample.version()?));
    ctx.print(format!(
        "  clock       {} ({}{})",
        clock.id,
        clock.reference,
        clock.origin_description.as_deref().map(|o| format!(": {o}")).unwrap_or_default()
    ));
    ctx.print(format!(
        "  records     {} events in {} records, {} documents, {} links; {} with unknown availability",
        summary["events"], summary["records"], summary["documents"], summary["links"], summary["unknown_availability"]
    ));
    ctx.print("\nevents");
    let rows: Vec<Vec<String>> = clinical.events.iter().map(event_row).collect();
    ctx.print(indent(&table(&rows, &EVENT_HEADERS)));
    if !clinical.documents().is_empty() {
        ctx.print("\ndocuments");
        let rows: Vec<Vec<String>> = clinical
            .documents()
            .iter()
            .map(|d| {
                vec![
                    d.document_id.clone(),
                    d.media_type.clone(),
                    d.language.clone().unwrap_or_else(|| "-".into()),
                    d.source_type.clone().unwrap_or_else(|| "-".into()),
                    thousands(d.n_bytes as i128),
                ]
            })
            .collect();
        ctx.print(indent(&table(&rows, &["document", "media type", "language", "source", "bytes"])));
    }
    if !clinical.links.is_empty() {
        ctx.print("\nlinks");
        let rows: Vec<Vec<String>> = clinical.links.iter().map(|l| vec![l.describe()]).collect();
        ctx.print(indent(&table(&rows, &["link"])));
    }
    Ok(EXIT_OK)
}

fn policy_of(m: &ArgMatches) -> medh5::Result<SelectionPolicy> {
    let mut policy = match get_str(m, "policy_file") {
        Some(file) => {
            let text = std::fs::read_to_string(file).map_err(|e| medh5::error::os_error(&e, file))?;
            let doc = medh5::json::loads(&text).map_err(|e| Error::Value(format!("{file} is not JSON: {e}")))?;
            SelectionPolicy::from_json(&doc)?
        }
        None => SelectionPolicy::strict(),
    };
    if let Some(selection) = get_str(m, "policy") {
        policy.selection = selection.to_string();
    }
    if let Some(w) = get_i64(m, "context_us") {
        policy.context_us = Some(w);
    }
    policy.check()?;
    Ok(policy)
}

fn select_cmd(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    let sample = open_sample(m)?;
    let clinical = clinical_of(&sample, path)?;
    let cutoff = match (get_i64(m, "cutoff_us"), get_f64(m, "cutoff_hours")) {
        (Some(us), None) => us,
        (None, Some(h)) => medh5::clinical::hours(h)?,
        _ => return Ok(ctx.usage("give the cutoff once: --cutoff-us or --cutoff-hours")),
    };
    let policy = policy_of(m)?;
    let selection = clinical.select(cutoff, &policy)?;
    if flag(m, "json") {
        ctx.emit(&selection.to_json(&clinical.events), true);
        return Ok(EXIT_OK);
    }
    ctx.print(format!(
        "{path} at {} ({} us): {} under {}",
        when(Some(medh5::clinical::model::Bounds::exact(cutoff))),
        cutoff,
        selection.status,
        selection.policy
    ));
    let rows: Vec<Vec<String>> = selection.events.iter().map(|s| event_row(&clinical.events[s.index])).collect();
    ctx.print(format!("\nadmitted events ({})", rows.len()));
    ctx.print(indent(&table(&rows, &EVENT_HEADERS)));
    if !selection.payloads.is_empty() {
        let listed: Vec<String> = selection.payloads.iter().map(|(_, k, i)| format!("{k} {}", repr_str(i))).collect();
        ctx.print(format!("\nadmitted payloads: {}", listed.join(", ")));
    }
    if !selection.uncertain_records.is_empty() {
        ctx.print(format!(
            "\nuncertifiable: a later revision of {} has unknown or straddling availability",
            selection.uncertain_records.join(", ")
        ));
    }
    if !selection.excluded.is_empty() {
        let listed: Vec<String> = selection.excluded.iter().map(|(k, v)| format!("{k}={v}")).collect();
        ctx.print(format!("excluded: {}", listed.join(", ")));
    }
    Ok(EXIT_OK)
}

fn export(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = req_str(m, "path");
    // The write truncates whatever `--out` names: the sample itself, named
    // again or through a link, became the JSON describing it, and exit 0.
    if let Some(out) = get_str(m, "out").filter(|out| same_file(Path::new(path), Path::new(out))) {
        return Err(Error::invalid(format!(
            "--out {} is {}, the file being exported; write the records to another file",
            repr_str(out),
            repr_str(path)
        )));
    }
    let sample = open_sample(m)?;
    let records = clinical_of(&sample, path)?.records()?;
    let text = pretty(&records.to_json()) + "\n";
    match get_str(m, "out") {
        Some(out) => {
            medh5::h5::file::write_atomic(Path::new(out), text.as_bytes())?;
            ctx.print(format!(
                "wrote {} events, {} documents and {} links to {out}",
                records.events.len(),
                records.documents.len(),
                records.links.len()
            ));
        }
        None => ctx.print(text.trim_end()),
    }
    Ok(EXIT_OK)
}

/// What `clinical augment --from-meds/--from-fhir` says when no Python
/// package can import.
const NO_IMPORTER: &str = "--from-meds and --from-fhir import in the Python package: install it (`pip install \
                           'medh5[meds]'` for MEDS) and run `medh5 clinical augment` from it, or point MEDH5_PYTHON at \
                           an interpreter that has it";

/// What `clinical augment-cohort` says when no Python package can import.
const NO_COHORT_IMPORTER: &str = "clinical augment-cohort imports in the Python package: install it (`pip install \
                                  'medh5[meds]'` for MEDS) and run it from there, or point MEDH5_PYTHON at an \
                                  interpreter that has it";

/// `clinical augment-cohort`: the importers are Python's, always.
fn augment_cohort(argv: &[String], m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let root = crate::app::command();
    let command = root
        .find_subcommand("clinical")
        .and_then(|c| c.find_subcommand("augment-cohort"))
        .expect("the grammar defines clinical augment-cohort");
    let parsed = crate::convert::parsed_args(command, m);
    let host = ctx.host;
    let (out, err) = ctx.streams();
    Ok(match host.convert(argv, "clinical-augment-cohort", &parsed, out, err) {
        Some(code) => code,
        None => ctx.fail(NO_COHORT_IMPORTER),
    })
}

fn augment(argv: &[String], m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    if get_str(m, "from_meds").is_some() || get_str(m, "from_fhir").is_some() {
        // The importers are Python's, as the format converters are.
        let root = crate::app::command();
        let command = root
            .find_subcommand("clinical")
            .and_then(|c| c.find_subcommand("augment"))
            .expect("the grammar defines clinical augment");
        let parsed = crate::convert::parsed_args(command, m);
        let host = ctx.host;
        let (out, err) = ctx.streams();
        return match host.convert(argv, "clinical-augment", &parsed, out, err) {
            Some(code) => Ok(code),
            None => Ok(ctx.fail(NO_IMPORTER)),
        };
    }
    let path = PathBuf::from(req_str(m, "path"));
    let file = req_str(m, "records");
    let text = std::fs::read_to_string(file).map_err(|e| medh5::error::os_error(&e, file))?;
    let doc = medh5::json::loads(&text).map_err(|e| Error::Value(format!("{file} is not JSON: {e}")))?;
    let records = ClinicalRecords::from_json(&doc)?;
    let out = get_str(m, "out").map(PathBuf::from);
    let report = medh5::clinical::augment::augment(&path, records, out.as_deref())?;
    if flag(m, "json") {
        ctx.emit(&report.to_json(), true);
        return Ok(EXIT_OK);
    }
    ctx.print(format!(
        "{}: MEDH5 {} -> {}; {} events, {} documents, {} links; {} payload digests unchanged",
        report.path,
        report.version_before,
        report.version_after,
        report.events,
        report.documents,
        report.links,
        report.unchanged_digests
    ));
    ctx.print(format!(
        "  content_id {} -> {}",
        report.content_id_before.as_deref().unwrap_or("-"),
        report.content_id_after.as_deref().unwrap_or("-")
    ));
    for line in &report.assumptions {
        ctx.print(format!("  unknown: {line}"));
    }
    Ok(EXIT_OK)
}

fn strip(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = PathBuf::from(req_str(m, "path"));
    let out = PathBuf::from(req_str(m, "out"));
    let report = medh5::clinical::augment::strip(&path, &out)?;
    if flag(m, "json") {
        ctx.emit(&report.to_json(), true);
        return Ok(EXIT_OK);
    }
    ctx.print(format!(
        "wrote the imaging projection to {} (MEDH5 {}): removed {} events, {} documents, {} links",
        medh5::pyval::py_path(&out),
        report.version_after,
        report.events_removed,
        report.documents_removed,
        report.links_removed
    ));
    Ok(EXIT_OK)
}

fn load_task(m: &ArgMatches) -> medh5::Result<(TaskManifest, Option<PathBuf>, Vec<Note>)> {
    let file = Path::new(req_str(m, "manifest"));
    let doc = TaskManifest::read_document(file)?;
    let manifest = TaskManifest::from_json(&doc)?;
    let base = get_str(m, "base").map(PathBuf::from).or_else(|| file.parent().map(Path::to_path_buf));
    Ok((manifest, base, notes_of(&doc)))
}

/// Where a task's sources may be opened from: confined to `--source-root`,
/// unconfined with `--any-source-path`, and otherwise unconfined with a note
/// on stderr naming the sources 3.0 will refuse (task-cache-1 §2).
fn source_access(
    m: &ArgMatches,
    manifest: &TaskManifest,
    base: Option<&Path>,
    ctx: &mut Ctx,
) -> medh5::Result<SourceAccess> {
    if get_str(m, "source_root").is_none() && !flag(m, "any_source_path") {
        // A base that does not exist confines nothing to note: its sources
        // fail to open (T301) whatever 3.0 does.
        if let Ok(confined) = SourceAccess::confined(base, None) {
            let outside = manifest.sources_outside(&confined);
            if let Some(first) = outside.first() {
                ctx.eprint(format!(
                    "note: {} source(s) lie outside the manifest's directory ({}{}): medh5 3.0 confines a task's \
                     sources to its directory; pass --any-source-path to keep reading them, or --source-root DIR",
                    outside.len(),
                    repr_str(&first.uri),
                    if outside.len() > 1 { ", ..." } else { "" }
                ));
            }
        }
    }
    access_of(m, base)
}

/// [`source_access`], without its note.
fn access_of(m: &ArgMatches, base: Option<&Path>) -> medh5::Result<SourceAccess> {
    match get_str(m, "source_root") {
        Some(root) => SourceAccess::confined(base, Some(Path::new(root))),
        None => Ok(SourceAccess::new(base)),
    }
}

fn findings_text(findings: &[medh5::companion::Finding], ctx: &mut Ctx) {
    for f in findings {
        ctx.print(format!("  {}", f.line()));
    }
}

/// What the manifest should state and does not: printed after the findings,
/// and never a failure.
fn notes_text(notes: &[Note], ctx: &mut Ctx) {
    for n in notes {
        ctx.print(format!("  {}", n.line()));
    }
}

fn task_validate(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let file = req_str(m, "manifest");
    let loaded = TaskManifest::read_document(Path::new(file))
        .and_then(|doc| Ok((TaskManifest::from_json(&doc)?, notes_of(&doc))));
    let (manifest, notes) = match loaded {
        Ok(t) => t,
        Err(e) if e.code().is_some() => {
            let finding = medh5::companion::Finding::new(e.code().unwrap_or("T101"), "/", e.message());
            if flag(m, "json") {
                ctx.emit(&json!({"path": file, "ok": false, "findings": [finding.to_json()]}), true);
            } else {
                ctx.print(format!("{file}: FAILED"));
                findings_text(&[finding], ctx);
            }
            return Ok(EXIT_ERROR);
        }
        Err(e) => return Err(e),
    };
    let findings = manifest.validate();
    if flag(m, "json") {
        ctx.emit(
            &json!({
                "path": file,
                "ok": findings.is_empty(),
                "task_fingerprint": manifest.task_fingerprint(),
                "manifest_fingerprint": manifest.manifest_fingerprint(),
                "findings": findings.iter().map(medh5::companion::Finding::to_json).collect::<Vec<_>>(),
                "notes": notes.iter().map(Note::to_json).collect::<Vec<_>>(),
            }),
            true,
        );
    } else {
        ctx.print(format!(
            "{file}: {} ({} {} v{}, {} subjects, {} rows)",
            if findings.is_empty() { "OK" } else { "FAILED" },
            medh5::companion::task::SCHEMA,
            manifest.task_id,
            manifest.task_version,
            manifest.subjects.len(),
            manifest.rows.len()
        ));
        ctx.print(format!("  task fingerprint      {}", manifest.task_fingerprint()));
        ctx.print(format!("  manifest fingerprint  {}", manifest.manifest_fingerprint()));
        findings_text(&findings, ctx);
        notes_text(&notes, ctx);
    }
    Ok(if findings.is_empty() { EXIT_OK } else { EXIT_ERROR })
}

/// Both forms take the subjects one at a time (`preflight_each`): `--json`
/// writes each subject's history as soon as it is merged, and the summary
/// keeps one line per row --- neither holds every subject's records.
fn task_preflight(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let (manifest, base, notes) = load_task(m)?;
    let access = source_access(m, &manifest, base.as_deref(), ctx)?;
    let deep = flag(m, "deep");
    // A manifest with findings reads no subject: nothing to share out.
    if workers::wanted(m) > 1 && manifest.subjects.len() > 1 && manifest.validate().is_empty() {
        return preflight_in_units(m, &manifest, &notes, ctx);
    }
    if flag(m, "json") {
        let ok = write_preflight_noted(&manifest, &access, deep, &notes, &mut Stdout(ctx))?;
        ctx.print("");
        return Ok(if ok { EXIT_OK } else { EXIT_ERROR });
    }
    let mut lines: Vec<Option<(String, Vec<String>)>> = vec![None; manifest.rows.len()];
    let done = preflight_each_in(&manifest, &access, deep, &mut |_, rows| {
        for (r, view) in rows {
            lines[r] = Some(row_line(&view));
        }
        Ok(())
    })?;
    let lines: Vec<(String, Vec<String>)> = lines
        .into_iter()
        .zip(&done.unclaimed)
        .map(|(line, blank)| line.or_else(|| blank.as_ref().map(row_line)).expect("every row"))
        .collect();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (status, _) in &lines {
        *counts.entry(status).or_default() += 1;
    }
    let ok = done.findings.is_empty();
    let counts: Vec<String> = counts.iter().map(|(k, v)| format!("{v} {k}")).collect();
    ctx.print(format!(
        "{}: {} ({} rows: {})",
        req_str(m, "manifest"),
        if ok { "OK" } else { "FAILED" },
        lines.len(),
        if counts.is_empty() { "none".into() } else { counts.join(", ") }
    ));
    findings_text(&done.findings, ctx);
    notes_text(&notes, ctx);
    let rows: Vec<Vec<String>> = lines.into_iter().map(|(_, line)| line).collect();
    ctx.print(String::new());
    ctx.print(table(&rows, &["row", "partition", "cutoff_us", "status", "events", "slots", "target", "why"]));
    Ok(if ok { EXIT_OK } else { EXIT_ERROR })
}

/// `task preflight` on `--workers` processes, a subject to a unit: the same
/// text, or the same bytes of `--json`, as one process writes.  Each
/// subject's history is written as soon as its turn comes; its rows, as
/// text, wait in a temporary file for the subjects before them.
fn preflight_in_units(m: &ArgMatches, manifest: &TaskManifest, notes: &[Note], ctx: &mut Ctx) -> CmdResult {
    let as_json = flag(m, "json");
    let units: Vec<workers::Unit> = manifest
        .subjects
        .iter()
        .enumerate()
        .map(|(i, s)| workers::Unit { task: json!(i), files: Vec::new(), label: s.subject_id.clone() })
        .collect();
    let mut findings: Vec<Finding> = Vec::new();
    let mut statuses: Vec<Option<String>> = vec![None; manifest.rows.len()];
    // Text: each row's cells.  `--json`: where each row's text is in `spill`.
    let mut cells: Vec<Vec<String>> = vec![Vec::new(); manifest.rows.len()];
    let mut spans: Vec<(u64, usize)> = vec![(0, 0); manifest.rows.len()];
    let mut spill = tempfile::tempfile()?;
    let mut end = 0u64;
    let mut subjects = 0usize;
    if as_json {
        let mut out = Stdout(ctx);
        write!(out, "{{\n  \"task_fingerprint\": {}", pretty_at(&json!(manifest.task_fingerprint()), 1))?;
        write!(
            out,
            ",\n  \"manifest_fingerprint\": {},\n  \"subjects\": ",
            pretty_at(&json!(manifest.manifest_fingerprint()), 1)
        )?;
    }
    workers::run_units("task preflight", m, ctx, &units, &mut |ctx, outcome| {
        let part = &outcome.data;
        if as_json {
            let mut out = Stdout(ctx);
            out.write_all(if subjects > 0 { b",\n    " } else { b"[\n    " })?;
            out.write_all(part["history"].as_str().unwrap_or_default().as_bytes())?;
        }
        subjects += 1;
        for row in part["rows"].as_array().into_iter().flatten() {
            let r = row["index"].as_u64().unwrap_or_default() as usize;
            let slot = statuses.get_mut(r).ok_or_else(|| Error::Index(format!("the manifest has no row {r}")))?;
            *slot = Some(row["status"].as_str().unwrap_or_default().to_string());
            if as_json {
                let text = row["json"].as_str().unwrap_or_default();
                spill.write_all(text.as_bytes())?;
                spans[r] = (end, text.len());
                end += text.len() as u64;
            } else {
                cells[r] = row["cells"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|c| c.as_str().unwrap_or_default().to_string())
                    .collect();
            }
        }
        findings.extend(part["findings"].as_array().into_iter().flatten().map(|f| {
            let text = |key: &str| f[key].as_str().unwrap_or_default().to_string();
            Finding { code: text("code"), location: text("location"), message: text("message") }
        }));
        Ok(())
    })?;
    // A valid manifest's every row names a declared subject (T201).
    let statuses: Vec<String> = statuses
        .into_iter()
        .enumerate()
        .map(|(r, s)| {
            s.ok_or_else(|| Error::Runtime(format!("row {} was read by no subject", manifest.rows[r].row_id)))
        })
        .collect::<medh5::Result<_>>()?;
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for status in &statuses {
        *counts.entry(status).or_default() += 1;
    }
    let ok = findings.is_empty();
    if as_json {
        let mut out = Stdout(ctx);
        out.write_all(if subjects > 0 { b"\n  ]" } else { b"[]" })?;
        out.write_all(b",\n  \"rows\": ")?;
        spill.seek(SeekFrom::Start(0))?;
        let mut text = Vec::new();
        for (r, (at, len)) in spans.iter().enumerate() {
            text.resize(*len, 0);
            spill.seek(SeekFrom::Start(*at))?;
            spill.read_exact(&mut text)?;
            out.write_all(if r > 0 { b",\n    " } else { b"[\n    " })?;
            out.write_all(&text)?;
        }
        out.write_all(if spans.is_empty() { b"[]" } else { b"\n  ]" })?;
        let findings: Vec<Value> = findings.iter().map(Finding::to_json).collect();
        let notes: Vec<Value> = notes.iter().map(Note::to_json).collect();
        write!(out, ",\n  \"ok\": {ok},\n  \"counts\": {}", pretty_at(&json!(counts), 1))?;
        write!(out, ",\n  \"findings\": {}", pretty_at(&json!(findings), 1))?;
        write!(out, ",\n  \"notes\": {}", pretty_at(&json!(notes), 1))?;
        out.write_all(b"\n}")?;
        ctx.print("");
        return Ok(if ok { EXIT_OK } else { EXIT_ERROR });
    }
    let counts: Vec<String> = counts.iter().map(|(k, v)| format!("{v} {k}")).collect();
    ctx.print(format!(
        "{}: {} ({} rows: {})",
        req_str(m, "manifest"),
        if ok { "OK" } else { "FAILED" },
        statuses.len(),
        if counts.is_empty() { "none".into() } else { counts.join(", ") }
    ));
    findings_text(&findings, ctx);
    notes_text(notes, ctx);
    ctx.print(String::new());
    ctx.print(table(&cells, &["row", "partition", "cutoff_us", "status", "events", "slots", "target", "why"]));
    Ok(if ok { EXIT_OK } else { EXIT_ERROR })
}

/// `task preflight --workers`: a unit is a subject, by its index in the
/// manifest.  A subject that cannot be read stops a preflight, so one whose
/// worker stopped does.
pub(crate) const PREFLIGHT: workers::Kind = workers::Kind {
    runner: |m| {
        let (manifest, base, _) = load_task(m)?;
        let access = access_of(m, base.as_deref())?;
        let mut rows_of: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, row) in manifest.rows.iter().enumerate() {
            rows_of.entry(row.subject_id.clone()).or_default().push(i);
        }
        let task_fingerprint = manifest.task_fingerprint();
        Ok(Box::new(SubjectReader {
            manifest,
            task_fingerprint,
            rows_of,
            access,
            deep: flag(m, "deep"),
            json: flag(m, "json"),
        }))
    },
    died: |unit, status| {
        Err(Error::Runtime(format!("subject {}: the worker reading it stopped ({status})", repr_str(&unit.label))))
    },
    writes: false,
};

struct SubjectReader {
    manifest: TaskManifest,
    task_fingerprint: String,
    rows_of: HashMap<String, Vec<usize>>,
    access: SourceAccess,
    deep: bool,
    json: bool,
}

impl workers::Runner for SubjectReader {
    fn run(&mut self, task: &Value) -> medh5::Result<workers::FileOutcome> {
        let index = task.as_u64().ok_or_else(|| Error::Value("a preflight task names no subject".into()))? as usize;
        let subject = self.manifest.subjects.get(index).ok_or_else(|| Error::Index(format!("no subject {index}")))?;
        let wanted = self.rows_of.get(&subject.subject_id).map(Vec::as_slice).unwrap_or(&[]);
        let (history, findings, rows) =
            preflight_subject_in(&self.manifest, &self.task_fingerprint, index, wanted, &self.access, self.deep)?;
        let rows: Vec<Value> = rows
            .iter()
            .map(|(r, view)| {
                if self.json {
                    json!({"index": r, "status": view.status, "json": pretty_at(&view.to_json(Some(&history)), 2)})
                } else {
                    json!({"index": r, "status": view.status, "cells": row_line(view).1})
                }
            })
            .collect();
        let mut outcome = workers::FileOutcome::new();
        outcome.data = json!({
            "history": if self.json { Value::String(pretty_at(&history.to_json(), 2)) } else { Value::Null },
            "rows": rows,
            "findings": findings.iter().map(Finding::to_json).collect::<Vec<_>>(),
        });
        Ok(outcome)
    }
}

/// A row's status and its line of the summary table.
fn row_line(r: &RowView) -> (String, Vec<String>) {
    let slots: Vec<String> =
        r.slots.iter().map(|s| format!("{}={}", s.slot, s.image_id.as_deref().unwrap_or("-"))).collect();
    let line = vec![
        r.row_id.clone(),
        r.partition.clone().unwrap_or_else(|| "-".into()),
        r.cutoff_us.to_string(),
        r.status.clone(),
        r.selection.as_ref().map_or(0, |s| s.events.len()).to_string(),
        if slots.is_empty() { "-".into() } else { slots.join(" ") },
        r.target.status.clone(),
        if r.reasons.is_empty() { "-".into() } else { r.reasons.join("; ") },
    ];
    (r.status.clone(), line)
}

fn task_repin(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let (mut manifest, base, _) = load_task(m)?;
    let access = source_access(m, &manifest, base.as_deref(), ctx)?;
    let only: Option<std::collections::BTreeSet<String>> =
        m.get_many::<String>("source").map(|ids| ids.cloned().collect());
    let out = get_str(m, "out").unwrap_or_else(|| req_str(m, "manifest"));
    let dry_run = flag(m, "dry_run");
    if !dry_run {
        manifest.check_output(Path::new(out), &access).map_err(|e| Error::invalid(format!("--out {}", e.message())))?;
    }
    let report = manifest.repin(&access, only.as_ref())?;
    if !dry_run && !report.changed.is_empty() {
        manifest.save(Path::new(out))?;
    }
    if flag(m, "json") {
        let mut doc = report.to_json();
        doc["dry_run"] = json!(dry_run);
        ctx.emit(&doc, true);
        return Ok(EXIT_OK);
    }
    let verb = if dry_run { "would repin" } else { "repinned" };
    for r in &report.changed {
        ctx.print(format!("{verb} {}: {} -> {}", r.source_id, r.before, r.after));
    }
    ctx.print(format!(
        "{}: {} {}, {} unchanged{}",
        req_str(m, "manifest"),
        verb,
        report.changed.len(),
        report.unchanged.len(),
        if report.rows.is_empty() {
            String::new()
        } else {
            format!("; {} row fingerprint(s) change, so their patient-level cache entries go stale", report.rows.len())
        }
    ));
    Ok(EXIT_OK)
}

/// A task drawn from a cohort (`TaskManifest::from_cohort`), its definition
/// from `--definition` and the flags, its rows from `--rows`.
fn task_new(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    use medh5::companion::{CohortTask, Partitions};
    let out = Path::new(req_str(m, "out"));
    let definition = match get_str(m, "definition") {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| medh5::error::os_error(&e, Path::new(path)))?;
            match medh5::json::loads(&text) {
                Ok(Value::Object(doc)) => doc,
                Ok(_) => return Err(Error::invalid(format!("--definition {}: not a JSON object", repr_str(path)))),
                Err(e) => return Err(Error::invalid(format!("--definition {}: not JSON: {e}", repr_str(path)))),
            }
        }
        None => serde_json::Map::new(),
    };
    let named = |flag: &str, key: Option<&str>| -> medh5::Result<String> {
        let given = get_str(m, flag).map(str::to_string);
        let defined = match key {
            Some(key) => definition.get("task").and_then(|t| t.get(key)),
            None => definition.get("identity_namespace"),
        };
        given.or_else(|| defined.and_then(Value::as_str).map(str::to_string)).ok_or_else(|| {
            Error::invalid(format!("the task needs --{flag}, or a definition that states it", flag = flag))
        })
    };
    let (task_id, version, namespace) =
        (named("id", Some("id"))?, named("version", Some("version"))?, named("namespace", None)?);
    let cohort = crate::dataset::load_manifest(req_str(m, "cohort"), ctx)?;
    let split = get_str(m, "split").map(|p| medh5::dataset::split::load(Path::new(p))).transpose()?;
    let fold =
        get_i64(m, "fold").map(|f| u32::try_from(f).map_err(|_| Error::invalid(format!("--fold {f} is not a fold"))));
    let fold = fold.transpose()?;
    let partitions = match (&split, get_str(m, "claims")) {
        (Some(split), _) => Partitions::Split { split, fold },
        (None, Some(_)) if fold.is_some() => {
            return Err(Error::invalid("--fold is a split's: claims already name their partition"))
        }
        (None, Some(set_id)) => Partitions::Claims { set_id },
        (None, None) => return Err(Error::invalid("give --split or --claims")),
    };
    let base = out.parent().map(|p| if p.as_os_str().is_empty() { Path::new(".") } else { p });
    let training = get_str(m, "training_partition").unwrap_or("train");
    let drawn = TaskManifest::from_cohort(
        &cohort,
        partitions,
        &CohortTask::new(&task_id, &version, &namespace).with_base(base).with_training_partition(training),
    )?;
    let mut doc = drawn.to_json();
    for key in ["policy", "slots", "target"] {
        if let Some(value) = definition.get(key) {
            doc[key] = value.clone();
        }
    }
    let description = get_str(m, "description").map(str::to_string).or_else(|| {
        definition.get("task").and_then(|t| t.get("description")).and_then(Value::as_str).map(str::to_string)
    });
    if let Some(d) = description {
        doc["task"]["description"] = json!(d);
    }
    if let Some(follow_up) = get_i64(m, "min_follow_up_us") {
        match doc.get_mut("target") {
            Some(Value::Object(target)) => {
                target.insert("min_follow_up_us".into(), json!(follow_up));
            }
            _ => return Err(Error::invalid("--min-follow-up-us is the target's, and the definition states none")),
        }
    }
    if let Some(note) = notes_of(&doc).into_iter().next() {
        return Err(Error::invalid(format!(
            "{}: state --min-follow-up-us, or min_follow_up_us in the definition",
            note.line()
        )));
    }
    let mut task = TaskManifest::from_json(&doc)?;
    if let Some(rows) = get_str(m, "rows") {
        task.rows = read_rows(Path::new(rows))?;
    }
    let findings = task.validate();
    if let Some(first) = findings.first() {
        return Err(Error::coded(
            &first.code,
            format!(
                "the new task fails validation ({} finding(s)): {} {}: {}",
                findings.len(),
                first.code,
                first.location,
                first.message
            ),
        ));
    }
    task.check_output(out, &SourceAccess::new(base))
        .map_err(|e| Error::invalid(format!("{} {}", req_str(m, "out"), e.message())))?;
    task.save(out)?;
    let mut counts: Vec<(String, usize)> = Vec::new();
    for s in &task.subjects {
        let p = s.partition.clone().unwrap_or_default();
        match counts.iter_mut().find(|(q, _)| *q == p) {
            Some((_, n)) => *n += 1,
            None => counts.push((p, 1)),
        }
    }
    if let Some((_, partitions)) = &task.split {
        counts.sort_by_key(|(p, _)| partitions.iter().position(|q| q == p).unwrap_or(usize::MAX));
    }
    if flag(m, "json") {
        ctx.emit(
            &json!({
                "path": req_str(m, "out"),
                "task_fingerprint": task.task_fingerprint(),
                "manifest_fingerprint": task.manifest_fingerprint(),
                "split": task.split.as_ref().map(|(set_id, partitions)| json!({"set_id": set_id, "partitions": partitions})),
                "subjects": counts.iter().map(|(p, n)| (p.clone(), json!(n))).collect::<serde_json::Map<_, _>>(),
                "sources": task.subjects.iter().map(|s| s.sources.len()).sum::<usize>(),
                "rows": task.rows.len(),
            }),
            true,
        );
        return Ok(EXIT_OK);
    }
    let placed: Vec<String> = counts.iter().map(|(p, n)| format!("{n} in {p}")).collect();
    ctx.print(format!(
        "wrote {}: {} subject(s) ({}), {} row(s); task fingerprint {}",
        req_str(m, "out"),
        task.subjects.len(),
        placed.join(", "),
        task.rows.len(),
        task.task_fingerprint()
    ));
    Ok(EXIT_OK)
}

/// Rows from a CSV of `row_id,subject_id,cutoff_us`, its header first.
fn read_rows(path: &Path) -> medh5::Result<Vec<medh5::companion::task::Row>> {
    let text = std::fs::read_to_string(path).map_err(|e| medh5::error::os_error(&e, path))?;
    let shown = repr_str(&path.to_string_lossy());
    let mut lines = text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty());
    let header: Vec<&str> = lines.next().map(|(_, l)| l.split(',').map(str::trim).collect()).unwrap_or_default();
    if header != ["row_id", "subject_id", "cutoff_us"] {
        return Err(Error::invalid(format!("{shown}: the first line must be the header row_id,subject_id,cutoff_us")));
    }
    let mut rows = Vec::new();
    for (n, line) in lines {
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        let [row_id, subject_id, cutoff] = cells[..] else {
            return Err(Error::invalid(format!("{shown} line {}: three cells, not {}", n + 1, cells.len())));
        };
        let cutoff_us = cutoff.parse::<i64>().map_err(|_| {
            Error::invalid(format!("{shown} line {}: cutoff_us {} is not an integer", n + 1, repr_str(cutoff)))
        })?;
        rows.push(medh5::companion::task::Row {
            row_id: row_id.to_string(),
            subject_id: subject_id.to_string(),
            cutoff_us,
        });
    }
    Ok(rows)
}

fn task_reconcile(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let (mut manifest, base, _) = load_task(m)?;
    let access = source_access(m, &manifest, base.as_deref(), ctx)?;
    // The save replaces whatever `--out` names: a source sample, named again
    // or through a link, became the manifest that pins it, and exit 0.
    let out = get_str(m, "out").unwrap_or_else(|| req_str(m, "manifest"));
    manifest.check_output(Path::new(out), &access).map_err(|e| Error::invalid(format!("--out {}", e.message())))?;
    let recorded = manifest.reconcile_all(&access)?;
    manifest.save(Path::new(out))?;
    ctx.print(format!("recorded {recorded} reconciled event(s); wrote {out}"));
    Ok(EXIT_OK)
}

/// What a cache holds, from its manifest: nothing is decoded.
fn cache_info(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let cache = FeatureCache::open(Path::new(req_str(m, "path")))?;
    let h = &cache.header;
    let sources: std::collections::BTreeSet<(&str, Option<&str>)> = cache
        .entries
        .iter()
        .flat_map(|e| &e.sources)
        .map(|s| (s.content_id.as_str(), s.sample_key.as_deref()))
        .collect();
    let info = json!({
        "path": cache.path.to_string_lossy(),
        "schema": medh5::companion::cache::SCHEMA,
        "level": h.level,
        "encoder": h.encoder,
        "preprocessing": h.preprocessing,
        "output": h.output,
        "task_fingerprint": h.task_fingerprint,
        "selection": h.selection,
        "fitted_on": h.fitted_on,
        "manifest_digest": cache.manifest_digest,
        "entries": cache.entries.len(),
        "sources": sources.len(),
    });
    if flag(m, "json") {
        ctx.emit(&info, true);
        return Ok(EXIT_OK);
    }
    let one = |v: &Value| match v {
        Value::Null => "-".to_string(),
        Value::String(s) => s.clone(),
        other => medh5::json::canonical(other),
    };
    ctx.print(format!(
        "{}: {} {}-level, {} entries over {} source versions",
        req_str(m, "path"),
        medh5::companion::cache::SCHEMA,
        h.level,
        cache.entries.len(),
        sources.len()
    ));
    for key in ["encoder", "preprocessing", "output", "task_fingerprint", "selection", "fitted_on", "manifest_digest"] {
        ctx.print(format!("  {key:<16}  {}", one(&info[key])));
    }
    Ok(EXIT_OK)
}

/// A cache's entries, as its manifest records them, optionally only those
/// reading one source version or of one row.
fn cache_ls(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let cache = FeatureCache::open(Path::new(req_str(m, "path")))?;
    let source = get_str(m, "source");
    let row = get_str(m, "row");
    let shown: Vec<&CacheEntry> = cache
        .entries
        .iter()
        .filter(|e| source.is_none_or(|c| e.sources.iter().any(|s| s.content_id == c)))
        .filter(|e| row.is_none_or(|r| e.row_id.as_deref() == Some(r)))
        .collect();
    if flag(m, "json") {
        ctx.emit(&Value::Array(shown.iter().map(|e| e.to_json()).collect()), true);
        return Ok(EXIT_OK);
    }
    let rows: Vec<Vec<String>> = shown
        .iter()
        .map(|e| {
            let what = match (&e.event_id, &e.row_id) {
                (Some(event), _) => format!("{event} {}", e.document_id.as_deref().unwrap_or("-")),
                (None, Some(row)) => format!("{row} @{}", e.cutoff_us.map_or("-".into(), |c| c.to_string())),
                (None, None) => "-".into(),
            };
            let pins: Vec<String> = e.sources.iter().map(|s| format!("{} {}", s.locator(), s.content_id)).collect();
            vec![e.entry_id.clone(), pins.join("; "), what, e.digest.clone()]
        })
        .collect();
    let what = if cache.header.level == "event" { "event document" } else { "row @cutoff" };
    ctx.print(table(&rows, &["entry", "sources", what, "digest"]));
    Ok(EXIT_OK)
}

fn cache_validate(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let path = Path::new(req_str(m, "path"));
    let mut base = get_str(m, "base").map(PathBuf::from);
    let task = match get_str(m, "task") {
        Some(file) => {
            let file = Path::new(file);
            let manifest = TaskManifest::load(file)?;
            let task_base = get_str(m, "task_base").map(PathBuf::from).or_else(|| file.parent().map(Path::to_path_buf));
            let access = source_access(m, &manifest, task_base.as_deref(), ctx)?;
            let admitted = Admitted::preflight_in(&manifest, &access, false)?;
            // An entry source the task pins is opened where the task says
            // (task-cache-1 §7.1); any other resolves by its own URI, which
            // copied the task's, relative to its base.
            base = base.or(task_base);
            Some((manifest, access, admitted))
        }
        None => None,
    };
    let base = base.or_else(|| path.parent().map(Path::to_path_buf));
    let entries = match get_str(m, "source_root") {
        Some(root) => SourceAccess::confined(base.as_deref(), Some(Path::new(root)))?,
        None => SourceAccess::new(base.as_deref()),
    };
    let mut check = CacheCheck::new(entries);
    if let Some((manifest, access, admitted)) = &task {
        check = check.with_task(manifest, access).with_admitted(admitted);
    }
    let report = validate_cache_in(path, &check)?;
    if flag(m, "json") {
        ctx.emit(&report.to_json(), true);
    } else {
        ctx.print(format!(
            "{}: {} ({} entries{}{})",
            report.path,
            if report.ok() { "OK" } else { "FAILED" },
            report.entries,
            if report.stale().is_empty() { String::new() } else { format!(", {} stale", report.stale().len()) },
            if report.corrupt().is_empty() { String::new() } else { format!(", {} corrupt", report.corrupt().len()) }
        ));
        findings_text(&report.findings, ctx);
    }
    Ok(if report.ok() { EXIT_OK } else { EXIT_ERROR })
}
