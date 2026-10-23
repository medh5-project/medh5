//! `convert` and `migrate` --- the importers and exporters.
//!
//! Every command writes a conversion report, because the interesting part of
//! an import is not that it succeeded but what it had to decide.  `migrate`
//! (0.x files, Appendix B) and the NIfTI converters are native; the other
//! format converters are Python integrations and run in the Python package,
//! through the [`Host`].

use std::path::{Component, Path, PathBuf};

use clap::{ArgAction, ArgMatches};
use serde_json::{json, Map, Value};

use medh5::convert::legacy::{
    build_label_set, load_sidecar, migrate_group, migrate_paths, plan_migration, write_sidecar, MigrationGroup,
    MigrationPlan,
};
use medh5::convert::nifti::{self, ExportOptions, ImportOptions, SegOptions};
use medh5::convert::report::{ConversionReport, Note};
use medh5::convert::{Occasion, SubjectGroup};
use medh5::h5::file::same_file;
use medh5::json::{pretty, repr_str};
use medh5::labels::registry::load_file;
use medh5::labels::{ClassKey, LabelSet};
use medh5::sample::writer::Annotated;
use medh5::Error;

use crate::app;
use crate::common::*;
use crate::workers;

/// What `--help` cannot say for a converter the standalone binary cannot run.
const NO_PYTHON: &str = "the format converters (DICOM, DICOM SEG, RTSTRUCT, ITK, nnU-Net) are part of the Python \
                         package: install it (`pip install 'medh5[dicom]'`) and run `medh5 convert` from it, or point \
                         MEDH5_PYTHON at an interpreter that has it";

/// The converters the engine runs itself, with or without Python.
const NATIVE: [&str; 3] = ["from-nifti", "import-seg-nifti", "to-nifti"];

/// The parsed arguments of one subcommand, keyed as the 1.x parser named
/// them, for a host that runs the command itself.
pub fn parsed_args(command: &clap::Command, m: &ArgMatches) -> Value {
    let mut out = Map::new();
    for arg in command.get_arguments() {
        let id = arg.get_id().as_str();
        if id == "help" || id == "version" {
            continue;
        }
        let value = match arg.get_action() {
            ArgAction::SetTrue => Value::Bool(m.get_flag(id)),
            ArgAction::Append => match m.get_many::<String>(id) {
                Some(v) => Value::Array(v.map(|s| Value::String(s.clone())).collect()),
                None => Value::Null,
            },
            _ if arg.get_num_args().is_some_and(|n| n.max_values() > 1) => match m.get_many::<String>(id) {
                Some(v) => Value::Array(v.map(|s| Value::String(s.clone())).collect()),
                None => Value::Null,
            },
            _ => {
                if let Ok(Some(v)) = m.try_get_one::<String>(id) {
                    Value::String(v.clone())
                } else if let Ok(Some(v)) = m.try_get_one::<i64>(id) {
                    Value::from(*v)
                } else if let Ok(Some(v)) = m.try_get_one::<f64>(id) {
                    medh5::json::num(*v)
                } else {
                    Value::Null
                }
            }
        };
        out.insert(id.to_string(), value);
    }
    Value::Object(out)
}

pub fn convert(argv: &[String], m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let Some((name, sub)) = m.subcommand() else {
        return Ok(ctx.usage("usage: medh5 convert COMMAND ... (see --help)"));
    };
    if NATIVE.contains(&name) {
        return convert_nifti(name, sub, ctx);
    }
    let root = app::command();
    let command = root
        .find_subcommand("convert")
        .and_then(|c| c.find_subcommand(name))
        .expect("the grammar defines every convert command");
    let parsed = parsed_args(command, sub);
    let host = ctx.host;
    let (out, err) = ctx.streams();
    match host.convert(argv, name, &parsed, out, err) {
        Some(code) => Ok(code),
        None => Ok(ctx.fail(NO_PYTHON)),
    }
}

/// Refuse a file `migrate` would write that is one it reads, before anything
/// is written: `--report` or `--write-labels` naming an input replaced it
/// with JSON, and a `--report` directory was refused only after the cohort
/// had been migrated.
fn check_writes(m: &ArgMatches, sources: &[&Path]) -> Result<(), String> {
    let report = get_str(m, "report").map(Path::new);
    let labels = get_str(m, "write_labels").map(Path::new);
    if let Some(path) = report {
        let shown = repr_str(&path.to_string_lossy());
        if path.is_dir() {
            return Err(format!("--report {shown} is a directory; name a file"));
        }
        // `migrate` creates its output directory, so a report may go there.
        let lexical = |p: &Path| -> PathBuf { p.components().filter(|c| *c != Component::CurDir).collect() };
        let out = get_str(m, "out").map(|o| lexical(Path::new(o)));
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty() && !p.is_dir()) {
            if out != Some(lexical(parent)) {
                return Err(format!("--report {shown} cannot be written: its directory does not exist"));
            }
        }
    }
    if let (Some(r), Some(l)) = (report, labels) {
        if r == l || same_file(r, l) {
            return Err(format!("--report and --write-labels both name {}", repr_str(&r.to_string_lossy())));
        }
    }
    let read: Vec<&Path> = sources.iter().copied().chain(get_str(m, "label_set").map(Path::new)).collect();
    for (flag, path) in [("--report", report), ("--write-labels", labels)] {
        let Some(path) = path else { continue };
        if let Some(input) = read.iter().find(|input| same_file(input, path)) {
            return Err(format!(
                "{flag} {} is the input {}; name a new file",
                repr_str(&path.to_string_lossy()),
                repr_str(&input.to_string_lossy())
            ));
        }
    }
    Ok(())
}

/// Write the report where asked, print it, and exit on whether it is ok.
pub fn finish(report: &ConversionReport, m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    if let Some(path) = get_str(m, "report") {
        // A report naming an output would replace the sample just written.
        if let Some(output) = report.outputs.iter().find(|o| same_file(Path::new(o), Path::new(path))) {
            return Ok(ctx.fail(format!(
                "--report {} is the output {}; the report was not written",
                repr_str(path),
                repr_str(output)
            )));
        }
        medh5::h5::file::write_atomic(Path::new(path), (pretty(&report.to_json()) + "\n").as_bytes())?;
    }
    if flag(m, "json") {
        ctx.emit(&report.to_json(), true);
    } else {
        ctx.print(report.format(true));
    }
    Ok(if report.ok() { EXIT_OK } else { EXIT_ERROR })
}

pub fn migrate(m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    let paths = get_paths(m, "paths");
    let sources: Vec<&Path> = paths.iter().map(Path::new).collect();
    if let Err(message) = check_writes(m, &sources) {
        return Ok(ctx.fail(message));
    }
    match migrated(m, &sources, ctx) {
        Ok(report) => finish(&report, m, ctx),
        Err(e) if e.is_medh5() || matches!(e, medh5::Error::Io(_)) => Ok(ctx.fail(e.to_string())),
        Err(e) => Err(e),
    }
}

/// What `migrate` reports: the minted label set's review, or the cohort's
/// migration, its samples written in `--workers` processes.
fn migrated(m: &ArgMatches, sources: &[&Path], ctx: &mut Ctx) -> medh5::Result<ConversionReport> {
    if let Some(target) = get_str(m, "write_labels") {
        let mut report = ConversionReport::new("migrate", "");
        let label_set = build_label_set(sources, Some(&mut report))?;
        let written = write_sidecar(&label_set, Path::new(target))?;
        let note = format!("{}: {} classes --- review before migrating", written.display(), label_set.len());
        // With `--json`, stdout is the report alone.
        if flag(m, "json") {
            ctx.eprint(note);
        } else {
            ctx.print(note);
        }
        return Ok(report);
    }
    let reviewed = match get_str(m, "label_set") {
        Some(p) => Some(load_sidecar(Path::new(p))?),
        None => None,
    };
    let (out, group_by, subject_key, codec) =
        (Path::new(req_str(m, "out")), req_str(m, "group_by"), get_str(m, "subject_key"), req_str(m, "codec"));
    if workers::wanted(m) == 1 {
        return migrate_paths(sources, out, group_by, subject_key, reviewed.as_ref(), codec);
    }
    // The plan is the cohort's: its label set, its grouping and the samples'
    // names.  Each sample is a unit, its inputs and output its files.
    let MigrationPlan { mut log, labels, groups } =
        plan_migration(sources, out, group_by, subject_key, reviewed.as_ref())?;
    let labels = labels.to_json(None);
    let units: Vec<workers::Unit> = groups
        .iter()
        .map(|g| workers::Unit {
            task: json!({"group": group_to_json(g), "labels": labels}),
            files: g.sources.iter().chain([&g.target]).map(|p| p.to_string_lossy().into_owned()).collect(),
            label: g.name.clone(),
        })
        .collect();
    workers::run_units("migrate", m, ctx, &units, &mut |_, outcome| {
        let part = &outcome.data;
        log.outputs
            .extend(part["outputs"].as_array().into_iter().flatten().filter_map(Value::as_str).map(String::from));
        log.notes.extend(part["notes"].as_array().into_iter().flatten().map(note_from_json));
        Ok(())
    })?;
    Ok(log)
}

/// `migrate --workers`: a unit is one sample of the plan, a subject's files.
pub(crate) const MIGRATE: workers::Kind = workers::Kind {
    runner: |m| Ok(Box::new(Migrator { codec: req_str(m, "codec").to_string() })),
    died: |unit, status| {
        let inputs: Vec<&str> = unit.task["group"]["occasions"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|o| o["key"].as_str())
            .collect();
        let mut part = ConversionReport::new("migrate", "");
        part.warn(
            "failed",
            format!("{}: the worker writing it stopped ({status})", inputs.join(", ")),
            json!({"inputs": inputs, "output": unit.task["group"]["target"]}),
        );
        Ok(migrated_part(&part))
    },
    writes: true,
};

struct Migrator {
    codec: String,
}

impl workers::Runner for Migrator {
    fn run(&mut self, task: &Value) -> medh5::Result<workers::FileOutcome> {
        let group = group_from_json(&task["group"])?;
        let labels = LabelSet::from_json(Some(&task["labels"]))?
            .ok_or_else(|| Error::Value("a migrate task holds no label set".into()))?;
        Ok(migrated_part(&migrate_group(&group, &labels, &self.codec)))
    }
}

/// What one sample adds to the cohort's report, as a worker answers it.
fn migrated_part(part: &ConversionReport) -> workers::FileOutcome {
    let mut outcome = workers::FileOutcome::new();
    outcome.data = json!({"outputs": part.outputs, "notes": part.notes.iter().map(Note::to_json).collect::<Vec<_>>()});
    outcome
}

fn note_from_json(note: &Value) -> Note {
    let text = |key: &str| note[key].as_str().unwrap_or_default().to_string();
    Note {
        kind: text("kind"),
        message: text("message"),
        severity: text("severity"),
        detail: note["detail"].as_object().cloned().unwrap_or_default(),
    }
}

fn group_to_json(g: &MigrationGroup) -> Value {
    let occasions: Vec<Value> = g
        .group
        .occasions
        .iter()
        .map(|o| {
            json!({
                "key": o.key,
                "subject_id": o.subject_id,
                "date": o.date,
                "order_hint": o.order_hint,
                "demographics": o.demographics.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(),
                "payload": o.payload,
            })
        })
        .collect();
    json!({
        "subject_id": g.group.subject_id,
        "ordered_by": g.group.ordered_by,
        "occasions": occasions,
        "sources": g.sources.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
        "name": g.name,
        "target": g.target.to_string_lossy(),
        "clash": g.clash.as_ref().map(|p| p.to_string_lossy()),
    })
}

fn group_from_json(v: &Value) -> medh5::Result<MigrationGroup> {
    let bad = || Error::Value("a malformed migrate task".into());
    let text = |v: &Value| v.as_str().map(str::to_string);
    let occasions = v["occasions"]
        .as_array()
        .ok_or_else(bad)?
        .iter()
        .map(|o| {
            Ok(Occasion {
                key: text(&o["key"]).ok_or_else(bad)?,
                subject_id: text(&o["subject_id"]),
                date: text(&o["date"]),
                order_hint: o["order_hint"].as_f64(),
                demographics: o["demographics"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|pair| Some((text(&pair[0])?, text(&pair[1])?)))
                    .collect(),
                payload: o["payload"].as_u64().ok_or_else(bad)? as usize,
            })
        })
        .collect::<medh5::Result<Vec<_>>>()?;
    let paths = |v: &Value| -> Vec<PathBuf> {
        v.as_array().into_iter().flatten().filter_map(Value::as_str).map(PathBuf::from).collect()
    };
    Ok(MigrationGroup {
        group: SubjectGroup {
            subject_id: text(&v["subject_id"]).ok_or_else(bad)?,
            occasions,
            ordered_by: text(&v["ordered_by"]).ok_or_else(bad)?,
        },
        sources: paths(&v["sources"]),
        name: text(&v["name"]).ok_or_else(bad)?,
        target: PathBuf::from(text(&v["target"]).ok_or_else(bad)?),
        clash: text(&v["clash"]).map(PathBuf::from),
    })
}

// -- NIfTI, in the engine -----------------------------------------------------
//
// What the Python package's `medh5.cli` did for these commands until 2.2.0,
// ported: the same checks before anything is read, the same messages, and
// the same report.  An error the package let escape as a traceback (a
// `ValueError`) is a handled one here.

/// Run NIfTI converter `name`.
fn convert_nifti(name: &str, m: &ArgMatches, ctx: &mut Ctx) -> CmdResult {
    if let Some(report) = get_str(m, "report") {
        if Path::new(report).is_dir() {
            return Ok(ctx.fail(format!("--report {report} is a directory; name the file to write")));
        }
    }
    if let Some(clash) = nifti_collision(name, m) {
        return Ok(ctx.fail(clash));
    }
    let run = match name {
        "from-nifti" => from_nifti(m).map(|report| (report, None)),
        "import-seg-nifti" => import_seg_nifti(m).map(|report| (report, None)),
        _ => to_nifti(m).map(|(report, written)| (report, Some(written))),
    };
    match run {
        Ok((report, text)) => finish_converted(&report, m, ctx, text),
        Err(e) => Ok(ctx.fail(failure(&e))),
    }
}

fn from_nifti(m: &ArgMatches) -> medh5::Result<ConversionReport> {
    let images = paths_of(pairs(m, "image", "image")?);
    let masks = paths_of(pairs(m, "mask", "mask")?);
    let modalities = pairs(m, "modality", "modality")?;
    let value_units = pairs(m, "value_units", "value-units")?;
    let mut options = ImportOptions::default()
        .with_masks(masks)
        .with_modalities(modalities)
        .with_coord_system(req_str(m, "coord_system"))
        .with_fourth_axis(req_str(m, "fourth_axis"))
        .with_assume_geometry(flag(m, "assume_geometry"))
        .with_plane_as_volume(flag(m, "plane_as_volume"))
        .with_value_units(value_units)
        .with_codec(req_str(m, "codec"))
        .with_annotated_classes(annotated(m)?);
    if let Some(id) = get_str(m, "sample_id") {
        options = options.with_sample_id(id);
    }
    if let Some(id) = get_str(m, "subject_id") {
        options = options.with_subject_id(id);
    }
    if let Some(path) = get_str(m, "label_set") {
        options = options.with_label_set(load_file(Path::new(path))?);
    }
    let mut report = ConversionReport::new("from-nifti", "");
    nifti::from_nifti(&images, Path::new(req_str(m, "out")), &options, &mut report)?;
    Ok(report)
}

fn import_seg_nifti(m: &ArgMatches) -> medh5::Result<ConversionReport> {
    let sample = one_sample(req_str(m, "sample"))?;
    let masks = paths_of(pairs(m, "mask", "mask")?);
    let mut options = SegOptions::default()
        .with_ann_id(req_str(m, "ann_id"))
        .with_annotated_classes(annotated(m)?)
        .with_assume_aligned(flag(m, "assume_aligned"))
        .with_assume_geometry(flag(m, "assume_geometry"))
        .with_plane_as_volume(flag(m, "plane_as_volume"));
    if let Some(grid) = get_str(m, "grid").filter(|g| !g.is_empty()) {
        options = options.with_grid(grid);
    }
    let mut report = ConversionReport::new("import-seg-nifti", "");
    nifti::import_seg_nifti(Path::new(sample), &masks, &options, &mut report)?;
    Ok(report)
}

/// The report, and the file written: an export prints the file, as it
/// always has.
fn to_nifti(m: &ArgMatches) -> medh5::Result<(ConversionReport, String)> {
    let sample = one_sample(req_str(m, "path"))?;
    let mut options = ExportOptions::default().with_physical(!flag(m, "stored"));
    if let Some(ann) = get_str(m, "annotation") {
        options = options.with_annotation(ann);
    }
    if let Some(key) = get_str(m, "class_key") {
        options = options.with_class_key(ClassKey::Key(key.to_string()));
    }
    let mut report = ConversionReport::new("to-nifti", sample);
    let written = nifti::to_nifti(
        Path::new(sample),
        req_str(m, "image"),
        Path::new(req_str(m, "out")),
        &options,
        Some(&mut report),
    )?;
    let written = shown(&written);
    report.outputs.push(written.clone());
    Ok((report, written))
}

/// `path`, as the one sample file a converter reads or amends: a collection
/// is refused with the way to its members.
fn one_sample(path: &str) -> medh5::Result<&str> {
    match sample_file_refusal(path) {
        Some(refusal) => Err(Error::invalid(refusal)),
        None => Ok(path),
    }
}

/// `--image NAME=PATH` and its kin, in order, a name given twice keeping its
/// first place and its last value.
fn pairs(m: &ArgMatches, id: &str, what: &str) -> medh5::Result<Vec<(String, String)>> {
    let mut out: Vec<(String, String)> = Vec::new();
    for entry in get_paths(m, id) {
        let Some((name, value)) = entry.split_once('=') else {
            return Err(Error::invalid(format!("--{what} expects NAME=VALUE, got {}", repr_str(&entry))));
        };
        match out.iter_mut().find(|(known, _)| known == name) {
            Some(slot) => slot.1 = value.to_string(),
            None => out.push((name.to_string(), value.to_string())),
        }
    }
    Ok(out)
}

fn paths_of(pairs: Vec<(String, String)>) -> Vec<(String, PathBuf)> {
    pairs.into_iter().map(|(name, path)| (name, PathBuf::from(path))).collect()
}

/// `--annotated-classes`: `all_given`, `all`, or class keys separated by
/// commas.
fn annotated(m: &ArgMatches) -> medh5::Result<Annotated> {
    Ok(match get_str(m, "annotated_classes") {
        None | Some("all_given") => Annotated::AllGiven,
        Some("all") => Annotated::All,
        Some(value) => {
            let keys: Vec<&str> = value.split(',').map(str::trim).collect();
            if keys.iter().any(|k| k.is_empty()) {
                return Err(Error::invalid(format!(
                    "--annotated-classes expects all_given, all, or class keys separated by commas, got {}",
                    repr_str(value)
                )));
            }
            Annotated::Classes(keys.into_iter().map(|k| ClassKey::Key(k.to_string())).collect())
        }
    })
}

/// Why a NIfTI converter's report or output would replace a file it reads,
/// or `None`.  Paths are compared resolved and, where both exist, as one
/// file, so `sub/../ct.nii.gz`, a link to it and a hard link are the file
/// itself.
fn nifti_collision(name: &str, m: &ArgMatches) -> Option<String> {
    let mut reads: Vec<(String, PathBuf)> = Vec::new();
    let mut writes: Vec<(String, PathBuf)> = Vec::new();
    if name == "to-nifti" {
        if let Some(path) = given(m, "path").filter(|p| !p.is_empty()) {
            reads.push(("the sample this command reads".into(), path.into()));
        }
    } else {
        for id in ["image", "mask"] {
            let entries = m.try_get_many::<String>(id).ok().flatten().into_iter().flatten();
            for entry in entries {
                let (given, path) = entry.split_once('=').unwrap_or((entry, ""));
                if !path.is_empty() {
                    reads.push((format!("{id} {}, which this command reads", repr_str(given)), path.into()));
                }
            }
        }
        let named = [("the label set this command reads", "label_set"), ("the sample this command amends", "sample")];
        for (what, id) in named {
            if let Some(path) = given(m, id).filter(|p| !p.is_empty()) {
                reads.push((what.into(), path.into()));
            }
        }
    }
    if let Some(out) = given(m, "out").filter(|p| !p.is_empty()) {
        writes.push(("the output this command writes".into(), out.into()));
    }
    if let Some(report) = given(m, "report").filter(|p| !p.is_empty()) {
        if let Some((what, _)) = reads.iter().chain(&writes).find(|(_, path)| same_path(Path::new(report), path)) {
            return Some(format!("--report {report} is {what}; name another file for the report"));
        }
    }
    for (_, out) in &writes {
        if let Some((what, _)) = reads.iter().find(|(_, path)| same_path(out, path)) {
            return Some(format!("the output {} is {what}; name another file to write", shown(out)));
        }
    }
    None
}

/// The value of argument `id`, which this command may not have.
fn given<'m>(m: &'m ArgMatches, id: &str) -> Option<&'m str> {
    m.try_get_one::<String>(id).ok().flatten().map(String::as_str)
}

/// Keep the report where `--report` says, print it (or `text`, the file an
/// export wrote), and exit on it.  A report naming an output is not
/// written, and the command fails once the report is printed.
fn finish_converted(report: &ConversionReport, m: &ArgMatches, ctx: &mut Ctx, text: Option<String>) -> CmdResult {
    let mut refusal = None;
    if let Some(target) = get_str(m, "report") {
        match report.outputs.iter().find(|o| same_path(Path::new(target), Path::new(o))) {
            None => {
                let json = pretty(&report.to_json()) + "\n";
                if let Err(e) = medh5::h5::file::write_atomic(Path::new(target), json.as_bytes()) {
                    return Ok(ctx.fail(failure(&e)));
                }
            }
            Some(clash) => {
                refusal = Some(format!(
                    "--report {target} is {clash}, which this command wrote; the report was not written --- name \
                     another file for it"
                ));
            }
        }
    }
    if flag(m, "json") {
        ctx.emit(&report.to_json(), true);
    } else {
        ctx.print(text.unwrap_or_else(|| report.format(true)));
    }
    if let Some(refusal) = refusal {
        return Ok(ctx.fail(refusal));
    }
    Ok(if report.ok() { EXIT_OK } else { EXIT_ERROR })
}

/// Whether two paths name one file: equal once resolved (an output not
/// written yet has no identity to compare), or one file on disk.
fn same_path(first: &Path, second: &Path) -> bool {
    matches!((resolved(first), resolved(second)), (Some(a), Some(b)) if a == b) || same_file(first, second)
}

/// `path` made absolute, its symbolic links followed as far as it exists and
/// the rest joined as written: Python's `Path.resolve()`.
fn resolved(path: &Path) -> Option<PathBuf> {
    let mut out = std::env::current_dir().ok()?.canonicalize().ok()?;
    let mut exists = true;
    for part in path.components() {
        match part {
            Component::Prefix(_) | Component::RootDir => out.push(part.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(name) => {
                out.push(name);
                if exists {
                    match out.canonicalize() {
                        Ok(real) => out = real,
                        Err(_) => exists = false,
                    }
                }
            }
        }
    }
    Some(out)
}

/// A path as Python's `str(Path(...))` spells it: without `.` components.
fn shown(path: &Path) -> String {
    let parts: PathBuf = path.components().filter(|c| *c != Component::CurDir).collect();
    if parts.as_os_str().is_empty() {
        ".".into()
    } else {
        parts.to_string_lossy().into_owned()
    }
}

/// What the command line says for a converter's error: an `OSError` as the
/// shell says one (`<file>: <why>`), a lookup by its key, anything else in
/// its own words.
fn failure(error: &Error) -> String {
    if let Error::Io(message) = error {
        if let Some(rest) = message.strip_prefix("[Errno ") {
            if let Some((_, rest)) = rest.split_once("] ") {
                let at = rest.find(": '").or_else(|| rest.find(": \""));
                if let Some(at) = at {
                    return format!("{}: {}", unliteral(&rest[at + 2..]), &rest[..at]);
                }
            }
            return message.clone();
        }
        if let Some((reason, code)) = message.rsplit_once(" (os error ") {
            if let Ok(code) = code.trim_end_matches(')').parse::<i32>() {
                return format!("[Errno {code}] {reason}");
            }
        }
        return message.clone();
    }
    top_level_message(error)
}

/// The string a Python string literal ([`repr_str`]'s) spells.
fn unliteral(literal: &str) -> String {
    let inner = literal.get(1..literal.len().saturating_sub(1)).unwrap_or_default();
    let mut out = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        let code = |chars: &mut std::str::Chars<'_>, n: usize| -> Option<char> {
            let digits: String = chars.by_ref().take(n).collect();
            u32::from_str_radix(&digits, 16).ok().and_then(char::from_u32)
        };
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('x') => out.extend(code(&mut chars, 2)),
            Some('u') => out.extend(code(&mut chars, 4)),
            Some('U') => out.extend(code(&mut chars, 8)),
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}
