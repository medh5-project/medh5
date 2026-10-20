//! `convert` and `migrate` --- the importers and exporters.
//!
//! Every command writes a conversion report, because the interesting part of
//! an import is not that it succeeded but what it had to decide.  `migrate`
//! (0.x files, Appendix B) and the NIfTI converters are native; the other
//! format converters are Python integrations and run in the Python package,
//! through the [`Host`].

use std::path::{Component, Path, PathBuf};

use clap::{ArgAction, ArgMatches};
use serde_json::{Map, Value};

use medh5::convert::legacy::{build_label_set, load_sidecar, migrate_paths, write_sidecar};
use medh5::convert::nifti::{self, ExportOptions, ImportOptions, SegOptions};
use medh5::convert::report::ConversionReport;
use medh5::h5::file::same_file;
use medh5::json::{pretty, repr_str};
use medh5::labels::registry::load_file;
use medh5::labels::ClassKey;
use medh5::sample::writer::Annotated;
use medh5::Error;

use crate::app;
use crate::common::*;

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
    let result = (|| -> medh5::Result<Option<ConversionReport>> {
        if let Some(target) = get_str(m, "write_labels") {
            let mut report = ConversionReport::new("migrate", "");
            let label_set = build_label_set(&sources, Some(&mut report))?;
            let written = write_sidecar(&label_set, Path::new(target))?;
            let note = format!("{}: {} classes --- review before migrating", written.display(), label_set.len());
            // With `--json`, stdout is the report alone.
            if flag(m, "json") {
                ctx.eprint(note);
            } else {
                ctx.print(note);
            }
            return Ok(Some(report));
        }
        let reviewed = match get_str(m, "label_set") {
            Some(p) => Some(load_sidecar(Path::new(p))?),
            None => None,
        };
        let report = migrate_paths(
            &sources,
            Path::new(req_str(m, "out")),
            req_str(m, "group_by"),
            get_str(m, "subject_key"),
            reviewed.as_ref(),
            req_str(m, "codec"),
        )?;
        Ok(Some(report))
    })();
    match result {
        Ok(Some(report)) => finish(&report, m, ctx),
        Ok(None) => Ok(EXIT_OK),
        Err(e) if e.is_medh5() || matches!(e, medh5::Error::Io(_)) => Ok(ctx.fail(e.to_string())),
        Err(e) => Err(e),
    }
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
