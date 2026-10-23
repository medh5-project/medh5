//! Validation (spec §15).
//!
//! Four levels, each a superset of the last:
//!
//! - `structural` --- layout, required attributes, dtypes, shapes,
//!   identifier syntax, JSON Schema;
//! - `semantic` --- cross-references resolve, geometry consistency, class ids
//!   in the label set, encoding invariants, profile requirements;
//! - `integrity` --- per-object digests, `content_id`, index currency;
//! - `strict` --- all of the above, with warnings promoted to failures.
//!
//! A validation pass never fails on a bad file --- it reports.  Curation needs
//! to see everything wrong with a file at once.

pub mod rules;

use std::path::Path;

use serde_json::{json, Value};

use crate::h5::attrs;
use crate::h5::ops;
use crate::integrity::AttrNameMap;
use crate::json::{repr_list, repr_str};
use crate::sample::reader::PROFILES;
use crate::{Error, Result};

/// The validation levels, in order.
pub const LEVELS: [&str; 4] = ["structural", "semantic", "integrity", "strict"];

/// The group a collection keeps its sample roots in (§2.2).
pub const SAMPLES_GROUP: &str = "samples";

/// One finding.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    /// The diagnostic code (spec §15.2, 1.1 §11.2): stable API.
    pub code: String,
    /// An HDF5 path, or a JSON pointer into `/meta` after `/meta#`.
    pub location: String,
    /// What is wrong, for a person.
    pub message: String,
    /// `error` or `warning`, as the code table gives the code; `strict`
    /// promotes a warning without changing this.
    pub severity: String,
    /// The level the pass ran at.
    pub level: String,
}

impl Diagnostic {
    /// The code table's one-line summary.
    pub fn summary(&self) -> &'static str {
        crate::codes::get(&self.code).map(|c| c.summary.as_str()).unwrap_or("")
    }

    /// The diagnostic as `validate --json` prints it.
    pub fn to_json(&self) -> Value {
        json!({
            "code": self.code,
            "severity": self.severity,
            "location": self.location,
            "message": self.message,
            "level": self.level,
        })
    }

    /// `SEVERITY CODE location: message`, severity padded to seven columns.
    pub fn line(&self) -> String {
        format!("{:7} {} {}: {}", self.severity.to_uppercase(), self.code, self.location, self.message)
    }
}

/// Everything a validation pass found.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// The file validated, as the caller named it.
    pub path: String,
    /// The level the pass ran at.
    pub level: String,
    /// The profiles the file was held to: those it declares, then any the
    /// caller required besides.
    pub profiles: Vec<String>,
    /// Every finding, errors first, then by code and location.
    pub diagnostics: Vec<Diagnostic>,
    /// What was checked: the rules run, and whether `/meta` reached its
    /// schema (a collection lists its sample roots instead).
    pub checked: Value,
}

impl Report {
    /// An empty report for `path` at `level`.
    pub fn new(path: &str, level: &str) -> Report {
        Report {
            path: path.into(),
            level: level.into(),
            profiles: Vec::new(),
            diagnostics: Vec::new(),
            checked: json!({}),
        }
    }

    /// Severity at this level, which is not always the table's: §15.1
    /// defines `strict` as the other levels with warnings promoted to
    /// errors, so at `strict` there are no warnings.  The diagnostic keeps
    /// its measured `severity`.
    pub fn effective_severity<'a>(&self, d: &'a Diagnostic) -> &'a str {
        if self.level == "strict" {
            "error"
        } else {
            &d.severity
        }
    }

    /// The diagnostics that are errors at this level, warnings promoted at
    /// `strict` included.
    pub fn errors(&self) -> Vec<&Diagnostic> {
        self.diagnostics.iter().filter(|d| self.effective_severity(d) == "error").collect()
    }

    /// The diagnostics that are warnings at this level (none at `strict`).
    pub fn warnings(&self) -> Vec<&Diagnostic> {
        self.diagnostics.iter().filter(|d| self.effective_severity(d) == "warning").collect()
    }

    /// Diagnostics the table calls warnings and this level treats as errors.
    pub fn promoted(&self) -> Vec<&Diagnostic> {
        if self.level != "strict" {
            return Vec::new();
        }
        self.diagnostics.iter().filter(|d| d.severity == "warning").collect()
    }

    /// Every code emitted, sorted and deduplicated --- the conformance key.
    pub fn codes(&self) -> Vec<String> {
        let mut out: Vec<String> = self.diagnostics.iter().map(|d| d.code.clone()).collect();
        out.sort();
        out.dedup();
        out
    }

    /// Whether the file conforms at this level.
    pub fn ok(&self) -> bool {
        self.errors().is_empty()
    }

    /// The report as `validate --json` prints it.
    pub fn to_json(&self) -> Value {
        json!({
            "path": self.path,
            "level": self.level,
            "profiles": self.profiles,
            "ok": self.ok(),
            "errors": self.errors().len(),
            "warnings": self.warnings().len(),
            "promoted": self.promoted().len(),
            "diagnostics": self.diagnostics.iter().map(Diagnostic::to_json).collect::<Vec<_>>(),
            "checked": self.checked,
        })
    }

    /// Human-readable text, one diagnostic per line.
    pub fn format(&self, verbose: bool) -> String {
        let profiles = if self.profiles.is_empty() { "-".to_string() } else { self.profiles.join(",") };
        let mut lines = vec![format!(
            "{}: {} [{}] profiles={profiles} ({} errors, {} warnings)",
            self.path,
            if self.ok() { "OK" } else { "FAILED" },
            self.level,
            self.errors().len(),
            self.warnings().len()
        )];
        for d in &self.diagnostics {
            lines.push(format!("  {}", d.line()));
            let summary = d.summary();
            if verbose && !summary.is_empty() {
                lines.push(format!("          -> {summary}"));
            }
        }
        lines.join("\n")
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        format!(
            "Report({}, ok={}, {} errors, {} warnings)",
            repr_str(&self.path),
            if self.ok() { "True" } else { "False" },
            self.errors().len(),
            self.warnings().len()
        )
    }

    fn sort(&mut self) {
        self.diagnostics.sort_by(|a, b| {
            ((a.severity != "error"), &a.code, &a.location).cmp(&((b.severity != "error"), &b.code, &b.location))
        });
    }
}

/// Combine several reports, prefixing each location with its file.
pub fn merge(reports: &[Report], path: &str) -> Report {
    let mut out = Report::new(path, reports.first().map(|r| r.level.as_str()).unwrap_or("structural"));
    for report in reports {
        for d in &report.diagnostics {
            out.diagnostics.push(Diagnostic { location: format!("{}:{}", report.path, d.location), ..d.clone() });
        }
    }
    out
}

/// Refuse a level the validator does not run.  Public for the frontends,
/// which refuse it before they reach the validator.
#[doc(hidden)]
pub fn check_level(level: &str) -> Result<()> {
    if !LEVELS.contains(&level) {
        return Err(Error::Value(format!(
            "unknown validation level {}; expected one of {}",
            repr_str(level),
            crate::json::repr_list(&LEVELS).replacen('[', "(", 1).replacen(']', ")", 1)
        )));
    }
    Ok(())
}

/// Refuse a requirement naming a profile this validator does not know: a
/// misspelt one would hold the file to nothing and pass it.
fn check_required(profiles: Option<&[String]>) -> Result<()> {
    let mut unknown: Vec<&String> =
        profiles.unwrap_or_default().iter().filter(|p| !PROFILES.contains(&p.as_str())).collect();
    if !unknown.is_empty() {
        unknown.sort();
        unknown.dedup();
        return Err(Error::Value(format!(
            "unknown profile(s) {}; expected any of {}",
            repr_list(&unknown),
            repr_list(&PROFILES)
        )));
    }
    Ok(())
}

/// Run `rules` in order, collecting what each finds, and the names of the
/// rules run.
///
/// A rule that stops on an error reports it as one finding --- under the
/// error's own code when it has one, the defect the rule met, and as E001
/// ("could not read the file") when it has none --- and the rules after it
/// still run.  A file of another major (E002) is not read past the container
/// rule: what this major's rules would say of it is not its content's.
fn run_rules(ctx: &mut rules::Context, rules: &[(&'static str, rules::Rule)]) -> (Vec<Diagnostic>, Vec<&'static str>) {
    let mut out = Vec::new();
    let mut run = Vec::new();
    for (name, rule) in rules {
        run.push(*name);
        let found = match rule(ctx) {
            Ok(found) => found,
            Err(e) => vec![match e.code() {
                Some(code) => ctx.err(code, "/", format!("{name} stopped: {}", e.message())),
                None => ctx.err("E001", "/", format!("{name} could not read the file: {}", e.python_line())),
            }],
        };
        let unsupported = found.iter().any(|d| d.code == "E002");
        out.extend(found);
        if unsupported {
            break;
        }
    }
    (out, run)
}

/// Validate an open sample root.
///
/// The root is held to the profiles it declares and to `profiles` besides: a
/// caller's requirement adds to the file's claims and never waives one.  A
/// profile this validator does not know is refused (`Error::Value`), as an
/// unknown level is.
///
/// `errors_only` skips the warning-only checks that read bulk data --- what
/// `SampleWriter::commit` asks for.
pub fn validate_root_with(
    root: &hdf5::Group,
    path: &str,
    level: &str,
    profiles: Option<&[String]>,
    attr_names: Option<&AttrNameMap>,
    errors_only: bool,
) -> Result<Report> {
    stage!("validate");
    check_level(level)?;
    check_required(profiles)?;
    let mut declared: Vec<String> = attrs::get_strs(root, "medh5_profiles").ok().flatten().unwrap_or_default();
    for profile in profiles.unwrap_or_default() {
        if !declared.contains(profile) {
            declared.push(profile.clone());
        }
    }
    let mut ctx = rules::Context::new(root.clone(), path, level, declared.clone(), errors_only);
    ctx.attr_names = attr_names.cloned();
    let mut report = Report::new(path, level);
    report.profiles = declared;
    let (diagnostics, run) = run_rules(&mut ctx, &rules::rules_for(level));
    report.diagnostics = diagnostics;
    report.checked = json!({
        "rules": run,
        "schema_checked": ctx.schema_checked,
    });
    report.sort();
    Ok(report)
}

/// Validate an open sample root at `level` (what the writer's commit gate
/// calls with `errors_only`).
pub fn validate_root(root: &hdf5::Group, path: Option<&Path>, level: &str, errors_only: bool) -> Result<Report> {
    let shown = path.map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| "<memory>".into());
    validate_root_with(root, &shown, level, None, None, errors_only)
}

/// The digest attribute map, needed only where integrity is checked.
/// Public for the frontends, which validate a root they hold.
#[doc(hidden)]
pub fn attr_names_for(root: &hdf5::Group, level: &str) -> Option<AttrNameMap> {
    if level != "integrity" && level != "strict" {
        return None;
    }
    crate::sample::attr_name_map_of(root).ok()
}

/// Validate a `collection` root and every sample in it (§2.2), each sample
/// held to its own profiles and to `profiles` besides.
pub fn validate_collection(root: &hdf5::Group, path: &str, level: &str, profiles: Option<&[String]>) -> Result<Report> {
    check_level(level)?;
    check_required(profiles)?;
    let mut ctx = rules::Context::new(root.clone(), path, level, Vec::new(), false);
    let header = rules::check_collection(&mut ctx)?;
    // Another major's members are not read as this major's (E002).
    let unsupported = header.iter().any(|d| d.code == "E002");
    let mut members = Vec::new();
    if let Some(node) = ops::child_group(root, SAMPLES_GROUP).filter(|_| !unsupported) {
        for key in ops::members(&node)? {
            if let Some(member) = ops::child_group(&node, &key) {
                let names = attr_names_for(&member, level);
                members.push(validate_root_with(
                    &member,
                    &format!("/{SAMPLES_GROUP}/{key}"),
                    level,
                    profiles,
                    names.as_ref(),
                    false,
                )?);
            }
        }
    }
    let mut combined = if members.is_empty() { Report::new(path, level) } else { merge(&members, path) };
    combined.level = level.into();
    let mut diagnostics = header;
    diagnostics.extend(combined.diagnostics);
    combined.diagnostics = diagnostics;
    combined.checked = json!({"samples": members.iter().map(|r| r.path.clone()).collect::<Vec<_>>()});
    combined.sort();
    Ok(combined)
}

fn read_failure(path: &str, level: &str, message: String) -> Report {
    let mut report = Report::new(path, level);
    report.diagnostics.push(Diagnostic {
        code: "E001".into(),
        location: "/".into(),
        message,
        severity: "error".into(),
        level: level.into(),
    });
    report
}

/// How to validate a file: the level, the profiles to require besides those
/// the file declares, and the revision of the code table to report under.
///
/// Built from [`Default`] (`semantic`, no extra profiles, the engine's own
/// revision) and the `with`-style setters.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidateOptions {
    /// One of [`LEVELS`].
    pub level: String,
    /// Profiles to hold the file to besides those it declares.
    pub profiles: Option<Vec<String>>,
    /// Report only codes this revision of the table (§15.2) had allocated;
    /// `None` reports every code the engine knows.
    pub codes_revision: Option<u32>,
}

impl Default for ValidateOptions {
    fn default() -> Self {
        ValidateOptions { level: "semantic".into(), profiles: None, codes_revision: None }
    }
}

impl ValidateOptions {
    /// Validate at `level` (one of [`LEVELS`]).
    pub fn level(mut self, level: &str) -> Self {
        self.level = level.into();
        self
    }

    /// Hold the file to `profiles` besides those it declares.
    pub fn profiles(mut self, profiles: &[String]) -> Self {
        self.profiles = Some(profiles.to_vec());
        self
    }

    /// Report only the codes revision `revision` of the table had allocated
    /// (§15.1, §15.2): a `strict` verdict that later codes cannot move.
    pub fn codes_revision(mut self, revision: u32) -> Self {
        self.codes_revision = Some(revision);
        self
    }
}

/// Refuse a revision the table does not have: revision 0 would drop every
/// code and pass any file, and one past the table's would promise codes this
/// engine cannot report.
fn check_revision(revision: Option<u32>) -> Result<()> {
    match revision {
        Some(r) if r < 1 || r > crate::codes::table_revision() => Err(Error::invalid(format!(
            "codes revision {r} is not a revision of the code table; this engine implements revisions 1 to {} \
             (§15.2)",
            crate::codes::table_revision()
        ))),
        _ => Ok(()),
    }
}

/// Drop every diagnostic whose code a later revision than `revision`
/// allocated --- before `strict` promotes anything, since promotion is
/// computed from what the report holds.
fn pin_revision(report: &mut Report, revision: Option<u32>) {
    if let Some(pinned) = revision {
        report.diagnostics.retain(|d| crate::codes::revision(&d.code).is_none_or(|r| r <= pinned));
    }
}

/// Validate one `.medh5` sample or `.medh5c` collection file, held to the
/// profiles it declares and to `profiles` besides.
///
/// A file that cannot be read is reported (E001), not returned as an error;
/// an unknown level or profile is an error.
pub fn validate_file(path: &Path, level: &str, profiles: Option<&[String]>) -> Result<Report> {
    let mut options = ValidateOptions::default().level(level);
    options.profiles = profiles.map(<[String]>::to_vec);
    validate_file_with(path, &options)
}

/// Validate one `.medh5` sample or `.medh5c` collection file as `options`
/// say.
///
/// A file that cannot be read is reported (E001), not returned as an error;
/// an unknown level or profile, or a codes revision outside the table, is an
/// error.
pub fn validate_file_with(path: &Path, options: &ValidateOptions) -> Result<Report> {
    let (level, profiles) = (options.level.as_str(), options.profiles.as_deref());
    check_level(level)?;
    check_required(profiles)?;
    check_revision(options.codes_revision)?;
    let text = path.to_string_lossy().into_owned();
    let handle = match crate::h5::file::open_read(path) {
        Ok(h) => h,
        Err(e) => return Ok(read_failure(&text, level, e.to_string())),
    };
    let result = (|| -> Result<Report> {
        let root = handle.as_group()?;
        let kind = attrs::get_str(&root, "medh5_kind")?.unwrap_or_else(|| "sample".into());
        if kind == "collection" {
            return validate_collection(&root, &text, level, profiles);
        }
        let names = attr_names_for(&root, level);
        validate_root_with(&root, &text, level, profiles, names.as_ref(), false)
    })();
    let mut report = match result {
        Ok(r) => r,
        Err(e) => read_failure(&text, level, format!("the file could not be read: {}", e.python_line())),
    };
    pin_revision(&mut report, options.codes_revision);
    Ok(report)
}

/// The clinical profile's errors in one sample root (E8xx, and the document
/// and declaration rules they rest on), without the bulk-data rules --- what a
/// task's preflight asks of each source it will select from --- and the
/// tables the rules read, document text deferred, so a caller that goes on to
/// use the profile reads it once
/// ([`Clinical::from_tables`](crate::clinical::Clinical::from_tables)).
pub fn clinical_checked(root: &hdf5::Group, path: &str) -> Result<(Vec<Diagnostic>, Option<rules::ClinicalTables>)> {
    let declared = attrs::get_strs(root, "medh5_profiles")?.unwrap_or_default();
    let mut ctx = rules::Context::new(root.clone(), path, "semantic", declared, true);
    let mut out = Vec::new();
    for rule in [rules::check_document as rules::Rule, rules::check_clinical, rules::check_clinical_records] {
        out.extend(rule(&mut ctx)?);
    }
    out.retain(|d| d.severity == "error" && (d.code.starts_with("E8") || d.code == "E009" || d.code == "E004"));
    Ok((out, ctx.clinical.take()))
}

/// Validate many files, one report each.
pub fn validate_paths(paths: &[&Path], level: &str, profiles: Option<&[String]>) -> Result<Vec<Report>> {
    paths.iter().map(|p| validate_file(p, level, profiles)).collect()
}

/// Validate many files as `options` say, one report each.
pub fn validate_paths_with(paths: &[&Path], options: &ValidateOptions) -> Result<Vec<Report>> {
    check_revision(options.codes_revision)?;
    paths.iter().map(|p| validate_file_with(p, options)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops_on_a_defect(_: &mut rules::Context) -> Result<Vec<Diagnostic>> {
        Err(Error::coded("E405", "6 instance ids for 3 class ids"))
    }

    fn cannot_read(_: &mut rules::Context) -> Result<Vec<Diagnostic>> {
        Err(Error::Io("H5Dread failed".into()))
    }

    fn finds_one(ctx: &mut rules::Context) -> Result<Vec<Diagnostic>> {
        Ok(vec![ctx.err("W903", "/meta#deidentification", "no de-identification record")])
    }

    #[test]
    fn s15_2_a_rule_that_stops_reports_its_own_code_and_the_rest_still_run() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("x.medh5"))).unwrap();
        let mut ctx = rules::Context::new(file.as_group().unwrap(), "x.medh5", "semantic", Vec::new(), false);
        let rules: [(&'static str, rules::Rule); 3] =
            [("check_a", stops_on_a_defect), ("check_b", cannot_read), ("check_c", finds_one)];
        let (found, run) = run_rules(&mut ctx, &rules);
        assert_eq!(run, ["check_a", "check_b", "check_c"]);
        let codes: Vec<&str> = found.iter().map(|d| d.code.as_str()).collect();
        assert_eq!(codes, ["E405", "E001", "W903"]);
        assert_eq!(found[0].message, "check_a stopped: 6 instance ids for 3 class ids");
        assert_eq!((found[0].severity.as_str(), found[0].level.as_str()), ("error", "semantic"));
        assert!(found[1].message.starts_with("check_b could not read the file: "), "{}", found[1].message);
        assert_eq!(found[2].severity, "warning");
    }

    #[test]
    fn s15_2_an_object_that_cannot_be_read_is_e001_where_it_is() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("x.medh5"))).unwrap();
        let ctx = rules::Context::new(file.as_group().unwrap(), "x.medh5", "semantic", Vec::new(), false);
        let defect = ctx.failed("/grids/g", &Error::coded("E104", "spacing (0.0, 1.0) must be finite and > 0"));
        assert_eq!((defect.code.as_str(), defect.location.as_str()), ("E104", "/grids/g"));
        assert_eq!(defect.message, "spacing (0.0, 1.0) must be finite and > 0");
        let unread = ctx.failed("/grids/g", &Error::Io("H5Aread failed".into()));
        assert_eq!((unread.code.as_str(), unread.location.as_str()), ("E001", "/grids/g"));
        assert!(unread.message.starts_with("could not be read: "), "{}", unread.message);
    }

    /// Revision 0 would drop every code and pass any file; one past the
    /// table's would promise codes this engine cannot report.
    #[test]
    fn s15_2_a_revision_outside_the_table_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("absent.medh5");
        for revision in [0, crate::codes::table_revision() + 1] {
            let options = ValidateOptions::default().codes_revision(revision);
            let err = validate_file_with(&path, &options).unwrap_err();
            assert_eq!(err.kind_name(), "MEDH5ValidationError", "{err:?}");
            assert!(err.message().contains(&format!("codes revision {revision}")), "{err}");
            assert!(validate_paths_with(&[path.as_path()], &options).is_err());
        }
        let pinned = ValidateOptions::default().codes_revision(crate::codes::table_revision());
        let report = validate_file_with(&path, &pinned).unwrap();
        assert_eq!(report.codes(), ["E001"]);
    }
}
