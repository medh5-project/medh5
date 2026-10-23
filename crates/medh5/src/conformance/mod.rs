//! The conformance corpus and the suite that publishes it (spec §15).
//!
//! Every case is a file plus the exact set of diagnostic codes a conforming
//! validator must emit for it.  Valid cases prove the format is writable;
//! invalid cases --- at least one per error code, built by mutating a valid
//! file --- prove the validator catches what the spec says it must.  The corpus
//! is a **shipped artifact**: [`build_corpus`] writes the files and an
//! `expected.json`, [`run_corpus`] checks this validator against it,
//! [`publish`] writes the distributable suite and [`score`] measures an
//! implementation that is not this one, in any language, from the codes it
//! reports back.
//!
//! Voxel data is drawn from [`Rng`](crate::rng::Rng) with fixed seeds, so a
//! build is reproducible.  No case depends on the values: what a case states
//! is its diagnostic codes.

mod build;
pub mod clinical;
pub mod companion;
#[doc(hidden)]
pub mod robustness;
pub mod suite;

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use serde_json::{json, Map, Value};

use crate::json::repr_str;
use crate::validate::validate_file;
use crate::{Error, Result};

pub use build::SEED;
pub use suite::{
    check_checksums, load_manifest, publish, score, score_with, summarize, ScoreOptions, CHECKSUMS, SCHEMA,
};

/// How a case's file is made.
pub type Build = Arc<dyn Fn(&Path) -> Result<()> + Send + Sync>;

/// One corpus entry.
#[derive(Clone)]
pub struct Case {
    /// The case's name, which is also its file's stem.
    pub name: String,
    /// What the file holds, in one sentence.
    pub description: String,
    /// The specification clause the case exercises.
    pub clause: String,
    /// How the file is made; `None` for a case read from a published
    /// manifest ([`Case::from_record`]).
    pub build: Option<Build>,
    /// The validation level the case is checked at.
    pub level: String,
    /// The error codes a conforming validator reports, and no others.
    pub errors: Vec<String>,
    /// The warning codes a conforming validator reports, and no others.
    pub warnings: Vec<String>,
    /// `.medh5c` for a collection case (§2.2).
    pub suffix: String,
    /// Built by editing a committed file, so its digests are deliberately
    /// stale: "no expected errors" does not mean "this file also verifies".
    pub mutated: bool,
}

impl fmt::Debug for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Case")
            .field("name", &self.name)
            .field("clause", &self.clause)
            .field("level", &self.level)
            .field("errors", &self.errors)
            .field("warnings", &self.warnings)
            .field("suffix", &self.suffix)
            .field("mutated", &self.mutated)
            .finish()
    }
}

impl Case {
    /// Whether the file conforms: it expects no error.
    pub fn valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// The format version the case is addressed to: the latest `since` among
    /// the codes it expects, and a later minor's for a clause of it (marked
    /// `(1.1)` or `(1.2)`).
    ///
    /// What a file declares is not it: the E009 and E011 cases declare 1.0
    /// and expect 1.1's answers, the W913 case declares 1.3.  A 1.0-only
    /// implementation is scored on the cases addressed to 1.0
    /// ([`suite::ScoreOptions`]).
    pub fn format_version(&self) -> String {
        let floor = crate::version::format_versions()
            .iter()
            .rev()
            .find(|v| self.clause.contains(&format!("({v})")))
            .copied()
            .unwrap_or(crate::version::BASE_VERSION);
        latest_version(floor, self.errors.iter().chain(&self.warnings).filter_map(|c| crate::codes::since(c)))
    }

    /// The case's entry in `expected.json`.
    pub fn to_json(&self) -> Value {
        let sorted = |v: &[String]| -> Vec<String> {
            let mut s = v.to_vec();
            s.sort();
            s
        };
        json!({
            "name": self.name,
            "description": self.description,
            "clause": self.clause,
            "level": self.level,
            "file_suffix": self.suffix,
            "valid": self.valid(),
            "mutated": self.mutated,
            "expect_errors": sorted(&self.errors),
            "expect_warnings": sorted(&self.warnings),
            "format_version": self.format_version(),
        })
    }

    /// A case standing in for a manifest entry this build does not know, so
    /// scoring a suite published by a newer medh5 reports on its cases.
    pub fn from_record(record: &Value) -> Case {
        let text = |k: &str| record.get(k).map(crate::json::py_str).unwrap_or_default();
        let list = |k: &str| -> Vec<String> {
            record
                .get(k)
                .and_then(Value::as_array)
                .map(|a| a.iter().map(crate::json::py_str).collect())
                .unwrap_or_default()
        };
        Case {
            name: text("name"),
            description: text("description"),
            clause: text("clause"),
            build: None,
            level: record.get("level").and_then(Value::as_str).unwrap_or("semantic").to_string(),
            errors: list("expect_errors"),
            warnings: list("expect_warnings"),
            suffix: record.get("file_suffix").and_then(Value::as_str).unwrap_or(".medh5").to_string(),
            mutated: record.get("mutated").and_then(Value::as_bool).unwrap_or(false),
        }
    }
}

/// A `major.minor` version as numbers, so `1.10` follows `1.9`; `None` for
/// anything else.
pub(crate) fn version_key(version: &str) -> Option<(u32, u32)> {
    let (major, minor) = version.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// The latest of `floor` and `versions`; one that is not a version is passed
/// over.
pub(crate) fn latest_version<'a>(floor: &'a str, versions: impl IntoIterator<Item = &'a str>) -> String {
    let mut latest = floor;
    for version in versions {
        if version_key(version) > version_key(latest) {
            latest = version;
        }
    }
    latest.to_string()
}

/// What running one case produced.
#[derive(Debug, Clone)]
pub struct CaseResult {
    /// The case run.
    pub case: Case,
    /// The file validated.
    pub path: String,
    /// The error codes the validator reported.
    pub got_errors: Vec<String>,
    /// The warning codes the validator reported.
    pub got_warnings: Vec<String>,
    /// Expected codes the validator did not report, at the severity
    /// expected: an error reported as a warning is missing as an error.
    pub missing: Vec<String>,
    /// Codes the validator reported that the case does not expect at the
    /// severity reported.
    pub unexpected: Vec<String>,
    /// Why the case could not be run at all, when it could not.
    pub error: Option<String>,
    /// Every diagnostic the validator reported, one line each.
    pub details: Vec<String>,
}

impl CaseResult {
    /// A result for `case` at `path`, with nothing reported yet.
    pub fn new(case: Case, path: String) -> CaseResult {
        CaseResult {
            case,
            path,
            got_errors: Vec::new(),
            got_warnings: Vec::new(),
            missing: Vec::new(),
            unexpected: Vec::new(),
            error: None,
            details: Vec::new(),
        }
    }

    /// Whether the validator reported exactly the expected codes.
    pub fn ok(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty() && self.error.is_none()
    }

    /// The result as `conformance run --json` prints it.
    pub fn to_json(&self) -> Value {
        let sorted = |v: &[String]| -> Vec<String> {
            let mut s = v.to_vec();
            s.sort();
            s
        };
        json!({
            "name": self.case.name,
            "ok": self.ok(),
            "expect_errors": sorted(&self.case.errors),
            "got_errors": sorted(&self.got_errors),
            "expect_warnings": sorted(&self.case.warnings),
            "got_warnings": sorted(&self.got_warnings),
            "missing": sorted(&self.missing),
            "unexpected": sorted(&self.unexpected),
            "error": self.error,
        })
    }

    /// Fill in what a validator reported against what the case expects.
    pub fn compare(&mut self, errors: BTreeSet<String>, warnings: BTreeSet<String>) {
        let (missing, unexpected) = differences(&self.case.errors, &self.case.warnings, &errors, &warnings);
        self.got_errors = errors.into_iter().collect();
        self.got_warnings = warnings.into_iter().collect();
        self.missing = missing;
        self.unexpected = unexpected;
    }
}

/// What was expected and not reported, and reported and not expected, each
/// at its severity: an error reported as a warning is both.  The codes alone
/// were compared, so a validator reporting every code as a warning ---
/// rejecting nothing --- passed every case.
pub(crate) fn differences(
    expect_errors: &[String],
    expect_warnings: &[String],
    errors: &BTreeSet<String>,
    warnings: &BTreeSet<String>,
) -> (Vec<String>, Vec<String>) {
    let expect_errors: BTreeSet<String> = expect_errors.iter().cloned().collect();
    let expect_warnings: BTreeSet<String> = expect_warnings.iter().cloned().collect();
    let missing: BTreeSet<String> =
        expect_errors.difference(errors).chain(expect_warnings.difference(warnings)).cloned().collect();
    let unexpected: BTreeSet<String> =
        errors.difference(&expect_errors).chain(warnings.difference(&expect_warnings)).cloned().collect();
    (missing.into_iter().collect(), unexpected.into_iter().collect())
}

/// Every case, in the order the corpus lists them.
pub fn cases() -> &'static [Case] {
    static CASES: OnceLock<Vec<Case>> = OnceLock::new();
    CASES.get_or_init(build::registry)
}

/// The case called `name`, or [`Error::Key`].
pub fn case_by_name(name: &str) -> Result<&'static Case> {
    cases()
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| Error::Key(format!("unknown conformance case {}", repr_str(name))))
}

fn selected(names: Option<&[String]>) -> Vec<&'static Case> {
    cases().iter().filter(|c| names.is_none_or(|n| n.contains(&c.name))).collect()
}

/// Write every case and an `expected.json` manifest beside them.
pub fn build_corpus(outdir: &Path, names: Option<&[String]>) -> Result<PathBuf> {
    std::fs::create_dir_all(outdir)?;
    for entry in selected(names) {
        build_case(outdir, entry)?;
    }
    write_expected(outdir, names)
}

/// Write `entry`'s file in `outdir`, over any there; its path.
fn build_case(outdir: &Path, entry: &Case) -> Result<PathBuf> {
    let path = outdir.join(format!("{}{}", entry.name, entry.suffix));
    if path.exists() {
        std::fs::remove_file(&path)?;
    }
    if let Some(build) = &entry.build {
        build(&path)?;
    }
    Ok(path)
}

/// The `expected.json` manifest of the cases `names` selects, written in
/// `outdir` --- what [`build_corpus`] writes once the cases are.  Public for
/// the frontends: `conformance run --workers` builds the cases in several
/// processes ([`run_case`]) and writes the manifest after them.
#[doc(hidden)]
pub fn write_expected(outdir: &Path, names: Option<&[String]>) -> Result<PathBuf> {
    let mut records = Vec::new();
    for entry in selected(names) {
        let mut record = match entry.to_json() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        record.insert("file".into(), json!(format!("{}{}", entry.name, entry.suffix)));
        records.push(Value::Object(record));
    }
    let manifest = json!({
        "format": crate::FORMAT_VERSION,
        "generator": format!("medh5 {}", crate::VERSION),
        "codes_revision": crate::codes::table_revision(),
        "cases": records,
    });
    let target = outdir.join("expected.json");
    std::fs::write(&target, crate::json::pretty(&manifest) + "\n")?;
    Ok(target)
}

/// Build every case, validate it, and compare against its expected codes.
pub fn run_corpus(outdir: &Path, names: Option<&[String]>) -> Result<Vec<CaseResult>> {
    build_corpus(outdir, names)?;
    Ok(selected(names).into_iter().map(|entry| check_case(outdir, entry)).collect())
}

/// One case of [`run_corpus`]: build it in `outdir` and check it, without
/// the manifest ([`write_expected`]).  Public for the frontends.
#[doc(hidden)]
pub fn run_case(outdir: &Path, entry: &Case) -> Result<CaseResult> {
    std::fs::create_dir_all(outdir)?;
    build_case(outdir, entry)?;
    Ok(check_case(outdir, entry))
}

/// Validate `entry`'s file in `outdir` against its expected codes.
fn check_case(outdir: &Path, entry: &Case) -> CaseResult {
    let path = outdir.join(format!("{}{}", entry.name, entry.suffix));
    let mut result = CaseResult::new(entry.clone(), crate::pyval::py_path(&path));
    match validate_file(&path, &entry.level, None) {
        Err(e) => result.error = Some(e.python_line()),
        Ok(report) => {
            // By the code table's severity, as a submission states it,
            // whatever the level promotes.
            let by = |severity: &str| -> BTreeSet<String> {
                report.diagnostics.iter().filter(|d| d.severity == severity).map(|d| d.code.clone()).collect()
            };
            result.compare(by("error"), by("warning"));
            result.details = report.diagnostics.iter().map(|d| d.line()).collect();
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::build::{mutate, restamp, set_str};
    use super::*;

    fn codes_at(report: &crate::validate::Report, location: &str) -> Vec<String> {
        report.diagnostics.iter().filter(|d| d.location == location).map(|d| d.code.clone()).collect()
    }

    /// 1.2 §2.2: a 1.2 file is read by a 1.1 implementation --- the reader
    /// medh5 2.1.0 is --- as its projection, the version W913 at `/` and
    /// nothing else wrong, and fully by this one.
    #[test]
    fn v1_2_s2_2_a_1_1_reader_projects_a_1_2_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("minimal.medh5");
        case_by_name("core-minimal").unwrap().build.as_ref().unwrap()(&path).unwrap();
        mutate(&path, |root| set_str(root, "", "medh5_version", "1.2")).unwrap();
        restamp(&path).unwrap();
        let here = validate_file(&path, "integrity", None).unwrap();
        assert!(here.ok() && codes_at(&here, "/").is_empty(), "{:?}", here.diagnostics);
        let earlier = crate::version::read_as("1.1", || validate_file(&path, "integrity", None)).unwrap();
        assert_eq!(codes_at(&earlier, "/"), ["W913"], "{:?}", earlier.diagnostics);
        assert!(earlier.ok(), "{:?}", earlier.diagnostics);
        assert_eq!(
            crate::version::read_as("1.1", || crate::version::support("1.2")),
            crate::version::Support::Projection
        );
    }

    /// 1.2 §2.2, §3: a 1.1 reader leaves each polylines annotation out as
    /// W913, recomputes the `content_id` and finds it matches, and validates
    /// the rest; this engine reads them as the 1.2 they are.
    #[test]
    fn v1_2_s3_a_1_1_reader_projects_polylines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("polylines.medh5");
        let case = case_by_name("shapes-polylines").unwrap();
        assert_eq!(case.format_version(), "1.2");
        case.build.as_ref().unwrap()(&path).unwrap();
        let here = validate_file(&path, "integrity", None).unwrap();
        assert_eq!(here.codes(), ["W912"], "{:?}", here.diagnostics);
        let earlier = crate::version::read_as("1.1", || validate_file(&path, "integrity", None)).unwrap();
        assert!(earlier.ok(), "{:?}", earlier.diagnostics);
        for at in ["/", "/annotations/catheter", "/annotations/centreline"] {
            assert_eq!(codes_at(&earlier, at), ["W913"], "{at}: {:?}", earlier.diagnostics);
        }
        let (mut warned, mut expected): (Vec<String>, Vec<&str>) = (earlier.codes(), vec!["W912", "W913"]);
        warned.dedup();
        expected.sort();
        assert_eq!(warned, expected, "{:?}", earlier.diagnostics);
        let sample = crate::version::read_as("1.1", || crate::sample::open_sample(&path)).unwrap();
        let listed = crate::version::read_as("1.1", || sample.annotations().map(|a| a.len())).unwrap();
        assert_eq!(listed, 0);
        let sample = crate::sample::open_sample(&path).unwrap();
        let mut ids: Vec<&String> = sample.annotations().unwrap().keys().collect();
        ids.sort();
        assert_eq!(ids, ["catheter", "centreline"]);
    }
}
