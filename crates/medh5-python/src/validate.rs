//! `medh5.validate`: the validator (spec §15) and the code table.
//!
//! Reports cross as their JSON form; `medh5/validate.py` turns them into
//! the `Report` and `Diagnostic` dataclasses 1.x returned.  What a report
//! says of itself --- its verdict, its counts at a level, its text and JSON
//! --- is the engine's: the `validate_report_*` functions rebuild the engine
//! report from the dataclass and ask it.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyList;

use medh5::validate as engine;

use crate::convert::{json_to_py, opt_strings, py_to_json_lenient, strings};
use crate::errors::R;
use crate::reader::SampleHandle;

/// The normative diagnostic code table (§15.2), as JSON text.
#[pyfunction]
fn codes_table() -> &'static str {
    medh5::codes::table_json()
}

/// The engine's options for `level`, `profiles` and `codes_revision`; a
/// revision below 0 is refused here (it is no `u32`), the rest by the engine.
fn options(level: &str, profiles: Option<Vec<String>>, codes_revision: Option<i64>) -> R<engine::ValidateOptions> {
    let mut options = engine::ValidateOptions::default().level(level);
    if let Some(profiles) = profiles {
        options = options.profiles(&profiles);
    }
    if let Some(n) = codes_revision {
        let revision = u32::try_from(n).map_err(|_| {
            medh5::Error::invalid(format!(
                "codes revision {n} is not a revision of the code table; this engine implements revisions 1 to {} \
                 (§15.2)",
                medh5::codes::table_revision()
            ))
        })?;
        options = options.codes_revision(revision);
    }
    Ok(options)
}

#[pyfunction]
#[pyo3(signature = (path, *, level="semantic", profiles=None, codes_revision=None))]
fn validate_file<'py>(
    py: Python<'py>,
    path: PathBuf,
    level: &str,
    profiles: Option<&Bound<'py, PyAny>>,
    codes_revision: Option<i64>,
) -> R<Bound<'py, PyAny>> {
    let options = options(level, opt_strings(profiles)?, codes_revision)?;
    let report = py.detach(move || engine::validate_file_with(&path, &options))?;
    Ok(json_to_py(py, &report.to_json())?)
}

#[pyfunction]
#[pyo3(signature = (paths, *, level="semantic", profiles=None, codes_revision=None))]
fn validate_paths<'py>(
    py: Python<'py>,
    paths: Vec<PathBuf>,
    level: &str,
    profiles: Option<&Bound<'py, PyAny>>,
    codes_revision: Option<i64>,
) -> R<Bound<'py, PyList>> {
    let options = options(level, opt_strings(profiles)?, codes_revision)?;
    let reports = py.detach(move || {
        let refs: Vec<&std::path::Path> = paths.iter().map(PathBuf::as_path).collect();
        engine::validate_paths_with(&refs, &options)
    })?;
    let out = PyList::empty(py);
    for r in reports {
        out.append(json_to_py(py, &r.to_json())?)?;
    }
    Ok(out)
}

/// Validate an open sample (`Sample`'s handle), with `errors_only` skipping
/// the warning-only checks that read bulk data.
#[pyfunction]
#[pyo3(signature = (sample, *, path=None, level="semantic", profiles=None, errors_only=false))]
fn validate_root<'py>(
    py: Python<'py>,
    sample: &Bound<'py, PyAny>,
    path: Option<String>,
    level: &str,
    profiles: Option<&Bound<'py, PyAny>>,
    errors_only: bool,
) -> R<Bound<'py, PyAny>> {
    engine::check_level(level)?;
    let handle = match sample.cast::<SampleHandle>() {
        Ok(h) => h.clone(),
        Err(_) => sample.getattr("_handle")?.cast_into::<SampleHandle>().map_err(PyErr::from)?,
    };
    let handle = handle.get();
    let engine_sample = crate::h5(py, || handle.sample())?;
    let profiles = opt_strings(profiles)?;
    let shown = path.or_else(|| engine_sample.path.as_ref().map(|p| p.to_string_lossy().into_owned()));
    let shown = shown.unwrap_or_else(|| "<memory>".into());
    let level = level.to_string();
    let report = py.detach(move || {
        let names = engine::attr_names_for(&engine_sample.root, &level);
        engine::validate_root_with(
            &engine_sample.root,
            &shown,
            &level,
            profiles.as_deref(),
            names.as_ref(),
            errors_only,
        )
    })?;
    Ok(json_to_py(py, &report.to_json())?)
}

/// Every rule run at `level`, in order.  An unknown level is refused: every
/// rule, which it was given, read as the answer for a misspelt level.
#[pyfunction]
fn rules_for(level: &str) -> R<Vec<&'static str>> {
    engine::check_level(level)?;
    Ok(engine::rules::rules_for(level).into_iter().map(|(n, _)| n).collect())
}

/// The engine value of a `Diagnostic` dataclass.
fn diagnostic_arg(obj: &Bound<'_, PyAny>) -> PyResult<engine::Diagnostic> {
    Ok(engine::Diagnostic {
        code: obj.getattr("code")?.extract()?,
        location: obj.getattr("location")?.extract()?,
        message: obj.getattr("message")?.extract()?,
        severity: obj.getattr("severity")?.extract()?,
        level: obj.getattr("level")?.extract()?,
    })
}

/// The engine value of a `Report` dataclass.
fn report_arg(obj: &Bound<'_, PyAny>) -> PyResult<engine::Report> {
    let mut diagnostics = Vec::new();
    for d in obj.getattr("diagnostics")?.try_iter()? {
        diagnostics.push(diagnostic_arg(&d?)?);
    }
    Ok(engine::Report {
        path: obj.getattr("path")?.extract()?,
        level: obj.getattr("level")?.extract()?,
        profiles: strings(&obj.getattr("profiles")?)?,
        diagnostics,
        checked: py_to_json_lenient(&obj.getattr("checked")?)?,
    })
}

/// The positions in `report.diagnostics` of its errors, its warnings and
/// the warnings it promotes, at its level.
#[pyfunction]
fn validate_report_positions(report: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, Vec<usize>, Vec<usize>)> {
    let report = report_arg(report)?;
    let at = |found: Vec<&engine::Diagnostic>| -> Vec<usize> {
        found.iter().filter_map(|d| report.diagnostics.iter().position(|g| std::ptr::eq(g, *d))).collect()
    };
    Ok((at(report.errors()), at(report.warnings()), at(report.promoted())))
}

/// A report as `validate --json` prints it.
#[pyfunction]
fn validate_report_json<'py>(py: Python<'py>, report: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &report_arg(report)?.to_json())
}

/// A report as `validate` prints it.
#[pyfunction]
#[pyo3(signature = (report, *, verbose=false))]
fn validate_report_format(report: &Bound<'_, PyAny>, verbose: bool) -> PyResult<String> {
    Ok(report_arg(report)?.format(verbose))
}

/// Several reports as one, each location prefixed with its file: the
/// merged report's JSON.
#[pyfunction]
#[pyo3(signature = (reports, path="<multiple>"))]
fn validate_merge<'py>(py: Python<'py>, reports: &Bound<'py, PyAny>, path: &str) -> PyResult<Bound<'py, PyAny>> {
    let mut all = Vec::new();
    for r in reports.try_iter()? {
        all.push(report_arg(&r?)?);
    }
    json_to_py(py, &engine::merge(&all, path).to_json())
}

/// A code's name, or the engine's `KeyError` for one the table lacks.
#[pyfunction]
fn code_lookup(name: &str) -> R<&'static str> {
    Ok(medh5::codes::code(name)?.code.as_str())
}

/// The codes of one domain, in table order.
#[pyfunction]
fn codes_for_domain(domain: &str) -> Vec<&'static str> {
    medh5::codes::for_domain(domain).into_iter().map(|c| c.code.as_str()).collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(codes_for_domain, m)?)?;
    m.add_function(wrap_pyfunction!(code_lookup, m)?)?;
    m.add("CODE_DOMAINS", pyo3::types::PyTuple::new(m.py(), medh5::codes::DOMAINS)?)?;
    m.add_function(wrap_pyfunction!(validate_report_positions, m)?)?;
    m.add_function(wrap_pyfunction!(validate_report_json, m)?)?;
    m.add_function(wrap_pyfunction!(validate_report_format, m)?)?;
    m.add_function(wrap_pyfunction!(validate_merge, m)?)?;
    m.add_function(wrap_pyfunction!(codes_table, m)?)?;
    m.add_function(wrap_pyfunction!(validate_file, m)?)?;
    m.add_function(wrap_pyfunction!(validate_paths, m)?)?;
    m.add_function(wrap_pyfunction!(validate_root, m)?)?;
    m.add_function(wrap_pyfunction!(rules_for, m)?)?;
    m.add("LEVELS", pyo3::types::PyTuple::new(m.py(), engine::LEVELS)?)?;
    Ok(())
}
