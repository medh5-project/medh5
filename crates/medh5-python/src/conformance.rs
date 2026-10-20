//! `medh5.conformance` and `medh5.bench`: the conformance corpus and suite
//! (spec §15), and the performance measurements.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::{json, Value};

use medh5::bench;
use medh5::conformance::{self as engine, suite};

use crate::convert::{json_to_py, py_to_json, py_to_json_lenient, strings};
use crate::errors::R;

/// A result as the Python `CaseResult` is built from: the case record, the
/// path, the codes and what went wrong.
fn result_json(r: &engine::CaseResult) -> Value {
    json!({
        "case": r.case.to_json(),
        "path": r.path,
        "got_errors": r.got_errors,
        "got_warnings": r.got_warnings,
        "missing": r.missing,
        "unexpected": r.unexpected,
        "error": r.error,
        "details": r.details,
    })
}

fn results_to_py<'py>(py: Python<'py>, results: &[engine::CaseResult]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for r in results {
        out.append(json_to_py(py, &result_json(r))?)?;
    }
    Ok(out)
}

/// Every case, as its manifest record, in corpus order.
#[pyfunction]
fn conformance_cases<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for case in engine::cases() {
        out.append(json_to_py(py, &case.to_json())?)?;
    }
    Ok(out)
}

/// Write one corpus case's file to `path`.
#[pyfunction]
fn conformance_build_case(py: Python<'_>, name: &str, path: PathBuf) -> R<()> {
    let case = engine::case_by_name(name)?;
    let Some(build) = case.build.clone() else {
        return Err(medh5::Error::invalid(format!("case {} has no builder", medh5::json::repr_str(name))).into());
    };
    Ok(py.detach(move || build(&path))?)
}

#[pyfunction]
#[pyo3(signature = (outdir, *, names=None))]
fn conformance_build_corpus(py: Python<'_>, outdir: PathBuf, names: Option<Vec<String>>) -> R<String> {
    let manifest = py.detach(move || engine::build_corpus(&outdir, names.as_deref()))?;
    Ok(manifest.to_string_lossy().into_owned())
}

#[pyfunction]
#[pyo3(signature = (outdir, *, names=None))]
fn conformance_run_corpus<'py>(py: Python<'py>, outdir: PathBuf, names: Option<Vec<String>>) -> R<Bound<'py, PyList>> {
    let results = py.detach(move || engine::run_corpus(&outdir, names.as_deref()))?;
    Ok(results_to_py(py, &results)?)
}

#[pyfunction]
#[pyo3(signature = (outdir, *, names=None))]
fn conformance_publish(py: Python<'_>, outdir: PathBuf, names: Option<Vec<String>>) -> R<String> {
    let root = py.detach(move || suite::publish(&outdir, names.as_deref()))?;
    Ok(root.to_string_lossy().into_owned())
}

#[pyfunction]
fn conformance_check_checksums(py: Python<'_>, root: PathBuf) -> R<Vec<String>> {
    Ok(py.detach(move || suite::check_checksums(&root))?)
}

#[pyfunction]
fn conformance_load_manifest<'py>(py: Python<'py>, root: PathBuf) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &suite::load_manifest(&root)?)?)
}

#[pyfunction]
#[pyo3(signature = (root, submitted, *, format_version=None, codes_revision=None))]
fn conformance_score<'py>(
    py: Python<'py>,
    root: PathBuf,
    submitted: &Bound<'py, PyAny>,
    format_version: Option<String>,
    codes_revision: Option<u32>,
) -> R<Bound<'py, PyList>> {
    let entries: Vec<Value> = match py_to_json(submitted)? {
        Value::Array(items) => items,
        other => vec![other],
    };
    let mut options = suite::ScoreOptions::default();
    if let Some(version) = &format_version {
        options = options.format_version(version);
    }
    if let Some(revision) = codes_revision {
        options = options.codes_revision(revision);
    }
    let results = suite::score_with(&root, &entries, &options)?;
    Ok(results_to_py(py, &results)?)
}

#[pyfunction]
fn conformance_score_companion<'py>(
    py: Python<'py>,
    root: PathBuf,
    submitted: &Bound<'py, PyAny>,
) -> R<Bound<'py, PyList>> {
    let entries: Vec<Value> = match py_to_json(submitted)? {
        Value::Array(items) => items,
        other => vec![other],
    };
    let results = engine::companion::score(&root, &entries)?;
    Ok(results_to_py(py, &results)?)
}

/// The engine value of a `Case` dataclass: what it expects, not how it is
/// built.
fn case_arg(obj: &Bound<'_, PyAny>) -> PyResult<engine::Case> {
    Ok(engine::Case {
        name: obj.getattr("name")?.extract()?,
        description: obj.getattr("description")?.extract()?,
        clause: obj.getattr("clause")?.extract()?,
        build: None,
        level: obj.getattr("level")?.extract()?,
        errors: strings(&obj.getattr("errors")?)?,
        warnings: strings(&obj.getattr("warnings")?)?,
        suffix: obj.getattr("suffix")?.extract()?,
        mutated: obj.getattr("mutated")?.is_truthy()?,
    })
}

/// The engine value of a `CaseResult` dataclass.
fn case_result_arg(obj: &Bound<'_, PyAny>) -> PyResult<engine::CaseResult> {
    Ok(engine::CaseResult {
        case: case_arg(&obj.getattr("case")?)?,
        path: obj.getattr("path")?.extract()?,
        got_errors: strings(&obj.getattr("got_errors")?)?,
        got_warnings: strings(&obj.getattr("got_warnings")?)?,
        missing: strings(&obj.getattr("missing")?)?,
        unexpected: strings(&obj.getattr("unexpected")?)?,
        error: obj.getattr("error")?.extract()?,
        details: strings(&obj.getattr("details")?)?,
    })
}

/// A case's entry in `expected.json`.
#[pyfunction]
fn conformance_case_json<'py>(py: Python<'py>, case: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &case_arg(case)?.to_json())
}

/// A result as `conformance run --json` prints it.
#[pyfunction]
fn conformance_result_json<'py>(py: Python<'py>, result: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &case_result_arg(result)?.to_json())
}

/// The `--json` summary of a run or a score.
#[pyfunction]
fn conformance_summary<'py>(py: Python<'py>, results: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let mut all = Vec::new();
    for r in results.try_iter()? {
        all.push(case_result_arg(&r?)?);
    }
    json_to_py(py, &suite::summarize(&all))
}

// -- bench -------------------------------------------------------------------------------

/// The engine value of a `Measurement` dataclass, or of its JSON form.
/// A NaN value is a measurement (a metric that measured nothing), which the
/// JSON form spells `null`.
fn measurement_arg(obj: &Bound<'_, PyAny>) -> R<bench::Measurement> {
    if obj.is_instance_of::<PyDict>() {
        return Ok(bench::Measurement::from_json(&py_to_json(obj)?)?);
    }
    let detail = match py_to_json_lenient(&obj.getattr("detail")?)? {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    Ok(bench::Measurement {
        name: obj.getattr("name")?.str()?.to_string(),
        value: obj.getattr("value")?.extract()?,
        unit: obj.getattr("unit")?.str()?.to_string(),
        target: obj.getattr("target")?.extract()?,
        description: obj.getattr("description")?.str()?.to_string(),
        detail,
    })
}

/// A measurement as `bench --json` prints it: a NaN as `null`.
#[pyfunction]
fn bench_measurement_json<'py>(py: Python<'py>, measurement: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &measurement_arg(measurement)?.to_json())?)
}

/// A count from Python, where a negative one reaches the engine as 0, which
/// it refuses for what 0 would measure: taken unsigned, it raised PyO3's
/// `OverflowError` instead.
fn count(value: i64) -> usize {
    if value < 0 {
        0
    } else {
        usize::try_from(value).unwrap_or(usize::MAX)
    }
}

/// Median milliseconds per call of a Python callable.
#[pyfunction]
#[pyo3(signature = (r#fn, *, repeats=20, warmup=3))]
fn bench_timed(py: Python<'_>, r#fn: Py<PyAny>, repeats: i64, warmup: i64) -> R<f64> {
    let Ok(warmup) = usize::try_from(warmup) else {
        return Err(medh5::Error::invalid(format!("warmup must be at least 0, not {warmup}")).into());
    };
    let repeats = count(repeats);
    let mut failure: Option<PyErr> = None;
    let value = bench::timed(
        || match r#fn.call0(py) {
            Ok(_) => Ok(()),
            Err(e) => {
                failure = Some(e);
                Err(medh5::Error::Runtime("the timed callable raised".into()))
            }
        },
        repeats,
        warmup,
    );
    if let Some(e) = failure {
        return Err(e.into());
    }
    Ok(value?)
}

fn measurements_to_py<'py>(py: Python<'py>, found: &[bench::Measurement]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for m in found {
        out.append(json_to_py(py, &m.to_json())?)?;
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (path, *, annotation=None, patch=64, repeats=20))]
fn bench_benchmark_file<'py>(
    py: Python<'py>,
    path: PathBuf,
    annotation: Option<String>,
    patch: i64,
    repeats: i64,
) -> R<Bound<'py, PyList>> {
    let (patch, repeats) = (count(patch), count(repeats));
    let found = py.detach(move || bench::benchmark_file(&path, annotation.as_deref(), patch, repeats))?;
    Ok(measurements_to_py(py, &found)?)
}

#[pyfunction]
#[pyo3(signature = (directory, *, shape=vec![64, 96, 96], codec="training".to_string(), seed=20260815))]
fn bench_synthetic_pair(py: Python<'_>, directory: PathBuf, shape: Vec<usize>, codec: String, seed: u64) -> R<String> {
    let path = py.detach(move || bench::synthetic_pair(&directory, &shape, &codec, seed))?;
    Ok(path.to_string_lossy().into_owned())
}

#[pyfunction]
#[pyo3(signature = (directory, *, shape=vec![192, 256, 256], classes=8, codec="training".to_string(), index=true,
    seed=20260815, name="bench.medh5".to_string()))]
#[allow(clippy::too_many_arguments)]
fn bench_synthetic_sample(
    py: Python<'_>,
    directory: PathBuf,
    shape: Vec<usize>,
    classes: usize,
    codec: String,
    index: bool,
    seed: u64,
    name: String,
) -> R<String> {
    let path = py.detach(move || bench::synthetic_sample(&directory, &shape, classes, &codec, index, seed, &name))?;
    Ok(path.to_string_lossy().into_owned())
}

#[pyfunction]
#[pyo3(signature = (directory, *, classes=bench::MANY_CLASSES))]
fn bench_synthetic_many_class_sample(py: Python<'_>, directory: PathBuf, classes: usize) -> R<String> {
    let path = py.detach(move || bench::synthetic_many_class_sample(&directory, classes))?;
    Ok(path.to_string_lossy().into_owned())
}

#[pyfunction]
#[pyo3(signature = (path, *, patch=64, repeats=20))]
fn bench_many_class_measurement<'py>(py: Python<'py>, path: PathBuf, patch: i64, repeats: i64) -> R<Bound<'py, PyAny>> {
    let (patch, repeats) = (count(patch), count(repeats));
    let found = py.detach(move || bench::many_class_measurement(&path, patch, repeats))?;
    Ok(json_to_py(py, &found.to_json())?)
}

/// The text report of measurements, given as `Measurement`s or their JSON
/// form.
#[pyfunction]
fn bench_report(measurements: &Bound<'_, PyAny>) -> R<String> {
    let mut found = Vec::new();
    for item in measurements.try_iter()? {
        found.push(measurement_arg(&item?)?);
    }
    Ok(bench::report(&found))
}

/// One measurement's report line (`str(measurement)`), given as a
/// `Measurement` or its JSON form.
#[pyfunction]
fn bench_line(measurement: &Bound<'_, PyAny>) -> R<String> {
    Ok(measurement_arg(measurement)?.to_string())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(conformance_cases, m)?,
        wrap_pyfunction!(conformance_build_case, m)?,
        wrap_pyfunction!(conformance_build_corpus, m)?,
        wrap_pyfunction!(conformance_run_corpus, m)?,
        wrap_pyfunction!(conformance_publish, m)?,
        wrap_pyfunction!(conformance_check_checksums, m)?,
        wrap_pyfunction!(conformance_load_manifest, m)?,
        wrap_pyfunction!(conformance_score, m)?,
        wrap_pyfunction!(conformance_case_json, m)?,
        wrap_pyfunction!(conformance_result_json, m)?,
        wrap_pyfunction!(conformance_summary, m)?,
        wrap_pyfunction!(bench_measurement_json, m)?,
        wrap_pyfunction!(conformance_score_companion, m)?,
        wrap_pyfunction!(bench_timed, m)?,
        wrap_pyfunction!(bench_benchmark_file, m)?,
        wrap_pyfunction!(bench_synthetic_pair, m)?,
        wrap_pyfunction!(bench_synthetic_sample, m)?,
        wrap_pyfunction!(bench_synthetic_many_class_sample, m)?,
        wrap_pyfunction!(bench_many_class_measurement, m)?,
        wrap_pyfunction!(bench_report, m)?,
        wrap_pyfunction!(bench_line, m)?,
    ] {
        m.add_function(f)?;
    }
    let py = m.py();
    m.add("CONFORMANCE_SEED", engine::SEED)?;
    m.add("CONFORMANCE_SCHEMA", suite::SCHEMA)?;
    m.add("CONFORMANCE_CHECKSUMS", suite::CHECKSUMS)?;
    let targets = PyDict::new(py);
    for (name, target, description) in bench::TARGETS {
        targets.set_item(name, (target, description))?;
    }
    m.add("BENCH_TARGETS", targets)?;
    m.add("BENCH_MANY_CLASSES", bench::MANY_CLASSES)?;
    Ok(())
}
