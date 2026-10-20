//! `medh5.io`: the conversion report, study grouping and the 0.x migration
//! (spec §3.7, Appendix B).
//!
//! The library-specific converters (NIfTI, DICOM, DICOM SEG, RTSTRUCT,
//! nnU-Net) are Python integrations; what they share with the native
//! migration --- the report's text and JSON, the grouping rules, the key and
//! file-name sanitisers --- is the engine's, bound here.  The report and the
//! grouping types stay the 1.x dataclasses: converters append to
//! `report.outputs` and `report.notes`, and occasions carry arbitrary Python
//! payloads, which travel through the engine as positions.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};
use serde_json::{Map, Value};

use medh5::convert::grouping as gr;
use medh5::convert::legacy as lg;
use medh5::convert::report::{self as rp, ConversionReport, Note};

use crate::convert::{array_to_py, json_to_py, map_to_py, nd_to_py, py_to_json_lenient, strings};
use crate::errors::R;
use crate::labels::{label_set_arg, LabelSet};

// -- the report ----------------------------------------------------------------------------

fn note_arg(obj: &Bound<'_, PyAny>) -> PyResult<Note> {
    let detail = match py_to_json_lenient(&obj.getattr("detail")?)? {
        Value::Object(m) => m,
        Value::Null => Map::new(),
        other => {
            let mut m = Map::new();
            m.insert("value".into(), other);
            m
        }
    };
    Ok(Note {
        kind: obj.getattr("kind")?.extract()?,
        message: obj.getattr("message")?.extract()?,
        severity: obj.getattr("severity")?.extract()?,
        detail,
    })
}

/// The engine value of a `ConversionReport` dataclass.
pub(crate) fn report_arg(obj: &Bound<'_, PyAny>) -> PyResult<ConversionReport> {
    let mut notes = Vec::new();
    for note in obj.getattr("notes")?.try_iter()? {
        notes.push(note_arg(&note?)?);
    }
    Ok(ConversionReport {
        source: obj.getattr("source")?.extract()?,
        converter: obj.getattr("converter")?.extract()?,
        outputs: strings(&obj.getattr("outputs")?)?,
        notes,
    })
}

fn note_fields<'py>(py: Python<'py>, note: &Note) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("kind", &note.kind)?;
    out.set_item("message", &note.message)?;
    out.set_item("severity", &note.severity)?;
    out.set_item("detail", map_to_py(py, &note.detail)?)?;
    Ok(out)
}

fn notes_fields<'py>(py: Python<'py>, notes: &[Note]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for note in notes {
        out.append(note_fields(py, note)?)?;
    }
    Ok(out)
}

/// A `ConversionReport`'s fields, notes as field dicts.
pub(crate) fn report_fields<'py>(py: Python<'py>, r: &ConversionReport) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("source", &r.source)?;
    out.set_item("converter", &r.converter)?;
    out.set_item("outputs", PyList::new(py, &r.outputs)?)?;
    out.set_item("notes", notes_fields(py, &r.notes)?)?;
    Ok(out)
}

/// A scratch report to collect what an engine step notes, for appending to
/// the caller's.
fn scratch() -> ConversionReport {
    ConversionReport::new("", "")
}

#[pyfunction]
fn io_report_ok(report: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(report_arg(report)?.ok())
}

#[pyfunction]
fn io_report_json<'py>(py: Python<'py>, report: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &report_arg(report)?.to_json())
}

#[pyfunction]
#[pyo3(signature = (report, *, verbose=false))]
fn io_report_format(report: &Bound<'_, PyAny>, verbose: bool) -> PyResult<String> {
    Ok(report_arg(report)?.format(verbose))
}

#[pyfunction]
fn io_note_line(note: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(note_arg(note)?.line())
}

#[pyfunction]
fn io_note_json<'py>(py: Python<'py>, note: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &note_arg(note)?.to_json())
}

// -- sanitisers -----------------------------------------------------------------------------

/// A label-set `key` from free text (§5.2): `^[a-z0-9][a-z0-9_]*$`.
#[pyfunction]
#[pyo3(signature = (name, *, fallback="class"))]
fn io_sanitize_key(name: &Bound<'_, PyAny>, fallback: &str) -> PyResult<String> {
    Ok(gr::sanitize_key(&name.str()?.to_string(), fallback))
}

/// A filename stem from free text: identifier characters only, truncated.
#[pyfunction]
#[pyo3(signature = (text, *, limit=200))]
fn io_sanitize_stem(text: &Bound<'_, PyAny>, limit: usize) -> PyResult<String> {
    Ok(gr::sanitize_stem(&text.str()?.to_string(), limit))
}

// -- grouping -------------------------------------------------------------------------------

fn opt_str(obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if obj.is_none() {
        Ok(None)
    } else {
        Ok(Some(obj.str()?.to_string()))
    }
}

/// The engine form of an `Occasion` dataclass; `payload` is its position.
fn occasion_arg(obj: &Bound<'_, PyAny>, position: usize) -> PyResult<gr::Occasion> {
    let mut demographics = IndexMap::new();
    let found = obj.getattr("demographics")?;
    if !found.is_none() {
        for item in found.call_method0("items")?.try_iter()? {
            let (k, v): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
            demographics.insert(k.str()?.to_string(), v.str()?.to_string());
        }
    }
    let order_hint = obj.getattr("order_hint")?;
    Ok(gr::Occasion {
        key: obj.getattr("key")?.str()?.to_string(),
        subject_id: opt_str(&obj.getattr("subject_id")?)?,
        date: opt_str(&obj.getattr("date")?)?,
        order_hint: if order_hint.is_none() { None } else { Some(order_hint.extract()?) },
        demographics,
        payload: position,
    })
}

fn occasions_arg(occasions: &Bound<'_, PyAny>) -> PyResult<Vec<gr::Occasion>> {
    occasions.try_iter()?.enumerate().map(|(i, o)| occasion_arg(&o?, i)).collect()
}

/// Group occasions into subjects: `(groups, notes)`, a group being
/// `(subject_id, positions, ordered_by)` and the notes what grouping recorded.
#[pyfunction]
#[pyo3(signature = (occasions, *, mode="subject"))]
fn io_group_by_subject<'py>(
    py: Python<'py>,
    occasions: &Bound<'py, PyAny>,
    mode: &str,
) -> R<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let mut log = scratch();
    let groups = gr::group_by_subject(occasions_arg(occasions)?, mode, Some(&mut log))?;
    let out = PyList::empty(py);
    for g in groups {
        let positions: Vec<usize> = g.occasions.iter().map(|o| o.payload).collect();
        out.append((g.subject_id, positions, g.ordered_by))?;
    }
    Ok((out, notes_fields(py, &log.notes)?))
}

/// Subject keys the sources contradict -> the facts that disagree.
#[pyfunction]
fn io_contradictions<'py>(py: Python<'py>, occasions: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let found = occasions_arg(occasions)?;
    let refs: Vec<&gr::Occasion> = found.iter().collect();
    let out = PyDict::new(py);
    for (k, v) in gr::contradictions(&refs) {
        out.set_item(k, v)?;
    }
    Ok(out)
}

fn group_of(subject_id: &str, keys: Vec<String>, dates: Vec<Option<String>>) -> gr::SubjectGroup {
    let mut group = gr::SubjectGroup::new(subject_id);
    for (i, key) in keys.into_iter().enumerate() {
        group.occasions.push(gr::Occasion {
            key,
            date: dates.get(i).cloned().flatten(),
            payload: i,
            ..Default::default()
        });
    }
    group
}

/// Days from the first visit for each date (`None` where one is missing).
#[pyfunction]
fn io_days_from_baseline(dates: Vec<Option<String>>) -> Vec<Option<i64>> {
    let keys = vec![String::new(); dates.len()];
    group_of("", keys, dates).days_from_baseline()
}

/// A unique filename stem for one group; adds it to `used`.
#[pyfunction]
#[pyo3(signature = (subject_id, keys, used, safe=None))]
fn io_output_name(
    subject_id: &str,
    keys: Vec<String>,
    used: &Bound<'_, PySet>,
    safe: Option<&Bound<'_, PyAny>>,
) -> PyResult<String> {
    let n = keys.len();
    let group = group_of(subject_id, keys, vec![None; n]);
    let mut taken: BTreeSet<String> = BTreeSet::new();
    for item in used.iter() {
        taken.insert(item.str()?.to_string());
    }
    let failure: RefCell<Option<PyErr>> = RefCell::new(None);
    let clean = |text: &str| -> String {
        match safe.filter(|s| !s.is_none()) {
            None => gr::sanitize_stem(text, 120),
            Some(f) => match f.call1((text,)).and_then(|v| v.extract::<String>()) {
                Ok(v) => v,
                Err(e) => {
                    failure.borrow_mut().get_or_insert(e);
                    String::new()
                }
            },
        }
    };
    let name = gr::output_name(&group, &mut taken, &clean);
    if let Some(e) = failure.into_inner() {
        return Err(e);
    }
    used.add(&name)?;
    Ok(name)
}

/// What a merged group records about instance ids (§7.4): note dicts.
#[pyfunction]
fn io_note_instance_ids<'py>(py: Python<'py>, subject_id: &str, occasions: usize) -> PyResult<Bound<'py, PyList>> {
    let group = group_of(subject_id, vec![String::new(); occasions], vec![None; occasions]);
    let mut log = scratch();
    gr::note_instance_ids(&group, &mut log);
    notes_fields(py, &log.notes)
}

// -- the 0.x reader and migration -------------------------------------------------------------

fn label_to_py<'py>(py: Python<'py>, label: &Option<lg::LegacyLabel>) -> PyResult<Bound<'py, PyAny>> {
    Ok(match label {
        None => py.None().into_bound(py),
        Some(lg::LegacyLabel::Int(i)) => i.into_pyobject(py)?.into_any(),
        Some(lg::LegacyLabel::Float(f)) => f.into_pyobject(py)?.into_any(),
        Some(lg::LegacyLabel::Bool(b)) => pyo3::types::PyBool::new(py, *b).to_owned().into_any(),
        Some(lg::LegacyLabel::Text(t)) => t.into_pyobject(py)?.into_any(),
    })
}

fn meta_fields<'py>(py: Python<'py>, meta: &lg::LegacyMeta) -> PyResult<Bound<'py, PyDict>> {
    let spatial = PyDict::new(py);
    spatial.set_item("spacing", &meta.spatial.spacing)?;
    spatial.set_item("origin", &meta.spatial.origin)?;
    spatial.set_item("direction", &meta.spatial.direction)?;
    spatial.set_item("axis_labels", &meta.spatial.axis_labels)?;
    spatial.set_item("coord_system", &meta.spatial.coord_system)?;
    let out = PyDict::new(py);
    out.set_item("spatial", spatial)?;
    out.set_item("shape", &meta.shape)?;
    out.set_item("image_names", &meta.image_names)?;
    out.set_item("seg_names", &meta.seg_names)?;
    out.set_item("label", label_to_py(py, &meta.label)?)?;
    out.set_item("label_name", &meta.label_name)?;
    out.set_item("patch_size", &meta.patch_size)?;
    out.set_item("extra", map_to_py(py, &meta.extra)?)?;
    out.set_item("schema_version", &meta.schema_version)?;
    Ok(out)
}

#[pyfunction]
fn legacy_is(py: Python<'_>, path: PathBuf) -> bool {
    crate::h5(py, || lg::is_legacy(&path))
}

/// A 0.x file's metadata, as `LegacyMeta` fields.
#[pyfunction]
fn legacy_read_meta(py: Python<'_>, path: PathBuf) -> R<Bound<'_, PyDict>> {
    let meta = py.detach(move || lg::read_meta(&path))?;
    Ok(meta_fields(py, &meta)?)
}

/// A whole 0.x file, as `LegacySample` fields (`meta` as `LegacyMeta` fields).
#[pyfunction]
fn legacy_read_sample(py: Python<'_>, path: PathBuf) -> R<Bound<'_, PyDict>> {
    let sample = py.detach(move || lg::read_sample(&path))?;
    let out = PyDict::new(py);
    let images = PyDict::new(py);
    for (name, array) in sample.images {
        images.set_item(name, nd_to_py(py, array))?;
    }
    out.set_item("images", images)?;
    let seg = PyDict::new(py);
    for (name, mask) in sample.seg {
        seg.set_item(name, array_to_py(py, mask))?;
    }
    out.set_item("seg", seg)?;
    out.set_item("bboxes", sample.bboxes.map(|b| array_to_py(py, b)))?;
    out.set_item(
        "bbox_scores",
        sample.bbox_scores.map(|s| array_to_py(py, ndarray::ArrayD::from_shape_vec(vec![s.len()], s).unwrap())),
    )?;
    out.set_item("bbox_labels", sample.bbox_labels)?;
    out.set_item("meta", meta_fields(py, &sample.meta)?)?;
    Ok(out)
}

fn paths_arg(paths: &Bound<'_, PyAny>) -> PyResult<Vec<PathBuf>> {
    paths.try_iter()?.map(|p| crate::convert::path(&p?)).collect()
}

/// One label set over a cohort: `(label_set, notes)`.
#[pyfunction]
fn legacy_build_label_set<'py>(py: Python<'py>, paths: &Bound<'py, PyAny>) -> R<(LabelSet, Bound<'py, PyList>)> {
    let owned = paths_arg(paths)?;
    let refs: Vec<&Path> = owned.iter().map(PathBuf::as_path).collect();
    let mut log = scratch();
    let label_set = crate::h5(py, || lg::build_label_set(&refs, Some(&mut log)))?;
    Ok((LabelSet::wrap(label_set), notes_fields(py, &log.notes)?))
}

fn label_set_opt(obj: Option<&Bound<'_, PyAny>>) -> R<Option<medh5::labels::LabelSet>> {
    obj.filter(|o| !o.is_none()).map(label_set_arg).transpose()
}

/// Migrate one 0.x file; the report's fields (`report` carried into it).
#[pyfunction]
#[pyo3(signature = (path, out, *, label_set=None, codec="balanced".to_string(), report=None))]
fn legacy_migrate<'py>(
    py: Python<'py>,
    path: PathBuf,
    out: PathBuf,
    label_set: Option<&Bound<'py, PyAny>>,
    codec: String,
    report: Option<&Bound<'py, PyAny>>,
) -> R<Bound<'py, PyDict>> {
    let labels = label_set_opt(label_set)?;
    let given = report.filter(|r| !r.is_none()).map(report_arg).transpose()?;
    let found = py.detach(move || lg::migrate(&path, &out, labels.as_ref(), &codec, given))?;
    Ok(report_fields(py, &found)?)
}

/// Migrate a cohort, minting one label set for all of it; the report's fields.
#[pyfunction]
#[pyo3(signature = (paths, outdir, *, group_by="study".to_string(), subject_key=None, label_set=None,
    codec="balanced".to_string()))]
fn legacy_migrate_paths<'py>(
    py: Python<'py>,
    paths: &Bound<'py, PyAny>,
    outdir: PathBuf,
    group_by: String,
    subject_key: Option<String>,
    label_set: Option<&Bound<'py, PyAny>>,
    codec: String,
) -> R<Bound<'py, PyDict>> {
    let owned = paths_arg(paths)?;
    let labels = label_set_opt(label_set)?;
    let found = py.detach(move || {
        let refs: Vec<&Path> = owned.iter().map(PathBuf::as_path).collect();
        lg::migrate_paths(&refs, &outdir, &group_by, subject_key.as_deref(), labels.as_ref(), &codec)
    })?;
    Ok(report_fields(py, &found)?)
}

#[pyfunction]
fn legacy_write_sidecar(label_set: &Bound<'_, PyAny>, path: PathBuf) -> R<String> {
    Ok(lg::write_sidecar(&label_set_arg(label_set)?, &path)?.to_string_lossy().into_owned())
}

#[pyfunction]
fn legacy_load_sidecar(path: PathBuf) -> R<LabelSet> {
    Ok(LabelSet::wrap(lg::load_sidecar(&path)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(io_report_ok, m)?,
        wrap_pyfunction!(io_report_json, m)?,
        wrap_pyfunction!(io_report_format, m)?,
        wrap_pyfunction!(io_note_line, m)?,
        wrap_pyfunction!(io_note_json, m)?,
        wrap_pyfunction!(io_sanitize_key, m)?,
        wrap_pyfunction!(io_sanitize_stem, m)?,
        wrap_pyfunction!(io_group_by_subject, m)?,
        wrap_pyfunction!(io_contradictions, m)?,
        wrap_pyfunction!(io_days_from_baseline, m)?,
        wrap_pyfunction!(io_output_name, m)?,
        wrap_pyfunction!(io_note_instance_ids, m)?,
        wrap_pyfunction!(legacy_is, m)?,
        wrap_pyfunction!(legacy_read_meta, m)?,
        wrap_pyfunction!(legacy_read_sample, m)?,
        wrap_pyfunction!(legacy_build_label_set, m)?,
        wrap_pyfunction!(legacy_migrate, m)?,
        wrap_pyfunction!(legacy_migrate_paths, m)?,
        wrap_pyfunction!(legacy_write_sidecar, m)?,
        wrap_pyfunction!(legacy_load_sidecar, m)?,
    ] {
        m.add_function(f)?;
    }
    let py = m.py();
    m.add("IO_SEVERITIES", PyTuple::new(py, rp::SEVERITIES)?)?;
    m.add("IO_FALLBACK_PREFIX", gr::FALLBACK_PREFIX)?;
    m.add("LEGACY_SCHEMA_VERSION", lg::SCHEMA_VERSION)?;
    m.add("LEGACY_BOX_SHIFT", lg::BOX_SHIFT)?;
    Ok(())
}
