//! `medh5.curation.agreement`, `.scrub` and `.splits`: the curation tools
//! that compare two annotations, sweep a file for identifiers, and audit split
//! claims across a cohort (spec §11.2, §11.4, §12.3).
//!
//! The result types stay the 1.x Python dataclasses, so they are built here
//! from engine values as plain fields, and their computed members (`value`,
//! `ok`, `format()`, `to_json()`, ...) come back through these functions, which
//! rebuild the engine value from the dataclass and ask it.

use std::path::PathBuf;

use indexmap::IndexMap;
use ndarray::{ArrayViewD, Dimension, IxDyn};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PyTuple};
use serde_json::Value;

use medh5::curation::agreement as ag;
use medh5::curation::scrub as sc;
use medh5::curation::splits as sp;
use medh5::labels::ClassKey;

use crate::annotations::annotation_handle;
use crate::convert::{
    asarray, class_key, class_keys, detach_if, f64_array, json_to_py, py_to_json, strings, DETACH_BYTES,
};
use crate::curation::{Agreement, SplitClaim};
use crate::errors::R;

fn tuple_of<'py, T: IntoPyObject<'py> + Clone>(py: Python<'py>, items: &[T]) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, items.iter().cloned())
}

/// Every `(key, value)` of a mapping argument.
fn items_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let mut out = Vec::new();
    for item in obj.call_method0("items")?.try_iter()? {
        out.push(item?.extract()?);
    }
    Ok(out)
}

fn opt_string(obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if obj.is_none() {
        Ok(None)
    } else {
        obj.extract().map(Some)
    }
}

// -- agreement ---------------------------------------------------------------------------

fn classes_arg(classes: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<ClassKey>>> {
    match classes {
        Some(c) if !c.is_none() => Ok(Some(class_keys(c)?)),
        _ => Ok(None),
    }
}

/// `map_classes` as Python gives it: `"id"`, `"key"`, or a mapping of
/// classes of `a` to classes of `b` (ids or keys, each resolved on its side).
fn class_map_arg(
    map: Option<&Bound<'_, PyAny>>,
    a: &medh5::annotations::Annotation,
    b: &medh5::annotations::Annotation,
) -> R<ag::ClassMap> {
    let Some(map) = map.filter(|m| !m.is_none()) else { return Ok(ag::ClassMap::Id) };
    if let Ok(name) = map.extract::<String>() {
        return match name.as_str() {
            "id" => Ok(ag::ClassMap::Id),
            "key" => Ok(ag::ClassMap::Key),
            _ => Err(PyValueError::new_err(format!(
                "map_classes is 'id', 'key' or a mapping of a's classes to b's; got {name:?}"
            ))
            .into()),
        };
    }
    let mut explicit = IndexMap::new();
    for (x, y) in items_of(map)? {
        explicit.insert(a.resolve_class(&class_key(&x)?)?, b.resolve_class(&class_key(&y)?)?);
    }
    Ok(ag::ClassMap::Explicit(explicit))
}

fn match_by_arg(match_by: &str) -> PyResult<ag::MatchBy> {
    match match_by {
        "auto" => Ok(ag::MatchBy::Auto),
        "iou" => Ok(ag::MatchBy::Iou),
        "instance_id" => Ok(ag::MatchBy::InstanceId),
        other => Err(PyValueError::new_err(format!("match_by is 'auto', 'iou' or 'instance_id'; got {other:?}"))),
    }
}

/// `{a's key: b's class id}` for the classes a comparison compared.
fn b_class_ids<'py>(py: Python<'py>, pairs: &IndexMap<String, (i64, i64)>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (key, (_, b)) in pairs {
        out.set_item(key, b)?;
    }
    Ok(out)
}

/// A `VoxelAgreement`'s fields, as its constructor takes them.
fn voxel_fields<'py>(py: Python<'py>, v: &ag::VoxelAgreement) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("metric", &v.metric)?;
    let per_class = PyDict::new(py);
    for (k, score) in &v.per_class {
        per_class.set_item(k, score)?;
    }
    out.set_item("per_class", per_class)?;
    out.set_item("skipped", tuple_of(py, &v.skipped)?)?;
    out.set_item("against", &v.against)?;
    let ids = PyDict::new(py);
    for (k, id) in &v.class_ids {
        ids.set_item(k, id)?;
    }
    out.set_item("class_ids", ids)?;
    Ok(out)
}

/// An `InstanceAgreement`'s fields, as its constructor takes them.
fn instance_fields<'py>(py: Python<'py>, v: &ag::InstanceAgreement) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("matched", tuple_of(py, &v.matched)?)?;
    out.set_item("only_in_a", tuple_of(py, &v.only_in_a)?)?;
    out.set_item("only_in_b", tuple_of(py, &v.only_in_b)?)?;
    out.set_item("matched_by", &v.matched_by)?;
    out.set_item("threshold", v.threshold)?;
    out.set_item("against", &v.against)?;
    out.set_item("class_mismatches", tuple_of(py, &v.class_mismatches)?)?;
    out.set_item("skipped", tuple_of(py, &v.skipped)?)?;
    Ok(out)
}

/// The engine value of a `VoxelAgreement` dataclass.
fn voxel_arg(obj: &Bound<'_, PyAny>) -> PyResult<ag::VoxelAgreement> {
    let mut per_class = IndexMap::new();
    for (k, v) in items_of(&obj.getattr("per_class")?)? {
        per_class.insert(k.str()?.to_string(), v.extract::<f64>()?);
    }
    let mut class_ids = IndexMap::new();
    for (k, v) in items_of(&obj.getattr("class_ids")?)? {
        class_ids.insert(k.str()?.to_string(), v.extract::<i64>()?);
    }
    Ok(ag::VoxelAgreement {
        metric: obj.getattr("metric")?.extract()?,
        per_class,
        skipped: strings(&obj.getattr("skipped")?)?,
        against: opt_string(&obj.getattr("against")?)?,
        class_ids,
    })
}

/// The engine value of an `InstanceAgreement` dataclass.
fn instance_arg(obj: &Bound<'_, PyAny>) -> PyResult<ag::InstanceAgreement> {
    Ok(ag::InstanceAgreement {
        matched: obj.getattr("matched")?.extract()?,
        only_in_a: obj.getattr("only_in_a")?.extract()?,
        only_in_b: obj.getattr("only_in_b")?.extract()?,
        matched_by: obj.getattr("matched_by")?.extract()?,
        threshold: obj.getattr("threshold")?.extract()?,
        against: opt_string(&obj.getattr("against")?)?,
        class_mismatches: obj.getattr("class_mismatches")?.extract()?,
        skipped: strings(&obj.getattr("skipped")?)?,
    })
}

/// Per-class Dice or IoU between two voxel annotations: the fields of a
/// `VoxelAgreement`.
#[pyfunction]
#[pyo3(signature = (a, b, *, metric="dice".to_string(), classes=None, map_classes=None))]
fn agreement_compare_voxel<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    metric: String,
    classes: Option<&Bound<'py, PyAny>>,
    map_classes: Option<&Bound<'py, PyAny>>,
) -> R<Bound<'py, PyDict>> {
    let (a, b) = (annotation_handle(a)?, annotation_handle(b)?);
    let mut options = ag::CompareOptions::default();
    options.classes = classes_arg(classes)?;
    options.map_classes = class_map_arg(map_classes, &a, &b)?;
    options.metric = Some(metric);
    let found = py.detach(move || ag::compare_voxel_with(&a, &b, &options))?;
    match found.comparison {
        ag::Comparison::Voxel(v) => {
            let fields = voxel_fields(py, &v)?;
            fields.set_item("b_class_ids", b_class_ids(py, &found.pairs)?)?;
            Ok(fields)
        }
        ag::Comparison::Instance(_) => unreachable!("compare_voxel_with compares voxels"),
    }
}

/// Object-level agreement between two annotations: the fields of an
/// `InstanceAgreement`.
#[pyfunction]
#[pyo3(signature = (a, b, *, threshold=ag::DEFAULT_IOU, classes=None, map_classes=None, match_by="auto"))]
fn agreement_compare_instances<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    threshold: f64,
    classes: Option<&Bound<'py, PyAny>>,
    map_classes: Option<&Bound<'py, PyAny>>,
    match_by: &str,
) -> R<Bound<'py, PyDict>> {
    let (a, b) = (annotation_handle(a)?, annotation_handle(b)?);
    let mut options = ag::CompareOptions::default();
    options.classes = classes_arg(classes)?;
    options.map_classes = class_map_arg(map_classes, &a, &b)?;
    options.match_by = match_by_arg(match_by)?;
    options.threshold = Some(threshold);
    let found = py.detach(move || ag::compare_instances_with(&a, &b, &options))?;
    match found.comparison {
        ag::Comparison::Instance(i) => {
            let fields = instance_fields(py, &i)?;
            fields.set_item("b_class_ids", b_class_ids(py, &found.pairs)?)?;
            Ok(fields)
        }
        ag::Comparison::Voxel(_) => unreachable!("compare_instances_with compares objects"),
    }
}

/// The comparison two annotations' kinds support:
/// `("voxel" | "instance", fields)`.
#[pyfunction]
#[pyo3(signature = (a, b, *, metric=None, threshold=None, classes=None, map_classes=None, match_by="auto"))]
#[allow(clippy::too_many_arguments)]
fn agreement_compare<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    metric: Option<String>,
    threshold: Option<f64>,
    classes: Option<&Bound<'py, PyAny>>,
    map_classes: Option<&Bound<'py, PyAny>>,
    match_by: &str,
) -> R<(&'static str, Bound<'py, PyDict>)> {
    let (a, b) = (annotation_handle(a)?, annotation_handle(b)?);
    let mut options = ag::CompareOptions::default();
    options.metric = metric;
    options.threshold = threshold;
    options.classes = classes_arg(classes)?;
    options.map_classes = class_map_arg(map_classes, &a, &b)?;
    options.match_by = match_by_arg(match_by)?;
    let found = py.detach(move || ag::compare_with(&a, &b, &options))?;
    let (kind, fields) = match &found.comparison {
        ag::Comparison::Voxel(v) => ("voxel", voxel_fields(py, v)?),
        ag::Comparison::Instance(i) => ("instance", instance_fields(py, i)?),
    };
    fields.set_item("b_class_ids", b_class_ids(py, &found.pairs)?)?;
    Ok((kind, fields))
}

#[pyfunction]
fn agreement_voxel_value(agreement: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    Ok(voxel_arg(agreement)?.value())
}

#[pyfunction]
#[pyo3(signature = (agreement, against=None))]
fn agreement_voxel_record(agreement: &Bound<'_, PyAny>, against: Option<&str>) -> R<Agreement> {
    let found = voxel_arg(agreement)?;
    Ok(Agreement::wrap(match against {
        Some(path) => found.to_record_against(path)?,
        None => found.to_record()?,
    }))
}

#[pyfunction]
fn agreement_voxel_json<'py>(py: Python<'py>, agreement: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &voxel_arg(agreement)?.to_json())
}

#[pyfunction]
fn agreement_instance_value(agreement: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    Ok(instance_arg(agreement)?.value())
}

#[pyfunction]
fn agreement_instance_mean_iou(agreement: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    Ok(instance_arg(agreement)?.mean_iou())
}

#[pyfunction]
#[pyo3(signature = (agreement, against=None))]
fn agreement_instance_record(agreement: &Bound<'_, PyAny>, against: Option<&str>) -> R<Agreement> {
    let found = instance_arg(agreement)?;
    Ok(Agreement::wrap(match against {
        Some(path) => found.to_record_against(path)?,
        None => found.to_record()?,
    }))
}

#[pyfunction]
fn agreement_instance_json<'py>(py: Python<'py>, agreement: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &instance_arg(agreement)?.to_json())
}

/// The shape two arrays broadcast to, by NumPy's rule.
fn broadcast_shape(a: &[usize], b: &[usize]) -> PyResult<Vec<usize>> {
    let n = a.len().max(b.len());
    let pad = |s: &[usize]| -> Vec<usize> { std::iter::repeat_n(1, n - s.len()).chain(s.iter().copied()).collect() };
    let (pa, pb) = (pad(a), pad(b));
    pa.iter()
        .zip(&pb)
        .map(|(&x, &y)| match (x, y) {
            _ if x == y => Ok(x),
            (1, _) => Ok(y),
            (_, 1) => Ok(x),
            _ => Err(PyValueError::new_err(format!(
                "operands could not be broadcast together with shapes {} {}",
                shape_text(a),
                shape_text(b)
            ))),
        })
        .collect()
}

fn shape_text(shape: &[usize]) -> String {
    match shape {
        [one] => format!("({one},)"),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",")),
    }
}

/// `count` of two masks as boolean arrays, broadcast to one shape as NumPy
/// would --- lazily, on views of the arrays --- and counted with the GIL
/// released once they reach [`DETACH_BYTES`].
fn count_pair(
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    count: fn(ArrayViewD<'_, bool>, ArrayViewD<'_, bool>) -> Option<f64>,
) -> PyResult<Option<f64>> {
    let py = a.py();
    let (a, b) = (asarray(a, Some("bool"))?, asarray(b, Some("bool"))?);
    let (a, b) = (a.cast::<PyArrayDyn<bool>>()?.readonly(), b.cast::<PyArrayDyn<bool>>()?.readonly());
    let (a, b) = (a.as_array(), b.as_array());
    let shape = IxDyn(&broadcast_shape(a.shape(), b.shape())?);
    let large = shape.size() >= DETACH_BYTES;
    match (a.broadcast(shape.clone()), b.broadcast(shape)) {
        (Some(x), Some(y)) => Ok(detach_if(py, large, || count(x, y))),
        _ => Err(PyValueError::new_err("the masks cannot be broadcast to one shape")),
    }
}

/// Sørensen--Dice, or `None` when both masks are empty.
#[pyfunction]
fn agreement_dice(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    count_pair(a, b, ag::dice_view)
}

/// Intersection over union, or `None` when both masks are empty.
#[pyfunction]
fn agreement_iou(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    count_pair(a, b, ag::iou_view)
}

fn box_arg(obj: &Bound<'_, PyAny>) -> PyResult<ndarray::Array2<f64>> {
    let array = f64_array(obj)?;
    let shape = array.shape().to_vec();
    match array.into_dimensionality::<ndarray::Ix2>() {
        Ok(m) if m.ncols() == 2 => Ok(m),
        _ => Err(PyValueError::new_err(format!("a box is an (S, 2) array, not shape {}", shape_text(&shape)))),
    }
}

/// IoU of two `(S, 2)` boxes in the same space.  Refused for boxes of
/// different numbers of axes or of none (`ValueError`), and for a bound that
/// is not finite or a `lo` above its `hi` (E406).
#[pyfunction]
fn agreement_box_iou(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> R<f64> {
    Ok(ag::box_iou_f64_checked(&box_arg(a)?, &box_arg(b)?)?)
}

// -- scrub -------------------------------------------------------------------------------

fn finding_fields<'py>(py: Python<'py>, f: &sc::Finding) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("rule", &f.rule)?;
    out.set_item("where", &f.r#where)?;
    out.set_item("detail", &f.detail)?;
    out.set_item("value", &f.value)?;
    out.set_item("actionable", f.actionable)?;
    out.set_item("fixable", f.fixable)?;
    Ok(out)
}

fn findings_fields<'py>(py: Python<'py>, found: &[sc::Finding]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for f in found {
        out.append(finding_fields(py, f)?)?;
    }
    Ok(out)
}

/// A `ScrubReport`'s fields, findings as field dicts.
fn report_fields<'py>(py: Python<'py>, r: &sc::ScrubReport) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("path", &r.path)?;
    out.set_item("profile", &r.profile)?;
    out.set_item("findings", findings_fields(py, &r.findings)?)?;
    out.set_item("actions", PyList::new(py, &r.actions)?)?;
    out.set_item("applied", r.applied)?;
    out.set_item("remaining", findings_fields(py, &r.remaining)?)?;
    let uid_map = PyDict::new(py);
    for (k, v) in &r.uid_map {
        uid_map.set_item(k, v)?;
    }
    out.set_item("uid_map", uid_map)?;
    out.set_item("not_checked", tuple_of(py, &r.not_checked)?)?;
    Ok(out)
}

fn finding_arg(obj: &Bound<'_, PyAny>) -> PyResult<sc::Finding> {
    Ok(sc::Finding {
        rule: obj.getattr("rule")?.extract()?,
        r#where: obj.getattr("where")?.extract()?,
        detail: obj.getattr("detail")?.extract()?,
        value: opt_string(&obj.getattr("value")?)?,
        actionable: obj.getattr("actionable")?.is_truthy()?,
        fixable: obj.getattr("fixable")?.is_truthy()?,
    })
}

fn findings_arg(obj: &Bound<'_, PyAny>) -> PyResult<Vec<sc::Finding>> {
    obj.try_iter()?.map(|f| finding_arg(&f?)).collect()
}

/// The engine value of a `ScrubReport` dataclass.
fn report_arg(obj: &Bound<'_, PyAny>) -> PyResult<sc::ScrubReport> {
    let mut uid_map = IndexMap::new();
    for (k, v) in items_of(&obj.getattr("uid_map")?)? {
        uid_map.insert(k.extract::<String>()?, v.extract::<String>()?);
    }
    Ok(sc::ScrubReport {
        path: obj.getattr("path")?.extract()?,
        profile: obj.getattr("profile")?.extract()?,
        findings: findings_arg(&obj.getattr("findings")?)?,
        actions: strings(&obj.getattr("actions")?)?,
        applied: obj.getattr("applied")?.is_truthy()?,
        remaining: findings_arg(&obj.getattr("remaining")?)?,
        uid_map,
        not_checked: strings(&obj.getattr("not_checked")?)?,
    })
}

/// Find identifiers in one file; changes nothing.  The report's fields.
#[pyfunction]
#[pyo3(signature = (path, *, profile="basic".to_string()))]
fn scrub_scan(py: Python<'_>, path: PathBuf, profile: String) -> R<Bound<'_, PyDict>> {
    let report = py.detach(move || sc::scan(&path, &profile))?;
    Ok(report_fields(py, &report)?)
}

/// The engine's options, from the Python API's keywords.  `date_shift_days`
/// and `date_shift_range` are two shifts, so giving both is refused.
#[allow(clippy::too_many_arguments)]
fn scrub_options(
    profile: String,
    salt: String,
    date_shift_days: Option<i64>,
    date_shift_range: Option<(i64, i64)>,
    performed_by: Option<String>,
    pseudonymise_ids: bool,
    withhold_shift: Option<bool>,
    out: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    attest_text: Option<String>,
    redact_text: bool,
    skip_refused: bool,
) -> R<sc::ScrubOptions> {
    let mut options =
        sc::ScrubOptions::default().with_profile(profile).with_salt(salt).with_pseudonymise_ids(pseudonymise_ids);
    options =
        match (date_shift_days, date_shift_range) {
            (Some(_), Some(_)) => return Err(medh5::Error::invalid(
                "--date-shift-days (date_shift_days) and --date-shift-range (date_shift_range) are two shifts for one \
                 sample: give one",
            )
            .into()),
            (Some(days), None) => options.with_date_shift(sc::DateShift::Days(days)),
            (None, Some((min, max))) => options.with_date_shift(sc::DateShift::Range(min, max)),
            (None, None) => options,
        };
    if let Some(by) = performed_by {
        options = options.with_performed_by(by);
    }
    if let Some(withhold) = withhold_shift {
        options = options.with_withhold_shift(withhold);
    }
    if let Some(out) = out {
        options = options.with_out(out);
    }
    if let Some(dir) = out_dir {
        options = options.with_out_dir(dir);
    }
    if let Some(tool) = attest_text {
        options = options.with_attest_text(tool);
    }
    Ok(options.with_redact_text(redact_text).with_skip_refused(skip_refused))
}

/// Refuse what these options cannot do, before any file is read
/// (`ScrubOptions::check`).
#[pyfunction]
#[pyo3(signature = (apply_changes, *, profile="basic".to_string(), salt="".to_string(), date_shift_days=None,
    date_shift_range=None, performed_by=None, pseudonymise_ids=false, withhold_shift=None, out=None, out_dir=None,
    attest_text=None, redact_text=false, skip_refused=false))]
#[allow(clippy::too_many_arguments)]
fn scrub_check(
    apply_changes: bool,
    profile: String,
    salt: String,
    date_shift_days: Option<i64>,
    date_shift_range: Option<(i64, i64)>,
    performed_by: Option<String>,
    pseudonymise_ids: bool,
    withhold_shift: Option<bool>,
    out: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    attest_text: Option<String>,
    redact_text: bool,
    skip_refused: bool,
) -> R<()> {
    let options = scrub_options(
        profile,
        salt,
        date_shift_days,
        date_shift_range,
        performed_by,
        pseudonymise_ids,
        withhold_shift,
        out,
        out_dir,
        attest_text,
        redact_text,
        skip_refused,
    )?;
    Ok(options.check(apply_changes)?)
}

/// Act on the actionable findings and write the §11.4 attestation.  The
/// report's fields.
#[pyfunction]
#[pyo3(signature = (path, *, profile="basic".to_string(), salt="".to_string(), date_shift_days=None,
    date_shift_range=None, performed_by=None, pseudonymise_ids=false, withhold_shift=None, out=None, out_dir=None,
    attest_text=None, redact_text=false, skip_refused=false))]
#[allow(clippy::too_many_arguments)]
fn scrub_apply(
    py: Python<'_>,
    path: PathBuf,
    profile: String,
    salt: String,
    date_shift_days: Option<i64>,
    date_shift_range: Option<(i64, i64)>,
    performed_by: Option<String>,
    pseudonymise_ids: bool,
    withhold_shift: Option<bool>,
    out: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    attest_text: Option<String>,
    redact_text: bool,
    skip_refused: bool,
) -> R<Bound<'_, PyDict>> {
    let options = scrub_options(
        profile,
        salt,
        date_shift_days,
        date_shift_range,
        performed_by,
        pseudonymise_ids,
        withhold_shift,
        out,
        out_dir,
        attest_text,
        redact_text,
        skip_refused,
    )?;
    let report = py.detach(move || sc::apply_with(&path, &options))?;
    Ok(report_fields(py, &report)?)
}

/// The engine document of a `SampleDocument`, or of anything with its JSON
/// form.
fn document_arg(obj: &Bound<'_, PyAny>) -> R<medh5::document::SampleDocument> {
    if let Ok(doc) = obj.cast::<crate::document::SampleDocument>() {
        return doc.get().current(obj.py());
    }
    let json = if obj.hasattr("to_json")? { obj.call_method0("to_json")? } else { obj.clone() };
    Ok(medh5::document::SampleDocument::from_json(&py_to_json(&json)?)?)
}

/// Every rule over one sample document, added to `report`: its fields after.
#[pyfunction]
fn scrub_scan_document<'py>(
    py: Python<'py>,
    document: &Bound<'py, PyAny>,
    report: &Bound<'py, PyAny>,
) -> R<Bound<'py, PyDict>> {
    let document = document_arg(document)?;
    let mut found = report_arg(report)?;
    sc::scan_document(&document, &mut found);
    Ok(report_fields(py, &found)?)
}

/// A finding as `ScrubReport.add` records it: the value previewed, and
/// actionable only where `--apply` may act.
#[pyfunction]
#[pyo3(signature = (rule, r#where, detail, value=None, *, actionable=false, fixable=true))]
fn scrub_finding<'py>(
    py: Python<'py>,
    rule: &str,
    r#where: &str,
    detail: &str,
    value: Option<String>,
    actionable: bool,
    fixable: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mut report = sc::ScrubReport::new("", "basic");
    report.add(rule, r#where, detail, value.as_deref(), actionable, fixable);
    finding_fields(py, &report.findings[0])
}

#[pyfunction]
fn scrub_finding_line(finding: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(finding_arg(finding)?.to_string())
}

#[pyfunction]
fn scrub_finding_json<'py>(py: Python<'py>, finding: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &finding_arg(finding)?.to_json())
}

#[pyfunction]
fn scrub_report_ok(report: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(report_arg(report)?.ok())
}

/// Positions of the findings on what the sample is *called*, in `remaining`
/// once applied and in `findings` before.
#[pyfunction]
fn scrub_report_open_identity(report: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let found = report_arg(report)?;
    let left = if found.applied { &found.remaining } else { &found.findings };
    Ok(found.open_identity().into_iter().filter_map(|f| left.iter().position(|g| std::ptr::eq(g, f))).collect())
}

#[pyfunction]
fn scrub_report_format(report: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(report_arg(report)?.format())
}

#[pyfunction]
fn scrub_report_json<'py>(py: Python<'py>, report: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &report_arg(report)?.to_json())
}

/// A stable pseudonym for a UID: same input, same output, everywhere.
#[pyfunction]
#[pyo3(signature = (uid, salt=""))]
fn scrub_pseudonymise(uid: &str, salt: &str) -> String {
    sc::pseudonymise(uid, salt)
}

// -- split audit -------------------------------------------------------------------------

fn membership_fields<'py>(py: Python<'py>, m: &sp::Membership) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("path", &m.path)?;
    out.set_item("sample_id", &m.sample_id)?;
    out.set_item("subject_id", &m.subject_id)?;
    out.set_item("group_id", &m.group_id)?;
    out.set_item("claim", Py::new(py, SplitClaim::wrap(m.claim.clone()))?)?;
    Ok(out)
}

fn leak_fields<'py>(py: Python<'py>, l: &sp::Leak) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("set_id", &l.set_id)?;
    out.set_item("group_id", &l.group_id)?;
    out.set_item("partitions", tuple_of(py, &l.partitions)?)?;
    out.set_item("paths", tuple_of(py, &l.paths)?)?;
    out.set_item("groups", tuple_of(py, &l.groups)?)?;
    out.set_item("subjects", tuple_of(py, &l.subjects)?)?;
    Ok(out)
}

fn conflict_fields<'py>(py: Python<'py>, c: &sp::Conflict) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("set_id", &c.set_id)?;
    out.set_item("manifests", tuple_of(py, &c.manifests)?)?;
    let by = PyDict::new(py);
    for (k, v) in &c.paths_by_manifest {
        by.set_item(k, tuple_of(py, v)?)?;
    }
    out.set_item("paths_by_manifest", by)?;
    Ok(out)
}

fn membership_arg(obj: &Bound<'_, PyAny>) -> R<sp::Membership> {
    Ok(sp::Membership {
        path: obj.getattr("path")?.extract()?,
        sample_id: obj.getattr("sample_id")?.extract()?,
        subject_id: obj.getattr("subject_id")?.extract()?,
        group_id: obj.getattr("group_id")?.extract()?,
        claim: SplitClaim::arg(&obj.getattr("claim")?)?,
    })
}

fn leak_arg(obj: &Bound<'_, PyAny>) -> PyResult<sp::Leak> {
    Ok(sp::Leak {
        set_id: obj.getattr("set_id")?.extract()?,
        group_id: obj.getattr("group_id")?.extract()?,
        partitions: strings(&obj.getattr("partitions")?)?,
        paths: strings(&obj.getattr("paths")?)?,
        groups: strings(&obj.getattr("groups")?)?,
        subjects: strings(&obj.getattr("subjects")?)?,
    })
}

fn conflict_arg(obj: &Bound<'_, PyAny>) -> PyResult<sp::Conflict> {
    let mut paths_by_manifest = std::collections::BTreeMap::new();
    for (k, v) in items_of(&obj.getattr("paths_by_manifest")?)? {
        paths_by_manifest.insert(k.extract::<String>()?, strings(&v)?);
    }
    Ok(sp::Conflict {
        set_id: obj.getattr("set_id")?.extract()?,
        manifests: strings(&obj.getattr("manifests")?)?,
        paths_by_manifest,
    })
}

/// The engine value of a `SplitAudit` dataclass.
fn audit_arg(obj: &Bound<'_, PyAny>) -> R<sp::SplitAudit> {
    let mut audit = sp::SplitAudit::default();
    for m in obj.getattr("memberships")?.try_iter()? {
        audit.memberships.push(membership_arg(&m?)?);
    }
    for c in obj.getattr("conflicts")?.try_iter()? {
        audit.conflicts.push(conflict_arg(&c?)?);
    }
    for l in obj.getattr("leaks")?.try_iter()? {
        audit.leaks.push(leak_arg(&l?)?);
    }
    audit.unclaimed = strings(&obj.getattr("unclaimed")?)?;
    audit.unreadable = obj.getattr("unreadable")?.extract()?;
    Ok(audit)
}

/// Read every file's claims and cross-check them (§12.3): the audit's fields.
#[pyfunction]
fn audit_splits<'py>(py: Python<'py>, paths: Vec<PathBuf>) -> PyResult<Bound<'py, PyDict>> {
    let audit = py.detach(move || sp::audit_splits(&paths));
    let out = PyDict::new(py);
    let memberships = PyList::empty(py);
    for m in &audit.memberships {
        memberships.append(membership_fields(py, m)?)?;
    }
    out.set_item("memberships", memberships)?;
    let conflicts = PyList::empty(py);
    for c in &audit.conflicts {
        conflicts.append(conflict_fields(py, c)?)?;
    }
    out.set_item("conflicts", conflicts)?;
    let leaks = PyList::empty(py);
    for l in &audit.leaks {
        leaks.append(leak_fields(py, l)?)?;
    }
    out.set_item("leaks", leaks)?;
    out.set_item("unclaimed", tuple_of(py, &audit.unclaimed)?)?;
    out.set_item("unreadable", tuple_of(py, &audit.unreadable)?)?;
    Ok(out)
}

/// Grouping key -> every grouping key sharing anatomy with it, sorted.
#[pyfunction]
fn audit_anatomy_units<'py>(py: Python<'py>, pairs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let mut owned: Vec<(String, String)> = Vec::new();
    for pair in pairs.try_iter()? {
        owned.push(pair?.extract()?);
    }
    let units = sp::anatomy_units(owned.iter().map(|(s, g)| (s.as_str(), g.as_str())));
    let out = PyDict::new(py);
    for (group, unit) in &units {
        out.set_item(group, tuple_of(py, unit)?)?;
    }
    Ok(out)
}

#[pyfunction]
fn audit_ok(audit: &Bound<'_, PyAny>) -> R<bool> {
    Ok(audit_arg(audit)?.ok())
}

#[pyfunction]
fn audit_set_ids<'py>(py: Python<'py>, audit: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
    Ok(tuple_of(py, &audit_arg(audit)?.set_ids())?)
}

/// `partition -> sample ids` for one split.
#[pyfunction]
fn audit_partitions<'py>(py: Python<'py>, audit: &Bound<'py, PyAny>, set_id: &str) -> R<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (partition, ids) in audit_arg(audit)?.partitions(set_id) {
        out.set_item(partition, tuple_of(py, &ids)?)?;
    }
    Ok(out)
}

#[pyfunction]
fn audit_counts<'py>(py: Python<'py>, audit: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let counts = serde_json::to_value(audit_arg(audit)?.counts()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(json_to_py(py, &counts)?)
}

#[pyfunction]
fn audit_json<'py>(py: Python<'py>, audit: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let json = audit_arg(audit)?.to_json();
    let out = json_to_py(py, &json)?;
    // `sets` is a tuple, as `set_ids` is.
    if let (Ok(dict), Some(Value::Array(items))) = (out.cast::<PyDict>(), json.get("sets")) {
        let names: Vec<String> = items.iter().filter_map(|v| v.as_str().map(str::to_string)).collect();
        dict.set_item("sets", tuple_of(py, &names)?)?;
    }
    Ok(out)
}

#[pyfunction]
fn audit_membership_json<'py>(py: Python<'py>, membership: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &membership_arg(membership)?.to_json())?)
}

#[pyfunction]
fn audit_leak_json<'py>(py: Python<'py>, leak: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &leak_arg(leak)?.to_json())
}

#[pyfunction]
fn audit_leak_line(leak: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(leak_arg(leak)?.to_string())
}

#[pyfunction]
fn audit_conflict_json<'py>(py: Python<'py>, conflict: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &conflict_arg(conflict)?.to_json())
}

#[pyfunction]
fn audit_conflict_line(conflict: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(conflict_arg(conflict)?.to_string())
}

fn frozenset_of<'py>(py: Python<'py>, items: &[&str]) -> PyResult<Bound<'py, PyFrozenSet>> {
    PyFrozenSet::new(py, items)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(agreement_compare_voxel, m)?,
        wrap_pyfunction!(agreement_compare_instances, m)?,
        wrap_pyfunction!(agreement_compare, m)?,
        wrap_pyfunction!(agreement_voxel_value, m)?,
        wrap_pyfunction!(agreement_voxel_record, m)?,
        wrap_pyfunction!(agreement_voxel_json, m)?,
        wrap_pyfunction!(agreement_instance_value, m)?,
        wrap_pyfunction!(agreement_instance_mean_iou, m)?,
        wrap_pyfunction!(agreement_instance_record, m)?,
        wrap_pyfunction!(agreement_instance_json, m)?,
        wrap_pyfunction!(agreement_dice, m)?,
        wrap_pyfunction!(agreement_iou, m)?,
        wrap_pyfunction!(agreement_box_iou, m)?,
        wrap_pyfunction!(scrub_scan, m)?,
        wrap_pyfunction!(scrub_apply, m)?,
        wrap_pyfunction!(scrub_check, m)?,
        wrap_pyfunction!(scrub_scan_document, m)?,
        wrap_pyfunction!(scrub_finding, m)?,
        wrap_pyfunction!(scrub_finding_line, m)?,
        wrap_pyfunction!(scrub_finding_json, m)?,
        wrap_pyfunction!(scrub_report_ok, m)?,
        wrap_pyfunction!(scrub_report_open_identity, m)?,
        wrap_pyfunction!(scrub_report_format, m)?,
        wrap_pyfunction!(scrub_report_json, m)?,
        wrap_pyfunction!(scrub_pseudonymise, m)?,
        wrap_pyfunction!(audit_splits, m)?,
        wrap_pyfunction!(audit_anatomy_units, m)?,
        wrap_pyfunction!(audit_ok, m)?,
        wrap_pyfunction!(audit_set_ids, m)?,
        wrap_pyfunction!(audit_partitions, m)?,
        wrap_pyfunction!(audit_counts, m)?,
        wrap_pyfunction!(audit_json, m)?,
        wrap_pyfunction!(audit_membership_json, m)?,
        wrap_pyfunction!(audit_leak_json, m)?,
        wrap_pyfunction!(audit_leak_line, m)?,
        wrap_pyfunction!(audit_conflict_json, m)?,
        wrap_pyfunction!(audit_conflict_line, m)?,
    ] {
        m.add_function(f)?;
    }
    let py = m.py();
    m.add("AGREEMENT_DEFAULT_IOU", ag::DEFAULT_IOU)?;
    m.add("AGREEMENT_OBJECT_KINDS", tuple_of(py, &ag::OBJECT_KINDS)?)?;
    m.add("SCRUB_PROFILES", tuple_of(py, &sc::PROFILES)?)?;
    m.add("SCRUB_IDENTIFYING_KEYS", frozenset_of(py, &sc::IDENTIFYING_KEYS)?)?;
    m.add("SCRUB_QUASI_IDENTIFYING_KEYS", frozenset_of(py, &sc::QUASI_IDENTIFYING_KEYS)?)?;
    m.add("SCRUB_DATE_KEYS", frozenset_of(py, &sc::DATE_KEYS)?)?;
    m.add("SCRUB_UID_KEYS", frozenset_of(py, &sc::UID_KEYS)?)?;
    m.add("SCRUB_MAX_DEPTH", sc::MAX_DEPTH)?;
    m.add("SCRUB_PSEUDONYM_PREFIX", sc::PSEUDONYM_PREFIX)?;
    m.add("SCRUB_PATH_REMOVED", sc::PATH_REMOVED)?;
    m.add("SCRUB_AGE_LIMIT", sc::AGE_LIMIT)?;
    m.add("SCRUB_INTERNAL_REFERENCES", tuple_of(py, &sc::INTERNAL_REFERENCES)?)?;
    m.add("SCRUB_FREE_TEXT", sc::FREE_TEXT)?;
    m.add("SCRUB_UNFIXABLE_LOCATIONS", tuple_of(py, &sc::UNFIXABLE_LOCATIONS)?)?;
    m.add("SCRUB_STRICT_RULES", tuple_of(py, sc::strict_rules())?)?;
    m.add("SCRUB_IDENTITY_RULES", tuple_of(py, &sc::IDENTITY_RULES)?)?;
    m.add("SCRUB_NOT_CHECKED", tuple_of(py, &sc::ScrubReport::new("", "basic").not_checked)?)?;
    Ok(())
}
