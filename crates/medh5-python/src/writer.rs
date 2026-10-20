//! `medh5.sample`: `SampleWriter`, `create` and `amend` (spec §14.4).
//!
//! Every method takes the 1.x keyword arguments and hands the engine's
//! `SampleWriter` the normalised values; every rule --- ids, geometry,
//! encodings, coverage, the validator gate in `commit` --- is the engine's.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{RwLockReadGuard, RwLockWriteGuard};

use numpy::{PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyString, PyTuple};
use serde_json::Value;

use medh5::annotations::encode::InstanceInput;
use medh5::annotations::encode_geometric::{Assertions, Polygon};
use medh5::annotations::select::OverlapStats;
use medh5::annotations::LabelValue;
use medh5::labels::ClassKey;
use medh5::sample::writer::{
    Annotated, GridOptions, ImageEdit, ImageOptions, QualityArg, SampleWriter as EngineWriter,
};
use medh5::sample::writer_annotations::{
    AnnotationOptions, FieldDtype, LabelmapSource, ObjectFields, Placement, SegmentationOptions, SegmentationSource,
    TransformSpec,
};

use crate::convert::{
    asarray, bool_array, class_key, class_keys, detach_if, f64_array, f64_vec, i64_array, i64_vec, kwargs_to_map,
    matrix as to_matrix, opt_strings, py_to_json, py_to_nd, strings, DETACH_ROWS,
};
use crate::curation::{Activity, Agent, Cohort, Deidentification, Identity, QualityRecord, SplitClaim, Timepoint};
use crate::errors::R;
use crate::exclusive::Exclusive;
use crate::geometry::Grid;
use crate::h5;
use crate::nodes::{Dataset, Group};

// -- arguments ----------------------------------------------------------------------------

fn is_given<'py>(obj: Option<&Bound<'py, PyAny>>) -> Option<Bound<'py, PyAny>> {
    obj.filter(|o| !o.is_none()).cloned()
}

/// An `Activity` (its id) or an activity id.
fn prov_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    let Some(o) = is_given(obj) else { return Ok(None) };
    if let Ok(s) = o.cast::<PyString>() {
        return Ok(Some(s.to_string()));
    }
    if let Ok(a) = o.cast::<Activity>() {
        return Ok(Some(a.get().inner.id.clone()));
    }
    Ok(Some(o.getattr("id")?.extract::<String>()?))
}

/// An `Agent` (its id) or an agent id.
fn agent_ref(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    let Some(o) = is_given(obj) else { return Ok(None) };
    if let Ok(s) = o.cast::<PyString>() {
        return Ok(Some(s.to_string()));
    }
    if let Ok(a) = o.cast::<Agent>() {
        return Ok(Some(a.get().inner.id.clone()));
    }
    Ok(Some(o.getattr("id")?.extract::<String>()?))
}

/// A quality reference: a record key, or the fields of a record to create.
fn quality_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<QualityArg>> {
    let Some(o) = is_given(obj) else { return Ok(None) };
    if let Ok(s) = o.cast::<PyString>() {
        return Ok(Some(QualityArg::Key(s.to_string())));
    }
    match py_to_json(&o)? {
        Value::Object(map) => Ok(Some(QualityArg::Record(map))),
        _ => Err(PyTypeError::new_err(format!(
            "quality must be a record key (str) or a mapping of record fields, not {}",
            o.get_type().name()?
        ))),
    }
}

/// `"all_given"`, `"all"`, or the classes looked for (§11.3).
pub(crate) fn annotated_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Annotated> {
    let Some(o) = is_given(obj) else { return Ok(Annotated::AllGiven) };
    if let Ok(s) = o.cast::<PyString>() {
        return Ok(match s.to_str()? {
            "all_given" => Annotated::AllGiven,
            "all" => Annotated::All,
            other => Annotated::Classes(vec![ClassKey::Key(other.to_string())]),
        });
    }
    Ok(Annotated::Classes(class_keys(&o)?))
}

#[allow(clippy::too_many_arguments)]
fn common(
    annotated_classes: Option<&Bound<'_, PyAny>>,
    closure: &str,
    timepoints: Option<&Bound<'_, PyAny>>,
    prov: Option<&Bound<'_, PyAny>>,
    quality: Option<&Bound<'_, PyAny>>,
    derived_from: Option<&Bound<'_, PyAny>>,
    task: Option<&str>,
    codec: Option<&str>,
) -> PyResult<AnnotationOptions> {
    Ok(AnnotationOptions {
        annotated_classes: annotated_arg(annotated_classes)?,
        closure: Some(closure.to_string()),
        timepoints: opt_strings(timepoints)?,
        prov: prov_arg(prov)?,
        quality: quality_arg(quality)?,
        derived_from: opt_strings(derived_from)?.unwrap_or_default(),
        task: task.map(str::to_string),
        codec: codec.map(str::to_string),
    })
}

fn u64s(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    let items = if obj.hasattr("tolist")? { obj.call_method0("tolist")? } else { obj.clone() };
    if let Ok(single) = items.extract::<u64>() {
        return Ok(vec![single]);
    }
    items.extract::<Vec<u64>>()
}

fn objects(
    instance_ids: Option<&Bound<'_, PyAny>>,
    scores: Option<&Bound<'_, PyAny>>,
    attributes: Option<&Bound<'_, PyAny>>,
) -> PyResult<ObjectFields> {
    let attributes = match is_given(attributes) {
        None => None,
        Some(items) => {
            let mut out = Vec::new();
            for item in items.try_iter()? {
                let item = item?;
                match py_to_json(&item)? {
                    Value::Object(m) => out.push(m),
                    _ => {
                        return Err(PyTypeError::new_err(format!(
                            "each entry of attributes is a mapping, not {}",
                            item.get_type().name()?
                        )))
                    }
                }
            }
            Some(out)
        }
    };
    Ok(ObjectFields {
        instance_ids: is_given(instance_ids).map(|o| u64s(&o)).transpose()?,
        scores: is_given(scores).map(|o| f64_vec(&o)).transpose()?,
        attributes,
    })
}

fn placement(grid: Option<&str>, space: Option<&str>, frame_uid: Option<&str>) -> Placement {
    Placement {
        grid: grid.map(str::to_string),
        space: space.map(str::to_string),
        frame_uid: frame_uid.map(str::to_string),
    }
}

fn opt_keys(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<ClassKey>>> {
    is_given(obj).map(|o| class_keys(&o)).transpose()
}

/// `{class: array}` items, in the mapping's order.
/// `label_values=`: each value's class, or classes (an nnU-Net region is a
/// union of values, so a value is in every region holding it).
fn label_values_arg(obj: &Bound<'_, PyAny>) -> PyResult<Vec<(i64, Vec<ClassKey>)>> {
    let mut out = Vec::new();
    for (value, classes) in items_of(obj)? {
        let classes = match class_key(&classes) {
            Ok(key) => vec![key],
            Err(_) => class_keys(&classes)?,
        };
        out.push((value.extract::<i64>()?, classes));
    }
    Ok(out)
}

/// `add_segmentation(labelmap=...)`, the volume borrowed from NumPy for the
/// whole write, which runs with the GIL released.
#[allow(clippy::too_many_arguments)]
fn add_labelmap(
    writer: &SampleWriter,
    ann_id: &str,
    grid: &str,
    labelmap: &Bound<'_, PyAny>,
    values: Option<Vec<(i64, Vec<ClassKey>)>>,
    ignore_value: Option<i64>,
    options: SegmentationOptions,
) -> R<(String, Option<OverlapStats>)> {
    fn write<T: numpy::Element + LabelValue>(
        writer: &SampleWriter,
        ann_id: &str,
        grid: &str,
        array: &Bound<'_, PyUntypedArray>,
        values: Option<Vec<(i64, Vec<ClassKey>)>>,
        ignore_value: Option<i64>,
        options: SegmentationOptions,
    ) -> R<(String, Option<OverlapStats>)> {
        let typed = array.cast::<PyArrayDyn<T>>().map_err(PyErr::from)?;
        let guard = typed.readonly();
        let view = guard.as_array();
        let mut held = writer.writer()?;
        let writer = &mut *held;
        Ok(array.py().detach(move || {
            let mut source = LabelmapSource::new(view);
            if let Some(values) = values {
                source = source.values(values);
            }
            if let Some(value) = ignore_value {
                source = source.ignore_value(value);
            }
            writer.add_segmentation_labelmap(ann_id, grid, source, options)
        })?)
    }
    let array = asarray(labelmap, None)?;
    let dtype = array.dtype();
    let (a, v, i) = (&array, values, ignore_value);
    match (dtype.kind() as char, dtype.itemsize()) {
        ('u', 1) => write::<u8>(writer, ann_id, grid, a, v, i, options),
        ('u', 2) => write::<u16>(writer, ann_id, grid, a, v, i, options),
        ('u', 4) => write::<u32>(writer, ann_id, grid, a, v, i, options),
        ('u', 8) => write::<u64>(writer, ann_id, grid, a, v, i, options),
        ('i', 1) => write::<i8>(writer, ann_id, grid, a, v, i, options),
        ('i', 2) => write::<i16>(writer, ann_id, grid, a, v, i, options),
        ('i', 4) => write::<i32>(writer, ann_id, grid, a, v, i, options),
        ('i', 8) => write::<i64>(writer, ann_id, grid, a, v, i, options),
        _ => Err(PyTypeError::new_err(format!(
            "labelmap= is an integer volume, not of dtype {}; pass masks= for boolean masks",
            dtype.str()?
        ))
        .into()),
    }
}

fn items_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let mut out = Vec::new();
    for item in obj.call_method0("items")?.try_iter()? {
        out.push(item?.extract()?);
    }
    Ok(out)
}

/// A boolean volume: a bool or integer array as NumPy reads it, non-zero
/// set; a floating-point one only when every value is 0 or 1.  Read as NumPy
/// reads it, a soft value --- 0.2 --- was set: a threshold nobody chose,
/// stored as which voxels are in the mask (E411, checked before the cast as
/// keypoint visibility is).
fn mask_arg(obj: &Bound<'_, PyAny>, what: &str) -> R<ndarray::ArrayD<bool>> {
    let array = crate::convert::asarray(obj, None)?;
    let kind: String = array.getattr("dtype")?.getattr("kind")?.extract()?;
    if kind == "f" {
        if let Some(bad) = f64_array(array.as_any())?.iter().find(|v| **v != 0.0 && **v != 1.0) {
            return Err(medh5::Error::coded(
                "E411",
                format!(
                    "{what} is floating point and holds {}, which is neither 0 nor 1: a mask is boolean, and reading \
                     every non-zero value as set is a threshold nobody chose --- threshold it first (`mask >= 0.5`), \
                     or pass a bool array",
                    medh5::json::py_float(*bad)
                ),
            )
            .into());
        }
    }
    Ok(bool_array(array.as_any())?)
}

/// `class_id` resolves a class key against the writer's label set, taking
/// the writer's lock only for the lookup, never while the items are read.
/// Instances as given, each with its class key: the keys are resolved under
/// the lock the write holds, so a label set changed by another thread cannot
/// land between two of them (E4).
fn instance_inputs(obj: &Bound<'_, PyAny>) -> R<Vec<(ClassKey, InstanceInput)>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        let item = item?;
        let get = |name: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
            Ok(match item.getattr(name) {
                Ok(v) if !v.is_none() => Some(v),
                _ => None,
            })
        };
        let class = get("class_id")?.ok_or_else(|| PyTypeError::new_err("an instance needs class_id"))?;
        let instance_id = get("instance_id")?.ok_or_else(|| PyTypeError::new_err("an instance needs instance_id"))?;
        out.push((
            class_key(&class)?,
            InstanceInput {
                class_id: 0,
                instance_id: instance_id.extract::<u64>()?,
                mask: get("mask")?.map(|m| mask_arg(&m, "an instance's mask")).transpose()?,
                bbox: get("box")?.map(|b| f64_vec(&b)).transpose()?,
                crop: get("crop")?.map(|c| mask_arg(&c, "an instance's crop")).transpose()?,
                score: get("score")?.map(|s| s.extract::<f64>()).transpose()?,
            },
        ));
    }
    Ok(out)
}

/// A polygon as given, its class not yet resolved (as `instance_inputs`).
struct PolygonArg {
    class: ClassKey,
    vertices: numpy::ndarray::ArrayD<f64>,
    plane: (i64, i64),
    role: String,
}

impl PolygonArg {
    fn resolve(self, class_id: impl Fn(&ClassKey) -> medh5::Result<i64>) -> R<Polygon> {
        Ok(Polygon::new(self.vertices, class_id(&self.class)?, self.plane, &self.role)?)
    }
}

fn polygons_of(obj: &Bound<'_, PyAny>) -> R<Vec<PolygonArg>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        let item = item?;
        let vertices = f64_array(&item.getattr("vertices")?)?;
        let class = class_key(&item.getattr("class_id")?)?;
        let plane: (i64, i64) = match item.getattr("plane") {
            Ok(p) if !p.is_none() => {
                let v = i64_vec(&p)?;
                if v.len() != 2 {
                    return Err(medh5::Error::coded("E405", "a polygon's plane is (axis, index)").into());
                }
                (v[0], v[1])
            }
            _ => (-1, 0),
        };
        let role: String = match item.getattr("role") {
            Ok(r) if !r.is_none() => r.extract()?,
            _ => "outer".into(),
        };
        out.push(PolygonArg { class, vertices, plane, role });
    }
    Ok(out)
}

/// An open curve as given, its class not yet resolved (as `PolygonArg`).
struct PolylineArg {
    class: ClassKey,
    vertices: numpy::ndarray::ArrayD<f64>,
    plane: Option<(i64, i64)>,
}

/// Open curves (1.2 §3): each item's `vertices`, `class_id` (an id or a
/// label-set key) and optional `plane` `(axis, index)`; `instance_ids`, one
/// per curve, when given.
fn polylines_of(obj: &Bound<'_, PyAny>) -> R<Vec<PolylineArg>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        let item = item?;
        let vertices = f64_array(&item.getattr("vertices")?)?;
        let class = class_key(&item.getattr("class_id")?)?;
        let mut plane = None;
        if let Ok(p) = item.getattr("plane") {
            if !p.is_none() {
                let v = i64_vec(&p)?;
                if v.len() != 2 {
                    return Err(medh5::Error::coded("E405", "a polyline's plane is (axis, index)").into());
                }
                plane = Some((v[0], v[1]));
            }
        }
        out.push(PolylineArg { class, vertices, plane });
    }
    Ok(out)
}

fn polylines(
    writer: &EngineWriter,
    given: Vec<PolylineArg>,
    instance_ids: Option<Vec<u64>>,
) -> R<Vec<medh5::annotations::Polyline>> {
    let mut out = Vec::new();
    for arg in given {
        let mut line = medh5::annotations::Polyline::new(arg.vertices, writer.class_id(&arg.class)?);
        if let Some((axis, index)) = arg.plane {
            line = line.plane(axis, index);
        }
        out.push(line);
    }
    if let Some(ids) = instance_ids {
        if ids.len() != out.len() {
            return Err(medh5::Error::coded(
                "E405",
                format!("instance_ids has {} entries for {} polylines; give one for each", ids.len(), out.len()),
            )
            .into());
        }
        out = out.into_iter().zip(ids).map(|(line, id)| line.instance_id(id)).collect();
    }
    Ok(out)
}

/// Classification rows as columns (§9): a mapping `class -> value`, or rows
/// `(class, value[, scope_id[, scheme, scheme_value]])`.
pub struct Rows<'py> {
    pub classes: Vec<Bound<'py, PyAny>>,
    pub values: Vec<f64>,
    pub scope_ids: Option<Vec<i64>>,
    pub schemes: Option<Vec<String>>,
    pub scheme_values: Option<Vec<String>>,
}

pub fn assertion_rows<'py>(labels: &Bound<'py, PyAny>) -> R<Rows<'py>> {
    let float = |v: &Bound<'py, PyAny>| -> PyResult<f64> { v.call_method0("__float__")?.extract::<f64>() };
    if labels.hasattr("keys")? && labels.hasattr("items")? {
        let mut classes = Vec::new();
        let mut values = Vec::new();
        for (k, v) in items_of(labels)? {
            classes.push(k);
            values.push(float(&v)?);
        }
        return Ok(Rows { classes, values, scope_ids: None, schemes: None, scheme_values: None });
    }
    let mut rows =
        Rows { classes: Vec::new(), values: Vec::new(), scope_ids: None, schemes: None, scheme_values: None };
    let mut units = Vec::new();
    let mut schemes = Vec::new();
    let mut scheme_values = Vec::new();
    let mut widths = std::collections::BTreeSet::new();
    for (index, row) in labels.try_iter()?.enumerate() {
        let row = row?;
        let is_sequence = !row.is_instance_of::<PyString>()
            && !row.is_instance_of::<pyo3::types::PyBytes>()
            && (row.is_instance_of::<pyo3::types::PyList>() || row.is_instance_of::<PyTuple>());
        if !is_sequence {
            return Err(medh5::Error::coded(
                "E405",
                format!(
                    "classification row {index} is {}; a row is (class, value), (class, value, scope_id) or (class, \
                     value, scope_id, scheme, scheme_value)",
                    row.repr()?
                ),
            )
            .into());
        }
        let fields: Vec<Bound<'py, PyAny>> = row.try_iter()?.collect::<PyResult<_>>()?;
        if ![2, 3, 5].contains(&fields.len()) {
            return Err(medh5::Error::coded(
                "E405",
                format!(
                    "classification row {index} has {} fields; a row is (class, value), (class, value, scope_id) or \
                     (class, value, scope_id, scheme, scheme_value)",
                    fields.len()
                ),
            )
            .into());
        }
        widths.insert(fields.len());
        rows.classes.push(fields[0].clone());
        rows.values.push(float(&fields[1])?);
        if fields.len() >= 3 {
            units.push(fields[2].extract::<i64>()?);
        }
        if fields.len() == 5 {
            schemes.push(fields[3].str()?.to_string());
            scheme_values.push(fields[4].str()?.to_string());
        }
    }
    if widths.len() > 1 {
        let listed: Vec<String> = widths.iter().map(|w| w.to_string()).collect();
        return Err(medh5::Error::coded(
            "E405",
            format!(
                "classification rows mix widths [{}]; every row supplies the same columns, so no assertion is left \
                 without a scope unit",
                listed.join(", ")
            ),
        )
        .into());
    }
    let width = widths.into_iter().next().unwrap_or(2);
    if width >= 3 {
        rows.scope_ids = Some(units);
    }
    if width == 5 {
        rows.schemes = Some(schemes);
        rows.scheme_values = Some(scheme_values);
    }
    Ok(rows)
}

/// Merge row columns with the keyword columns; a column given twice is refused.
pub fn assertion_columns(
    rows: Rows<'_>,
    class_ids: Vec<i64>,
    scope_ids: Option<&Bound<'_, PyAny>>,
    schemes: Option<&Bound<'_, PyAny>>,
    scheme_values: Option<&Bound<'_, PyAny>>,
) -> R<Assertions> {
    let given_scope = is_given(scope_ids).map(|o| i64_vec(&o)).transpose()?;
    let given_schemes = opt_strings(schemes)?;
    let given_values = opt_strings(scheme_values)?;
    for (name, from_rows, from_argument) in [
        ("scope_ids", rows.scope_ids.is_some(), given_scope.is_some()),
        ("schemes", rows.schemes.is_some(), given_schemes.is_some()),
        ("scheme_values", rows.scheme_values.is_some(), given_values.is_some()),
    ] {
        if from_rows && from_argument {
            return Err(medh5::Error::coded(
                "E405",
                format!("{name} is given both in the rows and as an argument; give it once"),
            )
            .into());
        }
    }
    Ok(Assertions {
        class_ids,
        values: rows.values,
        scope_ids: rows.scope_ids.or(given_scope),
        schemes: rows.schemes.or(given_schemes),
        scheme_values: rows.scheme_values.or(given_values),
    })
}

// -- the writer ------------------------------------------------------------------------------

/// Builder for one sample.  Every `add_*` validates immediately; `commit`
/// validates the whole and atomically replaces the target (§14.4).  Used by
/// one thread at a time (E4): a call while another thread's runs is refused.
#[pyclass(module = "medh5.sample", name = "SampleWriter", subclass, frozen)]
pub struct SampleWriter {
    inner: Exclusive<EngineWriter>,
}

impl SampleWriter {
    fn of(inner: EngineWriter) -> Self {
        SampleWriter { inner: Exclusive::new("a SampleWriter", inner) }
    }

    /// `f` of the writer's document, as it stands.
    pub(crate) fn read_document<T>(&self, f: impl FnOnce(&medh5::document::SampleDocument) -> T) -> R<T> {
        Ok(f(self.inner.read()?.document()))
    }

    /// `f` of the writer's document, to change, while it can still be
    /// written to.
    pub(crate) fn edit_document<T>(&self, f: impl FnOnce(&mut medh5::document::SampleDocument) -> T) -> R<T> {
        Ok(f(self.writer()?.document_mut()))
    }

    /// The engine writer, to read.
    fn state(&self) -> R<RwLockReadGuard<'_, EngineWriter>> {
        self.inner.read()
    }

    /// The writer's document, as `document` hands it out: a record the
    /// writer returns is read from it, so an edit to one of the record's
    /// dicts reaches the file.
    fn document_view<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(Bound::new(slf.py(), crate::document::SampleDocument::of_writer(slf.clone().unbind()))?.into_any())
    }

    /// The engine writer, alone, while it can still be written to.
    fn writer(&self) -> R<RwLockWriteGuard<'_, EngineWriter>> {
        let held = self.inner.write()?;
        if held.is_closed() {
            return Err(medh5::Error::invalid("this writer has already committed or aborted").into());
        }
        Ok(held)
    }
}

#[pymethods]
impl SampleWriter {
    #[new]
    #[pyo3(signature = (path, *, sample_id=None, subject_id=None, codec="balanced", profiles=None))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        sample_id: Option<String>,
        subject_id: Option<String>,
        codec: &str,
        profiles: Option<&Bound<'_, PyAny>>,
    ) -> R<Self> {
        let profiles = opt_strings(profiles)?.unwrap_or_default();
        if sample_id.is_none() {
            let ids = medh5::sample::writer::default_ids(&path)?;
            if ids.current != ids.stem {
                let (stem, current) = (medh5::json::repr_str(&ids.stem), medh5::json::repr_str(&ids.current));
                crate::deprecation::warn(
                    py,
                    &format!(
                        "medh5 3.0 names this sample {stem} (the file stem), not {current}: pass sample_id= (and \
                         subject_id=) to keep {current}"
                    ),
                    true,
                )?;
            }
        }
        let codec = codec.to_string();
        let inner = py.detach(move || {
            medh5::sample::writer::create(&path, sample_id.as_deref(), subject_id.as_deref(), &codec, &profiles)
        })?;
        Ok(SampleWriter::of(inner))
    }

    // -- lifecycle -------------------------------------------------------------------

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (exc_type=None, _exc=None, _tb=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> R<bool> {
        let mut held = self.inner.write()?;
        let inner = &mut *held;
        let aborted = is_given(exc_type).is_some();
        h5(py, move || {
            if aborted {
                inner.abort();
            } else if !inner.is_closed() {
                if let Err(e) = inner.commit(true) {
                    // The block is over, so no one is left to mend and commit
                    // again: a refused commit aborts, and the temporary
                    // sibling goes.
                    inner.abort();
                    return Err(e);
                }
            }
            Ok(())
        })?;
        Ok(false)
    }

    /// Discard the in-progress file, leaving any existing one untouched.
    fn abort(&self, py: Python<'_>) -> R<()> {
        let mut held = self.inner.write()?;
        let inner = &mut *held;
        h5(py, move || inner.abort());
        Ok(())
    }

    /// The target path.
    #[getter]
    fn path(&self) -> R<String> {
        Ok(self.state()?.path.to_string_lossy().into_owned())
    }

    /// The codec profile datasets default to.
    #[getter]
    fn codec(&self) -> R<String> {
        Ok(self.state()?.codec.clone())
    }

    /// Whether the writer has committed or been aborted.  A commit refused
    /// before it changed anything leaves it open.
    #[getter]
    fn closed(&self) -> R<bool> {
        Ok(self.state()?.is_closed())
    }

    /// The sample document being built.  Live: assigning one of its fields
    /// edits the writer's document, and `commit` validates the result.
    #[getter]
    fn document(slf: &Bound<'_, Self>) -> crate::document::SampleDocument {
        crate::document::SampleDocument::of_writer(slf.clone().unbind())
    }

    #[setter]
    fn set_document(slf: &Bound<'_, Self>, document: &Bound<'_, PyAny>) -> R<()> {
        let Ok(doc) = document.cast::<crate::document::SampleDocument>() else {
            return Err(medh5::Error::invalid(format!(
                "document must be a SampleDocument, not {}",
                document.get_type().name()?
            ))
            .into());
        };
        let doc = doc.get();
        if doc.is_view_of(slf.as_ptr()) {
            // `w.document = w.document`: already the writer's own.
            return Ok(());
        }
        let value = doc.current(slf.py())?;
        slf.get().writer()?.set_document(value);
        Ok(())
    }

    /// The file being built, for tools that rewrite what the builder does not
    /// model.  `commit` stamps what is written through it, and refuses
    /// (E701) to restamp a dataset the amended file already failed.
    #[getter]
    fn handle(&self, py: Python<'_>) -> R<Group> {
        h5(py, || Ok(Group::wrap(self.state()?.root()?)))
    }

    // -- document ------------------------------------------------------------------

    /// Merge fields into the identity (`sample_id`, `subject_id`, ...).
    #[pyo3(signature = (**fields))]
    fn identity(slf: &Bound<'_, Self>, fields: Option<&Bound<'_, PyDict>>) -> R<Identity> {
        let map = kwargs_to_map(fields)?;
        let made = slf.get().writer()?.identity(map)?;
        Ok(Identity::read_from(made, &Self::document_view(slf)?, "identity", None))
    }

    #[pyo3(signature = (**fields))]
    fn cohort(slf: &Bound<'_, Self>, fields: Option<&Bound<'_, PyDict>>) -> R<Cohort> {
        let map = kwargs_to_map(fields)?;
        let made = slf.get().writer()?.cohort(map)?;
        Ok(Cohort::read_from(made, &Self::document_view(slf)?, "cohort", None))
    }

    /// Declare a timepoint; the first explicit one replaces the implicit `tp0`.
    #[pyo3(signature = (timepoint_id, **fields))]
    fn add_timepoint(slf: &Bound<'_, Self>, timepoint_id: &str, fields: Option<&Bound<'_, PyDict>>) -> R<Timepoint> {
        let map = kwargs_to_map(fields)?;
        let made = slf.get().writer()?.add_timepoint(timepoint_id, map)?;
        Ok(Timepoint::read_from(made, &Self::document_view(slf)?, "timepoints", Some(timepoint_id)))
    }

    /// Set the label set, as it stands; the document's is returned, so an
    /// edit to a class's `properties` through it reaches the file.
    fn label_set(slf: &Bound<'_, Self>, label_set: &Bound<'_, PyAny>) -> R<crate::labels::LabelSet> {
        let ls = crate::labels::label_set_arg(label_set)?;
        let stored = slf.get().writer()?.label_set(ls);
        Ok(crate::labels::LabelSet::read_from(stored, &Self::document_view(slf)?))
    }

    #[pyo3(signature = (name, agent_id=None, **fields))]
    fn person(&self, name: &str, agent_id: Option<&str>, fields: Option<&Bound<'_, PyDict>>) -> R<Agent> {
        let map = kwargs_to_map(fields)?;
        Ok(Agent::wrap(self.writer()?.person(name, agent_id, map)?))
    }

    #[pyo3(signature = (name, version=None, **fields))]
    fn software(&self, name: &str, version: Option<&str>, fields: Option<&Bound<'_, PyDict>>) -> R<Agent> {
        let map = kwargs_to_map(fields)?;
        Ok(Agent::wrap(self.writer()?.software(name, version, map)?))
    }

    #[pyo3(signature = (name, **fields))]
    fn organization(&self, name: &str, fields: Option<&Bound<'_, PyDict>>) -> R<Agent> {
        let map = kwargs_to_map(fields)?;
        Ok(Agent::wrap(self.writer()?.organization(name, map)?))
    }

    /// Record an activity; ids are `act_<type>_<n>` unless given.
    #[pyo3(signature = (activity_type, *, agent=None, activity_id=None, **fields))]
    fn activity(
        slf: &Bound<'_, Self>,
        activity_type: &str,
        agent: Option<&Bound<'_, PyAny>>,
        activity_id: Option<&str>,
        fields: Option<&Bound<'_, PyDict>>,
    ) -> R<Activity> {
        let agent = agent_ref(agent)?;
        let map = kwargs_to_map(fields)?;
        let made = slf.get().writer()?.activity(activity_type, agent.as_deref(), activity_id, map)?;
        let id = made.id.clone();
        Ok(Activity::read_from(made, &Self::document_view(slf)?, "provenance.activities", Some(&id)))
    }

    /// Create or replace a quality record (status defaults to `draft`).
    #[pyo3(signature = (key, **fields))]
    fn set_quality(slf: &Bound<'_, Self>, key: &str, fields: Option<&Bound<'_, PyDict>>) -> R<QualityRecord> {
        let map = kwargs_to_map(fields)?;
        let made = slf.get().writer()?.set_quality(key, map)?;
        Ok(QualityRecord::read_from(made, &Self::document_view(slf)?, "quality", Some(key)))
    }

    /// Record a split claim, replacing any earlier claim for the same set.
    #[pyo3(signature = (**fields))]
    fn split(&self, fields: Option<&Bound<'_, PyDict>>) -> R<SplitClaim> {
        let map = kwargs_to_map(fields)?;
        Ok(SplitClaim::wrap(self.writer()?.split(map)?))
    }

    #[pyo3(signature = (**fields))]
    fn deidentification(slf: &Bound<'_, Self>, fields: Option<&Bound<'_, PyDict>>) -> R<Deidentification> {
        let map = kwargs_to_map(fields)?;
        let made = slf.get().writer()?.deidentification(map)?;
        Ok(Deidentification::read_from(made, &Self::document_view(slf)?, "deidentification", None))
    }

    /// Merge acquisition parameters for one image; returns them all, as the
    /// document's dict: an edit to it reaches the file.
    #[pyo3(signature = (image_id, **params))]
    fn acquisition<'py>(
        slf: &Bound<'py, Self>,
        image_id: &str,
        params: Option<&Bound<'py, PyDict>>,
    ) -> R<Bound<'py, PyAny>> {
        let map = kwargs_to_map(params)?;
        slf.get().writer()?.acquisition(image_id, map);
        Ok(Self::document_view(slf)?.getattr("acquisition")?.get_item(image_id)?)
    }

    /// Set a namespaced extension member of the document.
    fn extra(&self, namespace: &str, value: &Bound<'_, PyAny>) -> R<()> {
        let value = py_to_json(value)?;
        self.writer()?.extra(namespace, value);
        Ok(())
    }

    // -- grids ---------------------------------------------------------------------

    /// Declare a grid.  Geometry lives here and nowhere else.
    #[pyo3(signature = (grid_id, *, shape, spacing, origin=None, direction=None, axis_names=None, axis_kinds=None,
        coord_system="LPS", units="mm", timepoint=None, frame_uid=None, patch_hint=None, chunk_hint=None,
        time_values=None, time_units=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_grid(
        &self,
        py: Python<'_>,
        grid_id: &str,
        shape: &Bound<'_, PyAny>,
        spacing: &Bound<'_, PyAny>,
        origin: Option<&Bound<'_, PyAny>>,
        direction: Option<&Bound<'_, PyAny>>,
        axis_names: Option<&Bound<'_, PyAny>>,
        axis_kinds: Option<&Bound<'_, PyAny>>,
        coord_system: &str,
        units: &str,
        timepoint: Option<String>,
        frame_uid: Option<String>,
        patch_hint: Option<&Bound<'_, PyAny>>,
        chunk_hint: Option<&Bound<'_, PyAny>>,
        time_values: Option<&Bound<'_, PyAny>>,
        time_units: Option<String>,
    ) -> R<Grid> {
        let options = GridOptions {
            origin: is_given(origin).map(|o| f64_vec(&o)).transpose()?,
            direction: is_given(direction).map(|d| to_matrix(&d)).transpose()?,
            axis_names: opt_strings(axis_names)?,
            axis_kinds: opt_strings(axis_kinds)?,
            coord_system: Some(coord_system.to_string()),
            units: Some(units.to_string()),
            timepoint,
            frame_uid,
            patch_hint: is_given(patch_hint).map(|o| i64_vec(&o)).transpose()?,
            chunk_hint: is_given(chunk_hint).map(|o| i64_vec(&o)).transpose()?,
            time_values: is_given(time_values).map(|o| f64_vec(&o)).transpose()?,
            time_units,
        };
        let shape = i64_vec(shape)?;
        let spacing = f64_vec(spacing)?;
        h5(py, || Ok(Grid::wrap(self.writer()?.add_grid(grid_id, &shape, &spacing, options)?)))
    }

    /// Grids declared so far.
    #[getter]
    fn grids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let grids: Vec<_> = self.state()?.grids().iter().map(|(k, g)| (k.clone(), g.clone())).collect();
        let out = PyDict::new(py);
        for (k, g) in grids {
            out.set_item(k, Grid::wrap(g))?;
        }
        Ok(out)
    }

    /// Rename frames of reference everywhere they are named (§3.4).
    fn remap_frame_uids<'py>(&self, py: Python<'py>, mapping: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
        let mut map = HashMap::new();
        for (k, v) in items_of(mapping)? {
            map.insert(k.extract::<String>()?, v.extract::<String>()?);
        }
        let renamed = h5(py, || -> R<_> { Ok(self.writer()?.remap_frame_uids(&map)?) })?;
        Ok(PyTuple::new(py, renamed)?)
    }

    /// Every frame UID in the file so far, and the attributes naming it.
    fn frame_uids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let uids = h5(py, || -> R<_> { Ok(self.state()?.frame_uids()?) })?;
        let out = PyDict::new(py);
        for (k, v) in uids {
            out.set_item(k, PyTuple::new(py, v)?)?;
        }
        Ok(out)
    }

    // -- images --------------------------------------------------------------------

    /// Write one image, chunked for its grid's patch hint.
    #[pyo3(signature = (image_id, data, *, grid, modality, value_type="intensity", value_units=None,
        channel_names=None, rescale_slope=None, rescale_intercept=None, window_center=None, window_width=None,
        valid_mask=None, prov=None, codec=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_image(
        &self,
        py: Python<'_>,
        image_id: &str,
        data: &Bound<'_, PyAny>,
        grid: &str,
        modality: &str,
        value_type: &str,
        value_units: Option<String>,
        channel_names: Option<&Bound<'_, PyAny>>,
        rescale_slope: Option<f64>,
        rescale_intercept: Option<f64>,
        window_center: Option<&Bound<'_, PyAny>>,
        window_width: Option<&Bound<'_, PyAny>>,
        valid_mask: Option<String>,
        prov: Option<&Bound<'_, PyAny>>,
        codec: Option<String>,
    ) -> R<Dataset> {
        let options = ImageOptions {
            value_type: Some(value_type.to_string()),
            value_units,
            channel_names: opt_strings(channel_names)?,
            rescale_slope,
            rescale_intercept,
            window_center: is_given(window_center).map(|o| f64_vec(&o)).transpose()?,
            window_width: is_given(window_width).map(|o| f64_vec(&o)).transpose()?,
            valid_mask,
            prov: prov_arg(prov)?,
            codec,
        };
        let array = py_to_nd(data)?;
        let mut held = self.writer()?;
        let writer = &mut *held;
        let (id, grid, modality) = (image_id.to_string(), grid.to_string(), modality.to_string());
        let ds = py.detach(move || writer.add_image(&id, &array, &grid, &modality, options))?;
        Ok(Dataset::wrap(ds))
    }

    /// Write a multiscale image (§4.3); level geometry is checked here.
    #[pyo3(signature = (image_id, levels, *, grid_levels, modality, value_type="intensity", downsample_method="mean",
        value_units=None, rescale_slope=None, rescale_intercept=None, prov=None, codec=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_pyramid(
        &self,
        py: Python<'_>,
        image_id: &str,
        levels: &Bound<'_, PyAny>,
        grid_levels: &Bound<'_, PyAny>,
        modality: &str,
        value_type: &str,
        downsample_method: &str,
        value_units: Option<String>,
        rescale_slope: Option<f64>,
        rescale_intercept: Option<f64>,
        prov: Option<&Bound<'_, PyAny>>,
        codec: Option<String>,
    ) -> R<Group> {
        let arrays = levels.try_iter()?.map(|l| py_to_nd(&l?)).collect::<PyResult<Vec<_>>>()?;
        let grid_levels = strings(grid_levels)?;
        let options = ImageOptions {
            value_type: Some(value_type.to_string()),
            value_units,
            rescale_slope,
            rescale_intercept,
            prov: prov_arg(prov)?,
            codec,
            ..Default::default()
        };
        let mut held = self.writer()?;
        let writer = &mut *held;
        let (id, modality, method) = (image_id.to_string(), modality.to_string(), downsample_method.to_string());
        let group = py.detach(move || writer.add_pyramid(&id, &arrays, &grid_levels, &modality, &method, options))?;
        Ok(Group::wrap(group))
    }

    // -- voxel annotations -------------------------------------------------------------

    /// Write a voxel annotation, choosing the encoding by measurement.
    ///
    /// Returns the chosen `kind` and the overlap statistics behind the choice.
    /// `encoding` is `"auto"` or one of `TRANSCODABLE`: a `mask` has no
    /// classes, so it is refused (E404), as a transcode to it is.  Masks
    /// stored as `instances` are one object per class, each minted an id
    /// above every id the sample holds (§7.4).
    ///
    /// `labelmap` is the masks as one integer volume, borrowed rather than
    /// copied: each nonzero value the class of that id, or the classes
    /// `label_values` gives it, and `ignore_value` the ignore region.  What it
    /// writes is what the masks it implies write, byte for byte.
    #[pyo3(signature = (ann_id, *, grid, masks=None, probabilities=None, instances=None, labelmap=None,
        label_values=None, ignore_value=None, encoding="auto", threshold=None, annotated_classes=None,
        closure="explicit", ignore=None, ignore_mask=None, timepoints=None, prov=None, quality=None, derived_from=None,
        task="segmentation", codec=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_segmentation<'py>(
        &self,
        py: Python<'py>,
        ann_id: &str,
        grid: &str,
        masks: Option<&Bound<'py, PyAny>>,
        probabilities: Option<&Bound<'py, PyAny>>,
        instances: Option<&Bound<'py, PyAny>>,
        labelmap: Option<&Bound<'py, PyAny>>,
        label_values: Option<&Bound<'py, PyAny>>,
        ignore_value: Option<i64>,
        encoding: &str,
        threshold: Option<f64>,
        annotated_classes: Option<&Bound<'py, PyAny>>,
        closure: &str,
        ignore: Option<&Bound<'py, PyAny>>,
        ignore_mask: Option<String>,
        timepoints: Option<&Bound<'py, PyAny>>,
        prov: Option<&Bound<'py, PyAny>>,
        quality: Option<&Bound<'py, PyAny>>,
        derived_from: Option<&Bound<'py, PyAny>>,
        task: &str,
        codec: Option<&str>,
    ) -> R<Bound<'py, PyTuple>> {
        self.writer()?.grid_ref(grid)?;
        let sources =
            [("masks", masks), ("probabilities", probabilities), ("instances", instances), ("labelmap", labelmap)];
        let given: Vec<&str> = sources.into_iter().filter(|(_, v)| is_given(*v).is_some()).map(|(n, _)| n).collect();
        if given.len() > 1 {
            return Err(medh5::Error::coded(
                "E404",
                format!(
                    "annotation {}: pass one of masks=, probabilities=, instances= or labelmap=, not {}",
                    medh5::json::repr_str(ann_id),
                    given.join(" and ")
                ),
            )
            .into());
        }
        let labelmap = is_given(labelmap);
        let values = is_given(label_values).map(|v| label_values_arg(&v)).transpose()?;
        if labelmap.is_none() && (values.is_some() || ignore_value.is_some()) {
            return Err(medh5::Error::coded(
                "E404",
                format!(
                    "annotation {}: label_values= and ignore_value= apply to labelmap= only",
                    medh5::json::repr_str(ann_id)
                ),
            )
            .into());
        }
        // Instance classes are resolved with the write, under its lock.
        let mut keys = None;
        let source = if labelmap.is_some() {
            None
        } else if let Some(m) = is_given(masks) {
            let mut out = Vec::new();
            for (k, v) in items_of(&m)? {
                let what = format!("annotation {}: the mask of class {}", medh5::json::repr_str(ann_id), k.repr()?);
                out.push((class_key(&k)?, mask_arg(&v, &what)?));
            }
            Some(SegmentationSource::Masks(out))
        } else if let Some(p) = is_given(probabilities) {
            let mut out = Vec::new();
            for (k, v) in items_of(&p)? {
                out.push((class_key(&k)?, f64_array(&v)?));
            }
            Some(SegmentationSource::Probabilities(out))
        } else if let Some(i) = is_given(instances) {
            let (classes, given): (Vec<_>, Vec<_>) = instance_inputs(&i)?.into_iter().unzip();
            keys = Some(classes);
            Some(SegmentationSource::Instances(given))
        } else {
            return Err(medh5::Error::invalid(
                "add_segmentation needs one of masks=, probabilities=, instances= or labelmap=",
            )
            .into());
        };
        let options = SegmentationOptions {
            encoding: Some(encoding.to_string()),
            threshold,
            ignore: is_given(ignore)
                .map(|i| mask_arg(&i, &format!("annotation {}: ignore", medh5::json::repr_str(ann_id))))
                .transpose()?,
            ignore_mask,
            common: common(annotated_classes, closure, timepoints, prov, quality, derived_from, Some(task), codec)?,
        };
        let (kind, stats) = match (source, labelmap) {
            (Some(mut source), _) => {
                let mut held = self.writer()?;
                if let (SegmentationSource::Instances(given), Some(keys)) = (&mut source, keys) {
                    for (instance, key) in given.iter_mut().zip(&keys) {
                        instance.class_id = held.class_id(key)?;
                    }
                }
                let writer = &mut *held;
                py.detach(move || writer.add_segmentation(ann_id, grid, source, options))?
            }
            (None, Some(labelmap)) => add_labelmap(self, ann_id, grid, &labelmap, values, ignore_value, options)?,
            (None, None) => unreachable!("one source is given"),
        };
        let stats = match stats {
            Some(s) => Bound::new(py, crate::annotations::OverlapStats::wrap(s))?.into_any(),
            None => py.None().into_bound(py),
        };
        Ok(PyTuple::new(py, [PyString::new(py, &kind).into_any(), stats])?)
    }

    /// Write a boolean `mask` annotation (FOV, ignore region).  A
    /// floating-point mask holding a value other than 0 and 1 is refused
    /// (E411) rather than read as set wherever it is non-zero.
    #[pyo3(signature = (ann_id, mask, *, grid, task="other", prov=None, codec=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_mask(
        &self,
        py: Python<'_>,
        ann_id: &str,
        mask: &Bound<'_, PyAny>,
        grid: &str,
        task: &str,
        prov: Option<&Bound<'_, PyAny>>,
        codec: Option<&str>,
    ) -> R<Group> {
        let mask = mask_arg(mask, &format!("mask {}", medh5::json::repr_str(ann_id)))?;
        let prov = prov_arg(prov)?;
        let mut held = self.writer()?;
        let writer = &mut *held;
        let group = py.detach(move || -> medh5::Result<_> {
            writer.add_mask(ann_id, mask, grid, task, prov.as_deref(), codec)?;
            writer.root()?.group("annotations")?.group(ann_id).map_err(Into::into)
        })?;
        Ok(Group::wrap(group))
    }

    /// Drop an annotation, and any index entry derived from it.
    fn remove_annotation(&self, py: Python<'_>, ann_id: &str) -> R<()> {
        h5(py, || Ok(self.writer()?.remove_annotation(ann_id)?))
    }

    /// Drop a transform; what still names it is checked at commit.
    fn remove_transform(&self, py: Python<'_>, transform_id: &str) -> R<()> {
        h5(py, || Ok(self.writer()?.remove_transform(transform_id)?))
    }

    /// Drop an image, its `acquisition` entry and its `series_uids`
    /// mentions; what still names it, and the last image going (E201), are
    /// checked at commit.
    fn remove_image(&self, py: Python<'_>, image_id: &str) -> R<()> {
        h5(py, || Ok(self.writer()?.remove_image(image_id)?))
    }

    /// Drop a grid; what still names it (E101, E503), and the last grid
    /// going (E111), are checked at commit.
    fn remove_grid(&self, py: Python<'_>, grid_id: &str) -> R<()> {
        h5(py, || Ok(self.writer()?.remove_grid(grid_id)?))
    }

    /// Change an image's attributes without rewriting its samples (§4.1):
    /// `None` removes one.  Checked as `add_image` checks them.
    #[pyo3(signature = (image_id, **attrs))]
    fn set_image_attrs(&self, py: Python<'_>, image_id: &str, attrs: Option<&Bound<'_, PyDict>>) -> R<()> {
        let mut edit = ImageEdit::default();
        for (key, value) in attrs.into_iter().flat_map(|d| d.iter()) {
            let key: String = key.extract()?;
            let given = (!value.is_none()).then_some(&value);
            let text = |v: Option<&Bound<'_, PyAny>>| v.map(|v| v.extract::<String>()).transpose();
            match key.as_str() {
                "modality" => edit.modality = Some(text(given)?),
                "value_type" => edit.value_type = Some(text(given)?),
                "value_units" => edit.value_units = Some(text(given)?),
                "valid_mask" => edit.valid_mask = Some(text(given)?),
                "prov" => edit.prov = Some(prov_arg(given)?),
                "channel_names" => edit.channel_names = Some(opt_strings(given)?),
                "rescale_slope" => edit.rescale_slope = Some(given.map(|v| v.extract::<f64>()).transpose()?),
                "rescale_intercept" => edit.rescale_intercept = Some(given.map(|v| v.extract::<f64>()).transpose()?),
                "window_center" => edit.window_center = Some(given.map(f64_vec).transpose()?),
                "window_width" => edit.window_width = Some(given.map(f64_vec).transpose()?),
                _ => {
                    let why = match key.as_str() {
                        "grid" => "an image's grid is its geometry; write the image again on the other grid",
                        "digest" => "a digest is the commit's to stamp",
                        "levels" | "downsample_factors" | "downsample_method" | "grid_levels" => {
                            "a pyramid's levels are written by add_pyramid"
                        }
                        _ => "the specification defines no such image attribute (§4.1)",
                    };
                    return Err(medh5::Error::invalid(format!(
                        "set_image_attrs does not edit {}: {why}",
                        medh5::json::repr_str(&key)
                    ))
                    .into());
                }
            }
        }
        h5(py, || Ok(self.writer()?.set_image_attrs(image_id, edit)?))
    }

    /// Re-encode a voxel annotation in place, preserving its header (§7.6).
    /// A `probmap` is thresholded at its `threshold` (§7.5): another encoding
    /// keeps the voxels at or above it as the class, and none of the
    /// probabilities.
    #[pyo3(signature = (ann_id, to_kind, *, codec=None, drop_identity=false))]
    fn transcode_annotation(
        &self,
        py: Python<'_>,
        ann_id: &str,
        to_kind: &str,
        codec: Option<String>,
        drop_identity: bool,
    ) -> R<String> {
        let mut held = self.writer()?;
        let writer = &mut *held;
        let (id, kind) = (ann_id.to_string(), to_kind.to_string());
        Ok(py.detach(move || writer.transcode_annotation(&id, &kind, codec.as_deref(), drop_identity))?)
    }

    // -- geometric and classification annotations (§8, §9) ---------------------------------

    /// Axis-aligned boxes, `(N, S, 2)` in `[lo, hi]` at voxel edges (§8.2).
    #[pyo3(signature = (ann_id, boxes, class_ids, *, grid=None, space="index", frame_uid=None, instance_ids=None,
        scores=None, attributes=None, slice_index=None, annotated_classes=None, closure="explicit", timepoints=None,
        prov=None, quality=None, derived_from=None, task="detection", codec=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_boxes(
        &self,
        py: Python<'_>,
        ann_id: &str,
        boxes: &Bound<'_, PyAny>,
        class_ids: &Bound<'_, PyAny>,
        grid: Option<&str>,
        space: &str,
        frame_uid: Option<&str>,
        instance_ids: Option<&Bound<'_, PyAny>>,
        scores: Option<&Bound<'_, PyAny>>,
        attributes: Option<&Bound<'_, PyAny>>,
        slice_index: Option<&Bound<'_, PyAny>>,
        annotated_classes: Option<&Bound<'_, PyAny>>,
        closure: &str,
        timepoints: Option<&Bound<'_, PyAny>>,
        prov: Option<&Bound<'_, PyAny>>,
        quality: Option<&Bound<'_, PyAny>>,
        derived_from: Option<&Bound<'_, PyAny>>,
        task: &str,
        codec: Option<&str>,
    ) -> R<Group> {
        let boxes = f64_array(boxes)?;
        let keys = class_keys(class_ids)?;
        let objects = objects(instance_ids, scores, attributes)?;
        let n_boxes = boxes.shape().first().copied().unwrap_or(0);
        let slice_index =
            is_given(slice_index).map(|o| crate::annotations::slice_index_arg(&o, n_boxes)).transpose()?;
        let options = common(annotated_classes, closure, timepoints, prov, quality, derived_from, Some(task), codec)?;
        let at = placement(grid, Some(space), frame_uid);
        let mut held = self.writer()?;
        let writer = &mut *held;
        let group = detach_if(py, n_boxes >= DETACH_ROWS, move || {
            writer.add_boxes(ann_id, &boxes, &keys, objects, slice_index, at, options)
        })?;
        Ok(Group::wrap(group))
    }

    /// Oriented boxes: centre, full edge lengths, rotation (§8.3).
    #[pyo3(signature = (ann_id, centers, sizes, rotations, class_ids, *, grid=None, space="index", frame_uid=None,
        instance_ids=None, scores=None, attributes=None, annotated_classes=None, closure="explicit",
        timepoints=None, prov=None, quality=None, derived_from=None, task="detection", codec=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_obb(
        &self,
        py: Python<'_>,
        ann_id: &str,
        centers: &Bound<'_, PyAny>,
        sizes: &Bound<'_, PyAny>,
        rotations: &Bound<'_, PyAny>,
        class_ids: &Bound<'_, PyAny>,
        grid: Option<&str>,
        space: &str,
        frame_uid: Option<&str>,
        instance_ids: Option<&Bound<'_, PyAny>>,
        scores: Option<&Bound<'_, PyAny>>,
        attributes: Option<&Bound<'_, PyAny>>,
        annotated_classes: Option<&Bound<'_, PyAny>>,
        closure: &str,
        timepoints: Option<&Bound<'_, PyAny>>,
        prov: Option<&Bound<'_, PyAny>>,
        quality: Option<&Bound<'_, PyAny>>,
        derived_from: Option<&Bound<'_, PyAny>>,
        task: &str,
        codec: Option<&str>,
    ) -> R<Group> {
        let (centers, sizes, rotations) = (f64_array(centers)?, f64_array(sizes)?, f64_array(rotations)?);
        let keys = class_keys(class_ids)?;
        let objects = objects(instance_ids, scores, attributes)?;
        let options = common(annotated_classes, closure, timepoints, prov, quality, derived_from, Some(task), codec)?;
        let rows = centers.shape().first().copied().unwrap_or(0);
        let at = placement(grid, Some(space), frame_uid);
        let mut held = self.writer()?;
        let writer = &mut *held;
        let group = detach_if(py, rows >= DETACH_ROWS, move || {
            writer.add_obb(ann_id, &centers, &sizes, &rotations, &keys, objects, at, options)
        })?;
        Ok(Group::wrap(group))
    }

    /// `(N, K, S)` keypoints with per-slot classes (§8.4).
    #[pyo3(signature = (ann_id, points, keypoint_classes, class_ids, *, grid=None, space="index", frame_uid=None,
        visibility=None, instance_ids=None, scores=None, skeleton=None, annotated_classes=None, closure="explicit",
        timepoints=None, prov=None, quality=None, derived_from=None, task="detection", codec=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_keypoints(
        &self,
        py: Python<'_>,
        ann_id: &str,
        points: &Bound<'_, PyAny>,
        keypoint_classes: &Bound<'_, PyAny>,
        class_ids: &Bound<'_, PyAny>,
        grid: Option<&str>,
        space: &str,
        frame_uid: Option<&str>,
        visibility: Option<&Bound<'_, PyAny>>,
        instance_ids: Option<&Bound<'_, PyAny>>,
        scores: Option<&Bound<'_, PyAny>>,
        skeleton: Option<&str>,
        annotated_classes: Option<&Bound<'_, PyAny>>,
        closure: &str,
        timepoints: Option<&Bound<'_, PyAny>>,
        prov: Option<&Bound<'_, PyAny>>,
        quality: Option<&Bound<'_, PyAny>>,
        derived_from: Option<&Bound<'_, PyAny>>,
        task: &str,
        codec: Option<&str>,
    ) -> R<Group> {
        let points = f64_array(points)?;
        let slots = class_keys(keypoint_classes)?;
        let keys = class_keys(class_ids)?;
        let visibility = is_given(visibility).map(|v| i64_array(&v)).transpose()?;
        let objects = objects(instance_ids, scores, None)?;
        let options = common(annotated_classes, closure, timepoints, prov, quality, derived_from, Some(task), codec)?;
        let rows = points.shape().first().copied().unwrap_or(0) * slots.len().max(1);
        let at = placement(grid, Some(space), frame_uid);
        let mut held = self.writer()?;
        let writer = &mut *held;
        let group = detach_if(py, rows >= DETACH_ROWS, move || {
            writer.add_keypoints(ann_id, &points, &slots, &keys, visibility.as_ref(), objects, skeleton, at, options)
        })?;
        Ok(Group::wrap(group))
    }

    /// A point set: landmarks, seeds, or half a correspondence (§8.5).
    #[pyo3(signature = (ann_id, points, *, grid=None, space="index", frame_uid=None, class_ids=None,
        instance_ids=None, names=None, weights=None, correspondence=None, annotated_classes=None, closure="explicit",
        timepoints=None, prov=None, quality=None, derived_from=None, task="detection", codec=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_points(
        &self,
        py: Python<'_>,
        ann_id: &str,
        points: &Bound<'_, PyAny>,
        grid: Option<&str>,
        space: &str,
        frame_uid: Option<&str>,
        class_ids: Option<&Bound<'_, PyAny>>,
        instance_ids: Option<&Bound<'_, PyAny>>,
        names: Option<&Bound<'_, PyAny>>,
        weights: Option<&Bound<'_, PyAny>>,
        correspondence: Option<&str>,
        annotated_classes: Option<&Bound<'_, PyAny>>,
        closure: &str,
        timepoints: Option<&Bound<'_, PyAny>>,
        prov: Option<&Bound<'_, PyAny>>,
        quality: Option<&Bound<'_, PyAny>>,
        derived_from: Option<&Bound<'_, PyAny>>,
        task: &str,
        codec: Option<&str>,
    ) -> R<Group> {
        let points = f64_array(points)?;
        let keys = opt_keys(class_ids)?;
        let instance_ids = is_given(instance_ids).map(|i| u64s(&i)).transpose()?;
        let names = opt_strings(names)?;
        let weights = is_given(weights).map(|w| f64_vec(&w)).transpose()?;
        let options = common(annotated_classes, closure, timepoints, prov, quality, derived_from, Some(task), codec)?;
        let rows = points.shape().first().copied().unwrap_or(0);
        let at = placement(grid, Some(space), frame_uid);
        let mut held = self.writer()?;
        let writer = &mut *held;
        let group = detach_if(py, rows >= DETACH_ROWS, move || {
            writer.add_points(
                ann_id,
                &points,
                keys.as_deref(),
                instance_ids.as_deref(),
                names.as_deref(),
                weights.as_deref(),
                correspondence,
                at,
                options,
            )
        })?;
        Ok(Group::wrap(group))
    }

    /// Planar polygons (§8.6) --- the RTSTRUCT-shaped annotation.
    #[pyo3(signature = (ann_id, polygons, *, grid=None, space="index", frame_uid=None, annotated_classes=None,
        closure="explicit", timepoints=None, prov=None, quality=None, derived_from=None, task="segmentation",
        codec=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_contours(
        &self,
        py: Python<'_>,
        ann_id: &str,
        polygons: &Bound<'_, PyAny>,
        grid: Option<&str>,
        space: &str,
        frame_uid: Option<&str>,
        annotated_classes: Option<&Bound<'_, PyAny>>,
        closure: &str,
        timepoints: Option<&Bound<'_, PyAny>>,
        prov: Option<&Bound<'_, PyAny>>,
        quality: Option<&Bound<'_, PyAny>>,
        derived_from: Option<&Bound<'_, PyAny>>,
        task: &str,
        codec: Option<&str>,
    ) -> R<Group> {
        let given = polygons_of(polygons)?;
        let options = common(annotated_classes, closure, timepoints, prov, quality, derived_from, Some(task), codec)?;
        let at = placement(grid, Some(space), frame_uid);
        // Every class against the one label set the write holds (E4).
        let mut held = self.writer()?;
        let found = given.into_iter().map(|p| p.resolve(|k| held.class_id(k))).collect::<R<Vec<_>>>()?;
        let writer = &mut *held;
        let group = py.detach(move || writer.add_contours(ann_id, &found, at, options))?;
        Ok(Group::wrap(group))
    }

    /// Open curves (1.2 §3): centrelines, catheter and applicator paths,
    /// RTSTRUCT's open contours.  A sample holding one is written at format
    /// 1.2.
    #[pyo3(signature = (ann_id, polylines, *, grid=None, space="index", frame_uid=None, instance_ids=None,
        annotated_classes=None, closure="explicit", timepoints=None, prov=None, quality=None, derived_from=None,
        task="other", codec=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_polylines(
        &self,
        py: Python<'_>,
        ann_id: &str,
        polylines: &Bound<'_, PyAny>,
        grid: Option<&str>,
        space: &str,
        frame_uid: Option<&str>,
        instance_ids: Option<&Bound<'_, PyAny>>,
        annotated_classes: Option<&Bound<'_, PyAny>>,
        closure: &str,
        timepoints: Option<&Bound<'_, PyAny>>,
        prov: Option<&Bound<'_, PyAny>>,
        quality: Option<&Bound<'_, PyAny>>,
        derived_from: Option<&Bound<'_, PyAny>>,
        task: &str,
        codec: Option<&str>,
    ) -> R<Group> {
        let ids = is_given(instance_ids).map(|i| u64s(&i)).transpose()?;
        let given = polylines_of(polylines)?;
        let options = common(annotated_classes, closure, timepoints, prov, quality, derived_from, Some(task), codec)?;
        let at = placement(grid, Some(space), frame_uid);
        // Every class against the one label set the write holds (E4).
        let mut held = self.writer()?;
        let found = self::polylines(&held, given, ids)?;
        let writer = &mut *held;
        let group = py.detach(move || writer.add_polylines(ann_id, &found, &at, &options))?;
        Ok(Group::wrap(group))
    }

    /// A triangle surface mesh (§8.7); `space` defaults to `world`.
    #[pyo3(signature = (ann_id, vertices, faces, *, grid=None, space="world", frame_uid=None, normals=None,
        vertex_class_ids=None, mesh_offsets=None, mesh_class_ids=None, annotated_classes=None, closure="explicit",
        timepoints=None, prov=None, quality=None, derived_from=None, task="segmentation", codec=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_mesh(
        &self,
        py: Python<'_>,
        ann_id: &str,
        vertices: &Bound<'_, PyAny>,
        faces: &Bound<'_, PyAny>,
        grid: Option<&str>,
        space: &str,
        frame_uid: Option<&str>,
        normals: Option<&Bound<'_, PyAny>>,
        vertex_class_ids: Option<&Bound<'_, PyAny>>,
        mesh_offsets: Option<&Bound<'_, PyAny>>,
        mesh_class_ids: Option<&Bound<'_, PyAny>>,
        annotated_classes: Option<&Bound<'_, PyAny>>,
        closure: &str,
        timepoints: Option<&Bound<'_, PyAny>>,
        prov: Option<&Bound<'_, PyAny>>,
        quality: Option<&Bound<'_, PyAny>>,
        derived_from: Option<&Bound<'_, PyAny>>,
        task: &str,
        codec: Option<&str>,
    ) -> R<Group> {
        let vertices = f64_array(vertices)?;
        let faces = i64_array(faces)?;
        let normals = is_given(normals).map(|n| f64_array(&n)).transpose()?;
        let vertex_ids = opt_keys(vertex_class_ids)?;
        let offsets = is_given(mesh_offsets).map(|o| i64_vec(&o)).transpose()?;
        let mesh_ids = opt_keys(mesh_class_ids)?;
        let options = common(annotated_classes, closure, timepoints, prov, quality, derived_from, Some(task), codec)?;
        let at = placement(grid, Some(space), frame_uid);
        let mut held = self.writer()?;
        let writer = &mut *held;
        let group = py.detach(move || {
            writer.add_mesh(
                ann_id,
                &vertices,
                &faces,
                normals.as_ref(),
                vertex_ids.as_deref(),
                offsets.as_deref(),
                mesh_ids.as_deref(),
                at,
                options,
            )
        })?;
        Ok(Group::wrap(group))
    }

    /// A classification annotation (§9).  A change label is `scope="sample"`
    /// with explicit `timepoints`.  *labels* is a mapping `class -> value` or
    /// rows `(class, value[, scope_id[, scheme, scheme_value]])`.
    #[pyo3(signature = (ann_id, labels, *, scope="sample", multilabel=true, scope_ids=None, schemes=None,
        scheme_values=None, grid=None, annotated_classes=None, closure="explicit", timepoints=None, prov=None,
        quality=None, derived_from=None, codec=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_classification(
        &self,
        py: Python<'_>,
        ann_id: &str,
        labels: &Bound<'_, PyAny>,
        scope: &str,
        multilabel: bool,
        scope_ids: Option<&Bound<'_, PyAny>>,
        schemes: Option<&Bound<'_, PyAny>>,
        scheme_values: Option<&Bound<'_, PyAny>>,
        grid: Option<String>,
        annotated_classes: Option<&Bound<'_, PyAny>>,
        closure: &str,
        timepoints: Option<&Bound<'_, PyAny>>,
        prov: Option<&Bound<'_, PyAny>>,
        quality: Option<&Bound<'_, PyAny>>,
        derived_from: Option<&Bound<'_, PyAny>>,
        codec: Option<&str>,
    ) -> R<Group> {
        let rows = assertion_rows(labels)?;
        let keys = rows.classes.iter().map(class_key).collect::<PyResult<Vec<_>>>()?;
        let mut assertions = assertion_columns(rows, Vec::new(), scope_ids, schemes, scheme_values)?;
        let options = common(annotated_classes, closure, timepoints, prov, quality, derived_from, None, codec)?;
        // The classes against one label set, the one the write holds (E4).
        let mut held = self.writer()?;
        assertions.class_ids = keys.iter().map(|k| held.class_id(k)).collect::<medh5::Result<Vec<_>>>()?;
        let writer = &mut *held;
        let group = h5(py, move || writer.add_classification(ann_id, assertions, scope, multilabel, grid, options))?;
        Ok(Group::wrap(group))
    }

    // -- transforms (§10) --------------------------------------------------------------

    /// A transform mapping points from `from_frame` to `to_frame`:
    /// `x_M = T(x_F)`, the ITK convention.  `units` defaults to those of the
    /// grids in the two frames (§10.1).
    #[pyo3(signature = (transform_id, *, kind, from_frame, to_frame, matrix=None, field=None, control_points=None,
        components=None, field_grid=None, cp_grid=None, vector_space="world", interpolation="linear",
        extrapolation="zero", order=3, units=None, from_grid=None, to_grid=None, invertible=None, inverse_id=None,
        metrics=None, prov=None, codec=None, field_dtype="float32"))]
    #[allow(clippy::too_many_arguments)]
    fn add_transform(
        &self,
        py: Python<'_>,
        transform_id: &str,
        kind: &str,
        from_frame: &str,
        to_frame: &str,
        matrix: Option<&Bound<'_, PyAny>>,
        field: Option<&Bound<'_, PyAny>>,
        control_points: Option<&Bound<'_, PyAny>>,
        components: Option<&Bound<'_, PyAny>>,
        field_grid: Option<String>,
        cp_grid: Option<String>,
        vector_space: &str,
        interpolation: &str,
        extrapolation: &str,
        order: i64,
        units: Option<String>,
        from_grid: Option<String>,
        to_grid: Option<String>,
        invertible: Option<bool>,
        inverse_id: Option<String>,
        metrics: Option<&Bound<'_, PyAny>>,
        prov: Option<&Bound<'_, PyAny>>,
        codec: Option<String>,
        field_dtype: &str,
    ) -> R<Group> {
        let field_dtype = match field_dtype {
            "float32" => FieldDtype::Float32,
            "float16" => FieldDtype::Float16,
            "auto" => FieldDtype::Auto,
            other => {
                return Err(medh5::Error::invalid(format!(
                    "field_dtype {} is not one of 'float32', 'float16', 'auto'",
                    medh5::json::repr_str(other)
                ))
                .into())
            }
        };
        let spec = TransformSpec {
            matrix: is_given(matrix).map(|m| f64_array(&m)).transpose()?,
            field: is_given(field).map(|f| py_to_nd(&f)).transpose()?,
            control_points: is_given(control_points).map(|c| f64_array(&c)).transpose()?,
            components: opt_strings(components)?,
            field_grid,
            cp_grid,
            vector_space: Some(vector_space.to_string()),
            interpolation: Some(interpolation.to_string()),
            extrapolation: Some(extrapolation.to_string()),
            order: Some(order),
            units,
            from_grid,
            to_grid,
            invertible,
            inverse_id,
            metrics: quality_arg(metrics)?,
            prov: prov_arg(prov)?,
            codec,
        };
        let mut held = self.writer()?;
        let writer = &mut *held;
        let (id, kind, from, to) =
            (transform_id.to_string(), kind.to_string(), from_frame.to_string(), to_frame.to_string());
        let group = py.detach(move || writer.add_transform_with(&id, &kind, &from, &to, spec, field_dtype))?;
        Ok(Group::wrap(group))
    }

    // -- index and commit -----------------------------------------------------------------

    /// Build sampling indices (§14.3); every non-mask voxel annotation when
    /// `ann_ids` is `None`.  `occupancy=None` writes no occupancy planes.
    #[pyo3(signature = (ann_ids=None, *, max_coords=medh5::storage::index::DEFAULT_MAX_COORDS,
        occupancy=Some(medh5::storage::index::DEFAULT_OCCUPANCY_FACTOR), seed=0))]
    fn build_index<'py>(
        &self,
        py: Python<'py>,
        ann_ids: Option<&Bound<'py, PyAny>>,
        max_coords: usize,
        occupancy: Option<usize>,
        seed: u64,
    ) -> R<Bound<'py, PyTuple>> {
        let ids = opt_strings(ann_ids)?;
        let mut held = self.writer()?;
        let writer = &mut *held;
        let built = py.detach(move || writer.build_index(ids.as_deref(), Some(max_coords), Some(occupancy), seed))?;
        Ok(PyTuple::new(py, built)?)
    }

    /// Profiles this sample actually satisfies, unioned with declared ones.
    fn infer_profiles<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyFrozenSet>> {
        let profiles = h5(py, || -> R<_> { Ok(self.state()?.infer_profiles()?) })?;
        Ok(PyFrozenSet::new(py, profiles)?)
    }

    /// Validate, write `/meta`, stamp digests and atomically replace.
    ///
    /// `digests=True` recomputes every digest, and refuses (E701) to restamp
    /// one whose bytes the amended file already failed; `digests=False`
    /// stamps only datasets that carry none (or one in another algorithm than
    /// the file's).  An amended file keeps its `digest_algo`.  A commit
    /// refused before it changes anything --- no image (E201), a reference
    /// that does not resolve, a schema error (E005) --- leaves the writer
    /// open, to mend and commit again; one that fails after it began writing
    /// aborts it.  Returns the `content_id`, or `None` when already
    /// committed; committing an aborted writer raises.
    #[pyo3(signature = (*, digests=true))]
    fn commit(&self, py: Python<'_>, digests: bool) -> R<Option<String>> {
        let mut held = self.inner.write()?;
        let writer = &mut *held;
        Ok(py.detach(move || writer.commit(digests))?)
    }

    // -- the clinical profile (format 1.1) --------------------------------------------------------

    /// Declare the subject clock, starting the `clinical` profile (1.1 §3).
    #[pyo3(signature = (clock=None, **fields))]
    fn set_clock<'py>(
        &self,
        py: Python<'py>,
        clock: Option<&Bound<'py, PyAny>>,
        fields: Option<&Bound<'py, PyDict>>,
    ) -> R<Bound<'py, PyAny>> {
        let doc = crate::clinical::record_with(clock, fields)?;
        let clock = medh5::clinical::Clock::from_json(&doc)?;
        let added = h5(py, || -> R<_> { Ok(self.writer()?.set_clock(clock)?.to_json()) })?;
        Ok(crate::convert::json_to_py(py, &added)?)
    }

    /// Add one event version (1.1 §5): an `Event`, a dict, or keywords.
    #[pyo3(signature = (event=None, **fields))]
    fn add_event<'py>(
        &self,
        py: Python<'py>,
        event: Option<&Bound<'py, PyAny>>,
        fields: Option<&Bound<'py, PyDict>>,
    ) -> R<Bound<'py, PyAny>> {
        let event = crate::clinical::event_arg(event, fields)?;
        let added = h5(py, || -> R<_> { Ok(self.writer()?.add_event(event)?.to_json()) })?;
        Ok(crate::convert::json_to_py(py, &added)?)
    }

    /// Add one source document (1.1 §6).
    #[pyo3(signature = (document=None, **fields))]
    fn add_document<'py>(
        &self,
        py: Python<'py>,
        document: Option<&Bound<'py, PyAny>>,
        fields: Option<&Bound<'py, PyDict>>,
    ) -> R<Bound<'py, PyAny>> {
        let doc = medh5::clinical::Document::from_json(&crate::clinical::record_with(document, fields)?)?;
        let added = h5(py, || -> R<_> { Ok(self.writer()?.add_document(doc)?.to_json()) })?;
        Ok(crate::convert::json_to_py(py, &added)?)
    }

    /// Add one typed link (1.1 §7).
    #[pyo3(signature = (link=None, **fields))]
    fn add_link<'py>(
        &self,
        py: Python<'py>,
        link: Option<&Bound<'py, PyAny>>,
        fields: Option<&Bound<'py, PyDict>>,
    ) -> R<Bound<'py, PyAny>> {
        let link = medh5::clinical::Link::from_json(&crate::clinical::record_with(link, fields)?)?;
        let added = h5(py, || -> R<_> { Ok(self.writer()?.add_link(link)?.to_json()) })?;
        Ok(crate::convert::json_to_py(py, &added)?)
    }

    /// Add a logical-record bundle: `clinical`, `events`, `documents`, `links`.
    fn add_records(&self, py: Python<'_>, records: &Bound<'_, PyAny>) -> R<()> {
        let records = medh5::clinical::ClinicalRecords::from_json(&crate::clinical::record(records)?)?;
        let mut held = self.writer()?;
        let writer = &mut *held;
        Ok(py.detach(move || writer.add_records(records))?)
    }

    /// The clinical records so far (an amended file's once loaded), or `None`.
    fn clinical<'py>(&self, py: Python<'py>) -> R<Option<Bound<'py, PyAny>>> {
        let found = h5(py, || -> R<_> { Ok(self.writer()?.clinical()?.map(|r| r.to_json())) })?;
        Ok(found.map(|v| crate::convert::json_to_py(py, &v)).transpose()?)
    }

    #[getter]
    fn has_clinical(&self) -> R<bool> {
        Ok(self.state()?.has_clinical())
    }

    /// Remove the clinical profile: the imaging projection (1.1 §10), written
    /// at the lowest version what remains needs.  A sample with no clinical
    /// profile to remove keeps the version it declared (1.1 §2.1).
    fn drop_clinical(&self, py: Python<'_>) -> R<()> {
        h5(py, || Ok(self.writer()?.drop_clinical()?))
    }

    fn __repr__(&self) -> R<String> {
        let held = self.state()?;
        Ok(format!(
            "SampleWriter({}, codec={}{})",
            medh5::json::repr_str(&held.path.to_string_lossy()),
            medh5::json::repr_str(&held.codec),
            if held.is_closed() { ", closed" } else { "" }
        ))
    }
}

/// Create a new sample; use as a context manager, or call `commit()`.
#[pyfunction]
#[pyo3(signature = (path, *, sample_id=None, subject_id=None, codec="balanced", profiles=None))]
fn create(
    py: Python<'_>,
    path: PathBuf,
    sample_id: Option<String>,
    subject_id: Option<String>,
    codec: &str,
    profiles: Option<&Bound<'_, PyAny>>,
) -> R<SampleWriter> {
    SampleWriter::new(py, path, sample_id, subject_id, codec, profiles)
}

/// Copy-on-write amend: build a new file from the old and replace it.
///
/// On POSIX, anything holding the old file open keeps reading the old inode.
/// Windows cannot replace an open file: close every reader of it before the
/// commit, whose rename otherwise fails and leaves the file as it was.
#[pyfunction]
#[pyo3(signature = (path, *, codec=None))]
fn amend(py: Python<'_>, path: PathBuf, codec: Option<String>) -> R<SampleWriter> {
    let at = path.clone();
    let inner = py
        .detach(move || medh5::sample::writer::amend(&path, codec.as_deref()))
        .map_err(|e| crate::errors::path_error(e, &at))?;
    Ok(SampleWriter::of(inner))
}

/// The `medh5_version` a commit writes: the lowest version `profiles` and
/// annotation `kinds` need (`1.0`; `1.1` with `clinical`; `1.2` with a
/// `polylines` annotation), and never below `source`, the version an amended
/// file already declared (1.1 §2.3).  An amendment that removes the clinical
/// profile (`drop_clinical`), or the last `polylines` annotation, passes no
/// `source`: what remains is written at the lowest version it needs (1.1
/// §2.1, 1.2 §2.1).
#[pyfunction]
#[pyo3(signature = (source, profiles = Vec::new(), *, kinds = Vec::new()))]
fn written_version(source: Option<&str>, profiles: Vec<String>, kinds: Vec<String>) -> String {
    medh5::sample::writer::written_version_for_content(source, &profiles, &kinds)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SampleWriter>()?;
    m.add_function(wrap_pyfunction!(create, m)?)?;
    m.add_function(wrap_pyfunction!(amend, m)?)?;
    m.add_function(wrap_pyfunction!(written_version, m)?)?;
    let py = m.py();
    m.add("MANAGED_ROOT_ATTRS", PyTuple::new(py, medh5::sample::writer::MANAGED_ROOT_ATTRS)?)?;
    m.add("STANDARD_GROUPS", PyTuple::new(py, medh5::sample::writer::STANDARD_GROUPS)?)?;
    Ok(())
}
