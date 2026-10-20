//! `medh5.annotations`: payloads, the encoders, encoding selection and
//! transcoding (spec §6--§9).
//!
//! The encoders are exported for third-party converters: they validate
//! exactly as the writer does, because the writer calls the same engine
//! functions.

use std::collections::{BTreeMap, BTreeSet};

use ndarray::ArrayD;
use numpy::{PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PyString, PyTuple};
use serde_json::Value;

use medh5::annotations::encode as enc;
use medh5::annotations::encode_geometric as geo;
use medh5::annotations::payload::{Masks, Payload, PayloadData};
use medh5::annotations::select;
use medh5::array::DType;

use crate::convert::{
    array_to_py, attr_to_py, bool_array, detach_if, dtype_arg, dtype_to_py, f64_array, f64_vec, i64_array, i64_vec,
    json_to_py, nd_to_py, opt_strings, py_to_attr, py_to_json, py_to_nd, DETACH_ROWS,
};
use crate::errors::R;
use crate::exclusive::Exclusive;

// -- argument helpers -------------------------------------------------------------------

fn given<'py>(obj: Option<&Bound<'py, PyAny>>) -> Option<Bound<'py, PyAny>> {
    obj.filter(|o| !o.is_none()).cloned()
}

/// `int(obj)`, the conversion 1.x applied to every class id.
pub fn int_of(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(i);
    }
    obj.py().import("builtins")?.getattr("int")?.call1((obj,))?.extract::<i64>()
}

pub fn ints_of(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    // A 1-D NumPy array of integers that fit `i64` converts as an array: no
    // Python int per element, and a large copy runs without the GIL.
    if let Ok(array) = obj.cast::<PyUntypedArray>() {
        let dtype = array.dtype();
        let fits = dtype.kind() == b'i' || (dtype.kind() == b'u' && dtype.itemsize() < 8);
        if fits && array.ndim() == 1 {
            return i64_vec(obj);
        }
    }
    let items = if obj.hasattr("tolist")? { obj.call_method0("tolist")? } else { obj.clone() };
    if items.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err("class ids are integers, not a string"));
    }
    items.try_iter()?.map(|i| int_of(&i?)).collect()
}

fn u64s(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    let items = if obj.hasattr("tolist")? { obj.call_method0("tolist")? } else { obj.clone() };
    items.extract::<Vec<u64>>()
}

fn items_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let mut out = Vec::new();
    for item in obj.call_method0("items")?.try_iter()? {
        out.push(item?.extract()?);
    }
    Ok(out)
}

/// `{class_id: mask}` as boolean arrays (any dtype is coerced, as 1.x did).
pub fn masks_arg(obj: &Bound<'_, PyAny>) -> PyResult<Masks> {
    let mut out = Masks::new();
    for (k, v) in items_of(obj)? {
        out.insert(int_of(&k)?, bool_array(&v)?);
    }
    Ok(out)
}

pub fn masks_to_py<'py>(py: Python<'py>, masks: Masks) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (k, v) in masks {
        out.set_item(k, array_to_py(py, v))?;
    }
    Ok(out)
}

fn shape_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<usize>>> {
    match given(obj) {
        None => Ok(None),
        Some(o) => Ok(Some(i64_vec(&o)?.into_iter().map(|v| v.max(0) as usize).collect())),
    }
}

fn colouring_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<BTreeMap<i64, usize>>> {
    let Some(o) = given(obj) else { return Ok(None) };
    let mut out = BTreeMap::new();
    for (k, v) in items_of(&o)? {
        out.insert(int_of(&k)?, v.extract::<usize>()?);
    }
    Ok(Some(out))
}

/// The keyword options `encode_masks` and `transcode_payload` forward.
fn encode_options(kwargs: Option<&Bound<'_, PyDict>>) -> R<enc::EncodeOptions> {
    let mut options = enc::EncodeOptions::default();
    let Some(kw) = kwargs else { return Ok(options) };
    for (k, v) in kw.iter() {
        let name: String = k.extract()?;
        if v.is_none() {
            continue;
        }
        match name.as_str() {
            "ignore" => options.ignore = Some(bool_array(&v)?),
            "ignore_id" => options.ignore_id = Some(int_of(&v)?),
            "colouring" => options.colouring = colouring_arg(Some(&v))?,
            "dtype" => options.probmap_dtype = Some(dtype_arg(&v)?),
            "normalized" => options.normalized = v.is_truthy()?,
            "threshold" => options.threshold = Some(v.extract()?),
            "start_id" => options.start_id = Some(v.extract()?),
            "store_masks" => options.store_masks = Some(v.is_truthy()?),
            other => {
                return Err(PyTypeError::new_err(format!(
                    "unexpected keyword argument {}",
                    medh5::json::repr_str(other)
                ))
                .into())
            }
        }
    }
    Ok(options)
}

// -- AnnotationPayload ------------------------------------------------------------------------

/// Datasets and kind-specific attributes for one encoded annotation.
///
/// A mutable record, as 1.x's dataclass was: `datasets` and `attrs` are the
/// payload's own dicts, so an array edited in place, or a dataset or an
/// attribute added, is what the writer and the transcoder then read.  They
/// are handed out on first access --- the arrays move to NumPy uncopied ---
/// and read back when the payload is used.  Changed by one thread at a time
/// (E4); the dicts it hands out are Python's, shared as any dict is.
#[pyclass(module = "medh5.annotations.base", name = "AnnotationPayload", skip_from_py_object, frozen)]
pub struct AnnotationPayload {
    state: Exclusive<PayloadState>,
}

struct PayloadState {
    /// The kind, stacked axes and class ids; the datasets and attributes too,
    /// until they are handed out.
    inner: Payload,
    /// Once handed out, the payload's datasets (`inner`'s are then empty).
    datasets: Option<Py<PyDict>>,
    /// Once handed out, the payload's attributes (`inner`'s are then empty).
    attrs: Option<Py<PyDict>>,
}

impl AnnotationPayload {
    pub fn wrap(inner: Payload) -> Self {
        AnnotationPayload {
            state: Exclusive::new("an AnnotationPayload", PayloadState { inner, datasets: None, attrs: None }),
        }
    }

    /// The payload as it stands, for the engine.
    pub fn current(&self, py: Python<'_>) -> R<Payload> {
        // What the payload holds, taken under the lock: the dicts handed out
        // too, each taken whole in one step (another thread may be editing
        // it) while the lock still says they are the payload's.  A dict
        // replaced after, and edited by whoever kept it, is no longer read.
        // Their values are converted after the lock, since that runs NumPy
        // on what a caller put there.
        let (mut p, datasets, attrs) = {
            let state = self.state.read()?;
            let mut p = Payload::new(&state.inner.kind);
            p.stacked_axes = state.inner.stacked_axes;
            p.class_ids = state.inner.class_ids.clone();
            if state.datasets.is_none() {
                p.datasets = state.inner.datasets.clone();
            }
            if state.attrs.is_none() {
                p.attrs = state.inner.attrs.clone();
            }
            let handed =
                |d: &Option<Py<PyDict>>| d.as_ref().map(|d| crate::convert::dict_items(d.bind(py))).transpose();
            (p, handed(&state.datasets)?, handed(&state.attrs)?)
        };
        for (k, v) in datasets.into_iter().flatten() {
            p.datasets.insert(k.extract()?, data_arg(&v)?);
        }
        for (k, v) in attrs.into_iter().flatten() {
            p.attrs.push((k.extract()?, py_to_attr(&v)?));
        }
        Ok(p)
    }

    /// The datasets dict, handed out (and from then on the payload's).
    fn datasets_dict<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        if let Some(found) = &self.state.read()?.datasets {
            return Ok(found.bind(py).clone());
        }
        let mut state = self.state.write()?;
        if let Some(found) = &state.datasets {
            return Ok(found.bind(py).clone());
        }
        // Strings first, while nothing has moved: making them can fail.
        let mut strings = Vec::new();
        for (k, v) in &state.inner.datasets {
            if let PayloadData::Strings(values) = v {
                let list = PyList::new(py, values)?;
                strings.push((k.clone(), crate::convert::numpy(py)?.call_method1("array", (list, "O"))?));
            }
        }
        let out = PyDict::new(py);
        for (k, v) in std::mem::take(&mut state.inner.datasets) {
            match v {
                PayloadData::Array(a) => out.set_item(&k, nd_to_py(py, a))?,
                PayloadData::Strings(_) => {
                    let made = strings.iter().find(|(name, _)| *name == k).map(|(_, value)| value);
                    out.set_item(&k, made)?
                }
            }
        }
        state.datasets = Some(out.clone().unbind());
        Ok(out)
    }

    /// The attributes dict, handed out (and from then on the payload's).
    fn attrs_dict<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        if let Some(found) = &self.state.read()?.attrs {
            return Ok(found.bind(py).clone());
        }
        let mut state = self.state.write()?;
        if let Some(found) = &state.attrs {
            return Ok(found.bind(py).clone());
        }
        let out = PyDict::new(py);
        for (k, v) in &state.inner.attrs {
            out.set_item(k, attr_to_py(py, v)?)?;
        }
        state.inner.attrs.clear();
        state.attrs = Some(out.clone().unbind());
        Ok(out)
    }
}

/// A `datasets` or `attrs` value: the dict itself (as a dataclass field
/// holds what it is given), or any other mapping as a dict.
fn dict_arg<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    if let Ok(d) = value.cast::<PyDict>() {
        return Ok(d.clone());
    }
    let out = PyDict::new(value.py());
    out.update(value.cast::<pyo3::types::PyMapping>()?)?;
    Ok(out)
}

fn data_arg(obj: &Bound<'_, PyAny>) -> PyResult<PayloadData> {
    let np = crate::convert::numpy(obj.py())?;
    let array = np.call_method1("asarray", (obj,))?;
    let kind: String = array.getattr("dtype")?.getattr("kind")?.extract()?;
    if kind == "O" || kind == "U" || kind == "S" {
        let mut values = Vec::new();
        for item in array.call_method0("tolist")?.try_iter()? {
            let item = item?;
            values.push(match item.extract::<Vec<u8>>() {
                Ok(bytes) if item.is_instance_of::<pyo3::types::PyBytes>() => {
                    String::from_utf8_lossy(&bytes).into_owned()
                }
                _ => item.str()?.to_string(),
            });
        }
        return Ok(PayloadData::Strings(values));
    }
    Ok(PayloadData::Array(py_to_nd(&array)?))
}

/// A payload argument: this class, or anything with its attributes.
pub fn payload_arg(obj: &Bound<'_, PyAny>) -> R<Payload> {
    if let Ok(p) = obj.cast::<AnnotationPayload>() {
        return p.get().current(obj.py());
    }
    let mut p = Payload::new(&obj.getattr("kind")?.extract::<String>()?);
    if let Ok(datasets) = obj.getattr("datasets") {
        for (k, v) in items_of(&datasets)? {
            p.datasets.insert(k.extract()?, data_arg(&v)?);
        }
    }
    if let Ok(attrs) = obj.getattr("attrs") {
        for (k, v) in items_of(&attrs)? {
            p.attrs.push((k.extract()?, py_to_attr(&v)?));
        }
    }
    if let Ok(s) = obj.getattr("stacked_axes") {
        p.stacked_axes = s.extract()?;
    }
    if let Ok(c) = obj.getattr("class_ids") {
        p.class_ids = ints_of(&c)?;
    }
    Ok(p)
}

impl AnnotationPayload {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &["kind", "datasets", "attrs", "stacked_axes", "class_ids"];
}

#[pymethods]
impl AnnotationPayload {
    #[classattr]
    fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        crate::geometry::dataclass_fields(py, Self::FIELDS)
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::values::match_args(py, Self::FIELDS)
    }

    /// `copy.replace(obj, **changes)`.
    #[pyo3(signature = (**changes))]
    fn __replace__(slf: &Bound<'_, Self>, changes: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Py<PyAny>> {
        crate::values::dataclass_replace(slf.as_any(), changes)
    }

    #[new]
    #[pyo3(signature = (kind, datasets=None, attrs=None, stacked_axes=0, class_ids=None))]
    fn new(
        kind: &str,
        datasets: Option<&Bound<'_, PyAny>>,
        attrs: Option<&Bound<'_, PyAny>>,
        stacked_axes: usize,
        class_ids: Option<&Bound<'_, PyAny>>,
    ) -> R<Self> {
        let mut p = Payload::new(kind);
        if let Some(d) = given(datasets) {
            for (k, v) in items_of(&d)? {
                p.datasets.insert(k.extract()?, data_arg(&v)?);
            }
        }
        if let Some(a) = given(attrs) {
            for (k, v) in items_of(&a)? {
                p.attrs.push((k.extract()?, py_to_attr(&v)?));
            }
        }
        p.stacked_axes = stacked_axes;
        if let Some(c) = given(class_ids) {
            p.class_ids = ints_of(&c)?;
        }
        Ok(AnnotationPayload::wrap(p))
    }
    #[getter]
    fn kind(&self) -> R<String> {
        Ok(self.state.read()?.inner.kind.clone())
    }
    #[setter]
    fn set_kind(&self, kind: String) -> R<()> {
        self.state.write()?.inner.kind = kind;
        Ok(())
    }
    #[getter]
    fn datasets<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        self.datasets_dict(py)
    }
    #[setter]
    fn set_datasets(&self, datasets: &Bound<'_, PyAny>) -> R<()> {
        let datasets = dict_arg(datasets)?.unbind();
        let mut state = self.state.write()?;
        state.datasets = Some(datasets);
        state.inner.datasets.clear();
        Ok(())
    }
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        self.attrs_dict(py)
    }
    #[setter]
    fn set_attrs(&self, attrs: &Bound<'_, PyAny>) -> R<()> {
        let attrs = dict_arg(attrs)?.unbind();
        let mut state = self.state.write()?;
        state.attrs = Some(attrs);
        state.inner.attrs.clear();
        Ok(())
    }
    #[getter]
    fn stacked_axes(&self) -> R<usize> {
        Ok(self.state.read()?.inner.stacked_axes)
    }
    #[setter]
    fn set_stacked_axes(&self, stacked_axes: usize) -> R<()> {
        self.state.write()?.inner.stacked_axes = stacked_axes;
        Ok(())
    }
    #[getter]
    fn class_ids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        let ids = self.state.read()?.inner.class_ids.clone();
        Ok(PyTuple::new(py, ids)?)
    }
    #[setter]
    fn set_class_ids(&self, class_ids: &Bound<'_, PyAny>) -> R<()> {
        let ids = ints_of(class_ids)?;
        self.state.write()?.inner.class_ids = ids;
        Ok(())
    }
    /// The `data` dataset.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(self.datasets_dict(py)?.as_any().get_item("data")?)
    }
    /// Total bytes of every dataset.
    #[getter]
    fn nbytes(&self, py: Python<'_>) -> R<usize> {
        let handed = {
            let state = self.state.read()?;
            match &state.datasets {
                Some(found) => found.clone_ref(py),
                None => return Ok(state.inner.nbytes()),
            }
        };
        Ok(handed.bind(py).values().iter().map(|v| v.getattr("nbytes")?.extract::<usize>()).sum::<PyResult<_>>()?)
    }
    /// The kind, and each dataset's shape and dtype, for reports.
    fn describe<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.current(py)?.describe())?)
    }
    fn __repr__(&self, py: Python<'_>) -> R<String> {
        let (kind, class_ids, handed, names) = {
            let state = self.state.read()?;
            let handed = state.datasets.as_ref().map(|d| d.clone_ref(py));
            let names: Vec<String> = state.inner.datasets.keys().cloned().collect();
            (state.inner.kind.clone(), state.inner.class_ids.clone(), handed, names)
        };
        let names: Vec<String> = match handed {
            Some(found) => {
                found.bind(py).keys().iter().map(|k| k.str().map(|s| s.to_string())).collect::<PyResult<_>>()?
            }
            None => names,
        };
        Ok(format!(
            "AnnotationPayload(kind={}, datasets={}, class_ids={})",
            medh5::json::repr_str(&kind),
            medh5::json::repr_list(&names),
            medh5::json::repr_int_tuple(&class_ids)
        ))
    }
    /// Pickling and `copy`: the fields, as the constructor takes them.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_fields(slf.as_any(), None, Self::FIELDS)
    }
}

// -- AnnotationHeader -------------------------------------------------------------------------

const _: () = assert!(medh5::labels::IGNORE_ID == 65535, "AnnotationHeader's signature states the default ignore id");

/// The fixed attribute header every annotation carries (spec §6.2).
#[pyclass(module = "medh5.annotations.base", name = "AnnotationHeader", skip_from_py_object, frozen)]
#[derive(Clone)]
pub struct AnnotationHeader {
    pub inner: medh5::annotations::header::AnnotationHeader,
}

impl AnnotationHeader {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &[
        "kind",
        "task",
        "grid",
        "timepoints",
        "space",
        "frame_uid",
        "class_ids",
        "annotated_class_ids",
        "closure",
        "ignore_id",
        "ignore_mask",
        "prov",
        "quality",
        "derived_from",
        "extra",
    ];
}

#[pymethods]
impl AnnotationHeader {
    /// The header an annotation group carries (`sample.root["annotations/x"]`).
    #[classmethod]
    fn read(cls: &Bound<'_, pyo3::types::PyType>, group: &Bound<'_, PyAny>) -> R<Self> {
        let group = crate::integrity::group_of(group)?;
        let inner = crate::h5(cls.py(), || medh5::annotations::header::AnnotationHeader::read(&group))?;
        Ok(AnnotationHeader { inner })
    }

    #[classattr]
    fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        crate::geometry::dataclass_fields(py, Self::FIELDS)
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::values::match_args(py, Self::FIELDS)
    }

    /// `copy.replace(obj, **changes)`.
    #[pyo3(signature = (**changes))]
    fn __replace__(slf: &Bound<'_, Self>, changes: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Py<PyAny>> {
        crate::values::dataclass_replace(slf.as_any(), changes)
    }

    // `ignore_id`'s default is written out so that `inspect.signature`
    // shows it; the assertion before the class holds it to the engine's.
    #[new]
    #[pyo3(signature = (kind, task, grid=None, timepoints=None, space=None, frame_uid=None, class_ids=None,
        annotated_class_ids=None, closure="explicit", ignore_id=65535, ignore_mask=None,
        prov=None, quality=None, derived_from=None, extra=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kind: &str,
        task: &str,
        grid: Option<String>,
        timepoints: Option<&Bound<'_, PyAny>>,
        space: Option<String>,
        frame_uid: Option<String>,
        class_ids: Option<&Bound<'_, PyAny>>,
        annotated_class_ids: Option<&Bound<'_, PyAny>>,
        closure: &str,
        ignore_id: i64,
        ignore_mask: Option<String>,
        prov: Option<String>,
        quality: Option<String>,
        derived_from: Option<&Bound<'_, PyAny>>,
        extra: Option<&Bound<'_, PyAny>>,
    ) -> R<Self> {
        let mut h = medh5::annotations::header::AnnotationHeader::new(kind, task);
        h.grid = grid;
        h.timepoints = opt_strings(timepoints)?;
        h.space = space;
        h.frame_uid = frame_uid;
        h.class_ids = given(class_ids).map(|c| ints_of(&c)).transpose()?.unwrap_or_default();
        h.annotated_class_ids = given(annotated_class_ids).map(|c| ints_of(&c)).transpose()?.unwrap_or_default();
        h.closure = closure.to_string();
        h.ignore_id = ignore_id;
        h.ignore_mask = ignore_mask;
        h.prov = prov;
        h.quality = quality;
        h.derived_from = opt_strings(derived_from)?.unwrap_or_default();
        if let Some(e) = given(extra) {
            for (k, v) in items_of(&e)? {
                h.extra.push((k.extract()?, py_to_attr(&v)?));
            }
        }
        h.check()?;
        Ok(AnnotationHeader { inner: h })
    }
    #[getter]
    fn kind(&self) -> &str {
        &self.inner.kind
    }
    #[getter]
    fn task(&self) -> &str {
        &self.inner.task
    }
    #[getter]
    fn grid(&self) -> Option<&str> {
        self.inner.grid.as_deref()
    }
    #[getter]
    fn timepoints<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        self.inner.timepoints.as_ref().map(|t| PyTuple::new(py, t)).transpose()
    }
    #[getter]
    fn space(&self) -> Option<&str> {
        self.inner.space.as_deref()
    }
    #[getter]
    fn frame_uid(&self) -> Option<&str> {
        self.inner.frame_uid.as_deref()
    }
    #[getter]
    fn class_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.class_ids)
    }
    #[getter]
    fn annotated_class_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.annotated_class_ids)
    }
    #[getter]
    fn closure(&self) -> &str {
        &self.inner.closure
    }
    #[getter]
    fn ignore_id(&self) -> i64 {
        self.inner.ignore_id
    }
    #[getter]
    fn ignore_mask(&self) -> Option<&str> {
        self.inner.ignore_mask.as_deref()
    }
    #[getter]
    fn prov(&self) -> Option<&str> {
        self.inner.prov.as_deref()
    }
    #[getter]
    fn quality(&self) -> Option<&str> {
        self.inner.quality.as_deref()
    }
    #[getter]
    fn derived_from<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.derived_from)
    }
    #[getter]
    fn extra<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in &self.inner.extra {
            out.set_item(k, attr_to_py(py, v)?)?;
        }
        Ok(out)
    }
    /// The attributes this header writes.
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.inner.attrs() {
            out.set_item(k, attr_to_py(py, &v)?)?;
        }
        Ok(out)
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<AnnotationHeader>().map(|o| o.get().inner == self.inner).unwrap_or(false)
    }
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let names = [
            "kind",
            "task",
            "grid",
            "timepoints",
            "space",
            "frame_uid",
            "class_ids",
            "annotated_class_ids",
            "closure",
            "ignore_id",
            "ignore_mask",
            "prov",
            "quality",
            "derived_from",
            "extra",
        ];
        let fields = names.iter().map(|n| Ok((*n, slf.getattr(*n)?))).collect::<PyResult<Vec<_>>>()?;
        crate::values::dataclass_repr("AnnotationHeader", &fields)
    }
    /// Pickling and `copy`: the fields, as the constructor takes them.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_fields(slf.as_any(), None, Self::FIELDS)
    }
}

// -- OverlapStats and CostModel ---------------------------------------------------------------

/// Measured properties of a set of class masks, and what they imply (§7.6).
#[pyclass(module = "medh5.annotations.voxel", name = "OverlapStats", skip_from_py_object, frozen)]
#[derive(Clone)]
pub struct OverlapStats {
    pub inner: select::OverlapStats,
}

impl OverlapStats {
    pub fn wrap(inner: select::OverlapStats) -> Self {
        OverlapStats { inner }
    }
}

fn int_dict<'py, K: IntoPyObject<'py> + Copy, V: IntoPyObject<'py> + Copy>(
    py: Python<'py>,
    items: impl IntoIterator<Item = (K, V)>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (k, v) in items {
        out.set_item(k, v)?;
    }
    Ok(out)
}

impl OverlapStats {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] =
        &["class_ids", "spatial_shape", "counts", "edges", "colouring", "localized", "n_labelled_voxels"];
}

#[pymethods]
impl OverlapStats {
    #[classattr]
    fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        crate::geometry::dataclass_fields(py, Self::FIELDS)
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::values::match_args(py, Self::FIELDS)
    }

    #[getter]
    fn class_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.class_ids)
    }
    #[getter]
    fn spatial_shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.spatial_shape)
    }
    #[getter]
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        int_dict(py, self.inner.counts.iter().map(|(k, v)| (*k, *v)))
    }
    #[getter]
    fn edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        PyFrozenSet::new(py, self.inner.edges.iter().copied())
    }
    #[getter]
    fn colouring<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        int_dict(py, self.inner.colouring.iter().map(|(k, v)| (*k, *v)))
    }
    #[getter]
    fn localized<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        PyFrozenSet::new(py, self.inner.localized.iter().copied())
    }
    #[getter]
    fn n_labelled_voxels(&self) -> u64 {
        self.inner.n_labelled_voxels
    }
    #[getter]
    fn n_classes(&self) -> usize {
        self.inner.n_classes()
    }
    #[getter]
    fn n_voxels(&self) -> u64 {
        self.inner.n_voxels()
    }
    #[getter]
    fn total_foreground(&self) -> u64 {
        self.inner.total_foreground()
    }
    #[getter]
    fn fill(&self) -> f64 {
        self.inner.fill()
    }
    #[getter]
    fn depth(&self) -> f64 {
        self.inner.depth()
    }
    #[getter]
    fn n_layers(&self) -> usize {
        self.inner.n_layers()
    }
    #[getter]
    fn mean_degree(&self) -> f64 {
        self.inner.mean_degree()
    }
    #[getter]
    fn is_edgeless(&self) -> bool {
        self.inner.is_edgeless()
    }
    #[getter]
    fn n_planes(&self) -> usize {
        self.inner.n_planes()
    }
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.summary())
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<OverlapStats>().map(|o| o.get().inner == self.inner).unwrap_or(false)
    }
    fn __repr__(&self) -> String {
        format!(
            "OverlapStats(classes={}, voxels={}, fill={}, layers={}, planes={}, edges={})",
            self.inner.n_classes(),
            self.inner.n_voxels(),
            medh5::json::py_float(self.inner.fill()),
            self.inner.n_layers(),
            self.inner.n_planes(),
            self.inner.edges.len()
        )
    }
    /// Pickling and `copy`: the fields, through `_rebuild` (the engine alone
    /// measures one).
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_fields(slf.as_any(), Some("_rebuild"), Self::FIELDS)
    }
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    fn _rebuild(
        _cls: &Bound<'_, pyo3::types::PyType>,
        class_ids: Vec<i64>,
        spatial_shape: Vec<usize>,
        counts: BTreeMap<i64, u64>,
        edges: BTreeSet<(i64, i64)>,
        colouring: BTreeMap<i64, usize>,
        localized: BTreeSet<i64>,
        n_labelled_voxels: u64,
    ) -> Self {
        OverlapStats::wrap(select::OverlapStats {
            class_ids,
            spatial_shape,
            counts,
            edges,
            colouring,
            localized,
            n_labelled_voxels,
        })
    }
}

/// Raw (pre-compression) bytes per encoding.
#[pyclass(module = "medh5.annotations.voxel", name = "CostModel", skip_from_py_object, frozen)]
pub struct CostModel {
    pub inner: select::CostModel,
}

impl CostModel {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &["labelmap", "layers", "bitmask", "instances", "probmap", "detail"];
}

#[pymethods]
impl CostModel {
    #[classattr]
    fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        crate::geometry::dataclass_fields(py, Self::FIELDS)
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::values::match_args(py, Self::FIELDS)
    }

    #[getter]
    fn labelmap(&self) -> Option<u64> {
        self.inner.labelmap
    }
    #[getter]
    fn layers(&self) -> u64 {
        self.inner.layers
    }
    #[getter]
    fn bitmask(&self) -> u64 {
        self.inner.bitmask
    }
    #[getter]
    fn instances(&self) -> u64 {
        self.inner.instances
    }
    #[getter]
    fn probmap(&self) -> u64 {
        self.inner.probmap
    }
    #[getter]
    fn detail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.detail)
    }
    /// The cheapest encoding (`labelmap` only when it can hold the masks).
    fn best(&self) -> &'static str {
        self.inner.best()
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.to_json())
    }
    fn __repr__(&self) -> String {
        format!(
            "CostModel(labelmap={}, layers={}, bitmask={}, instances={}, probmap={})",
            self.inner.labelmap.map(|v| v.to_string()).unwrap_or_else(|| "None".into()),
            self.inner.layers,
            self.inner.bitmask,
            self.inner.instances,
            self.inner.probmap
        )
    }
    /// Pickling and `copy`: the fields, through `_rebuild` (the engine alone
    /// costs an encoding).
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_fields(slf.as_any(), Some("_rebuild"), Self::FIELDS)
    }
    #[classmethod]
    fn _rebuild(
        _cls: &Bound<'_, pyo3::types::PyType>,
        labelmap: Option<u64>,
        layers: u64,
        bitmask: u64,
        instances: u64,
        probmap: u64,
        detail: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let detail = py_to_json(detail)?;
        Ok(CostModel { inner: select::CostModel { labelmap, layers, bitmask, instances, probmap, detail } })
    }
}

// -- selection ------------------------------------------------------------------------------

/// Measure a set of class masks: counts, the overlap graph, a colouring and localization (§7.6).
///
/// `masks` maps class id to a boolean array, of `spatial_shape` when it is
/// given. The result, an `OverlapStats`, is what `select_encoding` and
/// `cost_model` decide by.
#[pyfunction]
#[pyo3(signature = (masks, spatial_shape=None))]
fn analyse(py: Python<'_>, masks: &Bound<'_, PyAny>, spatial_shape: Option<&Bound<'_, PyAny>>) -> R<OverlapStats> {
    let masks = masks_arg(masks)?;
    let shape = shape_arg(spatial_shape)?;
    Ok(OverlapStats::wrap(py.detach(|| select::analyse(&masks, shape.as_deref()))?))
}

/// Colour an overlap graph greedily, highest degree first: a dict of class id to layer (§7.6).
///
/// `edges` are `(a, b)` pairs of classes that overlap; two classes joined by an
/// edge never share a colour, so each colour can be one `layers` plane.
#[pyfunction]
fn greedy_colour<'py>(
    py: Python<'py>,
    class_ids: &Bound<'py, PyAny>,
    edges: &Bound<'py, PyAny>,
) -> R<Bound<'py, PyDict>> {
    let ids = ints_of(class_ids)?;
    let mut set = BTreeSet::new();
    for e in edges.try_iter()? {
        let (a, b): (i64, i64) = e?.extract()?;
        set.insert((a, b));
    }
    Ok(int_dict(py, select::greedy_colour(&ids, &set))?)
}

/// Group a colouring of class id to layer into one tuple of class ids per layer, ascending.
#[pyfunction]
fn layers_from_colouring<'py>(py: Python<'py>, colouring: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
    let map = colouring_arg(Some(colouring))?.unwrap_or_default();
    let layers = select::layers_from_colouring(&map);
    Ok(PyTuple::new(py, layers.into_iter().map(|l| PyTuple::new(py, l)).collect::<PyResult<Vec<_>>>()?)?)
}

/// Return the bytes per voxel of a labelmap for these class ids: 1 (`uint8`) or 2 (`uint16`).
///
/// 1 when every id is at most 254 and no ignore value is painted (`ignore`),
/// else 2 (§7.1).
#[pyfunction]
#[pyo3(signature = (class_ids, *, ignore=false))]
fn label_dtype_size(class_ids: &Bound<'_, PyAny>, ignore: bool) -> R<usize> {
    Ok(select::label_dtype_size(&ints_of(class_ids)?, ignore))
}

/// Return the raw, uncompressed bytes each voxel encoding would take for the analysed masks.
///
/// `stats` is an `analyse` result; `ignore` says an ignore value will be
/// painted. The `labelmap` cost is `None` when classes overlap, since a
/// labelmap cannot hold them (§7.6).
#[pyfunction]
#[pyo3(signature = (stats, *, ignore=false))]
fn cost_model(stats: &Bound<'_, OverlapStats>, ignore: bool) -> CostModel {
    CostModel { inner: select::cost_model(&stats.get().inner, ignore) }
}

/// Choose an encoding by measurement: `(kind, stats)`.
#[pyfunction]
#[pyo3(signature = (masks=None, spatial_shape=None, *, stats=None, soft=false, prefer=None, ignore=false))]
fn select_encoding<'py>(
    py: Python<'py>,
    masks: Option<&Bound<'py, PyAny>>,
    spatial_shape: Option<&Bound<'py, PyAny>>,
    stats: Option<&Bound<'py, OverlapStats>>,
    soft: bool,
    prefer: Option<&str>,
    ignore: bool,
) -> R<Bound<'py, PyTuple>> {
    let stats = match (stats, given(masks)) {
        (Some(s), _) => s.get().inner.clone(),
        (None, Some(m)) => {
            let masks = masks_arg(&m)?;
            let shape = shape_arg(spatial_shape)?;
            py.detach(|| select::analyse(&masks, shape.as_deref()))?
        }
        (None, None) => return Err(medh5::Error::invalid("select_encoding needs masks= or stats=").into()),
    };
    let kind = select::select_encoding_checked(&stats, soft, prefer, ignore)?;
    Ok(PyTuple::new(py, [PyString::new(py, &kind).into_any(), Bound::new(py, OverlapStats::wrap(stats))?.into_any()])?)
}

// -- voxel encoders ---------------------------------------------------------------------------

/// Pack mutually exclusive class masks into one integer volume (§7.1).
/// `ignore` is painted as `ignore_id`; a region that overlaps a class (E404:
/// §7.7 puts it in a sibling `mask`), a region of another shape (E405), or an
/// `ignore_id` outside 1..65535 or equal to a class id (E404) is refused.
#[pyfunction]
#[pyo3(signature = (masks, spatial_shape=None, *, ignore=None, ignore_id=medh5::labels::IGNORE_ID))]
fn encode_labelmap(
    py: Python<'_>,
    masks: &Bound<'_, PyAny>,
    spatial_shape: Option<&Bound<'_, PyAny>>,
    ignore: Option<&Bound<'_, PyAny>>,
    ignore_id: i64,
) -> R<AnnotationPayload> {
    let masks = masks_arg(masks)?;
    let shape = shape_arg(spatial_shape)?;
    let ignore = given(ignore).map(|i| bool_array(&i)).transpose()?;
    let payload = py.detach(|| enc::encode_labelmap(&masks, shape.as_deref(), ignore.as_ref(), ignore_id))?;
    Ok(AnnotationPayload::wrap(payload))
}

/// Colour the overlap graph and pack each colour into one labelmap (§7.2);
/// `ignore` and `ignore_id` are checked and painted as `encode_labelmap`
/// does.
#[pyfunction]
#[pyo3(signature = (masks, spatial_shape=None, *, colouring=None, ignore=None, ignore_id=medh5::labels::IGNORE_ID))]
fn encode_layers(
    py: Python<'_>,
    masks: &Bound<'_, PyAny>,
    spatial_shape: Option<&Bound<'_, PyAny>>,
    colouring: Option<&Bound<'_, PyAny>>,
    ignore: Option<&Bound<'_, PyAny>>,
    ignore_id: i64,
) -> R<AnnotationPayload> {
    let masks = masks_arg(masks)?;
    let shape = shape_arg(spatial_shape)?;
    let colouring = colouring_arg(colouring)?;
    let ignore = given(ignore).map(|i| bool_array(&i)).transpose()?;
    let payload =
        py.detach(|| enc::encode_layers(&masks, shape.as_deref(), colouring.as_ref(), ignore.as_ref(), ignore_id))?;
    Ok(AnnotationPayload::wrap(payload))
}

/// Pack class masks into `uint64` bitplanes, one bit per class, as a `bitmask` payload (§7.3).
///
/// `masks` maps class id to a boolean array of `spatial_shape`; `ceil(C / 64)`
/// planes are stored, least significant bit first.
#[pyfunction]
#[pyo3(signature = (masks, spatial_shape=None))]
fn encode_bitmask(
    py: Python<'_>,
    masks: &Bound<'_, PyAny>,
    spatial_shape: Option<&Bound<'_, PyAny>>,
) -> R<AnnotationPayload> {
    let masks = masks_arg(masks)?;
    let shape = shape_arg(spatial_shape)?;
    Ok(AnnotationPayload::wrap(py.detach(|| enc::encode_bitmask(&masks, shape.as_deref()))?))
}

/// Wrap one boolean volume as a `mask` payload.
#[pyfunction]
fn encode_mask(py: Python<'_>, mask: &Bound<'_, PyAny>) -> R<AnnotationPayload> {
    let mask = bool_array(mask)?;
    Ok(AnnotationPayload::wrap(py.detach(move || enc::encode_mask(mask))))
}

fn planes_arg(obj: &Bound<'_, PyAny>) -> PyResult<BTreeMap<i64, ArrayD<f64>>> {
    let mut out = BTreeMap::new();
    for (k, v) in items_of(obj)? {
        out.insert(int_of(&k)?, f64_array(&v)?);
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (probabilities, spatial_shape=None, *, dtype=None, normalized=false, threshold=None))]
fn encode_probmap(
    py: Python<'_>,
    probabilities: &Bound<'_, PyAny>,
    spatial_shape: Option<&Bound<'_, PyAny>>,
    dtype: Option<&Bound<'_, PyAny>>,
    normalized: bool,
    threshold: Option<f64>,
) -> R<AnnotationPayload> {
    let planes = planes_arg(probabilities)?;
    let shape = shape_arg(spatial_shape)?;
    let dtype = match given(dtype) {
        Some(d) => dtype_arg(&d)?,
        None => DType::F16,
    };
    let payload = py.detach(|| enc::encode_probmap(&planes, shape.as_deref(), dtype, normalized, threshold))?;
    Ok(AnnotationPayload::wrap(payload))
}

/// `values >= threshold`, decided in the stored precision (§7.5).
#[pyfunction]
fn contains_at<'py>(py: Python<'py>, values: &Bound<'py, PyAny>, threshold: f64) -> R<Bound<'py, PyAny>> {
    let values = py_to_nd(values)?;
    Ok(array_to_py(py, py.detach(|| enc::contains_at(&values, threshold))))
}

#[pyfunction]
#[pyo3(signature = (planes, threshold, requested=None))]
fn storage_dtype<'py>(
    py: Python<'py>,
    planes: &Bound<'py, PyAny>,
    threshold: f64,
    requested: Option<&Bound<'py, PyAny>>,
) -> R<Bound<'py, PyAny>> {
    let arrays: Vec<ArrayD<f64>> = if planes.hasattr("values")? && planes.hasattr("items")? {
        planes.call_method0("values")?.try_iter()?.map(|p| f64_array(&p?)).collect::<PyResult<_>>()?
    } else {
        planes.try_iter()?.map(|p| f64_array(&p?)).collect::<PyResult<_>>()?
    };
    let requested = match given(requested) {
        Some(d) => dtype_arg(&d)?,
        None => DType::F16,
    };
    Ok(dtype_to_py(py, enc::storage_dtype(&arrays, threshold, requested)?)?)
}

/// `uint32` unless an id needs the wider form (§7.4, §8.2).
#[pyfunction]
fn instance_id_dtype<'py>(py: Python<'py>, ids: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(dtype_to_py(py, enc::instance_id_dtype(&u64s(ids)?))?)
}

/// An `InstanceInput`-shaped object (attributes, as 1.x took them).
pub fn instance_input(item: &Bound<'_, PyAny>) -> PyResult<enc::InstanceInput> {
    let get = |name: &str| -> Option<Bound<'_, PyAny>> {
        match item.getattr(name) {
            Ok(v) if !v.is_none() => Some(v),
            _ => None,
        }
    };
    let class = get("class_id").ok_or_else(|| PyTypeError::new_err("an instance needs class_id"))?;
    let instance_id = get("instance_id").ok_or_else(|| PyTypeError::new_err("an instance needs instance_id"))?;
    Ok(enc::InstanceInput {
        class_id: int_of(&class)?,
        instance_id: instance_id.extract::<u64>()?,
        mask: get("mask").map(|m| bool_array(&m)).transpose()?,
        bbox: get("box").map(|b| f64_vec(&b)).transpose()?,
        crop: get("crop").map(|c| bool_array(&c)).transpose()?,
        score: get("score").map(|s| s.extract::<f64>()).transpose()?,
    })
}

#[pyfunction]
#[pyo3(signature = (objects, spatial_shape=None, *, store_masks=true, class_ids=None))]
fn encode_instances(
    py: Python<'_>,
    objects: &Bound<'_, PyAny>,
    spatial_shape: Option<&Bound<'_, PyAny>>,
    store_masks: bool,
    class_ids: Option<&Bound<'_, PyAny>>,
) -> R<AnnotationPayload> {
    let objects = objects.try_iter()?.map(|o| instance_input(&o?)).collect::<PyResult<Vec<_>>>()?;
    let shape = shape_arg(spatial_shape)?;
    let ids = given(class_ids).map(|c| ints_of(&c)).transpose()?;
    let payload = py.detach(|| enc::encode_instances(&objects, shape.as_deref(), store_masks, ids.as_deref()))?;
    Ok(AnnotationPayload::wrap(payload))
}

/// One object per class mask, ids from `start_id`: the field values of
/// `InstanceInput`s, as `(class_id, instance_id, mask)` triples.  An id past
/// the largest `uint64` is refused (E404).
#[pyfunction]
#[pyo3(signature = (masks, *, start_id=1))]
fn instances_from_masks<'py>(py: Python<'py>, masks: &Bound<'py, PyAny>, start_id: u64) -> R<Bound<'py, PyList>> {
    let masks = masks_arg(masks)?;
    let found = py.detach(|| enc::instances_from_masks(&masks, start_id))?;
    let out = PyList::empty(py);
    for i in found {
        let mask = match i.mask {
            Some(m) => array_to_py(py, m),
            None => py.None().into_bound(py),
        };
        out.append((i.class_id, i.instance_id, mask))?;
    }
    Ok(out)
}

/// Decode any voxel payload into a dict of class id to boolean mask.
///
/// A `probmap` is thresholded at `threshold` when given, else at its own
/// declared threshold. Decoding `instances` needs `spatial_shape` (E405), and a
/// `mask` decodes as class 1.
#[pyfunction]
#[pyo3(signature = (payload, *, spatial_shape=None, threshold=None))]
fn payload_to_masks<'py>(
    py: Python<'py>,
    payload: &Bound<'py, PyAny>,
    spatial_shape: Option<&Bound<'py, PyAny>>,
    threshold: Option<f64>,
) -> R<Bound<'py, PyDict>> {
    let payload = payload_arg(payload)?;
    let shape = shape_arg(spatial_shape)?;
    let masks = py.detach(|| enc::payload_to_masks(&payload, shape.as_deref(), threshold))?;
    Ok(masks_to_py(py, masks)?)
}

/// Encode class masks into the voxel encoding `kind` and return the payload.
///
/// `kind` is one of `labelmap`, `layers`, `bitmask`, `probmap`, `mask` or
/// `instances`. The keyword arguments are the encoders' own --- `ignore`,
/// `ignore_id`, `colouring`, `dtype`, `normalized`, `threshold`, `start_id`,
/// `store_masks` --- each used by the encodings that take it; another raises
/// `TypeError`.
#[pyfunction]
#[pyo3(signature = (masks, kind, spatial_shape=None, **kwargs))]
fn encode_masks(
    py: Python<'_>,
    masks: &Bound<'_, PyAny>,
    kind: &str,
    spatial_shape: Option<&Bound<'_, PyAny>>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> R<AnnotationPayload> {
    let masks = masks_arg(masks)?;
    let shape = shape_arg(spatial_shape)?;
    let options = encode_options(kwargs)?;
    Ok(AnnotationPayload::wrap(py.detach(|| enc::encode_masks(&masks, kind, shape.as_deref(), &options))?))
}

/// Encode class masks, choosing the encoding by measurement when asked to:
/// `(payload, stats)`.  An ignore region rides in band only under
/// `labelmap`/`layers`, and only where it meets no class; anything else is
/// refused (E404), because one payload cannot carry the §7.7 sibling mask.
#[pyfunction]
#[pyo3(signature = (masks, spatial_shape=None, *, encoding="auto", ignore=None, **kwargs))]
fn encode_voxels<'py>(
    py: Python<'py>,
    masks: &Bound<'py, PyAny>,
    spatial_shape: Option<&Bound<'py, PyAny>>,
    encoding: &str,
    ignore: Option<&Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> R<Bound<'py, PyTuple>> {
    let masks = masks_arg(masks)?;
    let shape = shape_arg(spatial_shape)?;
    let shape = medh5::annotations::payload::normalize_masks(&masks, shape.as_deref())?;
    let ignore = given(ignore).map(|i| bool_array(&i)).transpose()?;
    let stats = py.detach(|| select::analyse(&masks, Some(&shape)))?;
    let prefer = if encoding == "auto" { None } else { Some(encoding) };
    let kind = select::select_encoding_checked(&stats, false, prefer, ignore.is_some())?;
    let mut options = encode_options(kwargs)?;
    if let Some(region) = ignore {
        if !enc::IN_BAND_IGNORE_KINDS.contains(&kind.as_str()) {
            return Err(medh5::Error::coded(
                "E404",
                format!(
                    "encode_voxels: {} cannot hold an ignore region in band; §7.7 puts it in a separate `mask` \
                     annotation, which a single payload cannot carry. Use SampleWriter.add_segmentation, which writes \
                     the sibling mask, or choose 'labelmap' or 'layers'.",
                    medh5::json::repr_str(&kind)
                ),
            )
            .into());
        }
        if options.ignore.is_none() {
            options.ignore = Some(region);
        }
    }
    let payload = py.detach(|| enc::encode_masks(&masks, &kind, Some(&shape), &options))?;
    Ok(PyTuple::new(
        py,
        [
            Bound::new(py, AnnotationPayload::wrap(payload))?.into_any(),
            Bound::new(py, OverlapStats::wrap(stats))?.into_any(),
        ],
    )?)
}

/// Re-encode a payload; the same payload comes back when it already has
/// `to_kind`.
#[pyfunction]
#[pyo3(signature = (payload, to_kind, *, spatial_shape=None, threshold=None, drop_identity=false, **kwargs))]
fn transcode_payload<'py>(
    payload: &Bound<'py, PyAny>,
    to_kind: &str,
    spatial_shape: Option<&Bound<'py, PyAny>>,
    threshold: Option<f64>,
    drop_identity: bool,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> R<Bound<'py, PyAny>> {
    let py = payload.py();
    let source = payload_arg(payload)?;
    if source.kind == to_kind {
        enc::check_target(to_kind)?;
        return Ok(payload.clone());
    }
    let shape = shape_arg(spatial_shape)?;
    let options = encode_options(kwargs)?;
    let out =
        py.detach(|| enc::transcode_payload(&source, to_kind, shape.as_deref(), threshold, drop_identity, &options))?;
    Ok(Bound::new(py, AnnotationPayload::wrap(out))?.into_any())
}

/// Convert an open voxel annotation to another encoding (§7.6).
/// A `probmap` is thresholded at its `threshold` (§7.5): another encoding
/// keeps the voxels at or above it as the class, and none of the
/// probabilities.
#[pyfunction]
#[pyo3(signature = (annotation, to_kind, *, drop_identity=false))]
fn transcode(
    py: Python<'_>,
    annotation: &Bound<'_, PyAny>,
    to_kind: &str,
    drop_identity: bool,
) -> R<AnnotationPayload> {
    let handle = annotation_handle(annotation)?;
    let payload = py.detach(|| medh5::sample::writer_annotations::transcode(&handle, to_kind, drop_identity))?;
    Ok(AnnotationPayload::wrap(payload))
}

/// Every class of a voxel annotation as a full mask.
#[pyfunction]
#[pyo3(signature = (annotation, classes=None))]
fn annotation_to_masks<'py>(
    py: Python<'py>,
    annotation: &Bound<'py, PyAny>,
    classes: Option<&Bound<'py, PyAny>>,
) -> R<Bound<'py, PyDict>> {
    let handle = annotation_handle(annotation)?;
    let keys = given(classes).map(|c| crate::convert::class_keys(&c)).transpose()?;
    let masks = py.detach(|| medh5::sample::writer_annotations::annotation_to_masks(&handle, keys.as_deref()))?;
    Ok(masks_to_py(py, masks)?)
}

/// The engine annotation of an `AnnotationHandle` or a facade holding one.
pub fn annotation_handle(obj: &Bound<'_, PyAny>) -> PyResult<std::sync::Arc<medh5::annotations::Annotation>> {
    if let Ok(h) = obj.cast::<crate::reader::AnnotationHandle>() {
        return Ok(h.get().inner.clone());
    }
    let inner = obj.getattr("_handle")?;
    Ok(inner.cast::<crate::reader::AnnotationHandle>()?.get().inner.clone())
}

/// Whether two mask dicts hold the same classes with identical masks: the losslessness check.
#[pyfunction]
fn masks_equal(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> R<bool> {
    let (a, b) = (masks_arg(a)?, masks_arg(b)?);
    Ok(py.detach(|| enc::masks_equal(&a, &b)))
}

/// Whether transcoding a payload to `to_kind` keeps every class mask and its ignore region (§7.6).
///
/// The payload and its transcode are both decoded and compared, the in-band
/// ignore region (§7.7) included. A conversion `transcode_payload` refuses
/// raises as it does.
#[pyfunction]
#[pyo3(signature = (payload, to_kind, *, spatial_shape=None))]
fn check_roundtrip(
    py: Python<'_>,
    payload: &Bound<'_, PyAny>,
    to_kind: &str,
    spatial_shape: Option<&Bound<'_, PyAny>>,
) -> R<bool> {
    let payload = payload_arg(payload)?;
    let shape = shape_arg(spatial_shape)?;
    Ok(py.detach(|| enc::check_roundtrip(&payload, to_kind, shape.as_deref()))?)
}

#[pyfunction]
#[pyo3(signature = (masks, spatial_shape=None))]
fn normalize_masks<'py>(
    py: Python<'py>,
    masks: &Bound<'py, PyAny>,
    spatial_shape: Option<&Bound<'py, PyAny>>,
) -> R<Bound<'py, PyTuple>> {
    // `asarray(..., order="C")` returns a C-contiguous boolean array as it is,
    // and casts anything else (with the GIL released, as NumPy casts): the
    // shapes are checked on the arrays themselves, never on a copy.
    let np = crate::convert::numpy(py)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "bool")?;
    kwargs.set_item("order", "C")?;
    let mut given: BTreeMap<i64, Bound<'py, PyUntypedArray>> = BTreeMap::new();
    for (k, v) in items_of(masks)? {
        let class_id = int_of(&k)?;
        let array = np.call_method("asarray", (v,), Some(&kwargs))?;
        given.insert(class_id, array.cast_into::<PyUntypedArray>().map_err(PyErr::from)?);
    }
    let shape = shape_arg(spatial_shape)?;
    let found = medh5::annotations::payload::normalize_mask_shapes(
        given.iter().map(|(class_id, array)| (*class_id, array.shape())),
        shape.as_deref(),
    )?;
    let out = PyDict::new(py);
    for (class_id, array) in given {
        out.set_item(class_id, array)?;
    }
    Ok(PyTuple::new(py, [out.into_any(), PyTuple::new(py, found)?.into_any()])?)
}

/// Return `class_id` if it is an assignable class id, 1 to 65534 (§5.3).
///
/// 0 (background) and 65535 (ignore) raise `MEDH5ValidationError` (E303); an id
/// outside the range raises E306.
#[pyfunction]
fn check_class_id(class_id: &Bound<'_, PyAny>) -> R<i64> {
    Ok(i64::from(medh5::labels::check_class_id(int_of(class_id)?)?))
}

// -- geometric encoders -----------------------------------------------------------------------

struct Columns {
    class_ids: Vec<i64>,
    instance_ids: Option<Vec<u64>>,
    scores: Option<Vec<f64>>,
    attributes: Option<Vec<serde_json::Map<String, Value>>>,
}

impl Columns {
    fn of(
        class_ids: &Bound<'_, PyAny>,
        instance_ids: Option<&Bound<'_, PyAny>>,
        scores: Option<&Bound<'_, PyAny>>,
        attributes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Columns> {
        let attributes = match given(attributes) {
            None => None,
            Some(items) => Some(
                items
                    .try_iter()?
                    .map(|i| match py_to_json(&i?)? {
                        Value::Object(m) => Ok(m),
                        _ => Err(PyTypeError::new_err("each entry of attributes is a mapping")),
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            ),
        };
        Ok(Columns {
            class_ids: ints_of(class_ids)?,
            instance_ids: given(instance_ids).map(|i| u64s(&i)).transpose()?,
            scores: given(scores).map(|s| f64_vec(&s)).transpose()?,
            attributes,
        })
    }

    fn view(&self) -> geo::ObjectColumns<'_> {
        geo::ObjectColumns {
            class_ids: &self.class_ids,
            instance_ids: self.instance_ids.as_deref(),
            scores: self.scores.as_deref(),
            attributes: self.attributes.as_deref(),
        }
    }
}

/// Pack axis-aligned boxes, `(N, S, 2)` in `[lo, hi]` at voxel edges, as a `boxes` payload (§8.2).
///
/// `class_ids` has one id per box; `instance_ids`, `scores` and per-box
/// `attributes` dicts are optional columns. `slice_index` names one plane per
/// box, for 2-D boxes on a slice of a 3-D grid. A box with `lo > hi` or a bound
/// that is not finite raises E406, a column of the wrong length E405.
#[pyfunction]
#[pyo3(signature = (boxes, class_ids, *, instance_ids=None, scores=None, attributes=None, slice_index=None))]
fn encode_boxes(
    py: Python<'_>,
    boxes: &Bound<'_, PyAny>,
    class_ids: &Bound<'_, PyAny>,
    instance_ids: Option<&Bound<'_, PyAny>>,
    scores: Option<&Bound<'_, PyAny>>,
    attributes: Option<&Bound<'_, PyAny>>,
    slice_index: Option<&Bound<'_, PyAny>>,
) -> R<AnnotationPayload> {
    let boxes = f64_array(boxes)?;
    let cols = Columns::of(class_ids, instance_ids, scores, attributes)?;
    let n_boxes = boxes.shape().first().copied().unwrap_or(0);
    let planes = given(slice_index).map(|s| slice_index_arg(&s, n_boxes)).transpose()?;
    // `move`: the inputs are freed with the GIL released too, and freeing tens
    // of megabytes is milliseconds.
    let encoded =
        detach_if(py, n_boxes >= DETACH_ROWS, move || geo::encode_boxes(&boxes, &cols.view(), planes.as_deref()))?;
    Ok(AnnotationPayload::wrap(encoded))
}

/// Pack oriented boxes as centres, full edge lengths and rotation matrices, an `obb` payload (§8.3).
///
/// `centers` and `sizes` are `(N, S)` and `rotations` `(N, S, S)`, column *k*
/// the direction of the box's axis *k*. Sizes are full edge lengths, not
/// half-extents; each rotation must be proper (E407), and centres and sizes
/// finite (E406).
#[pyfunction]
#[pyo3(signature = (centers, sizes, rotations, class_ids, *, instance_ids=None, scores=None, attributes=None))]
#[allow(clippy::too_many_arguments)]
fn encode_obb(
    py: Python<'_>,
    centers: &Bound<'_, PyAny>,
    sizes: &Bound<'_, PyAny>,
    rotations: &Bound<'_, PyAny>,
    class_ids: &Bound<'_, PyAny>,
    instance_ids: Option<&Bound<'_, PyAny>>,
    scores: Option<&Bound<'_, PyAny>>,
    attributes: Option<&Bound<'_, PyAny>>,
) -> R<AnnotationPayload> {
    let cols = Columns::of(class_ids, instance_ids, scores, attributes)?;
    let (centers, sizes, rotations) = (f64_array(centers)?, f64_array(sizes)?, f64_array(rotations)?);
    let rows = centers.shape().first().copied().unwrap_or(0);
    let encoded =
        detach_if(py, rows >= DETACH_ROWS, move || geo::encode_obb(&centers, &sizes, &rotations, &cols.view()))?;
    Ok(AnnotationPayload::wrap(encoded))
}

/// Pack `(N, K, S)` keypoints of `N` objects as a `keypoints` payload (§8.4).
///
/// `keypoint_class_ids` gives the class of each of the `K` slots and
/// `class_ids` the class of each object. `visibility` is `(N, K)`: 0 not
/// labelled, 1 occluded, 2 visible. `skeleton` names a skeleton of the label
/// set (§5.5).
#[pyfunction]
#[pyo3(signature = (points, keypoint_class_ids, class_ids, *, visibility=None, instance_ids=None, scores=None, skeleton=None))]
#[allow(clippy::too_many_arguments)]
fn encode_keypoints(
    py: Python<'_>,
    points: &Bound<'_, PyAny>,
    keypoint_class_ids: &Bound<'_, PyAny>,
    class_ids: &Bound<'_, PyAny>,
    visibility: Option<&Bound<'_, PyAny>>,
    instance_ids: Option<&Bound<'_, PyAny>>,
    scores: Option<&Bound<'_, PyAny>>,
    skeleton: Option<&str>,
) -> R<AnnotationPayload> {
    let cols = Columns::of(class_ids, instance_ids, scores, None)?;
    let slots = ints_of(keypoint_class_ids)?;
    let visibility = given(visibility).map(|v| i64_array(&v)).transpose()?;
    let points = f64_array(points)?;
    let rows = points.shape().first().copied().unwrap_or(0) * slots.len().max(1);
    let encoded = detach_if(py, rows >= DETACH_ROWS, move || {
        geo::encode_keypoints(&points, &slots, &cols.view(), visibility.as_ref(), skeleton)
    })?;
    Ok(AnnotationPayload::wrap(encoded))
}

/// Pack an `(N, S)` point set --- landmarks or seeds --- as a `points` payload (§8.5).
///
/// The per-point columns are optional: `class_ids`, `instance_ids`, `names`,
/// `weights`. `correspondence` names the paired `points` annotation of a
/// landmark pair (§10.6).
#[pyfunction]
#[pyo3(signature = (points, *, class_ids=None, instance_ids=None, names=None, weights=None, correspondence=None))]
fn encode_points(
    py: Python<'_>,
    points: &Bound<'_, PyAny>,
    class_ids: Option<&Bound<'_, PyAny>>,
    instance_ids: Option<&Bound<'_, PyAny>>,
    names: Option<&Bound<'_, PyAny>>,
    weights: Option<&Bound<'_, PyAny>>,
    correspondence: Option<&str>,
) -> R<AnnotationPayload> {
    let ids = given(class_ids).map(|c| ints_of(&c)).transpose()?;
    let instance_ids = given(instance_ids).map(|i| u64s(&i)).transpose()?;
    let names = opt_strings(names)?;
    let weights = given(weights).map(|w| f64_vec(&w)).transpose()?;
    let points = f64_array(points)?;
    let rows = points.shape().first().copied().unwrap_or(0);
    let encoded = detach_if(py, rows >= DETACH_ROWS, move || {
        geo::encode_points(
            &points,
            ids.as_deref(),
            instance_ids.as_deref(),
            names.as_deref(),
            weights.as_deref(),
            correspondence,
        )
    })?;
    Ok(AnnotationPayload::wrap(encoded))
}

pub fn polygon_arg(item: &Bound<'_, PyAny>) -> R<geo::Polygon> {
    let vertices = f64_array(&item.getattr("vertices")?)?;
    let class_id = int_of(&item.getattr("class_id")?)?;
    let plane = match item.getattr("plane") {
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
    Ok(geo::Polygon::new(vertices, class_id, plane, &role)?)
}

/// Concatenate planar polygons into a `contours` payload with an offset table (§8.6).
///
/// Each polygon carries `vertices`, `class_id`, an optional `plane` `(axis,
/// index)` and a `role`, `outer` or `hole`. An empty list needs `ndim` to shape
/// the vertex table (E410).
#[pyfunction]
#[pyo3(signature = (polygons, *, ndim=None))]
fn encode_contours(py: Python<'_>, polygons: &Bound<'_, PyAny>, ndim: Option<usize>) -> R<AnnotationPayload> {
    // The polygons are Python objects, walked with the GIL; their packing is
    // not.
    let found = polygons.try_iter()?.map(|p| polygon_arg(&p?)).collect::<R<Vec<_>>>()?;
    let vertices: usize = found.iter().map(|p| p.vertices.shape().first().copied().unwrap_or(0)).sum();
    let encoded = detach_if(py, vertices >= DETACH_ROWS, move || geo::encode_contours(&found, ndim))?;
    Ok(AnnotationPayload::wrap(encoded))
}

/// A contour role, or E411.
#[pyfunction]
fn check_contour_role(role: &str) -> R<String> {
    if !geo::CONTOUR_ROLES.contains(&role) {
        return Err(medh5::Error::coded(
            "E411",
            format!(
                "contour role {} must be one of {}",
                medh5::json::repr_str(role),
                medh5::json::repr_list(&geo::CONTOUR_ROLES)
            ),
        )
        .into());
    }
    Ok(role.to_string())
}

/// Each of one class's polygons in one plane, its depth: how many of the
/// others enclose it (§8.6).  Each is `(V, 2)` in-plane coordinates.
#[pyfunction]
fn polygon_depths(polygons: &Bound<'_, PyAny>) -> R<Vec<usize>> {
    let mut found = Vec::new();
    for item in polygons.try_iter()? {
        let array = f64_array(&item?)?;
        let flat = array
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|_| medh5::Error::coded("E405", "a polygon's in-plane vertices are (V, 2)"))?;
        if flat.ncols() != 2 {
            return Err(medh5::Error::coded("E405", "a polygon's in-plane vertices are (V, 2)").into());
        }
        found.push(flat);
    }
    let views: Vec<_> = found.iter().map(|a| a.view()).collect();
    Ok(medh5::annotations::depth::polygon_depths(&views))
}

/// Pack a triangle surface mesh, `(V, 3)` vertices and `(F, 3)` faces, as a `mesh` payload (§8.7).
///
/// `normals` and `vertex_class_ids` are per vertex. Several meshes share one
/// payload through `mesh_offsets` (`M + 1` entries; mesh *m* is faces `[o[m],
/// o[m+1])`) and `mesh_class_ids`. A face indexing past the vertices raises
/// E405, offsets that do not delimit the faces E408.
#[pyfunction]
#[pyo3(signature = (vertices, faces, *, normals=None, vertex_class_ids=None, mesh_offsets=None, mesh_class_ids=None))]
fn encode_mesh(
    py: Python<'_>,
    vertices: &Bound<'_, PyAny>,
    faces: &Bound<'_, PyAny>,
    normals: Option<&Bound<'_, PyAny>>,
    vertex_class_ids: Option<&Bound<'_, PyAny>>,
    mesh_offsets: Option<&Bound<'_, PyAny>>,
    mesh_class_ids: Option<&Bound<'_, PyAny>>,
) -> R<AnnotationPayload> {
    let normals = given(normals).map(|n| f64_array(&n)).transpose()?;
    let vertex_ids = given(vertex_class_ids).map(|v| ints_of(&v)).transpose()?;
    let offsets = given(mesh_offsets).map(|o| i64_vec(&o)).transpose()?;
    let mesh_ids = given(mesh_class_ids).map(|m| ints_of(&m)).transpose()?;
    let (vertices, faces) = (f64_array(vertices)?, i64_array(faces)?);
    let payload = py.detach(|| {
        geo::encode_mesh(
            &vertices,
            &faces,
            normals.as_ref(),
            vertex_ids.as_deref(),
            offsets.as_deref(),
            mesh_ids.as_deref(),
        )
    })?;
    Ok(AnnotationPayload::wrap(payload))
}

/// A `slice_index` argument as planes: an array of exactly one plane per box
/// (§8.2), refused with E405 otherwise --- before flattening, because a column
/// of the right length can be the wrong shape.
pub fn slice_index_arg(obj: &Bound<'_, PyAny>, n_boxes: usize) -> R<Vec<i64>> {
    let array = crate::convert::i64_array(obj)?;
    if let Some(problem) = geo::check_slice_index_shape(array.shape(), n_boxes) {
        return Err(medh5::Error::coded("E405", problem).into());
    }
    Ok(array.iter().copied().collect())
}

/// Return what is wrong with a `slice_index` column of one plane per box, or `None` (§8.2).
///
/// The shape `(n_boxes,)` is checked always. With the index-space `boxes` and
/// the grid `shape`, each plane of a box flat on one axis is also checked to
/// lie inside the grid.
#[pyfunction]
#[pyo3(signature = (planes, n_boxes, *, boxes=None, shape=None))]
fn check_slice_index(
    planes: &Bound<'_, PyAny>,
    n_boxes: usize,
    boxes: Option<&Bound<'_, PyAny>>,
    shape: Option<&Bound<'_, PyAny>>,
) -> R<Option<String>> {
    let array = crate::convert::i64_array(planes)?;
    if let Some(problem) = geo::check_slice_index_shape(array.shape(), n_boxes) {
        return Ok(Some(problem));
    }
    let planes: Vec<i64> = array.iter().copied().collect();
    let rows: Option<Vec<Vec<f64>>> = match given(boxes) {
        None => None,
        Some(b) => {
            let array = f64_array(&b)?;
            let n = array.shape().first().copied().unwrap_or(0);
            Some((0..n).map(|i| array.index_axis(ndarray::Axis(0), i).iter().copied().collect()).collect())
        }
    };
    let shape = shape_arg(shape)?;
    Ok(geo::check_slice_index(&planes, n_boxes, rows.as_deref(), shape.as_deref()))
}

/// Return `space` if it is `index` or `world`, else raise E412 (§8.1).
#[pyfunction]
fn check_space(space: &str) -> R<String> {
    geo::check_space(space)?;
    Ok(space.to_string())
}

/// Return `scope` if it is a classification scope, else raise E412 (§9).
///
/// The scopes are `sample`, `timepoint`, `grid`, `roi`, `slice` and `instance`.
#[pyfunction]
fn check_scope(scope: &str) -> R<String> {
    geo::check_scope(scope)?;
    Ok(scope.to_string())
}

// -- classification -------------------------------------------------------------------------

/// `(classes, values, scope_ids, schemes, scheme_values)` from a mapping or
/// assertion rows (§9).
#[pyfunction]
fn assertion_rows<'py>(py: Python<'py>, labels: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
    let rows = crate::writer::assertion_rows(labels)?;
    let opt = |v: Option<Bound<'py, PyAny>>| v.unwrap_or_else(|| py.None().into_bound(py));
    Ok(PyTuple::new(
        py,
        [
            PyList::new(py, rows.classes)?.into_any(),
            PyList::new(py, rows.values)?.into_any(),
            opt(rows.scope_ids.map(|s| PyList::new(py, s)).transpose()?.map(Bound::into_any)),
            opt(rows.schemes.map(|s| PyList::new(py, s)).transpose()?.map(Bound::into_any)),
            opt(rows.scheme_values.map(|s| PyList::new(py, s)).transpose()?.map(Bound::into_any)),
        ],
    )?)
}

/// Pack label assertions as a `classification` payload over one `scope` (§9).
///
/// `labels` is a mapping of class to value, or rows `(class, value[, scope_id[,
/// scheme, scheme_value]])`; a value is in `[0, 1]`, 1.0 a hard positive and
/// 0.0 an explicit negative (E404 otherwise). `multilabel=False` allows at most
/// one positive class per scope unit, and that is checked here.
#[pyfunction]
#[pyo3(signature = (labels, *, scope="sample", multilabel=true, scope_ids=None, schemes=None, scheme_values=None))]
fn encode_classification(
    labels: &Bound<'_, PyAny>,
    scope: &str,
    multilabel: bool,
    scope_ids: Option<&Bound<'_, PyAny>>,
    schemes: Option<&Bound<'_, PyAny>>,
    scheme_values: Option<&Bound<'_, PyAny>>,
) -> R<AnnotationPayload> {
    geo::check_scope(scope)?;
    let rows = crate::writer::assertion_rows(labels)?;
    let class_ids = rows.classes.iter().map(int_of).collect::<PyResult<Vec<_>>>()?;
    let assertions = crate::writer::assertion_columns(rows, class_ids, scope_ids, schemes, scheme_values)?;
    Ok(AnnotationPayload::wrap(geo::encode_classification(&assertions, scope, multilabel)?))
}

// -- header ---------------------------------------------------------------------------------

#[pyfunction]
fn default_task_for_kind(kind: &str) -> Option<&'static str> {
    medh5::annotations::header::default_task_for_kind(kind)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AnnotationPayload>()?;
    m.add_class::<AnnotationHeader>()?;
    m.add_class::<OverlapStats>()?;
    m.add_class::<CostModel>()?;
    for f in [
        wrap_pyfunction!(analyse, m)?,
        wrap_pyfunction!(greedy_colour, m)?,
        wrap_pyfunction!(layers_from_colouring, m)?,
        wrap_pyfunction!(label_dtype_size, m)?,
        wrap_pyfunction!(cost_model, m)?,
        wrap_pyfunction!(select_encoding, m)?,
        wrap_pyfunction!(encode_labelmap, m)?,
        wrap_pyfunction!(encode_layers, m)?,
        wrap_pyfunction!(encode_bitmask, m)?,
        wrap_pyfunction!(encode_mask, m)?,
        wrap_pyfunction!(encode_probmap, m)?,
        wrap_pyfunction!(contains_at, m)?,
        wrap_pyfunction!(storage_dtype, m)?,
        wrap_pyfunction!(instance_id_dtype, m)?,
        wrap_pyfunction!(encode_instances, m)?,
        wrap_pyfunction!(instances_from_masks, m)?,
        wrap_pyfunction!(payload_to_masks, m)?,
        wrap_pyfunction!(encode_masks, m)?,
        wrap_pyfunction!(encode_voxels, m)?,
        wrap_pyfunction!(transcode_payload, m)?,
        wrap_pyfunction!(transcode, m)?,
        wrap_pyfunction!(annotation_to_masks, m)?,
        wrap_pyfunction!(masks_equal, m)?,
        wrap_pyfunction!(check_roundtrip, m)?,
        wrap_pyfunction!(normalize_masks, m)?,
        wrap_pyfunction!(check_class_id, m)?,
        wrap_pyfunction!(encode_boxes, m)?,
        wrap_pyfunction!(encode_obb, m)?,
        wrap_pyfunction!(encode_keypoints, m)?,
        wrap_pyfunction!(encode_points, m)?,
        wrap_pyfunction!(encode_contours, m)?,
        wrap_pyfunction!(check_contour_role, m)?,
        wrap_pyfunction!(polygon_depths, m)?,
        wrap_pyfunction!(encode_mesh, m)?,
        wrap_pyfunction!(check_slice_index, m)?,
        wrap_pyfunction!(check_space, m)?,
        wrap_pyfunction!(check_scope, m)?,
        wrap_pyfunction!(assertion_rows, m)?,
        wrap_pyfunction!(encode_classification, m)?,
        wrap_pyfunction!(default_task_for_kind, m)?,
    ] {
        m.add_function(f)?;
    }
    let py = m.py();
    use medh5::annotations::header as h;
    m.add("VOXEL_KINDS", PyTuple::new(py, h::VOXEL_KINDS)?)?;
    let latest = medh5::version::LATEST_VERSION;
    m.add("GEOMETRIC_KINDS", PyTuple::new(py, h::geometric_kinds(latest).unwrap_or_default())?)?;
    m.add("ANNOTATION_KINDS", PyTuple::new(py, h::kinds(latest).unwrap_or_default())?)?;
    m.add("RESERVED_KINDS", PyTuple::new(py, h::RESERVED_KINDS)?)?;
    m.add("TASKS", PyTuple::new(py, h::TASKS)?)?;
    m.add("SPEC_ANNOTATION_ATTRS", PyTuple::new(py, h::SPEC_ANNOTATION_ATTRS)?)?;
    m.add("BITS_PER_PLANE", enc::BITS_PER_PLANE)?;
    m.add("DEFAULT_THRESHOLD", enc::DEFAULT_THRESHOLD)?;
    m.add("TRANSCODABLE", PyTuple::new(py, enc::TRANSCODABLE)?)?;
    m.add("IN_BAND_IGNORE_KINDS", PyTuple::new(py, enc::IN_BAND_IGNORE_KINDS)?)?;
    m.add("SPACES", PyTuple::new(py, geo::SPACES)?)?;
    m.add("CONTOUR_ROLES", PyTuple::new(py, geo::CONTOUR_ROLES)?)?;
    m.add("SCOPES", PyTuple::new(py, geo::SCOPES)?)?;
    m.add("ROTATION_TOL", geo::ROTATION_TOL)?;
    let visibility = PyDict::new(py);
    for (k, v) in geo::VISIBILITY {
        visibility.set_item(k, v)?;
    }
    m.add("VISIBILITY", visibility)?;
    m.add("LOCALIZED_BBOX_FRACTION", select::LOCALIZED_BBOX_FRACTION)?;
    m.add("SPARSE_FILL", select::SPARSE_FILL)?;
    m.add("SLAB_BYTES", medh5::annotations::payload::SLAB_BYTES)?;
    Ok(())
}
