//! Values across the boundary: JSON documents, NumPy arrays, class keys,
//! regions of interest.

use std::path::PathBuf;

use ndarray::ArrayD;
use numpy::{PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PySlice, PyString, PyTuple};
use serde_json::{Map, Value};

use medh5::array::{DType, Index, NdArray, Slice};
use medh5::labels::ClassKey;

use crate::errors::R;

// -- modules --------------------------------------------------------------------------

pub fn numpy(py: Python<'_>) -> PyResult<&Bound<'_, PyModule>> {
    static NUMPY: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    Ok(NUMPY.get_or_try_init(py, || py.import("numpy").map(Bound::unbind))?.bind(py))
}

// -- JSON -----------------------------------------------------------------------------

/// A JSON document as Python objects: dicts keep their order, integers stay
/// `int`, every float is a `float`.
pub fn json_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any()
            } else {
                n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any()
            }
        }
        Value::String(s) => PyString::new(py, s).into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            dict.into_any()
        }
    })
}

fn key_text(key: &Bound<'_, PyAny>) -> PyResult<String> {
    // `json.dumps` keys: str as is; int, float, bool and None by their JSON spelling.
    if let Ok(s) = key.cast::<PyString>() {
        return Ok(s.to_string());
    }
    if key.is_none() {
        return Ok("null".into());
    }
    if let Ok(b) = key.cast::<PyBool>() {
        return Ok(if b.is_true() { "true" } else { "false" }.into());
    }
    if key.is_instance_of::<PyInt>() || key.is_instance_of::<PyFloat>() {
        return Ok(key.str()?.to_string());
    }
    if let Ok(i) = key.extract::<i64>() {
        return Ok(i.to_string());
    }
    Err(PyTypeError::new_err(format!("keys must be str, int, float, bool or None, not {}", key.get_type().name()?)))
}

/// A float as a JSON number --- which NaN and infinity are not.
///
/// 1.x let `json.dumps` write them as `NaN`/`Infinity`, which made `/meta`
/// something other than the JSON §2.4 requires; turning them into `null`
/// would change the data without a word.  So they are refused, and the caller
/// says what a missing value is (`None`).
fn finite(value: f64) -> PyResult<Value> {
    if value.is_finite() {
        return Ok(medh5::json::num(value));
    }
    Err(crate::errors::BindError::from(medh5::Error::invalid(format!(
        "{} is not a JSON number: JSON has no NaN or infinity (use None for a missing value)",
        medh5::json::py_float(value)
    )))
    .into())
}

/// What an integer beyond 64 bits becomes.
#[derive(Clone, Copy)]
enum Wide {
    /// Refused: a JSON integer is written in decimal, as itself (§5.1), and
    /// the engine holds one in 64 bits; rounding a larger one to a float would
    /// store another number, of another type, without a word.
    Refused,
    /// The float a JSON parser makes of its digits, for a document the engine
    /// then validates as it would the same JSON read from a file.
    Parsed,
}

/// An integer as a JSON number.
fn integer(obj: &Bound<'_, PyAny>, wide: Wide) -> PyResult<Value> {
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::from(i));
    }
    if let Ok(u) = obj.extract::<u64>() {
        return Ok(Value::from(u));
    }
    if let Wide::Parsed = wide {
        return finite(obj.extract::<f64>()?);
    }
    Err(crate::errors::BindError::from(medh5::Error::invalid(format!(
        "{} is outside the 64-bit integers medh5 stores exactly ({} to {}); store it as a string",
        obj.str()?,
        i64::MIN,
        u64::MAX
    )))
    .into())
}

/// `collections.abc.Sequence`: what a JSON array is made from, besides a
/// list or a tuple --- ordered, and the same on every pass.
fn is_sequence(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    static SEQUENCE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = obj.py();
    let abc = SEQUENCE
        .get_or_try_init(py, || py.import("collections.abc").and_then(|m| m.getattr("Sequence")).map(Bound::unbind))?;
    obj.is_instance(abc.bind(py))
}

/// A Python value as JSON, the way `json.dumps` would see it --- plus NumPy
/// scalars and arrays, which 1.x callers pass freely, mappings and sequences.
///
/// A set is refused, as `json.dumps` refuses it: its order is its hash order,
/// which varies from one interpreter to the next (`PYTHONHASHSEED`), so the
/// bytes written --- and the `content_id` --- would too.  So is an iterator,
/// which a write would consume, and an integer beyond 64 bits ([`Wide`]).
/// A dict's items as they stand, taken in one step (`dict.items()` is
/// atomic): iterating the dict itself panics if another thread resizes it
/// meanwhile, as one may on free-threaded CPython.
pub(crate) fn dict_items<'py>(dict: &Bound<'py, PyDict>) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    dict.items().iter().map(|item| item.extract()).collect()
}

pub fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    json_of(obj, Wide::Refused)
}

/// [`py_to_json`] for a document an engine contract validates as it would
/// the same JSON in a file --- a task manifest --- so that an integer beyond
/// 64 bits is reported by the contract's own rule (T101), not refused here.
pub fn py_to_json_as_parsed(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    json_of(obj, Wide::Parsed)
}

fn json_of(obj: &Bound<'_, PyAny>, wide: Wide) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::String(s.to_string()));
    }
    if obj.is_instance_of::<PyInt>() {
        return integer(obj, wide);
    }
    if let Ok(f) = obj.cast::<PyFloat>() {
        return finite(f.value());
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = Map::new();
        for (k, v) in dict_items(dict)? {
            map.insert(key_text(&k)?, json_of(&v, wide)?);
        }
        return Ok(Value::Object(map));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let mut items = Vec::new();
        for item in obj.try_iter()? {
            items.push(json_of(&item?, wide)?);
        }
        return Ok(Value::Array(items));
    }
    // NumPy scalars and arrays: through their Python equivalents.
    if obj.hasattr("dtype")? && obj.hasattr("tolist")? {
        return json_of(&obj.call_method0("tolist")?, wide);
    }
    if obj.hasattr("keys")? && obj.hasattr("__getitem__")? {
        let mut map = Map::new();
        for key in obj.call_method0("keys")?.try_iter()? {
            let key = key?;
            let value = obj.get_item(&key)?;
            map.insert(key_text(&key)?, json_of(&value, wide)?);
        }
        return Ok(Value::Object(map));
    }
    if obj.hasattr("to_json")? {
        return json_of(&obj.call_method0("to_json")?, wide);
    }
    // Numbers of other types: an integer through `__index__`, anything else
    // (`Decimal`, `Fraction`) through `__float__` --- finite, as a float is.
    if obj.hasattr("__index__")? {
        return integer(&obj.call_method0("__index__")?, wide);
    }
    if let Ok(f) = obj.extract::<f64>() {
        return finite(f);
    }
    let binary = obj.is_instance_of::<pyo3::types::PyBytes>()
        || obj.is_instance_of::<pyo3::types::PyByteArray>()
        || obj.is_instance_of::<pyo3::types::PyMemoryView>();
    if !binary && is_sequence(obj)? {
        let mut items = Vec::new();
        for item in obj.try_iter()? {
            items.push(json_of(&item?, wide)?);
        }
        return Ok(Value::Array(items));
    }
    Err(PyTypeError::new_err(format!("Object of type {} is not JSON serializable", obj.get_type().name()?)))
}

/// [`py_to_json`], with `str()` for a value JSON has no form for: for records
/// whose details are free-form and must always serialise.
pub fn py_to_json_lenient(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = Map::new();
        for (k, v) in dict_items(dict)? {
            map.insert(key_text(&k).unwrap_or(k.str()?.to_string()), py_to_json_lenient(&v)?);
        }
        return Ok(Value::Object(map));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let mut items = Vec::new();
        for item in obj.try_iter()? {
            items.push(py_to_json_lenient(&item?)?);
        }
        return Ok(Value::Array(items));
    }
    match py_to_json(obj) {
        Ok(v) => Ok(v),
        Err(_) => Ok(Value::String(obj.str()?.to_string())),
    }
}

/// A mapping of keyword fields as a JSON object.
pub fn kwargs_to_map(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Map<String, Value>> {
    let mut map = Map::new();
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            map.insert(k.extract::<String>()?, py_to_json(&v)?);
        }
    }
    Ok(map)
}

pub fn map_to_py<'py>(py: Python<'py>, map: &Map<String, Value>) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in map {
        dict.set_item(k, json_to_py(py, v)?)?;
    }
    Ok(dict)
}

// -- arrays ---------------------------------------------------------------------------

/// An engine array as a NumPy array (moved, not copied).
pub fn nd_to_py(py: Python<'_>, array: NdArray) -> Bound<'_, PyAny> {
    match array {
        NdArray::Bool(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::I8(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::U8(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::I16(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::U16(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::I32(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::U32(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::I64(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::U64(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::F16(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::F32(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
        NdArray::F64(a) => numpy::PyArray::from_owned_array(py, a).into_any(),
    }
}

pub fn array_to_py<T: numpy::Element>(py: Python<'_>, array: ArrayD<T>) -> Bound<'_, PyAny> {
    numpy::PyArray::from_owned_array(py, array).into_any()
}

/// `numpy.asarray(obj[, dtype])`, native byte order.
pub fn asarray<'py>(obj: &Bound<'py, PyAny>, dtype: Option<&str>) -> PyResult<Bound<'py, PyUntypedArray>> {
    let py = obj.py();
    let np = numpy(py)?;
    let kwargs = PyDict::new(py);
    if let Some(d) = dtype {
        kwargs.set_item("dtype", d)?;
    }
    let mut array = np.call_method("asarray", (obj,), Some(&kwargs))?;
    let byteorder: String = array.getattr("dtype")?.getattr("byteorder")?.extract()?;
    if byteorder == ">" || (byteorder == "<" && cfg!(target_endian = "big")) {
        let native = array.getattr("dtype")?.call_method1("newbyteorder", ("=",))?;
        array = array.call_method1("astype", (native,))?;
    }
    Ok(array.cast_into::<PyUntypedArray>()?)
}

/// Arguments of this many bytes or more are copied with the GIL released.
pub const DETACH_BYTES: usize = 1 << 20;

/// Calls of this many rows or more run with the GIL released, once their
/// arguments are converted: below it, releasing and taking the GIL back costs
/// more than the work.
pub const DETACH_ROWS: usize = 4096;

/// `f()`, with the GIL released when `large`.  On free-threaded CPython
/// always detached: there is no GIL to hand back, and a call that may wait
/// for HDF5's lock waits detached (`crate::h5`).
pub fn detach_if<T: Ungil, F: Ungil + FnOnce() -> T>(py: Python<'_>, large: bool, f: F) -> T {
    if large || cfg!(Py_GIL_DISABLED) {
        py.detach(f)
    } else {
        f()
    }
}

/// An owned copy in standard (C) layout, whatever the input's: HDF5 reads and
/// writes row-major buffers, and a Fortran-ordered, transposed or flipped
/// array --- nibabel's `get_fdata()`, `x.T`, `np.flip(x, 0)` --- is as valid
/// an input as any.
///
/// From [`DETACH_BYTES`] the copy runs with the GIL released, the read-only
/// borrow held: another Python thread writing the array during the call may
/// leave a torn copy, as NumPy's own GIL-free copies may, so an argument is not
/// to be changed while a call reads it.
fn typed<T: numpy::Element + Clone + Send + Sync>(array: &Bound<'_, PyUntypedArray>) -> PyResult<ArrayD<T>> {
    let typed = array.cast::<PyArrayDyn<T>>()?;
    let guard = typed.readonly();
    let view = guard.as_array();
    let large = view.len().saturating_mul(std::mem::size_of::<T>()) >= DETACH_BYTES;
    Ok(detach_if(array.py(), large, || view.as_standard_layout().into_owned()))
}

/// A NumPy array (or anything `numpy.asarray` accepts) as an engine array.
pub fn py_to_nd(obj: &Bound<'_, PyAny>) -> PyResult<NdArray> {
    let array = asarray(obj, None)?;
    untyped_to_nd(&array)
}

pub fn untyped_to_nd(array: &Bound<'_, PyUntypedArray>) -> PyResult<NdArray> {
    let dtype = array.dtype();
    let kind = dtype.kind() as char;
    let size = dtype.itemsize();
    Ok(match (kind, size) {
        ('b', 1) => NdArray::Bool(typed(array)?),
        ('i', 1) => NdArray::I8(typed(array)?),
        ('u', 1) => NdArray::U8(typed(array)?),
        ('i', 2) => NdArray::I16(typed(array)?),
        ('u', 2) => NdArray::U16(typed(array)?),
        ('i', 4) => NdArray::I32(typed(array)?),
        ('u', 4) => NdArray::U32(typed(array)?),
        ('i', 8) => NdArray::I64(typed(array)?),
        ('u', 8) => NdArray::U64(typed(array)?),
        ('f', 2) => NdArray::F16(typed(array)?),
        ('f', 4) => NdArray::F32(typed(array)?),
        ('f', 8) => NdArray::F64(typed(array)?),
        _ => {
            return Err(PyTypeError::new_err(format!(
                "arrays of dtype {} cannot be stored; use a boolean, integer or floating dtype",
                dtype.str()?
            )))
        }
    })
}

/// `numpy.asarray(obj, dtype=dtype)`, `T` being `dtype`'s element type.
pub fn typed_array<T: numpy::Element + Clone + Send + Sync>(
    obj: &Bound<'_, PyAny>,
    dtype: &str,
) -> PyResult<ArrayD<T>> {
    typed(&asarray(obj, Some(dtype))?)
}

/// `numpy.asarray(obj, dtype=bool)`.
pub fn bool_array(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<bool>> {
    typed(&asarray(obj, Some("bool"))?)
}

/// `numpy.asarray(obj, dtype=float64)`.
pub fn f64_array(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<f64>> {
    typed(&asarray(obj, Some("float64"))?)
}

/// `numpy.asarray(obj, dtype=int64)`.
pub fn i64_array(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<i64>> {
    typed(&asarray(obj, Some("int64"))?)
}

pub fn f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    Ok(into_vec(f64_array(obj)?))
}

pub fn i64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    Ok(into_vec(i64_array(obj)?))
}

/// The elements in order, taking the array's own buffer when it is laid out
/// as one, as `typed` returns it: walking a dynamic-rank array element by
/// element costs ~10 ms a million, with the GIL held.
fn into_vec<T: Clone>(array: ArrayD<T>) -> Vec<T> {
    if !array.is_standard_layout() {
        return array.iter().cloned().collect();
    }
    let n = array.len();
    // C order: the elements are the buffer's `n` from the offset on.
    let (mut data, offset) = array.into_raw_vec_and_offset();
    match offset.unwrap_or(0) {
        0 => {
            data.truncate(n);
            data
        }
        start => data[start..start + n].to_vec(),
    }
}

/// A 2-D float matrix.
pub fn matrix(obj: &Bound<'_, PyAny>) -> PyResult<ndarray::Array2<f64>> {
    let a = f64_array(obj)?;
    if a.ndim() != 2 {
        return Err(PyValueError::new_err(format!("expected a 2-D matrix, got shape {:?}", a.shape())));
    }
    a.into_dimensionality::<ndarray::Ix2>().map_err(|e| PyValueError::new_err(e.to_string()))
}

/// `numpy.dtype(obj)` as an engine dtype.
pub fn dtype_arg(obj: &Bound<'_, PyAny>) -> R<DType> {
    let np = numpy(obj.py())?;
    let text: String = np.call_method1("dtype", (obj,))?.getattr("str")?.extract()?;
    Ok(DType::parse(&text)?)
}

pub fn dtype_to_py(py: Python<'_>, dtype: DType) -> PyResult<Bound<'_, PyAny>> {
    numpy(py)?.call_method1("dtype", (dtype.name(),))
}

// -- keys, regions, paths ----------------------------------------------------------------

/// A class given by id (`int`, NumPy integer) or by key (`str`).
pub fn class_key(obj: &Bound<'_, PyAny>) -> PyResult<ClassKey> {
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(ClassKey::Key(s.to_string()));
    }
    if obj.is_instance_of::<PyBool>() {
        return Ok(ClassKey::Id(i64::from(obj.extract::<bool>()?)));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(ClassKey::Id(i));
    }
    if let Ok(id) = obj.getattr("id") {
        if let Ok(i) = id.extract::<i64>() {
            return Ok(ClassKey::Id(i));
        }
    }
    Err(PyTypeError::new_err(format!("a class is an int id or a str key, not {}", obj.get_type().name()?)))
}

pub fn class_keys(obj: &Bound<'_, PyAny>) -> PyResult<Vec<ClassKey>> {
    if obj.is_instance_of::<PyString>() {
        return Ok(vec![class_key(obj)?]);
    }
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(class_key(&item?)?);
    }
    Ok(out)
}

pub fn class_ids(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(item?.extract::<i64>()?);
    }
    Ok(out)
}

pub fn strings(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(vec![s.to_string()]);
    }
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(item?.str()?.to_string());
    }
    Ok(out)
}

pub fn opt_strings(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<String>>> {
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => strings(o).map(Some),
    }
}

fn slice_bound(value: Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(value.extract::<i64>()?))
    }
}

/// One element of a region: `slice` or an integer position.
pub fn index_item(obj: &Bound<'_, PyAny>) -> PyResult<Index> {
    if let Ok(s) = obj.cast::<PySlice>() {
        return Ok(Index::Slice(Slice {
            start: slice_bound(s.getattr("start")?)?,
            stop: slice_bound(s.getattr("stop")?)?,
            step: slice_bound(s.getattr("step")?)?,
        }));
    }
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err("a region is slices or integer positions, not bools"));
    }
    Ok(Index::At(obj.extract::<i64>()?))
}

/// A region of interest as a list of index items; `None` stays `None`.
pub fn region(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Index>>> {
    let Some(obj) = obj else { return Ok(None) };
    if obj.is_none() {
        return Ok(None);
    }
    if obj.is_instance_of::<PySlice>() {
        return Ok(Some(vec![index_item(obj)?]));
    }
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(index_item(&item?)?);
    }
    Ok(Some(out))
}

/// [`region`] over an array of `ndim` axes, where a single Ellipsis stands for
/// as many whole axes as the other items leave, as NumPy reads it; two are an
/// `IndexError`.
pub fn region_for(obj: Option<&Bound<'_, PyAny>>, ndim: usize) -> PyResult<Option<Vec<Index>>> {
    let Some(obj) = obj else { return Ok(None) };
    let ellipsis = pyo3::types::PyEllipsis::get(obj.py());
    if obj.is(ellipsis) {
        return Ok(Some(vec![Index::Slice(Slice::full()); ndim]));
    }
    if obj.is_none() || obj.is_instance_of::<PySlice>() {
        return region(Some(obj));
    }
    let items: Vec<Bound<'_, PyAny>> = obj.try_iter()?.collect::<PyResult<_>>()?;
    match items.iter().filter(|i| i.is(ellipsis)).count() {
        0 => region(Some(obj)),
        1 => {
            let fill = (ndim + 1).saturating_sub(items.len());
            let mut out = Vec::with_capacity(ndim);
            for item in &items {
                if item.is(ellipsis) {
                    out.extend(std::iter::repeat_n(Index::Slice(Slice::full()), fill));
                } else {
                    out.push(index_item(item)?);
                }
            }
            Ok(Some(out))
        }
        _ => Err(pyo3::exceptions::PyIndexError::new_err("an index can only have a single ellipsis ('...')")),
    }
}

/// A region made of slices only (the engine APIs that crop), every axis kept.
///
/// An integer position is refused rather than read as a one-wide slice: NumPy
/// drops the axis an integer indexes and counts a negative one from the end,
/// and a position past the end is an `IndexError` there, not an empty read.
pub fn slices(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<Slice>>> {
    let Some(items) = region(obj)? else { return Ok(None) };
    items
        .into_iter()
        .map(|i| match i {
            Index::Slice(s) => Ok(s),
            Index::At(v) => {
                let one = match v.checked_add(1) {
                    Some(next) if next != 0 => format!("slice({v}, {next})"),
                    _ => format!("slice({v}, None)"),
                };
                Err(PyTypeError::new_err(format!(
                    "a region of interest here is one slice per spatial axis, not the position {v}; \
                     {one} reads that one position and keeps its axis"
                )))
            }
        })
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

pub fn path(obj: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
    obj.extract::<PathBuf>()
}

// -- HDF5 attribute values ------------------------------------------------------------------

use medh5::h5::attrs::AttrValue;

/// An attribute value as h5py would hand it back (scalars as Python
/// scalars, arrays as NumPy arrays, strings as `str`).
pub fn attr_to_py<'py>(py: Python<'py>, value: &AttrValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        AttrValue::Str(s) => PyString::new(py, s).into_any(),
        AttrValue::Strs(items) => PyList::new(py, items)?.into_any(),
        AttrValue::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        AttrValue::Int(i) => i.into_pyobject(py)?.into_any(),
        AttrValue::Float(f) => f.into_pyobject(py)?.into_any(),
        AttrValue::Array(a) => nd_to_py(py, a.clone()),
        AttrValue::Unsupported(d) => PyString::new(py, d).into_any(),
    })
}

/// A Python value as an attribute (spec §2.5): strings as UTF-8 text, string
/// sequences as string arrays (never a JSON blob), scalars with an explicit
/// width (`int64`, `float64`, `bool`), numeric sequences as `int64`, `float64`
/// or `bool` arrays, arrays as they are.  Anything else is refused.
pub fn py_to_attr(obj: &Bound<'_, PyAny>) -> PyResult<AttrValue> {
    let np = numpy(obj.py())?;
    let is = |name: &str| -> PyResult<bool> { obj.is_instance(&np.getattr(name)?) };
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(AttrValue::Str(s.to_string()));
    }
    if obj.is_instance_of::<PyBool>() || is("bool_")? {
        return Ok(AttrValue::Bool(obj.is_truthy()?));
    }
    if obj.is_instance_of::<PyInt>() || is("integer")? {
        return Ok(AttrValue::Int(obj.extract()?));
    }
    if obj.is_instance_of::<PyFloat>() || is("floating")? {
        return Ok(AttrValue::Float(obj.extract()?));
    }
    // A fixed-length string as h5py reads one back: text, not code points.
    if obj.is_instance_of::<pyo3::types::PyBytes>() || is("bytes_")? {
        let bytes: Vec<u8> = obj.extract()?;
        return Ok(AttrValue::Str(String::from_utf8_lossy(&bytes).into_owned()));
    }
    if is("ndarray")? {
        let kind: String = obj.getattr("dtype")?.getattr("kind")?.extract()?;
        if matches!(kind.as_str(), "U" | "S" | "O") && obj.getattr("ndim")?.extract::<usize>()? == 1 {
            return py_to_attr(&obj.call_method0("tolist")?);
        }
        return Ok(AttrValue::Array(py_to_nd(obj)?));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let items: Vec<Bound<'_, PyAny>> = obj.try_iter()?.collect::<PyResult<_>>()?;
        if items.is_empty() {
            return Ok(AttrValue::ints(&[]));
        }
        let all = |test: &dyn Fn(&Bound<'_, PyAny>) -> PyResult<bool>| -> PyResult<bool> {
            for item in &items {
                if !test(item)? {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        if all(&|i| Ok(i.is_instance_of::<PyString>()))? {
            return Ok(AttrValue::Strs(items.iter().map(|i| i.to_string()).collect()));
        }
        let boolean = |i: &Bound<'_, PyAny>| -> PyResult<bool> {
            Ok(i.is_instance_of::<PyBool>() || i.is_instance(&np.getattr("bool_")?)?)
        };
        let integer = |i: &Bound<'_, PyAny>| -> PyResult<bool> {
            Ok(!boolean(i)? && (i.is_instance_of::<PyInt>() || i.is_instance(&np.getattr("integer")?)?))
        };
        let real = |i: &Bound<'_, PyAny>| -> PyResult<bool> {
            Ok(integer(i)? || i.is_instance_of::<PyFloat>() || i.is_instance(&np.getattr("floating")?)?)
        };
        if all(&boolean)? {
            return Ok(AttrValue::Array(NdArray::from(bool_array(obj)?)));
        }
        if all(&integer)? {
            return Ok(AttrValue::Array(NdArray::from(i64_array(obj)?)));
        }
        if all(&real)? {
            return Ok(AttrValue::Array(NdArray::from(f64_array(obj)?)));
        }
        return Ok(AttrValue::Array(py_to_nd(obj)?));
    }
    Err(crate::errors::BindError::from(medh5::Error::invalid(format!(
        "cannot encode attribute value of type {}",
        obj.get_type().repr()?
    )))
    .into())
}
