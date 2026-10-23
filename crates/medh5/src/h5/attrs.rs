//! Attribute encoding (spec §2.5).
//!
//! | Logical type | HDF5 encoding |
//! |---|---|
//! | string | variable-length UTF-8 scalar |
//! | string list | 1-D array of variable-length UTF-8, never a JSON string |
//! | boolean | `int8` enum `FALSE=0, TRUE=1` (NumPy `bool_`) |
//! | integer / list | `int64` scalar / 1-D `int64` |
//! | float / list | `float64` scalar / 1-D `float64` |
//! | matrix | 2-D array, stored 2-D |
//!
//! Readers accept fixed-length as well as variable-length strings, ASCII as
//! well as UTF-8, and normalise to `String`.

use hdf5::types::{TypeDescriptor as TD, VarLenUnicode};
use ndarray::{ArrayD, IxDyn};
use serde_json::Value;

use crate::array::{DType, NdArray};
use crate::json::{num, repr_str};
use crate::{with_array, with_dtype, Error, Result};

/// One attribute value.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    /// A scalar string.
    Str(String),
    /// An array of strings: a 1-D attribute's, or an n-D one's flattened in
    /// C order (the shape is not kept).
    Strs(Vec<String>),
    /// A scalar boolean.
    Bool(bool),
    /// A scalar integer (written `int64`).
    Int(i64),
    /// A scalar float (written `float64`).
    Float(f64),
    /// A numeric or boolean array of any dtype and shape.
    Array(NdArray),
    /// A type this codec does not decode (compound, reference, ...).
    Unsupported(String),
}

impl AttrValue {
    /// A 1-D `int64` array.
    pub fn ints(values: &[i64]) -> AttrValue {
        AttrValue::Array(NdArray::from(ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).unwrap()))
    }

    /// A 1-D `float64` array.
    pub fn floats(values: &[f64]) -> AttrValue {
        AttrValue::Array(NdArray::from(ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).unwrap()))
    }

    /// A 2-D `float64` matrix.
    pub fn matrix(rows: usize, cols: usize, values: &[f64]) -> AttrValue {
        AttrValue::Array(NdArray::from(ArrayD::from_shape_vec(IxDyn(&[rows, cols]), values.to_vec()).unwrap()))
    }

    /// A string list.
    pub fn strs<S: AsRef<str>>(values: &[S]) -> AttrValue {
        AttrValue::Strs(values.iter().map(|s| s.as_ref().to_string()).collect())
    }

    /// The value as a string (spec §2.5: readers accept bytes and str).
    pub fn as_str(&self) -> Option<String> {
        match self {
            AttrValue::Str(s) => Some(s.clone()),
            AttrValue::Strs(v) if v.len() == 1 => Some(v[0].clone()),
            _ => None,
        }
    }

    /// The value as a string list; a scalar string is a list of one.
    pub fn as_str_list(&self) -> Option<Vec<String>> {
        match self {
            AttrValue::Str(s) => Some(vec![s.clone()]),
            AttrValue::Strs(v) => Some(v.clone()),
            _ => None,
        }
    }

    /// Every element as `f64` (scalars are lists of one).
    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            AttrValue::Bool(b) => Some(vec![if *b { 1.0 } else { 0.0 }]),
            AttrValue::Int(i) => Some(vec![*i as f64]),
            AttrValue::Float(f) => Some(vec![*f]),
            AttrValue::Array(a) => Some(a.to_f64().iter().copied().collect()),
            _ => None,
        }
    }

    /// Every element as `i64` (floats truncated; scalars are lists of one).
    pub fn as_i64_vec(&self) -> Option<Vec<i64>> {
        match self {
            AttrValue::Bool(b) => Some(vec![i64::from(*b)]),
            AttrValue::Int(i) => Some(vec![*i]),
            AttrValue::Float(f) => Some(vec![*f as i64]),
            AttrValue::Array(a) => Some(a.cast::<i64>().iter().copied().collect()),
            _ => None,
        }
    }

    /// The value as one integer.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            AttrValue::Array(a) if a.len() == 1 => self.as_i64_vec().map(|v| v[0]),
            AttrValue::Array(_) => None,
            _ => self.as_i64_vec().and_then(|v| v.first().copied()),
        }
    }

    /// The value as one float.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            AttrValue::Array(a) if a.len() == 1 => self.as_f64_vec().map(|v| v[0]),
            AttrValue::Array(_) => None,
            _ => self.as_f64_vec().and_then(|v| v.first().copied()),
        }
    }

    /// The value as one boolean (`bool(np.asarray(v).reshape(()).item())`).
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            AttrValue::Bool(b) => Some(*b),
            AttrValue::Int(i) => Some(*i != 0),
            AttrValue::Float(f) => Some(*f != 0.0),
            AttrValue::Array(a) if a.len() == 1 => Some(a.to_f64().iter().next().copied().unwrap_or(0.0) != 0.0),
            AttrValue::Str(s) => Some(!s.is_empty()),
            _ => None,
        }
    }

    /// The value as a 2-D float matrix `(rows, cols, row-major values)`.
    pub fn as_matrix(&self) -> Option<(usize, usize, Vec<f64>)> {
        match self {
            AttrValue::Array(a) if a.ndim() == 2 => {
                let shape = a.shape();
                Some((shape[0], shape[1], a.to_f64().iter().copied().collect()))
            }
            _ => None,
        }
    }

    /// The shape of the stored value (`[]` for scalars).
    pub fn shape(&self) -> Vec<usize> {
        match self {
            AttrValue::Strs(v) => vec![v.len()],
            AttrValue::Array(a) => a.shape(),
            _ => Vec::new(),
        }
    }

    /// The value as JSON, as the 1.x reference serialized attributes for
    /// `canonical_attrs` (§13.2): arrays as nested lists, scalars as items.
    pub fn to_json(&self) -> Value {
        match self {
            AttrValue::Str(s) => Value::String(s.clone()),
            AttrValue::Strs(v) => Value::Array(v.iter().map(|s| Value::String(s.clone())).collect()),
            AttrValue::Bool(b) => Value::Bool(*b),
            AttrValue::Int(i) => Value::from(*i),
            AttrValue::Float(f) => num(*f),
            AttrValue::Array(a) => array_to_json(a),
            AttrValue::Unsupported(d) => Value::String(d.clone()),
        }
    }

    /// The value in canonical JSON (§5.1), as `canonical_attrs` hashes it.
    ///
    /// Written from the value rather than through [`to_json`](Self::to_json):
    /// a JSON value has no NaN or infinity, and an attribute can hold one ---
    /// 1.x hashed it as `NaN`/`Infinity`, so a `content_id` stamped then
    /// depends on spelling it the same way.
    pub fn write_canonical(&self, out: &mut String) {
        fn float(out: &mut String, value: f64) {
            out.push_str(&crate::json::float_repr(value));
        }
        fn list<T>(out: &mut String, items: impl IntoIterator<Item = T>, mut each: impl FnMut(&mut String, T)) {
            out.push('[');
            for (i, item) in items.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                each(out, item);
            }
            out.push(']');
        }
        fn nested(out: &mut String, shape: &[usize], offset: usize, flat: &dyn Fn(&mut String, usize)) {
            match shape.split_first() {
                None => flat(out, offset),
                Some((&n, inner)) => {
                    let stride: usize = inner.iter().product();
                    list(out, 0..n, |out, i| nested(out, inner, offset + i * stride, flat));
                }
            }
        }
        match self {
            AttrValue::Str(s) | AttrValue::Unsupported(s) => crate::json::write_string(out, s, false),
            AttrValue::Strs(v) => list(out, v, |out, s| crate::json::write_string(out, s, false)),
            AttrValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            AttrValue::Int(i) => out.push_str(&i.to_string()),
            AttrValue::Float(f) => float(out, *f),
            AttrValue::Array(array) => {
                let shape = array.shape();
                match array {
                    NdArray::Bool(a) => {
                        let flat: Vec<bool> = a.iter().copied().collect();
                        nested(out, &shape, 0, &|out, i| out.push_str(if flat[i] { "true" } else { "false" }));
                    }
                    NdArray::F16(_) | NdArray::F32(_) | NdArray::F64(_) => {
                        let flat: Vec<f64> = array.to_f64().iter().copied().collect();
                        nested(out, &shape, 0, &|out, i| float(out, flat[i]));
                    }
                    NdArray::U64(a) => {
                        let flat: Vec<u64> = a.iter().copied().collect();
                        nested(out, &shape, 0, &|out, i| out.push_str(&flat[i].to_string()));
                    }
                    other => {
                        let flat: Vec<i64> = other.cast::<i64>().iter().copied().collect();
                        nested(out, &shape, 0, &|out, i| out.push_str(&flat[i].to_string()));
                    }
                }
            }
        }
    }

    /// A short type description, for diagnostics.
    pub fn describe(&self) -> String {
        match self {
            AttrValue::Str(_) => "str".into(),
            AttrValue::Strs(v) => format!("str[{}]", v.len()),
            AttrValue::Bool(_) => "bool".into(),
            AttrValue::Int(_) => "int64".into(),
            AttrValue::Float(_) => "float64".into(),
            AttrValue::Array(a) => format!("{}{:?}", a.dtype().name(), a.shape()),
            AttrValue::Unsupported(d) => d.clone(),
        }
    }
}

/// Nested JSON lists of an array (NumPy `tolist()`).
pub fn array_to_json(array: &NdArray) -> Value {
    fn nest(values: &[Value], shape: &[usize]) -> Value {
        if shape.is_empty() {
            return values.first().cloned().unwrap_or(Value::Null);
        }
        if shape.len() == 1 {
            return Value::Array(values.to_vec());
        }
        let stride: usize = shape[1..].iter().product();
        let mut out = Vec::with_capacity(shape[0]);
        for i in 0..shape[0] {
            out.push(nest(&values[i * stride..(i + 1) * stride], &shape[1..]));
        }
        Value::Array(out)
    }
    let flat: Vec<Value> = match array {
        NdArray::Bool(a) => a.iter().map(|v| Value::Bool(*v)).collect(),
        NdArray::F16(_) | NdArray::F32(_) | NdArray::F64(_) => array.to_f64().iter().map(|v| num(*v)).collect(),
        NdArray::U64(a) => a.iter().map(|v| Value::from(*v)).collect(),
        other => other.cast::<i64>().iter().map(|v| Value::from(*v)).collect(),
    };
    nest(&flat, &array.shape())
}

fn descriptor(attr: &hdf5::Attribute) -> Result<TD> {
    Ok(attr.dtype()?.to_descriptor()?)
}

/// Read one attribute, or `None` when it is absent.
pub fn read(obj: &hdf5::Location, name: &str) -> Result<Option<AttrValue>> {
    phase!(Metadata);
    super::alive(obj)?;
    if !has(obj, name) {
        return Ok(None);
    }
    let attr = obj.attr(name)?;
    Ok(Some(read_attribute(&attr)?))
}

/// Decode an open attribute.
pub fn read_attribute(attr: &hdf5::Attribute) -> Result<AttrValue> {
    let td = match descriptor(attr) {
        Ok(td) => td,
        Err(e) => return Ok(AttrValue::Unsupported(e.to_string())),
    };
    let shape = attr.shape();
    let scalar = shape.is_empty();
    let value = match td {
        TD::VarLenUnicode | TD::VarLenAscii | TD::FixedAscii(_) | TD::FixedUnicode(_) => {
            let strings = read_strings_attr(attr, &td)?;
            if scalar {
                AttrValue::Str(strings.into_iter().next().unwrap_or_default())
            } else {
                AttrValue::Strs(strings)
            }
        }
        TD::Boolean => {
            let a = attr.read_dyn::<bool>()?;
            if scalar {
                AttrValue::Bool(a.iter().next().copied().unwrap_or(false))
            } else {
                AttrValue::Array(NdArray::Bool(a))
            }
        }
        TD::Integer(_) | TD::Unsigned(_) | TD::Float(_) => {
            let dtype = numeric_dtype(&td).expect("numeric descriptor");
            let array = with_dtype!(dtype, T => NdArray::from(attr.read_dyn::<T>()?));
            if scalar {
                match &array {
                    NdArray::F16(_) | NdArray::F32(_) | NdArray::F64(_) => {
                        AttrValue::Float(array.to_f64().iter().next().copied().unwrap_or(0.0))
                    }
                    NdArray::U64(a) if a.iter().next().copied().unwrap_or(0) > i64::MAX as u64 => {
                        AttrValue::Array(array)
                    }
                    _ => AttrValue::Int(array.cast::<i64>().iter().next().copied().unwrap_or(0)),
                }
            } else {
                AttrValue::Array(array)
            }
        }
        TD::Enum(e) => {
            // A non-boolean enum reads as its base integer, as NumPy sees it.
            let dtype = enum_base(&e);
            let raw = read_enum_raw(attr, dtype)?;
            if scalar {
                AttrValue::Int(raw.cast::<i64>().iter().next().copied().unwrap_or(0))
            } else {
                AttrValue::Array(raw)
            }
        }
        other => AttrValue::Unsupported(format!("{other}")),
    };
    Ok(value)
}

fn read_strings_attr(attr: &hdf5::Attribute, td: &TD) -> Result<Vec<String>> {
    // Sized before it is read, as a dataset is, and by what it decodes to:
    // a `String` per element beyond its stored bytes, since an attribute
    // cannot be read in parts.
    let itemsize = match td {
        TD::FixedAscii(width) | TD::FixedUnicode(width) => (*width).max(1).saturating_mul(2),
        _ => std::mem::size_of::<VarLenUnicode>(),
    };
    super::data::ensure_holdable(
        &format!("attribute {}", repr_str(&attr.name())),
        &[attr.size().max(1)],
        itemsize.saturating_add(super::data::STRING_COST),
    )?;
    Ok(match td {
        // Bytes, decoded here: the types' `as_str` trusts the file to hold valid
        // UTF-8, and a damaged one does not (`text` keeps that a finding, not
        // undefined behaviour).
        TD::VarLenUnicode => attr.read_raw::<VarLenUnicode>()?.iter().map(|s| text(s.as_bytes().to_vec())).collect(),
        TD::VarLenAscii => {
            attr.read_raw::<hdf5::types::VarLenAscii>()?.iter().map(|s| text(s.as_bytes().to_vec())).collect()
        }
        TD::FixedAscii(_) | TD::FixedUnicode(_) => {
            read_fixed_strings(attr.id(), attr.size(), true)?.into_iter().map(text).collect()
        }
        _ => Vec::new(),
    })
}

/// Text from stored bytes: kept as they are when they are UTF-8, and
/// invalid UTF-8 replaced rather than trusted.
pub(crate) fn text(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

/// The stored bytes of `n` fixed-length strings, each without its trailing
/// NUL padding --- what h5py and NumPy read.
///
/// Read in the stored type, so no character set is converted (HDF5 refuses
/// to convert between ASCII and UTF-8) and no byte is decoded: a caller
/// wanting text decodes them ([`text`]).
pub(crate) fn read_fixed_strings(id: crate::h5sys::h5i::hid_t, n: usize, is_attr: bool) -> Result<Vec<Vec<u8>>> {
    use crate::h5sys::{h5a, h5d, h5p, h5s, h5t};
    super::locked(|| unsafe {
        let ftype = if is_attr { h5a::H5Aget_type(id) } else { h5d::H5Dget_type(id) };
        if ftype < 0 {
            return Err(Error::Io("could not read a fixed-length string".into()));
        }
        let size = h5t::H5Tget_size(ftype).max(1);
        // The callers size the read first; the product is checked here too,
        // and the buffer reserved fallibly, so no declared size can panic.
        let mut buf = Vec::new();
        let bytes = size
            .checked_mul(n.max(1))
            .filter(|bytes| super::data::holdable(&[*bytes], 1))
            .filter(|bytes| buf.try_reserve_exact(*bytes).is_ok());
        let Some(bytes) = bytes else {
            h5t::H5Tclose(ftype);
            return Err(Error::Io(format!(
                "{} fixed-length strings of {size} bytes are more than this process can hold",
                n.max(1)
            )));
        };
        buf.resize(bytes, 0);
        let status = if is_attr {
            h5a::H5Aread(id, ftype, buf.as_mut_ptr().cast())
        } else {
            h5d::H5Dread(id, ftype, h5s::H5S_ALL, h5s::H5S_ALL, h5p::H5P_DEFAULT, buf.as_mut_ptr().cast())
        };
        h5t::H5Tclose(ftype);
        if status < 0 {
            return Err(Error::Io("could not read a fixed-length string".into()));
        }
        Ok(buf.chunks(size).take(n).map(unpadded).collect())
    })
}

/// A fixed-length string's stored bytes without the NULs padding it.
fn unpadded(item: &[u8]) -> Vec<u8> {
    let end = item.iter().rposition(|b| *b != 0).map(|i| i + 1).unwrap_or(0);
    item[..end].to_vec()
}

/// Rows `[start, start + count)` of a fixed-length string dataset of shape
/// `shape`, along its first axis with every other axis whole, read as
/// [`read_fixed_strings`] reads them.  The caller has sized the read.
pub(crate) fn read_fixed_string_rows(
    ds: &hdf5::Dataset,
    shape: &[usize],
    start: usize,
    count: usize,
) -> Result<Vec<Vec<u8>>> {
    use crate::h5sys::{h5d, h5p, h5s, h5t};
    let mut first = vec![0u64; shape.len()];
    let mut counts: Vec<u64> = shape.iter().map(|n| *n as u64).collect();
    first[0] = start as u64;
    counts[0] = count as u64;
    let n = counts.iter().try_fold(1usize, |n, c| n.checked_mul(*c as usize));
    super::locked(|| unsafe {
        let ftype = h5d::H5Dget_type(ds.id());
        if ftype < 0 {
            return Err(Error::Io(format!("could not read the strings of {}", ds.name())));
        }
        let size = h5t::H5Tget_size(ftype).max(1);
        let mut buf = Vec::new();
        let bytes = n
            .and_then(|n| n.checked_mul(size))
            .filter(|bytes| super::data::holdable(&[*bytes], 1))
            .filter(|bytes| buf.try_reserve_exact(*bytes).is_ok());
        let (Some(bytes), Some(n)) = (bytes, n) else {
            h5t::H5Tclose(ftype);
            return Err(Error::Io(format!(
                "{} strings of {size} bytes of {} are more than this process can hold",
                count,
                ds.name()
            )));
        };
        buf.resize(bytes, 0);
        let fspace = h5d::H5Dget_space(ds.id());
        let mut status = h5s::H5Sselect_hyperslab(
            fspace,
            h5s::H5S_seloper_t::H5S_SELECT_SET,
            first.as_ptr(),
            std::ptr::null(),
            counts.as_ptr(),
            std::ptr::null(),
        );
        let mspace = h5s::H5Screate_simple(counts.len() as i32, counts.as_ptr(), std::ptr::null());
        if status >= 0 {
            status = h5d::H5Dread(ds.id(), ftype, mspace, fspace, h5p::H5P_DEFAULT, buf.as_mut_ptr().cast());
        }
        h5s::H5Sclose(mspace);
        h5s::H5Sclose(fspace);
        h5t::H5Tclose(ftype);
        if status < 0 {
            return Err(Error::Io(format!("could not read the strings of {}", ds.name())));
        }
        let mut out = Vec::new();
        out.try_reserve_exact(n)
            .map_err(|_| Error::Io(format!("{n} strings of {} are more than this process can hold", ds.name())))?;
        out.extend(buf.chunks(size).take(n).map(unpadded));
        Ok(out)
    })
}

fn enum_base(e: &hdf5::types::EnumType) -> DType {
    use hdf5::types::IntSize;
    match (e.size, e.signed) {
        (IntSize::U1, true) => DType::I8,
        (IntSize::U1, false) => DType::U8,
        (IntSize::U2, true) => DType::I16,
        (IntSize::U2, false) => DType::U16,
        (IntSize::U4, true) => DType::I32,
        (IntSize::U4, false) => DType::U32,
        (IntSize::U8, true) => DType::I64,
        (IntSize::U8, false) => DType::U64,
    }
}

fn read_enum_raw(attr: &hdf5::Attribute, dtype: DType) -> Result<NdArray> {
    // Read the bytes in the file's own enum type and reinterpret them as the
    // base integer: HDF5 has no enum -> integer conversion path.
    use crate::h5sys::{h5a, h5t};
    let shape = attr.shape();
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut buf = vec![0u8; n * dtype.itemsize()];
    super::locked(|| unsafe {
        let ftype = h5a::H5Aget_type(attr.id());
        let mtype = h5t::H5Tget_native_type(ftype, h5t::H5T_direction_t::H5T_DIR_ASCEND);
        let status = h5a::H5Aread(attr.id(), mtype, buf.as_mut_ptr().cast());
        h5t::H5Tclose(mtype);
        h5t::H5Tclose(ftype);
        if status < 0 {
            return Err(Error::Io("could not read an enum attribute".into()));
        }
        Ok(())
    })?;
    bytes_to_array(&buf, dtype, if shape.is_empty() { &[] } else { &shape })
}

/// Native-endian bytes to an array of `dtype` and `shape`.
pub(crate) fn bytes_to_array(buf: &[u8], dtype: DType, shape: &[usize]) -> Result<NdArray> {
    let n: usize = shape.iter().product::<usize>();
    let n = if shape.is_empty() { 1 } else { n };
    let size = dtype.itemsize();
    if buf.len() < n * size {
        return Err(Error::Io("short read".into()));
    }
    macro_rules! decode {
        ($t:ty) => {{
            let values: Vec<$t> =
                (0..n).map(|i| <$t>::from_ne_bytes(buf[i * size..(i + 1) * size].try_into().unwrap())).collect();
            NdArray::from(ArrayD::from_shape_vec(IxDyn(shape), values)?)
        }};
    }
    Ok(match dtype {
        DType::I8 => decode!(i8),
        DType::U8 => decode!(u8),
        DType::I16 => decode!(i16),
        DType::U16 => decode!(u16),
        DType::I32 => decode!(i32),
        DType::U32 => decode!(u32),
        DType::I64 => decode!(i64),
        DType::U64 => decode!(u64),
        DType::Bool => NdArray::from(ArrayD::from_shape_vec(IxDyn(shape), buf[..n].iter().map(|b| *b != 0).collect())?),
        DType::F16 => NdArray::from(ArrayD::from_shape_vec(
            IxDyn(shape),
            (0..n).map(|i| half::f16::from_ne_bytes(buf[i * 2..i * 2 + 2].try_into().unwrap())).collect(),
        )?),
        DType::F32 => decode!(f32),
        DType::F64 => decode!(f64),
    })
}

/// The array dtype of a numeric descriptor.
pub fn numeric_dtype(td: &TD) -> Option<DType> {
    use hdf5::types::{FloatSize, IntSize};
    Some(match td {
        TD::Boolean => DType::Bool,
        TD::Integer(IntSize::U1) => DType::I8,
        TD::Integer(IntSize::U2) => DType::I16,
        TD::Integer(IntSize::U4) => DType::I32,
        TD::Integer(IntSize::U8) => DType::I64,
        TD::Unsigned(IntSize::U1) => DType::U8,
        TD::Unsigned(IntSize::U2) => DType::U16,
        TD::Unsigned(IntSize::U4) => DType::U32,
        TD::Unsigned(IntSize::U8) => DType::U64,
        TD::Float(FloatSize::U2) => DType::F16,
        TD::Float(FloatSize::U4) => DType::F32,
        TD::Float(FloatSize::U8) => DType::F64,
        TD::Enum(e) => enum_base(e),
        _ => return None,
    })
}

/// Whether an attribute exists.
///
/// Asked of HDF5 by name: listing every name to find one made each attribute
/// read cost as many reads as the object has attributes.
pub fn has(obj: &hdf5::Location, name: &str) -> bool {
    phase!(Metadata);
    let Ok(cname) = std::ffi::CString::new(name) else { return false };
    // SAFETY: a NUL-terminated name and HDF5's own id, under its lock; a
    // closed object is an error, which reads as absent, as it did.
    super::locked(|| unsafe { crate::h5sys::h5a::H5Aexists(obj.id(), cname.as_ptr()) > 0 })
}

/// Every attribute name, sorted (HDF5's name order).
pub fn names(obj: &hdf5::Location) -> Result<Vec<String>> {
    phase!(Metadata);
    super::alive(obj)?;
    let mut names = obj.attr_names()?;
    names.sort();
    Ok(names)
}

/// Delete an attribute if present.
pub fn delete(obj: &hdf5::Location, name: &str) -> Result<()> {
    phase!(Metadata);
    if has(obj, name) {
        obj.delete_attr(name)?;
    }
    Ok(())
}

/// Write (or overwrite) one attribute.
pub fn write(obj: &hdf5::Location, name: &str, value: &AttrValue) -> Result<()> {
    phase!(Metadata);
    delete(obj, name)?;
    match value {
        AttrValue::Str(s) => {
            let v: VarLenUnicode = s.parse().map_err(|e| Error::Value(format!("{e}")))?;
            obj.new_attr::<VarLenUnicode>().shape(()).create(name)?.write_scalar(&v)?;
        }
        AttrValue::Strs(values) => {
            let parsed: Vec<VarLenUnicode> = values
                .iter()
                .map(|s| s.parse())
                .collect::<std::result::Result<_, _>>()
                .map_err(|e| Error::Value(format!("{e}")))?;
            let arr = ndarray::Array1::from(parsed);
            obj.new_attr::<VarLenUnicode>().shape(values.len()).create(name)?.write(&arr)?;
        }
        AttrValue::Bool(b) => {
            obj.new_attr::<bool>().shape(()).create(name)?.write_scalar(b)?;
        }
        AttrValue::Int(i) => {
            obj.new_attr::<i64>().shape(()).create(name)?.write_scalar(i)?;
        }
        AttrValue::Float(f) => {
            obj.new_attr::<f64>().shape(()).create(name)?.write_scalar(f)?;
        }
        AttrValue::Array(array) => {
            with_array!(array, a => {
                obj.new_attr_builder().with_data(a.view()).create(name)?;
            });
        }
        AttrValue::Unsupported(d) => {
            return Err(Error::Value(format!("cannot write attribute {} of unsupported type {d}", repr_str(name))));
        }
    }
    Ok(())
}

/// Write a mapping of attributes, skipping `None` values.
pub fn write_all(obj: &hdf5::Location, values: &[(&str, Option<AttrValue>)]) -> Result<()> {
    for (name, value) in values {
        if let Some(v) = value {
            write(obj, name, v)?;
        }
    }
    Ok(())
}

// -- typed getters -----------------------------------------------------------

/// A string attribute, or `None` when absent.
pub fn get_str(obj: &hdf5::Location, name: &str) -> Result<Option<String>> {
    Ok(read(obj, name)?.map(|v| v.as_str().unwrap_or_else(|| stringify_value(&v))))
}

/// A string-list attribute, or `None` when absent.
pub fn get_strs(obj: &hdf5::Location, name: &str) -> Result<Option<Vec<String>>> {
    Ok(read(obj, name)?.map(|v| v.as_str_list().unwrap_or_default()))
}

/// An integer-list attribute, or `None` when absent.
pub fn get_i64s(obj: &hdf5::Location, name: &str) -> Result<Option<Vec<i64>>> {
    Ok(read(obj, name)?.and_then(|v| v.as_i64_vec()))
}

/// A float-list attribute, or `None` when absent.
pub fn get_f64s(obj: &hdf5::Location, name: &str) -> Result<Option<Vec<f64>>> {
    Ok(read(obj, name)?.and_then(|v| v.as_f64_vec()))
}

/// A scalar integer attribute, or `None` when absent.
pub fn get_i64(obj: &hdf5::Location, name: &str) -> Result<Option<i64>> {
    Ok(read(obj, name)?.and_then(|v| v.as_i64()))
}

/// A scalar float attribute, or `None` when absent.
pub fn get_f64(obj: &hdf5::Location, name: &str) -> Result<Option<f64>> {
    Ok(read(obj, name)?.and_then(|v| v.as_f64()))
}

/// A scalar boolean attribute, or `None` when absent.
pub fn get_bool(obj: &hdf5::Location, name: &str) -> Result<Option<bool>> {
    Ok(read(obj, name)?.and_then(|v| v.as_bool()))
}

/// Python's `str()` of an attribute value that is not a string.
pub fn stringify_value(value: &AttrValue) -> String {
    match value {
        AttrValue::Str(s) => s.clone(),
        AttrValue::Int(i) => i.to_string(),
        AttrValue::Float(f) => crate::json::py_float(*f),
        AttrValue::Bool(b) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        other => crate::json::py_str(&other.to_json()),
    }
}

/// The name of the attribute's object, for messages.
pub fn object_name(obj: &hdf5::Location) -> String {
    obj.name()
}

/// Fetch a required attribute, raising a coded validation error when absent.
pub fn require(obj: &hdf5::Location, name: &str, code: &str) -> Result<AttrValue> {
    read(obj, name)?
        .ok_or_else(|| Error::coded(code, format!("{}: required attribute {} is missing", obj.name(), repr_str(name))))
}

/// Copy one attribute byte-for-byte, keeping its stored type and shape.
///
/// An amend copies the root attributes it does not manage this way, so an
/// attribute this writer does not interpret --- an extension's --- survives
/// exactly as written: an `int32` stays `int32`, a fixed-length string stays
/// fixed-length.  (A file of a later minor version is not amended at all.)
pub fn copy_raw(src: &hdf5::Location, dst: &hdf5::Location, name: &str) -> Result<()> {
    RawAttr::capture(src, name)?.restore(dst)
}

/// An attribute captured exactly as stored --- its datatype, dataspace and
/// bytes --- so it can be recreated after the object holding it is rebuilt,
/// with nothing a typed decode would normalise (§16).
pub struct RawAttr {
    name: std::ffi::CString,
    dtype: crate::h5sys::h5i::hid_t,
    space: crate::h5sys::h5i::hid_t,
    buf: Vec<u8>,
    /// The buffer holds variable-length data HDF5 allocated, to reclaim.
    vlen: bool,
}

// SAFETY: the identifiers and the variable-length memory they own are only
// touched under HDF5's global lock.
unsafe impl Send for RawAttr {}

impl RawAttr {
    /// Read attribute `name` of `obj` as stored.
    pub fn capture(obj: &hdf5::Location, name: &str) -> Result<RawAttr> {
        use crate::h5sys::{h5a, h5p, h5s, h5t};
        let cname = std::ffi::CString::new(name).map_err(|e| Error::Value(e.to_string()))?;
        super::locked(|| unsafe {
            let attr = h5a::H5Aopen(obj.id(), cname.as_ptr(), h5p::H5P_DEFAULT);
            if attr < 0 {
                return Err(Error::Io(format!("could not open attribute {}", repr_str(name))));
            }
            let stored = h5a::H5Aget_type(attr);
            let dtype = h5t::H5Tcopy(stored);
            h5t::H5Tclose(stored);
            let space = h5a::H5Aget_space(attr);
            let size = h5t::H5Tget_size(dtype);
            let npoints = h5s::H5Sget_simple_extent_npoints(space).max(1) as usize;
            let mut buf = vec![0u8; size * npoints];
            let status = h5a::H5Aread(attr, dtype, buf.as_mut_ptr().cast());
            h5a::H5Aclose(attr);
            let vlen =
                h5t::H5Tdetect_class(dtype, h5t::H5T_class_t::H5T_VLEN) > 0 || h5t::H5Tis_variable_str(dtype) > 0;
            let raw = RawAttr { name: cname, dtype, space, buf, vlen: vlen && status >= 0 };
            if status < 0 {
                return Err(Error::Io(format!("could not read attribute {}", repr_str(name))));
            }
            Ok(raw)
        })
    }

    /// The attribute's name.
    pub fn name(&self) -> &str {
        self.name.to_str().unwrap_or_default()
    }

    /// Write it to `obj`, replacing an attribute of the same name.
    pub fn restore(&self, obj: &hdf5::Location) -> Result<()> {
        use crate::h5sys::{h5a, h5p};
        super::locked(|| unsafe {
            if h5a::H5Aexists(obj.id(), self.name.as_ptr()) > 0 {
                h5a::H5Adelete(obj.id(), self.name.as_ptr());
            }
            let out = h5a::H5Acreate2(
                obj.id(),
                self.name.as_ptr(),
                self.dtype,
                self.space,
                h5p::H5P_DEFAULT,
                h5p::H5P_DEFAULT,
            );
            let status = if out < 0 { -1 } else { h5a::H5Awrite(out, self.dtype, self.buf.as_ptr().cast()) };
            if out >= 0 {
                h5a::H5Aclose(out);
            }
            if status < 0 {
                return Err(Error::Io(format!("could not copy attribute {}", repr_str(&self.name.to_string_lossy()))));
            }
            Ok(())
        })
    }
}

impl Drop for RawAttr {
    fn drop(&mut self) {
        use crate::h5sys::{h5p, h5s, h5t};
        super::locked(|| unsafe {
            if self.vlen {
                h5t::H5Treclaim(self.dtype, self.space, h5p::H5P_DEFAULT, self.buf.as_mut_ptr().cast());
            }
            h5s::H5Sclose(self.space);
            h5t::H5Tclose(self.dtype);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> (tempfile::TempDir, hdf5::File) {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::create_truncate(&dir.path().join("a.h5")).unwrap();
        (dir, file)
    }

    /// Every logical type of §2.5 reads back as itself, stored as the table
    /// says.
    #[test]
    fn s2_5_types_round_trip() {
        let (_dir, file) = scratch();
        let cases = [
            ("str", AttrValue::Str("x".into()), TD::VarLenUnicode),
            ("strs", AttrValue::strs(&["a", "b"]), TD::VarLenUnicode),
            ("int", AttrValue::Int(3), TD::Integer(hdf5::types::IntSize::U8)),
            ("ints", AttrValue::ints(&[1, 2]), TD::Integer(hdf5::types::IntSize::U8)),
            ("float", AttrValue::Float(1.5), TD::Float(hdf5::types::FloatSize::U8)),
            ("floats", AttrValue::floats(&[1.5, 2.5]), TD::Float(hdf5::types::FloatSize::U8)),
            ("bool", AttrValue::Bool(true), TD::Boolean),
            ("matrix", AttrValue::matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]), TD::Float(hdf5::types::FloatSize::U8)),
        ];
        for (name, value, stored) in &cases {
            write(&file, name, value).unwrap();
            assert_eq!(read(&file, name).unwrap().as_ref(), Some(value), "{name}");
            assert_eq!(descriptor(&file.attr(name).unwrap()).unwrap(), *stored, "{name}");
        }
        assert_eq!(read(&file, "str").unwrap().unwrap().as_str().as_deref(), Some("x"));
        assert_eq!(read(&file, "strs").unwrap().unwrap().as_str_list(), Some(vec!["a".into(), "b".into()]));
        assert_eq!(read(&file, "int").unwrap().unwrap().as_i64(), Some(3));
        assert_eq!(read(&file, "ints").unwrap().unwrap().as_i64_vec(), Some(vec![1, 2]));
        assert_eq!(read(&file, "float").unwrap().unwrap().as_f64(), Some(1.5));
        assert_eq!(read(&file, "floats").unwrap().unwrap().as_f64_vec(), Some(vec![1.5, 2.5]));
        assert_eq!(read(&file, "bool").unwrap().unwrap().as_bool(), Some(true));
        assert_eq!(read(&file, "absent").unwrap(), None);
        // A scalar string is a list of one.
        assert_eq!(AttrValue::Str("solo".into()).as_str_list(), Some(vec!["solo".into()]));
    }

    /// Readers accept fixed-length strings as well as variable-length ones.
    #[test]
    fn s2_5_fixed_length_strings_read_as_strings() {
        let (_dir, file) = scratch();
        let fixed: hdf5::types::FixedAscii<4> = hdf5::types::FixedAscii::from_ascii(b"x").unwrap();
        file.new_attr::<hdf5::types::FixedAscii<4>>().shape(()).create("fixed").unwrap().write_scalar(&fixed).unwrap();
        assert_eq!(read(&file, "fixed").unwrap(), Some(AttrValue::Str("x".into())));
    }

    /// An attribute is found by name, and a closed object or a name HDF5
    /// cannot hold has none.
    #[test]
    fn s2_5_an_attribute_is_found_by_name() {
        let (_dir, file) = scratch();
        write(&file, "here", &AttrValue::Int(1)).unwrap();
        assert!(has(&file, "here"));
        assert!(!has(&file, "absent"));
        assert!(!has(&file, "he\0re"));
        let group = file.create_group("g").unwrap();
        write(&group, "inner", &AttrValue::Int(2)).unwrap();
        assert!(has(&group, "inner") && !has(&file, "inner"));
        crate::h5::file::close_everything(file, false).unwrap();
        assert!(!has(&group, "inner"));
    }

    /// Matrices stay two-dimensional: a flat array is not a matrix.
    #[test]
    fn s2_5_matrices_stay_two_dimensional() {
        let eye = AttrValue::matrix(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(eye.as_matrix().map(|(r, c, _)| (r, c)), Some((3, 3)));
        assert_eq!(AttrValue::floats(&[0.0; 9]).as_matrix(), None);
        assert_eq!(AttrValue::Float(1.0).as_matrix(), None);
    }

    /// An empty list is stored `int64` of shape `(0,)`, as NumPy would.
    #[test]
    fn s2_5_empty_lists_and_overwrites() {
        let (_dir, file) = scratch();
        write(&file, "empty", &AttrValue::ints(&[])).unwrap();
        let back = read(&file, "empty").unwrap().unwrap();
        assert_eq!(back.shape(), vec![0]);
        assert_eq!(back.describe(), "int64[0]");
        // Writing again replaces the value and its type.
        write(&file, "empty", &AttrValue::Str("now a string".into())).unwrap();
        assert_eq!(get_str(&file, "empty").unwrap().as_deref(), Some("now a string"));
        assert!(write(&file, "bad", &AttrValue::Unsupported("compound".into())).is_err());
    }
}

#[cfg(test)]
mod canonical_tests {
    use super::*;

    /// `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
    /// of the value 1.x read with h5py --- non-finite floats included.
    #[test]
    fn s13_2_canonical_attribute_values_are_python_json_to_the_byte() {
        let cases: Vec<(AttrValue, &str)> = vec![
            (AttrValue::Str("ünï \"q\"\n".into()), "\"ünï \\\"q\\\"\\n\""),
            (AttrValue::strs(&["a", "b"]), "[\"a\",\"b\"]"),
            (AttrValue::Bool(true), "true"),
            (AttrValue::Int(-3), "-3"),
            (AttrValue::Float(1.0), "1.0"),
            (AttrValue::Float(f64::NAN), "NaN"),
            (AttrValue::Float(f64::NEG_INFINITY), "-Infinity"),
            (AttrValue::floats(&[0.1, 1e-5, 1e16]), "[0.1,1e-05,1e+16]"),
            (AttrValue::ints(&[]), "[]"),
            (AttrValue::matrix(2, 2, &[1.0, 0.0, f64::INFINITY, -0.0]), "[[1.0,0.0],[Infinity,-0.0]]"),
            (AttrValue::Array(NdArray::from(ndarray::arr1(&[true, false]).into_dyn())), "[true,false]"),
            (AttrValue::Array(NdArray::from(ndarray::arr1(&[u64::MAX]).into_dyn())), "[18446744073709551615]"),
            (AttrValue::Array(NdArray::from(ndarray::arr1(&[0.5f32]).into_dyn())), "[0.5]"),
        ];
        for (value, expected) in cases {
            let mut out = String::new();
            value.write_canonical(&mut out);
            assert_eq!(out, expected, "{value:?}");
        }
    }
}
