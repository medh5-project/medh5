//! Dense arrays of any element type the format stores.
//!
//! Images are whatever dtype the scanner produced; labels are `uint8`/`uint16`;
//! bitplanes are `uint64`; probabilities are `float16`/`float32`; masks are
//! `bool`.  [`NdArray`] carries one of them with its shape, in C order, and is
//! what every reader returns and every writer accepts.  The dtype names and
//! strings are NumPy's (`"int16"`, `"<i2"`), because the §13.1 digest is
//! defined over NumPy's dtype string.

use half::f16;
use ndarray::{ArrayD, IxDyn};

use crate::{Error, Result};

/// The element types the format stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DType {
    Bool,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F16,
    F32,
    F64,
}

impl DType {
    /// Every dtype, in promotion order.
    pub const ALL: [DType; 12] = [
        DType::Bool,
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::I32,
        DType::U32,
        DType::I64,
        DType::U64,
        DType::F16,
        DType::F32,
        DType::F64,
    ];

    /// NumPy's dtype string with explicit little-endian byte order (§13.1).
    pub fn numpy_str(self) -> &'static str {
        match self {
            DType::Bool => "|b1",
            DType::I8 => "|i1",
            DType::U8 => "|u1",
            DType::I16 => "<i2",
            DType::U16 => "<u2",
            DType::I32 => "<i4",
            DType::U32 => "<u4",
            DType::I64 => "<i8",
            DType::U64 => "<u8",
            DType::F16 => "<f2",
            DType::F32 => "<f4",
            DType::F64 => "<f8",
        }
    }

    /// NumPy's dtype name.
    pub fn name(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::I8 => "int8",
            DType::U8 => "uint8",
            DType::I16 => "int16",
            DType::U16 => "uint16",
            DType::I32 => "int32",
            DType::U32 => "uint32",
            DType::I64 => "int64",
            DType::U64 => "uint64",
            DType::F16 => "float16",
            DType::F32 => "float32",
            DType::F64 => "float64",
        }
    }

    /// Parse a NumPy dtype name or string (`"uint8"`, `"<u2"`, `"|b1"`, `"f4"`).
    pub fn parse(text: &str) -> Result<DType> {
        let t = text.trim().trim_start_matches(['<', '>', '=', '|']);
        let found = match t {
            "bool" | "b1" | "?" | "bool_" => DType::Bool,
            "int8" | "i1" => DType::I8,
            "uint8" | "u1" => DType::U8,
            "int16" | "i2" => DType::I16,
            "uint16" | "u2" => DType::U16,
            "int32" | "i4" => DType::I32,
            "uint32" | "u4" => DType::U32,
            "int64" | "i8" => DType::I64,
            "uint64" | "u8" => DType::U64,
            "float16" | "f2" | "half" => DType::F16,
            "float32" | "f4" | "single" => DType::F32,
            "float64" | "f8" | "double" | "float" => DType::F64,
            _ => return Err(Error::Type(format!("data type {text:?} not understood"))),
        };
        Ok(found)
    }

    /// Bytes per element.
    pub fn itemsize(self) -> usize {
        match self {
            DType::Bool | DType::I8 | DType::U8 => 1,
            DType::I16 | DType::U16 | DType::F16 => 2,
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::I64 | DType::U64 | DType::F64 => 8,
        }
    }

    /// NumPy's kind character: `b`, `i`, `u` or `f`.
    pub fn kind(self) -> char {
        match self {
            DType::Bool => 'b',
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => 'i',
            DType::U8 | DType::U16 | DType::U32 | DType::U64 => 'u',
            DType::F16 | DType::F32 | DType::F64 => 'f',
        }
    }

    pub fn is_float(self) -> bool {
        self.kind() == 'f'
    }

    pub fn is_integer(self) -> bool {
        matches!(self.kind(), 'i' | 'u')
    }

    /// Whether the integer `value` comes through a conversion to this dtype
    /// unchanged: inside an integer type's range, exactly representable in a
    /// float type (every integer up to 2048 in `float16`, 2**24 in
    /// `float32`), `0` or `1` in `bool`.
    pub fn holds(self, value: i128) -> bool {
        crate::with_dtype!(self, T => T::from_i128(value).to_i128() == value)
    }

    /// The inclusive range of an integer dtype.
    pub fn int_range(self) -> Option<(i128, i128)> {
        Some(match self {
            DType::I8 => (i8::MIN as i128, i8::MAX as i128),
            DType::U8 => (0, u8::MAX as i128),
            DType::I16 => (i16::MIN as i128, i16::MAX as i128),
            DType::U16 => (0, u16::MAX as i128),
            DType::I32 => (i32::MIN as i128, i32::MAX as i128),
            DType::U32 => (0, u32::MAX as i128),
            DType::I64 => (i64::MIN as i128, i64::MAX as i128),
            DType::U64 => (0, u64::MAX as i128),
            DType::Bool => (0, 1),
            _ => return None,
        })
    }
}

/// An element type an [`NdArray`] can hold.
pub trait Element: hdf5::H5Type + Copy + Default + PartialEq + Send + Sync + std::fmt::Debug + 'static {
    const DTYPE: DType;
    fn to_f64(self) -> f64;
    fn from_f64(value: f64) -> Self;
    fn to_i128(self) -> i128;
    /// Saturating conversion from an exact integer.
    fn from_i128(value: i128) -> Self;
    fn write_le(self, out: &mut Vec<u8>);
    fn wrap(array: ArrayD<Self>) -> NdArray;
    fn unwrap_ref(array: &NdArray) -> Option<&ArrayD<Self>>;
}

macro_rules! int_element {
    ($t:ty, $variant:ident) => {
        impl Element for $t {
            const DTYPE: DType = DType::$variant;
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_f64(value: f64) -> Self {
                // `as` saturates and maps NaN to 0, which is what a checked
                // cast after a range test needs.
                value as $t
            }
            fn to_i128(self) -> i128 {
                self as i128
            }
            fn from_i128(value: i128) -> Self {
                value.clamp(<$t>::MIN as i128, <$t>::MAX as i128) as $t
            }
            fn write_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn wrap(array: ArrayD<Self>) -> NdArray {
                NdArray::$variant(array)
            }
            fn unwrap_ref(array: &NdArray) -> Option<&ArrayD<Self>> {
                match array {
                    NdArray::$variant(a) => Some(a),
                    _ => None,
                }
            }
        }
    };
}

int_element!(i8, I8);
int_element!(u8, U8);
int_element!(i16, I16);
int_element!(u16, U16);
int_element!(i32, I32);
int_element!(u32, U32);
int_element!(i64, I64);
int_element!(u64, U64);

impl Element for bool {
    const DTYPE: DType = DType::Bool;
    fn to_f64(self) -> f64 {
        if self {
            1.0
        } else {
            0.0
        }
    }
    fn from_f64(value: f64) -> Self {
        value != 0.0
    }
    fn to_i128(self) -> i128 {
        self as i128
    }
    fn from_i128(value: i128) -> Self {
        value != 0
    }
    fn write_le(self, out: &mut Vec<u8>) {
        out.push(self as u8);
    }
    fn wrap(array: ArrayD<Self>) -> NdArray {
        NdArray::Bool(array)
    }
    fn unwrap_ref(array: &NdArray) -> Option<&ArrayD<Self>> {
        match array {
            NdArray::Bool(a) => Some(a),
            _ => None,
        }
    }
}

macro_rules! float_element {
    ($t:ty, $variant:ident, $to:expr, $from:expr) => {
        impl Element for $t {
            const DTYPE: DType = DType::$variant;
            fn to_f64(self) -> f64 {
                ($to)(self)
            }
            fn from_f64(value: f64) -> Self {
                ($from)(value)
            }
            fn to_i128(self) -> i128 {
                ($to)(self) as i128
            }
            fn from_i128(value: i128) -> Self {
                ($from)(value as f64)
            }
            fn write_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn wrap(array: ArrayD<Self>) -> NdArray {
                NdArray::$variant(array)
            }
            fn unwrap_ref(array: &NdArray) -> Option<&ArrayD<Self>> {
                match array {
                    NdArray::$variant(a) => Some(a),
                    _ => None,
                }
            }
        }
    };
}

float_element!(f16, F16, |v: f16| v.to_f64(), f16::from_f64);
float_element!(f32, F32, |v: f32| v as f64, |v: f64| v as f32);
float_element!(f64, F64, |v: f64| v, |v: f64| v);

/// A dense array of one of the format's element types.
#[derive(Debug, Clone, PartialEq)]
pub enum NdArray {
    Bool(ArrayD<bool>),
    I8(ArrayD<i8>),
    U8(ArrayD<u8>),
    I16(ArrayD<i16>),
    U16(ArrayD<u16>),
    I32(ArrayD<i32>),
    U32(ArrayD<u32>),
    I64(ArrayD<i64>),
    U64(ArrayD<u64>),
    F16(ArrayD<f16>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
}

/// Run `$body` with `$a` bound to the typed array inside an [`NdArray`].
#[macro_export]
macro_rules! with_array {
    ($arr:expr, $a:ident => $body:expr) => {
        match $arr {
            $crate::array::NdArray::Bool($a) => $body,
            $crate::array::NdArray::I8($a) => $body,
            $crate::array::NdArray::U8($a) => $body,
            $crate::array::NdArray::I16($a) => $body,
            $crate::array::NdArray::U16($a) => $body,
            $crate::array::NdArray::I32($a) => $body,
            $crate::array::NdArray::U32($a) => $body,
            $crate::array::NdArray::I64($a) => $body,
            $crate::array::NdArray::U64($a) => $body,
            $crate::array::NdArray::F16($a) => $body,
            $crate::array::NdArray::F32($a) => $body,
            $crate::array::NdArray::F64($a) => $body,
        }
    };
}

/// Run `$body` with `$T` bound to the Rust element type of a [`DType`].
#[macro_export]
macro_rules! with_dtype {
    ($dtype:expr, $T:ident => $body:expr) => {
        match $dtype {
            $crate::array::DType::Bool => {
                type $T = bool;
                $body
            }
            $crate::array::DType::I8 => {
                type $T = i8;
                $body
            }
            $crate::array::DType::U8 => {
                type $T = u8;
                $body
            }
            $crate::array::DType::I16 => {
                type $T = i16;
                $body
            }
            $crate::array::DType::U16 => {
                type $T = u16;
                $body
            }
            $crate::array::DType::I32 => {
                type $T = i32;
                $body
            }
            $crate::array::DType::U32 => {
                type $T = u32;
                $body
            }
            $crate::array::DType::I64 => {
                type $T = i64;
                $body
            }
            $crate::array::DType::U64 => {
                type $T = u64;
                $body
            }
            $crate::array::DType::F16 => {
                type $T = half::f16;
                $body
            }
            $crate::array::DType::F32 => {
                type $T = f32;
                $body
            }
            $crate::array::DType::F64 => {
                type $T = f64;
                $body
            }
        }
    };
}

impl<T: Element> From<ArrayD<T>> for NdArray {
    fn from(array: ArrayD<T>) -> Self {
        T::wrap(array)
    }
}

impl NdArray {
    /// A zero-filled array.
    pub fn zeros(dtype: DType, shape: &[usize]) -> NdArray {
        with_dtype!(dtype, T => NdArray::from(ArrayD::<T>::default(IxDyn(shape))))
    }

    /// An array from a flat vector in C order.
    pub fn from_vec<T: Element>(shape: &[usize], data: Vec<T>) -> Result<NdArray> {
        Ok(NdArray::from(ArrayD::from_shape_vec(IxDyn(shape), data)?))
    }

    pub fn dtype(&self) -> DType {
        match self {
            NdArray::Bool(_) => DType::Bool,
            NdArray::I8(_) => DType::I8,
            NdArray::U8(_) => DType::U8,
            NdArray::I16(_) => DType::I16,
            NdArray::U16(_) => DType::U16,
            NdArray::I32(_) => DType::I32,
            NdArray::U32(_) => DType::U32,
            NdArray::I64(_) => DType::I64,
            NdArray::U64(_) => DType::U64,
            NdArray::F16(_) => DType::F16,
            NdArray::F32(_) => DType::F32,
            NdArray::F64(_) => DType::F64,
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        with_array!(self, a => a.shape().to_vec())
    }

    pub fn ndim(&self) -> usize {
        with_array!(self, a => a.ndim())
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        with_array!(self, a => a.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn nbytes(&self) -> usize {
        self.len() * self.dtype().itemsize()
    }

    /// The typed array, when the dtype matches.
    pub fn as_typed<T: Element>(&self) -> Option<&ArrayD<T>> {
        T::unwrap_ref(self)
    }

    /// Every element as `f64`, C order.
    pub fn to_f64(&self) -> ArrayD<f64> {
        with_array!(self, a => a.mapv(|v| v.to_f64()))
    }

    /// Every element converted to `T`, **saturating**: a value outside `T`'s
    /// range becomes the nearest value `T` holds, never another value by
    /// wrapping --- `300` is `255` as `uint8` and `-1` is `0`, where NumPy's
    /// `astype` gives `44` and `255`.  A float becomes an integer by
    /// truncation toward zero, and NaN becomes `0`; an integer too large for
    /// a float type becomes infinity.  Integer to integer goes through
    /// `i128`, so 64-bit values stay exact.
    pub fn cast<T: Element>(&self) -> ArrayD<T> {
        if let Some(same) = self.as_typed::<T>() {
            return same.clone();
        }
        match self {
            NdArray::Bool(a) => a.mapv(|v| T::from_f64(if v { 1.0 } else { 0.0 })),
            other => {
                // Integer -> integer through i128 keeps 64-bit values exact.
                if T::DTYPE.is_integer() && other.dtype().is_integer() {
                    with_array!(other, a => a.mapv(|v| T::from_i128(v.to_i128())))
                } else {
                    with_array!(other, a => a.mapv(|v| T::from_f64(v.to_f64())))
                }
            }
        }
    }

    /// The array converted to another dtype, saturating as [`Self::cast`]
    /// does --- NumPy's `astype` except at values the target cannot hold,
    /// which NumPy wraps.  A reader that must not lose a value checks it
    /// fits first ([`DType::holds`]), as `Annotation::labelmap` does.
    pub fn astype(&self, dtype: DType) -> NdArray {
        if dtype == self.dtype() {
            return self.clone();
        }
        with_dtype!(dtype, T => NdArray::from(self.cast::<T>()))
    }

    /// `value != 0` everywhere.
    pub fn nonzero_mask(&self) -> ArrayD<bool> {
        with_array!(self, a => a.mapv(|v| v.to_f64() != 0.0))
    }

    /// The C-order little-endian bytes of every element (§13.1).
    pub fn le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.nbytes());
        with_array!(self, a => {
            for v in a.iter() {
                v.write_le(&mut out);
            }
        });
        out
    }

    /// [`le_bytes`](Self::le_bytes), handed to `each` about `piece` bytes at
    /// a time: what a digest reads without a copy of the array beside it.
    pub(crate) fn for_each_le_piece(&self, piece: usize, each: &mut dyn FnMut(&[u8])) {
        let mut out = Vec::with_capacity(piece.min(self.nbytes()) + 16);
        with_array!(self, a => {
            for v in a.iter() {
                v.write_le(&mut out);
                if out.len() >= piece {
                    each(&out);
                    out.clear();
                }
            }
        });
        if !out.is_empty() {
            each(&out);
        }
    }

    /// Minimum and maximum as `f64`, or `None` for an empty array.
    pub fn min_max(&self) -> Option<(f64, f64)> {
        if self.is_empty() {
            return None;
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        with_array!(self, a => {
            for v in a.iter() {
                let f = v.to_f64();
                if f < lo {
                    lo = f;
                }
                if f > hi {
                    hi = f;
                }
            }
        });
        Some((lo, hi))
    }

    /// The array reshaped (C order), copying when needed.
    pub fn reshape(&self, shape: &[usize]) -> Result<NdArray> {
        with_array!(self, a => {
            let flat: Vec<_> = a.iter().copied().collect();
            NdArray::from_vec(shape, flat)
        })
    }

    /// A compact description: `uint8 (4, 5)`.
    pub fn describe(&self) -> String {
        format!("{} {:?}", self.dtype().name(), self.shape())
    }
}

/// A NumPy-style slice over one axis: `start:stop:step`, any part optional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Slice {
    pub start: Option<i64>,
    pub stop: Option<i64>,
    pub step: Option<i64>,
}

impl Slice {
    /// `start:stop`.
    pub fn new(start: i64, stop: i64) -> Slice {
        Slice { start: Some(start), stop: Some(stop), step: None }
    }

    /// `:` --- the whole axis.
    pub fn full() -> Slice {
        Slice::default()
    }

    /// Resolve against an axis of length `n` (NumPy semantics, positive step).
    ///
    /// Returns `(start, stop, step)` with `0 <= start <= stop <= n`.
    pub fn resolve(&self, n: usize) -> Result<(usize, usize, usize)> {
        let step = self.step.unwrap_or(1);
        if step <= 0 {
            return Err(Error::Value("slice step must be positive".into()));
        }
        let n_i = n as i64;
        let clip = |v: i64| -> i64 {
            let v = if v < 0 { v + n_i } else { v };
            v.clamp(0, n_i)
        };
        let start = self.start.map(clip).unwrap_or(0);
        let stop = self.stop.map(clip).unwrap_or(n_i).max(start);
        Ok((start as usize, stop as usize, step as usize))
    }

    /// The number of elements selected on an axis of length `n`.
    pub fn count(&self, n: usize) -> Result<usize> {
        let (start, stop, step) = self.resolve(n)?;
        Ok((stop - start).div_ceil(step))
    }
}

/// One axis of a selection: a slice, or a single index that drops the axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Index {
    Slice(Slice),
    At(i64),
}

impl From<Slice> for Index {
    fn from(s: Slice) -> Self {
        Index::Slice(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtypes_and_bytes() {
        let a = NdArray::from_vec(&[2, 2], vec![1i16, -2, 3, 4]).unwrap();
        assert_eq!(a.dtype().numpy_str(), "<i2");
        assert_eq!(a.le_bytes(), vec![1, 0, 0xfe, 0xff, 3, 0, 4, 0]);
        for piece in [1, 3, 4, 64] {
            let mut pieces = Vec::new();
            a.for_each_le_piece(piece, &mut |bytes| pieces.extend_from_slice(bytes));
            assert_eq!(pieces, a.le_bytes(), "piece {piece}");
        }
        assert_eq!(a.astype(DType::F32).to_f64().iter().copied().collect::<Vec<_>>(), vec![1.0, -2.0, 3.0, 4.0]);
        let big = NdArray::from_vec(&[1], vec![u64::MAX - 1]).unwrap();
        assert_eq!(big.cast::<u64>()[[0]], u64::MAX - 1);
        assert_eq!(big.cast::<i64>()[[0]], i64::MAX);
        assert_eq!(DType::parse("<u2").unwrap(), DType::U16);
    }

    #[test]
    fn slices_follow_numpy() {
        assert_eq!(Slice::new(2, 100).resolve(10).unwrap(), (2, 10, 1));
        assert_eq!(Slice::new(-3, -1).resolve(10).unwrap(), (7, 9, 1));
        assert_eq!(Slice::new(5, 2).resolve(10).unwrap(), (5, 5, 1));
        assert_eq!(Slice { start: None, stop: None, step: Some(3) }.count(10).unwrap(), 4);
    }
}
