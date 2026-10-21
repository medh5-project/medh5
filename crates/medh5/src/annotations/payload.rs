//! The intermediate an encoder produces before anything touches HDF5.
//!
//! Keeping encoders pure --- arrays in, arrays out --- is what lets the
//! transcoding matrix be tested without a file, and what lets the writer
//! decide chunking and codecs in one place instead of in every encoder.

use std::collections::BTreeMap;

use indexmap::IndexMap;
use ndarray::ArrayD;
use serde_json::{json, Value};

use crate::array::{Index, NdArray, Slice};
use crate::h5::attrs::AttrValue;
use crate::h5::data;
use crate::json::repr_int_tuple;
use crate::labels::check_class_id;
use crate::{Error, Result};

/// One dataset of a payload: numbers, or strings.
#[derive(Debug, Clone, PartialEq)]
pub enum PayloadData {
    Array(NdArray),
    Strings(Vec<String>),
}

impl PayloadData {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            PayloadData::Array(a) => a.shape(),
            PayloadData::Strings(s) => vec![s.len()],
        }
    }

    pub fn nbytes(&self) -> usize {
        match self {
            PayloadData::Array(a) => a.nbytes(),
            // An object array of references, as NumPy counts it.
            PayloadData::Strings(s) => s.len() * 8,
        }
    }

    pub fn dtype_str(&self) -> String {
        match self {
            PayloadData::Array(a) => a.dtype().numpy_str().to_string(),
            PayloadData::Strings(_) => "|O".to_string(),
        }
    }

    pub fn as_array(&self) -> Option<&NdArray> {
        match self {
            PayloadData::Array(a) => Some(a),
            PayloadData::Strings(_) => None,
        }
    }
}

impl From<NdArray> for PayloadData {
    fn from(a: NdArray) -> Self {
        PayloadData::Array(a)
    }
}

/// Datasets and kind-specific attributes for one encoded annotation.
#[derive(Debug, Clone, PartialEq)]
pub struct Payload {
    pub kind: String,
    pub datasets: IndexMap<String, PayloadData>,
    pub attrs: Vec<(String, AttrValue)>,
    /// Leading axes of `data` that must get chunk extent 1 (spec §14.1).
    pub stacked_axes: usize,
    pub class_ids: Vec<i64>,
}

impl Payload {
    /// An empty payload of `kind`.
    pub fn new(kind: &str) -> Payload {
        Payload {
            kind: kind.into(),
            datasets: IndexMap::new(),
            attrs: Vec::new(),
            stacked_axes: 0,
            class_ids: Vec::new(),
        }
    }

    /// The `data` dataset.
    pub fn data(&self) -> Result<&NdArray> {
        self.array("data")
    }

    /// A numeric dataset by name.
    pub fn array(&self, name: &str) -> Result<&NdArray> {
        match self.datasets.get(name) {
            Some(PayloadData::Array(a)) => Ok(a),
            Some(PayloadData::Strings(_)) => Err(Error::Type(format!("{name:?} holds strings"))),
            None => Err(Error::Key(crate::json::repr_str(name))),
        }
    }

    /// A kind-specific attribute.
    pub fn attr(&self, name: &str) -> Option<&AttrValue> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }

    /// Set (or replace) a kind-specific attribute.
    pub fn set_attr(&mut self, name: &str, value: AttrValue) {
        if let Some(slot) = self.attrs.iter_mut().find(|(k, _)| k == name) {
            slot.1 = value;
        } else {
            self.attrs.push((name.into(), value));
        }
    }

    /// Total bytes of every dataset.
    pub fn nbytes(&self) -> usize {
        self.datasets.values().map(PayloadData::nbytes).sum()
    }

    /// Shapes and dtypes, for reports.
    pub fn describe(&self) -> Value {
        let datasets: serde_json::Map<String, Value> = self
            .datasets
            .iter()
            .map(|(name, d)| (name.clone(), json!({"shape": d.shape(), "dtype": d.dtype_str()})))
            .collect();
        json!({"kind": self.kind, "datasets": datasets, "nbytes": self.nbytes()})
    }
}

/// Class id -> boolean occupancy over the grid's spatial shape, sorted by id.
pub type Masks = BTreeMap<i64, ArrayD<bool>>;

/// Check every mask has one agreed shape and a writable class id.
pub fn normalize_masks(masks: &Masks, spatial_shape: Option<&[usize]>) -> Result<Vec<usize>> {
    normalize_mask_shapes(masks.iter().map(|(class_id, mask)| (*class_id, mask.shape())), spatial_shape)
}

/// [`normalize_masks`] on the masks' shapes alone, `(class_id, shape)` in
/// class order: a caller holding the masks elsewhere checks them without
/// copying them into [`Masks`].
pub fn normalize_mask_shapes<'a>(
    masks: impl IntoIterator<Item = (i64, &'a [usize])>,
    spatial_shape: Option<&[usize]>,
) -> Result<Vec<usize>> {
    let mut shape: Option<Vec<usize>> = spatial_shape.map(<[usize]>::to_vec);
    for (class_id, found) in masks {
        match &shape {
            None => shape = Some(found.to_vec()),
            Some(s) if found != s.as_slice() => {
                return Err(Error::coded(
                    "E405",
                    format!(
                        "mask for class {class_id} has shape {}, expected {}",
                        repr_int_tuple(found),
                        repr_int_tuple(s)
                    ),
                ))
            }
            _ => {}
        }
        check_class_id(class_id)?;
    }
    shape.ok_or_else(|| Error::coded("E410", "no masks were supplied"))
}

/// Read budget for a scan that only needs a yes/no or a count.
pub const SLAB_BYTES: usize = 8 * 1024 * 1024;

/// Selections that read `ds` in slabs within [`SLAB_BYTES`], with `prefix`
/// fixing leading indices.  A row over the budget descends an axis instead.
pub fn slabs(shape: &[usize], itemsize: usize, prefix: &[usize]) -> Vec<Vec<Index>> {
    slabs_within(shape, itemsize, prefix, SLAB_BYTES)
}

/// [`slabs`] for a read budget of `budget` bytes.
pub fn slabs_within(shape: &[usize], itemsize: usize, prefix: &[usize], budget: usize) -> Vec<Vec<Index>> {
    let rest = &shape[prefix.len()..];
    let mut out = Vec::new();
    if rest.is_empty() || rest[0] == 0 {
        return out;
    }
    // A file states any extents, and a row past `usize` wrapped.  No read
    // holds one, in slabs or whole: the one selection of all of it is
    // refused by the read, as more than a process can hold.
    let Some(per_row) = rest[1..].iter().try_fold(itemsize, |n, e| n.checked_mul(*e)) else {
        let mut sel: Vec<Index> = prefix.iter().map(|i| Index::At(*i as i64)).collect();
        sel.push(Index::Slice(Slice::full()));
        return vec![sel];
    };
    if per_row > budget && rest.len() > 1 {
        for index in 0..rest[0] {
            let mut p = prefix.to_vec();
            p.push(index);
            out.extend(slabs_within(shape, itemsize, &p, budget));
        }
        return out;
    }
    let step = (budget / per_row.max(1)).min(rest[0]).max(1);
    let mut start = 0;
    while start < rest[0] {
        let mut sel: Vec<Index> = prefix.iter().map(|i| Index::At(*i as i64)).collect();
        sel.push(Index::Slice(Slice::new(start as i64, (start + step) as i64)));
        out.push(sel);
        start += step;
    }
    out
}

/// Whether any element of a dataset equals `value`, scanning in slabs.
pub fn contains_value(ds: &hdf5::Dataset, value: i64) -> Result<bool> {
    contains_value_within(ds, value, SLAB_BYTES)
}

/// [`contains_value`] reading at most `budget` bytes at a time where a row
/// fits --- with no cap on the dataset's size: a check that declined to look
/// past some element count would report what it had not found.
pub fn contains_value_within(ds: &hdf5::Dataset, value: i64, budget: usize) -> Result<bool> {
    let dtype = data::dtype(ds)?;
    for sel in slabs_within(&ds.shape(), dtype.itemsize(), &[], budget) {
        let block = data::read_region(ds, &sel)?;
        if block.cast::<i64>().iter().any(|v| *v == value) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether any element of a dataset is nonzero, scanning in slabs and
/// stopping at the first.
pub fn any_nonzero(ds: &hdf5::Dataset) -> Result<bool> {
    let dtype = data::dtype(ds)?;
    for sel in slabs(&ds.shape(), dtype.itemsize(), &[]) {
        if data::read_region(ds, &sel)?.nonzero_mask().iter().any(|v| *v) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The number of nonzero elements of a dataset, in slabs.
pub fn count_nonzero(ds: &hdf5::Dataset) -> Result<u64> {
    count_nonzero_within(ds, SLAB_BYTES)
}

/// [`count_nonzero`] reading at most `budget` bytes at a time where a row fits.
pub fn count_nonzero_within(ds: &hdf5::Dataset, budget: usize) -> Result<u64> {
    let dtype = data::dtype(ds)?;
    let mut total = 0u64;
    for sel in slabs_within(&ds.shape(), dtype.itemsize(), &[], budget) {
        let block = data::read_region(ds, &sel)?;
        total += block.nonzero_mask().iter().filter(|v| **v).count() as u64;
    }
    Ok(total)
}

/// How many voxels hold each value up to `ceiling`, in slabs.
pub fn value_counts(ds: &hdf5::Dataset, ceiling: i64) -> Result<BTreeMap<i64, u64>> {
    value_counts_within(ds, ceiling, SLAB_BYTES)
}

/// [`value_counts`] reading at most `budget` bytes at a time where a row fits.
pub fn value_counts_within(ds: &hdf5::Dataset, ceiling: i64, budget: usize) -> Result<BTreeMap<i64, u64>> {
    let dtype = data::dtype(ds)?;
    let mut totals = vec![0u64; (ceiling.max(0) + 1) as usize];
    for sel in slabs_within(&ds.shape(), dtype.itemsize(), &[], budget) {
        let block = data::read_region(ds, &sel)?;
        for v in block.cast::<i64>().iter() {
            if *v >= 0 && *v <= ceiling {
                totals[*v as usize] += 1;
            }
        }
    }
    Ok(totals.into_iter().enumerate().filter(|(_, c)| *c > 0).map(|(v, c)| (v as i64, c)).collect())
}

/// Per-plane population counts of a packed `uint64` bitmask: `(P, 64)`.
pub fn popcounts(ds: &hdf5::Dataset) -> Result<Vec<[u64; 64]>> {
    popcounts_within(ds, SLAB_BYTES)
}

/// [`popcounts`] reading at most `budget` bytes at a time where a row fits:
/// one plane is read slab by slab, never whole.
pub fn popcounts_within(ds: &hdf5::Dataset, budget: usize) -> Result<Vec<[u64; 64]>> {
    let shape = ds.shape();
    let planes = shape.first().copied().unwrap_or(0);
    let mut out = vec![[0u64; 64]; planes];
    for (plane, counts) in out.iter_mut().enumerate() {
        for sel in slabs_within(&shape, 8, &[plane], budget) {
            let block = data::read_region(ds, &sel)?;
            for word in block.cast::<u64>().iter() {
                let mut w = *word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    counts[bit] += 1;
                    w &= w - 1;
                }
            }
        }
    }
    Ok(out)
}

/// `np.packbits` of a flat boolean sequence: MSB-first within each byte.
pub fn packbits(bits: impl IntoIterator<Item = bool>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut byte = 0u8;
    let mut n = 0;
    for bit in bits {
        byte = (byte << 1) | u8::from(bit);
        n += 1;
        if n == 8 {
            out.push(byte);
            byte = 0;
            n = 0;
        }
    }
    if n > 0 {
        out.push(byte << (8 - n));
    }
    out
}

/// `np.unpackbits(packed)[:n]`, MSB-first.
pub fn unpackbits(packed: &[u8], n: usize) -> Vec<bool> {
    let mut out = Vec::with_capacity(n);
    for byte in packed {
        for k in (0..8).rev() {
            if out.len() == n {
                return out;
            }
            out.push((byte >> k) & 1 == 1);
        }
    }
    out.resize(n, false);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bytes one selection reads from a dataset of `shape`.
    fn selection_bytes(shape: &[usize], itemsize: usize, sel: &[Index]) -> usize {
        let mut n = itemsize;
        for (axis, extent) in shape.iter().enumerate() {
            n *= match sel.get(axis) {
                Some(Index::At(_)) => 1,
                Some(Index::Slice(s)) => {
                    let start = s.start.unwrap_or(0).max(0) as usize;
                    let stop = (s.stop.unwrap_or(*extent as i64).max(0) as usize).min(*extent);
                    stop.saturating_sub(start)
                }
                _ => *extent,
            };
        }
        n
    }

    #[test]
    fn s14_slabs_stay_within_the_budget_and_tile_the_dataset() {
        for (shape, itemsize, prefix) in [
            (vec![2, 16, 32, 32], 2, vec![]),  // layers, read whole: descends to rows
            (vec![1, 16, 32, 32], 8, vec![0]), // one bitmask plane
            (vec![3, 7, 5], 1, vec![]),        // rows smaller than the budget
        ] {
            let sels = slabs_within(&shape, itemsize, &prefix, 4096);
            assert!(sels.iter().all(|sel| selection_bytes(&shape, itemsize, sel) <= 4096), "{shape:?}");
            let whole: usize = shape[prefix.len()..].iter().product::<usize>() * itemsize;
            let read: usize = sels.iter().map(|sel| selection_bytes(&shape, itemsize, sel)).sum();
            assert_eq!(read, whole, "{shape:?}");
        }
    }

    /// A row of `2**62 x 4` two-byte values wrapped its size to 0, and the
    /// whole dataset became one slab.  It is still one selection, now on
    /// purpose: no slab of it can be held, and the read refuses it.
    #[test]
    fn s14_a_row_past_the_integers_is_one_selection_the_read_refuses() {
        let sels = slabs_within(&[3, 1 << 62, 4], 2, &[], 4096);
        assert_eq!(sels, vec![vec![Index::Slice(Slice::full())]]);
        let sels = slabs_within(&[2, 3, 1 << 62, 4], 2, &[1], 4096);
        assert_eq!(sels, vec![vec![Index::At(1), Index::Slice(Slice::full())]]);
    }

    #[test]
    fn s14_plane_counts_read_within_the_slab_budget() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("planes.h5"))).unwrap();
        let mut layers = ndarray::Array4::<u16>::zeros((2, 16, 32, 32));
        layers.slice_mut(ndarray::s![0, 2..10, 2..20, 2..20]).fill(1);
        layers.slice_mut(ndarray::s![1, .., .., 10..30]).fill(2);
        let ds = file.new_dataset_builder().with_data(&layers).create("layers").unwrap();
        let expected =
            BTreeMap::from([(0, 2 * 16 * 32 * 32 - 8 * 18 * 18 - 16 * 32 * 20), (1, 8 * 18 * 18), (2, 16 * 32 * 20)]);
        assert_eq!(value_counts_within(&ds, 2, 4096).unwrap(), expected);
        assert_eq!(value_counts(&ds, 2).unwrap(), expected);

        let mut words = ndarray::Array4::<u64>::zeros((1, 16, 32, 32));
        words.slice_mut(ndarray::s![0, 2..10, 2..20, 2..20]).fill(0b01);
        words.slice_mut(ndarray::s![0, .., .., 10..30]).mapv_inplace(|w| w | 0b10);
        let ds = file.new_dataset_builder().with_data(&words).create("bitmask").unwrap();
        let counts = popcounts_within(&ds, 4096).unwrap();
        assert_eq!((counts[0][0], counts[0][1]), (8 * 18 * 18, 16 * 32 * 20));
        assert_eq!(popcounts(&ds).unwrap(), counts);
    }

    #[test]
    fn s7_7_scans_reach_the_last_slab() {
        // One row per slab, the only ignore voxel and the only foreground in
        // the last one: a scan that stopped early would miss both.
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("scan.h5"))).unwrap();
        let mut labels = ndarray::Array3::<u16>::zeros((8, 4, 4));
        labels[[7, 3, 3]] = 65535;
        labels[[7, 3, 2]] = 1;
        let ds = file.new_dataset_builder().with_data(&labels).create("labels").unwrap();
        for budget in [1, 32, SLAB_BYTES] {
            assert!(contains_value_within(&ds, 65535, budget).unwrap(), "budget {budget}");
            assert!(!contains_value_within(&ds, 2, budget).unwrap(), "budget {budget}");
            assert_eq!(count_nonzero_within(&ds, budget).unwrap(), 2, "budget {budget}");
        }
    }

    #[test]
    fn bits_pack_like_numpy() {
        // np.packbits([1,0,1,1,0,0,0,0,1]) == [176, 128]
        let bits = [true, false, true, true, false, false, false, false, true];
        assert_eq!(packbits(bits), vec![176, 128]);
        assert_eq!(unpackbits(&[176, 128], 9), bits.to_vec());
    }
}
