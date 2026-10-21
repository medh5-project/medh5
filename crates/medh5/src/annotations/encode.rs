//! The voxel encoders (spec §7.1-§7.5) and payload-level transcoding (§7.6).
//!
//! Every encoder takes per-class boolean masks (or probabilities) and returns a
//! [`Payload`]; nothing here touches HDF5.  Transcoding decodes a payload back
//! to masks and re-encodes them, so `contains(class, voxel)` is preserved by
//! construction --- `probmap` excepted, which is lossless only under its
//! declared threshold.

use std::collections::BTreeMap;

use half::f16;
use ndarray::{ArrayD, ArrayViewD, Axis, Dimension, IxDyn, Slice as NdSlice};

use super::layout::{column_problems, mask_problems, table_problems};
use super::payload::{normalize_masks, packbits, unpackbits, Masks, Payload, PayloadData};
use super::select::{analyse, bbox, label_dtype_size, layers_from_colouring};
use crate::array::{DType, NdArray};
use crate::geometry::affine::{box_to_slices, slices_to_box};
use crate::h5::attrs::AttrValue;
use crate::json::{repr_int_list, repr_int_tuple, repr_list, repr_str};
use crate::labels::{check_class_id, IGNORE_ID};
use crate::{Error, Result};

/// Bits per `uint64` plane.
pub const BITS_PER_PLANE: usize = 64;
/// The probmap decision threshold when none is declared (§7.5).
pub const DEFAULT_THRESHOLD: f64 = 0.5;
/// The encodings a transcode may target.
pub const TRANSCODABLE: [&str; 5] = ["labelmap", "layers", "bitmask", "instances", "probmap"];
/// Encodings that can hold an ignore region in the data itself (§7.7).
pub const IN_BAND_IGNORE_KINDS: [&str; 2] = ["labelmap", "layers"];

/// Options the encoders accept; each uses only its own.
#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    /// An in-band ignore region (`labelmap`, `layers`).
    pub ignore: Option<ArrayD<bool>>,
    /// The ignore value (default 65535).
    pub ignore_id: Option<i64>,
    /// A precomputed layer colouring (`layers`).
    pub colouring: Option<BTreeMap<i64, usize>>,
    /// The narrowest storage dtype wanted (`probmap`, default float16).
    pub probmap_dtype: Option<DType>,
    /// Whether channels sum to one (`probmap`).
    pub normalized: bool,
    /// The decision threshold (`probmap`).
    pub threshold: Option<f64>,
    /// The first minted instance id (`instances` from masks).
    pub start_id: Option<u64>,
    /// Whether to store per-object masks (`instances`, default true).
    pub store_masks: Option<bool>,
}

fn first_true(mask: &ArrayD<bool>) -> Option<Vec<usize>> {
    mask.indexed_iter().find(|(_, v)| **v).map(|(idx, _)| idx.slice().to_vec())
}

/// Whether `[lo, hi]` bounds one axis of a box (§8.2): finite, and `lo` not
/// past `hi`.  A NaN fails `lo <= hi` --- and so read as a box --- and an
/// infinite bound spans no grid.
pub(crate) fn box_axis_ok(lo: f32, hi: f32) -> bool {
    lo <= hi && lo.is_finite() && hi.is_finite()
}

/// Refuse an in-band ignore region the volume cannot hold (§6.2, §7.7): a
/// region of another shape than the masks (E405), or an `ignore_id` that is
/// no `uint16`, is background, or is one of the classes (E404) --- stored
/// beside the class ids, it would wrap to another value or read as one.
fn check_ignore(ignore: Option<&ArrayD<bool>>, shape: &[usize], ignore_id: i64, class_ids: &[i64]) -> Result<()> {
    if let Some(region) = ignore {
        if region.shape() != shape {
            return Err(Error::coded(
                "E405",
                format!(
                    "ignore has shape {}; the masks' spatial shape is {}",
                    repr_int_tuple(region.shape()),
                    repr_int_tuple(shape)
                ),
            ));
        }
    }
    if !(1..=i64::from(u16::MAX)).contains(&ignore_id) {
        return Err(Error::coded(
            "E404",
            format!(
                "ignore_id {ignore_id} is outside 1..65535: it is stored as a uint16 beside the class ids, and 0 is \
                 background (§6.2)"
            ),
        ));
    }
    if class_ids.contains(&ignore_id) {
        return Err(Error::coded(
            "E404",
            format!(
                "ignore_id {ignore_id} is also a class id; one value cannot mark both the class and the ignore region"
            ),
        ));
    }
    Ok(())
}

/// Refuse an in-band ignore region that meets a class (E404, §7.7): one value
/// per voxel cannot say both "this class" and "ignored".  Painted only where
/// no class was, the region lost every voxel it shared with a class --- and in
/// `layers` it went into the voxel's other layers instead, an in-band region
/// over a class.  Such a region is a separate `mask` annotation named by
/// `ignore_mask`, as `SampleWriter::add_segmentation` writes it.  `claimed`
/// is where `classes` holds a class.
fn refuse_ignore_over_class(ignore: &ArrayD<bool>, claimed: &ArrayD<bool>, classes: ArrayViewD<'_, u16>) -> Result<()> {
    if !ndarray::Zip::from(ignore).and(claimed).fold(false, |acc, i, c| acc || (*i && *c)) {
        return Ok(());
    }
    let first = first_true(&ndarray::Zip::from(ignore).and(claimed).map_collect(|i, c| *i && *c)).unwrap_or_default();
    Err(Error::coded(
        "E404",
        format!(
            "the ignore region overlaps class {} at voxel {}: one value per voxel cannot hold both, so a region that \
             meets a class is a separate `mask` annotation named by `ignore_mask` (§7.7), which \
             SampleWriter.add_segmentation writes",
            classes[first.as_slice()],
            repr_int_tuple(&first)
        ),
    ))
}

// -- labelmap -------------------------------------------------------------------

/// Pack mutually exclusive class masks into one integer volume (§7.1).
///
/// Raises E404 when two classes claim the same voxel: silently letting the last
/// writer win would turn a curation error into a training set nobody can audit.
/// `ignore` is painted as `ignore_id`; a region that overlaps a class (E404:
/// §7.7 puts it in a sibling `mask`), a region of another shape (E405), or an
/// `ignore_id` outside `1..=65535` or equal to a class id (E404) is refused.
/// The volume is `uint16` only when a class id or a painted ignore voxel
/// needs it.
pub fn encode_labelmap(
    masks: &Masks,
    spatial_shape: Option<&[usize]>,
    ignore: Option<&ArrayD<bool>>,
    ignore_id: i64,
) -> Result<Payload> {
    let shape = normalize_masks(masks, spatial_shape)?;
    let class_ids: Vec<i64> = masks.keys().copied().collect();
    check_ignore(ignore, &shape, ignore_id, &class_ids)?;
    let mut data = ArrayD::<u16>::zeros(IxDyn(&shape));
    let mut claimed = ArrayD::from_elem(IxDyn(&shape), false);
    for (class_id, mask) in masks {
        let clash = ndarray::Zip::from(mask).and(&claimed).map_collect(|m, c| *m && *c);
        if let Some(first) = first_true(&clash) {
            return Err(Error::coded(
                "E404",
                format!(
                    "labelmap requires mutually exclusive classes; class {class_id} overlaps an earlier class at \
                     voxel {}. Use `layers` or `bitmask` for overlapping structures.",
                    repr_int_tuple(&first)
                ),
            ));
        }
        ndarray::Zip::from(&mut data).and(mask).and(&mut claimed).for_each(|d, m, c| {
            if *m {
                *d = *class_id as u16;
                *c = true;
            }
        });
    }
    // The dtype follows what is stored: an ignore region no voxel carries
    // (all false) leaves a `uint8` volume `uint8`.
    let mut ignored = false;
    if let Some(ign) = ignore {
        refuse_ignore_over_class(ign, &claimed, data.view())?;
        let value = ignore_id as u16;
        ndarray::Zip::from(&mut data).and(ign).for_each(|d, i| {
            if *i {
                *d = value;
                ignored = true;
            }
        });
    }
    let itemsize = label_dtype_size(&class_ids, ignored);
    let stored = if itemsize == 1 { NdArray::U8(data.mapv(|v| v as u8)) } else { NdArray::U16(data) };
    let mut p = Payload::new("labelmap");
    p.datasets.insert("data".into(), stored.into());
    p.class_ids = class_ids;
    Ok(p)
}

// -- layers ----------------------------------------------------------------------

/// Colour the overlap graph and pack each colour into one labelmap (§7.2).
///
/// `ignore` and `ignore_id` are checked and painted per layer as
/// [`encode_labelmap`] does.
pub fn encode_layers(
    masks: &Masks,
    spatial_shape: Option<&[usize]>,
    colouring: Option<&BTreeMap<i64, usize>>,
    ignore: Option<&ArrayD<bool>>,
    ignore_id: i64,
) -> Result<Payload> {
    let shape = normalize_masks(masks, spatial_shape)?;
    let class_ids: Vec<i64> = masks.keys().copied().collect();
    let colouring: BTreeMap<i64, usize> = match colouring {
        None => analyse(masks, Some(&shape))?.colouring,
        Some(given) => {
            let mut missing: Vec<i64> = class_ids.iter().filter(|c| !given.contains_key(c)).copied().collect();
            missing.sort();
            if !missing.is_empty() {
                return Err(Error::coded("E404", format!("colouring omits classes {}", repr_int_list(&missing))));
            }
            class_ids.iter().map(|c| (*c, given[c])).collect()
        }
    };
    check_ignore(ignore, &shape, ignore_id, &class_ids)?;
    let buckets = layers_from_colouring(&colouring);
    let n_layers = buckets.len().max(1);
    let mut ignored = false;
    let mut full_shape = vec![n_layers];
    full_shape.extend(&shape);
    let mut data = ArrayD::<u16>::zeros(IxDyn(&full_shape));
    for (layer, bucket) in buckets.iter().enumerate() {
        let mut plane = data.index_axis_mut(Axis(0), layer);
        let mut claimed = ArrayD::from_elem(IxDyn(&shape), false);
        for class_id in bucket {
            let mask = &masks[class_id];
            let overlap = ndarray::Zip::from(mask).and(&claimed).fold(false, |acc, m, c| acc || (*m && *c));
            if overlap {
                return Err(Error::coded(
                    "E404",
                    format!(
                        "classes {} were assigned to layer {layer} but overlap; the colouring is not a valid \
                         colouring of the overlap graph",
                        repr_int_tuple(bucket)
                    ),
                ));
            }
            ndarray::Zip::from(&mut plane).and(mask).and(&mut claimed).for_each(|d, m, c| {
                if *m {
                    *d = *class_id as u16;
                    *c = true;
                }
            });
        }
        if let Some(ign) = ignore {
            refuse_ignore_over_class(ign, &claimed, plane.view())?;
            let value = ignore_id as u16;
            ndarray::Zip::from(&mut plane).and(ign).for_each(|d, i| {
                if *i {
                    *d = value;
                    ignored = true;
                }
            });
        }
    }
    let itemsize = label_dtype_size(&class_ids, ignored);
    let width = buckets.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut table = ArrayD::<u16>::zeros(IxDyn(&[n_layers, width]));
    for (layer, bucket) in buckets.iter().enumerate() {
        for (k, cid) in bucket.iter().enumerate() {
            table[[layer, k]] = *cid as u16;
        }
    }
    let stored = if itemsize == 1 { NdArray::U8(data.mapv(|v| v as u8)) } else { NdArray::U16(data) };
    let mut p = Payload::new("layers");
    p.datasets.insert("data".into(), stored.into());
    p.datasets.insert("layer_class_ids".into(), NdArray::U16(table).into());
    p.stacked_axes = 1;
    p.class_ids = class_ids;
    Ok(p)
}

// -- bitmask ----------------------------------------------------------------------

/// Pack class masks into `ceil(C/64)` `uint64` bitplanes, LSB-first (§7.3).
pub fn encode_bitmask(masks: &Masks, spatial_shape: Option<&[usize]>) -> Result<Payload> {
    let shape = normalize_masks(masks, spatial_shape)?;
    let class_ids: Vec<i64> = masks.keys().copied().collect();
    let n_planes = class_ids.len().div_ceil(BITS_PER_PLANE).max(1);
    let mut full_shape = vec![n_planes];
    full_shape.extend(&shape);
    let mut data = ArrayD::<u64>::zeros(IxDyn(&full_shape));
    for (position, class_id) in class_ids.iter().enumerate() {
        let (plane, bit) = (position / BITS_PER_PLANE, position % BITS_PER_PLANE);
        let flag = 1u64 << bit;
        let mut view = data.index_axis_mut(Axis(0), plane);
        ndarray::Zip::from(&mut view).and(&masks[class_id]).for_each(|d, m| {
            if *m {
                *d |= flag;
            }
        });
    }
    let ids = ArrayD::from_shape_vec(IxDyn(&[class_ids.len()]), class_ids.iter().map(|c| *c as u16).collect())?;
    let mut p = Payload::new("bitmask");
    p.datasets.insert("data".into(), NdArray::U64(data).into());
    p.datasets.insert("bit_class_ids".into(), NdArray::U16(ids).into());
    p.stacked_axes = 1;
    p.class_ids = class_ids;
    Ok(p)
}

// -- mask --------------------------------------------------------------------------

/// Wrap a boolean volume as a `mask` payload.
pub fn encode_mask(mask: ArrayD<bool>) -> Payload {
    let mut p = Payload::new("mask");
    p.datasets.insert("data".into(), NdArray::Bool(mask).into());
    p
}

// -- probmap ---------------------------------------------------------------------

/// `values >= threshold`, decided in the precision `values` are stored in (§7.5).
///
/// The threshold is a float64 attribute and the data is usually float16;
/// rounding the threshold the way the data was rounded makes equal values
/// compare equal.
pub fn contains_at(values: &NdArray, threshold: f64) -> ArrayD<bool> {
    match values {
        NdArray::F16(a) => {
            let t = f16::from_f64(threshold);
            a.mapv(|v| v >= t)
        }
        NdArray::F32(a) => {
            let t = threshold as f32;
            a.mapv(|v| v >= t)
        }
        NdArray::F64(a) => a.mapv(|v| v >= threshold),
        other => other.to_f64().mapv(|v| v >= threshold),
    }
}

/// The narrowest allowed dtype, no narrower than `requested`, under which every
/// voxel lands on the same side of `threshold` as it was given (§7.5).
///
/// When none does --- a value within `float32` rounding of the threshold, on
/// the other side of it --- the map is refused (E411) rather than stored as
/// `float32` anyway, which moved that voxel across the threshold at write time
/// while §7.5 says a writer MUST NOT.
pub fn storage_dtype(planes: &[ArrayD<f64>], threshold: f64, requested: DType) -> Result<DType> {
    let allowed = [DType::F16, DType::F32];
    let mut candidates: Vec<DType> = allowed.iter().copied().filter(|d| d.itemsize() >= requested.itemsize()).collect();
    if candidates.is_empty() {
        candidates.push(requested);
    }
    let flipped = |arr: &ArrayD<f64>, candidate: DType| {
        let stored = contains_at(&NdArray::F64(arr.clone()).astype(candidate), threshold);
        arr.indexed_iter().zip(stored.iter()).find(|((_, v), s)| (**v >= threshold) != **s).map(|((i, v), _)| (i, *v))
    };
    for candidate in &candidates {
        if planes.iter().all(|arr| flipped(arr, *candidate).is_none()) {
            return Ok(*candidate);
        }
    }
    let widest = *candidates.last().unwrap();
    for (plane, arr) in planes.iter().enumerate() {
        if let Some((index, value)) = flipped(arr, widest) {
            return Err(Error::coded(
                "E411",
                format!(
                    "probability {} (map {plane}, voxel {}) is on the other side of the threshold {} once stored as \
                     {}, the widest dtype §7.5 allows; move it or the threshold off the rounding",
                    crate::json::py_float(value),
                    repr_int_tuple(index.slice()),
                    crate::json::py_float(threshold),
                    widest.name(),
                ),
            ));
        }
    }
    Ok(widest)
}

/// How far `sum`, the per-voxel sum of a `normalized` map's `classes` values
/// stored as `dtype`, may be from 1 (§7.5): the 1e-6 a writer allows the
/// values it is given, the rounding of each stored value --- half a unit in
/// its last place, at most `u·v` for a normal `v` and half the smallest
/// subnormal below them --- and the float64 sums of the writer and the
/// reader, `n·ε/2` of the sum each, which with the writer's own allowance for
/// its sum make three.
///
/// The rounding is bounded by the values stored, not by the class count
/// alone: `classes · epsilon` reached 1 at 1,024 float16 classes, and a map
/// whose every value was zeroed then validated as normalized.  This bound
/// stays below 2e-3 at every count a class id allows.
pub fn normalization_tolerance(dtype: DType, classes: usize, sum: f64) -> f64 {
    // The unit roundoff and the smallest positive subnormal.
    let (unit, tiny) = match dtype {
        DType::F16 => (2f64.powi(-11), 2f64.powi(-24)),
        DType::F32 => (2f64.powi(-24), 2f64.powi(-149)),
        _ => (0.0, 0.0),
    };
    let n = classes as f64;
    let sum = sum.abs();
    1e-6 + unit * sum + n * tiny / 2.0 + 3.0 * n * f64::EPSILON * sum.max(1.0)
}

/// Stack per-class probability volumes on a leading class axis (§7.5).
pub fn encode_probmap(
    probabilities: &BTreeMap<i64, ArrayD<f64>>,
    spatial_shape: Option<&[usize]>,
    dtype: DType,
    normalized: bool,
    threshold: Option<f64>,
) -> Result<Payload> {
    let mut attrs = vec![("normalized".to_string(), AttrValue::Bool(normalized))];
    if let Some(t) = threshold {
        if !(0.0..=1.0).contains(&t) || t.is_nan() {
            return Err(Error::coded("E404", format!("threshold {} must lie in [0, 1]", crate::json::py_float(t))));
        }
        attrs.push(("threshold".to_string(), AttrValue::Float(t)));
    }
    let mut class_ids = Vec::new();
    for c in probabilities.keys() {
        class_ids.push(check_class_id(*c)? as i64);
    }
    class_ids.sort();
    let decide = threshold.unwrap_or(DEFAULT_THRESHOLD);
    let mut shape: Option<Vec<usize>> = spatial_shape.map(<[usize]>::to_vec);
    let mut planes = Vec::new();
    for class_id in &class_ids {
        let arr = &probabilities[class_id];
        match &shape {
            None => shape = Some(arr.shape().to_vec()),
            Some(s) if arr.shape() != s.as_slice() => {
                return Err(Error::coded(
                    "E405",
                    format!(
                        "probability map for class {class_id} has shape {}, expected {}",
                        repr_int_tuple(arr.shape()),
                        repr_int_tuple(s)
                    ),
                ))
            }
            _ => {}
        }
        // `f64::min`/`max` skip NaN, so a fold over them passed a map of NaN:
        // every value is tested.
        if let Some(bad) = arr.iter().find(|v| !(0.0..=1.0).contains(*v)) {
            let what = if bad.is_nan() { "values that are not numbers" } else { "values outside [0, 1]" };
            return Err(Error::coded("E411", format!("probability map for class {class_id} has {what}")));
        }
        planes.push(arr.clone());
    }
    let shape = shape.ok_or_else(|| Error::coded("E410", "no probability maps were supplied"))?;
    if normalized && !planes.is_empty() {
        let mut sums = ArrayD::<f64>::zeros(IxDyn(&shape));
        for plane in &planes {
            sums += plane;
        }
        // The values as given, to 1e-6 and the error of this sum: what the
        // stored map is then held to adds only their rounding (see
        // `normalization_tolerance`), so a map written here always validates.
        let n = planes.len() as f64;
        if let Some((index, sum)) =
            sums.indexed_iter().find(|(_, s)| (**s - 1.0).abs() > 1e-6 + n * f64::EPSILON * s.abs().max(1.0))
        {
            return Err(Error::coded(
                "E404",
                format!(
                    "`normalized` is true, but the classes sum to {} at voxel {}; §7.5 has them sum to 1 at every \
                     voxel",
                    crate::json::py_float(*sum),
                    repr_int_tuple(index.slice())
                ),
            ));
        }
    }
    let chosen = storage_dtype(&planes, decide, dtype)?;
    let mut full_shape = vec![planes.len()];
    full_shape.extend(&shape);
    let mut stacked = ArrayD::<f64>::zeros(IxDyn(&full_shape));
    for (i, plane) in planes.iter().enumerate() {
        stacked.index_axis_mut(Axis(0), i).assign(plane);
    }
    let mut p = Payload::new("probmap");
    p.datasets.insert("data".into(), NdArray::F64(stacked).astype(chosen).into());
    p.attrs = attrs;
    p.stacked_axes = 1;
    p.class_ids = class_ids;
    Ok(p)
}

// -- instances --------------------------------------------------------------------

/// One object handed to [`encode_instances`].
#[derive(Debug, Clone, Default)]
pub struct InstanceInput {
    pub class_id: i64,
    pub instance_id: u64,
    /// A full-grid mask; its tight crop and box are derived.
    pub mask: Option<ArrayD<bool>>,
    /// A box, flat `[lo0, hi0, lo1, hi1, ...]`.
    pub bbox: Option<Vec<f64>>,
    /// The bbox-local crop, with `bbox`.
    pub crop: Option<ArrayD<bool>>,
    pub score: Option<f64>,
}

/// The `uint32`/`uint64` width an instance-id column needs (§7.4).
pub fn instance_id_dtype(ids: &[u64]) -> DType {
    if ids.iter().any(|i| *i > 0xFFFF_FFFF) {
        DType::U64
    } else {
        DType::U32
    }
}

/// An instance-id column in the width it needs.
pub fn instance_id_array(ids: &[u64]) -> NdArray {
    let shape = [ids.len()];
    match instance_id_dtype(ids) {
        DType::U64 => NdArray::U64(ArrayD::from_shape_vec(IxDyn(&shape), ids.to_vec()).unwrap()),
        _ => NdArray::U32(ArrayD::from_shape_vec(IxDyn(&shape), ids.iter().map(|v| *v as u32).collect()).unwrap()),
    }
}

/// Pack objects into boxes, ids, offsets and one concatenated bit stream (§7.4).
///
/// `class_ids` names classes the annotation can express beyond its objects'
/// own --- a class searched for and not found must stay in it.  The objects'
/// classes are always declared with them: §6.2 has `class_ids` hold every
/// class the data can encode, and `contains()` answers for every object's
/// class.  No objects at all is written as `N = 0` columns: "examined, none
/// found".
///
/// A `crop` given with its `bbox` is the object's mask cut to that box, so
/// its shape is the box's extent; and with `spatial_shape`, a box has the
/// grid's spatial rank and a crop no more voxels than the grid.  A reader
/// holds a stored annotation to the same (E405), so nothing written here
/// reads back as other than it was given.
pub fn encode_instances(
    objects: &[InstanceInput],
    spatial_shape: Option<&[usize]>,
    store_masks: bool,
    class_ids: Option<&[i64]>,
) -> Result<Payload> {
    if objects.is_empty() {
        let (Some(ids), Some(shape)) = (class_ids.filter(|c| !c.is_empty()), spatial_shape) else {
            return Err(Error::coded(
                "E410",
                "no instances were supplied; an empty `instances` annotation needs class_ids (what was examined) \
                 and spatial_shape",
            ));
        };
        return empty_instances(shape.len(), ids, store_masks);
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut shared = std::collections::BTreeSet::new();
    for o in objects {
        if !seen.insert(o.instance_id) {
            shared.insert(o.instance_id);
        }
    }
    if !shared.is_empty() {
        let shared: Vec<u64> = shared.into_iter().collect();
        return Err(Error::coded(
            "E404",
            format!(
                "instance id(s) {} name more than one object in this annotation; an `instance_id` names one \
                 physical object, and two distinct objects MUST NOT share one (§7.4)",
                repr_int_list(&shared)
            ),
        ));
    }
    let mut ndim: Option<usize> = None;
    let mut boxes: Vec<f32> = Vec::new();
    let mut crops: Vec<ArrayD<bool>> = Vec::new();
    let mut object_classes: Vec<i64> = Vec::new();
    let mut instance_ids: Vec<u64> = Vec::new();
    let mut scores: Vec<f32> = Vec::new();
    let has_scores = objects.iter().any(|o| o.score.is_some());
    for obj in objects {
        let mut crop = obj.crop.clone();
        let mut bx: Option<Vec<f32>> = obj.bbox.as_ref().map(|b| b.iter().map(|v| *v as f32).collect());
        if let Some(mask) = &obj.mask {
            if let Some(shape) = spatial_shape {
                if mask.shape() != shape {
                    return Err(Error::coded(
                        "E405",
                        format!(
                            "instance {}: mask shape {} != grid {}",
                            obj.instance_id,
                            repr_int_tuple(mask.shape()),
                            repr_int_tuple(shape)
                        ),
                    ));
                }
            }
            let Some(slices) = bbox(mask) else {
                return Err(Error::coded("E404", format!("instance {} has an empty mask", obj.instance_id)));
            };
            let mut view = mask.view();
            for (axis, (lo, hi)) in slices.iter().enumerate() {
                view.slice_axis_inplace(Axis(axis), NdSlice::from(*lo..*hi));
            }
            crop = Some(view.to_owned());
            let pairs: Vec<(i64, i64)> = slices.iter().map(|(a, b)| (*a as i64, *b as i64)).collect();
            bx = Some(slices_to_box(&pairs));
        }
        let Some(bx) = bx else {
            return Err(Error::coded("E410", format!("instance {} needs either a mask or a box", obj.instance_id)));
        };
        if bx.len() % 2 != 0 {
            return Err(Error::coded("E405", format!("instance {}: a box is (S, 2)", obj.instance_id)));
        }
        let s = bx.len() / 2;
        if (0..s).any(|k| !box_axis_ok(bx[2 * k], bx[2 * k + 1])) {
            return Err(Error::coded(
                "E406",
                format!("instance {}: box has lo > hi, or a bound that is not finite", obj.instance_id),
            ));
        }
        match ndim {
            None => ndim = Some(s),
            Some(n) if n != s => return Err(Error::coded("E405", "instances disagree on dimensionality")),
            _ => {}
        }
        if let Some(shape) = spatial_shape.filter(|shape| shape.len() != s) {
            return Err(Error::coded(
                "E405",
                format!(
                    "instance {}: a box with {s} axes on a grid with {} spatial axes",
                    obj.instance_id,
                    shape.len()
                ),
            ));
        }
        if let Some(c) = &crop {
            let flat: Vec<f64> = bx.iter().map(|v| f64::from(*v)).collect();
            // Saturating, as `mask_problems` does: a box from one end of the
            // integers to the other overflowed the subtraction.
            let extent: Vec<i64> = box_to_slices(&flat, None)?.iter().map(|(lo, hi)| hi.saturating_sub(*lo)).collect();
            let given: Vec<i64> = c.shape().iter().map(|v| *v as i64).collect();
            if given != extent {
                return Err(Error::coded(
                    "E405",
                    format!(
                        "instance {}: the crop is {} voxels and its box spans {}; a crop is the object's mask cut to its \
                         box",
                        obj.instance_id,
                        repr_int_tuple(&given),
                        repr_int_tuple(&extent)
                    ),
                ));
            }
            if let Some(shape) = spatial_shape.filter(|shape| c.len() > shape.iter().product::<usize>()) {
                return Err(Error::coded(
                    "E405",
                    format!(
                        "instance {}: the crop is {} voxels, more than the grid {} it is cut from",
                        obj.instance_id,
                        repr_int_tuple(&given),
                        repr_int_tuple(shape)
                    ),
                ));
            }
        }
        boxes.extend(&bx);
        object_classes.push(check_class_id(obj.class_id)? as i64);
        instance_ids.push(obj.instance_id);
        scores.push(obj.score.map(|v| v as f32).unwrap_or(f32::NAN));
        if store_masks {
            let Some(c) = crop else {
                return Err(Error::coded(
                    "E410",
                    format!("instance {}: store_masks=True needs a mask or crop", obj.instance_id),
                ));
            };
            crops.push(c);
        }
    }
    let n = objects.len();
    let s = ndim.unwrap_or(0);
    let mut p = Payload::new("instances");
    p.datasets.insert("boxes".into(), NdArray::from_vec(&[n, s, 2], boxes)?.into());
    p.datasets.insert(
        "class_ids".into(),
        NdArray::from_vec(&[n], object_classes.iter().map(|c| *c as u16).collect::<Vec<_>>())?.into(),
    );
    p.datasets.insert("instance_ids".into(), instance_id_array(&instance_ids).into());
    if has_scores {
        p.datasets.insert("scores".into(), NdArray::from_vec(&[n], scores)?.into());
    }
    if store_masks {
        let packed: Vec<Vec<u8>> = crops.iter().map(|c| packbits(c.iter().copied())).collect();
        let mut offsets = vec![0u64];
        for chunk in &packed {
            offsets.push(offsets.last().unwrap() + chunk.len() as u64);
        }
        let shapes: Vec<i32> =
            crops.iter().flat_map(|c| c.shape().iter().map(|v| *v as i32).collect::<Vec<_>>()).collect();
        let data: Vec<u8> = packed.concat();
        let crop_ndim = crops.first().map(|c| c.ndim()).unwrap_or(s);
        p.datasets.insert("mask_offsets".into(), NdArray::from_vec(&[n + 1], offsets)?.into());
        p.datasets.insert("mask_shapes".into(), NdArray::from_vec(&[n, crop_ndim], shapes)?.into());
        let len = data.len();
        p.datasets.insert("mask_data".into(), NdArray::from_vec(&[len], data)?.into());
    }
    let mut declared = object_classes;
    for c in class_ids.unwrap_or(&[]) {
        declared.push(check_class_id(*c)? as i64);
    }
    declared.sort();
    declared.dedup();
    p.class_ids = declared;
    Ok(p)
}

fn empty_instances(ndim: usize, class_ids: &[i64], store_masks: bool) -> Result<Payload> {
    let mut p = Payload::new("instances");
    p.datasets.insert("boxes".into(), NdArray::zeros(DType::F32, &[0, ndim, 2]).into());
    p.datasets.insert("class_ids".into(), NdArray::zeros(DType::U16, &[0]).into());
    p.datasets.insert("instance_ids".into(), NdArray::zeros(DType::U32, &[0]).into());
    if store_masks {
        p.datasets.insert("mask_offsets".into(), NdArray::zeros(DType::U64, &[1]).into());
        p.datasets.insert("mask_shapes".into(), NdArray::zeros(DType::I32, &[0, ndim]).into());
        p.datasets.insert("mask_data".into(), NdArray::zeros(DType::U8, &[0]).into());
    }
    let mut ids = Vec::new();
    for c in class_ids {
        ids.push(check_class_id(*c)? as i64);
    }
    ids.sort();
    ids.dedup();
    p.class_ids = ids;
    Ok(p)
}

/// The id minted for the `i`-th class from `start_id`, refused (E404) past
/// the largest `uint64` rather than wrapped to 0 --- another object's id.
pub(crate) fn minted_id(start_id: u64, i: usize) -> Result<u64> {
    start_id.checked_add(i as u64).ok_or_else(|| {
        Error::coded(
            "E404",
            format!("instance ids minted from {start_id} run past {}, the largest a uint64 holds", u64::MAX),
        )
    })
}

/// One object per class --- what a converter can honestly infer from masks
/// --- the `i`-th class's minted `start_id + i`.  An id past the largest
/// `uint64` is refused (E404).
///
/// A class whose mask is empty has no object, and none is made for it ---
/// [`encode_instances`] refuses an empty object --- but it keeps its place
/// in the numbering, as the `instances` transcode numbers objects; pass the
/// classes to [`encode_instances`] as `class_ids` to keep them examined.
pub fn instances_from_masks(masks: &Masks, start_id: u64) -> Result<Vec<InstanceInput>> {
    masks
        .iter()
        .enumerate()
        .filter(|(_, (_, mask))| mask.iter().any(|v| *v))
        .map(|(i, (class_id, mask))| {
            Ok(InstanceInput {
                class_id: *class_id,
                instance_id: minted_id(start_id, i)?,
                mask: Some(mask.clone()),
                ..Default::default()
            })
        })
        .collect()
}

/// Decode one object's bbox-local crop from the packed columns.
///
/// Columns that cannot describe the object are refused rather than read past
/// an edge or padded: an index with no offsets or no shape row, or a negative
/// or overflowing shape (E405), and offsets that do not delimit exactly the
/// bytes that shape packs into, inside `mask_data` (E408).
pub fn decode_crop(mask_data: &[u8], offsets: &[u64], shapes: &ArrayD<i64>, index: usize) -> Result<ArrayD<bool>> {
    let refuse = |code: &str, why: String| Error::coded(code, format!("object {index}: {why}"));
    let (Some(start), Some(stop)) = (offsets.get(index), offsets.get(index + 1)) else {
        return Err(refuse("E405", format!("`mask_offsets` has {} entries, none of them its own", offsets.len())));
    };
    if shapes.ndim() != 2 || index >= shapes.shape()[0] {
        return Err(refuse("E405", format!("`mask_shapes` {} has no row for it", repr_int_tuple(shapes.shape()))));
    }
    let row: Vec<i64> = shapes.index_axis(Axis(0), index).iter().copied().collect();
    let shape: Option<Vec<usize>> = row.iter().map(|v| usize::try_from(*v).ok()).collect();
    let size = shape.as_ref().and_then(|s| s.iter().try_fold(1usize, |a, b| a.checked_mul(*b)));
    let (Some(shape), Some(n)) = (shape, size) else {
        return Err(refuse("E405", format!("`mask_shapes` gives it the shape {}", repr_int_tuple(&row))));
    };
    let (start, stop) = (usize::try_from(*start).unwrap_or(usize::MAX), usize::try_from(*stop).unwrap_or(usize::MAX));
    if start > stop || stop > mask_data.len() || stop - start != n.div_ceil(8) {
        return Err(refuse(
            "E408",
            format!(
                "`mask_offsets` give it bytes {start}..{stop} of the {} in `mask_data`, and its {} mask packs into {}",
                mask_data.len(),
                repr_int_tuple(&row),
                n.div_ceil(8)
            ),
        ));
    }
    Ok(ArrayD::from_shape_vec(IxDyn(&shape), unpackbits(&mask_data[start..stop], n))?)
}

// -- decoding and transcoding -------------------------------------------------------

/// Decode any voxel payload to per-class boolean masks.
///
/// A `probmap` is thresholded at `threshold` when given, else at its own
/// declared threshold.
pub fn payload_to_masks(payload: &Payload, spatial_shape: Option<&[usize]>, threshold: Option<f64>) -> Result<Masks> {
    check_planes(payload)?;
    let mut out = Masks::new();
    match payload.kind.as_str() {
        "labelmap" => {
            let data = payload.data()?.cast::<i64>();
            for c in &payload.class_ids {
                out.insert(*c, data.mapv(|v| v == *c));
            }
        }
        "layers" => {
            let data = payload.data()?.cast::<i64>();
            let table = payload.array("layer_class_ids")?.cast::<i64>();
            for layer in 0..table.shape()[0] {
                let plane = data.index_axis(Axis(0), layer);
                for value in table.index_axis(Axis(0), layer).iter() {
                    if *value != 0 {
                        out.insert(*value, plane.mapv(|v| v == *value));
                    }
                }
            }
        }
        "bitmask" => {
            let data = payload.data()?.cast::<u64>();
            let ids = payload.array("bit_class_ids")?.cast::<i64>();
            for (position, value) in ids.iter().enumerate() {
                let (plane, bit) = (position / BITS_PER_PLANE, position % BITS_PER_PLANE);
                let flag = 1u64 << bit;
                out.insert(*value, data.index_axis(Axis(0), plane).mapv(|w| w & flag != 0));
            }
        }
        "probmap" => {
            let data = payload.data()?;
            let cut = threshold
                .or_else(|| payload.attr("threshold").and_then(AttrValue::as_f64))
                .unwrap_or(DEFAULT_THRESHOLD);
            for (i, c) in payload.class_ids.iter().enumerate() {
                let plane = crate::with_array!(data, a => NdArray::from(a.index_axis(Axis(0), i).to_owned()));
                out.insert(*c, contains_at(&plane, cut));
            }
        }
        "mask" => {
            out.insert(1, payload.data()?.nonzero_mask());
        }
        "instances" => {
            let Some(shape) = spatial_shape else {
                return Err(Error::coded("E405", "decoding `instances` needs the grid's spatial shape"));
            };
            return instances_to_masks(payload, shape);
        }
        other => return Err(Error::coded("E401", format!("cannot decode voxel kind {}", repr_str(other)))),
    }
    Ok(out)
}

/// Hold a hand-built `layers`, `bitmask` or `probmap` payload to what a
/// stored one is held to when it is opened, before a plane is indexed: a
/// class table with one row or bit per stored plane and each class once
/// (E405, E404), and one probability plane per class (E405).  A payload
/// with `class_ids` is held to them too.
fn check_planes(payload: &Payload) -> Result<()> {
    let kind = payload.kind.as_str();
    if !matches!(kind, "layers" | "bitmask" | "probmap") {
        return Ok(());
    }
    let planes = payload.datasets.get("data").map(|d| d.shape().first().copied().unwrap_or(0));
    if kind == "probmap" {
        if let Some(planes) = planes.filter(|p| *p != payload.class_ids.len()) {
            return Err(Error::coded(
                "E405",
                format!(
                    "`data` has {planes} plane(s) for {} class id(s); a probmap stores one plane per class, in \
                     `class_ids` order",
                    payload.class_ids.len()
                ),
            ));
        }
        return Ok(());
    }
    let table = if kind == "layers" { "layer_class_ids" } else { "bit_class_ids" };
    let shape_of = |name: &str| payload.datasets.get(name).map(PayloadData::shape);
    let rows: Option<Vec<Vec<i64>>> = match payload.datasets.get(table) {
        None => None,
        Some(_) => {
            let values = payload.array(table)?.cast::<i64>();
            Some(if kind == "layers" && values.ndim() == 2 {
                values.outer_iter().map(|row| row.iter().copied().collect()).collect()
            } else {
                vec![values.iter().copied().collect()]
            })
        }
    };
    let declared = (!payload.class_ids.is_empty()).then_some(payload.class_ids.as_slice());
    if let Some(problem) = table_problems(kind, &shape_of, planes, rows.as_deref(), declared).first() {
        return Err(Error::coded(problem.code, problem.message.clone()));
    }
    Ok(())
}

fn instances_to_masks(payload: &Payload, spatial_shape: &[usize]) -> Result<Masks> {
    // The columns are held to what a stored annotation is held to when it is
    // opened: a hand-built payload is input like any file.
    let shape_of = |name: &str| payload.datasets.get(name).map(PayloadData::shape);
    if let Some(problem) = column_problems("instances", &shape_of, Some(spatial_shape.len())).first() {
        return Err(Error::coded(problem.code, problem.message.clone()));
    }
    let boxes = payload.array("boxes")?.to_f64();
    let classes = payload.array("class_ids")?.cast::<i64>();
    let mut declared: Vec<i64> = payload.class_ids.clone();
    if declared.is_empty() {
        declared = classes.iter().copied().collect();
        declared.sort();
        declared.dedup();
    }
    let packed = if payload.datasets.contains_key("mask_data") {
        let offsets: Vec<i64> = payload.array("mask_offsets")?.cast::<i64>().iter().copied().collect();
        let shapes = payload.array("mask_shapes")?.cast::<i64>();
        let mask_data: Vec<u8> = payload.array("mask_data")?.cast::<u8>().iter().copied().collect();
        if let Some(problem) = mask_problems(&boxes, &offsets, &shapes, mask_data.len(), Some(spatial_shape)).first() {
            return Err(Error::coded(problem.code, problem.message.clone()));
        }
        Some((offsets.iter().map(|v| *v as u64).collect::<Vec<u64>>(), shapes, mask_data))
    } else {
        None
    };
    let mut out: Masks = declared.iter().map(|c| (*c, ArrayD::from_elem(IxDyn(spatial_shape), false))).collect();
    let n = boxes.shape()[0];
    for index in 0..n {
        let bx: Vec<f64> = boxes.index_axis(Axis(0), index).iter().copied().collect();
        let class_id = classes[[index]];
        let target = out.entry(class_id).or_insert_with(|| ArrayD::from_elem(IxDyn(spatial_shape), false));
        // Unclipped, deliberately: these slices are the frame the stored crop
        // was cut in, and the object paints the part of it inside the grid,
        // as a reader paints it.
        let mut inside = Vec::with_capacity(spatial_shape.len());
        let mut local = Vec::with_capacity(spatial_shape.len());
        for ((lo, hi), extent) in box_to_slices(&bx, None)?.into_iter().zip(spatial_shape) {
            let (a, b) = (lo.max(0), hi.min(*extent as i64));
            inside.push((a, b));
            // Saturating: no crop fits a box that starts at the end of the
            // integers (`mask_problems` holds a crop to its box), so only an
            // object without one gets here, and it does not read `local`.
            local.push((a.saturating_sub(lo), b.saturating_sub(lo)));
        }
        if inside.iter().any(|(a, b)| b <= a) {
            continue;
        }
        let mut view = target.view_mut();
        for (axis, (a, b)) in inside.iter().enumerate() {
            view.slice_axis_inplace(Axis(axis), NdSlice::from(*a as usize..*b as usize));
        }
        if let Some((offsets, shapes, mask_data)) = &packed {
            // `mask_problems` has held the crop to its box's extent.
            let crop = decode_crop(mask_data, offsets, shapes, index)?;
            let mut part = crop.view();
            for (axis, (a, b)) in local.iter().enumerate() {
                part.slice_axis_inplace(Axis(axis), NdSlice::from(*a as usize..*b as usize));
            }
            ndarray::Zip::from(&mut view).and(&part).for_each(|t, c| *t |= *c);
        } else {
            view.fill(true);
        }
    }
    Ok(out)
}

/// Encode per-class boolean masks into any voxel encoding.
pub fn encode_masks(
    masks: &Masks,
    kind: &str,
    spatial_shape: Option<&[usize]>,
    options: &EncodeOptions,
) -> Result<Payload> {
    let ignore_id = options.ignore_id.unwrap_or(IGNORE_ID);
    match kind {
        "labelmap" => encode_labelmap(masks, spatial_shape, options.ignore.as_ref(), ignore_id),
        "layers" => encode_layers(masks, spatial_shape, options.colouring.as_ref(), options.ignore.as_ref(), ignore_id),
        "bitmask" => encode_bitmask(masks, spatial_shape),
        "probmap" => {
            let probabilities: BTreeMap<i64, ArrayD<f64>> =
                masks.iter().map(|(c, m)| (*c, m.mapv(|v| if v { 1.0 } else { 0.0 }))).collect();
            encode_probmap(
                &probabilities,
                spatial_shape,
                options.probmap_dtype.unwrap_or(DType::F16),
                options.normalized,
                options.threshold,
            )
        }
        "mask" => {
            normalize_masks(masks, spatial_shape)?;
            let mut planes = masks.values();
            let Some(first) = planes.next() else {
                return Err(Error::coded("E410", "no masks were supplied"));
            };
            let mut merged = first.clone();
            for plane in planes {
                ndarray::Zip::from(&mut merged).and(plane).for_each(|m, p| *m |= *p);
            }
            Ok(encode_mask(merged))
        }
        "instances" => {
            let start_id = options.start_id.unwrap_or(1);
            let mut objects = Vec::new();
            for (i, (c, m)) in masks.iter().enumerate().filter(|(_, (_, m))| m.iter().any(|v| *v)) {
                let instance_id = minted_id(start_id, i)?;
                objects.push(InstanceInput { class_id: *c, instance_id, mask: Some(m.clone()), ..Default::default() });
            }
            let ids: Vec<i64> = masks.keys().copied().collect();
            encode_instances(&objects, spatial_shape, options.store_masks.unwrap_or(true), Some(&ids))
        }
        other => Err(Error::coded("E401", format!("cannot encode voxel kind {}", repr_str(other)))),
    }
}

/// Refuse a transcode target that cannot hold what `contains` promises.
pub fn check_target(to_kind: &str) -> Result<()> {
    if to_kind == "mask" {
        return Err(Error::coded(
            "E404",
            "cannot transcode to 'mask': a `mask` has no classes (§4.4), so every class would merge into one \
             volume and the coverage contract (`class_ids`, `annotated_class_ids`) would be lost. For a deliberate \
             union, build one with `encode_mask` and write it with `add_mask`.",
        ));
    }
    if !TRANSCODABLE.contains(&to_kind) {
        return Err(Error::coded(
            "E401",
            format!("{} is not a voxel encoding; expected one of {}", repr_str(to_kind), repr_list(&TRANSCODABLE)),
        ));
    }
    Ok(())
}

/// Object identity leaves only when the caller says so (§7.4).
pub fn check_identity(from_kind: &str, to_kind: &str, drop_identity: bool) -> Result<()> {
    if from_kind == "instances" && to_kind != "instances" && !drop_identity {
        return Err(Error::coded(
            "E404",
            format!(
                "transcoding 'instances' to {} keeps every voxel and drops every instance_id --- the field tracking \
                 joins on across visits (§7.4). Pass drop_identity=True (--drop-identity) to do it deliberately; the \
                 transcode is then recorded in the provenance.",
                repr_str(to_kind)
            ),
        ));
    }
    Ok(())
}

/// Convert a payload to another encoding, preserving `contains`.
///
/// Refused as `SampleWriter.transcode_annotation` refuses a stored one:
/// object identity cannot be made from a dense encoding (E404, §7.4), and
/// leaves `instances` only with `drop_identity`; class identity cannot go
/// into a `mask` (E404).  An in-band ignore region (§7.7) --- the voxels of
/// a `labelmap` or `layers` payload holding `options.ignore_id`, 65535 by
/// default --- is carried in band to `labelmap` and `layers`, beside any
/// `options.ignore`, and refused (E404) for an encoding that cannot hold it,
/// where it would have read as background: verified absent.
pub fn transcode_payload(
    payload: &Payload,
    to_kind: &str,
    spatial_shape: Option<&[usize]>,
    threshold: Option<f64>,
    drop_identity: bool,
    options: &EncodeOptions,
) -> Result<Payload> {
    if to_kind == payload.kind {
        return Ok(payload.clone());
    }
    check_target(to_kind)?;
    check_identity(&payload.kind, to_kind, drop_identity)?;
    if to_kind == "instances" {
        return Err(Error::coded(
            "E404",
            format!(
                "cannot transcode {} to 'instances': a dense encoding carries no object identity, so every object of \
                 a class would merge into one with a newly minted instance_id (spec §7.4). Re-derive the objects \
                 from the source that had them.",
                repr_str(&payload.kind)
            ),
        ));
    }
    let masks = payload_to_masks(payload, spatial_shape, threshold)?;
    let shape: Vec<usize> = match spatial_shape {
        Some(s) => s.to_vec(),
        None => masks.values().next().map(|m| m.shape().to_vec()).unwrap_or_default(),
    };
    let ignore_id = options.ignore_id.unwrap_or(IGNORE_ID);
    let Some(region) = in_band_ignore(payload, ignore_id)? else {
        return encode_masks(&masks, to_kind, Some(&shape), options);
    };
    if !IN_BAND_IGNORE_KINDS.contains(&to_kind) {
        return Err(Error::coded(
            "E404",
            format!(
                "cannot transcode {} to {}: the payload carries an in-band ignore region, which {} cannot hold \
                 (spec §7.7), and its voxels would read as verified absent. Keep the region as a separate `mask` \
                 annotation named by `ignore_mask`, or transcode to 'labelmap' or 'layers' instead.",
                repr_str(&payload.kind),
                repr_str(to_kind),
                repr_str(to_kind)
            ),
        ));
    }
    let mut carried = options.clone();
    carried.ignore_id = Some(ignore_id);
    carried.ignore = Some(match &options.ignore {
        Some(given) if given.shape() == region.shape() => {
            ndarray::Zip::from(given).and(&region).map_collect(|g, r| *g || *r)
        }
        // A region of another shape is the encoder's to refuse (E405).
        Some(given) => given.clone(),
        None => region,
    });
    encode_masks(&masks, to_kind, Some(&shape), &carried)
}

/// The in-band ignore region (§7.7) of a `labelmap` or `layers` payload ---
/// where any plane holds `ignore_id` --- or `None` where no voxel does.  An
/// `ignore_id` that is one of the payload's classes marks that class, not a
/// region.
fn in_band_ignore(payload: &Payload, ignore_id: i64) -> Result<Option<ArrayD<bool>>> {
    if !IN_BAND_IGNORE_KINDS.contains(&payload.kind.as_str()) || payload.class_ids.contains(&ignore_id) {
        return Ok(None);
    }
    let data = payload.data()?.cast::<i64>();
    let region = if payload.kind == "layers" && data.ndim() > 0 {
        data.map_axis(Axis(0), |lane| lane.iter().any(|v| *v == ignore_id))
    } else {
        data.mapv(|v| v == ignore_id)
    };
    Ok(region.iter().any(|v| *v).then_some(region))
}

/// Whether two mask sets agree on every class --- the losslessness check.
pub fn masks_equal(a: &Masks, b: &Masks) -> bool {
    a.len() == b.len() && a.iter().all(|(k, v)| b.get(k) == Some(v))
}

/// Whether transcoding to `to_kind` keeps what the payload holds: every
/// class mask, and the in-band ignore region (§7.7) under the default
/// `ignore_id`, decode alike from the payload and from its transcode
/// (§7.6).  A conversion [`transcode_payload`] refuses raises as it does.
pub fn check_roundtrip(payload: &Payload, to_kind: &str, spatial_shape: Option<&[usize]>) -> Result<bool> {
    let shape: Option<Vec<usize>> = match spatial_shape {
        Some(s) => Some(s.to_vec()),
        None => payload.datasets.get("data").and_then(|d| d.shape().get(payload.stacked_axes..).map(<[usize]>::to_vec)),
    };
    let original = payload_to_masks(payload, shape.as_deref(), None)?;
    let converted = transcode_payload(payload, to_kind, shape.as_deref(), None, false, &EncodeOptions::default())?;
    let decoded = payload_to_masks(&converted, shape.as_deref(), None)?;
    Ok(masks_equal(&original, &decoded)
        && in_band_ignore(payload, IGNORE_ID)? == in_band_ignore(&converted, IGNORE_ID)?)
}

/// A payload dataset as `PayloadData`, for writers building payloads by hand.
#[deprecated(since = "2.1.0", note = "medh5 3.0 removes array_data: use PayloadData::Array(array)")]
pub fn array_data(array: NdArray) -> PayloadData {
    PayloadData::Array(array)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(shape: &[usize], origin: &[usize], size: usize) -> ArrayD<bool> {
        let mut m = ArrayD::from_elem(IxDyn(shape), false);
        for (idx, v) in m.indexed_iter_mut() {
            if (0..shape.len()).all(|a| idx[a] >= origin[a] && idx[a] < origin[a] + size) {
                *v = true;
            }
        }
        m
    }

    fn masks() -> Masks {
        let shape = [16, 24, 24];
        let mut m = Masks::new();
        m.insert(1, block(&shape, &[2, 2, 2], 8));
        m.insert(2, block(&shape, &[2, 14, 2], 6));
        m.insert(3, block(&shape, &[4, 4, 4], 3));
        m
    }

    fn wide(crop: Option<ArrayD<bool>>) -> InstanceInput {
        let bbox = Some([-3e38, 3e38].repeat(3));
        InstanceInput { class_id: 1, instance_id: 1, mask: None, bbox, crop, score: None }
    }

    /// A box from one end of the integers to the other: `hi - lo`
    /// overflowed checking its crop's extent.  The extent saturates, as
    /// `mask_problems` has it, and no crop is that large.
    #[test]
    fn s7_4_a_crop_for_a_box_past_the_integers_is_e405() {
        let crop = ArrayD::from_elem(IxDyn(&[2, 2, 2]), true);
        let err = encode_instances(&[wide(Some(crop))], Some(&[4, 5, 6]), true, None).unwrap_err();
        assert_eq!(err.code(), Some("E405"));
        assert!(err.to_string().contains(&format!("box spans ({0}, {0}, {0})", i64::MAX)), "{err}");
    }

    /// Painting a box-only object from past the integers' start subtracted
    /// `i64::MIN` from the grid's first voxel.  It paints the grid.
    #[test]
    fn s7_4_a_box_past_the_integers_decodes_to_the_grid() {
        let payload = encode_instances(&[wide(None)], Some(&[4, 5, 6]), false, None).unwrap();
        let masks = payload_to_masks(&payload, Some(&[4, 5, 6]), None).unwrap();
        assert!(masks[&1].iter().all(|v| *v));
    }

    #[test]
    fn every_transcode_is_lossless() {
        let masks = masks();
        let shape = vec![16usize, 24, 24];
        for from in ["layers", "bitmask", "instances", "probmap"] {
            let payload = encode_masks(&masks, from, Some(&shape), &EncodeOptions::default()).unwrap();
            let decoded = payload_to_masks(&payload, Some(&shape), None).unwrap();
            assert!(masks_equal(&decoded, &masks), "{from}");
            for to in ["layers", "bitmask", "probmap"] {
                let opts = EncodeOptions::default();
                let out = if from == "instances" {
                    transcode_payload(&payload, to, Some(&shape), None, true, &opts).unwrap()
                } else {
                    transcode_payload(&payload, to, Some(&shape), None, false, &opts).unwrap()
                };
                let back = payload_to_masks(&out, Some(&shape), None).unwrap();
                assert!(masks_equal(&back, &masks), "{from}->{to}");
            }
        }
        assert_eq!(encode_labelmap(&masks, None, None, IGNORE_ID).unwrap_err().code(), Some("E404"));
    }

    #[test]
    fn s7_4_an_id_minted_past_the_largest_uint64_is_refused() {
        // Unchecked, `start_id + i` panicked in a debug build and wrapped to
        // 0 --- another object's id --- in a release one.
        let masks = masks();
        let shape = vec![16usize, 24, 24];
        let top = u64::MAX - 1;
        let first_two: Masks = masks.iter().take(2).map(|(c, m)| (*c, m.clone())).collect();
        let minted: Vec<u64> = instances_from_masks(&first_two, top).unwrap().iter().map(|o| o.instance_id).collect();
        assert_eq!(minted, vec![top, u64::MAX]);
        assert_eq!(instances_from_masks(&masks, top).unwrap_err().code(), Some("E404"));
        let options = EncodeOptions { start_id: Some(top), ..Default::default() };
        assert_eq!(encode_masks(&masks, "instances", Some(&shape), &options).unwrap_err().code(), Some("E404"));
    }

    #[test]
    fn probmap_widens_when_float16_would_move_a_voxel() {
        let mut probs = BTreeMap::new();
        probs.insert(1, ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0 / 3.0, 0.0, 1.0]).unwrap());
        let p = encode_probmap(&probs, None, DType::F16, false, Some(1.0 / 3.0)).unwrap();
        // float16(1/3) is 0.33325 < 1/3, so containment survives only in float16
        // because the threshold is rounded the same way.
        let data = p.data().unwrap();
        assert_eq!(data.dtype(), DType::F16);
        let back = payload_to_masks(&p, None, None).unwrap();
        assert_eq!(back[&1].iter().copied().collect::<Vec<_>>(), vec![true, false, true]);
    }

    /// The stored sums, as the validator forms them: class by class, in
    /// float64, from the values the payload stores.
    fn stored_sums(p: &Payload) -> Vec<f64> {
        let data = p.data().unwrap().to_f64();
        let classes = data.shape()[0];
        let voxels = data.len() / classes;
        let mut sums = vec![0.0; voxels];
        for (k, v) in data.iter().enumerate() {
            sums[k % voxels] += *v;
        }
        sums
    }

    /// The allowance is bounded by the values stored, at every class count a
    /// class id allows: `classes · epsilon` was 1.000001 at 1,024 float16
    /// classes, and a map whose every value was zeroed validated as normalized.
    #[test]
    fn s7_5_no_class_count_admits_a_lost_mass() {
        for dtype in [DType::F16, DType::F32] {
            for classes in [1, 2, 1024, 65536] {
                assert!(normalization_tolerance(dtype, classes, 0.0) < 2e-3, "{dtype:?}, {classes}");
                assert!(normalization_tolerance(dtype, classes, 1.0) < 3e-3, "{dtype:?}, {classes}");
            }
        }
        assert!(normalization_tolerance(DType::F16, 1024, 0.0) < 1e-4);
        assert!(normalization_tolerance(DType::F32, 1024, 1.0) < 2e-6);
    }

    /// Whatever the writer accepts, its stored map meets the bound: here 1,000
    /// classes each just below the midpoint above a float16 power of two, so
    /// every one rounds down by almost half a unit in its last place --- the
    /// worst float16 can do --- and a softmax over 1,024 classes.
    #[test]
    fn s7_5_a_map_the_writer_stores_meets_the_bound() {
        let shape = [2usize, 2];
        let b = 2f64.powi(-10);
        let below = b + 2f64.powi(-21) * (1.0 - 2f64.powi(-20));
        let mut worst: BTreeMap<i64, ArrayD<f64>> =
            (1..=1000).map(|c| (c, ArrayD::from_elem(IxDyn(&shape), below))).collect();
        worst.insert(1001, ArrayD::from_elem(IxDyn(&shape), 1.0 - 1000.0 * below));
        let mut softmax: BTreeMap<i64, ArrayD<f64>> = BTreeMap::new();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let logits: Vec<f64> = (0..1024)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 10_000) as f64 / 1_000.0
            })
            .collect();
        let total: f64 = logits.iter().map(|l| l.exp()).sum();
        for (c, l) in logits.iter().enumerate() {
            softmax.insert(c as i64 + 1, ArrayD::from_elem(IxDyn(&shape), l.exp() / total));
        }
        for planes in [worst, softmax] {
            let n = planes.len();
            let p = encode_probmap(&planes, None, DType::F16, true, None).unwrap();
            assert_eq!(p.data().unwrap().dtype(), DType::F16);
            for sum in stored_sums(&p) {
                assert!((sum - 1.0).abs() <= normalization_tolerance(DType::F16, n, sum), "{n} classes sum to {sum}");
            }
        }
        // The worst case is not slack the bound has to spare.
        let p = encode_probmap(
            &(1..=1000).map(|c| (c, ArrayD::from_elem(IxDyn(&shape), below))).collect(),
            None,
            DType::F16,
            false,
            None,
        )
        .unwrap();
        let lost = 1000.0 * below - stored_sums(&p)[0];
        assert!(lost > 4.7e-4 && lost <= normalization_tolerance(DType::F16, 1000, 1.0), "{lost}");
    }
}
