//! The sampling index: what a patch sampler would otherwise recompute (§14.3).
//!
//! A cached subsample of foreground coordinates answers "where is class c?"
//! in O(1) in volume size.  Every entry carries `source_digest`, so a stale
//! entry is detectable rather than silently wrong (§13.3).

use std::collections::HashMap;
use std::sync::OnceLock;

use indexmap::IndexMap;
use ndarray::{Array2, Array3, ArrayD, Axis, IxDyn};
use serde_json::{json, Map, Value};

use super::codecs::{dataset_layout, CodecProfile, Role};
use crate::annotations::read::walk_rows;
use crate::annotations::Annotation;
use crate::array::{Index, NdArray, Slice};
use crate::geometry::affine::slices_to_box;
use crate::h5::attrs::{self, AttrValue};
use crate::h5::{data, ops};
use crate::json::repr_str;
use crate::labels::ClassKey;
use crate::rng::Rng;
use crate::{Error, Result};

/// Coordinates cached per class.
pub const DEFAULT_MAX_COORDS: usize = 4096;
/// Block edge of the occupancy map.
pub const DEFAULT_OCCUPANCY_FACTOR: usize = 8;

/// The datasets of one annotation's index entry.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexPayload {
    pub ann_id: String,
    pub class_ids: Vec<u16>,
    pub voxel_counts: Vec<i64>,
    /// `(C, S, 2)`, NaN for an empty class.
    pub class_bboxes: Array3<f32>,
    pub fg_coords: IndexMap<i64, Array2<i32>>,
    pub occupancy: Option<ArrayD<bool>>,
    pub source_digest: Option<String>,
    pub max_coords: usize,
    pub seed: u64,
    pub total_foreground: i64,
}

/// Uniformly subsample foreground voxel coordinates.
///
/// Every foreground voxel when there are at most `max_coords`; otherwise
/// `max_coords` of them drawn without replacement, in C order.
pub fn sample_foreground_coords(mask: &ArrayD<bool>, max_coords: usize, rng: &mut Rng) -> Result<Array2<i32>> {
    let mask = mask.as_standard_layout();
    let values = mask.as_slice().expect("standard layout");
    let mut picker = Picker::new(values.iter().filter(|v| **v).count(), max_coords, rng)?;
    for (flat, v) in values.iter().enumerate() {
        if *v {
            picker.offer(flat);
        }
    }
    picker.coords(mask.shape())
}

/// The foreground voxels [`sample_foreground_coords`] keeps of one class,
/// offered in C order: every one, or the draw it made from the count.
struct Picker {
    picks: Option<Vec<usize>>,
    ordinal: usize,
    next: usize,
    chosen: Vec<usize>,
}

impl Picker {
    /// The draw is made here, from the class's voxel count alone, so a
    /// class's picks depend only on its count and the classes drawn before.
    fn new(total: usize, max_coords: usize, rng: &mut Rng) -> Result<Picker> {
        let picks = if total == 0 || total <= max_coords {
            None
        } else {
            let mut p = rng.sample_without_replacement(total, max_coords)?;
            p.sort_unstable();
            Some(p)
        };
        Ok(Picker { picks, ordinal: 0, next: 0, chosen: Vec::with_capacity(total.min(max_coords)) })
    }

    fn offer(&mut self, flat: usize) {
        match &self.picks {
            None => self.chosen.push(flat),
            Some(p) => {
                if self.next < p.len() && p[self.next] == self.ordinal {
                    self.chosen.push(flat);
                    self.next += 1;
                }
            }
        }
        self.ordinal += 1;
    }

    /// The kept voxels' `(n, S)` indices in a volume of `shape`.
    fn coords(self, shape: &[usize]) -> Result<Array2<i32>> {
        let ndim = shape.len();
        let unravel = |flat: usize| -> Vec<i32> {
            let mut rest = flat;
            let mut out = vec![0i32; ndim];
            for axis in (0..ndim).rev() {
                out[axis] = (rest % shape[axis]) as i32;
                rest /= shape[axis];
            }
            out
        };
        let flat: Vec<i32> = self.chosen.into_iter().flat_map(unravel).collect();
        Ok(Array2::from_shape_vec((flat.len() / ndim.max(1), ndim), flat)?)
    }
}

/// Low-resolution "is there anything in this block" map, for rejection
/// sampling: one bit per `factor`-cube of voxels.  A factor of 0 has no
/// blocks, and is refused (`ValueError`).
pub fn occupancy(mask: &ArrayD<bool>, factor: usize) -> Result<ArrayD<bool>> {
    if factor == 0 {
        return Err(Error::Value("an occupancy factor must be at least 1".into()));
    }
    let mask = mask.as_standard_layout();
    let mut tally = Tally::new(mask.shape(), Some(factor));
    if mask.ndim() == 0 {
        // One voxel, and one block holding it.
        if mask.iter().any(|v| *v) {
            tally.occupied.as_mut().expect("a factor").fill(true);
        }
    } else {
        walk_rows(mask.as_slice().expect("standard layout"), mask.shape(), |row, outer, k, v| {
            if *v {
                tally.add(row, outer, k);
            }
        });
    }
    tally.occupancy()
}

/// One class's count, bounds and occupancy, gathered a voxel at a time in
/// one pass --- the rows of the last axis in C order, as
/// [`Annotation::scan_classes`] and [`walk_rows`] hand them out.
struct Tally {
    count: u64,
    bounds: Vec<(usize, usize)>,
    /// The occupancy map's shape, its C-order bits, and the block edge.
    coarse: Vec<usize>,
    occupied: Option<Vec<bool>>,
    factor: usize,
    /// The row whose occupancy offset `base` is: rows come in order.
    row: Option<usize>,
    base: usize,
}

impl Tally {
    fn new(shape: &[usize], factor: Option<usize>) -> Tally {
        let factor = factor.unwrap_or(0);
        let coarse: Vec<usize> = shape.iter().map(|n| n.div_ceil(factor.max(1)).max(1)).collect();
        let occupied = (factor > 0).then(|| vec![false; coarse.iter().product()]);
        Tally { count: 0, bounds: vec![(usize::MAX, 0); shape.len()], coarse, occupied, factor, row: None, base: 0 }
    }

    fn add(&mut self, row: usize, outer: &[usize], k: usize) {
        self.count += 1;
        for ((lo, hi), index) in self.bounds.iter_mut().zip(outer.iter().chain([&k])) {
            *lo = (*lo).min(*index);
            *hi = (*hi).max(index + 1);
        }
        if let Some(occupied) = self.occupied.as_mut() {
            if self.row != Some(row) {
                // The block row's offset: C order over the coarse shape.
                let mut base = 0;
                for (index, extent) in outer.iter().zip(&self.coarse) {
                    base = base * extent + index / self.factor;
                }
                self.base = base * self.coarse.last().copied().unwrap_or(1);
                self.row = Some(row);
            }
            occupied[self.base + k / self.factor] = true;
        }
    }

    /// Tight `(start, stop)` per axis, `None` for an empty class.
    fn bbox(&self) -> Option<Vec<(usize, usize)>> {
        (self.count > 0).then(|| self.bounds.clone())
    }

    fn occupancy(self) -> Result<ArrayD<bool>> {
        let bits = self.occupied.unwrap_or_default();
        Ok(ArrayD::from_shape_vec(IxDyn(&self.coarse), bits)?)
    }
}

/// Compute the sampling index for one voxel annotation.
///
/// `labelmap`, `layers` and `bitmask` are read a stored plane at a time:
/// one pass counts, bounds and blocks every class, the coordinate draws are
/// made in class order from the counts, and a second pass keeps the drawn
/// voxels --- the entries the per-class reading gives, from two reads of
/// each plane rather than four passes and a read per class.
pub fn build_index(
    annotation: &Annotation,
    classes: Option<&[ClassKey]>,
    max_coords: usize,
    occupancy_factor: Option<usize>,
    seed: u64,
    source_digest: Option<String>,
) -> Result<IndexPayload> {
    let ids = annotation.resolve_classes(classes)?;
    let mut rng = Rng::new(seed);
    let shape = annotation.spatial_shape()?;
    // `fg_coords` are `int32` (§14.3): an index past them wrapped, so a grid
    // with one has no index.
    if let Some(n) = shape.iter().find(|n| **n > 1 << 31) {
        return Err(Error::Value(format!(
            "annotation {}: an index stores int32 coordinates (§14.3), and an axis of {n} voxels has indices past \
             them",
            repr_str(&annotation.ann_id)
        )));
    }
    let factor = occupancy_factor.filter(|f| *f > 0);
    let mut tallies: Vec<Tally> = ids.iter().map(|_| Tally::new(&shape, factor)).collect();
    let mut picked: Vec<Array2<i32>> = Vec::with_capacity(ids.len());
    if annotation.scan_classes(&ids, &mut |entry, row, outer, k| tallies[entry].add(row, outer, k))? {
        let mut pickers: Vec<Picker> =
            tallies.iter().map(|t| Picker::new(t.count as usize, max_coords, &mut rng)).collect::<Result<_>>()?;
        let width = shape.last().copied().unwrap_or(1);
        annotation.scan_classes(&ids, &mut |entry, row, _, k| pickers[entry].offer(row * width + k))?;
        for picker in pickers {
            picked.push(picker.coords(&shape)?);
        }
    } else {
        let window = annotation.window(None)?;
        for (entry, class_id) in ids.iter().enumerate() {
            let mask = annotation.dense_class(*class_id, &window)?;
            let mask = mask.as_standard_layout();
            let values = mask.as_slice().expect("standard layout");
            let tally = &mut tallies[entry];
            walk_rows(values, &shape, |row, outer, k, v| {
                if *v {
                    tally.add(row, outer, k);
                }
            });
            let mut picker = Picker::new(tally.count as usize, max_coords, &mut rng)?;
            for (flat, v) in values.iter().enumerate() {
                if *v {
                    picker.offer(flat);
                }
            }
            picked.push(picker.coords(&shape)?);
        }
    }
    let mut counts = vec![0i64; ids.len()];
    let mut bboxes = Array3::<f32>::zeros((ids.len(), shape.len(), 2));
    let mut coords = IndexMap::new();
    let mut planes = Vec::new();
    for (i, ((class_id, tally), found)) in ids.iter().zip(tallies).zip(picked).enumerate() {
        counts[i] = tally.count as i64;
        match tally.bbox() {
            Some(b) => {
                let slices: Vec<(i64, i64)> = b.iter().map(|(lo, hi)| (*lo as i64, *hi as i64)).collect();
                for (k, v) in slices_to_box(&slices).into_iter().enumerate() {
                    bboxes[[i, k / 2, k % 2]] = v;
                }
            }
            None => bboxes.index_axis_mut(Axis(0), i).fill(f32::NAN),
        }
        coords.insert(*class_id, found);
        if factor.is_some() {
            planes.push(tally.occupancy()?);
        }
    }
    let occupancy = if planes.is_empty() {
        None
    } else {
        let views: Vec<_> = planes.iter().map(|p| p.view()).collect();
        Some(ndarray::stack(Axis(0), &views)?)
    };
    Ok(IndexPayload {
        ann_id: annotation.ann_id.clone(),
        class_ids: ids.iter().map(|c| *c as u16).collect(),
        total_foreground: counts.iter().sum(),
        voxel_counts: counts,
        class_bboxes: bboxes,
        fg_coords: coords,
        occupancy,
        source_digest,
        max_coords,
        seed,
    })
}

/// Write an index entry under `index/<ann_id>`, through the label codec.
pub fn write_index(root: &hdf5::Group, payload: &IndexPayload, codec: &CodecProfile) -> Result<hdf5::Group> {
    let index_root = match ops::child_group(root, "index") {
        Some(g) => g,
        None => ops::new_group(root, "index")?,
    };
    if ops::exists(&index_root, &payload.ann_id) {
        ops::unlink(&index_root, &payload.ann_id)?;
    }
    let group = ops::new_group(&index_root, &payload.ann_id)?;
    let store = |parent: &hdf5::Group, name: &str, array: NdArray, chunks: Option<Vec<usize>>| -> Result<()> {
        let layout = dataset_layout(&array.shape(), array.dtype().itemsize(), codec, Role::Label, chunks);
        data::create(parent, name, &array, &layout)?;
        Ok(())
    };
    store(
        &group,
        "class_ids",
        NdArray::from(ArrayD::from_shape_vec(IxDyn(&[payload.class_ids.len()]), payload.class_ids.clone())?),
        None,
    )?;
    store(
        &group,
        "voxel_counts",
        NdArray::from(ArrayD::from_shape_vec(IxDyn(&[payload.voxel_counts.len()]), payload.voxel_counts.clone())?),
        None,
    )?;
    store(&group, "class_bboxes", NdArray::from(payload.class_bboxes.clone().into_dyn()), None)?;
    let coords = ops::new_group(&group, "fg_coords")?;
    for (class_id, arr) in &payload.fg_coords {
        store(&coords, &class_id.to_string(), NdArray::from(arr.clone().into_dyn()), None)?;
    }
    if let Some(occ) = &payload.occupancy {
        let mut chunks = vec![1];
        chunks.extend(&occ.shape()[1..]);
        store(&group, "occupancy", NdArray::from(occ.clone()), Some(chunks))?;
    }
    if let Some(d) = &payload.source_digest {
        attrs::write(&group, "source_digest", &AttrValue::Str(d.clone()))?;
    }
    attrs::write(&group, "max_coords", &AttrValue::Int(payload.max_coords as i64))?;
    attrs::write(&group, "seed", &AttrValue::Int(payload.seed as i64))?;
    Ok(group)
}

/// The index's class ids, and each id's row.
type ClassTable = (Vec<i64>, HashMap<i64, usize>);

/// Reader for one `index/<ann_id>` entry.  Small tables are read once.
#[derive(Debug)]
pub struct SamplingIndex {
    pub ann_id: String,
    pub group: hdf5::Group,
    class_ids: OnceLock<Result<ClassTable>>,
    counts: OnceLock<Result<IndexMap<i64, i64>>>,
}

impl SamplingIndex {
    pub fn new(ann_id: &str, group: hdf5::Group) -> SamplingIndex {
        SamplingIndex { ann_id: ann_id.into(), group, class_ids: OnceLock::new(), counts: OnceLock::new() }
    }

    /// The entry's group, refused once its file is closed: a closed group
    /// has no members, which reads as an entry without its tables.
    fn live(&self) -> Result<&hdf5::Group> {
        crate::h5::alive(&self.group)?;
        Ok(&self.group)
    }

    fn table(&self) -> Result<&(Vec<i64>, HashMap<i64, usize>)> {
        let group = self.live()?;
        match self.class_ids.get_or_init(|| {
            let ds = group.dataset("class_ids")?;
            let ids: Vec<i64> = data::read(&ds)?.cast::<i64>().iter().copied().collect();
            let positions = ids.iter().enumerate().map(|(i, c)| (*c, i)).collect();
            Ok((ids, positions))
        }) {
            Ok(t) => Ok(t),
            Err(e) => Err(e.clone()),
        }
    }

    /// The indexed classes, in stored order.
    pub fn class_ids(&self) -> Result<Vec<i64>> {
        Ok(self.table()?.0.clone())
    }

    /// Whether a class is indexed.
    pub fn has_class(&self, class_id: i64) -> Result<bool> {
        Ok(self.table()?.1.contains_key(&class_id))
    }

    /// The digest of the annotation this entry was built from.
    pub fn source_digest(&self) -> Result<Option<String>> {
        attrs::get_str(&self.group, "source_digest")
    }

    /// Coordinates cached per class.
    pub fn max_coords(&self) -> Result<i64> {
        Ok(attrs::get_i64(&self.group, "max_coords")?.unwrap_or(DEFAULT_MAX_COORDS as i64))
    }

    /// Voxel count per class.
    pub fn voxel_counts(&self) -> Result<IndexMap<i64, i64>> {
        let group = self.live()?;
        match self.counts.get_or_init(|| {
            let ids = self.class_ids()?;
            let counts: Vec<i64> = data::read(&group.dataset("voxel_counts")?)?.cast::<i64>().iter().copied().collect();
            Ok(ids.into_iter().zip(counts).collect())
        }) {
            Ok(c) => Ok(c.clone()),
            Err(e) => Err(e.clone()),
        }
    }

    /// A class's `(S, 2)` bounding box, `None` when empty.
    pub fn bbox(&self, class_id: i64) -> Result<Option<Array2<f32>>> {
        let Some(position) = self.table()?.1.get(&class_id).copied() else {
            return Err(Error::Value(format!("index {} has no class {class_id}", repr_str(&self.ann_id))));
        };
        let row =
            data::read_region(&self.live()?.dataset("class_bboxes")?, &[Index::At(position as i64)])?.cast::<f32>();
        if row.iter().any(|v| v.is_nan()) {
            return Ok(None);
        }
        let s = row.len() / 2;
        Ok(Some(row.into_shape_with_order((s, 2))?))
    }

    fn coord_node(&self, class_id: i64) -> Result<hdf5::Dataset> {
        let group = self.live()?.group("fg_coords")?;
        ops::child_dataset(&group, &class_id.to_string()).ok_or_else(|| {
            Error::Key(format!("index {} has no coordinates for class {class_id}", repr_str(&self.ann_id)))
        })
    }

    /// Every cached coordinate of a class, `(N, S)`.
    pub fn coords(&self, class_id: i64) -> Result<Array2<i32>> {
        let a = data::read(&self.coord_node(class_id)?)?.cast::<i32>();
        let n = a.shape().first().copied().unwrap_or(0);
        let s = a.shape().get(1).copied().unwrap_or(0);
        Ok(a.into_shape_with_order((n, s))?)
    }

    /// Draw `n` foreground voxel coordinates in O(1) time and memory.
    pub fn sample_foreground(&self, class_id: i64, n: usize, rng: &mut Rng) -> Result<Array2<i32>> {
        let pool = self.coord_node(class_id)?;
        let size = pool.shape().first().copied().unwrap_or(0);
        if size == 0 {
            return Err(Error::invalid(format!(
                "index {}: class {class_id} has no foreground voxels",
                repr_str(&self.ann_id)
            )));
        }
        let picks = rng.integers(0, size as i64, n)?;
        if n == 1 {
            let row = data::read_region(&pool, &[Index::At(picks[0])])?.cast::<i32>();
            let s = row.len();
            return Ok(row.into_shape_with_order((1, s))?);
        }
        let all = self.coords(class_id)?;
        let s = all.ncols();
        let mut out = Array2::<i32>::zeros((n, s));
        for (i, p) in picks.iter().enumerate() {
            out.row_mut(i).assign(&all.row(*p as usize));
        }
        Ok(out)
    }

    /// Sampling weights derived from `voxel_counts`.
    pub fn class_weights(&self, mode: &str) -> Result<IndexMap<i64, f64>> {
        let counts = self.voxel_counts()?;
        match mode {
            "uniform" => Ok(counts.keys().map(|c| (*c, 1.0)).collect()),
            "inverse_frequency" => {
                let weights: IndexMap<i64, f64> =
                    counts.iter().map(|(c, n)| (*c, if *n != 0 { 1.0 / *n as f64 } else { 0.0 })).collect();
                let total: f64 = weights.values().sum();
                Ok(if total > 0.0 {
                    weights.into_iter().map(|(c, w)| (c, w / total)).collect()
                } else {
                    counts.keys().map(|c| (*c, 0.0)).collect()
                })
            }
            other => Err(Error::invalid(format!("unknown weighting mode {}", repr_str(other)))),
        }
    }

    /// Whether this entry was built from the annotation with this digest.
    ///
    /// The caller computes `annotation_digest` under the algorithm the
    /// entry's `source_digest` names, its prefix (§13.3) ---
    /// `group_digest(group, root, algo)` --- which need not be the file's
    /// `digest_algo`.
    pub fn is_current(&self, annotation_digest: &str) -> Result<bool> {
        Ok(self.source_digest()?.as_deref() == Some(annotation_digest))
    }

    /// Whether an occupancy map is stored.
    pub fn has_occupancy(&self) -> Result<bool> {
        Ok(ops::exists(self.live()?, "occupancy"))
    }

    /// Read one class's occupancy plane.
    pub fn occupancy_plane(&self, position: usize) -> Result<Option<ArrayD<bool>>> {
        match ops::child_dataset(self.live()?, "occupancy") {
            None => Ok(None),
            Some(ds) => Ok(Some(
                data::read_region(&ds, &[Index::At(position as i64), Index::Slice(Slice::full())])?.nonzero_mask(),
            )),
        }
    }

    pub fn summary(&self) -> Result<Value> {
        let counts: Map<String, Value> = self.voxel_counts()?.iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        Ok(json!({
            "id": self.ann_id,
            "classes": self.class_ids()?,
            "voxel_counts": counts,
            "max_coords": self.max_coords()?,
            "has_occupancy": self.has_occupancy()?,
            "source_digest": self.source_digest()?,
        }))
    }
}

/// Every index entry under a sample root, by annotation id.
pub fn read_indices(root: &hdf5::Group) -> Result<IndexMap<String, SamplingIndex>> {
    let Some(node) = ops::child_group(root, "index") else {
        return Ok(IndexMap::new());
    };
    let mut out = IndexMap::new();
    for name in ops::members(&node)? {
        if let Some(g) = ops::child_group(&node, &name) {
            out.insert(name.clone(), SamplingIndex::new(&name, g));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `fg_coords` are `int32`, and a coordinate past them wrapped: a grid
    /// with an axis of more than `2**31` voxels has no index.
    #[test]
    fn s14_3_a_grid_past_int32_coordinates_has_no_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.medh5");
        crate::annotations::read::tests::wide_box(&path);
        {
            let file = crate::h5::file::open_rw(&path).unwrap();
            let grid = file.group("grids/g").unwrap();
            attrs::write(&grid, "shape", &AttrValue::ints(&[(1 << 31) + 1, 5, 6])).unwrap();
            drop(grid);
            file.close().unwrap();
        }
        let sample = crate::sample::open_sample(&path).unwrap();
        let err = build_index(sample.annotation("lesions").unwrap(), None, 16, None, 0, None).unwrap_err();
        assert!(err.to_string().contains("an axis of 2147483649 voxels"), "{err}");
    }
}
