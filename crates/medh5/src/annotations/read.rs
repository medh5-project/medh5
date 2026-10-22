//! The uniform read contract (spec §6, §7.6): one [`Annotation`] per stored
//! group, whose methods dispatch on its `kind`.
//!
//! Every voxel encoding answers the same predicate ---
//! `contains(class, voxel) -> bool` --- so callers ask for classes and a
//! region of interest and never for a layout.  The **coverage contract** rides
//! alongside: `class_ids` is what an annotation can express,
//! `annotated_class_ids` what the annotator committed to finding, and `0` at a
//! voxel means "verified absent" only for the second.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use indexmap::IndexMap;
use ndarray::{Array2, ArrayD, Axis, IxDyn};
use serde_json::{json, Map, Value};

use super::encode::{contains_at, decode_crop, BITS_PER_PLANE, DEFAULT_THRESHOLD};
use super::header::{is_geometric_kind, is_voxel_kind, AnnotationHeader};
use super::layout;
use super::payload::{any_nonzero, contains_value, count_nonzero, popcounts, value_counts};
use crate::array::{DType, Element, Index, NdArray, Slice};
use crate::geometry::affine::{box_to_slices, slices_to_box};
use crate::geometry::grid::Grid;
use crate::h5::{attrs, data, ops};
use crate::json::{py_float, repr_str};
use crate::labels::{ClassKey, LabelClass, LabelSet, BACKGROUND_ID, IGNORE_ID};
use crate::{with_array, Error, Result};

/// The grids an annotation can be read on, by id.
pub type Grids = IndexMap<String, Grid>;

/// A grid named in a call: the annotation's own, one by id, or one in hand.
#[derive(Debug, Clone, Copy)]
pub enum GridRef<'a> {
    Own,
    Id(&'a str),
    Grid(&'a Grid),
}

/// One physical object: a box, a class, an id and optionally a cropped mask.
///
/// `instance_id` is **sample-scoped**: the same lesion observed at several
/// timepoints reuses its id, so lesion tracking is a join on this field.
#[derive(Debug, Clone, PartialEq)]
pub struct Instance {
    pub index: usize,
    pub instance_id: u64,
    pub class_id: i64,
    /// `(S, 2)` box in continuous index coordinates.
    pub bbox: Array2<f32>,
    pub mask: Option<ArrayD<bool>>,
    pub score: Option<f64>,
}

impl Instance {
    /// The box as index slices, unclipped.
    pub fn slices(&self) -> Result<Vec<(i64, i64)>> {
        let flat: Vec<f64> = self.bbox.iter().map(|v| f64::from(*v)).collect();
        box_to_slices(&flat, None)
    }

    /// Foreground voxels: the mask's, or the box's when there is no mask.
    pub fn voxel_count(&self) -> Result<u64> {
        match &self.mask {
            Some(mask) => Ok(mask.iter().filter(|v| **v).count() as u64),
            // Exact or refused, never wrapped: an axis from one end of the
            // integers to the other spans 2**64 - 1 voxels, which a `u64`
            // holds, and two such axes more than one counts.
            None => self.slices()?.iter().try_fold(1u64, |n, (a, b)| {
                let extent = (i128::from(*b) - i128::from(*a)).max(0) as u64;
                n.checked_mul(extent).ok_or_else(|| {
                    Error::Value(format!("{}: its box spans more voxels than 64 bits count", self.repr()))
                })
            }),
        }
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        format!("Instance(id={}, class={}, box={})", self.instance_id, self.class_id, box_repr(&self.bbox))
    }
}

/// `box.tolist()` of a float32 `(S, 2)` array, as Python prints it.
pub fn box_repr(bbox: &Array2<f32>) -> String {
    let rows: Vec<String> = bbox
        .outer_iter()
        .map(|row| format!("[{}]", row.iter().map(|v| py_float(f64::from(*v))).collect::<Vec<_>>().join(", ")))
        .collect();
    format!("[{}]", rows.join(", "))
}

/// The `layers` table: `(L, K)` class ids and class id -> layer.
#[derive(Debug)]
struct LayerTable {
    table: Array2<u16>,
    layer_of: IndexMap<i64, usize>,
}

/// The `bitmask` table: bit position -> class id, and its inverse.
#[derive(Debug)]
struct BitTable {
    ids: Vec<u16>,
    position_of: HashMap<i64, usize>,
}

/// The per-object columns of an `instances` annotation, read once.
#[derive(Debug)]
struct InstanceColumns {
    boxes: ArrayD<f32>,
    class_ids: Vec<u16>,
    instance_ids: Vec<u64>,
    scores: Option<Vec<f32>>,
    mask_offsets: Option<Vec<u64>>,
    mask_shapes: Option<ArrayD<i64>>,
}

/// One annotation group, opened (spec §6).
///
/// The small per-kind tables --- `layers`' layer table, `bitmask`'s bit
/// table, `instances`' object columns --- whether any voxel is ignored, and
/// the hierarchy a `closure = "implicit"` annotation is read under are read
/// once per open annotation and kept: a sample is read-only and `amend`
/// replaces the inode rather than editing in place, so an open reader cannot
/// see them change.
#[derive(Debug)]
pub struct Annotation {
    pub ann_id: String,
    pub group: hdf5::Group,
    pub header: AnnotationHeader,
    grids: Arc<Grids>,
    label_set: Option<Arc<LabelSet>>,
    layers: OnceLock<Result<LayerTable>>,
    bits: OnceLock<Result<BitTable>>,
    columns: OnceLock<Result<InstanceColumns>>,
    /// Whether any voxel is ignored, in band or in the sibling mask.
    ignored: OnceLock<Result<bool>>,
    /// Whether the data holds the ignore id in band (`labelmap`, `layers`).
    in_band: OnceLock<Result<bool>>,
    /// The sibling `mask` the header's `ignore_mask` names, opened once.
    sibling: OnceLock<Result<Option<Box<Annotation>>>>,
    /// Class id -> the classes in `class_ids` that imply it (§5.4).
    implied: OnceLock<Result<HashMap<i64, Vec<i64>>>>,
}

/// What [`Annotation::scan_classes`] calls for each stored voxel of a class:
/// `(entry, row, outer, k)`.
pub(crate) type ClassVisitor<'a> = dyn FnMut(usize, usize, &[usize], usize) + 'a;

fn cached<T>(cell: &OnceLock<Result<T>>, init: impl FnOnce() -> Result<T>) -> Result<&T> {
    match crate::h5::once(cell, init) {
        Ok(v) => Ok(v),
        Err(e) => Err(e.clone()),
    }
}

/// A region of interest resolved against a grid: `(start, stop)` per axis.
pub type Window = Vec<(usize, usize)>;

fn window_shape(window: &Window) -> Vec<usize> {
    window.iter().map(|(a, b)| b.saturating_sub(*a)).collect()
}

fn window_index(prefix: Option<usize>, window: &Window) -> Vec<Index> {
    let mut out: Vec<Index> = prefix.map(|p| Index::At(p as i64)).into_iter().collect();
    out.extend(window.iter().map(|(a, b)| Index::Slice(Slice::new(*a as i64, *b as i64))));
    out
}

/// Elementwise `block == value`, compared in the block's own type.
pub fn equals(block: &NdArray, value: i64) -> ArrayD<bool> {
    match block {
        NdArray::F16(_) | NdArray::F32(_) | NdArray::F64(_) => block.to_f64().mapv(|v| v == value as f64),
        _ => with_array!(block, a => a.mapv(|v| v.to_i128() == i128::from(value))),
    }
}

/// Call `f(owner, k)` for every voxel `k` (C order) of `block` equal to the
/// class id of an `(owner, class id)` in `wanted`, compared as [`equals`]
/// compares.  A `uint8` or `uint16` block --- a label volume as §7.1 stores
/// it --- is split by value in one pass over it.
fn each_match(block: &NdArray, wanted: &[(usize, i64)], mut f: impl FnMut(usize, usize)) {
    let mut by_value = |values: &mut dyn Iterator<Item = usize>| {
        let (slot, owners) = value_table(wanted);
        if owners.is_empty() {
            return;
        }
        for (k, value) in values.enumerate() {
            if let Some(entry) = slot.get(value).filter(|e| **e != u32::MAX) {
                for owner in &owners[*entry as usize] {
                    f(*owner, k);
                }
            }
        }
    };
    match block {
        NdArray::U8(a) => by_value(&mut a.iter().map(|v| usize::from(*v))),
        NdArray::U16(a) => by_value(&mut a.iter().map(|v| usize::from(*v))),
        _ => {
            for (owner, class_id) in wanted {
                for (k, hit) in equals(block, *class_id).iter().enumerate() {
                    if *hit {
                        f(*owner, k);
                    }
                }
            }
        }
    }
}

/// OR into plane `p` of `flat` --- `(planes, voxels)` values in C order ---
/// every voxel of `block` equal to `c`, for each `(p, c)` in `wanted`.
fn paint_values(block: &NdArray, wanted: &[(usize, i64)], flat: &mut [bool], voxels: usize) {
    each_match(block, wanted, |position, k| flat[position * voxels + k] = true);
}

/// `value -> owners` for the `(owner, class id)` pairs a `uint16` can hold:
/// `slot[value]` indexes the owners of `value`, `u32::MAX` where it has none.
fn value_table(wanted: &[(usize, i64)]) -> (Vec<u32>, Vec<Vec<usize>>) {
    let top = wanted.iter().filter_map(|(_, c)| u16::try_from(*c).ok()).max();
    let mut slot = vec![u32::MAX; top.map_or(0, |t| usize::from(t) + 1)];
    let mut owners: Vec<Vec<usize>> = Vec::new();
    for (owner, class_id) in wanted {
        let Ok(value) = u16::try_from(*class_id) else { continue };
        let entry = &mut slot[usize::from(value)];
        if *entry == u32::MAX {
            *entry = owners.len() as u32;
            owners.push(Vec::new());
        }
        owners[*entry as usize].push(*owner);
    }
    (slot, owners)
}

/// Call `f(row, outer, k, value)` for every value of a C-order volume of
/// `shape` (at least one axis), a row of its last axis at a time: `row`
/// counts the rows, `outer` is the row's index along the other axes and `k`
/// the value's place in the row.
pub(crate) fn walk_rows<T>(values: &[T], shape: &[usize], mut f: impl FnMut(usize, &[usize], usize, &T)) {
    let Some((&width, outer)) = shape.split_last() else { return };
    if width == 0 {
        return;
    }
    let mut at = vec![0usize; outer.len()];
    for (row, chunk) in values.chunks_exact(width).enumerate() {
        for (k, value) in chunk.iter().enumerate() {
            f(row, &at, k, value);
        }
        for (index, extent) in at.iter_mut().zip(outer).rev() {
            *index += 1;
            if *index < *extent {
                break;
            }
            *index = 0;
        }
    }
}

/// OR `mask`, in C order, into each plane of `flat` in `positions`.
fn paint_mask(mask: &ArrayD<bool>, positions: &[usize], flat: &mut [bool], voxels: usize) {
    for position in positions {
        let plane = &mut flat[position * voxels..(position + 1) * voxels];
        for (o, m) in plane.iter_mut().zip(mask.iter()) {
            *o |= *m;
        }
    }
}

impl Annotation {
    /// Open an annotation group as the reader matching its `kind`.
    pub fn open(
        ann_id: &str,
        group: hdf5::Group,
        grids: Arc<Grids>,
        label_set: Option<Arc<LabelSet>>,
    ) -> Result<Annotation> {
        let header = AnnotationHeader::read(&group)?;
        let annotation = Annotation {
            ann_id: ann_id.to_string(),
            group,
            header,
            grids,
            label_set,
            layers: OnceLock::new(),
            bits: OnceLock::new(),
            columns: OnceLock::new(),
            ignored: OnceLock::new(),
            in_band: OnceLock::new(),
            sibling: OnceLock::new(),
            implied: OnceLock::new(),
        };
        annotation.check_columns()?;
        annotation.check_tables()?;
        Ok(annotation)
    }

    /// Refuse a `layers` or `bitmask` table that does not map the stored
    /// planes to the declared classes one to one (§7.2, §7.3): a class in no
    /// layer or bit read as examined and absent, a class in two as whichever
    /// came last, and a table class outside `class_ids` was answered for by
    /// `contains()` and painted by no `labelmap()`.  The tables are small,
    /// and read here once for every later read.
    fn check_tables(&self) -> Result<()> {
        match self.kind() {
            "layers" if self.optional_dataset("layer_class_ids")?.is_some() => self.layer_table().map(|_| ()),
            "bitmask" if self.optional_dataset("bit_class_ids")?.is_some() => self.bit_table().map(|_| ()),
            _ => Ok(()),
        }
    }

    /// Refuse an annotation whose datasets disagree in shape (E405): every
    /// reader of one indexes its columns by row, and a column one row short
    /// was read past its end; every voxel read is a window of the grid, and a
    /// volume of another shape cannot answer one.  Shapes only --- nothing is
    /// read.
    fn check_columns(&self) -> Result<()> {
        let grid = self.header.grid.as_deref().and_then(|g| self.grids.get(g).map(|grid| (g, grid)));
        let rank =
            grid.filter(|_| layout::ranked(self.kind(), self.header.space.as_deref())).map(|(_, g)| g.n_spatial());
        let shape_of = |name: &str| ops::child_dataset(&self.group, name).map(|d| d.shape());
        if let Some(problem) = layout::column_problems(self.kind(), &shape_of, rank).first() {
            return Err(problem.error(&self.ann_id));
        }
        let data = if is_voxel_kind(self.kind()) { self.optional_dataset("data")? } else { None };
        let planes = data.as_ref().map(|d| d.shape().first().copied().unwrap_or(0));
        if let Some(problem) = layout::table_problems(self.kind(), &shape_of, planes, None, None).first() {
            return Err(problem.error(&self.ann_id));
        }
        if let (Some((gid, grid)), Some(data)) = (grid, data) {
            let shape = data.shape();
            let stacked = matches!(self.kind(), "layers" | "bitmask" | "probmap");
            let tail = if stacked { shape.get(1..).unwrap_or(&[]) } else { &shape[..] };
            let spatial = grid.spatial_shape();
            if tail != spatial.as_slice() {
                return Err(Error::coded(
                    "E405",
                    format!(
                        "annotation {}: `data` has spatial shape {}, but grid {} has spatial shape {}; a voxel \
                         annotation covers its grid exactly",
                        repr_str(&self.ann_id),
                        crate::json::repr_int_tuple(tail),
                        repr_str(gid),
                        crate::json::repr_int_tuple(&spatial)
                    ),
                ));
            }
            if self.kind() == "probmap" && shape.first() != Some(&self.class_ids().len()) {
                return Err(Error::coded(
                    "E405",
                    format!(
                        "annotation {}: `data` has {} plane(s) for {} class id(s); a probmap stores one plane per \
                         class, in `class_ids` order",
                        repr_str(&self.ann_id),
                        shape.first().copied().unwrap_or(0),
                        self.class_ids().len()
                    ),
                ));
            }
        }
        Ok(())
    }

    // -- header passthrough -------------------------------------------------

    pub fn kind(&self) -> &str {
        &self.header.kind
    }

    pub fn task(&self) -> &str {
        &self.header.task
    }

    pub fn class_ids(&self) -> &[i64] {
        &self.header.class_ids
    }

    pub fn annotated_class_ids(&self) -> &[i64] {
        &self.header.annotated_class_ids
    }

    pub fn closure(&self) -> &str {
        &self.header.closure
    }

    pub fn ignore_id(&self) -> i64 {
        self.header.ignore_id
    }

    pub fn prov(&self) -> Option<&str> {
        self.header.prov.as_deref()
    }

    pub fn quality_key(&self) -> Option<&str> {
        self.header.quality.as_deref()
    }

    pub fn label_set(&self) -> Option<&LabelSet> {
        self.label_set.as_deref()
    }

    pub fn grid_id(&self) -> Option<&str> {
        self.header.grid.as_deref()
    }

    /// The grids this annotation can be read on.
    pub fn grids(&self) -> &Grids {
        &self.grids
    }

    /// Whether this is a voxel kind (§7).
    pub fn is_voxel(&self) -> bool {
        is_voxel_kind(self.kind())
    }

    /// Whether this is a geometric kind (§8).
    pub fn is_geometric(&self) -> bool {
        is_geometric_kind(self.kind())
    }

    /// The reader class name the Python bindings expose for this kind.
    pub fn class_name(&self) -> &'static str {
        match self.kind() {
            "labelmap" => "LabelmapAnnotation",
            "layers" => "LayersAnnotation",
            "bitmask" => "BitmaskAnnotation",
            "instances" => "InstancesAnnotation",
            "probmap" => "ProbmapAnnotation",
            "mask" => "MaskAnnotation",
            "boxes" => "BoxesAnnotation",
            "obb" => "ObbAnnotation",
            "keypoints" => "KeypointsAnnotation",
            "points" => "PointsAnnotation",
            "contours" => "ContoursAnnotation",
            "polylines" => "PolylinesAnnotation",
            "mesh" => "MeshAnnotation",
            _ => "ClassificationAnnotation",
        }
    }

    /// The annotation's grid, or an error naming why there is none.
    pub fn grid(&self) -> Result<&Grid> {
        let gid = self.header.grid.as_deref().ok_or_else(|| {
            Error::invalid(format!(
                "annotation {} of kind {} has no grid",
                repr_str(&self.ann_id),
                repr_str(self.kind())
            ))
        })?;
        self.grids.get(gid).ok_or_else(|| {
            Error::coded(
                "E101",
                format!("annotation {} names grid {}, which does not exist", repr_str(&self.ann_id), repr_str(gid)),
            )
        })
    }

    /// Timepoints this annotation pertains to, inherited from the grid (§3.7).
    pub fn timepoints(&self) -> Vec<String> {
        if let Some(tps) = &self.header.timepoints {
            return tps.clone();
        }
        match self.header.grid.as_deref().and_then(|g| self.grids.get(g)) {
            Some(grid) => grid.timepoint.iter().filter(|t| !t.is_empty()).cloned().collect(),
            None => Vec::new(),
        }
    }

    // -- coverage -----------------------------------------------------------

    /// Whether the annotator committed to finding this class (§11.3).
    pub fn is_annotated(&self, key: &ClassKey) -> Result<bool> {
        let id = self.resolve_class(key)?;
        Ok(self.header.annotated_class_ids.contains(&id))
    }

    /// Whether every class this annotation can express was looked for.
    pub fn is_fully_covered(&self) -> bool {
        let mut a = self.header.annotated_class_ids.clone();
        let mut b = self.header.class_ids.clone();
        a.sort_unstable();
        a.dedup();
        b.sort_unstable();
        b.dedup();
        a == b
    }

    /// Whether any voxel is ignored (§7.7), in band or in the sibling `mask`
    /// the header's `ignore_mask` names: exactly when [`Self::ignore_region`]
    /// has one.  A named mask with no voxel set ignores nothing.
    pub fn has_ignore_region(&self) -> Result<bool> {
        Ok(*cached(&self.ignored, || self.ignore_present())?)
    }

    /// Whether the data itself carries the ignore id (`labelmap`, `layers`):
    /// a scan, in slabs, made once per open annotation.  Data whose integer
    /// type cannot hold the ignore id carries none, and is not read.
    pub fn encodes_ignore(&self) -> Result<bool> {
        Ok(*cached(&self.in_band, || match self.kind() {
            "labelmap" | "layers" => {
                let data = self.data()?;
                if !self.may_hold_ignore(&data)? {
                    return Ok(false);
                }
                contains_value(&data, self.ignore_id())
            }
            _ => Ok(false),
        })?)
    }

    /// Whether `data` has a type that can hold the ignore id: every float
    /// type is taken to, an integer type only when the id is in its range.
    fn may_hold_ignore(&self, data: &hdf5::Dataset) -> Result<bool> {
        let dtype = data::dtype(data)?;
        Ok(dtype.int_range().is_none() || dtype.holds(i128::from(self.ignore_id())))
    }

    /// Whether a window read of the data has to look for the ignore id in
    /// band: `false` when it is known to hold none --- from the scan
    /// [`Self::encodes_ignore`] made, or because its type cannot hold the id
    /// --- so a window read skips the part of the read that finds the ignore
    /// region.  Never scans the data itself.
    pub(crate) fn may_encode_ignore(&self) -> Result<bool> {
        if !matches!(self.kind(), "labelmap" | "layers") {
            return Ok(false);
        }
        if let Some(known) = self.in_band.get() {
            return known.clone();
        }
        let possible = self.may_hold_ignore(&self.data()?)?;
        if !possible {
            let _ = self.in_band.set(Ok(false));
        }
        Ok(possible)
    }

    /// Resolve a class id or key to an id, using the label set for keys.
    pub fn resolve_class(&self, key: &ClassKey) -> Result<i64> {
        match key {
            ClassKey::Id(id) => Ok(*id),
            ClassKey::Key(name) => match &self.label_set {
                None => Err(Error::invalid(format!(
                    "annotation {}: cannot resolve class name {} without a label set",
                    repr_str(&self.ann_id),
                    repr_str(name)
                ))),
                Some(ls) => Ok(ls.lookup(key)?.id),
            },
        }
    }

    /// Resolve requested class keys to ids, refusing the reserved ids.
    ///
    /// `65535` --- or the `ignore_id` the header declares --- is not a class:
    /// §5.2 says it MUST NOT appear in `classes`, so nothing that returns
    /// per-class planes can answer for it, and an all-zero plane would be
    /// indistinguishable from a class examined and found absent.  Nor is `0`,
    /// background: a `labelmap` would answer with every voxel no class holds,
    /// and every other encoding with none (§6.2).
    pub fn resolve_classes(&self, keys: Option<&[ClassKey]>) -> Result<Vec<i64>> {
        let Some(keys) = keys else {
            return Ok(self.header.class_ids.clone());
        };
        let ids = keys.iter().map(|k| self.resolve_class(k)).collect::<Result<Vec<_>>>()?;
        let id = repr_str(&self.ann_id);
        if let Some(ignore) = ids.iter().find(|c| **c == IGNORE_ID || **c == self.ignore_id()) {
            let what = if *ignore == IGNORE_ID { "the reserved ignore id" } else { "this annotation's ignore id" };
            return Err(Error::coded(
                "E404",
                format!(
                    "annotation {id}: {ignore} is {what}, not a class, so no plane can be returned for it; read the \
                     ignore region with `ignore_mask()` where the encoding carries it in band, or through the \
                     `mask` annotation named by `header.ignore_mask`"
                ),
            ));
        }
        if ids.contains(&BACKGROUND_ID) {
            return Err(Error::coded(
                "E404",
                format!(
                    "annotation {id}: 0 is background, not a class, so no plane can be returned for it --- a \
                     `labelmap` would answer with every voxel no class holds, and every other encoding with none \
                     (§6.2); the voxels no class holds are those `labelmap()` returns as 0"
                ),
            ));
        }
        Ok(ids)
    }

    /// The classes in `class_ids` whose voxels also hold `class_id` (§5.4).
    ///
    /// Under `closure = "implicit"` every `is_a` ancestor of a listed class
    /// holds where the class does, so a class is present wherever one of its
    /// descendants in the label set's DAG is --- the classes here, in
    /// `class_ids` order.  Under `explicit` none is: readers MUST NOT infer
    /// ancestors there.  Without a label set there is no DAG to read, and a
    /// class unknown to it has no descendants.
    pub(crate) fn implying(&self, class_id: i64) -> Result<&[i64]> {
        let table = cached(&self.implied, || {
            let mut table: HashMap<i64, Vec<i64>> = HashMap::new();
            let Some(labels) = self.label_set.as_deref().filter(|_| self.closure() == "implicit") else {
                return Ok(table);
            };
            let mut seen = std::collections::HashSet::new();
            for stored in self.class_ids() {
                if !seen.insert(*stored) || !labels.contains_id(*stored) {
                    continue;
                }
                // `c` is an ancestor of `stored` exactly when `stored` is
                // among `c`'s descendants.
                for ancestor in labels.ancestors(&ClassKey::Id(*stored))? {
                    table.entry(ancestor).or_default().push(*stored);
                }
            }
            Ok(table)
        })?;
        Ok(table.get(&class_id).map(Vec::as_slice).unwrap_or(&[]))
    }

    /// Each requested class's plane position with the stored classes that
    /// paint it: the class itself and, under `implicit`, its descendants.
    fn sources(&self, ids: &[i64]) -> Result<Vec<(usize, i64)>> {
        let mut out = Vec::with_capacity(ids.len());
        for (position, class_id) in ids.iter().enumerate() {
            out.push((position, *class_id));
            out.extend(self.implying(*class_id)?.iter().map(|s| (position, *s)));
        }
        Ok(out)
    }

    /// Resolved class entries; empty when the label set is unavailable.
    pub fn classes(&self) -> Vec<LabelClass> {
        self.resolved_entries(&self.header.class_ids)
    }

    /// Resolved entries of the classes that were looked for.
    pub fn annotated_classes(&self) -> Vec<LabelClass> {
        self.resolved_entries(&self.header.annotated_class_ids)
    }

    fn resolved_entries(&self, ids: &[i64]) -> Vec<LabelClass> {
        match &self.label_set {
            None => Vec::new(),
            Some(ls) => ids.iter().filter_map(|i| ls.by_id(*i).cloned()).collect(),
        }
    }

    /// The label-set key of a class id, or the id as text.
    pub fn class_key(&self, class_id: i64) -> String {
        self.label_set
            .as_ref()
            .and_then(|ls| ls.by_id(class_id))
            .map(|c| c.key.clone())
            .unwrap_or_else(|| class_id.to_string())
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        format!(
            "{}({}, kind={}, {} classes)",
            self.class_name(),
            repr_str(&self.ann_id),
            repr_str(self.kind()),
            self.class_ids().len()
        )
    }

    // -- datasets -------------------------------------------------------------

    /// A member dataset, or `None` when absent.  A closed file is refused:
    /// HDF5 finds no member in one, which read as an absent dataset (E410,
    /// or no masks, no scores).
    pub fn optional_dataset(&self, name: &str) -> Result<Option<hdf5::Dataset>> {
        crate::h5::alive(&self.group)?;
        Ok(ops::child_dataset(&self.group, name))
    }

    /// A member dataset the kind requires (E410 when absent).
    pub fn dataset(&self, name: &str) -> Result<hdf5::Dataset> {
        self.optional_dataset(name)?.ok_or_else(|| {
            let id = repr_str(&self.ann_id);
            let message = match self.kind() {
                k if is_geometric_kind(k) => {
                    format!("annotation {id}: kind {} requires a {} dataset", repr_str(k), repr_str(name))
                }
                "labelmap" | "layers" | "bitmask" | "probmap" | "mask" if name == "data" => {
                    format!("annotation {id}: `{}` requires a `data` dataset", self.kind())
                }
                k => format!("annotation {id}: `{k}` requires a {} dataset", repr_str(name)),
            };
            Error::coded("E410", message)
        })
    }

    /// The `data` dataset of a voxel annotation.
    pub fn data(&self) -> Result<hdf5::Dataset> {
        self.dataset("data")
    }

    /// A whole numeric dataset cast to `T`.
    pub fn read_as<T: Element>(&self, name: &str) -> Result<ArrayD<T>> {
        Ok(data::read(&self.dataset(name)?)?.cast::<T>())
    }

    /// A whole numeric dataset cast to `T`, or `None` when absent.
    pub fn read_optional<T: Element>(&self, name: &str) -> Result<Option<ArrayD<T>>> {
        match self.optional_dataset(name)? {
            None => Ok(None),
            Some(ds) => Ok(Some(data::read(&ds)?.cast::<T>())),
        }
    }

    /// A string dataset, or `None` when absent.
    pub fn read_strings_optional(&self, name: &str) -> Result<Option<Vec<String>>> {
        match self.optional_dataset(name)? {
            None => Ok(None),
            Some(ds) => Ok(Some(data::read_strings(&ds)?)),
        }
    }

    /// The shape of a required dataset.
    pub fn dataset_shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.dataset(name)?.shape())
    }

    // -- voxel kinds: the region of interest --------------------------------

    /// The spatial shape of the annotation's grid.
    pub fn spatial_shape(&self) -> Result<Vec<usize>> {
        Ok(self.grid()?.spatial_shape())
    }

    /// Resolve a region of interest against the grid; `None` is the whole grid.
    ///
    /// A window is contiguous: a slice with a step other than 1 is refused
    /// (`ValueError`) rather than read as if it had none --- `slice(0, 4, 2)`
    /// would hand back four planes where `Image.read` returns two, and a
    /// label patch would no longer line up with its image patch.
    pub fn window(&self, roi: Option<&[Slice]>) -> Result<Window> {
        let shape = self.spatial_shape()?;
        let Some(roi) = roi else {
            return Ok(shape.iter().map(|n| (0, *n)).collect());
        };
        if roi.len() != shape.len() {
            return Err(Error::invalid(format!("roi has {} axes; grid has {} spatial axes", roi.len(), shape.len())));
        }
        roi.iter()
            .zip(&shape)
            .map(|(s, n)| {
                let (a, b, step) = s.resolve(*n)?;
                if step != 1 {
                    return Err(Error::Value(format!(
                        "annotation {}: a region of interest is read whole, so a slice with step {step} is refused; \
                         read the region, then slice the result with the step",
                        repr_str(&self.ann_id)
                    )));
                }
                Ok((a, b))
            })
            .collect()
    }

    fn voxel_window(&self, voxel: &[i64]) -> Result<Window> {
        let shape = self.spatial_shape()?;
        if voxel.len() != shape.len() {
            return Err(Error::invalid(format!(
                "voxel has {} axes; grid has {} spatial axes",
                voxel.len(),
                shape.len()
            )));
        }
        voxel
            .iter()
            .zip(&shape)
            .map(|(v, n)| {
                if *v < 0 || *v >= *n as i64 {
                    Err(Error::Index(format!("voxel index {v} is out of bounds for an axis of {n} voxels")))
                } else {
                    Ok((*v as usize, *v as usize + 1))
                }
            })
            .collect()
    }

    fn require_voxel(&self, what: &str) -> Result<()> {
        if self.is_voxel() {
            Ok(())
        } else {
            Err(Error::Type(format!(
                "{what} is defined on voxel annotations; {} is a {} annotation",
                repr_str(&self.ann_id),
                repr_str(self.kind())
            )))
        }
    }

    fn require_kind(&self, kinds: &[&str], what: &str) -> Result<()> {
        if kinds.contains(&self.kind()) {
            Ok(())
        } else {
            Err(Error::Type(format!(
                "{what} is defined on {} annotations; {} is a {} annotation",
                kinds.iter().map(|k| format!("`{k}`")).collect::<Vec<_>>().join("/"),
                repr_str(&self.ann_id),
                repr_str(self.kind())
            )))
        }
    }

    // -- voxel kinds: the contract -------------------------------------------

    /// Boolean occupancy of one class over a window, as §7.6's `contains`
    /// answers it: the class's own voxels and, under `closure = "implicit"`,
    /// those of every class in `class_ids` it is an ancestor of (§5.4,
    /// [`Self::implying`]) --- in every encoding alike.
    pub fn dense_class(&self, class_id: i64, window: &Window) -> Result<ArrayD<bool>> {
        self.read_planes(&self.sources(&[class_id])?, window, 1).map(|p| p.index_axis_move(Axis(0), 0))
    }

    /// One class's voxels as stored, whatever the `closure`: what an encoder
    /// was given for it, and what a transcode or a flatten re-encodes.
    pub(crate) fn stored_class(&self, class_id: i64, window: &Window) -> Result<ArrayD<bool>> {
        self.read_planes(&[(0, class_id)], window, 1).map(|p| p.index_axis_move(Axis(0), 0))
    }

    /// `(planes, *window)` occupancy: plane `p` holds the voxels of every
    /// stored class `c` of a `(p, c)` in `sources`.  Each stored volume,
    /// layer or bitplane is read once, however many classes it holds --- a
    /// `labelmap` window is read once, not once per class.
    fn read_planes(&self, sources: &[(usize, i64)], window: &Window, planes: usize) -> Result<ArrayD<bool>> {
        let mut shape = vec![planes];
        shape.extend(window_shape(window));
        let mut out = ArrayD::from_elem(IxDyn(&shape), false);
        let voxels: usize = shape[1..].iter().product();
        if sources.is_empty() {
            self.require_voxel("dense()")?;
            return Ok(out);
        }
        let flat = out.as_slice_mut().expect("standard layout");
        match self.kind() {
            "labelmap" => {
                let block = data::read_region(&self.data()?, &window_index(None, window))?;
                paint_values(&block, sources, flat, voxels);
            }
            "layers" => {
                let table = self.layer_table()?;
                let mut by_layer: std::collections::BTreeMap<usize, Vec<(usize, i64)>> = Default::default();
                for (position, class_id) in sources {
                    if let Some(layer) = table.layer_of.get(class_id) {
                        by_layer.entry(*layer).or_default().push((*position, *class_id));
                    }
                }
                let ds = self.data()?;
                for (layer, wanted) in by_layer {
                    let block = data::read_region(&ds, &window_index(Some(layer), window))?;
                    paint_values(&block, &wanted, flat, voxels);
                }
            }
            "bitmask" => {
                let table = self.bit_table()?;
                let mut by_plane: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
                for (position, class_id) in sources {
                    if let Some(slot) = table.position_of.get(class_id) {
                        by_plane.entry(slot / BITS_PER_PLANE).or_default().push((*position, slot % BITS_PER_PLANE));
                    }
                }
                let ds = self.data()?;
                for (plane, bits) in by_plane {
                    let block = data::read_region(&ds, &window_index(Some(plane), window))?.cast::<u64>();
                    for (k, word) in block.iter().enumerate() {
                        if *word == 0 {
                            continue;
                        }
                        for (position, bit) in &bits {
                            if (word >> bit) & 1 == 1 {
                                flat[position * voxels + k] = true;
                            }
                        }
                    }
                }
            }
            "probmap" => {
                let mut by_plane: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
                for (position, class_id) in sources {
                    if let Some(stored) = self.class_ids().iter().position(|c| c == class_id) {
                        by_plane.entry(stored).or_default().push(*position);
                    }
                }
                let ds = self.data()?;
                let threshold = if by_plane.is_empty() { DEFAULT_THRESHOLD } else { self.threshold()? };
                for (stored, positions) in by_plane {
                    let block = data::read_region(&ds, &window_index(Some(stored), window))?;
                    paint_mask(&contains_at(&block, threshold), &positions, flat, voxels);
                }
            }
            "instances" => self.paint_instances(sources, window, flat, voxels)?,
            "mask" => {
                let mask = data::read_region(&self.data()?, &window_index(None, window))?.nonzero_mask();
                let positions: Vec<usize> = sources.iter().map(|(p, _)| *p).collect();
                paint_mask(&mask, &positions, flat, voxels);
            }
            _ => self.require_voxel("dense()")?,
        }
        Ok(out)
    }

    /// `(C, *roi_shape)` boolean occupancy, one plane per requested class,
    /// each as [`Self::dense_class`] answers it.
    ///
    /// Each stored volume, layer or bitplane is read once rather than once
    /// per class; `mask` has no classes and answers with itself as one plane.
    pub fn dense(&self, classes: Option<&[ClassKey]>, roi: Option<&[Slice]>) -> Result<ArrayD<bool>> {
        self.require_voxel("dense()")?;
        if self.kind() == "mask" {
            if classes.is_some() {
                self.resolve_classes(classes)?;
            }
            return Ok(self.read_mask(roi)?.insert_axis(Axis(0)));
        }
        let ids = self.resolve_classes(classes)?;
        let window = self.window(roi)?;
        self.read_planes(&self.sources(&ids)?, &window, ids.len())
    }

    /// The uniform predicate of §7.6: is `class` present at `voxel`?
    ///
    /// Background (`0`) and the ignore id are no class, and are refused
    /// (E404) as [`Self::resolve_classes`] refuses them.
    pub fn contains(&self, class: &ClassKey, voxel: &[i64]) -> Result<bool> {
        self.require_voxel("contains()")?;
        let class_id = self.resolve_classes(Some(std::slice::from_ref(class)))?[0];
        let window = self.voxel_window(voxel)?;
        Ok(self.dense_class(class_id, &window)?.iter().next().copied().unwrap_or(false))
    }

    /// Flatten to one integer volume, breaking overlap ties explicitly.
    ///
    /// `priority` is ordered highest-precedence first; classes it omits are
    /// painted first, in `class_ids` order.  The §7.7 ignore region is
    /// painted over them as `ignore_id`, whichever encoding holds it --- in
    /// band, or in the sibling `mask` the header names --- so an ignored
    /// voxel never reads as `0`, which means verified absent.
    ///
    /// `dtype` must hold every class id, and `ignore_id` where the
    /// annotation has an ignore region; a narrower one is refused
    /// (`ValueError`) rather than saturating two values into one --- class 300
    /// and the ignore id both read as 255 in `uint8` --- and `bool` holds no
    /// class id at all.
    ///
    /// Returns the volume and, when no priority was given, how many voxels a
    /// later class overwrote --- the frontends warn on a nonzero count with
    /// [`Annotation::flatten_warning`].
    pub fn labelmap(
        &self,
        roi: Option<&[Slice]>,
        priority: Option<&[ClassKey]>,
        dtype: DType,
    ) -> Result<(NdArray, u64, Vec<i64>)> {
        self.require_voxel("labelmap()")?;
        self.check_label_dtype(dtype)?;
        let window = self.window(roi)?;
        let (mut out, overwritten, ordered, region) = if self.kind() == "labelmap" {
            // Cast once, from the stored width: its in-band region already
            // reads as `ignore_id`.
            let block = data::read_region(&self.data()?, &window_index(None, &window))?;
            (block.astype(dtype), 0, self.class_ids().to_vec(), self.with_sibling(&window, None)?)
        } else {
            let mut ordered = self.class_ids().to_vec();
            if let Some(priority) = priority {
                let ranked = self.resolve_classes(Some(priority))?;
                let mut rest: Vec<i64> = ordered.iter().copied().filter(|c| !ranked.contains(c)).collect();
                rest.extend(ranked.iter().rev());
                ordered = rest;
            }
            // The stored classes, painted in `ordered` order: the last one
            // present at a voxel holds it, and every one before it there was
            // overwritten.  Literal under every `closure`: one integer per
            // voxel holds the most specific class, as stored.
            let (last, present, in_band) = self.rank_classes(&ordered, &window)?;
            let mut out = ArrayD::<i64>::zeros(IxDyn(&window_shape(&window)));
            let mut overwritten = 0u64;
            for ((o, l), p) in out.iter_mut().zip(&last).zip(&present) {
                if *l > 0 {
                    *o = ordered[*l as usize - 1];
                    overwritten += u64::from(*p - 1);
                }
            }
            let overwritten = if priority.is_none() { overwritten } else { 0 };
            (NdArray::from(out).astype(dtype), overwritten, ordered, self.with_sibling(&window, in_band)?)
        };
        // `check_label_dtype` has made sure `dtype` holds the ignore id.
        if let Some(region) = region {
            let ignore = i128::from(self.ignore_id());
            with_array!(&mut out, a => a.zip_mut_with(&region, |o, r| {
                if *r {
                    *o = Element::from_i128(ignore);
                }
            }));
        }
        Ok((out, overwritten, ordered))
    }

    /// For each voxel of a window, in C order: one past the position in
    /// `ordered` of the last class stored there (0 for none), and how many
    /// entries of `ordered` are stored there --- with the in-band ignore
    /// region (§7.7) where it was read along.
    ///
    /// `layers` and `bitmask` read each stored layer or bitplane once, and
    /// a `layers` window's ignore region comes from the layers just read;
    /// the other encodings decode class by class.
    #[allow(clippy::type_complexity)]
    fn rank_classes(&self, ordered: &[i64], window: &Window) -> Result<(Vec<u32>, Vec<u32>, Option<ArrayD<bool>>)> {
        let shape = window_shape(window);
        let voxels: usize = shape.iter().product();
        let mut last = vec![0u32; voxels];
        let mut present = vec![0u32; voxels];
        let mut mark = |rank: usize, k: usize| {
            last[k] = last[k].max(rank as u32 + 1);
            present[k] += 1;
        };
        let mut in_band = None;
        match self.kind() {
            "layers" => {
                let table = self.layer_table()?;
                let mut by_layer: std::collections::BTreeMap<usize, Vec<(usize, i64)>> = Default::default();
                for (rank, class_id) in ordered.iter().enumerate() {
                    if let Some(layer) = table.layer_of.get(class_id) {
                        by_layer.entry(*layer).or_default().push((rank, *class_id));
                    }
                }
                let ds = self.data()?;
                // Every layer may hold the ignore id, so all are read when
                // one might; otherwise only those holding a wanted class.
                let layers: Vec<usize> = if self.may_encode_ignore()? {
                    in_band = Some(ArrayD::from_elem(IxDyn(&shape), false));
                    (0..ds.shape().first().copied().unwrap_or(0)).collect()
                } else {
                    by_layer.keys().copied().collect()
                };
                for layer in layers {
                    let block = data::read_region(&ds, &window_index(Some(layer), window))?;
                    if let Some(wanted) = by_layer.get(&layer) {
                        each_match(&block, wanted, &mut mark);
                    }
                    if let Some(region) = in_band.as_mut() {
                        region.zip_mut_with(&equals(&block, self.ignore_id()), |r, i| *r |= *i);
                    }
                }
            }
            "bitmask" => {
                let table = self.bit_table()?;
                let mut by_plane: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
                for (rank, class_id) in ordered.iter().enumerate() {
                    if let Some(slot) = table.position_of.get(class_id) {
                        by_plane.entry(slot / BITS_PER_PLANE).or_default().push((rank, slot % BITS_PER_PLANE));
                    }
                }
                let ds = self.data()?;
                for (plane, bits) in by_plane {
                    let block = data::read_region(&ds, &window_index(Some(plane), window))?.cast::<u64>();
                    for (k, word) in block.iter().enumerate() {
                        if *word == 0 {
                            continue;
                        }
                        for (rank, bit) in &bits {
                            if (word >> bit) & 1 == 1 {
                                mark(*rank, k);
                            }
                        }
                    }
                }
            }
            _ => {
                for (rank, class_id) in ordered.iter().enumerate() {
                    for (k, hit) in self.stored_class(*class_id, window)?.iter().enumerate() {
                        if *hit {
                            mark(rank, k);
                        }
                    }
                }
            }
        }
        Ok((last, present, in_band))
    }

    /// Refuse a `labelmap()` dtype that cannot hold the values the volume
    /// holds: every class id, and the ignore id where a voxel is ignored.
    fn check_label_dtype(&self, dtype: DType) -> Result<()> {
        let id = repr_str(&self.ann_id);
        if dtype == DType::Bool {
            return Err(Error::Value(format!(
                "annotation {id}: labelmap() returns class ids, which a bool volume cannot hold; dense() returns one \
                 boolean plane per class"
            )));
        }
        if let Some(class_id) = self.class_ids().iter().find(|c| !dtype.holds(i128::from(**c))) {
            return Err(Error::Value(format!(
                "annotation {id}: {} cannot hold class id {class_id}, and labelmap() returns class ids as they are; \
                 ask for a dtype that holds every one (uint16 does)",
                dtype.name()
            )));
        }
        let ignore = self.ignore_id();
        if !dtype.holds(i128::from(ignore)) && *cached(&self.ignored, || self.ignore_present())? {
            return Err(Error::Value(format!(
                "annotation {id}: {} cannot hold the ignore id {ignore}, which labelmap() writes at every ignored \
                 voxel; ask for a dtype that holds it (uint16 does)",
                dtype.name()
            )));
        }
        Ok(())
    }

    /// The warning a lossy, unprioritised flatten raises.
    pub fn flatten_warning(&self, overwritten: u64, ordered: &[i64]) -> String {
        format!(
            "labelmap() flattened {overwritten} overlapping voxel(s) in {}: classes {} overlap and one integer \
             volume cannot hold both, so later classes overwrote earlier ones. Pass priority=[...] to choose \
             which class survives, or use dense()/contains() to keep the overlap.",
            repr_str(&self.ann_id),
            crate::json::repr_int_list(ordered)
        )
    }

    /// Foreground voxel count per class, over the whole grid, each as
    /// [`Self::dense_class`] counts it.
    ///
    /// `labelmap` and `layers` answer every class from one pass of value
    /// counts, `bitmask` from one population count per plane; the other
    /// encodings, and a class other classes imply (§5.4), decode per class.
    pub fn voxel_counts(&self, classes: Option<&[ClassKey]>) -> Result<IndexMap<i64, u64>> {
        self.require_voxel("voxel_counts()")?;
        let ids = self.resolve_classes(classes)?;
        let counted = self.counts_from_planes()?;
        let window = self.window(None)?;
        let mut out = IndexMap::new();
        for class_id in ids {
            let count = match &counted {
                Some(counted) if self.implying(class_id)?.is_empty() => counted.get(&class_id).copied().unwrap_or(0),
                _ => self.dense_class(class_id, &window)?.iter().filter(|v| **v).count() as u64,
            };
            out.insert(class_id, count);
        }
        Ok(out)
    }

    fn counts_from_planes(&self) -> Result<Option<HashMap<i64, u64>>> {
        match self.kind() {
            "labelmap" | "layers" => {
                let Some(ceiling) = self.class_ids().iter().max() else {
                    return Ok(Some(HashMap::new()));
                };
                Ok(Some(value_counts(&self.data()?, *ceiling)?.into_iter().collect()))
            }
            "bitmask" => {
                let table = self.bit_table()?;
                if table.ids.is_empty() {
                    return Ok(Some(HashMap::new()));
                }
                let counts = popcounts(&self.data()?)?;
                let mut out = HashMap::new();
                for (position, class_id) in table.ids.iter().enumerate() {
                    let (plane, bit) = (position / BITS_PER_PLANE, position % BITS_PER_PLANE);
                    if plane < counts.len() {
                        out.insert(i64::from(*class_id), counts[plane][bit]);
                    }
                }
                Ok(Some(out))
            }
            _ => Ok(None),
        }
    }

    /// Tight `(S, 2)` index-space bounds per class; `None` when empty.
    ///
    /// Each stored volume, layer or bitplane is read once, and walked once
    /// for every class it holds ([`Self::scan_classes`]).
    pub fn class_bboxes(&self, classes: Option<&[ClassKey]>) -> Result<IndexMap<i64, Option<Array2<f32>>>> {
        self.require_voxel("class_bboxes()")?;
        let ids = self.resolve_classes(classes)?;
        let rank = self.spatial_shape()?.len();
        let mut bounds: Vec<Option<Vec<(usize, usize)>>> = vec![None; ids.len()];
        let scanned = self.scan_classes(&ids, &mut |entry, _, outer, k| {
            let found = bounds[entry].get_or_insert_with(|| vec![(usize::MAX, 0); rank]);
            for ((lo, hi), index) in found.iter_mut().zip(outer.iter().chain([&k])) {
                *lo = (*lo).min(*index);
                *hi = (*hi).max(index + 1);
            }
        })?;
        if !scanned {
            let window = self.window(None)?;
            for (entry, class_id) in ids.iter().enumerate() {
                bounds[entry] = super::select::bbox(&self.dense_class(*class_id, &window)?);
            }
        }
        let mut out = IndexMap::new();
        for (class_id, found) in ids.into_iter().zip(bounds) {
            out.insert(
                class_id,
                found.map(|b| {
                    let slices: Vec<(i64, i64)> = b.iter().map(|(lo, hi)| (*lo as i64, *hi as i64)).collect();
                    let flat = slices_to_box(&slices);
                    Array2::from_shape_vec((slices.len(), 2), flat).expect("(S, 2)")
                }),
            );
        }
        Ok(out)
    }

    /// Walk the stored voxels of each class in `entries` over the whole grid,
    /// reading each stored volume, layer or bitplane once: `visit(entry,
    /// row, outer, k)` for each voxel of class `entries[entry]`, where `row`
    /// counts the rows of the last axis in C order, `outer` is the row's
    /// index along the other axes and `k` the voxel's place in the row.  An
    /// entry's voxels come in C order.
    ///
    /// `false`, having visited nothing, where the data is not walked this
    /// way --- `probmap`, `instances` and `mask`, label data stored as
    /// another type than `uint8` or `uint16`, or a class others imply under
    /// `closure = "implicit"` --- and the caller decodes class by class with
    /// [`Self::dense_class`], which answers alike.
    pub(crate) fn scan_classes(&self, entries: &[i64], visit: &mut ClassVisitor<'_>) -> Result<bool> {
        if !matches!(self.kind(), "labelmap" | "layers" | "bitmask") {
            return Ok(false);
        }
        for class_id in entries {
            if !self.implying(*class_id)?.is_empty() {
                return Ok(false);
            }
        }
        let shape = self.spatial_shape()?;
        let ds = self.data()?;
        if shape.is_empty() || (self.kind() != "bitmask" && !matches!(data::dtype(&ds)?, DType::U8 | DType::U16)) {
            return Ok(false);
        }
        let window = self.window(None)?;
        match self.kind() {
            "bitmask" => {
                let table = self.bit_table()?;
                let mut by_plane: std::collections::BTreeMap<usize, Vec<(usize, usize)>> = Default::default();
                for (entry, class_id) in entries.iter().enumerate() {
                    if let Some(slot) = table.position_of.get(class_id) {
                        by_plane.entry(slot / BITS_PER_PLANE).or_default().push((entry, slot % BITS_PER_PLANE));
                    }
                }
                for (plane, bits) in by_plane {
                    let block = data::read_region(&ds, &window_index(Some(plane), &window))?.cast::<u64>();
                    let block = block.as_standard_layout();
                    walk_rows(block.as_slice().expect("standard layout"), &shape, |row, outer, k, word| {
                        if *word != 0 {
                            for (entry, bit) in &bits {
                                if (word >> bit) & 1 == 1 {
                                    visit(*entry, row, outer, k);
                                }
                            }
                        }
                    });
                }
            }
            kind => {
                let mut groups: std::collections::BTreeMap<Option<usize>, Vec<(usize, i64)>> = Default::default();
                let table = if kind == "layers" { Some(self.layer_table()?) } else { None };
                for (entry, class_id) in entries.iter().enumerate() {
                    match table {
                        None => groups.entry(None).or_default().push((entry, *class_id)),
                        Some(table) => {
                            if let Some(layer) = table.layer_of.get(class_id) {
                                groups.entry(Some(*layer)).or_default().push((entry, *class_id));
                            }
                        }
                    }
                }
                for (layer, wanted) in groups {
                    let block = data::read_region(&ds, &window_index(layer, &window))?;
                    let (slot, owners) = value_table(&wanted);
                    let mut hit = |row: usize, outer: &[usize], k: usize, value: usize| {
                        if let Some(index) = slot.get(value).filter(|i| **i != u32::MAX) {
                            for entry in &owners[*index as usize] {
                                visit(*entry, row, outer, k);
                            }
                        }
                    };
                    match &block {
                        NdArray::U8(a) => {
                            let a = a.as_standard_layout();
                            walk_rows(a.as_slice().expect("standard layout"), &shape, |row, outer, k, v| {
                                hit(row, outer, k, usize::from(*v))
                            });
                        }
                        NdArray::U16(a) => {
                            let a = a.as_standard_layout();
                            walk_rows(a.as_slice().expect("standard layout"), &shape, |row, outer, k, v| {
                                hit(row, outer, k, usize::from(*v))
                            });
                        }
                        _ => unreachable!("checked to be uint8 or uint16"),
                    }
                }
            }
        }
        Ok(true)
    }

    /// The in-band ignore region (§7.7) of a `labelmap` or `layers`
    /// annotation: a voxel is ignored where any layer marks it.
    pub fn ignore_mask(&self, roi: Option<&[Slice]>) -> Result<ArrayD<bool>> {
        self.require_kind(&["labelmap", "layers"], "ignore_mask()")?;
        let window = self.window(roi)?;
        Ok(match self.in_band_ignore(&window)? {
            Some(region) => region,
            None => ArrayD::from_elem(IxDyn(&window_shape(&window)), false),
        })
    }

    /// [`Self::ignore_mask`] over a window, or `None` without a read when the
    /// data holds no ignore id in band ([`Self::may_encode_ignore`]).
    fn in_band_ignore(&self, window: &Window) -> Result<Option<ArrayD<bool>>> {
        if !self.may_encode_ignore()? {
            return Ok(None);
        }
        let ignore = self.ignore_id();
        if self.kind() == "labelmap" {
            return Ok(Some(equals(&data::read_region(&self.data()?, &window_index(None, window))?, ignore)));
        }
        let mut index = vec![Index::Slice(Slice::full())];
        index.extend(window_index(None, window));
        let block = equals(&data::read_region(&self.data()?, &index)?, ignore);
        Ok(Some(block.map_axis(Axis(0), |lane| lane.iter().any(|v| *v))))
    }

    /// The §7.7 ignore region over the whole grid, from the annotation alone:
    /// in band (`labelmap`, `layers`) and the sibling `mask` its
    /// `ignore_mask` names, read beside it in the same `annotations/` group.
    /// `None` where no voxel is ignored.
    pub fn ignore_region(&self) -> Result<Option<ArrayD<bool>>> {
        let window = self.window(None)?;
        let in_band = if matches!(self.kind(), "labelmap" | "layers") { self.in_band_ignore(&window)? } else { None };
        self.with_sibling(&window, in_band)
    }

    /// `region` --- the in-band part over `window`, where it was read --- with
    /// the sibling `mask`'s part added; `None` where no voxel is ignored.
    fn with_sibling(&self, window: &Window, mut region: Option<ArrayD<bool>>) -> Result<Option<ArrayD<bool>>> {
        if let Some(sibling) = self.ignore_sibling()? {
            let slices: Vec<Slice> = window.iter().map(|(a, b)| Slice::new(*a as i64, *b as i64)).collect();
            let mask = sibling.read_mask(Some(&slices))?;
            if mask.shape() != window_shape(window).as_slice() {
                return Err(Error::coded(
                    "E413",
                    format!(
                        "{}: ignore_mask names {}, whose grid is not this annotation's; a mask delimits the voxels \
                         of the grid it shares",
                        repr_str(&self.ann_id),
                        repr_str(&sibling.ann_id)
                    ),
                ));
            }
            region = Some(match region {
                Some(mut inband) => {
                    inband.zip_mut_with(&mask, |r, m| *r |= *m);
                    inband
                }
                None => mask,
            });
        }
        Ok(region.filter(|r| r.iter().any(|v| *v)))
    }

    /// Whether any voxel of the annotation is ignored (§7.7): a scan, in
    /// slabs, of the in-band data and of the sibling mask.
    fn ignore_present(&self) -> Result<bool> {
        if self.encodes_ignore()? {
            return Ok(true);
        }
        match self.ignore_sibling()? {
            Some(sibling) => any_nonzero(&sibling.data()?),
            None => Ok(false),
        }
    }

    /// The sibling `mask` annotation the header's `ignore_mask` names, read
    /// beside this one in the same `annotations/` group (E413 when there is
    /// no such `mask`), opened once.
    fn ignore_sibling(&self) -> Result<Option<&Annotation>> {
        let sibling = cached(&self.sibling, || {
            let Some(reference) = self.header.ignore_mask.clone() else {
                return Ok(None);
            };
            let name = crate::sample::annotation_id(&reference);
            let path = self.group.name();
            let parent = path.rsplit_once('/').map_or("/", |(p, _)| p);
            let sibling = ops::child_group(&self.group.file()?.group(parent)?, name)
                .map(|g| Annotation::open(name, g, self.grids.clone(), self.label_set.clone()))
                .transpose()?
                .filter(|a| a.kind() == "mask")
                .ok_or_else(|| {
                    Error::coded(
                        "E413",
                        format!(
                            "{}: ignore_mask names {}, which is not a `mask` annotation in this file",
                            repr_str(&self.ann_id),
                            repr_str(name)
                        ),
                    )
                })?;
            Ok(Some(Box::new(sibling)))
        })?;
        Ok(sibling.as_deref())
    }

    /// A `mask` annotation's volume over a window.
    pub fn read_mask(&self, roi: Option<&[Slice]>) -> Result<ArrayD<bool>> {
        self.require_kind(&["mask"], "read()")?;
        let window = self.window(roi)?;
        Ok(data::read_region(&self.data()?, &window_index(None, &window))?.nonzero_mask())
    }

    // -- layers -------------------------------------------------------------

    fn layer_table(&self) -> Result<&LayerTable> {
        cached(&self.layers, || {
            let ds = self.optional_dataset("layer_class_ids")?.ok_or_else(|| {
                Error::coded(
                    "E410",
                    format!("annotation {}: `layers` requires `layer_class_ids`", repr_str(&self.ann_id)),
                )
            })?;
            let raw = data::read(&ds)?;
            let stored = raw.cast::<i64>();
            let rows: Vec<Vec<i64>> = if stored.ndim() == 2 {
                stored.outer_iter().map(|row| row.iter().copied().collect()).collect()
            } else {
                vec![stored.iter().copied().collect()]
            };
            let planes = self.optional_dataset("data")?.map(|d| d.shape().first().copied().unwrap_or(0));
            let shape_of = |name: &str| (name == "layer_class_ids").then(|| ds.shape());
            let found = layout::table_problems("layers", &shape_of, planes, Some(&rows), Some(self.class_ids()));
            if let Some(problem) = found.first() {
                return Err(problem.error(&self.ann_id));
            }
            let raw = raw.cast::<u16>();
            let table = if raw.ndim() == 2 {
                raw.into_dimensionality::<ndarray::Ix2>()?
            } else {
                let n = raw.len();
                raw.into_shape_with_order((1, n))?
            };
            let mut layer_of = IndexMap::new();
            for (layer, row) in table.outer_iter().enumerate() {
                for value in row.iter().filter(|v| **v != 0) {
                    layer_of.insert(i64::from(*value), layer);
                }
            }
            Ok(LayerTable { table, layer_of })
        })
    }

    /// `(L, K)` class ids per layer, zero-padded.
    pub fn layer_class_ids(&self) -> Result<Array2<u16>> {
        self.require_kind(&["layers"], "layer_class_ids")?;
        Ok(self.layer_table()?.table.clone())
    }

    /// The number of stored layers.
    pub fn n_layers(&self) -> Result<usize> {
        self.require_kind(&["layers"], "n_layers")?;
        Ok(self.data()?.shape().first().copied().unwrap_or(0))
    }

    /// Class id -> layer index.  Every class appears in exactly one layer.
    pub fn layer_of(&self) -> Result<IndexMap<i64, usize>> {
        self.require_kind(&["layers"], "layer_of")?;
        Ok(self.layer_table()?.layer_of.clone())
    }

    /// The classes of each layer, padding dropped.
    pub fn layer_classes(&self) -> Result<Vec<Vec<i64>>> {
        self.require_kind(&["layers"], "layer_classes()")?;
        Ok(self
            .layer_table()?
            .table
            .outer_iter()
            .map(|row| row.iter().filter(|v| **v != 0).map(|v| i64::from(*v)).collect())
            .collect())
    }

    /// One layer's labelmap, sliced in a single call (spec §14.5).
    pub fn read_layer(&self, layer: usize, roi: Option<&[Slice]>) -> Result<NdArray> {
        self.require_kind(&["layers"], "read_layer()")?;
        let window = self.window(roi)?;
        data::read_region(&self.data()?, &window_index(Some(layer), &window))
    }

    // -- bitmask ------------------------------------------------------------

    fn bit_table(&self) -> Result<&BitTable> {
        cached(&self.bits, || {
            let ds = self.optional_dataset("bit_class_ids")?.ok_or_else(|| {
                Error::coded(
                    "E410",
                    format!("annotation {}: `bitmask` requires `bit_class_ids`", repr_str(&self.ann_id)),
                )
            })?;
            let raw = data::read(&ds)?;
            let rows = vec![raw.cast::<i64>().iter().copied().collect::<Vec<i64>>()];
            let planes = self.optional_dataset("data")?.map(|d| d.shape().first().copied().unwrap_or(0));
            let shape_of = |name: &str| (name == "bit_class_ids").then(|| ds.shape());
            let found = layout::table_problems("bitmask", &shape_of, planes, Some(&rows), Some(self.class_ids()));
            if let Some(problem) = found.first() {
                return Err(problem.error(&self.ann_id));
            }
            let ids: Vec<u16> = raw.cast::<u16>().iter().copied().collect();
            let position_of = ids.iter().enumerate().map(|(i, c)| (i64::from(*c), i)).collect();
            Ok(BitTable { ids, position_of })
        })
    }

    /// Bit position -> class id.
    pub fn bit_class_ids(&self) -> Result<Vec<u16>> {
        self.require_kind(&["bitmask"], "bit_class_ids")?;
        Ok(self.bit_table()?.ids.clone())
    }

    /// The number of `uint64` planes.
    pub fn n_planes(&self) -> Result<usize> {
        self.require_kind(&["bitmask"], "n_planes")?;
        Ok(self.data()?.shape().first().copied().unwrap_or(0))
    }

    /// Class id -> bit position.
    pub fn position_of(&self) -> Result<HashMap<i64, usize>> {
        self.require_kind(&["bitmask"], "position_of")?;
        Ok(self.bit_table()?.position_of.clone())
    }

    /// Every class present at one voxel, in O(P) reads.
    pub fn classes_at(&self, voxel: &[i64]) -> Result<Vec<i64>> {
        self.require_kind(&["bitmask"], "classes_at()")?;
        let window = self.voxel_window(voxel)?;
        let ids = &self.bit_table()?.ids;
        let ds = self.data()?;
        let mut out = Vec::new();
        for plane in 0..ds.shape().first().copied().unwrap_or(0) {
            let word = data::read_region(&ds, &window_index(Some(plane), &window))?
                .cast::<u64>()
                .iter()
                .next()
                .copied()
                .unwrap_or(0);
            if word == 0 {
                continue;
            }
            for bit in 0..BITS_PER_PLANE {
                if (word >> bit) & 1 == 1 {
                    let position = plane * BITS_PER_PLANE + bit;
                    if position < ids.len() {
                        out.push(i64::from(ids[position]));
                    }
                }
            }
        }
        Ok(out)
    }

    // -- probmap ------------------------------------------------------------

    /// Whether the probabilities sum to one per voxel (§7.5).
    pub fn normalized(&self) -> Result<bool> {
        Ok(attrs::get_bool(&self.group, "normalized")?.unwrap_or(false))
    }

    /// The declared decision threshold (§7.5), or the 0.5 default.
    pub fn threshold(&self) -> Result<f64> {
        Ok(attrs::get_f64(&self.group, "threshold")?.unwrap_or(DEFAULT_THRESHOLD))
    }

    /// `(C, *roi_shape)` float probabilities for the requested classes.
    pub fn probabilities(&self, classes: Option<&[ClassKey]>, roi: Option<&[Slice]>) -> Result<ArrayD<f32>> {
        self.require_kind(&["probmap"], "probabilities()")?;
        let ids = self.resolve_classes(classes)?;
        let window = self.window(roi)?;
        let mut shape = vec![ids.len()];
        shape.extend(window_shape(&window));
        let mut out = ArrayD::<f32>::zeros(IxDyn(&shape));
        let ds = self.data()?;
        for (i, class_id) in ids.iter().enumerate() {
            if let Some(position) = self.class_ids().iter().position(|c| c == class_id) {
                let block = data::read_region(&ds, &window_index(Some(position), &window))?.cast::<f32>();
                out.index_axis_mut(Axis(0), i).assign(&block);
            }
        }
        Ok(out)
    }

    // -- instances ----------------------------------------------------------

    /// The object columns, read once.  Stored masks are held to their boxes
    /// and their grid here, before any is decoded: a mask shape is an
    /// allocation, and one nobody bounded aborted the process (§7.4).
    fn instance_columns(&self) -> Result<&InstanceColumns> {
        cached(&self.columns, || {
            let boxes = self.read_as::<f32>("boxes")?;
            let (mask_offsets, mask_shapes) = if self.has_masks()? {
                let offsets: Vec<i64> = self.read_as::<i64>("mask_offsets")?.iter().copied().collect();
                let shapes = self.read_as::<i64>("mask_shapes")?;
                let stored = self.dataset_shape("mask_data")?.iter().product();
                let grid = self.spatial_shape()?;
                let found = layout::mask_problems(&boxes.mapv(f64::from), &offsets, &shapes, stored, Some(&grid));
                if let Some(problem) = found.first() {
                    return Err(problem.error(&self.ann_id));
                }
                (Some(offsets.iter().map(|v| *v as u64).collect()), Some(shapes))
            } else {
                (None, None)
            };
            Ok(InstanceColumns {
                boxes,
                class_ids: self.read_as::<u16>("class_ids")?.iter().copied().collect(),
                instance_ids: self.read_as::<u64>("instance_ids")?.iter().copied().collect(),
                scores: self.read_optional::<f32>("scores")?.map(|a| a.iter().copied().collect()),
                mask_offsets,
                mask_shapes,
            })
        })
    }

    /// Whether an `instances` annotation stores per-object masks.
    pub fn has_masks(&self) -> Result<bool> {
        Ok(self.optional_dataset("mask_data")?.is_some())
    }

    /// `(N, S, 2)` object boxes (`instances` and `boxes`).
    pub fn boxes(&self) -> Result<ArrayD<f32>> {
        match self.kind() {
            "instances" => Ok(self.instance_columns()?.boxes.clone()),
            "boxes" => self.read_as::<f32>("boxes"),
            _ => self.require_kind(&["instances", "boxes"], "boxes").map(|_| unreachable!()),
        }
    }

    /// Per-object class ids.
    pub fn object_class_ids(&self) -> Result<Vec<u16>> {
        match self.kind() {
            "instances" => Ok(self.instance_columns()?.class_ids.clone()),
            "points" => match self.read_optional::<u16>("class_ids")? {
                Some(a) => Ok(a.iter().copied().collect()),
                None => Ok(vec![0; self.dataset_shape("points")?.first().copied().unwrap_or(0)]),
            },
            "contours" => Ok(self.read_as::<u16>("contour_class_ids")?.iter().copied().collect()),
            "polylines" => Ok(self.read_as::<u16>("polyline_class_ids")?.iter().copied().collect()),
            "mesh" => match self.read_optional::<u16>("mesh_class_ids")? {
                Some(a) => Ok(a.iter().copied().collect()),
                None => Ok(self.class_ids().iter().map(|c| *c as u16).collect()),
            },
            k if is_geometric_kind(k) => Ok(self.read_as::<u16>("class_ids")?.iter().copied().collect()),
            _ => self
                .require_kind(
                    &["instances", "boxes", "obb", "keypoints", "points", "contours", "polylines", "mesh"],
                    "object_class_ids",
                )
                .map(|_| unreachable!()),
        }
    }

    /// Object ids, widened to `uint64` and never narrowed (§7.4, §8.2).
    pub fn instance_ids(&self) -> Result<Option<Vec<u64>>> {
        match self.kind() {
            "instances" => Ok(Some(self.instance_columns()?.instance_ids.clone())),
            k if is_geometric_kind(k) => {
                Ok(self.read_optional::<u64>("instance_ids")?.map(|a| a.iter().copied().collect()))
            }
            _ => self.require_kind(&["instances"], "instance_ids").map(|_| unreachable!()),
        }
    }

    /// Per-object scores, when stored.
    pub fn scores(&self) -> Result<Option<Vec<f32>>> {
        match self.kind() {
            "instances" => Ok(self.instance_columns()?.scores.clone()),
            k if is_geometric_kind(k) => Ok(self.read_optional::<f32>("scores")?.map(|a| a.iter().copied().collect())),
            _ => self.require_kind(&["instances"], "scores").map(|_| unreachable!()),
        }
    }

    /// The number of objects of an `instances` annotation.
    pub fn n_objects(&self) -> Result<usize> {
        self.require_kind(&["instances"], "n_objects")?;
        Ok(self.instance_columns()?.boxes.shape().first().copied().unwrap_or(0))
    }

    /// Decode one object's bbox-local mask.
    pub fn crop(&self, index: usize) -> Result<Option<ArrayD<bool>>> {
        self.require_kind(&["instances"], "crop()")?;
        if !self.has_masks()? {
            return Ok(None);
        }
        let columns = self.instance_columns()?;
        let offsets = columns.mask_offsets.as_ref().expect("masks present");
        let shapes = columns.mask_shapes.as_ref().expect("masks present");
        if index + 1 >= offsets.len() {
            return Err(Error::Index(format!(
                "index {index} is out of bounds for {} objects",
                offsets.len().saturating_sub(1)
            )));
        }
        let (start, stop) = (offsets[index] as i64, offsets[index + 1] as i64);
        let packed =
            data::read_region(&self.dataset("mask_data")?, &[Index::Slice(Slice::new(start, stop))])?.cast::<u8>();
        let packed: Vec<u8> = packed.iter().copied().collect();
        let local_offsets = [0u64, packed.len() as u64];
        let shape = shapes.index_axis(Axis(0), index).to_owned().insert_axis(Axis(0));
        Ok(Some(decode_crop(&packed, &local_offsets, &shape.into_dyn(), 0)?))
    }

    /// Every object, in stored order (`instances`, `boxes`).
    ///
    /// A score that is not finite --- the NaN a writer stores for an object
    /// given none --- is no score.  A `boxes` annotation without
    /// `instance_ids` (§8.2 makes them optional) identifies a box by its row:
    /// `instance_id` is then its 0-based position, which names no object
    /// beyond this annotation and joins nothing across visits.
    pub fn instances(&self) -> Result<Vec<Instance>> {
        let finite = |scores: Option<&Vec<f32>>, i: usize| {
            scores.and_then(|s| s.get(i)).filter(|v| v.is_finite()).map(|v| f64::from(*v))
        };
        match self.kind() {
            "instances" => {
                let columns = self.instance_columns()?;
                let n = columns.boxes.shape().first().copied().unwrap_or(0);
                // One read of the packed masks, rather than one per object.
                let packed: Option<Vec<u8>> = match self.optional_dataset("mask_data")? {
                    Some(ds) if n > 0 => Some(data::read(&ds)?.cast::<u8>().iter().copied().collect()),
                    _ => None,
                };
                (0..n)
                    .map(|i| {
                        let mask = match (&packed, &columns.mask_offsets, &columns.mask_shapes) {
                            (Some(packed), Some(offsets), Some(shapes)) => {
                                Some(decode_crop(packed, offsets, shapes, i)?)
                            }
                            _ => None,
                        };
                        Ok(Instance {
                            index: i,
                            instance_id: columns.instance_ids[i],
                            class_id: i64::from(columns.class_ids[i]),
                            bbox: box_at(&columns.boxes, i)?,
                            mask,
                            score: finite(columns.scores.as_ref(), i),
                        })
                    })
                    .collect()
            }
            "boxes" => {
                let boxes = self.boxes()?;
                let classes = self.object_class_ids()?;
                let ids = self.instance_ids()?;
                let scores = self.scores()?;
                let n = boxes.shape().first().copied().unwrap_or(0);
                (0..n)
                    .map(|i| {
                        Ok(Instance {
                            index: i,
                            instance_id: ids.as_ref().map(|v| v[i]).unwrap_or(i as u64),
                            class_id: i64::from(classes[i]),
                            bbox: box_at(&boxes, i)?,
                            mask: None,
                            score: finite(scores.as_ref(), i),
                        })
                    })
                    .collect()
            }
            k if is_voxel_kind(k) => Err(Error::invalid(format!(
                "annotation {} of kind {} does not carry instance identity, and it cannot be recovered from a \
                 dense encoding: transcoding to `instances` would merge every object of a class into one and mint \
                 an id that belongs to none of them (§7.4). Re-derive the objects from whatever source had them.",
                repr_str(&self.ann_id),
                repr_str(k)
            ))),
            _ => self.require_kind(&["instances", "boxes"], "instances()").map(|_| unreachable!()),
        }
    }

    /// One object by its id (`KeyError` when absent): the first object
    /// carrying it, the only one decoded.
    pub fn instance(&self, instance_id: u64) -> Result<Instance> {
        self.require_kind(&["instances"], "instance()")?;
        let columns = self.instance_columns()?;
        let Some(index) = columns.instance_ids.iter().position(|i| *i == instance_id) else {
            return Err(Error::Key(format!("annotation {} has no instance {instance_id}", repr_str(&self.ann_id))));
        };
        Ok(Instance {
            index,
            instance_id,
            class_id: i64::from(columns.class_ids[index]),
            bbox: box_at(&columns.boxes, index)?,
            mask: self.crop(index)?,
            score: columns.scores.as_ref().and_then(|s| s.get(index)).filter(|v| v.is_finite()).map(|v| f64::from(*v)),
        })
    }

    /// `instance_id -> class_id`: the join key for longitudinal tracking.
    pub fn tracking(&self) -> Result<IndexMap<u64, i64>> {
        self.require_kind(&["instances"], "tracking()")?;
        let columns = self.instance_columns()?;
        Ok(columns.instance_ids.iter().zip(&columns.class_ids).map(|(i, c)| (*i, i64::from(*c))).collect())
    }

    /// [`Self::read_planes`] for `instances`: each object of a wanted class
    /// that meets the window is decoded once and painted into every plane
    /// that wants its class.
    fn paint_instances(
        &self,
        sources: &[(usize, i64)],
        window: &Window,
        flat: &mut [bool],
        voxels: usize,
    ) -> Result<()> {
        let shape = window_shape(window);
        let mut planes_of: HashMap<i64, Vec<usize>> = HashMap::new();
        for (position, class_id) in sources {
            let planes = planes_of.entry(*class_id).or_default();
            if !planes.contains(position) {
                planes.push(*position);
            }
        }
        let columns = self.instance_columns()?;
        for (index, object_class) in columns.class_ids.iter().enumerate() {
            let Some(planes) = planes_of.get(&i64::from(*object_class)) else { continue };
            // Unclipped, deliberately: these slices are the frame the stored
            // crop was cut in; intersecting with the window does the clipping.
            let bbox: Vec<f64> = box_at(&columns.boxes, index)?.iter().map(|v| f64::from(*v)).collect();
            let obj = box_to_slices(&bbox, None)?;
            let mut local = Vec::with_capacity(obj.len());
            let mut target = Vec::with_capacity(obj.len());
            let mut empty = false;
            for ((want_lo, want_hi), (have_lo, have_hi)) in window.iter().zip(&obj) {
                let (want_lo, want_hi) = (*want_lo as i64, *want_hi as i64);
                let lo = want_lo.max(*have_lo);
                let hi = want_hi.min(*have_hi);
                if hi <= lo {
                    empty = true;
                    break;
                }
                // Saturating, for the same reason as the painter's
                // (`instances_to_masks`): only a crop reads it, and none fits
                // a box that starts at the end of the integers.
                local.push((lo.saturating_sub(*have_lo) as usize, hi.saturating_sub(*have_lo) as usize));
                target.push(((lo - want_lo) as usize, (hi - want_lo) as usize));
            }
            if empty {
                continue;
            }
            let crop = self.crop(index)?;
            let src = match &crop {
                None => None,
                Some(crop) => {
                    let mut src = crop.view();
                    for (axis, (a, b)) in local.iter().enumerate() {
                        let n = src.shape()[axis];
                        src.slice_axis_inplace(Axis(axis), ndarray::Slice::from((*a).min(n)..(*b).min(n)));
                    }
                    Some(src)
                }
            };
            for plane in planes {
                let part = &mut flat[plane * voxels..(plane + 1) * voxels];
                let mut view = ndarray::ArrayViewMutD::from_shape(IxDyn(&shape), part)?;
                for (axis, (a, b)) in target.iter().enumerate() {
                    view.slice_axis_inplace(Axis(axis), ndarray::Slice::from(*a..*b));
                }
                match &src {
                    None => view.fill(true),
                    Some(src) if src.shape() == view.shape() => {
                        ndarray::Zip::from(&mut view).and(src).for_each(|o, s| *o |= *s);
                    }
                    Some(_) => {
                        return Err(Error::coded(
                            "E405",
                            format!(
                                "annotation {}: object {index}'s stored mask does not cover its box",
                                repr_str(&self.ann_id)
                            ),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    // -- summaries ------------------------------------------------------------

    /// JSON-safe description for `medh5 info`.
    pub fn summary(&self) -> Result<Value> {
        if self.is_voxel() {
            return self.voxel_summary();
        }
        if self.is_geometric() {
            return self.geometric_summary();
        }
        self.classification_summary()
    }

    fn voxel_summary(&self) -> Result<Value> {
        let mut out = Map::new();
        out.insert("id".into(), json!(self.ann_id));
        out.insert("kind".into(), json!(self.kind()));
        out.insert("task".into(), json!(self.task()));
        out.insert("grid".into(), json!(self.grid_id()));
        out.insert("timepoints".into(), json!(self.timepoints()));
        out.insert("classes".into(), json!(self.class_ids().len()));
        out.insert("annotated_classes".into(), json!(self.annotated_class_ids().len()));
        out.insert("fully_covered".into(), json!(self.is_fully_covered()));
        out.insert("closure".into(), json!(self.closure()));
        out.insert("quality".into(), json!(self.quality_key()));
        out.insert("prov".into(), json!(self.prov()));
        match self.kind() {
            "mask" => {
                out.insert("true_voxels".into(), json!(count_nonzero(&self.data()?)?));
            }
            "layers" => {
                out.insert("layers".into(), json!(self.n_layers()?));
            }
            "bitmask" => {
                out.insert("planes".into(), json!(self.n_planes()?));
            }
            "probmap" => {
                out.insert("normalized".into(), json!(self.normalized()?));
                out.insert("threshold".into(), crate::json::num(self.threshold()?));
            }
            "instances" => {
                out.insert("objects".into(), json!(self.n_objects()?));
                out.insert("has_masks".into(), json!(self.has_masks()?));
                out.insert("instance_ids".into(), json!(self.instance_columns()?.instance_ids));
            }
            _ => {}
        }
        Ok(Value::Object(out))
    }
}

/// Row `i` of an `(N, S, 2)` box array as an `(S, 2)` array.
pub fn box_at(boxes: &ArrayD<f32>, i: usize) -> Result<Array2<f32>> {
    let row = boxes.index_axis(Axis(0), i).to_owned();
    let s = row.len() / 2;
    Ok(row.into_shape_with_order((s, 2))?)
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::annotations::InstanceInput;
    use crate::sample::{create, open_sample, GridOptions, SegmentationOptions, SegmentationSource};

    #[allow(unused_imports)]
    use super::*;

    /// A box-only object from one end of the integers to the other on every
    /// axis, as a file can state one: `float32` bounds of +-3e38 floor to
    /// `i64::MIN` and `i64::MAX`.  Written with a mask, which is then
    /// removed and the box widened.
    pub(crate) fn wide_box(path: &std::path::Path) {
        let mut w = create(path, Some("wide"), None, "portable", &[]).unwrap();
        w.add_grid("g", &[4, 5, 6], &[1.0; 3], GridOptions::default()).unwrap();
        let ct = crate::array::NdArray::from_vec(&[4, 5, 6], vec![0i16; 120]).unwrap();
        w.add_image("CT", &ct, "g", "CT", crate::sample::ImageOptions::default()).unwrap();
        let mut mask = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[4, 5, 6]), false);
        mask[[1, 1, 1]] = true;
        let object =
            InstanceInput { class_id: 1, instance_id: 1, mask: Some(mask), bbox: None, crop: None, score: None };
        let source = SegmentationSource::Instances(vec![object]);
        w.add_segmentation("lesions", "g", source, SegmentationOptions::default()).unwrap();
        w.commit(true).unwrap();
        let file = crate::h5::file::open_rw(path).unwrap();
        let group = file.group("annotations/lesions").unwrap();
        for name in ["mask_data", "mask_offsets", "mask_shapes"] {
            group.unlink(name).unwrap();
        }
        let wide = ndarray::Array3::from_shape_vec((1, 3, 2), [-3e38f32, 3e38].repeat(3)).unwrap();
        group.dataset("boxes").unwrap().write(&wide).unwrap();
        drop(group);
        file.close().unwrap();
    }

    /// The box spans `2**64 - 1` voxels an axis, which `b - a` overflowed:
    /// one axis of it is a count a `u64` holds, three are refused.
    #[test]
    fn s7_4_a_box_past_the_integers_counts_exactly_or_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        wide_box(&dir.path().join("w.medh5"));
        let sample = open_sample(&dir.path().join("w.medh5")).unwrap();
        let mut object = sample.annotation("lesions").unwrap().instances().unwrap().remove(0);
        let err = object.voxel_count().unwrap_err();
        assert!(err.to_string().contains("more voxels than 64 bits count"), "{err}");
        object.bbox = ndarray::array![[-3e38f32, 3e38]];
        assert_eq!(object.voxel_count().unwrap(), u64::MAX);
        object.bbox = ndarray::array![[-3e38f32, 3e38], [1.5, 2.5]];
        assert_eq!(object.voxel_count().unwrap(), u64::MAX);
        object.bbox = ndarray::array![[-3e38f32, 3e38], [1.5, 3.5]];
        assert!(object.voxel_count().is_err(), "twice 2**64 - 1");
    }

    /// Painting the object subtracted its box's start from the window's:
    /// `0 - i64::MIN` overflowed.  It paints the whole grid.
    #[test]
    fn s7_4_a_box_past_the_integers_paints_the_grid() {
        let dir = tempfile::tempdir().unwrap();
        wide_box(&dir.path().join("w.medh5"));
        let sample = open_sample(&dir.path().join("w.medh5")).unwrap();
        let ann = sample.annotation("lesions").unwrap();
        let mask = ann.dense_class(1, &ann.window(None).unwrap()).unwrap();
        assert_eq!((mask.shape(), mask.iter().all(|v| *v)), (&[4, 5, 6][..], true));
    }
}
