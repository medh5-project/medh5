//! Annotations follow their images through a resample (1.0 §6–§9).
//!
//! Every annotation on a source grid is carried onto the target under the
//! outputs' suffix, naming its source in `derived_from` and keeping its
//! classes and its coverage: what was looked for on the source was looked
//! for on the copy.  Voxel encodings move nearest in their own encoding;
//! `instances` move object by object under their own ids; a probability map
//! moves linearly.  Points, keypoints and meshes map exactly through the same
//! mapping as the voxels.  A box stays a box only where the mapping keeps the
//! axes, an oriented box only where it keeps right angles, and a contour or
//! polyline only where the mapping is affine and its plane lands on a plane
//! of the target grid.  Anything else is refused, never approximated.
//!
//! Target voxels outside the source were never examined, so a class-bearing
//! voxel annotation carried onto them gets an ignore region (§7.7) over
//! them: its own ignore mask, carried and widened, or a new mask.

use std::collections::{BTreeMap, HashMap, HashSet};

use indexmap::IndexMap;
use ndarray::{Array2, ArrayD, Dimension, IxDyn};
use serde_json::{json, Value};

use super::{blocks, linear_sample, unit_scale, Body, Mapping, ResamplePlan, Transform};
use crate::annotations::encode::{encode_instances, encode_probmap, InstanceInput};
use crate::annotations::header::AnnotationHeader;
use crate::annotations::payload::{Payload, PayloadData};
use crate::annotations::read::Annotation;
use crate::array::{DType, Element, NdArray};
use crate::geometry::affine::{box_to_slices, slices_to_box};
use crate::geometry::grid::Grid;
use crate::geometry::linalg::{det, inv};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::{data, ops};
use crate::ids::validate_id;
use crate::json::{float_repr, repr_str};
use crate::sample::reader::{KeyKind, Sample};
use crate::sample::writer::SampleWriter;
use crate::{Error, Result};

/// One annotation a resample carried onto the target.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ResampledAnnotation {
    /// The annotation carried.
    pub source: String,
    /// The annotation written.
    pub annotation_id: String,
    pub kind: String,
    /// The ignore mask the written annotation names (§7.7): its own carried,
    /// or a new one over the target voxels outside the source.
    pub ignore_mask: Option<String>,
    /// What the carry changed beyond the coordinates, when anything.
    pub note: Option<String>,
}

impl ResampledAnnotation {
    pub fn to_json(&self) -> Value {
        json!({
            "source": self.source,
            "annotation_id": self.annotation_id,
            "kind": self.kind,
            "ignore_mask": self.ignore_mask,
            "note": self.note,
        })
    }
}

/// The voxel kinds that hold classes: what an ignore region applies to.
const CLASS_VOXEL_KINDS: [&str; 5] = ["labelmap", "layers", "bitmask", "instances", "probmap"];
/// The voxel kinds carried by gathering their stored values, nearest.
const DENSE_KINDS: [&str; 4] = ["labelmap", "layers", "bitmask", "mask"];
/// The attributes the header writes; every other attribute is carried as
/// stored, unless the payload sets it.
const HEADER_ATTRS: [&str; 15] = [
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
    "digest",
];

/// Target voxels one mapping block holds.
const BLOCK_VOXELS: usize = 1 << 18;

/// How a class-bearing voxel annotation's ignore region is carried.
#[derive(Debug, Clone, PartialEq)]
enum Ignore {
    /// It names none, and the target holds no voxel outside the source.
    None,
    /// Its own ignore mask, carried under this id.
    Carried(String),
    /// A new mask under this id, written when any target voxel lies outside
    /// the source.
    New(String),
}

/// One annotation to carry.
struct Item {
    source: String,
    output: String,
    kind: String,
    /// The source grid it is carried from.
    grid: String,
    /// Bound to that grid (false: a world-space annotation of its frame).
    on_grid: bool,
    /// Carried voxel by voxel (an `instances` annotation without masks is
    /// carried as its boxes).
    voxel: bool,
    ignore: Ignore,
    /// A mask a carried annotation names as its ignore mask: the target
    /// voxels outside the source join it.
    widen: bool,
}

/// What [`prepare`] decided: the annotations to carry, and the ones left.
pub(super) struct Carry {
    items: Vec<Item>,
    geometry: IndexMap<String, Geometry>,
    pub(super) not_carried: Vec<String>,
}

impl Carry {
    /// The annotations read, as sample paths.
    pub(super) fn inputs(&self) -> Vec<String> {
        self.items.iter().map(|item| format!("annotations/{}", item.source)).collect()
    }
}

/// How coordinates on a source grid map onto the target.
struct Geometry {
    source: Grid,
    target: Grid,
    /// Source index to target index, `(S+1, S+1)`, when the mapping is affine.
    index: Option<Array2<f64>>,
    /// Source world to target world, in each grid's units, when affine.
    world: Option<Array2<f64>>,
    /// The transform from the source's frame to the target's, and the unit
    /// factors into and out of it, for a path that is not affine.
    forward: Option<(Transform, f64, f64)>,
}

impl Geometry {
    fn new(sample: &Sample, mapping: &Mapping, source: &Grid, target: &Grid) -> Result<Geometry> {
        let s = source.n_spatial();
        match mapping {
            Mapping::Affine(m) => {
                let factor = source.world_scale_into(target)?;
                Ok(Geometry {
                    source: source.clone(),
                    target: target.clone(),
                    index: Some(inv(m)?),
                    world: Some(scale(s, factor)),
                    forward: None,
                })
            }
            Mapping::Through(_) => {
                let forward = sample.transform_between_as(&source.grid_id, &target.grid_id, Some(KeyKind::Grid))?;
                let Some(transform) = forward else {
                    return Ok(Geometry {
                        source: source.clone(),
                        target: target.clone(),
                        index: None,
                        world: None,
                        forward: None,
                    });
                };
                let units = transform.units().to_string();
                let (Some(into), Some(out)) = (unit_scale(&source.units, &units), unit_scale(&units, &target.units))
                else {
                    return Err(Error::coded(
                        "E506",
                        format!(
                            "transform {} is in {} and grids {} and {} are in {} and {}; no factor relates them (§3.5)",
                            repr_str(&transform.transform_id),
                            repr_str(&units),
                            repr_str(&source.grid_id),
                            repr_str(&target.grid_id),
                            repr_str(&source.units),
                            repr_str(&target.units)
                        ),
                    ));
                };
                let world = path_matrix(&transform)?
                    .filter(|m| m.nrows() == s + 1)
                    .map(|m| scale(s, out).dot(&m).dot(&scale(s, into)));
                let index = match &world {
                    Some(w) => Some(inv(&target.affine())?.dot(w).dot(&source.affine())),
                    None => None,
                };
                Ok(Geometry {
                    source: source.clone(),
                    target: target.clone(),
                    index,
                    world,
                    forward: Some((transform, into, out)),
                })
            }
        }
    }

    /// The affine map of `world` (else index) coordinates, when there is one.
    fn linear(&self, world: bool) -> Option<&Array2<f64>> {
        if world {
            self.world.as_ref()
        } else {
            self.index.as_ref()
        }
    }

    /// Whether points can be mapped at all.
    fn can_move(&self) -> bool {
        self.index.is_some() || self.forward.is_some()
    }

    /// `(N, S)` points in the source's `world` (else index) coordinates, in
    /// the target's.
    fn points(&self, world: bool, points: &Array2<f64>) -> Result<Array2<f64>> {
        if let Some(m) = self.linear(world) {
            return Ok(apply(m, points));
        }
        let Some((transform, into, out)) = &self.forward else {
            return Err(Error::invalid("no transform path maps the source's coordinates onto the target"));
        };
        let (n, s) = points.dim();
        let flat: Vec<f64> = points.iter().copied().collect();
        let source_world: Vec<f64> =
            if world { flat } else { self.source.index_to_world(&flat)? }.into_iter().map(|v| v * into).collect();
        let moved = transform.transform_points(&ArrayD::from_shape_vec(IxDyn(&[n, s]), source_world)?)?;
        let target_world: Vec<f64> = moved.iter().map(|v| v * out).collect();
        let result = if world { target_world } else { self.target.world_to_index(&target_world)? };
        Ok(Array2::from_shape_vec((n, s), result)?)
    }
}

/// The homogeneous `(S+1, S+1)` matrix scaling by `factor`.
fn scale(s: usize, factor: f64) -> Array2<f64> {
    let mut m = Array2::<f64>::eye(s + 1);
    for k in 0..s {
        m[[k, k]] = factor;
    }
    m
}

/// `(N, S)` points through a homogeneous `(S+1, S+1)` matrix.
fn apply(m: &Array2<f64>, points: &Array2<f64>) -> Array2<f64> {
    let s = m.nrows() - 1;
    let mut out = points.dot(&m.slice(ndarray::s![..s, ..s]).t());
    for mut row in out.outer_iter_mut() {
        for (r, v) in row.iter_mut().enumerate() {
            *v += m[[r, s]];
        }
    }
    out
}

/// A resolved path as one matrix, when every step is affine (or identity).
fn path_matrix(transform: &Transform) -> Result<Option<Array2<f64>>> {
    match &transform.body {
        Body::Chain(steps) => {
            let mut total: Option<Array2<f64>> = None;
            for step in steps {
                let Some(m) = path_matrix(step)? else {
                    return Ok(None);
                };
                total = Some(match total {
                    None => m,
                    Some(t) if t.nrows() == m.nrows() => m.dot(&t),
                    Some(_) => return Ok(None),
                });
            }
            Ok(total)
        }
        Body::Inverse(inner) => match path_matrix(inner)? {
            Some(m) => Ok(Some(inv(&m)?)),
            None => Ok(None),
        },
        Body::Stored => match transform.kind() {
            "affine" => Ok(Some(transform.matrix()?)),
            _ => Ok(None),
        },
    }
}

/// Where each source axis goes under a linear part that keeps the axes (a
/// signed permutation with scaling): `(target axis, factor)` per source
/// axis, or `None` when an axis would be mixed with another.
fn axis_map(m: &Array2<f64>) -> Option<Vec<(usize, f64)>> {
    let s = m.nrows() - 1;
    let largest = (0..s).flat_map(|r| (0..s).map(move |k| (r, k))).map(|(r, k)| m[[r, k]].abs()).fold(0.0, f64::max);
    let tolerance = 1e-9 * largest.max(1.0);
    let mut used = vec![false; s];
    let mut out = Vec::with_capacity(s);
    for k in 0..s {
        let rows: Vec<usize> = (0..s).filter(|r| m[[*r, k]].abs() > tolerance).collect();
        let [r] = rows[..] else {
            return None;
        };
        if used[r] {
            return None;
        }
        used[r] = true;
        out.push((r, m[[r, k]]));
    }
    Some(out)
}

/// Where the plane `x[axis] = at` of the source lands under `m`: the target
/// axis it is normal to and its index there, when it lands on an axis plane.
fn plane_map(m: &Array2<f64>, axis: usize, at: f64) -> Result<Option<(usize, f64)>> {
    let s = m.nrows() - 1;
    let back = inv(m)?;
    let largest = (0..s).map(|j| back[[axis, j]].abs()).fold(0.0, f64::max);
    let rows: Vec<usize> = (0..s).filter(|j| back[[axis, *j]].abs() > 1e-9 * largest.max(1.0)).collect();
    let [j] = rows[..] else {
        return Ok(None);
    };
    Ok(Some((j, (at - back[[axis, s]]) / back[[axis, j]])))
}

/// Whether `value` is a whole number, to rounding.
fn whole(value: f64) -> bool {
    (value - value.round()).abs() <= 1e-6
}

fn refused(item: &str, kind: &str, target: &str, why: &str) -> Error {
    Error::invalid(format!(
        "annotation {} ({kind}) cannot be carried onto grid {}: {why}; it is refused rather than approximated, so \
         resample without annotations (--no-annotations, annotations=False) to move the images alone",
        repr_str(item),
        repr_str(target)
    ))
}

const NO_PATH: &str = "no transform path maps its coordinates from the source's frame into the target's";
const NOT_AFFINE: &str =
    "the transform path to the target is not affine, so its straight edges would not stay straight";

/// Choose the annotations to carry, refuse any that cannot be carried
/// exactly, and claim their ids.  `excluded` are the selected images' own
/// valid masks, which their outputs' masks already take in.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(
    sample: &Sample,
    plan: &ResamplePlan,
    target: &Grid,
    mappings: &IndexMap<String, Mapping>,
    sources: &IndexMap<String, Grid>,
    excluded: &HashSet<String>,
    suffix: &str,
    taken: &mut HashSet<String>,
) -> Result<Carry> {
    let annotations = sample.annotations()?;
    let target_frame = target.frame_uid.clone().filter(|f| !f.is_empty());
    let mut candidates: Vec<(&Annotation, String, bool)> = Vec::new();
    for (id, annotation) in annotations {
        if excluded.contains(id) {
            continue;
        }
        let header = &annotation.header;
        match &header.grid {
            Some(grid) if sources.contains_key(grid) => candidates.push((annotation, grid.clone(), true)),
            Some(_) => {}
            None => {
                // A world-space annotation of a source frame the target is
                // not in: it moves with the frame.
                let Some(frame) = header.frame_uid.as_ref().filter(|_| header.space.as_deref() == Some("world")) else {
                    continue;
                };
                if target_frame.as_ref() == Some(frame) {
                    continue;
                }
                if let Some((grid, _)) = sources.iter().find(|(_, g)| g.frame_uid.as_ref() == Some(frame)) {
                    candidates.push((annotation, grid.clone(), false));
                }
            }
        }
    }
    let mut geometry: IndexMap<String, Geometry> = IndexMap::new();
    if !plan.annotations {
        return Ok(Carry {
            items: Vec::new(),
            geometry,
            not_carried: candidates.iter().map(|(a, _, _)| a.ann_id.clone()).collect(),
        });
    }
    let chosen: HashSet<&str> = candidates.iter().map(|(a, _, _)| a.ann_id.as_str()).collect();
    let mut ignore_masks: HashSet<String> = HashSet::new();
    for (annotation, _, _) in &candidates {
        if let Some(mask) = &annotation.header.ignore_mask {
            if !chosen.contains(mask.as_str()) {
                return Err(Error::coded(
                    "E413",
                    format!(
                        "annotation {} names {} as its ignore mask, which is not on its grid, so the ignore region \
                         cannot be carried with it (§7.7)",
                        repr_str(&annotation.ann_id),
                        repr_str(mask)
                    ),
                ));
            }
            ignore_masks.insert(mask.clone());
        }
    }
    let claim = |id: &str, taken: &mut HashSet<String>| -> Result<()> {
        validate_id(id, "annotation id")?;
        if annotations.contains_key(id) || !taken.insert(id.to_string()) {
            return Err(Error::invalid(format!(
                "annotation {} is already in the sample, so the resample cannot write it; resampling twice onto one \
                 grid needs another suffix",
                repr_str(id)
            )));
        }
        Ok(())
    };
    let mut items = Vec::new();
    for (annotation, grid, on_grid) in &candidates {
        let id = &annotation.ann_id;
        let kind = annotation.kind().to_string();
        let output = format!("{id}{suffix}");
        claim(&output, taken)?;
        if !geometry.contains_key(grid) {
            let source = &sources[grid];
            geometry.insert(grid.clone(), Geometry::new(sample, &mappings[grid], source, target)?);
        }
        check(annotation, &kind, &geometry[grid], *on_grid, target)?;
        let voxel = *on_grid && is_voxel(&kind) && (kind != "instances" || annotation.has_masks()?);
        let ignore = if voxel && CLASS_VOXEL_KINDS.contains(&kind.as_str()) {
            match &annotation.header.ignore_mask {
                Some(mask) => Ignore::Carried(format!("{mask}{suffix}")),
                None => {
                    let new = format!("{output}_ignore");
                    claim(&new, taken)?;
                    Ignore::New(new)
                }
            }
        } else {
            Ignore::None
        };
        items.push(Item {
            source: id.clone(),
            output,
            kind,
            grid: grid.clone(),
            on_grid: *on_grid,
            voxel,
            ignore,
            widen: ignore_masks.contains(id),
        });
    }
    Ok(Carry { items, geometry, not_carried: Vec::new() })
}

/// Refuse an annotation its kind cannot carry exactly onto the target.
fn check(annotation: &Annotation, kind: &str, geometry: &Geometry, on_grid: bool, target: &Grid) -> Result<()> {
    let id = &annotation.ann_id;
    let world = annotation.header.space.as_deref() == Some("world");
    let fail = |why: &str| Err(refused(id, kind, &target.grid_id, why));
    match kind {
        "labelmap" | "layers" | "bitmask" | "mask" | "probmap" => Ok(()),
        "instances" if annotation.has_masks()? => Ok(()),
        "instances" | "boxes" => {
            let Some(m) = geometry.linear(world) else {
                return fail(if geometry.can_move() { NOT_AFFINE } else { NO_PATH });
            };
            let Some(axes) = axis_map(m) else {
                return fail("the mapping onto it turns the box axes, and a rotated box is no axis-aligned box");
            };
            if kind == "boxes" && on_grid {
                check_slices(annotation, geometry, &axes, &|why| refused(id, kind, &target.grid_id, why))?;
            }
            Ok(())
        }
        "obb" => {
            let Some(m) = geometry.linear(world) else {
                return fail(if geometry.can_move() { NOT_AFFINE } else { NO_PATH });
            };
            let rotations = annotation.read_as::<f64>("rotations")?;
            let s = m.nrows() - 1;
            let linear = m.slice(ndarray::s![..s, ..s]).to_owned();
            for (i, rotation) in rotations.outer_iter().enumerate() {
                let r = rotation.into_dimensionality::<ndarray::Ix2>()?;
                if !orthogonal(&linear.dot(&r)) {
                    return fail(&format!("the mapping onto it shears oriented box {i}, which would not stay a box"));
                }
            }
            Ok(())
        }
        "points" | "keypoints" => {
            if geometry.can_move() {
                Ok(())
            } else {
                fail(NO_PATH)
            }
        }
        "mesh" => {
            if !geometry.can_move() {
                return fail(NO_PATH);
            }
            if geometry.linear(world).is_none() && annotation.optional_dataset("normals")?.is_some() {
                return fail("the transform path to the target is not affine, so its normals cannot be carried");
            }
            Ok(())
        }
        "contours" | "polylines" => {
            let Some(m) = geometry.linear(world) else {
                return fail(if geometry.can_move() { NOT_AFFINE } else { NO_PATH });
            };
            let _ = m;
            let column = if kind == "contours" { "contour_plane" } else { "polyline_plane" };
            if !on_grid {
                return Ok(());
            }
            let Some(planes) = annotation.read_optional::<i64>(column)? else {
                return Ok(());
            };
            let Some(index) = geometry.index.as_ref() else {
                return fail(NOT_AFFINE);
            };
            for plane in planes.outer_iter() {
                let (axis, at) = (plane[0], plane[1]);
                if axis < 0 {
                    continue;
                }
                match plane_map(index, axis as usize, at as f64)? {
                    Some((_, landed)) if whole(landed) => {}
                    Some((j, landed)) => {
                        return fail(&format!(
                            "plane {at} of axis {axis} lands at {} on axis {j}, between the target's planes",
                            float_repr(landed)
                        ))
                    }
                    None => {
                        return fail(&format!("plane {at} of axis {axis} lands obliquely, on no plane of the target"))
                    }
                }
            }
            Ok(())
        }
        "classification" => match attrs::get_str(&annotation.group, "scope")?.as_deref() {
            Some(scope @ ("slice" | "roi")) => fail(&format!(
                "its scope is a {scope} of grid {}, which the target does not have",
                repr_str(&geometry.source.grid_id)
            )),
            _ => Ok(()),
        },
        other => fail(&format!("this resample does not know how to carry a {} annotation", repr_str(other))),
    }
}

/// Refuse a 2-D box whose slice does not land on a slice of the target.
fn check_slices(
    annotation: &Annotation,
    geometry: &Geometry,
    axes: &[(usize, f64)],
    refuse: &dyn Fn(&str) -> Error,
) -> Result<()> {
    let Some(slices) = annotation.read_optional::<i64>("slice_index")? else {
        return Ok(());
    };
    let boxes = annotation.read_as::<f64>("boxes")?;
    let index = geometry.index.as_ref().expect("checked by the caller");
    let s = index.nrows() - 1;
    for (i, k) in slices.iter().enumerate() {
        let Some(axis) = degenerate_axis(&boxes, i) else {
            continue;
        };
        let (r, factor) = axes[axis];
        let landed = factor * *k as f64 + index[[r, s]];
        if !whole(landed) {
            return Err(refuse(&format!(
                "box {i} is drawn on slice {k} of axis {axis}, which lands at {} on axis {r}, between the target's \
                 slices",
                float_repr(landed)
            )));
        }
    }
    Ok(())
}

/// The axis along which box `i` has no extent (`lo == hi`), if one.
fn degenerate_axis(boxes: &ArrayD<f64>, i: usize) -> Option<usize> {
    let s = boxes.shape()[1];
    (0..s).find(|k| boxes[[i, *k, 0]] == boxes[[i, *k, 1]])
}

/// Whether a matrix's columns are mutually orthogonal.
fn orthogonal(m: &Array2<f64>) -> bool {
    let s = m.ncols();
    for a in 0..s {
        for b in (a + 1)..s {
            let (u, v) = (m.column(a), m.column(b));
            let tolerance = 1e-6 * u.dot(&u).sqrt() * v.dot(&v).sqrt();
            if u.dot(&v).abs() > tolerance {
                return false;
            }
        }
    }
    true
}

/// The target voxels' nearest source voxels, and which lie outside it.
struct Nearest {
    /// Per target voxel in C order, the flat index of its nearest source
    /// voxel; `u32::MAX` outside the source.
    index: Vec<u32>,
    /// The target's spatial shape.
    shape: Vec<usize>,
    outside: Option<ArrayD<bool>>,
}

/// The block shape over `shape` that holds at most [`BLOCK_VOXELS`].
fn block_shape(shape: &[usize]) -> Vec<usize> {
    let mut block = shape.to_vec();
    while block.iter().product::<usize>() > BLOCK_VOXELS {
        let (axis, _) = block.iter().enumerate().max_by_key(|(_, n)| **n).expect("a spatial axis");
        block[axis] = block[axis].div_ceil(2);
    }
    block
}

/// The `(n, S)` target indices of a block, in C order.
fn block_indices(block: &[(usize, usize)]) -> (Array2<f64>, Vec<usize>) {
    let dims: Vec<usize> = block.iter().map(|(a, b)| b - a).collect();
    let s = dims.len();
    let n: usize = dims.iter().product();
    let mut indices = Array2::<f64>::zeros((n, s));
    for (row, mut point) in indices.outer_iter_mut().enumerate() {
        let mut rest = row;
        for k in (0..s).rev() {
            point[k] = (block[k].0 + rest % dims[k]) as f64;
            rest /= dims[k];
        }
    }
    (indices, dims)
}

/// The flat C-order index of `at` in `shape`.
fn flat_index(at: impl Iterator<Item = usize>, shape: &[usize]) -> usize {
    at.zip(shape).fold(0, |flat, (i, n)| flat * n + i)
}

fn inside(point: ndarray::ArrayView1<f64>, shape: &[usize]) -> bool {
    point.iter().zip(shape).all(|(v, m)| *v >= -0.5 && *v <= *m as f64 - 0.5)
}

fn nearest(mapping: &Mapping, source: &[usize], target: &[usize]) -> Result<Nearest> {
    let total: usize = source.iter().product();
    if total >= u32::MAX as usize {
        return Err(Error::invalid(format!(
            "a source grid of {total} voxels is more than a resample carries annotations from; resample without \
             annotations (--no-annotations, annotations=False)"
        )));
    }
    let voxels: usize = target.iter().product();
    let mut index = vec![u32::MAX; voxels];
    let mut outside: Option<ArrayD<bool>> = None;
    for block in blocks(target, &block_shape(target)) {
        let (indices, _) = block_indices(&block);
        let points = mapping.map(&indices)?;
        for (row, point) in points.outer_iter().enumerate() {
            let at = indices.row(row);
            let flat = flat_index(at.iter().map(|v| *v as usize), target);
            if inside(point, source) {
                let voxel = point.iter().zip(source).map(|(v, m)| ((v + 0.5).floor().max(0.0) as usize).min(m - 1));
                index[flat] = flat_index(voxel, source) as u32;
            } else {
                outside
                    .get_or_insert_with(|| ArrayD::from_elem(IxDyn(target), false))
                    .as_slice_mut()
                    .expect("standard")[flat] = true;
            }
        }
    }
    Ok(Nearest { index, shape: target.to_vec(), outside })
}

/// Write every carried annotation; what was written is added to `written`.
pub(super) fn write(
    sample: &Sample,
    writer: &mut SampleWriter,
    carry: &Carry,
    mappings: &IndexMap<String, Mapping>,
    target: &Grid,
    activity: &str,
    written: &mut Vec<String>,
) -> Result<Vec<ResampledAnnotation>> {
    let annotations = sample.annotations()?;
    let target_spatial = target.spatial_shape();
    let mut nearest_by_grid: HashMap<String, Nearest> = HashMap::new();
    for item in &carry.items {
        if item.voxel && !nearest_by_grid.contains_key(&item.grid) {
            let source = carry.geometry[&item.grid].source.spatial_shape();
            nearest_by_grid.insert(item.grid.clone(), nearest(&mappings[&item.grid], &source, &target_spatial)?);
        }
    }
    let mut results: Vec<(usize, ResampledAnnotation)> = Vec::new();
    // Masks first, so an ignore mask exists before what names it.
    let order: Vec<usize> = (0..carry.items.len())
        .filter(|i| carry.items[*i].kind == "mask")
        .chain((0..carry.items.len()).filter(|i| carry.items[*i].kind != "mask"))
        .collect();
    for position in order {
        let item = &carry.items[position];
        let annotation = &annotations[&item.source];
        let geometry = &carry.geometry[&item.grid];
        let mut header = carried_header(annotation, item, geometry, target, activity);
        let mut note = None;
        let mut overrides: Vec<(String, AttrValue)> = Vec::new();
        let payload = if item.voxel {
            let near = &nearest_by_grid[&item.grid];
            match item.kind.as_str() {
                "probmap" => {
                    let (payload, filled) = probmap_payload(annotation, &mappings[&item.grid], geometry, target)?;
                    if filled > 0 && annotation.normalized()? {
                        note = Some(format!(
                            "{filled} target voxel(s) lie outside the source and hold no probability, so the map is \
                             written with normalized false"
                        ));
                    }
                    overrides.extend(payload.attrs.iter().cloned());
                    payload
                }
                "instances" => instances_payload(annotation, near, &geometry.source.spatial_shape())?,
                _ => dense_payload(annotation, near, item.widen)?,
            }
        } else {
            geometric_payload(annotation, &item.kind, geometry, item.on_grid)?
        };
        if item.voxel && CLASS_VOXEL_KINDS.contains(&item.kind.as_str()) {
            let near = &nearest_by_grid[&item.grid];
            header.ignore_mask = match &item.ignore {
                Ignore::Carried(mask) => Some(mask.clone()),
                Ignore::New(mask) => match &near.outside {
                    Some(outside) => {
                        writer.add_mask(mask, outside.clone(), &target.grid_id, "other", Some(activity), None)?;
                        written.push(format!("annotations/{mask}"));
                        Some(mask.clone())
                    }
                    None => None,
                },
                Ignore::None => None,
            };
        }
        write_carried(writer, annotation, &item.output, &header, &payload, target, item.on_grid, &overrides)?;
        written.push(format!("annotations/{}", item.output));
        results.push((
            position,
            ResampledAnnotation {
                source: item.source.clone(),
                annotation_id: item.output.clone(),
                kind: item.kind.clone(),
                ignore_mask: header.ignore_mask.clone(),
                note,
            },
        ));
    }
    results.sort_by_key(|(position, _)| *position);
    Ok(results.into_iter().map(|(_, r)| r).collect())
}

fn is_voxel(kind: &str) -> bool {
    DENSE_KINDS.contains(&kind) || kind == "probmap" || kind == "instances"
}

/// The source's header as the carried copy states it.
fn carried_header(
    annotation: &Annotation,
    item: &Item,
    geometry: &Geometry,
    target: &Grid,
    activity: &str,
) -> AnnotationHeader {
    let mut header = annotation.header.clone();
    if item.on_grid {
        header.grid = Some(target.grid_id.clone());
        let source_tp = geometry.source.timepoint.clone();
        if header.timepoints.is_none() && source_tp.is_some() && source_tp != target.timepoint {
            // A copy on another visit's grid is still of its own visit.
            header.timepoints = source_tp.map(|tp| vec![tp]);
        }
    }
    if header.space.as_deref() == Some("world") && geometry.forward.is_some() {
        header.frame_uid = target.frame_uid.clone();
    }
    header.prov = Some(activity.to_string());
    // A review was of the source, not of the copy.
    header.quality = None;
    header.derived_from = vec![annotation.ann_id.clone()];
    header.ignore_mask = None;
    header.extra = Vec::new();
    header
}

/// Write the carried annotation, then carry every attribute of the source
/// the header does not write as stored, unless the payload sets it.
#[allow(clippy::too_many_arguments)]
fn write_carried(
    writer: &mut SampleWriter,
    annotation: &Annotation,
    output: &str,
    header: &AnnotationHeader,
    payload: &Payload,
    target: &Grid,
    on_grid: bool,
    overrides: &[(String, AttrValue)],
) -> Result<()> {
    let group = writer.write_annotation(output, header, payload, on_grid.then_some(target), None)?;
    for name in attrs::names(&annotation.group)? {
        if HEADER_ATTRS.contains(&name.as_str()) || overrides.iter().any(|(k, _)| *k == name) {
            continue;
        }
        attrs::copy_raw(&annotation.group, &group, &name)?;
    }
    for (name, value) in overrides {
        attrs::write(&group, name, value)?;
    }
    Ok(())
}

/// Every dataset of the source, as stored, for a payload to change.
fn copy_payload(annotation: &Annotation, skip: &[&str]) -> Result<Payload> {
    let mut payload = Payload::new(annotation.kind());
    for name in ops::members(&annotation.group)? {
        if skip.contains(&name.as_str()) {
            continue;
        }
        let Some(dataset) = ops::child_dataset(&annotation.group, &name) else {
            continue;
        };
        let value = if data::is_strings(&dataset) {
            PayloadData::Strings(data::read_strings(&dataset)?)
        } else {
            PayloadData::Array(data::read(&dataset)?)
        };
        payload.datasets.insert(name, value);
    }
    payload.class_ids = annotation.class_ids().to_vec();
    Ok(payload)
}

/// A dense voxel encoding gathered nearest, in its own encoding: every
/// stored plane moves as it is.  Outside the source it holds 0, or for a
/// widened ignore mask `true`.
fn dense_payload(annotation: &Annotation, near: &Nearest, widen: bool) -> Result<Payload> {
    let stored = data::read(&annotation.data()?)?;
    let shape = stored.shape();
    let s = annotation.spatial_shape()?.len();
    let leading = shape[..shape.len() - s].to_vec();
    let lanes: usize = leading.iter().product::<usize>().max(1);
    let voxels = near.index.len();
    let mut full = leading.clone();
    full.extend(&near.shape);
    let gathered = crate::with_array!(&stored, a => {
        let flat: Vec<_> = a.iter().copied().collect();
        let per_lane = flat.len() / lanes;
        let fill = Element::from_f64(if widen { 1.0 } else { 0.0 });
        let mut out = Vec::with_capacity(lanes * voxels);
        for lane in 0..lanes {
            let base = lane * per_lane;
            out.extend(near.index.iter().map(|i| if *i == u32::MAX { fill } else { flat[base + *i as usize] }));
        }
        NdArray::from_vec(&full, out)?
    });
    let mut payload = copy_payload(annotation, &["data"])?;
    payload.datasets.insert("data".into(), gathered.into());
    payload.stacked_axes = leading.len();
    Ok(payload)
}

/// A probability map sampled linearly, renormalised where the source was
/// normalised; and the target voxels outside the source, which hold 0 in
/// every class.
fn probmap_payload(
    annotation: &Annotation,
    mapping: &Mapping,
    geometry: &Geometry,
    target: &Grid,
) -> Result<(Payload, u64)> {
    let source = geometry.source.spatial_shape();
    let target_spatial = target.spatial_shape();
    let field = annotation.probabilities(None, None)?.mapv(f64::from);
    let classes = annotation.class_ids().to_vec();
    let normalized = annotation.normalized()?;
    let voxels: usize = target_spatial.iter().product();
    let mut planes = vec![vec![0.0f64; voxels]; classes.len()];
    let mut filled = 0u64;
    for block in blocks(&target_spatial, &block_shape(&target_spatial)) {
        let (indices, _) = block_indices(&block);
        let points = mapping.map(&indices)?;
        let rows: Vec<usize> = (0..points.nrows()).filter(|r| inside(points.row(*r), &source)).collect();
        filled += (points.nrows() - rows.len()) as u64;
        if rows.is_empty() {
            continue;
        }
        let coords = Array2::from_shape_fn((rows.len(), source.len()), |(i, k)| {
            points[[rows[i], k]].clamp(0.0, source[k] as f64 - 1.0)
        });
        let values = linear_sample(&field, &coords, "nearest")?;
        for (i, row) in rows.iter().enumerate() {
            let flat = flat_index(indices.row(*row).iter().map(|v| *v as usize), &target_spatial);
            let sum: f64 = values.row(i).sum();
            for (k, plane) in planes.iter_mut().enumerate() {
                let v = values[[i, k]];
                plane[flat] = if normalized && sum > 0.0 { v / sum } else { v }.clamp(0.0, 1.0);
            }
        }
    }
    let mut map: BTreeMap<i64, ArrayD<f64>> = BTreeMap::new();
    for (id, plane) in classes.iter().zip(planes) {
        map.insert(*id, ArrayD::from_shape_vec(IxDyn(&target_spatial), plane)?);
    }
    let threshold = if attrs::has(&annotation.group, "threshold") { Some(annotation.threshold()?) } else { None };
    let dtype = data::dtype(&annotation.data()?)?;
    let payload = encode_probmap(&map, Some(&target_spatial), dtype, normalized && filled == 0, threshold)?;
    Ok((payload, filled))
}

/// The C-order coordinates of a flat index.
fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut at = vec![0usize; shape.len()];
    for k in (0..shape.len()).rev() {
        at[k] = flat % shape[k];
        flat /= shape[k];
    }
    at
}

/// Objects moved one by one under their own ids: a target voxel belongs to
/// every object holding its nearest source voxel, so two objects of one
/// class stay two, and overlapping objects keep their overlap.  Each box is
/// recomputed from its mask.  An object left with no voxel is refused.
fn instances_payload(annotation: &Annotation, near: &Nearest, source: &[usize]) -> Result<Payload> {
    let n = annotation.n_objects()?;
    let boxes = annotation.boxes()?;
    let classes = annotation.object_class_ids()?;
    let ids = annotation.instance_ids()?;
    let scores = annotation.scores()?;
    let s = source.len();
    // Each source voxel's objects, as an interned set: most voxels hold one
    // object or none, so the sets are few.
    let mut label = vec![0u32; source.iter().product()];
    let mut sets: Vec<Vec<u32>> = vec![Vec::new()];
    let mut interned: HashMap<Vec<u32>, u32> = HashMap::new();
    for i in 0..n {
        let Some(crop) = annotation.crop(i)? else {
            continue;
        };
        let flat_box: Vec<f64> = boxes.slice(ndarray::s![i, .., ..]).iter().map(|v| f64::from(*v)).collect();
        let origin: Vec<i64> = box_to_slices(&flat_box, None)?.iter().map(|(lo, _)| *lo).collect();
        for (at, on) in crop.indexed_iter() {
            if !*on {
                continue;
            }
            let voxel: Vec<i64> = at.slice().iter().zip(&origin).map(|(j, o)| *j as i64 + o).collect();
            if voxel.iter().zip(source).any(|(v, m)| *v < 0 || *v >= *m as i64) {
                continue;
            }
            let f = flat_index(voxel.iter().map(|v| *v as usize), source);
            let mut set = sets[label[f] as usize].clone();
            set.push(i as u32);
            let next = sets.len() as u32;
            let id = match interned.get(&set) {
                Some(id) => *id,
                None => {
                    interned.insert(set.clone(), next);
                    sets.push(set);
                    next
                }
            };
            label[f] = id;
        }
    }
    let mut voxels: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (t, at) in near.index.iter().enumerate() {
        if *at == u32::MAX {
            continue;
        }
        for object in &sets[label[*at as usize] as usize] {
            voxels[*object as usize].push(t);
        }
    }
    let mut objects = Vec::with_capacity(n);
    for (i, held) in voxels.iter().enumerate() {
        let id = ids.as_ref().map(|ids| ids[i]);
        if held.is_empty() {
            return Err(Error::invalid(format!(
                "object {} of annotation {} has no voxel on the target grid: each of its voxels falls between the \
                 target's voxel centres, and an object is refused, never dropped (§7.4); resample onto a finer grid, \
                 or without annotations (--no-annotations, annotations=False)",
                id.map_or_else(|| format!("at row {i}"), |v| v.to_string()),
                repr_str(&annotation.ann_id)
            )));
        }
        let coords: Vec<Vec<usize>> = held.iter().map(|f| unravel(*f, &near.shape)).collect();
        let lo: Vec<usize> = (0..s).map(|k| coords.iter().map(|c| c[k]).min().unwrap_or(0)).collect();
        let hi: Vec<usize> = (0..s).map(|k| coords.iter().map(|c| c[k] + 1).max().unwrap_or(0)).collect();
        let extent: Vec<usize> = lo.iter().zip(&hi).map(|(a, b)| b - a).collect();
        let mut crop = ArrayD::from_elem(IxDyn(&extent), false);
        for c in &coords {
            let local: Vec<usize> = c.iter().zip(&lo).map(|(v, o)| v - o).collect();
            crop[IxDyn(&local)] = true;
        }
        let pairs: Vec<(i64, i64)> = lo.iter().zip(&hi).map(|(a, b)| (*a as i64, *b as i64)).collect();
        objects.push(InstanceInput {
            class_id: i64::from(classes[i]),
            instance_id: id.unwrap_or(i as u64),
            mask: None,
            bbox: Some(slices_to_box(&pairs).iter().map(|v| f64::from(*v)).collect()),
            crop: Some(crop),
            score: scores.as_ref().and_then(|s| s.get(i)).filter(|v| v.is_finite()).map(|v| f64::from(*v)),
        });
    }
    let mut payload = encode_instances(&objects, Some(&near.shape), true, Some(annotation.class_ids()))?;
    if ids.is_none() {
        // The rows were numbered to encode them; they name no object.
        payload.datasets.shift_remove("instance_ids");
    }
    Ok(payload)
}

/// A payload dataset as `f64`, with its stored dtype and shape.
fn column(payload: &Payload, name: &str) -> Result<(ArrayD<f64>, DType)> {
    let array = payload.array(name)?;
    Ok((array.to_f64(), array.dtype()))
}

fn put(payload: &mut Payload, name: &str, values: ArrayD<f64>, dtype: DType) {
    payload.datasets.insert(name.into(), NdArray::F64(values).astype(dtype).into());
}

/// Map a coordinate dataset whose last axis is the point's.
fn move_points(payload: &mut Payload, name: &str, geometry: &Geometry, world: bool) -> Result<()> {
    if !payload.datasets.contains_key(name) {
        return Ok(());
    }
    let (values, dtype) = column(payload, name)?;
    let shape = values.shape().to_vec();
    let s = *shape.last().unwrap_or(&0);
    let n = values.len().checked_div(s).unwrap_or(0);
    let flat = Array2::from_shape_vec((n, s), values.iter().copied().collect())?;
    let moved = if n == 0 { flat } else { geometry.points(world, &flat)? };
    put(payload, name, moved.into_dyn().into_shape_with_order(IxDyn(&shape))?, dtype);
    Ok(())
}

/// A geometric or classification annotation's datasets, its coordinates
/// mapped onto the target.
fn geometric_payload(annotation: &Annotation, kind: &str, geometry: &Geometry, on_grid: bool) -> Result<Payload> {
    let world = annotation.header.space.as_deref() == Some("world");
    let mut payload = copy_payload(annotation, &[])?;
    match kind {
        "boxes" | "instances" => {
            let m = geometry.linear(world).expect("checked in prepare");
            let axes = axis_map(m).expect("checked in prepare");
            let s = m.nrows() - 1;
            let (values, dtype) = column(&payload, "boxes")?;
            let mut moved = ArrayD::<f64>::zeros(values.raw_dim());
            for i in 0..values.shape()[0] {
                for (k, (r, factor)) in axes.iter().enumerate() {
                    let a = factor * values[[i, k, 0]] + m[[*r, s]];
                    let b = factor * values[[i, k, 1]] + m[[*r, s]];
                    moved[[i, *r, 0]] = a.min(b);
                    moved[[i, *r, 1]] = a.max(b);
                }
            }
            if let (true, Some(index), true) = (on_grid, &geometry.index, payload.datasets.contains_key("slice_index"))
            {
                let index_axes = axis_map(index).expect("checked in prepare");
                let (slices, slice_dtype) = column(&payload, "slice_index")?;
                let mut landed = slices.clone();
                for (i, k) in slices.iter().enumerate() {
                    if let Some(axis) = degenerate_axis(&values, i) {
                        let (r, factor) = index_axes[axis];
                        landed[[i]] = (factor * k + index[[r, s]]).round();
                    }
                }
                put(&mut payload, "slice_index", landed, slice_dtype);
            }
            put(&mut payload, "boxes", moved, dtype);
        }
        "obb" => {
            move_points(&mut payload, "centers", geometry, world)?;
            let m = geometry.linear(world).expect("checked in prepare");
            let s = m.nrows() - 1;
            let linear = m.slice(ndarray::s![..s, ..s]).to_owned();
            let (sizes, size_dtype) = column(&payload, "sizes")?;
            let (rotations, rotation_dtype) = column(&payload, "rotations")?;
            let mut new_sizes = sizes.clone();
            let mut new_rotations = rotations.clone();
            for i in 0..rotations.shape()[0] {
                let r = rotations.slice(ndarray::s![i, .., ..]).to_owned();
                let mut moved = linear.dot(&r);
                for j in 0..s {
                    let norm = moved.column(j).dot(&moved.column(j)).sqrt();
                    if norm > 0.0 {
                        moved.column_mut(j).mapv_inplace(|v| v / norm);
                    }
                    new_sizes[[i, j]] = sizes[[i, j]] * norm;
                }
                if s > 0 && det(&moved) < 0.0 {
                    // A box is symmetric: turning one axis keeps it, and
                    // keeps the rotation proper.
                    moved.column_mut(s - 1).mapv_inplace(|v| -v);
                }
                new_rotations.slice_mut(ndarray::s![i, .., ..]).assign(&moved);
            }
            put(&mut payload, "sizes", new_sizes, size_dtype);
            put(&mut payload, "rotations", new_rotations, rotation_dtype);
        }
        "points" | "keypoints" => move_points(&mut payload, "points", geometry, world)?,
        "contours" | "polylines" => {
            move_points(&mut payload, "vertices", geometry, world)?;
            let name = if kind == "contours" { "contour_plane" } else { "polyline_plane" };
            if let (true, Some(index), true) = (on_grid, &geometry.index, payload.datasets.contains_key(name)) {
                let (planes, dtype) = column(&payload, name)?;
                let mut landed = planes.clone();
                for (i, plane) in planes.outer_iter().enumerate() {
                    if plane[0] < 0.0 {
                        continue;
                    }
                    let (j, at) = plane_map(index, plane[0] as usize, plane[1])?.expect("checked in prepare");
                    landed[[i, 0]] = j as f64;
                    landed[[i, 1]] = at.round();
                }
                put(&mut payload, name, landed, dtype);
            }
        }
        "mesh" => {
            move_points(&mut payload, "vertices", geometry, world)?;
            if let Some(m) = geometry.linear(world) {
                let s = m.nrows() - 1;
                let linear = m.slice(ndarray::s![..s, ..s]).to_owned();
                if payload.datasets.contains_key("normals") {
                    // Normals move by the inverse transpose, renormalised.
                    let normal_map = inv(&linear)?.reversed_axes();
                    let (normals, dtype) = column(&payload, "normals")?;
                    let flat = normals.into_dimensionality::<ndarray::Ix2>()?;
                    let mut moved = flat.dot(&normal_map.t());
                    for mut row in moved.outer_iter_mut() {
                        let norm = row.dot(&row).sqrt();
                        if norm > 0.0 {
                            row.mapv_inplace(|v| v / norm);
                        }
                    }
                    put(&mut payload, "normals", moved.into_dyn(), dtype);
                }
                if det(&linear) < 0.0 && payload.datasets.contains_key("faces") {
                    // A reflection turns every face inside out; rewinding
                    // keeps the outward side outward.
                    let (faces, dtype) = column(&payload, "faces")?;
                    let mut rewound = faces.clone();
                    for i in 0..faces.shape()[0] {
                        rewound[[i, 1]] = faces[[i, 2]];
                        rewound[[i, 2]] = faces[[i, 1]];
                    }
                    put(&mut payload, "faces", rewound, dtype);
                }
            }
        }
        _ => {}
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn homogeneous(linear: [[f64; 3]; 3], offset: [f64; 3]) -> Array2<f64> {
        let mut m = Array2::<f64>::eye(4);
        for r in 0..3 {
            for k in 0..3 {
                m[[r, k]] = linear[r][k];
            }
            m[[r, 3]] = offset[r];
        }
        m
    }

    #[test]
    fn s8_2_a_signed_permutation_keeps_a_box_a_box() {
        let m = homogeneous([[0.0, -2.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 3.0]], [1.0, 2.0, 3.0]);
        assert_eq!(axis_map(&m), Some(vec![(1, 0.5), (0, -2.0), (2, 3.0)]));
        let (c, s) = (0.1f64.cos(), 0.1f64.sin());
        let rotated = homogeneous([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], [0.0; 3]);
        assert_eq!(axis_map(&rotated), None);
    }

    #[test]
    fn s8_6_a_plane_lands_on_a_plane_only_at_a_whole_index() {
        // Index j = (i - 0.5) / 2 along every axis: plane 2.5 is plane 1.
        let m = homogeneous([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]], [-0.25; 3]);
        let (axis, at) = plane_map(&m, 0, 2.5).unwrap().unwrap();
        assert_eq!(axis, 0);
        assert!(whole(at) && at.round() == 1.0);
        let (_, at) = plane_map(&m, 0, 3.0).unwrap().unwrap();
        assert!(!whole(at));
        let swap = homogeneous([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [0.0; 3]);
        assert_eq!(plane_map(&swap, 0, 4.0).unwrap(), Some((1, 4.0)));
    }

    #[test]
    fn s8_3_a_shear_is_refused_for_an_oriented_box() {
        let shear = ndarray::arr2(&[[1.0, 0.5], [0.0, 1.0]]);
        assert!(!orthogonal(&shear));
        let (c, s) = (0.3f64.cos(), 0.3f64.sin());
        let rotation_and_scale = ndarray::arr2(&[[2.0 * c, -3.0 * s], [2.0 * s, 3.0 * c]]);
        assert!(orthogonal(&rotation_and_scale));
    }

    #[test]
    fn a_block_holds_at_most_its_budget() {
        let block = block_shape(&[300, 400, 500]);
        assert!(block.iter().product::<usize>() <= BLOCK_VOXELS);
        assert_eq!(block_shape(&[4, 4]), vec![4, 4]);
    }
}
