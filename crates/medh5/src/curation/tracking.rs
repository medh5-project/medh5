//! Longitudinal joins on `instance_id` (§7.4, §11.3).
//!
//! Tracking is a **join, not a structure**: grouping objects on
//! `instance_id` recovers the lesion a radiologist followed across visits.
//! Absence is not a measurement --- it is `resolved` only where the object
//! was looked for, and `unexamined` otherwise.  [`locate_tracks`] gives every
//! unseen (object, visit) an [`Absence`] saying why: it maps the object's
//! boxes from the visits that saw it into each annotation of the visit that
//! did not, and an object outside that grid, outside its images' acquired
//! voxels or inside its ignore region (§7.7) was not looked for there.  A
//! track carrying several class ids is reported (W909), never resolved by a
//! rule.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use indexmap::IndexMap;
use ndarray::Array2;
use serde_json::{json, Map, Value};

use crate::annotations::{Annotation, Instance};
use crate::array::Slice;
use crate::geometry::affine::{box_corners, box_to_slices, voxel_volume};
use crate::geometry::grid::{mm_per_unit, Grid};
use crate::json::{format_g, num, repr_str};
use crate::labels::ClassKey;
use crate::sample::Sample;
use crate::transforms::{stored_inverse, Body, Transform};
use crate::{Error, Result};

pub const PRESENT: &str = "present";
pub const RESOLVED: &str = "resolved";
pub const UNEXAMINED: &str = "unexamined";
pub const STATES: [&str; 3] = [PRESENT, RESOLVED, UNEXAMINED];

/// One object seen once: a row of one annotation.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub timepoint: String,
    pub annotation: String,
    /// The annotation's kind: `instances` (a mask), `boxes`, `obb`, or
    /// `points` or `polylines` (presence only).
    pub kind: String,
    pub index: usize,
    pub instance_id: u64,
    pub class_id: i64,
    pub bbox: Array2<f32>,
    pub voxel_count: Option<u64>,
    /// Physical volume in `units**S`, `S` the box's axes; `None` for a
    /// point, which marks an object without measuring it.
    pub volume: Option<f64>,
    /// The grid's `units`, which `volume` is in.
    pub units: Option<String>,
    pub score: Option<f64>,
    pub grid: Option<String>,
}

impl Observation {
    /// Box centre, in whatever space the annotation stores its boxes.
    pub fn centroid(&self) -> Vec<f64> {
        self.bbox.outer_iter().map(|r| (f64::from(r[0]) + f64::from(r[1])) / 2.0).collect()
    }

    pub fn extent(&self) -> Vec<f64> {
        self.bbox.outer_iter().map(|r| f64::from(r[1]) - f64::from(r[0])).collect()
    }

    /// `volume` in millimetres to the power of the box's axes, or `None`
    /// where it is unmeasured or its grid is not calibrated in a length.
    pub fn volume_mm(&self) -> Option<f64> {
        let scale = mm_per_unit(self.units.as_deref()?)?;
        Some(self.volume? * scale.powi(i32::try_from(self.bbox.nrows()).ok()?))
    }

    pub fn to_json(&self) -> Value {
        let rows: Vec<Value> =
            self.bbox.outer_iter().map(|r| json!(r.iter().map(|v| num(f64::from(*v))).collect::<Vec<_>>())).collect();
        json!({
            "timepoint": self.timepoint,
            "annotation": self.annotation,
            "kind": self.kind,
            "index": self.index,
            "class_id": self.class_id,
            "box": rows,
            "voxel_count": self.voxel_count,
            "volume": self.volume.map(num),
            "units": self.units,
            "volume_mm": self.volume_mm().map(num),
            "score": self.score.map(num),
        })
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        let volume = self.volume.map(|v| format_g(v, 4)).unwrap_or_else(|| "-".into());
        format!(
            "Observation({}[{}] @{}, class={}, volume={volume})",
            self.annotation, self.index, self.timepoint, self.class_id
        )
    }
}

/// Every observation of one physical object, ordered by timepoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub instance_id: u64,
    pub class_ids: Vec<i64>,
    pub observations: Vec<Observation>,
    pub class_key: Option<String>,
}

impl Track {
    /// The object's class.  See [`Track::has_class_conflict`] first.
    pub fn class_id(&self) -> i64 {
        self.class_ids[0]
    }

    /// Whether this id carries more than one class id (W909).
    pub fn has_class_conflict(&self) -> bool {
        self.class_ids.len() > 1
    }

    pub fn timepoints(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for o in &self.observations {
            if !out.contains(&o.timepoint) {
                out.push(o.timepoint.clone());
            }
        }
        out
    }

    /// Every observation at `timepoint`: one per annotation that saw the
    /// object there --- several where raters annotated the same visit, or a
    /// detection sits beside a segmentation.
    pub fn observations_at(&self, timepoint: &str) -> Vec<&Observation> {
        self.observations.iter().filter(|o| o.timepoint == timepoint).collect()
    }

    /// The observations that measure the object at `timepoint`: its masks
    /// (`instances`), or --- where no mask saw it --- its boxes, axis-aligned
    /// or oriented.  A box beside a mask of the same object is a detection of
    /// it, and its volume is its bounding box's.  A point or a curve marks the
    /// object and measures nothing, so it answers only where nothing else saw
    /// it.
    pub fn measurements_at(&self, timepoint: &str) -> Vec<&Observation> {
        let seen = self.observations_at(timepoint);
        for kinds in [&["instances"][..], &["boxes", "obb"]] {
            let found: Vec<&Observation> = seen.iter().copied().filter(|o| kinds.contains(&o.kind.as_str())).collect();
            if !found.is_empty() {
                return found;
            }
        }
        seen
    }

    /// The observation that answers for `timepoint` --- the one
    /// [`Track::measurements_at`] finds --- or `None` where the object was not
    /// seen.  Refused where it finds two, two raters' masks of one visit: the
    /// first rater's answered `at` and `volume` and the last one's `volumes`,
    /// so one track reported two volumes for one visit.
    /// Every accessor below answers by this rule.
    pub fn at(&self, timepoint: &str) -> Result<Option<&Observation>> {
        let found = self.measurements_at(timepoint);
        if found.len() > 1 {
            let names: Vec<String> = found.iter().map(|o| o.annotation.clone()).collect();
            return Err(Error::invalid(format!(
                "instance {} is measured at {} by {} annotations ({}); which is meant is the caller's to choose --- \
                 observations_at() lists them",
                self.instance_id,
                repr_str(timepoint),
                found.len(),
                crate::json::repr_list(&names)
            )));
        }
        Ok(found.first().copied())
    }

    pub fn volume(&self, timepoint: &str) -> Result<Option<f64>> {
        Ok(self.at(timepoint)?.and_then(|o| o.volume))
    }

    /// The volume at every timepoint the object was seen at, under the same
    /// rule as [`Track::at`].
    pub fn volumes(&self) -> Result<IndexMap<String, Option<f64>>> {
        self.timepoints().into_iter().map(|t| Ok((t.clone(), self.volume(&t)?))).collect()
    }

    /// `(v2 - v1) / v1` between two timepoints, or `None` if unmeasured.
    ///
    /// The two volumes are compared in millimetres: each observation's are
    /// its grid's `units` cubed, and a visit on a grid in metres beside one
    /// in millimetres read as a millionfold change.  Two volumes that share
    /// no physical measure --- an uncalibrated (`px`) grid beside a
    /// calibrated one, an area beside a volume --- have no change between
    /// them, and are `None`.
    pub fn relative_change(&self, first: &str, second: &str) -> Result<Option<f64>> {
        let (Some(before), Some(after)) = (self.at(first)?, self.at(second)?) else {
            return Ok(None);
        };
        let Some((before, after)) = comparable(before, after) else { return Ok(None) };
        if before <= 0.0 {
            return Ok(None);
        }
        Ok(Some((after - before) / before))
    }

    pub fn to_json(&self) -> Value {
        json!({
            "instance_id": self.instance_id,
            "class_ids": self.class_ids,
            "class_key": self.class_key,
            "timepoints": self.timepoints(),
            "observations": self.observations.iter().map(Observation::to_json).collect::<Vec<_>>(),
        })
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        let class = self.class_key.clone().unwrap_or_else(|| self.class_id().to_string());
        format!("Track({}, class={class}, seen at {})", self.instance_id, crate::json::repr_list(&self.timepoints()))
    }
}

/// Two observations' volumes in one measure: millimetres where both grids
/// are calibrated, their own units where both are the same uncalibrated
/// one, and `None` where no measure is shared or the boxes have different
/// numbers of axes.
fn comparable(before: &Observation, after: &Observation) -> Option<(f64, f64)> {
    if before.bbox.nrows() != after.bbox.nrows() {
        return None;
    }
    match (before.volume_mm(), after.volume_mm()) {
        (Some(b), Some(a)) => Some((b, a)),
        (None, None) if before.units == after.units => Some((before.volume?, after.volume?)),
        _ => None,
    }
}

/// The result of joining objects on `instance_id` across a sample.
#[derive(Debug, Clone, PartialEq)]
pub struct Tracking {
    pub tracks: BTreeMap<u64, Track>,
    pub timepoints: Vec<String>,
    /// `timepoint -> class ids whose absence there is a finding` (§11.3): the
    /// `annotated_class_ids` of each annotation at that visit that ignores
    /// no voxel of its grid (§7.7).
    pub coverage: IndexMap<String, BTreeSet<i64>>,
    /// `annotation -> why` for each annotation that carries `instance_ids`
    /// and that no track could join.
    pub skipped: IndexMap<String, String>,
}

#[allow(deprecated)]
impl Tracking {
    /// `present`, `resolved` or `unexamined`.
    #[deprecated(
        since = "2.1.0",
        note = "medh5 3.0 removes the class-level states, under which an object outside the visit's field of view \
                reads as resolved: use locate_tracks"
    )]
    pub fn state_at(&self, instance_id: u64, timepoint: &str) -> Option<&'static str> {
        let track = self.tracks.get(&instance_id)?;
        if !track.observations_at(timepoint).is_empty() {
            return Some(PRESENT);
        }
        let examined = self.coverage.get(timepoint);
        let looked = examined.is_some_and(|e| track.class_ids.iter().any(|c| e.contains(c)));
        Some(if looked { RESOLVED } else { UNEXAMINED })
    }

    #[deprecated(
        since = "2.1.0",
        note = "medh5 3.0 removes the class-level states, under which an object outside the visit's field of view \
                reads as resolved: use locate_tracks"
    )]
    pub fn states(&self, instance_id: u64) -> IndexMap<String, &'static str> {
        self.timepoints.iter().map(|tp| (tp.clone(), self.state_at(instance_id, tp).unwrap_or(UNEXAMINED))).collect()
    }

    /// Absent-then-present: not seen at a baseline that looked for it
    /// (`resolved` there), seen later.
    #[deprecated(
        since = "2.1.0",
        note = "medh5 3.0 removes the class-level states, under which an object outside the visit's field of view \
                reads as resolved: use locate_tracks"
    )]
    pub fn is_new(&self, instance_id: u64) -> bool {
        let states = self.states(instance_id);
        match self.timepoints.first() {
            None => false,
            Some(first) => states[first] == RESOLVED && states.values().any(|s| *s == PRESENT),
        }
    }

    /// Present-then-gone, where the later visit did look for it.
    #[deprecated(
        since = "2.1.0",
        note = "medh5 3.0 removes the class-level states, under which an object outside the visit's field of view \
                reads as resolved: use locate_tracks"
    )]
    pub fn is_resolved(&self, instance_id: u64) -> bool {
        if self.timepoints.len() < 2 {
            return false;
        }
        let states = self.states(instance_id);
        states[&self.timepoints[0]] == PRESENT && states[&self.timepoints[self.timepoints.len() - 1]] == RESOLVED
    }

    pub fn is_persistent(&self, instance_id: u64) -> bool {
        self.states(instance_id).values().all(|s| *s == PRESENT)
    }

    /// Instance ids carrying more than one class id (W909).
    pub fn class_conflicts(&self) -> BTreeMap<u64, Vec<i64>> {
        self.tracks.iter().filter(|(_, t)| t.has_class_conflict()).map(|(i, t)| (*i, t.class_ids.clone())).collect()
    }

    /// `timepoint -> instance ids` not seen there, whose class no annotation
    /// there examined without ignoring part of its grid.
    pub fn unexamined(&self) -> IndexMap<String, Vec<u64>> {
        let mut out = IndexMap::new();
        for tp in &self.timepoints {
            let ids: Vec<u64> =
                self.tracks.keys().copied().filter(|i| self.state_at(*i, tp) == Some(UNEXAMINED)).collect();
            if !ids.is_empty() {
                out.insert(tp.clone(), ids);
            }
        }
        out
    }

    pub fn repr(&self) -> String {
        format!("Tracking({} tracks over {} timepoints)", self.tracks.len(), self.timepoints.len())
    }

    pub fn to_json(&self) -> Value {
        self.json_with(&|instance_id, entry| {
            entry.insert(
                "states".into(),
                Value::Object(self.states(instance_id).into_iter().map(|(k, v)| (k, json!(v))).collect()),
            );
        })
    }

    /// [`Tracking::to_json`], each track's entry completed by `per_track`.
    fn json_with(&self, per_track: &dyn Fn(u64, &mut Map<String, Value>)) -> Value {
        let mut coverage: Vec<(&String, &BTreeSet<i64>)> = self.coverage.iter().collect();
        coverage.sort_by(|a, b| a.0.cmp(b.0));
        let coverage: Map<String, Value> = coverage.into_iter().map(|(k, v)| (k.clone(), json!(v))).collect();
        let tracks: Vec<Value> = self
            .tracks
            .iter()
            .map(|(i, t)| {
                let mut v = t.to_json();
                if let Value::Object(m) = &mut v {
                    per_track(*i, m);
                }
                v
            })
            .collect();
        let conflicts: Map<String, Value> =
            self.class_conflicts().into_iter().map(|(k, v)| (k.to_string(), json!(v))).collect();
        json!({
            "timepoints": self.timepoints,
            "coverage": coverage,
            "tracks": tracks,
            "class_conflicts": conflicts,
            "skipped": self.skipped,
        })
    }

    pub fn summary(&self) -> Value {
        self.summary_with(&|i| self.is_new(i), &|i| self.is_resolved(i), &|i| self.is_persistent(i))
    }

    fn summary_with(
        &self,
        new: &dyn Fn(u64) -> bool,
        resolved: &dyn Fn(u64) -> bool,
        persistent: &dyn Fn(u64) -> bool,
    ) -> Value {
        let pick = |f: &dyn Fn(u64) -> bool| -> Vec<u64> { self.tracks.keys().copied().filter(|i| f(*i)).collect() };
        json!({
            "tracks": self.tracks.len(),
            "timepoints": self.timepoints,
            "new": pick(new),
            "resolved": pick(resolved),
            "persistent": pick(persistent),
            "class_conflicts": self.class_conflicts().keys().copied().collect::<Vec<_>>(),
            "skipped": self.skipped.keys().collect::<Vec<_>>(),
        })
    }
}

/// The kinds a track joins in a 1.0 or 1.1 file.  A mask or a box measures
/// the object it follows; a point marks it (§10.6), and says only that it was
/// there.  [`tracked_kinds`] adds 1.2's `polylines`.
pub const TRACKED_KINDS: [&str; 4] = ["instances", "boxes", "obb", "points"];

/// The kinds a track joins: [`TRACKED_KINDS`] and 1.2's `polylines`, whose
/// `instance_ids` are those of the other kinds (1.2 §3, 1.0 §7.4).  A curve,
/// like a point, marks the object it traces and measures nothing.
pub fn tracked_kinds() -> &'static [&'static str] {
    &["instances", "boxes", "obb", "points", "polylines"]
}

/// Whether an annotation declares object identity a join can trust: a
/// `boxes` annotation without `instance_ids` numbers its rows positionally.
/// A closed file is refused rather than read as one without them.
pub fn carries_instance_ids(ann: &Annotation) -> Result<bool> {
    Ok(ann.kind() == "instances" || ann.optional_dataset("instance_ids")?.is_some())
}

/// Why no track can join an annotation that carries `instance_ids`, or
/// `None` when one can.
///
/// Every annotation [`carries_instance_ids`] admits was joined, and only
/// masks and axis-aligned boxes could be read as objects: an oriented box
/// or a landmark carrying the ids §10.6 asks for aborted the whole join,
/// and `medh5 track` with it.  A kind no track
/// reads is reported, never fatal.
fn unjoinable(ann: &Annotation) -> Option<String> {
    let kind = ann.kind();
    if !tracked_kinds().contains(&kind) {
        return Some(format!(
            "a {} annotation is neither measured nor marked by a track (they join {})",
            repr_str(kind),
            crate::json::repr_list(tracked_kinds())
        ));
    }
    if kind == "points" && matches!(ann.optional_dataset("class_ids"), Ok(None)) {
        return Some("its points carry no `class_ids`, so no track could say what they follow".into());
    }
    None
}

/// Whether `ann` declares part of its grid unexamined (§7.7, §11.3): an
/// ignore region holding at least one voxel.  One that cannot be read counts
/// as one, since nothing then shows where the annotator looked.
fn ignores_part(ann: &Annotation) -> bool {
    match ann.has_ignore_region() {
        Ok(false) => false,
        Ok(true) => !matches!(ann.ignore_region(), Ok(None)),
        Err(_) => true,
    }
}

/// The rows of a joinable annotation, each with the axis-aligned box a join
/// compares: a mask's or a box's own, an oriented box's enclosing box, a
/// point's degenerate `[p, p]`, and the box enclosing a curve's vertices.
fn objects(ann: &Annotation) -> Result<Vec<Instance>> {
    let bboxes: Vec<Array2<f32>> = match ann.kind() {
        "obb" => ann.obb_as_aabb()?.outer_iter().map(|b| b.mapv(|v| v as f32)).collect(),
        "points" => {
            let points = ann.read_as::<f32>("points")?.into_dimensionality::<ndarray::Ix2>()?;
            points.outer_iter().map(|p| Array2::from_shape_fn((p.len(), 2), |(k, _)| p[k])).collect()
        }
        "polylines" => ann
            .polylines()?
            .iter()
            .map(|line| {
                let columns = line.vertices.shape().get(1).copied().unwrap_or(0);
                Array2::from_shape_fn((columns, 2), |(k, end)| {
                    let column = line.vertices.index_axis(ndarray::Axis(1), k);
                    let bound = if end == 0 {
                        column.iter().copied().fold(f64::INFINITY, f64::min)
                    } else {
                        column.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    };
                    bound as f32
                })
            })
            .collect(),
        _ => return ann.instances(),
    };
    let n = bboxes.len();
    let classes = ann.object_class_ids()?;
    let ids = ann.instance_ids()?.unwrap_or_default();
    let scores = ann.scores()?;
    if classes.len() != n || ids.len() != n || scores.as_ref().is_some_and(|s| s.len() != n) {
        return Err(Error::coded(
            "E405",
            format!(
                "annotation {}: {n} object(s), but {} class id(s), {} instance id(s) and {} score(s)",
                repr_str(&ann.ann_id),
                classes.len(),
                ids.len(),
                scores.as_ref().map_or(n, Vec::len)
            ),
        ));
    }
    Ok(bboxes
        .into_iter()
        .enumerate()
        .map(|(i, bbox)| Instance {
            index: i,
            instance_id: ids[i],
            class_id: i64::from(classes[i]),
            bbox,
            mask: None,
            score: scores.as_ref().map(|s| f64::from(s[i])).filter(|v| v.is_finite()),
        })
        .collect())
}

/// An object's volume, and its voxel count where it has a mask.
///
/// `oriented` is an oriented box's product of edge lengths, along its own
/// axes: a rotation moves the box, not its volume.  In index space each
/// length is in voxels.  A point or a curve measures nothing.
fn measure(
    obj: &Instance,
    grid: Option<&Grid>,
    ann: &Annotation,
    oriented: Option<f64>,
) -> Result<(Option<f64>, Option<u64>)> {
    let space = ann.header.space.clone().unwrap_or_else(|| "index".into());
    if matches!(ann.kind(), "points" | "polylines") {
        return Ok((None, None));
    }
    if let Some(product) = oriented {
        if space == "world" {
            return Ok((Some(product), None));
        }
        return Ok((grid.map(|g| product * voxel_volume(&g.spacing)), None));
    }
    if let Some(mask) = &obj.mask {
        let count = mask.iter().filter(|v| **v).count() as u64;
        return Ok(match grid {
            None => (None, Some(count)),
            Some(g) => (Some(count as f64 * voxel_volume(&g.spacing)), Some(count)),
        });
    }
    if obj.bbox.ncols() != 2 {
        return Ok((None, None));
    }
    let extent: Vec<f64> = obj.bbox.outer_iter().map(|r| f64::from(r[1]) - f64::from(r[0])).collect();
    if space == "world" {
        return Ok((Some(extent.iter().product()), None));
    }
    let Some(g) = grid else { return Ok((None, None)) };
    if ann.kind() == "instances" {
        let flat: Vec<f64> = obj.bbox.iter().map(|v| f64::from(*v)).collect();
        // In floating point: a box from one end of the integers to the other
        // overflowed the subtraction, and two wide axes the product.
        let counted: f64 =
            box_to_slices(&flat, None)?.iter().map(|(a, b)| (i128::from(*b) - i128::from(*a)) as f64).product();
        return Ok((Some(counted * voxel_volume(&g.spacing)), None));
    }
    Ok((Some(extent.iter().zip(&g.spacing).map(|(e, s)| e * s).product()), None))
}

/// Join every instance-carrying annotation in `sample` on `instance_id`.
pub fn build_tracks(sample: &Sample, class_key: Option<&ClassKey>, do_measure: bool) -> Result<Tracking> {
    let declared = sample.timepoints()?.ids();
    let mut coverage: IndexMap<String, BTreeSet<i64>> = declared.iter().map(|t| (t.clone(), BTreeSet::new())).collect();
    let mut tracks: BTreeMap<u64, Vec<Observation>> = BTreeMap::new();
    let implicit: Vec<String> = if declared.len() == 1 { declared.clone() } else { vec![String::new()] };
    let mut skipped = IndexMap::new();
    for ann in sample.annotations()?.values() {
        if !carries_instance_ids(ann)? {
            continue;
        }
        if let Some(why) = unjoinable(ann) {
            skipped.insert(ann.ann_id.clone(), why);
            continue;
        }
        let wanted = match class_key {
            Some(k) => Some(ann.resolve_class(k)?),
            None => None,
        };
        let own = ann.timepoints();
        let timepoints = if own.is_empty() { implicit.clone() } else { own };
        // An annotation that ignores part of its grid examined its classes
        // only outside that region, and nothing here shows that an object it
        // did not find lay outside it, so its classes cover nothing: a
        // follow-up ignored as a whole resolved every lesion of the baseline.
        let examined: &[i64] = if ignores_part(ann) { &[] } else { ann.annotated_class_ids() };
        for tp in &timepoints {
            coverage.entry(tp.clone()).or_default().extend(examined.iter().copied());
        }
        let grid = ann.grid_id().and_then(|g| sample.grids().ok().and_then(|gs| gs.get(g).cloned()));
        let oriented = if do_measure && ann.kind() == "obb" { Some(ann.obb_volumes()?) } else { None };
        for obj in objects(ann)? {
            if wanted.is_some_and(|w| obj.class_id != w) {
                continue;
            }
            let (volume, count) = if do_measure {
                measure(&obj, grid.as_ref(), ann, oriented.as_ref().and_then(|v| v.get(obj.index).copied()))?
            } else {
                (None, None)
            };
            for tp in &timepoints {
                tracks.entry(obj.instance_id).or_default().push(Observation {
                    timepoint: tp.clone(),
                    annotation: ann.ann_id.clone(),
                    kind: ann.kind().to_string(),
                    index: obj.index,
                    instance_id: obj.instance_id,
                    class_id: obj.class_id,
                    bbox: obj.bbox.clone(),
                    voxel_count: count,
                    volume,
                    units: grid.as_ref().map(|g| g.units.clone()),
                    score: obj.score,
                    grid: ann.grid_id().map(str::to_string),
                });
            }
        }
    }
    let order: IndexMap<&String, usize> = declared.iter().enumerate().map(|(i, t)| (t, i)).collect();
    let label_set = sample.label_set()?;
    let mut built = BTreeMap::new();
    for (instance_id, mut observations) in tracks {
        observations.sort_by(|a, b| {
            let ka = (order.get(&a.timepoint).copied().unwrap_or(1 << 30), a.annotation.clone());
            let kb = (order.get(&b.timepoint).copied().unwrap_or(1 << 30), b.annotation.clone());
            ka.cmp(&kb)
        });
        let class_ids: Vec<i64> =
            observations.iter().map(|o| o.class_id).collect::<BTreeSet<_>>().into_iter().collect();
        let class_key = label_set.and_then(|ls| ls.by_id(class_ids[0])).map(|c| c.key.clone());
        built.insert(instance_id, Track { instance_id, class_ids, observations, class_key });
    }
    Ok(Tracking { tracks: built, timepoints: declared, coverage, skipped })
}

// -- why an absence is, or is not, a finding (§7.4) ---------------------------

/// Why an object unseen at a visit is, or is not, a finding there (§7.4).
///
/// The first two leave the object `resolved` there; the rest `unexamined`.
/// Where the visit's annotations disagree, the first in this order wins: one
/// annotation that looked at the place outweighs another that did not.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Absence {
    /// The object's located extent lies within an annotation's grid, within
    /// its images' acquired voxels, and outside its ignore region.
    Located,
    /// No geometry relates the object to the annotation's grid (no shared
    /// frame, no affine path), and the annotation examined the class and
    /// ignores no voxel: 1.0's class-level rule.  A registration would decide.
    ClassExamined,
    /// The located extent leaves the grid, or a voxel of it was not acquired
    /// (an image's `valid_mask`, §4.4).
    OutsideFov,
    /// The located extent holds an ignored voxel (§7.7).
    Ignored,
    /// No geometry relates the object to the annotation's grid, and the
    /// annotation ignores some voxels, which may be where the object is.
    Unrelated,
    /// The annotation examined the class but carries no object identity
    /// (`instance_ids`) a track could join, so nothing says the object is not
    /// among what it found.
    NoIdentity,
    /// Nothing at the visit examined the object's class.
    NotExamined,
}

/// Every [`Absence`], as [`Absence::as_str`] spells it, in precedence order.
pub const REASONS: &[&str] =
    &["located", "class_examined", "outside_fov", "ignored", "unrelated", "no_identity", "not_examined"];

const ABSENCES: [Absence; 7] = [
    Absence::Located,
    Absence::ClassExamined,
    Absence::OutsideFov,
    Absence::Ignored,
    Absence::Unrelated,
    Absence::NoIdentity,
    Absence::NotExamined,
];

impl Absence {
    pub fn as_str(self) -> &'static str {
        REASONS[self as usize]
    }

    /// The absence of a spelling [`Absence::as_str`] gives, or `None`.
    pub fn parse(reason: &str) -> Option<Absence> {
        REASONS.iter().position(|r| *r == reason).map(|i| ABSENCES[i])
    }

    /// Whether the absence is a finding: the object was looked for there.
    pub fn is_finding(self) -> bool {
        matches!(self, Absence::Located | Absence::ClassExamined)
    }

    /// `resolved` for a finding, `unexamined` otherwise.
    pub fn state(self) -> &'static str {
        if self.is_finding() {
            RESOLVED
        } else {
            UNEXAMINED
        }
    }
}

/// A [`Tracking`] with the reason for every unseen (object, visit).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct LocatedTracking {
    pub tracking: Tracking,
    /// `(instance_id, timepoint) -> why`, for every declared timepoint at
    /// which a track was not seen.
    pub absences: BTreeMap<(u64, String), Absence>,
}

impl LocatedTracking {
    /// A tracking and its absences, as [`locate_tracks`] found them.
    pub fn new(tracking: Tracking, absences: BTreeMap<(u64, String), Absence>) -> LocatedTracking {
        LocatedTracking { tracking, absences }
    }

    /// `present`, `resolved` or `unexamined`; `None` for an id no track has.
    pub fn state_at(&self, instance_id: u64, timepoint: &str) -> Option<&'static str> {
        let track = self.tracking.tracks.get(&instance_id)?;
        if !track.observations_at(timepoint).is_empty() {
            return Some(PRESENT);
        }
        Some(self.reason_at(instance_id, timepoint).map_or(UNEXAMINED, Absence::state))
    }

    /// Why the object is absent at `timepoint`; `None` where it was seen, or
    /// for a timepoint or id the tracking does not have.
    pub fn reason_at(&self, instance_id: u64, timepoint: &str) -> Option<Absence> {
        self.absences.get(&(instance_id, timepoint.to_string())).copied()
    }

    /// `{timepoint: state}` over every timepoint, in order.
    pub fn states(&self, instance_id: u64) -> IndexMap<String, &'static str> {
        self.tracking
            .timepoints
            .iter()
            .map(|tp| (tp.clone(), self.state_at(instance_id, tp).unwrap_or(UNEXAMINED)))
            .collect()
    }

    /// `{timepoint: reason}` over every timepoint, in order; `None` where the
    /// object was seen.
    pub fn reasons(&self, instance_id: u64) -> IndexMap<String, Option<Absence>> {
        self.tracking.timepoints.iter().map(|tp| (tp.clone(), self.reason_at(instance_id, tp))).collect()
    }

    /// Whether the absence at `timepoint` is a finding; with `located`, only
    /// where the object was located there (not by 1.0's class-level rule).
    fn looked(&self, instance_id: u64, timepoint: &str, located: bool) -> bool {
        match self.reason_at(instance_id, timepoint) {
            Some(reason) if located => reason == Absence::Located,
            Some(reason) => reason.is_finding(),
            None => false,
        }
    }

    /// Absent-then-present: not seen at a baseline that looked for it, seen
    /// later.  A baseline that did not image the place finds nothing new.
    pub fn is_new(&self, instance_id: u64, located: bool) -> bool {
        let Some(first) = self.tracking.timepoints.first() else { return false };
        self.looked(instance_id, first, located) && self.states(instance_id).values().any(|s| *s == PRESENT)
    }

    /// Present-then-gone, where the last visit looked for it.
    pub fn is_resolved(&self, instance_id: u64, located: bool) -> bool {
        let timepoints = &self.tracking.timepoints;
        if timepoints.len() < 2 {
            return false;
        }
        self.state_at(instance_id, &timepoints[0]) == Some(PRESENT)
            && self.looked(instance_id, &timepoints[timepoints.len() - 1], located)
    }

    pub fn is_persistent(&self, instance_id: u64) -> bool {
        self.states(instance_id).values().all(|s| *s == PRESENT)
    }

    /// `timepoint -> instance ids` not seen there and not looked for there.
    pub fn unexamined(&self) -> IndexMap<String, Vec<u64>> {
        let mut out = IndexMap::new();
        for tp in &self.tracking.timepoints {
            let ids: Vec<u64> =
                self.tracking.tracks.keys().copied().filter(|i| self.state_at(*i, tp) == Some(UNEXAMINED)).collect();
            if !ids.is_empty() {
                out.insert(tp.clone(), ids);
            }
        }
        out
    }

    /// [`Tracking::to_json`], each track with its `states` and `reasons`.
    pub fn to_json(&self) -> Value {
        self.tracking.json_with(&|instance_id, entry| {
            entry.insert(
                "states".into(),
                Value::Object(self.states(instance_id).into_iter().map(|(k, v)| (k, json!(v))).collect()),
            );
            entry.insert(
                "reasons".into(),
                Value::Object(
                    self.reasons(instance_id).into_iter().map(|(k, v)| (k, json!(v.map(Absence::as_str)))).collect(),
                ),
            );
        })
    }

    pub fn summary(&self) -> Value {
        self.tracking
            .summary_with(&|i| self.is_new(i, false), &|i| self.is_resolved(i, false), &|i| self.is_persistent(i))
    }

    pub fn repr(&self) -> String {
        self.tracking.repr()
    }
}

/// Join every instance-carrying annotation in `sample` on `instance_id`, and
/// say for every unseen (object, visit) why the absence is or is not a
/// finding (§7.4).
///
/// An object's *located extent* in an annotation is the box, rounded out to
/// whole voxels of its grid, that encloses the corners of every box observed
/// for it, each mapped into that grid through the shared frame or a path of
/// affine transforms.  A path with a nonlinear step does not locate it ---
/// a warped box's corners do not bound it --- and an observation that does
/// not map leaves the object unlocated.  Only the extent's window of an
/// ignore region and of each `valid_mask` is read.
pub fn locate_tracks(sample: &Sample, class_key: Option<&ClassKey>, do_measure: bool) -> Result<LocatedTracking> {
    let tracking = build_tracks(sample, class_key, do_measure)?;
    let implicit: Vec<String> =
        if tracking.timepoints.len() == 1 { tracking.timepoints.clone() } else { vec![String::new()] };
    let mut at: IndexMap<String, Vec<&Annotation>> = IndexMap::new();
    for ann in sample.annotations()?.values() {
        let own = ann.timepoints();
        for tp in if own.is_empty() { implicit.clone() } else { own } {
            at.entry(tp).or_default().push(ann);
        }
    }
    let mut locator = Locator { sample, valid_masks: HashMap::new() };
    let mut absences = BTreeMap::new();
    for (instance_id, track) in &tracking.tracks {
        for tp in &tracking.timepoints {
            if !track.observations_at(tp).is_empty() {
                continue;
            }
            let mut reason = Absence::NotExamined;
            for ann in at.get(tp).map(Vec::as_slice).unwrap_or(&[]) {
                reason = reason.min(locator.reason(track, ann)?);
                if reason == Absence::Located {
                    break;
                }
            }
            absences.insert((*instance_id, tp.clone()), reason);
        }
    }
    Ok(LocatedTracking { tracking, absences })
}

/// The voxel window of a located extent, one `(start, stop)` per spatial axis.
type Extent = Vec<(i64, i64)>;

/// What [`locate_tracks`] reads once per sample.
struct Locator<'a> {
    sample: &'a Sample,
    /// `grid -> images on it that declare a valid_mask`.
    valid_masks: HashMap<String, Vec<String>>,
}

impl Locator<'_> {
    /// Why `track`, unseen by `ann`, is or is not a finding there.
    fn reason(&mut self, track: &Track, ann: &Annotation) -> Result<Absence> {
        if !track.class_ids.iter().any(|c| ann.annotated_class_ids().contains(c)) {
            return Ok(Absence::NotExamined);
        }
        if !carries_instance_ids(ann)? || unjoinable(ann).is_some() {
            return Ok(Absence::NoIdentity);
        }
        let Some((grid, extent)) = self.extent(track, ann)? else {
            return Ok(if ignores_part(ann) { Absence::Unrelated } else { Absence::ClassExamined });
        };
        let shape = grid.spatial_shape();
        if extent.iter().zip(&shape).any(|((a, b), n)| *a < 0 || *b > *n as i64) || !self.acquired(&grid, &extent) {
            return Ok(Absence::OutsideFov);
        }
        if ignores_within(self.sample, ann, &extent) {
            return Ok(Absence::Ignored);
        }
        Ok(Absence::Located)
    }

    /// The located extent of `track` in `ann`'s grid, with that grid; `None`
    /// where any observation does not map into it.
    fn extent(&self, track: &Track, ann: &Annotation) -> Result<Option<(Grid, Extent)>> {
        let grids = self.sample.grids()?;
        let Some(target) = ann.grid_id().and_then(|g| grids.get(g)) else { return Ok(None) };
        let s = target.n_spatial();
        if track.observations.is_empty() || s == 0 {
            return Ok(None);
        }
        let (mut lo, mut hi) = (vec![f64::INFINITY; s], vec![f64::NEG_INFINITY; s]);
        for observation in &track.observations {
            let Some(corners) = self.corners_in(observation, target)? else { return Ok(None) };
            for corner in corners.chunks(s) {
                for k in 0..s {
                    lo[k] = lo[k].min(corner[k]);
                    hi[k] = hi[k].max(corner[k]);
                }
            }
        }
        if lo.iter().chain(&hi).any(|v| !v.is_finite()) {
            return Ok(None);
        }
        Ok(Some((target.clone(), rounded_out(&lo, &hi))))
    }

    /// An observation's box corners as continuous indices of `target`, flat;
    /// `None` where nothing relates the two grids by an affine.
    fn corners_in(&self, observation: &Observation, target: &Grid) -> Result<Option<Vec<f64>>> {
        let grids = self.sample.grids()?;
        let Some(source) = observation.grid.as_deref().and_then(|g| grids.get(g)) else { return Ok(None) };
        let s = target.n_spatial();
        let bbox = &observation.bbox;
        if bbox.ncols() != 2 || bbox.nrows() != s || source.n_spatial() != s {
            return Ok(None);
        }
        let flat: Vec<f64> = bbox.iter().map(|v| f64::from(*v)).collect();
        let corners = box_corners(&flat)?.concat();
        let space = self.sample.annotation(&observation.annotation)?.header.space.clone();
        let world = if space.as_deref() == Some("world") { corners } else { source.index_to_world(&corners)? };
        let Some(world) = self.world_between(source, target, &world) else { return Ok(None) };
        Ok(Some(target.world_to_index(&world)?))
    }

    /// `source`'s flat world points in `target`'s world: through the shared
    /// frame, or a path of affine transforms between their frames.
    fn world_between(&self, source: &Grid, target: &Grid, points: &[f64]) -> Option<Vec<f64>> {
        if source.grid_id == target.grid_id || source.comparable_with(target) {
            return source.world_into(target, points).ok();
        }
        let frame = |g: &Grid| g.frame_uid.clone().filter(|f| !f.is_empty());
        let (from, to) = (frame(source)?, frame(target)?);
        let transform = self.sample.resolve_frames(&from, &to).ok()??;
        if !is_linear(&transform) {
            return None;
        }
        let into = unit_scale(&source.units, transform.units())?;
        let out = unit_scale(transform.units(), &target.units)?;
        let s = source.n_spatial();
        let scaled: Vec<f64> = points.iter().map(|v| v * into).collect();
        let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[points.len() / s, s]), scaled).ok()?;
        let mapped = transform.transform_points(&array).ok()?;
        Some(mapped.iter().map(|v| v * out).collect())
    }

    /// Whether every voxel of `extent` was acquired by every image on `grid`
    /// that declares a `valid_mask` (§4.4).  One that cannot be read shows
    /// nothing acquired.
    fn acquired(&mut self, grid: &Grid, extent: &Extent) -> bool {
        let sample = self.sample;
        let images = self.valid_masks.entry(grid.grid_id.clone()).or_insert_with(|| match sample.images() {
            Ok(images) => images
                .iter()
                .filter(|(_, image)| {
                    image.grid_id().ok().as_deref() == Some(grid.grid_id.as_str())
                        && image.valid_mask().ok().flatten().is_some()
                })
                .map(|(id, _)| id.clone())
                .collect(),
            Err(_) => Vec::new(),
        });
        let roi = slices_of(extent);
        images.iter().all(|id| sample.valid_region(id, Some(&roi)).is_ok_and(|region| region.iter().all(|v| *v)))
    }
}

/// Whether `ann` ignores a voxel of `extent` (§7.7).  An ignore region that
/// cannot be read counts as covering it, as [`ignores_part`] has it.
fn ignores_within(sample: &Sample, ann: &Annotation, extent: &Extent) -> bool {
    if !ignores_part(ann) {
        return false;
    }
    let roi = slices_of(extent);
    if ann.is_voxel() {
        return sample.ignore_region(&ann.ann_id, Some(&roi)).map_or(true, |region| region.iter().any(|v| *v));
    }
    match ann.ignore_region() {
        Ok(Some(region)) => {
            let index: Vec<ndarray::SliceInfoElem> =
                extent.iter().map(|(a, b)| ndarray::SliceInfoElem::from(*a as usize..*b as usize)).collect();
            region.ndim() != extent.len() || region.slice(index.as_slice()).iter().any(|v| *v)
        }
        Ok(None) => false,
        Err(_) => true,
    }
}

fn slices_of(extent: &Extent) -> Vec<Slice> {
    extent.iter().map(|(a, b)| Slice::new(*a, *b)).collect()
}

/// Continuous index bounds rounded out to whole voxels: voxel `i` spans
/// `[i - 0.5, i + 0.5)`, so a bound on a voxel edge takes no voxel beyond it
/// (§3.3, §8.1), and one a hair past an edge from the affine's arithmetic is
/// read as on it.  A degenerate extent (a point on an edge) keeps one voxel.
fn rounded_out(lo: &[f64], hi: &[f64]) -> Extent {
    let snap = |v: f64| if (v - v.round()).abs() < 1e-6 { v.round() } else { v };
    lo.iter()
        .zip(hi)
        .map(|(l, h)| {
            let start = snap(l + 0.5).floor() as i64;
            let stop = snap(h + 0.5).ceil() as i64;
            (start, stop.max(start.saturating_add(1)))
        })
        .collect()
}

/// The factor taking world coordinates in `from` to `to`, or `None` where
/// no factor relates them (an uncalibrated unit against another).
fn unit_scale(from: &str, to: &str) -> Option<f64> {
    if from == to {
        return Some(1.0);
    }
    Some(mm_per_unit(from)? / mm_per_unit(to)?)
}

/// Whether a transform maps boxes to boxes' corners: identity and affine
/// steps only.  One that cannot be read is not.
fn is_linear(transform: &Transform) -> bool {
    match &transform.body {
        Body::Chain(steps) => steps.iter().all(is_linear),
        Body::Inverse(inner) => {
            is_linear(inner)
                && match stored_inverse(inner) {
                    Ok(Some(stored)) => is_linear(&stored),
                    Ok(None) => true,
                    Err(_) => false,
                }
        }
        Body::Stored => match transform.kind() {
            "identity" | "affine" => true,
            "displacement" | "bspline" => false,
            _ => transform.components().is_ok_and(|components| components.iter().all(is_linear)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s7_4_reason_precedence_prefers_a_finding() {
        // The visit's annotations disagree: the first reason in REASONS wins.
        let reasons: Vec<Absence> = REASONS.iter().map(|r| Absence::parse(r).expect("a reason")).collect();
        assert_eq!(reasons, ABSENCES);
        assert!(reasons.windows(2).all(|w| w[0] < w[1]));
        let disagreeing = [Absence::Ignored, Absence::ClassExamined, Absence::OutsideFov];
        assert_eq!(disagreeing.into_iter().min(), Some(Absence::ClassExamined));
        assert_eq!(reasons.iter().filter(|r| r.is_finding()).count(), 2);
        assert!(reasons.iter().all(|r| r.as_str() == REASONS[*r as usize]));
        assert_eq!(Absence::parse("resolved"), None);
    }

    #[test]
    fn s7_4_an_extent_rounds_out_to_whole_voxels() {
        // A bound on a voxel edge takes no voxel past it; one inside a voxel
        // takes that voxel; arithmetic noise at an edge is read as the edge.
        assert_eq!(rounded_out(&[0.5], &[4.5]), vec![(1, 5)]);
        assert_eq!(rounded_out(&[0.6], &[4.4]), vec![(1, 5)]);
        assert_eq!(rounded_out(&[0.4], &[4.6]), vec![(0, 6)]);
        assert_eq!(rounded_out(&[0.5 + 1e-9], &[4.5 - 1e-9]), vec![(1, 5)]);
        assert_eq!(rounded_out(&[3.0], &[3.0]), vec![(3, 4)]);
        assert_eq!(rounded_out(&[2.5], &[2.5]), vec![(3, 4)]);
    }

    /// A box-only object's volume multiplied `b - a` over its axes, and
    /// `i64::MAX - i64::MIN` overflowed.  It is measured in floating point.
    #[test]
    fn s7_4_a_box_past_the_integers_is_measured_without_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.medh5");
        crate::annotations::read::tests::wide_box(&path);
        let sample = crate::sample::open_sample(&path).unwrap();
        let tracking = build_tracks(&sample, None, true).unwrap();
        let observation = &tracking.tracks[&1].observations[0];
        assert_eq!(observation.voxel_count, None);
        assert_eq!(observation.volume, Some((u64::MAX as f64).powi(3)));
    }

    /// A bound past the integers saturates to their end, and the voxel a
    /// degenerate extent keeps overflowed past it.
    #[test]
    fn s7_4_an_extent_past_the_integers_rounds_out_without_a_panic() {
        assert_eq!(rounded_out(&[1e300], &[1e300]), vec![(i64::MAX, i64::MAX)]);
        assert_eq!(rounded_out(&[-1e300], &[1e300]), vec![(i64::MIN, i64::MAX)]);
    }

    #[test]
    fn s7_4_units_scale_only_where_calibrated() {
        assert_eq!(unit_scale("mm", "mm"), Some(1.0));
        assert_eq!(unit_scale("m", "mm"), Some(1000.0));
        assert_eq!(unit_scale("px", "px"), Some(1.0));
        assert_eq!(unit_scale("px", "mm"), None);
    }
}
