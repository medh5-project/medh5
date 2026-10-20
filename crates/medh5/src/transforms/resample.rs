//! Resampling images onto another grid, as a recorded derivation (1.0 §10.2,
//! §4.4, §11.1).
//!
//! The format stores native geometry so that resampling is never done
//! silently at ingest (§3.4's rationale); this is the deliberate, recorded
//! operation instead.  [`resample`] adds a resampled copy of each image to the
//! sample --- the sources stay untouched --- on an existing grid, or on a new
//! grid in the source's frame at another spacing, and records one `resample`
//! activity naming the inputs, the target grid, the transform path and the
//! interpolation, which every object it writes names in its `prov`.
//!
//! **Geometry is never invented.**  Two grids are mapped by their affines
//! alone only when §3.3 rule 4 makes them comparable; otherwise through the
//! sample's transforms, evaluated at the target's voxel centres as §10.2 says
//! ("evaluate T at each F-grid point and sample M at `T(x)`").  Without a path
//! the resample is refused, never aligned by assumption.
//!
//! The work is done a block of the target at a time, a block being one chunk
//! of the image written, so the memory a resample takes follows the target's
//! chunk shape, not its volume.  Linear and nearest read only the part of the
//! source a block maps into; cubic prefilters the whole source once, as SciPy
//! does, because its B-spline coefficients depend on every sample of a line.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use ndarray::{Array2, ArrayD, IxDyn};
use serde_json::{json, Map, Value};

use super::apply::{linear_sample, CubicCoefficients};
use super::model::{Body, Transform};
use crate::array::{DType, NdArray, Slice};
use crate::geometry::grid::{mm_per_unit, Grid};
use crate::geometry::linalg::inv;
use crate::h5::data;
use crate::ids::validate_id;
use crate::json::{float_repr, repr_list, repr_str};
use crate::sample::reader::{annotation_id, open_sample, KeyKind, Sample};
use crate::sample::writer::{amend_into, GridOptions, ImageOptions, SampleWriter};
use crate::{Error, Result, VERSION};

mod annotations;
pub use annotations::ResampledAnnotation;

/// The interpolations a resample takes: `linear` (the default), `nearest`
/// and `cubic`, the engine's SciPy-exact cubic B-spline.
pub const RESAMPLE_INTERPOLATIONS: &[&str] = &["linear", "nearest", "cubic"];

/// Where a resample puts its images.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ResampleTarget {
    /// An existing grid of the sample.
    Onto(String),
    /// A new grid, `grid_id`, in the source grid's frame, coordinate system
    /// and timepoint, at `spacing` (one value per spatial axis, in stored
    /// order), covering the source's extent edge to edge.
    Spacing { grid_id: String, spacing: Vec<f64> },
}

/// What [`resample`] does: which images, onto what, and how.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ResamplePlan {
    /// The images to resample, in the order their outputs are written.
    pub image_ids: Vec<String>,
    pub target: ResampleTarget,
    /// Appended to every source id to name its output; `_on_<target grid>`
    /// when `None`.
    pub suffix: Option<String>,
    /// One of [`RESAMPLE_INTERPOLATIONS`].
    pub interpolation: String,
    /// The stored value a target voxel takes when its source point lies
    /// outside the source.
    pub fill: f64,
    /// The dtype written: the source's under `nearest`, `float32` under
    /// `linear` and `cubic`, when `None`.
    pub dtype: Option<DType>,
    /// A new file to write; `None` amends the source in place.
    pub out: Option<PathBuf>,
    /// Carry the annotations on the source grids onto the target (the
    /// default); `false` moves the images alone.
    pub annotations: bool,
}

impl ResamplePlan {
    fn new(image_ids: Vec<String>, target: ResampleTarget) -> ResamplePlan {
        ResamplePlan {
            image_ids,
            target,
            suffix: None,
            interpolation: "linear".into(),
            fill: 0.0,
            dtype: None,
            out: None,
            annotations: true,
        }
    }

    /// Resample `image_ids` onto the existing grid `grid`.
    pub fn onto<I, S>(image_ids: I, grid: impl Into<String>) -> ResamplePlan
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        ResamplePlan::new(image_ids.into_iter().map(Into::into).collect(), ResampleTarget::Onto(grid.into()))
    }

    /// Resample `image_ids` onto a new grid `grid_id` at `spacing`.
    pub fn spacing<I, S>(image_ids: I, grid_id: impl Into<String>, spacing: Vec<f64>) -> ResamplePlan
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        ResamplePlan::new(
            image_ids.into_iter().map(Into::into).collect(),
            ResampleTarget::Spacing { grid_id: grid_id.into(), spacing },
        )
    }

    /// The plan the Python function and the command build from their
    /// arguments: exactly one of `onto` and `spacing`, and `grid_id` with
    /// `spacing` only, where it is required.  A `ValueError` otherwise.
    pub fn from_options(
        image_ids: Vec<String>,
        onto: Option<String>,
        spacing: Option<Vec<f64>>,
        grid_id: Option<String>,
    ) -> Result<ResamplePlan> {
        match (onto, spacing, grid_id) {
            (Some(grid), None, None) => Ok(ResamplePlan::onto(image_ids, grid)),
            (Some(_), None, Some(_)) => {
                Err(Error::Value("grid_id names a new grid, so it goes with spacing, not with onto".into()))
            }
            (None, Some(spacing), Some(grid_id)) => Ok(ResamplePlan::spacing(image_ids, grid_id, spacing)),
            (None, Some(_), None) => {
                Err(Error::Value("a new spacing makes a new grid, so grid_id must name it".into()))
            }
            (Some(_), Some(_), _) => Err(Error::Value("give one of onto and spacing, not both".into())),
            (None, None, _) => Err(Error::Value(
                "give onto (an existing grid) or spacing with grid_id (a new grid) to resample onto".into(),
            )),
        }
    }

    pub fn with_suffix(mut self, suffix: impl Into<String>) -> ResamplePlan {
        self.suffix = Some(suffix.into());
        self
    }

    pub fn with_interpolation(mut self, interpolation: impl Into<String>) -> ResamplePlan {
        self.interpolation = interpolation.into();
        self
    }

    pub fn with_fill(mut self, fill: f64) -> ResamplePlan {
        self.fill = fill;
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> ResamplePlan {
        self.dtype = Some(dtype);
        self
    }

    pub fn with_out(mut self, out: impl Into<PathBuf>) -> ResamplePlan {
        self.out = Some(out.into());
        self
    }

    pub fn with_annotations(mut self, annotations: bool) -> ResamplePlan {
        self.annotations = annotations;
        self
    }

    /// The target grid's id: the existing grid's, or the new one's.
    pub fn target_grid_id(&self) -> &str {
        match &self.target {
            ResampleTarget::Onto(grid) => grid,
            ResampleTarget::Spacing { grid_id, .. } => grid_id,
        }
    }

    /// The suffix outputs take: the given one, or `_on_<target grid>`.
    pub fn output_suffix(&self) -> String {
        self.suffix.clone().unwrap_or_else(|| format!("_on_{}", self.target_grid_id()))
    }
}

/// One image a resample wrote.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ResampledImage {
    /// The image resampled.
    pub source: String,
    /// The image written.
    pub image_id: String,
    /// The dtype written.
    pub dtype: DType,
    /// Its `valid_mask` (§4.4), written when any voxel holds no data.
    pub valid_mask: Option<String>,
    /// Target voxels whose source point lay outside the source, which took
    /// the fill value (spatial voxels, counted once whatever the channels).
    pub filled_voxels: u64,
}

/// What a resample wrote.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Resampled {
    /// The file written: the source, amended, or the plan's `out`.
    pub path: PathBuf,
    /// The `resample` activity every output names in its `prov`.
    pub activity_id: String,
    /// The target grid.
    pub grid_id: String,
    /// Whether the target grid was written by this resample.
    pub new_grid: bool,
    pub images: Vec<ResampledImage>,
    /// The ids of the transforms on each path, by source grid, for the
    /// source grids reached through a transform (an inverted step is
    /// `T⁻¹`).  A source grid comparable with the target (§3.3 rule 4) is
    /// mapped by the affines alone and is not listed.
    pub transforms: IndexMap<String, Vec<String>>,
    /// The annotations carried onto the target, in the sample's order.
    pub annotations: Vec<ResampledAnnotation>,
    /// The annotations on the source grids left where they are: every one
    /// when the plan carries none.
    pub not_carried: Vec<String>,
    /// Every object written, as a path in the sample (`images/CT_on_ref`).
    pub written: Vec<String>,
    /// The sum of every image's `filled_voxels`.
    pub filled_voxels: u64,
    /// The written file's `content_id`.
    pub content_id: Option<String>,
}

impl Resampled {
    /// The report as `medh5 resample --json` prints it.
    pub fn to_json(&self) -> Value {
        let images: Vec<Value> = self
            .images
            .iter()
            .map(|image| {
                json!({
                    "source": image.source,
                    "image_id": image.image_id,
                    "dtype": image.dtype.name(),
                    "valid_mask": image.valid_mask,
                    "filled_voxels": image.filled_voxels,
                })
            })
            .collect();
        json!({
            "path": self.path.to_string_lossy(),
            "activity_id": self.activity_id,
            "grid_id": self.grid_id,
            "new_grid": self.new_grid,
            "images": images,
            "transforms": self.transforms,
            "annotations": self.annotations.iter().map(ResampledAnnotation::to_json).collect::<Vec<_>>(),
            "not_carried": self.not_carried,
            "written": self.written,
            "filled_voxels": self.filled_voxels,
            "content_id": self.content_id,
        })
    }

    /// The report in a few lines, as `medh5 resample` prints it.
    pub fn lines(&self) -> Vec<String> {
        let grid = if self.new_grid { "new grid" } else { "grid" };
        let mut out = vec![format!(
            "resampled {} image{} onto {grid} {} ({})",
            self.images.len(),
            if self.images.len() == 1 { "" } else { "s" },
            repr_str(&self.grid_id),
            self.activity_id
        )];
        for image in &self.images {
            let mut line = format!("  {} -> {} ({})", image.source, image.image_id, image.dtype.name());
            if let Some(mask) = &image.valid_mask {
                line.push_str(&format!(", {} voxel(s) filled, valid_mask {mask}", image.filled_voxels));
            }
            out.push(line);
        }
        for annotation in &self.annotations {
            let mut line = format!("  {} -> {} ({})", annotation.source, annotation.annotation_id, annotation.kind);
            if let Some(mask) = &annotation.ignore_mask {
                line.push_str(&format!(", ignore_mask {mask}"));
            }
            out.push(line);
            if let Some(note) = &annotation.note {
                out.push(format!("    {note}"));
            }
        }
        if !self.not_carried.is_empty() {
            out.push(format!("  not carried: {}", self.not_carried.join(", ")));
        }
        for (grid, path) in &self.transforms {
            out.push(format!("  grid {} through {}", repr_str(grid), path.join(" -> ")));
        }
        out
    }
}

/// Resample images of the sample at `source` as `plan` says, writing the
/// results into the sample (or into `plan.out`) through one amendment.
///
/// Every output is refused before anything is written when its id is
/// taken, invalid (E003) or its geometry cannot be related to its source's
/// without inventing an alignment: a pair with no transform path (§10.2), or
/// two grids naming one frame that §3.3 rule 4 still keeps apart (W917).
pub fn resample(source: &Path, plan: &ResamplePlan) -> Result<Resampled> {
    if !RESAMPLE_INTERPOLATIONS.contains(&plan.interpolation.as_str()) {
        return Err(Error::Value(format!(
            "interpolation must be one of {}, not {}",
            repr_list(RESAMPLE_INTERPOLATIONS),
            repr_str(&plan.interpolation)
        )));
    }
    let out = match &plan.out {
        None => source.to_path_buf(),
        Some(out) => {
            let same = out == source
                || matches!((std::fs::canonicalize(out), std::fs::canonicalize(source)), (Ok(a), Ok(b)) if a == b);
            if !same && out.exists() {
                return Err(Error::invalid(format!(
                    "{} exists; a resample to a copy writes a new file and does not overwrite one",
                    repr_str(&out.to_string_lossy())
                )));
            }
            out.clone()
        }
    };
    let sample = open_sample(source)?;
    let prepared = prepare(&sample, plan)?;
    let mut writer = amend_into(source, &out, None, &[])?;
    let written = write(&sample, &mut writer, plan, &prepared);
    let mut result = match written {
        Ok(result) => result,
        Err(e) => {
            writer.abort();
            return Err(e);
        }
    };
    // Everything opened on the source closes before the amendment replaces
    // it: the resolved transforms hold groups of their own, which outlive the
    // sample's file identifier, and Windows cannot replace an open file.
    drop(prepared);
    drop(sample);
    result.content_id = writer.commit(true)?;
    result.path = out;
    Ok(result)
}

/// One image to resample, checked.
struct Job {
    source: String,
    output: String,
    mask: Option<String>,
    /// The source's own `valid_mask`, an annotation id.
    source_mask: Option<String>,
    source_grid: Grid,
    dtype: DType,
}

/// Everything decided before the amendment begins.
struct Prepared {
    target: Grid,
    new_grid: bool,
    jobs: Vec<Job>,
    /// Source grid id to its mapping onto the target.
    mappings: IndexMap<String, Mapping>,
    /// The annotations to carry.
    carry: annotations::Carry,
}

fn prepare(sample: &Sample, plan: &ResamplePlan) -> Result<Prepared> {
    let mut ids: Vec<String> = Vec::new();
    for id in &plan.image_ids {
        if !ids.contains(id) {
            ids.push(id.clone());
        }
    }
    if ids.is_empty() {
        return Err(Error::Value("a resample needs at least one image".into()));
    }
    let suffix = plan.output_suffix();
    let images = sample.images()?;
    let grids = sample.grids()?;
    let mut sources: Vec<(String, Grid)> = Vec::new();
    for id in &ids {
        let image = sample.image(id)?.level(0)?;
        sources.push((id.clone(), image.grid()?.clone()));
    }
    let (target, new_grid) = match &plan.target {
        ResampleTarget::Onto(grid_id) => {
            let Some(grid) = grids.get(grid_id) else {
                return Err(Error::coded(
                    "E101",
                    format!(
                        "grid {} is not in the sample; resample onto one of {}",
                        repr_str(grid_id),
                        repr_list(&grids.keys().cloned().collect::<Vec<_>>())
                    ),
                ));
            };
            (grid.clone(), false)
        }
        ResampleTarget::Spacing { grid_id, spacing } => {
            let first = &sources[0].1;
            if let Some((other, grid)) = sources.iter().find(|(_, g)| g.grid_id != first.grid_id) {
                return Err(Error::invalid(format!(
                    "images {} (grid {}) and {} (grid {}) are on different grids; a new spacing derives one grid \
                     from one source grid, so resample them separately, or onto an existing grid",
                    repr_str(&sources[0].0),
                    repr_str(&first.grid_id),
                    repr_str(other),
                    repr_str(&grid.grid_id)
                )));
            }
            validate_id(grid_id, "grid id")?;
            if grids.contains_key(grid_id) {
                return Err(Error::invalid(format!(
                    "grid {} is already in the sample; name the new grid another way, or resample onto it",
                    repr_str(grid_id)
                )));
            }
            (respaced(first, grid_id, spacing)?, true)
        }
    };
    let annotations = sample.annotations()?;
    let mut taken: HashSet<String> = HashSet::new();
    let mut jobs = Vec::new();
    for (id, source_grid) in sources {
        let output = format!("{id}{suffix}");
        validate_id(&output, "image id")?;
        let mask = format!("{output}_valid");
        validate_id(&mask, "annotation id")?;
        if images.contains_key(&output) {
            return Err(Error::invalid(format!(
                "image {} is already in the sample, so {} cannot be resampled onto {} under it; resampling twice \
                 onto one grid needs another suffix",
                repr_str(&output),
                repr_str(&id),
                repr_str(&target.grid_id)
            )));
        }
        if annotations.contains_key(&mask) {
            return Err(Error::invalid(format!(
                "annotation {} is already in the sample, and the resampled {} would name it as its valid_mask; \
                 choose another suffix",
                repr_str(&mask),
                repr_str(&output)
            )));
        }
        if !taken.insert(output.clone()) {
            return Err(Error::invalid(format!("two images would be written as {}", repr_str(&output))));
        }
        check_axes(&id, &source_grid, &target, new_grid)?;
        let image = sample.image(&id)?.level(0)?;
        let source_dtype = image.dtype()?;
        let dtype = match (plan.dtype, plan.interpolation.as_str()) {
            (Some(d), _) => d,
            (None, "nearest") => source_dtype,
            (None, _) => DType::F32,
        };
        check_fill(plan.fill, dtype)?;
        let source_mask = image.valid_mask()?.map(|m| annotation_id(&m).to_string());
        jobs.push(Job { source: id, output, mask: Some(mask), source_mask, source_grid, dtype });
    }
    let mut mappings: IndexMap<String, Mapping> = IndexMap::new();
    for job in &jobs {
        if mappings.contains_key(&job.source_grid.grid_id) {
            continue;
        }
        let mapping = if new_grid {
            // Derived from the source grid itself: one world by construction,
            // whether or not the grid names a frame.
            Mapping::affine(&target, &job.source_grid)?
        } else {
            mapping_between(sample, &target, &job.source_grid)?
        };
        mappings.insert(job.source_grid.grid_id.clone(), mapping);
    }
    let sources: IndexMap<String, Grid> =
        jobs.iter().map(|job| (job.source_grid.grid_id.clone(), job.source_grid.clone())).collect();
    let excluded: HashSet<String> = jobs.iter().filter_map(|job| job.source_mask.clone()).collect();
    // Images and annotations are separate groups, so an annotation may share
    // an image's id; what it may not share is another annotation's, the
    // outputs' valid masks among them.
    let mut annotation_ids: HashSet<String> = jobs.iter().filter_map(|job| job.mask.clone()).collect();
    let carry =
        annotations::prepare(sample, plan, &target, &mappings, &sources, &excluded, &suffix, &mut annotation_ids)?;
    Ok(Prepared { target, new_grid, jobs, mappings, carry })
}

/// The grid at `spacing` over `source`'s extent: its voxel edges kept, not
/// its first centre.  Along an axis of `n` voxels of size `s` from the edge
/// `o − s/2`, the new grid has `ceil(n·s / s′)` voxels, the first centred at
/// `o − s/2 + s′/2`.
pub fn respaced(source: &Grid, grid_id: &str, spacing: &[f64]) -> Result<Grid> {
    let s = source.n_spatial();
    if spacing.len() != s {
        return Err(Error::Value(format!(
            "spacing has {} value(s) and grid {} has {s} spatial axes; give one per axis, in stored order",
            spacing.len(),
            repr_str(&source.grid_id)
        )));
    }
    if spacing.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(Error::Value(format!(
            "spacing ({}) must be finite and positive",
            spacing.iter().map(|v| float_repr(*v)).collect::<Vec<_>>().join(", ")
        )));
    }
    let spatial = source.spatial_axes();
    let mut shape = source.shape.clone();
    let mut offsets = Vec::with_capacity(s);
    for (k, axis) in spatial.iter().enumerate() {
        let span = source.shape[*axis] as f64 * source.spacing[k];
        let count = span / spacing[k];
        let rounded = count.round();
        // `2 · 0.3 / 0.6` is 1.0000000000000002: an edge that lands on the
        // source's within rounding is the source's edge, not a voxel more.
        let n = if (count - rounded).abs() <= 1e-9 * rounded.max(1.0) { rounded } else { count.ceil() };
        shape[*axis] = (n as i64).max(1);
        offsets.push((spacing[k] - source.spacing[k]) / 2.0);
    }
    let mut origin = source.origin.clone();
    for (r, value) in origin.iter_mut().enumerate() {
        for (k, offset) in offsets.iter().enumerate() {
            *value += source.direction[[r, k]] * offset;
        }
    }
    let grid = Grid {
        grid_id: grid_id.to_string(),
        shape,
        axis_names: source.axis_names.clone(),
        axis_kinds: source.axis_kinds.clone(),
        spacing: spacing.to_vec(),
        origin,
        direction: source.direction.clone(),
        coord_system: source.coord_system.clone(),
        units: source.units.clone(),
        timepoint: source.timepoint.clone(),
        frame_uid: source.frame_uid.clone(),
        time_values: source.time_values.clone(),
        time_units: source.time_units.clone(),
        chunk_hint: None,
        patch_hint: None,
        extra: Vec::new(),
    };
    grid.check()?;
    Ok(grid)
}

/// Refuse a source whose non-spatial axes the target does not have: an
/// image takes its grid's shape (§4.1), and only spatial axes are resampled.
fn check_axes(image_id: &str, source: &Grid, target: &Grid, new_grid: bool) -> Result<()> {
    if new_grid {
        return Ok(());
    }
    let leading = |g: &Grid| -> Vec<(String, i64)> {
        g.axis_kinds
            .iter()
            .zip(&g.shape)
            .filter(|(kind, _)| *kind != "spatial")
            .map(|(kind, n)| (kind.clone(), *n))
            .collect()
    };
    let describe = |axes: &[(String, i64)]| -> String {
        if axes.is_empty() {
            "none".into()
        } else {
            axes.iter().map(|(kind, n)| format!("{kind} of {n}")).collect::<Vec<_>>().join(", ")
        }
    };
    let (ours, theirs) = (leading(source), leading(target));
    if ours != theirs {
        return Err(Error::invalid(format!(
            "image {} has non-spatial axes ({}) that grid {} does not ({}); an image takes its grid's shape \
             (§4.1) and a resample moves only the spatial axes, so resample onto a new spacing instead",
            repr_str(image_id),
            describe(&ours),
            repr_str(&target.grid_id),
            describe(&theirs)
        )));
    }
    if !same_times(source, target) {
        return Err(Error::invalid(format!(
            "image {} has a time axis at acquisition times {} and grid {} at {}; a resample moves only the \
             spatial axes and carries each time point as it is, so the target must hold the same times (§3.2)",
            repr_str(image_id),
            describe_times(source),
            repr_str(&target.grid_id),
            describe_times(target)
        )));
    }
    if source.n_spatial() != target.n_spatial() {
        return Err(Error::invalid(format!(
            "image {} has {} spatial axes and grid {} has {}; a resample keeps the number of spatial axes",
            repr_str(image_id),
            source.n_spatial(),
            repr_str(&target.grid_id),
            target.n_spatial()
        )));
    }
    Ok(())
}

/// Refuse a fill value the dtype written cannot hold exactly.
fn check_fill(fill: f64, dtype: DType) -> Result<()> {
    let holds = if dtype.is_float() {
        fill.is_nan() || DType::F64 == dtype || NdArray::from_vec(&[1], vec![fill])?.astype(dtype).to_f64()[[0]] == fill
    } else {
        fill.fract() == 0.0 && fill.is_finite() && dtype.holds(fill as i128)
    };
    if holds {
        return Ok(());
    }
    Err(Error::Value(format!("fill {} cannot be stored as {} without changing it", float_repr(fill), dtype.name())))
}

/// How the target's voxel centres map to a source grid's continuous indices.
enum Mapping {
    /// One matrix, `(S+1, S+1)`: target index to source index.
    Affine(Array2<f64>),
    /// Target index to target world, through a transform into the source's
    /// frame, then to source index, with the unit factors on either side.
    Through(Box<Through>),
}

/// [`Mapping::Through`]'s grids, path and unit factors.
struct Through {
    target: Grid,
    source: Grid,
    transform: Transform,
    into: f64,
    out: f64,
}

impl Mapping {
    /// Grids sharing one world (§3.3 rule 4): the source's inverse affine
    /// after the target's, through the unit factor between them.
    fn affine(target: &Grid, source: &Grid) -> Result<Mapping> {
        let scale = target.world_scale_into(source)?;
        let s = target.n_spatial();
        let mut units = Array2::<f64>::eye(s + 1);
        for k in 0..s {
            units[[k, k]] = scale;
        }
        Ok(Mapping::Affine(inv(&source.affine())?.dot(&units).dot(&target.affine())))
    }

    /// `(N, S)` target indices to `(N, S)` source indices.
    fn map(&self, indices: &Array2<f64>) -> Result<Array2<f64>> {
        match self {
            Mapping::Affine(m) => {
                let s = m.nrows() - 1;
                let linear = m.slice(ndarray::s![..s, ..s]);
                let mut out = indices.dot(&linear.t());
                for mut row in out.outer_iter_mut() {
                    for (r, v) in row.iter_mut().enumerate() {
                        *v += m[[r, s]];
                    }
                }
                Ok(out)
            }
            Mapping::Through(through) => {
                let Through { target, source, transform, into, out } = through.as_ref();
                let s = target.n_spatial();
                let flat: Vec<f64> = indices.iter().copied().collect();
                let world: Vec<f64> = target.index_to_world(&flat)?.into_iter().map(|v| v * into).collect();
                let points = ArrayD::from_shape_vec(IxDyn(&[indices.nrows(), s]), world)?;
                let moved: Vec<f64> = transform.transform_points(&points)?.iter().map(|v| v * out).collect();
                let index = source.world_to_index(&moved)?;
                Ok(Array2::from_shape_vec((indices.nrows(), s), index)?)
            }
        }
    }

    fn transform_ids(&self) -> Option<Vec<String>> {
        match self {
            Mapping::Affine(_) => None,
            Mapping::Through(through) => Some(path_ids(&through.transform)),
        }
    }

    /// The stored transforms the mapping reads, as sample paths.
    fn inputs(&self) -> Vec<String> {
        match self {
            Mapping::Affine(_) => Vec::new(),
            Mapping::Through(through) => {
                let mut out = Vec::new();
                stored_ids(&through.transform, &mut out);
                out.into_iter().map(|id| format!("transforms/{id}")).collect()
            }
        }
    }
}

/// The ids of a resolved path's steps, an inverted one as `T⁻¹`.
fn path_ids(transform: &Transform) -> Vec<String> {
    match &transform.body {
        Body::Chain(steps) => steps.iter().flat_map(path_ids).collect(),
        _ => vec![transform.transform_id.clone()],
    }
}

/// The ids of the stored transforms a resolved path evaluates.
fn stored_ids(transform: &Transform, out: &mut Vec<String>) {
    match &transform.body {
        Body::Chain(steps) => steps.iter().for_each(|step| stored_ids(step, out)),
        Body::Inverse(inner) => stored_ids(inner, out),
        Body::Stored => {
            if !out.contains(&transform.transform_id) {
                out.push(transform.transform_id.clone());
            }
        }
    }
}

fn unit_scale(from: &str, to: &str) -> Option<f64> {
    if from == to {
        return Some(1.0);
    }
    Some(mm_per_unit(from)? / mm_per_unit(to)?)
}

/// How `source`'s grid maps onto `target`'s: by the affines when §3.3 rule 4
/// makes them comparable, else through the transform path from the target's
/// frame to the source's (§10.2), else refused.
fn mapping_between(sample: &Sample, target: &Grid, source: &Grid) -> Result<Mapping> {
    if target.grid_id == source.grid_id || target.comparable_with(source) {
        return Mapping::affine(target, source);
    }
    let frame = |g: &Grid| g.frame_uid.clone().filter(|f| !f.is_empty());
    let (to, from) = match (frame(target), frame(source)) {
        (Some(t), Some(s)) => (t, s),
        (t, _) => {
            let missing = if t.is_none() { target } else { source };
            return Err(Error::invalid(format!(
                "grid {} names no frame of reference, so nothing relates it to grid {} and a resample between \
                 them would assume an alignment nobody computed (§3.4)",
                repr_str(&missing.grid_id),
                repr_str(if t.is_none() { &source.grid_id } else { &target.grid_id })
            )));
        }
    };
    if to == from {
        // W917: one frame named, but another coord_system or S.
        return Err(Error::invalid(format!(
            "grids {} and {} name one frame of reference, {}, but are not comparable (§3.3 rule 4: {}); no \
             transform relates them, so neither can be resampled onto the other",
            repr_str(&source.grid_id),
            repr_str(&target.grid_id),
            repr_str(&to),
            if source.coord_system != target.coord_system {
                format!("coord_system {} against {}", repr_str(&source.coord_system), repr_str(&target.coord_system))
            } else {
                format!("{} spatial axes against {}", source.n_spatial(), target.n_spatial())
            }
        )));
    }
    let Some(transform) = sample.transform_between_as(&target.grid_id, &source.grid_id, Some(KeyKind::Grid))? else {
        return Err(Error::invalid(format!(
            "no transform path relates frame {} (grid {}) to frame {} (grid {}), so {} cannot be resampled onto \
             {} without assuming an alignment (§10.2); store the registration between them as a transform first",
            repr_str(&to),
            repr_str(&target.grid_id),
            repr_str(&from),
            repr_str(&source.grid_id),
            repr_str(&source.grid_id),
            repr_str(&target.grid_id)
        )));
    };
    let units = transform.units().to_string();
    let (Some(into), Some(out)) = (unit_scale(&target.units, &units), unit_scale(&units, &source.units)) else {
        return Err(Error::coded(
            "E506",
            format!(
                "transform {} is in {} and grids {} and {} are in {} and {}; no factor relates them (§3.5)",
                repr_str(&transform.transform_id),
                repr_str(&units),
                repr_str(&target.grid_id),
                repr_str(&source.grid_id),
                repr_str(&target.units),
                repr_str(&source.units)
            ),
        ));
    };
    Ok(Mapping::Through(Box::new(Through { target: target.clone(), source: source.clone(), transform, into, out })))
}

fn write(sample: &Sample, writer: &mut SampleWriter, plan: &ResamplePlan, prepared: &Prepared) -> Result<Resampled> {
    let target = &prepared.target;
    let tool = writer.software("medh5", Some(VERSION), Map::new())?;
    // The activity is recorded first, so each object names it as it is
    // written; its outputs are completed below.
    let mut written: Vec<String> = Vec::new();
    if prepared.new_grid {
        written.push(format!("grids/{}", target.grid_id));
    }
    for job in &prepared.jobs {
        written.push(format!("images/{}", job.output));
    }
    let mut inputs: Vec<String> = prepared.jobs.iter().map(|job| format!("images/{}", job.source)).collect();
    for job in &prepared.jobs {
        if let Some(mask) = &job.source_mask {
            let path = format!("annotations/{mask}");
            if !inputs.contains(&path) {
                inputs.push(path);
            }
        }
    }
    for path in prepared.carry.inputs() {
        if !inputs.contains(&path) {
            inputs.push(path);
        }
    }
    if !prepared.new_grid {
        inputs.push(format!("grids/{}", target.grid_id));
    }
    let mut transforms: IndexMap<String, Vec<String>> = IndexMap::new();
    for (grid, mapping) in &prepared.mappings {
        if let Some(ids) = mapping.transform_ids() {
            transforms.insert(grid.clone(), ids);
        }
        for path in mapping.inputs() {
            if !inputs.contains(&path) {
                inputs.push(path);
            }
        }
    }
    let mut params = Map::new();
    match &plan.target {
        ResampleTarget::Onto(grid) => {
            params.insert("onto".into(), json!(grid));
        }
        ResampleTarget::Spacing { grid_id, spacing } => {
            params.insert("spacing".into(), json!(spacing));
            params.insert("grid_id".into(), json!(grid_id));
        }
    }
    params.insert("transforms".into(), json!(transforms));
    params.insert("interpolation".into(), json!(plan.interpolation));
    params.insert("fill".into(), fill_json(plan.fill));
    params.insert("dtype".into(), json!(plan.dtype.map(DType::name)));
    params.insert("annotations".into(), json!(plan.annotations));
    if prepared.new_grid {
        let options = GridOptions {
            origin: Some(target.origin.clone()),
            direction: Some(target.direction.clone()),
            axis_names: Some(target.axis_names.clone()),
            axis_kinds: Some(target.axis_kinds.clone()),
            coord_system: Some(target.coord_system.clone()),
            units: Some(target.units.clone()),
            timepoint: target.timepoint.clone(),
            frame_uid: target.frame_uid.clone(),
            time_values: target.time_values.clone(),
            time_units: target.time_units.clone(),
            ..Default::default()
        };
        writer.add_grid(&target.grid_id, &target.shape, &target.spacing, options)?;
    }
    let mut provisional = Map::new();
    provisional.insert("tool".into(), json!("medh5 resample"));
    provisional.insert("inputs".into(), json!(inputs));
    provisional.insert("params".into(), Value::Object(params.clone()));
    let activity = writer.activity("resample", Some(&tool.id), None, provisional)?;
    let mut images = Vec::new();
    for job in &prepared.jobs {
        let mapping = &prepared.mappings[&job.source_grid.grid_id];
        let (filled, valid) = resample_image(sample, writer, job, target, mapping, plan, &activity.id)?;
        let mask = if let Some(valid) = valid {
            let name = job.mask.clone().expect("named in prepare");
            writer.add_mask(&name, valid, &target.grid_id, "other", Some(&activity.id), None)?;
            let images_attr = writer.root()?.group("images")?.dataset(&job.output)?;
            crate::h5::attrs::write(&images_attr, "valid_mask", &crate::h5::attrs::AttrValue::Str(name.clone()))?;
            written.push(format!("annotations/{name}"));
            Some(name)
        } else {
            None
        };
        images.push(ResampledImage {
            source: job.source.clone(),
            image_id: job.output.clone(),
            dtype: job.dtype,
            valid_mask: mask,
            filled_voxels: filled,
        });
    }
    let carried =
        annotations::write(sample, writer, &prepared.carry, &prepared.mappings, target, &activity.id, &mut written)?;
    let mut fields = Map::new();
    fields.insert("tool".into(), json!("medh5 resample"));
    fields.insert("inputs".into(), json!(inputs));
    fields.insert("outputs".into(), json!(written));
    fields.insert("params".into(), Value::Object(params));
    let recorded = crate::curation::provenance::Activity::from_json(&{
        let mut doc = Map::new();
        doc.insert("id".into(), json!(activity.id));
        doc.insert("type".into(), json!("resample"));
        doc.insert("agent".into(), json!(tool.id));
        doc.extend(fields);
        Value::Object(doc)
    })?;
    writer.document_mut().provenance.add_activity(recorded, true)?;
    let filled_voxels = images.iter().map(|i| i.filled_voxels).sum();
    Ok(Resampled {
        path: PathBuf::new(),
        activity_id: activity.id,
        grid_id: target.grid_id.clone(),
        new_grid: prepared.new_grid,
        images,
        transforms,
        annotations: carried,
        not_carried: prepared.carry.not_carried.clone(),
        written,
        filled_voxels,
        content_id: None,
    })
}

/// `fill` as the activity records it: a number, or for a non-finite fill
/// the string Python spells it with (`"nan"`, `"inf"`, `"-inf"`) --- JSON has
/// no NaN, and `null` would record a fill nobody used.
fn fill_json(fill: f64) -> Value {
    if fill.is_nan() {
        json!("nan")
    } else if fill.is_infinite() {
        json!(if fill > 0.0 { "inf" } else { "-inf" })
    } else {
        json!(fill)
    }
}

/// A grid's `time_units`: `"s"` unless it says `"ms"` (§3.2).
fn time_units(grid: &Grid) -> &str {
    if grid.time_units.as_deref() == Some("ms") {
        "ms"
    } else {
        "s"
    }
}

/// Whether two grids state the same acquisition times.  In the same units the
/// values must be equal; across units the seconds are taken to milliseconds,
/// one rounding, and may differ by that rounding only --- a tolerance relative
/// to the value would let two times a second apart pass near 1e9 s.
fn same_times(a: &Grid, b: &Grid) -> bool {
    let (x, y) = match (&a.time_values, &b.time_values) {
        (None, None) => return true,
        (Some(x), Some(y)) if x.len() == y.len() => (x, y),
        _ => return false,
    };
    let ms = |grid: &Grid, t: f64| if time_units(grid) == "s" { t * 1e3 } else { t };
    if time_units(a) == time_units(b) {
        return x == y;
    }
    x.iter().zip(y).all(|(p, q)| {
        let (p, q) = (ms(a, *p), ms(b, *q));
        p == q || (p - q).abs() <= 2.0 * f64::EPSILON * p.abs().max(q.abs())
    })
}

fn describe_times(grid: &Grid) -> String {
    match &grid.time_values {
        None => "none".into(),
        Some(values) => format!("{values:?} {}", grid.time_units.as_deref().unwrap_or("s")),
    }
}

/// The blocks covering `shape`, one chunk extent each (the whole axis where
/// there is no chunking), in C order: `(start, stop)` per axis.  Generated
/// one at a time, so a fine chunking costs no memory up front.
fn blocks(shape: &[usize], chunk: &[usize]) -> impl Iterator<Item = Vec<(usize, usize)>> {
    let steps: Vec<usize> = shape.iter().zip(chunk).map(|(n, c)| (*c).clamp(1, (*n).max(1))).collect();
    let counts: Vec<usize> = shape.iter().zip(&steps).map(|(n, step)| n.div_ceil(*step)).collect();
    let total: usize = counts.iter().product();
    let shape = shape.to_vec();
    (0..total).map(move |mut flat| {
        let mut block = vec![(0, 0); shape.len()];
        for k in (0..shape.len()).rev() {
            let start = (flat % counts[k]) * steps[k];
            flat /= counts[k];
            block[k] = (start, (start + steps[k]).min(shape[k]));
        }
        block
    })
}

#[cfg(test)]
thread_local! {
    /// The most target voxels one block held, for the test that a resample
    /// works a chunk at a time.
    static LARGEST_BLOCK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The most source voxels one block read.
    static LARGEST_CROP: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// [`CROP_BUDGET`], lowered by a test.
    static TEST_CROP_BUDGET: std::cell::Cell<usize> = const { std::cell::Cell::new(CROP_BUDGET) };
}

/// The most source voxels (times the lanes of a leading chunk) one block of
/// the target reads: 2^22, 32 MiB as `f64`.
const CROP_BUDGET: usize = 1 << 22;

fn crop_budget() -> usize {
    #[cfg(test)]
    return TEST_CROP_BUDGET.with(|budget| budget.get());
    #[cfg(not(test))]
    CROP_BUDGET
}

/// Resample one image block by block; the voxels filled, and where the
/// result holds data (§4.4) --- `None` when every voxel does, so a resample
/// that fills nothing allocates no target-wide mask.
fn resample_image(
    sample: &Sample,
    writer: &mut SampleWriter,
    job: &Job,
    target: &Grid,
    mapping: &Mapping,
    plan: &ResamplePlan,
    activity: &str,
) -> Result<(u64, Option<ArrayD<bool>>)> {
    let image = sample.image(&job.source)?.level(0)?;
    let source = &job.source_grid;
    let s = source.n_spatial();
    let source_spatial = source.spatial_shape();
    let leading_shape: Vec<usize> = source.shape_usize()[..source.ndim() - s].to_vec();
    let lanes: usize = leading_shape.iter().product::<usize>().max(1);
    let (slope, intercept) = (image.attr("rescale_slope")?, image.attr("rescale_intercept")?);
    let window = image.window()?;
    let options = ImageOptions {
        value_type: Some(image.value_type()?),
        value_units: image.value_units()?,
        channel_names: image.channel_names()?,
        rescale_slope: slope.and_then(|v| v.as_f64()),
        rescale_intercept: intercept.and_then(|v| v.as_f64()),
        window_center: window.as_ref().map(|(c, _)| c.clone()),
        window_width: window.map(|(_, w)| w),
        prov: Some(activity.to_string()),
        ..Default::default()
    };
    let (dataset, chunks) =
        writer.create_image(&job.output, job.dtype, &target.grid_id, &image.modality()?, options)?;
    let target_spatial = target.spatial_shape();
    let (leading_chunk, chunk): (Vec<usize>, Vec<usize>) = match &chunks {
        Some(c) => (c[..c.len() - s].to_vec(), c[c.len() - s..].to_vec()),
        None => (leading_shape.clone(), target_spatial.clone()),
    };
    let cubic = if plan.interpolation == "cubic" {
        let whole = image.read(None, false, Some(DType::F64))?.to_f64();
        let mut field_shape = vec![lanes];
        field_shape.extend(&source_spatial);
        let field = whole.into_shape_with_order(IxDyn(&field_shape))?;
        Some(CubicCoefficients::new(&field, "zero")?)
    } else {
        None
    };
    let group_max: usize = leading_shape.iter().zip(&leading_chunk).map(|(n, c)| (*c).clamp(1, (*n).max(1))).product();
    let mut valid: Option<ArrayD<bool>> = None;
    let mut filled = 0u64;
    let mut pending: Vec<Vec<(usize, usize)>> = Vec::new();
    for chunk_block in blocks(&target_spatial, &chunk) {
        pending.push(chunk_block);
        while let Some(block) = pending.pop() {
            let dims: Vec<usize> = block.iter().map(|(a, b)| b - a).collect();
            let n: usize = dims.iter().product();
            let mut indices = Array2::<f64>::zeros((n, s));
            for (row, mut point) in indices.outer_iter_mut().enumerate() {
                let mut rest = row;
                for k in (0..s).rev() {
                    point[k] = (block[k].0 + rest % dims[k]) as f64;
                    rest /= dims[k];
                }
            }
            let points = mapping.map(&indices)?;
            let inside: Vec<bool> = points
                .outer_iter()
                .map(|p| p.iter().zip(&source_spatial).all(|(v, m)| *v >= -0.5 && *v <= *m as f64 - 0.5))
                .collect();
            let clamped = Array2::from_shape_fn((n, s), |(row, k)| {
                if inside[row] {
                    points[[row, k]].clamp(0.0, source_spatial[k] as f64 - 1.0)
                } else {
                    0.0
                }
            });
            // The part of the source the block's inside points fall in.
            let crop: Option<Vec<(usize, usize)>> = (0..s)
                .map(|k| {
                    let values = clamped.column(k);
                    let mut lo = f64::INFINITY;
                    let mut hi = f64::NEG_INFINITY;
                    for (row, v) in values.iter().enumerate() {
                        if inside[row] {
                            lo = lo.min(*v);
                            hi = hi.max(*v);
                        }
                    }
                    if lo > hi {
                        return None;
                    }
                    let lo = lo.floor() as usize;
                    let hi = (hi.ceil() as usize).min(source_spatial[k] - 1);
                    Some((lo, hi + 1))
                })
                .collect();
            // A chunk whose points spread over much of the source (a strong
            // downsampling, a large warp) is halved until the source it reads fits
            // the budget: one target voxel reads at most two source voxels an axis.
            if let Some(crop) = &crop {
                let source_voxels: usize = crop.iter().map(|(a, b)| b - a).product();
                if n > 1 && source_voxels.saturating_mul(group_max) > crop_budget() {
                    let (axis, _) = dims.iter().enumerate().max_by_key(|(_, d)| **d).expect("a spatial axis");
                    let middle = block[axis].0 + dims[axis] / 2;
                    let (mut first, mut second) = (block.clone(), block);
                    first[axis].1 = middle;
                    second[axis].0 = middle;
                    pending.push(second);
                    pending.push(first);
                    continue;
                }
                #[cfg(test)]
                LARGEST_CROP.with(|largest| largest.set(largest.get().max(source_voxels * group_max)));
            }
            let roi: Option<Vec<Slice>> =
                crop.as_ref().map(|c| c.iter().map(|(a, b)| Slice::new(*a as i64, *b as i64)).collect());
            let nearest_rows = |crop: &[(usize, usize)]| -> Vec<Option<Vec<usize>>> {
                (0..n)
                    .map(|row| {
                        inside[row].then(|| {
                            (0..s)
                                .map(|k| {
                                    let i = (clamped[[row, k]] + 0.5).floor() as usize;
                                    i.min(source_spatial[k] - 1) - crop[k].0
                                })
                                .collect()
                        })
                    })
                    .collect()
            };
            // Where the source holds data, for the points inside it.
            if let (Some(_), Some(crop), Some(roi)) = (&job.source_mask, &crop, &roi) {
                let region = sample.valid_region(&job.source, Some(roi))?;
                for (row, at) in nearest_rows(crop).into_iter().enumerate() {
                    if let Some(at) = at {
                        if !region[IxDyn(&at)] {
                            invalidate(&mut valid, &target_spatial, &block, &dims, row);
                        }
                    }
                }
            }
            for (row, ok) in inside.iter().enumerate() {
                if !ok {
                    filled += 1;
                    invalidate(&mut valid, &target_spatial, &block, &dims, row);
                }
            }
            let rows = crop.as_ref().map(|crop| nearest_rows(crop));
            let local = crop
                .as_ref()
                .map(|crop| Array2::from_shape_fn((n, s), |(row, k)| clamped[[row, k]] - crop[k].0 as f64));
            // Each chunk of the leading (time, channel) axes in turn, so a block
            // holds one chunk of the image and no more, however many frames.
            for lead in blocks(&leading_shape, &leading_chunk) {
                let lead_dims: Vec<usize> = lead.iter().map(|(a, b)| b - a).collect();
                let group: usize = lead_dims.iter().product();
                #[cfg(test)]
                LARGEST_BLOCK.with(|largest| largest.set(largest.get().max(n * group)));
                let roi: Option<Vec<Slice>> = roi.as_ref().map(|spatial| {
                    let mut full: Vec<Slice> = lead.iter().map(|(a, b)| Slice::new(*a as i64, *b as i64)).collect();
                    full.extend(spatial.iter().cloned());
                    full
                });
                let values: NdArray = match plan.interpolation.as_str() {
                    "nearest" => match (&rows, &roi) {
                        (Some(rows), Some(roi)) => {
                            let stored = image.read(Some(roi), false, None)?;
                            gather(&stored, group, rows, plan.fill, job.dtype)?
                        }
                        _ => constant(job.dtype, group, n, plan.fill)?,
                    },
                    interpolation => {
                        let mut out = Array2::<f64>::from_elem((n, group), plan.fill);
                        if let (Some(crop), Some(roi), Some(local)) = (&crop, &roi, &local) {
                            let computed = if interpolation == "cubic" {
                                let lanes = lane_indices(&leading_shape, &lead);
                                cubic.as_ref().expect("prefiltered above").sample_components(&clamped, &lanes)?
                            } else {
                                let stored = image.read(Some(roi), false, Some(DType::F64))?.to_f64();
                                let mut field_shape = vec![group];
                                field_shape.extend(crop.iter().map(|(a, b)| b - a));
                                let field = stored.into_shape_with_order(IxDyn(&field_shape))?;
                                linear_sample(&field, local, "nearest")?
                            };
                            for (row, ok) in inside.iter().enumerate() {
                                if *ok {
                                    out.row_mut(row).assign(&computed.row(row));
                                }
                            }
                        }
                        convert(&out, job.dtype, &job.source)?
                    }
                };
                let mut shape = lead_dims;
                shape.extend(&dims);
                let mut starts: Vec<usize> = lead.iter().map(|(a, _)| *a).collect();
                starts.extend(block.iter().map(|(a, _)| *a));
                data::write_region(&dataset, &values.reshape(&shape)?, &starts)?;
            }
        }
    }
    Ok((filled, valid))
}

/// The flat (C-order) lanes of the leading axes a block of them covers.
fn lane_indices(shape: &[usize], block: &[(usize, usize)]) -> Vec<usize> {
    let mut lanes = vec![0usize];
    for (k, (a, b)) in block.iter().enumerate() {
        lanes = lanes.iter().flat_map(|lane| (*a..*b).map(move |i| lane * shape[k] + i)).collect();
    }
    lanes
}

/// Mark one voxel of a block as holding no data, making the target-wide mask
/// the first time one does.
fn invalidate(valid: &mut Option<ArrayD<bool>>, shape: &[usize], block: &[(usize, usize)], dims: &[usize], row: usize) {
    let valid = valid.get_or_insert_with(|| ArrayD::from_elem(IxDyn(shape), true));
    let s = dims.len();
    let mut at = vec![0usize; s];
    let mut rest = row;
    for k in (0..s).rev() {
        at[k] = block[k].0 + rest % dims[k];
        rest /= dims[k];
    }
    valid[IxDyn(&at)] = false;
}

/// `(lanes, n)` values of `dtype`, all `fill`.
fn constant(dtype: DType, lanes: usize, n: usize, fill: f64) -> Result<NdArray> {
    convert(&Array2::from_elem((n, lanes), fill), dtype, "")
}

/// Nearest-neighbour values from a stored `(lanes…, crop…)` block, in its own
/// type, laid out `(lanes, n)`; `None` rows take `fill`.  Converted to `dtype`
/// only when it is not the stored type, so a nearest resample keeps every
/// stored value exactly, 64-bit integers included; and from one integer type
/// to another through `i128`, clipped, never through `f64`.
fn gather(stored: &NdArray, lanes: usize, rows: &[Option<Vec<usize>>], fill: f64, dtype: DType) -> Result<NdArray> {
    let n = rows.len();
    let crop: Vec<usize> = stored.shape()[stored.ndim() - rows.iter().flatten().next().map_or(0, Vec::len)..].to_vec();
    let mut flat_shape = vec![lanes];
    flat_shape.extend(&crop);
    let same = stored.dtype() == dtype;
    if let (false, Some(_), Some((lo, hi))) = (same, stored.dtype().int_range(), dtype.int_range()) {
        // `check_fill` has made the fill an integer the target holds.
        let fill = fill as i128;
        let exact: Vec<i128> = crate::with_array!(stored, a => {
            let a = a.view().into_shape_with_order(IxDyn(&flat_shape))?;
            let mut out = Vec::with_capacity(lanes * n);
            let mut index = vec![0usize; crop.len() + 1];
            for lane in 0..lanes {
                index[0] = lane;
                for row in rows {
                    out.push(match row {
                        Some(at) => {
                            index[1..].copy_from_slice(at);
                            crate::array::Element::to_i128(a[IxDyn(&index)]).clamp(lo, hi)
                        }
                        None => fill,
                    });
                }
            }
            out
        });
        return Ok(crate::with_dtype!(dtype, T => NdArray::from_vec(
            &[lanes, n],
            exact.into_iter().map(<T as crate::array::Element>::from_i128).collect::<Vec<T>>()
        )?));
    }
    crate::with_array!(stored, a => {
        let a = a.view().into_shape_with_order(IxDyn(&flat_shape))?;
        let fill_value = crate::array::Element::from_f64(fill);
        let mut out = Vec::with_capacity(lanes * n);
        let mut index = vec![0usize; crop.len() + 1];
        for lane in 0..lanes {
            index[0] = lane;
            for row in rows {
                match row {
                    Some(at) => {
                        index[1..].copy_from_slice(at);
                        out.push(a[IxDyn(&index)]);
                    }
                    None => out.push(fill_value),
                }
            }
        }
        let gathered = NdArray::from_vec(&[lanes, n], out)?;
        if same {
            Ok(gathered)
        } else {
            let as_rows = gathered.to_f64().reversed_axes().into_dimensionality::<ndarray::Ix2>()?;
            let mut values = as_rows.to_owned();
            for (row, at) in rows.iter().enumerate() {
                if at.is_none() {
                    values.row_mut(row).fill(fill);
                }
            }
            convert(&values, dtype, "")
        }
    })
}

/// `(n, lanes)` values as `(lanes, n)` of `dtype`: floats as they are, but
/// for a finite value beyond the type's range, which is clipped to it;
/// integers rounded half to even and clipped.  A NaN is refused for an
/// integer type, which has none.
fn convert(values: &Array2<f64>, dtype: DType, image: &str) -> Result<NdArray> {
    let lanes_first = values.t();
    let flat: Vec<f64> = lanes_first.iter().copied().collect();
    let shape = [values.ncols(), values.nrows()];
    if dtype.is_float() {
        let max = match dtype {
            DType::F16 => 65504.0,
            DType::F32 => f32::MAX as f64,
            _ => f64::MAX,
        };
        let clipped: Vec<f64> = flat.into_iter().map(|v| if v.is_finite() { v.clamp(-max, max) } else { v }).collect();
        return Ok(NdArray::from_vec(&shape, clipped)?.astype(dtype));
    }
    let (lo, hi) = dtype.int_range().expect("an integer or bool dtype");
    if flat.iter().any(|v| v.is_nan()) {
        return Err(Error::Value(format!(
            "resampling {} gives NaN, which {} cannot hold; resample to a float dtype",
            repr_str(image),
            dtype.name()
        )));
    }
    let rounded: Vec<f64> = flat.into_iter().map(|v| v.round_ties_even().clamp(lo as f64, hi as f64)).collect();
    if dtype == DType::I64 || dtype == DType::U64 {
        // Through i128, so the clipped extremes stay exact.
        let exact: Vec<i128> = rounded.iter().map(|v| (*v as i128).clamp(lo, hi)).collect();
        return Ok(crate::with_dtype!(dtype, T => NdArray::from_vec(
            &shape,
            exact.into_iter().map(<T as crate::array::Element>::from_i128).collect::<Vec<T>>()
        )?));
    }
    Ok(NdArray::from_vec(&shape, rounded)?.astype(dtype))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::writer::create;

    /// A sample with one float32 image `img` on grid `g`, and a grid `fine`
    /// at half its spacing, chunked by `chunk_hint`.
    fn sample(path: &Path, shape: [i64; 3], chunk_hint: Option<Vec<i64>>) {
        let mut w = create(path, Some("s"), Some("s"), "portable", &[]).unwrap();
        let frame = || GridOptions { frame_uid: Some("F0".into()), ..Default::default() };
        w.add_grid("g", &shape, &[1.0, 1.0, 1.0], frame()).unwrap();
        let n: usize = shape.iter().map(|v| *v as usize).product();
        let values: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dims: Vec<usize> = shape.iter().map(|v| *v as usize).collect();
        let image = NdArray::from_vec(&dims, values).unwrap();
        w.add_image("img", &image, "g", "CT", ImageOptions::default()).unwrap();
        let fine: Vec<i64> = shape.iter().map(|v| v * 2).collect();
        let options = GridOptions { origin: Some(vec![-0.25; 3]), chunk_hint, ..frame() };
        w.add_grid("fine", &fine, &[0.5, 0.5, 0.5], options).unwrap();
        w.commit(true).unwrap();
    }

    /// The work of a resample is a block of the target at a time, one chunk
    /// of the image written: what it holds follows the chunk shape, not the
    /// target's volume.
    #[test]
    fn a_large_resample_is_computed_a_chunk_at_a_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.medh5");
        sample(&path, [8, 32, 32], Some(vec![4, 16, 16]));
        LARGEST_BLOCK.with(|largest| largest.set(0));
        let result = resample(&path, &ResamplePlan::onto(["img"], "fine")).unwrap();
        let volume = 16 * 64 * 64;
        let largest = LARGEST_BLOCK.with(|largest| largest.get());
        assert_eq!(largest, 4 * 16 * 16, "the target has {volume} voxels");
        assert_eq!(result.filled_voxels, 0);
        // The blocks put together are the whole image.
        let sample = open_sample(&path).unwrap();
        let written = sample.image("img_on_fine").unwrap();
        assert_eq!(written.chunks().unwrap(), Some(vec![4, 16, 16]));
        let values = written.read(None, false, Some(DType::F64)).unwrap().to_f64();
        // Each fine voxel lies a quarter voxel inside a source voxel, so a
        // linear sample of an index ramp is the ramp at the fine centre.
        let ramp = |z: f64, y: f64, x: f64| z * 32.0 * 32.0 + y * 32.0 + x;
        for &(z, y, x) in &[(0usize, 0usize, 0usize), (5, 17, 40), (15, 63, 63), (9, 0, 33)] {
            let centre = |i: usize, n: usize| ((i as f64) * 0.5 - 0.25).clamp(0.0, n as f64 - 1.0);
            let want = ramp(centre(z, 8), centre(y, 32), centre(x, 32));
            assert!((values[[z, y, x]] - want).abs() < 1e-3, "{z} {y} {x}: {} vs {want}", values[[z, y, x]]);
        }
    }

    /// The leading (time) axes are chunked too: a block holds one frame of
    /// one spatial chunk, not every frame of it.
    #[test]
    fn a_time_series_is_resampled_a_frame_chunk_at_a_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.medh5");
        let mut w = create(&path, Some("s"), Some("s"), "portable", &[]).unwrap();
        let options = |origin: f64, chunk_hint: Option<Vec<i64>>| GridOptions {
            frame_uid: Some("F0".into()),
            axis_kinds: Some(vec!["time".into(), "spatial".into(), "spatial".into(), "spatial".into()]),
            time_values: Some(vec![0.0, 1.0, 2.0]),
            origin: Some(vec![origin; 3]),
            chunk_hint,
            ..Default::default()
        };
        w.add_grid("g", &[3, 4, 16, 16], &[1.0, 1.0, 1.0], options(0.0, None)).unwrap();
        let values: Vec<f32> = (0..3 * 4 * 16 * 16).map(|i| i as f32).collect();
        let image = NdArray::from_vec(&[3, 4, 16, 16], values).unwrap();
        w.add_image("img", &image, "g", "MR", ImageOptions::default()).unwrap();
        let fine = options(-0.25, Some(vec![1, 4, 16, 16]));
        w.add_grid("fine", &[3, 8, 32, 32], &[0.5, 0.5, 0.5], fine).unwrap();
        w.commit(true).unwrap();
        for interpolation in ["nearest", "linear", "cubic"] {
            LARGEST_BLOCK.with(|largest| largest.set(0));
            let mut plan = ResamplePlan::onto(["img"], "fine");
            plan.interpolation = interpolation.into();
            plan.suffix = Some(format!("_{interpolation}"));
            resample(&path, &plan).unwrap();
            assert_eq!(LARGEST_BLOCK.with(|largest| largest.get()), 4 * 16 * 16, "{interpolation}");
            let sample = open_sample(&path).unwrap();
            let written = sample.image(&format!("img_{interpolation}")).unwrap();
            let values = written.read(None, false, Some(DType::F64)).unwrap().to_f64();
            // Every frame is its own: frame t is frame 0 shifted by t·1024,
            // which nearest, linear and cubic all preserve exactly.
            for &(z, y, x) in &[(0usize, 0usize, 0usize), (5, 17, 30), (7, 31, 31)] {
                for t in 1..3 {
                    let shift = values[[t, z, y, x]] - values[[0, z, y, x]];
                    assert!((shift - 1024.0 * t as f64).abs() < 1e-2, "{interpolation} {t}: {shift}");
                }
            }
        }
    }

    /// A strong downsampling spreads one target chunk over the whole
    /// source; the chunk is split until what it reads fits the budget, and
    /// the result is the same.
    #[test]
    fn a_chunk_spread_over_the_source_is_split_to_the_budget() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.medh5");
        let mut w = create(&path, Some("s"), Some("s"), "portable", &[]).unwrap();
        let frame = || GridOptions { frame_uid: Some("F0".into()), ..Default::default() };
        w.add_grid("g", &[32, 32, 32], &[1.0, 1.0, 1.0], frame()).unwrap();
        let values: Vec<f32> = (0..32 * 32 * 32).map(|i| ((i * 7919) % 1000) as f32).collect();
        w.add_image("img", &NdArray::from_vec(&[32, 32, 32], values).unwrap(), "g", "CT", ImageOptions::default())
            .unwrap();
        let coarse = GridOptions { origin: Some(vec![3.5; 3]), ..frame() };
        w.add_grid("coarse", &[4, 4, 4], &[8.0, 8.0, 8.0], coarse).unwrap();
        w.commit(true).unwrap();
        for interpolation in ["nearest", "linear"] {
            let mut results = Vec::new();
            for budget in [CROP_BUDGET, 64] {
                TEST_CROP_BUDGET.with(|b| b.set(budget));
                LARGEST_CROP.with(|largest| largest.set(0));
                let mut plan = ResamplePlan::onto(["img"], "coarse");
                plan.interpolation = interpolation.into();
                plan.suffix = Some(format!("_{interpolation}_{budget}"));
                resample(&path, &plan).unwrap();
                let read = LARGEST_CROP.with(|largest| largest.get());
                assert!(read <= budget, "{interpolation}: read {read} source voxels, budget {budget}");
                let sample = open_sample(&path).unwrap();
                let image = sample.image(&format!("img_{interpolation}_{budget}")).unwrap();
                results.push(image.read(None, false, Some(DType::F64)).unwrap().to_f64());
            }
            TEST_CROP_BUDGET.with(|b| b.set(CROP_BUDGET));
            assert_eq!(results[0], results[1], "{interpolation}");
        }
    }

    /// A resample in place through a transform closes everything it opened
    /// on the source before the amendment replaces it: Windows cannot
    /// replace an open file, and the resolved transform held one.
    #[test]
    fn s14_4_the_source_is_closed_before_the_replace() {
        use crate::sample::writer_annotations::TransformSpec;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.medh5");
        let mut w = create(&path, Some("s"), Some("s"), "portable", &[]).unwrap();
        let frame = |uid: &str| GridOptions { frame_uid: Some(uid.into()), ..Default::default() };
        w.add_grid("g", &[4, 4, 4], &[1.0, 1.0, 1.0], frame("F0")).unwrap();
        let image = NdArray::from_vec(&[4, 4, 4], vec![1.0f32; 64]).unwrap();
        w.add_image("img", &image, "g", "CT", ImageOptions::default()).unwrap();
        w.add_grid("h", &[4, 4, 4], &[1.0, 1.0, 1.0], frame("F1")).unwrap();
        let identity = TransformSpec { matrix: Some(Array2::<f64>::eye(4).into_dyn()), ..Default::default() };
        w.add_transform("t", "affine", "F1", "F0", identity).unwrap();
        w.commit(true).unwrap();
        let seen = std::rc::Rc::new(std::cell::Cell::new(None));
        let record = seen.clone();
        crate::h5::file::before_rename(Some(Box::new(move |target: &Path| {
            record.set(Some(crate::h5::file::held_on(target)))
        })));
        let result = resample(&path, &ResamplePlan::onto(["img"], "h"));
        crate::h5::file::before_rename(None);
        result.unwrap();
        assert_eq!(seen.get(), Some(0), "the source was still open at the rename");
    }

    #[test]
    fn s3_2_equal_units_compare_times_exactly() {
        let grid = |units: Option<&str>, values: Vec<f64>| {
            Grid::from_spec(crate::geometry::grid::GridSpec {
                grid_id: "g".into(),
                shape: vec![values.len() as i64, 2, 2],
                spacing: vec![1.0, 1.0],
                axis_kinds: Some(vec!["time".into(), "spatial".into(), "spatial".into()]),
                time_values: Some(values),
                time_units: units.map(String::from),
                ..Default::default()
            })
            .unwrap()
        };
        let big = 1.0e9;
        assert!(same_times(&grid(None, vec![big]), &grid(Some("s"), vec![big])));
        assert!(!same_times(&grid(None, vec![big]), &grid(None, vec![big + 1.0])));
        assert!(!same_times(&grid(None, vec![big]), &grid(Some("ms"), vec![big * 1e3 + 1.0])));
        assert!(same_times(&grid(None, vec![big, 0.1]), &grid(Some("ms"), vec![big * 1e3, 100.0])));
    }

    #[test]
    fn leading_blocks_name_their_flat_lanes() {
        assert_eq!(lane_indices(&[], &[]), vec![0]);
        assert_eq!(lane_indices(&[4, 3], &[(1, 3), (0, 1)]), vec![3, 6]);
        assert_eq!(lane_indices(&[4, 3], &[(2, 3), (1, 3)]), vec![7, 8]);
    }

    #[test]
    fn s3_3_a_respaced_grid_keeps_the_source_edges() {
        let source = Grid::from_spec(crate::geometry::grid::GridSpec {
            grid_id: "g".into(),
            shape: vec![2, 3, 5],
            spacing: vec![1.0, 0.3, 1.0],
            origin: Some(vec![10.0, 0.0, -2.0]),
            ..Default::default()
        })
        .unwrap();
        let grid = respaced(&source, "c", &[2.0, 0.6, 3.0]).unwrap();
        // 2·1/2 = 1, 3·0.3/0.6 = 1.5 -> 2, 5·1/3 = 1.67 -> 2; and 2·0.3/0.6
        // computed in floating point is just over 1, which is still 1.
        assert_eq!(grid.shape, vec![1, 2, 2]);
        for k in 0..3 {
            let edge = source.origin[k] - source.spacing[k] / 2.0;
            assert!((grid.origin[k] - grid.spacing[k] / 2.0 - edge).abs() < 1e-12);
        }
        let exact = respaced(&source, "d", &[1.0, 0.15, 1.0]).unwrap();
        assert_eq!(exact.shape, vec![2, 6, 5]);
    }

    #[test]
    fn blocks_cover_the_shape_once() {
        let covered: Vec<_> = blocks(&[5, 3], &[2, 3]).collect();
        assert_eq!(covered, vec![vec![(0, 2), (0, 3)], vec![(2, 4), (0, 3)], vec![(4, 5), (0, 3)]]);
        assert_eq!(blocks(&[4], &[9]).collect::<Vec<_>>(), vec![vec![(0, 4)]]);
        let fine: Vec<_> = blocks(&[2, 3], &[1, 2]).collect();
        assert_eq!(fine, vec![vec![(0, 1), (0, 2)], vec![(0, 1), (2, 3)], vec![(1, 2), (0, 2)], vec![(1, 2), (2, 3)]]);
        assert_eq!(blocks(&[0, 3], &[1, 1]).count(), 0);
    }

    /// A nearest gather from one integer type to another keeps every value
    /// the target holds exactly, beyond 2^53 too, and clips the rest.
    #[test]
    fn integers_change_type_without_a_float_between() {
        let big = (1u64 << 53) + 1;
        let stored = NdArray::from_vec(&[3], vec![big, u64::MAX, 7]).unwrap();
        let rows = vec![Some(vec![0]), Some(vec![1]), None, Some(vec![2])];
        let out = gather(&stored, 1, &rows, -4.0, DType::I64).unwrap();
        assert_eq!(out.cast::<i64>().iter().copied().collect::<Vec<_>>(), vec![big as i64, i64::MAX, -4, 7]);
        let small = gather(&stored, 1, &rows, 0.0, DType::U8).unwrap();
        assert_eq!(small.cast::<u8>().iter().copied().collect::<Vec<_>>(), vec![255, 255, 0, 7]);
    }

    #[test]
    fn integers_round_half_to_even_and_clip_and_refuse_nan() {
        let values = Array2::from_shape_vec((5, 1), vec![0.5, 1.5, 2.5, -3.0, 1e9]).unwrap();
        let out = convert(&values, DType::I16, "x").unwrap();
        assert_eq!(out.to_f64().iter().copied().collect::<Vec<_>>(), vec![0.0, 2.0, 2.0, -3.0, 32767.0]);
        let big = Array2::from_shape_vec((1, 1), vec![1e30]).unwrap();
        assert_eq!(convert(&big, DType::U64, "x").unwrap().cast::<u64>()[[0, 0]], u64::MAX);
        let nan = Array2::from_shape_vec((1, 1), vec![f64::NAN]).unwrap();
        assert!(convert(&nan, DType::U8, "x").unwrap_err().to_string().contains("NaN"));
        assert!(convert(&nan, DType::F32, "x").unwrap().to_f64()[[0, 0]].is_nan());
    }
}
