//! Streaming cohort statistics: intensity moments and class frequencies.
//!
//! One pass per file, constant memory per worker, and an exact merge.  Means
//! are not averaged per file (the Welford/Chan merge weights by voxel count),
//! an unexamined class is not a zero (§11.3), class statistics come from
//! voxel annotations only, and moments are over the voxels an image holds data
//! in (`valid_mask`, §4.4).  Intensities are **physical** by default --- what
//! the loaders hand a model --- and `physical = false` measures stored values.
//!
//! The moments are over finite values.  A NaN or an infinity (PET outside its
//! field of view, resampled data) is counted, in `nonfinite`, and left out:
//! folded in, it made every moment NaN, the minimum and maximum depend on the
//! order of the files, and a statistics file written with NaN as `null`
//! reloaded as a mean of 0 and a standard deviation of 1.
//!
//! The arithmetic over finite values is NumPy's, operation for operation
//! (pairwise sums, the same merge formula evaluated in the same order), so the
//! numbers are the ones 1.x computed.  An image is read a chunk's planes at a
//! time and measured plane by plane, as 1.x measured it, so reading more than
//! one plane per read changes no bit of the result.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use ndarray::{ArrayViewD, Axis, IxDyn};
use serde_json::{json, Map, Value};

use crate::array::Slice;
use crate::collection::{open_any, AnyFile};
use crate::json::{num, repr_int_list, repr_str};
use crate::numeric::pairwise_sum;
use crate::sample::{open_sample, Sample};
use crate::{Error, Result};

/// Count, mean and M2 for one image key --- a Welford accumulator over the
/// finite values folded in.
#[derive(Debug, Clone, PartialEq)]
pub struct Moments {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
    pub minimum: f64,
    pub maximum: f64,
    /// Values left out of the moments because they are NaN or infinite.
    pub nonfinite: u64,
}

impl Default for Moments {
    fn default() -> Self {
        Moments { count: 0, mean: 0.0, m2: 0.0, minimum: f64::INFINITY, maximum: f64::NEG_INFINITY, nonfinite: 0 }
    }
}

/// A finite number a statistics document holds under `name`, or why not.
fn finite(doc: &Value, name: &str, count: u64) -> Result<f64> {
    match doc.get(name).and_then(Value::as_f64) {
        Some(v) if v.is_finite() => Ok(v),
        _ => Err(Error::Value(format!(
            "statistics over {count} value(s) hold no {name} ({}): a moment that was not a number was written as \
             null, and reading it as 0 normalised with (0, 1) --- recompute the statistics",
            match doc.get(name) {
                None => "missing".into(),
                Some(Value::Null) => "null".into(),
                Some(other) => crate::json::repr(other),
            }
        ))),
    }
}

impl Moments {
    /// Fold in one block of values; a NaN or an infinity is counted in
    /// `nonfinite`, not folded in.
    pub fn update(&mut self, block: &[f64]) {
        let values: Cow<'_, [f64]> = if block.iter().all(|v| v.is_finite()) {
            Cow::Borrowed(block)
        } else {
            Cow::Owned(block.iter().copied().filter(|v| v.is_finite()).collect())
        };
        self.nonfinite += (block.len() - values.len()) as u64;
        if values.is_empty() {
            return;
        }
        let n = values.len() as f64;
        let mean = pairwise_sum(&values) / n;
        let squares: Vec<f64> = values.iter().map(|v| (v - mean) * (v - mean)).collect();
        self.merge(&Moments {
            count: values.len() as u64,
            mean,
            m2: pairwise_sum(&squares),
            minimum: values.iter().copied().fold(f64::INFINITY, f64::min),
            maximum: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            nonfinite: 0,
        });
    }

    /// Chan-Golub-LeVeque parallel merge --- exact, not an approximation.
    pub fn merge(&mut self, other: &Moments) {
        let nonfinite = self.nonfinite + other.nonfinite;
        if other.count > 0 && self.count == 0 {
            *self = other.clone();
        } else if other.count > 0 {
            let total = self.count + other.count;
            let delta = other.mean - self.mean;
            self.mean += delta * other.count as f64 / total as f64;
            self.m2 += other.m2 + delta * delta * self.count as f64 * other.count as f64 / total as f64;
            self.count = total;
            // Python's builtin `min`/`max`: the first argument unless the
            // second is strictly better --- and a NaN, which no comparison
            // replaces, is replaced, so the result does not depend on which
            // operand came first.
            if other.minimum < self.minimum || self.minimum.is_nan() {
                self.minimum = other.minimum;
            }
            if other.maximum > self.maximum || self.maximum.is_nan() {
                self.maximum = other.maximum;
            }
        }
        self.nonfinite = nonfinite;
    }

    pub fn std(&self) -> f64 {
        if self.count > 1 {
            (self.m2 / self.count as f64).sqrt()
        } else {
            0.0
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "count": self.count,
            "mean": num(self.mean),
            "std": num(self.std()),
            "min": if self.count == 0 { Value::Null } else { num(self.minimum) },
            "max": if self.count == 0 { Value::Null } else { num(self.maximum) },
            "nonfinite": self.nonfinite,
        })
    }

    /// The moments a statistics document describes.  Statistics over any
    /// value need every moment: `null` there is a NaN written as JSON, and
    /// reading it as 0 turned a normalisation into (0, 1) without a word.
    pub fn from_json(doc: &Value) -> Result<Moments> {
        let count = doc.get("count").and_then(Value::as_u64).ok_or_else(|| Error::Key("'count'".into()))?;
        let nonfinite = doc.get("nonfinite").and_then(Value::as_u64).unwrap_or(0);
        if count == 0 {
            return Ok(Moments { nonfinite, ..Default::default() });
        }
        let std = finite(doc, "std", count)?;
        Ok(Moments {
            count,
            mean: finite(doc, "mean", count)?,
            m2: std * std * count as f64,
            minimum: finite(doc, "min", count)?,
            maximum: finite(doc, "max", count)?,
            nonfinite,
        })
    }
}

/// How often a class occurs, over the samples that examined it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClassStats {
    pub class_id: i64,
    /// What the id means: its key in the label set of the samples measured,
    /// `None` where none of them declares one (no label set, or one by
    /// reference).  A class id is local to its label set, so statistics
    /// that would count one id under two keys are refused.
    pub key: Option<String>,
    pub voxels: u64,
    /// Samples that examined the class and found it (a longitudinal sample
    /// counts once).  A class found where nobody looked for it says nothing
    /// about how often it occurs (§11.3): counted, it made the prevalence of
    /// a class examined in one sample and drawn in three 3.0.
    pub present_in: u64,
    /// Samples that looked for the class.
    pub examined_in: u64,
}

impl ClassStats {
    pub fn new(class_id: i64) -> ClassStats {
        ClassStats { class_id, ..Default::default() }
    }

    /// Fraction of *examining* samples in which the class was found; `None`
    /// when no sample examined it, which says nothing about how often it
    /// occurs (§11.3) --- 0.0 read as "looked for everywhere, found nowhere".
    pub fn examined_prevalence(&self) -> Option<f64> {
        (self.examined_in > 0).then(|| self.present_in as f64 / self.examined_in as f64)
    }

    /// Fraction of *examining* samples in which the class was found, 0.0
    /// when none examined it.
    #[deprecated(
        since = "2.1.0",
        note = "0.0 for a class no sample examined; use examined_prevalence; Option<f64> in 3.0"
    )]
    pub fn prevalence(&self) -> f64 {
        self.examined_prevalence().unwrap_or(0.0)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "class_id": self.class_id,
            "key": self.key,
            "voxels": self.voxels,
            "present_in": self.present_in,
            "examined_in": self.examined_in,
            // `null` for a class nobody examined: a 0.0 here recorded a
            // measurement that was never made.
            "prevalence": self.examined_prevalence().map(num),
        })
    }
}

/// Why statistics that give one class id two keys are refused.
const CLASH: &str = "their label sets give the id two meanings, and statistics over both would count one class's \
                     voxels as the other's, under one weight --- measure each label set's samples apart, or relabel \
                     them to one vocabulary (`medh5 dataset check` reports the clash as C102)";

/// Everything one streaming pass over a cohort learned.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetStats {
    pub samples: u64,
    pub images: IndexMap<String, Moments>,
    pub classes: IndexMap<i64, ClassStats>,
    pub total_voxels: u64,
    pub failures: Vec<String>,
    /// Whether the image moments are over physical values (rescale applied).
    pub physical: bool,
}

impl Default for DatasetStats {
    fn default() -> Self {
        DatasetStats {
            samples: 0,
            images: IndexMap::new(),
            classes: IndexMap::new(),
            total_voxels: 0,
            failures: Vec::new(),
            physical: true,
        }
    }
}

impl DatasetStats {
    /// Fold another pass in; refused, and nothing changed, when the two do
    /// not describe one thing.
    pub fn merge(&mut self, other: &DatasetStats) -> Result<()> {
        if other.samples > 0 {
            // One pass, one convention: a mean over physical values merged
            // with one over stored values describes no image anybody can load.
            if self.samples > 0 && other.physical != self.physical {
                return Err(Error::File(
                    "cannot merge statistics over physical values with statistics over stored values; compute both \
                     passes the same way"
                        .into(),
                ));
            }
        }
        if let Some((class_id, mine, theirs)) = self.key_clash(other) {
            return Err(Error::invalid(format!(
                "class {class_id} is {} in these statistics and {} in the ones merged in: {CLASH}",
                repr_str(&mine),
                repr_str(&theirs)
            )));
        }
        if other.samples > 0 {
            self.physical = other.physical;
        }
        self.samples += other.samples;
        self.total_voxels = self.total_voxels.saturating_add(other.total_voxels);
        self.failures.extend(other.failures.iter().cloned());
        for (key, moments) in &other.images {
            self.images.entry(key.clone()).or_default().merge(moments);
        }
        for (class_id, stats) in &other.classes {
            let mine = self.classes.entry(*class_id).or_insert_with(|| ClassStats::new(*class_id));
            if mine.key.is_none() {
                mine.key.clone_from(&stats.key);
            }
            mine.voxels = mine.voxels.saturating_add(stats.voxels);
            mine.present_in += stats.present_in;
            mine.examined_in += stats.examined_in;
        }
        Ok(())
    }

    /// The first class id `other` gives another key than these statistics
    /// do, with the two keys.  A class id means its key in its label set:
    /// binned by the id alone, "liver" in one label set and "spleen" in
    /// another were one class, whose counts fed one loss weight.  An id
    /// without a key (no label set, or one by reference) says nothing either
    /// way.
    fn key_clash(&self, other: &DatasetStats) -> Option<(i64, String, String)> {
        other.classes.iter().find_map(|(class_id, theirs)| {
            let mine = self.classes.get(class_id)?.key.as_deref()?;
            let theirs = theirs.key.as_deref()?;
            (mine != theirs).then(|| (*class_id, mine.to_string(), theirs.to_string()))
        })
    }

    /// `(mean, std)` for a z-score transform, or `(0, 1)` if unseen.
    pub fn normalization(&self, image_key: &str) -> (f64, f64) {
        match self.images.get(image_key) {
            Some(m) if m.count > 0 => (m.mean, if m.std() == 0.0 { 1.0 } else { m.std() }),
            _ => (0.0, 1.0),
        }
    }

    /// Classes with no voxels in these statistics: they get no loss weight.
    pub fn unweighted_classes(&self) -> Vec<i64> {
        let mut empty: Vec<i64> = self.classes.iter().filter(|(_, s)| s.voxels == 0).map(|(c, _)| *c).collect();
        empty.sort_unstable();
        empty
    }

    /// The warning [`class_weights`](Self::class_weights) has for a caller,
    /// when some class gets no weight.
    pub fn class_weights_warning(&self) -> Option<String> {
        let empty = self.unweighted_classes();
        if empty.is_empty() {
            return None;
        }
        Some(format!(
            "class_weights: class(es) {} have no voxels in these statistics, so they get no weight; a weight for an \
             unseen class is a guess, and the inverse of zero is not one",
            repr_int_list(&empty)
        ))
    }

    /// Loss weights from measured voxel frequencies, normalised to a mean of
    /// 1 so switching schemes does not silently rescale the learning rate.
    pub fn class_weights(&self, scheme: &str) -> Result<BTreeMap<i64, f64>> {
        let counts: Vec<(i64, u64)> =
            self.classes.iter().filter(|(_, s)| s.voxels > 0).map(|(c, s)| (*c, s.voxels)).collect();
        if counts.is_empty() {
            return Ok(BTreeMap::new());
        }
        let raw: Vec<(i64, f64)> = match scheme {
            "inverse_frequency" => counts.iter().map(|(c, v)| (*c, 1.0 / *v as f64)).collect(),
            "inverse_sqrt" => counts.iter().map(|(c, v)| (*c, 1.0 / (*v as f64).sqrt())).collect(),
            "uniform" => counts.iter().map(|(c, _)| (*c, 1.0)).collect(),
            other => {
                return Err(Error::File(format!(
                    "unknown weighting scheme {}; expected inverse_frequency, inverse_sqrt or uniform",
                    repr_str(other)
                )))
            }
        };
        let total = raw.iter().fold(0.0, |acc, (_, v)| acc + v);
        let scale = raw.len() as f64 / total;
        Ok(raw.into_iter().map(|(c, v)| (c, v * scale)).collect())
    }

    pub fn to_json(&self) -> Value {
        let mut images: Vec<(&String, &Moments)> = self.images.iter().collect();
        images.sort_by(|a, b| a.0.cmp(b.0));
        let mut classes: Vec<&ClassStats> = self.classes.values().collect();
        classes.sort_by_key(|s| s.class_id);
        json!({
            "samples": self.samples,
            "physical": self.physical,
            "total_voxels": self.total_voxels,
            "images": images.into_iter().map(|(k, v)| (k.clone(), v.to_json())).collect::<Map<String, Value>>(),
            "classes": classes.iter().map(|s| s.to_json()).collect::<Vec<_>>(),
            "failures": self.failures,
        })
    }

    pub fn from_json(doc: &Value) -> Result<DatasetStats> {
        let mut images = IndexMap::new();
        if let Some(Value::Object(m)) = doc.get("images") {
            for (k, v) in m {
                let moments = Moments::from_json(v).map_err(|e| match e {
                    Error::Value(why) => Error::Value(format!("image {}: {why}", repr_str(k))),
                    other => other,
                })?;
                images.insert(k.clone(), moments);
            }
        }
        let mut classes = IndexMap::new();
        if let Some(Value::Array(items)) = doc.get("classes") {
            for c in items {
                let id = c.get("class_id").and_then(Value::as_i64).ok_or_else(|| Error::Key("'class_id'".into()))?;
                let count = |k: &str| c.get(k).and_then(Value::as_u64).unwrap_or(0);
                classes.insert(
                    id,
                    ClassStats {
                        class_id: id,
                        key: c.get("key").and_then(Value::as_str).map(str::to_string),
                        voxels: count("voxels"),
                        present_in: count("present_in"),
                        examined_in: count("examined_in"),
                    },
                );
            }
        }
        Ok(DatasetStats {
            samples: doc.get("samples").and_then(Value::as_u64).unwrap_or(0),
            physical: doc.get("physical").and_then(Value::as_bool).unwrap_or(true),
            total_voxels: doc.get("total_voxels").and_then(Value::as_u64).unwrap_or(0),
            images,
            classes,
            failures: doc
                .get("failures")
                .and_then(Value::as_array)
                .map(|f| f.iter().map(crate::json::py_str).collect())
                .unwrap_or_default(),
        })
    }
}

/// What to measure in each file.
#[derive(Debug, Clone)]
pub struct StatsOptions {
    /// Image ids to include; `None` for every image.
    pub images: Option<Vec<String>>,
    /// Annotation ids to include; `None` for every annotation.
    pub annotations: Option<Vec<String>>,
    /// Read every Nth slab along the first axis (approximate, opt-in).
    pub sample_stride: usize,
    pub physical: bool,
}

impl Default for StatsOptions {
    fn default() -> Self {
        StatsOptions { images: None, annotations: None, sample_stride: 1, physical: true }
    }
}

impl StatsOptions {
    /// Refuse what no read could honour: a stride of 0 read every plane as a
    /// stride of 1 does, and said nothing.
    fn check(&self) -> Result<()> {
        if self.sample_stride == 0 {
            return Err(Error::invalid("sample_stride must be at least 1: it reads every Nth plane"));
        }
        Ok(())
    }
}

/// A sample by its path, or by `path::key` for a member of a collection ---
/// the locator a split names it by (`Split::paths`), so a partition's list
/// can be measured as it is.  A file whose own name holds `::` is still
/// opened by that name.
fn open_located(path: &Path) -> Result<Sample> {
    if !path.exists() {
        if let Some((file, key)) = path.to_string_lossy().rsplit_once("::") {
            if !key.is_empty() && Path::new(file).is_file() {
                return match open_any(Path::new(file), Some(key))? {
                    AnyFile::Sample(sample) => Ok(sample),
                    AnyFile::Collection(_) => {
                        Err(Error::File(format!("{} names a collection, not a sample in it", repr_str(file))))
                    }
                };
            }
        }
    }
    open_sample(path)
}

/// One file's contribution, read slab by slab rather than whole.
///
/// `path` is a file or, for a member of a collection, `path::key`.
pub fn stats_for(path: &Path, options: &StatsOptions) -> Result<DatasetStats> {
    options.check()?;
    let mut out = DatasetStats { samples: 1, physical: options.physical, ..Default::default() };
    let sample = open_located(path)?;
    let mut measured: BTreeSet<String> = BTreeSet::new();
    for (key, image) in sample.images()? {
        if options.images.as_ref().is_some_and(|wanted| !wanted.contains(key)) {
            continue;
        }
        let moments = out.images.entry(key.clone()).or_default();
        for_each_block(&sample, key, options.sample_stride, options.physical, |block| moments.update(block))?;
        measured.insert(image.grid_id()?);
    }
    let fresh = sample.fresh_indices()?.clone();
    let mut examined: BTreeSet<i64> = BTreeSet::new();
    let mut present: BTreeSet<i64> = BTreeSet::new();
    for (key, annotation) in sample.annotations()? {
        if options.annotations.as_ref().is_some_and(|wanted| !wanted.contains(key)) {
            continue;
        }
        // Voxel classes only: a `mask` has none, and a classification or a
        // geometric annotation names classes it holds no voxels of.
        if !annotation.is_voxel() || annotation.kind() == "mask" {
            continue;
        }
        let counts: IndexMap<i64, u64> = if fresh.contains(key) {
            match sample.index()?.get(key) {
                Some(index) => index.voxel_counts()?.into_iter().map(|(c, n)| (c, n.max(0) as u64)).collect(),
                None => IndexMap::new(),
            }
        } else {
            annotation.voxel_counts(None)?
        };
        let looked_for: BTreeSet<i64> = annotation.annotated_class_ids().iter().copied().collect();
        examined.extend(looked_for.iter().copied());
        let mut classes: BTreeSet<i64> = annotation.class_ids().iter().copied().collect();
        classes.extend(looked_for.iter().copied());
        let label_set = annotation.label_set();
        for class_id in classes {
            let voxels = counts.get(&class_id).copied().unwrap_or(0);
            let stats = out.classes.entry(class_id).or_insert_with(|| ClassStats::new(class_id));
            if stats.key.is_none() {
                stats.key = label_set.and_then(|l| l.by_id(class_id)).map(|c| c.key.clone());
            }
            // Saturating: an index's counts are any `int64` its file states,
            // and three of `2**63 - 1` wrapped the sum.
            stats.voxels = stats.voxels.saturating_add(voxels);
            if voxels > 0 {
                present.insert(class_id);
            }
        }
    }
    // Once per sample, not once per annotation.
    for class_id in present.intersection(&examined) {
        out.classes.entry(*class_id).or_insert_with(|| ClassStats::new(*class_id)).present_in += 1;
    }
    for class_id in examined {
        out.classes.entry(class_id).or_insert_with(|| ClassStats::new(class_id)).examined_in += 1;
    }
    // Each measured image's own grid, once.
    for grid_id in measured {
        // A grid holds at most `2**63 - 1` voxels (§3.1); a cohort's sum
        // saturates rather than wraps.
        let voxels = sample.grid(&grid_id)?.spatial_shape().iter().map(|v| *v as u64).product::<u64>();
        out.total_voxels = out.total_voxels.saturating_add(voxels);
    }
    Ok(out)
}

/// The most bytes of `f64` one read of an image holds.
const SLAB_BYTES: usize = 32 << 20;

/// The planes of an image along its first spatial axis, valid voxels only,
/// one at a time; every `stride`th plane.
///
/// A read covers a chunk's extent along that axis (fewer planes where they
/// would pass [`SLAB_BYTES`]), so a chunk is decompressed once, where reading
/// plane by plane decompressed it once for every plane it holds.  The planes
/// are still visited one by one and in order, so the moments are, to the
/// bit, the ones a plane-by-plane read computes.  Memory stays at one read.
fn for_each_block(
    sample: &Sample,
    key: &str,
    stride: usize,
    physical: bool,
    mut visit: impl FnMut(&[f64]),
) -> Result<()> {
    let image = sample.image(key)?;
    let spatial = image.grid()?.spatial_shape();
    if spatial.is_empty() {
        return Ok(());
    }
    let masked = image.valid_mask()?.is_some();
    // The spatial axes trail (§3.1), so the first of them is this one.
    let shape = image.shape()?;
    let axis = shape.len().saturating_sub(spatial.len());
    // Saturating: an image's extents are any its file states, and a plane
    // past `usize` is a plane over the slab, read one at a time.
    let plane = shape.iter().enumerate().filter(|(i, _)| *i != axis).fold(1usize, |n, (_, e)| n.saturating_mul(*e));
    let fits = (SLAB_BYTES / plane.max(1).saturating_mul(std::mem::size_of::<f64>())).max(1);
    let per_read = image.chunks()?.and_then(|c| c.get(axis).copied()).unwrap_or(fits).clamp(1, fits);
    let stride = stride.max(1);
    for start in (0..spatial[0]).step_by(per_read) {
        let stop = (start + per_read).min(spatial[0]);
        let planes: Vec<usize> = (start.div_ceil(stride) * stride..stop).step_by(stride).collect();
        let (Some(&first), Some(&last)) = (planes.first(), planes.last()) else { continue };
        let mut roi = vec![Slice::new(first as i64, last as i64 + 1)];
        roi.extend(std::iter::repeat_n(Slice::full(), spatial.len() - 1));
        let block = image.read(Some(&roi), physical, None)?.to_f64();
        let valid = if masked { Some(sample.valid_region(key, Some(&roi))?) } else { None };
        for at in planes.iter().map(|p| p - first) {
            let values = block.index_axis(Axis(axis), at);
            match &valid {
                Some(valid) => visit(&select(values, valid.index_axis(Axis(0), at))?),
                None => visit(&values.iter().copied().collect::<Vec<_>>()),
            }
        }
    }
    Ok(())
}

/// `block[np.broadcast_to(valid, block.shape)]`.
fn select(block: ArrayViewD<'_, f64>, valid: ArrayViewD<'_, bool>) -> Result<Vec<f64>> {
    let mask = valid.broadcast(IxDyn(block.shape())).ok_or_else(|| {
        Error::Value(format!(
            "operands could not be broadcast together with remapped shapes [original->remapped]: {:?} and requested \
             shape {:?}",
            valid.shape(),
            block.shape()
        ))
    })?;
    Ok(block.iter().zip(mask.iter()).filter(|(_, keep)| **keep).map(|(v, _)| *v).collect())
}

/// One file, with failure recorded rather than raised.
fn guarded(path: &Path, options: &StatsOptions) -> Result<DatasetStats> {
    match stats_for(path, options) {
        Ok(s) => Ok(s),
        Err(e) if e.is_medh5() || matches!(e, Error::Io(_)) => Ok(DatasetStats {
            failures: vec![format!("{}: {}", path.display(), e.python_str())],
            physical: options.physical,
            ..Default::default()
        }),
        Err(e) => Err(e),
    }
}

/// Fold one file's statistics into the cohort's; a class id the file gives
/// another key than the files before it is refused, naming the file.
fn merge_file(total: &mut DatasetStats, stats: &DatasetStats, path: &Path) -> Result<()> {
    if let Some((class_id, mine, theirs)) = total.key_clash(stats) {
        return Err(Error::invalid(format!(
            "class {class_id} is {} in {} and {} in the files before it: {CLASH}",
            repr_str(&theirs),
            path.display(),
            repr_str(&mine)
        )));
    }
    total.merge(stats)
}

/// Stream a whole cohort, optionally across `workers` threads.
///
/// Workers return accumulators, not arrays, and the results are merged in
/// path order, so the statistics do not depend on the worker count.
pub fn compute_stats<P: AsRef<Path> + Sync>(
    paths: &[P],
    options: &StatsOptions,
    workers: usize,
) -> Result<DatasetStats> {
    options.check()?;
    let mut total = DatasetStats { physical: options.physical, ..Default::default() };
    if workers <= 1 || paths.len() <= 1 {
        for path in paths {
            merge_file(&mut total, &guarded(path.as_ref(), options)?, path.as_ref())?;
        }
        return Ok(total);
    }
    let owned: Vec<PathBuf> = paths.iter().map(|p| p.as_ref().to_path_buf()).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Option<Result<DatasetStats>>> = (0..owned.len()).map(|_| None).collect();
    let slots: Vec<std::sync::Mutex<Option<Result<DatasetStats>>>> =
        (0..owned.len()).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..workers.min(owned.len()) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= owned.len() {
                    break;
                }
                let result = guarded(&owned[i], options);
                if let Ok(mut slot) = slots[i].lock() {
                    *slot = Some(result);
                }
            });
        }
    });
    for (i, slot) in slots.into_iter().enumerate() {
        results[i] = slot.into_inner().ok().flatten();
    }
    for (result, path) in results.into_iter().zip(&owned) {
        let stats = result.ok_or_else(|| Error::Runtime("a statistics worker died".into()))??;
        merge_file(&mut total, &stats, path)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A class drawn where no sample looked for it reported 0.0: "never
    /// found where looked", for a class nobody looked for.
    #[test]
    fn s11_3_examined_prevalence_is_none_without_an_examiner() {
        let drawn = ClassStats { voxels: 27, present_in: 0, examined_in: 0, ..ClassStats::new(2) };
        assert_eq!(drawn.examined_prevalence(), None);
        assert_eq!(drawn.to_json()["prevalence"], Value::Null);
        let looked = ClassStats { present_in: 1, examined_in: 4, ..ClassStats::new(1) };
        assert_eq!(looked.examined_prevalence(), Some(0.25));
        assert_eq!(looked.to_json()["prevalence"], json!(0.25));
        #[allow(deprecated)] // the 2.x answer, held until 3.0
        let legacy = drawn.prevalence();
        assert_eq!(legacy, 0.0);
    }

    /// An image's extents are any its file states: a plane of `2**62 x 2**62`
    /// values wrapped to 0, and the read was sized from that.  The plane is
    /// read one at a time, and refused as more than a process holds.
    #[test]
    fn s11_3_an_image_past_the_integers_is_refused_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.medh5");
        crate::annotations::read::tests::wide_box(&path);
        {
            let file = crate::h5::file::open_rw(&path).unwrap();
            let images = file.group("images").unwrap();
            let ct = images.dataset("CT").unwrap();
            let kept: Vec<_> = crate::h5::attrs::names(&ct)
                .unwrap()
                .into_iter()
                .map(|name| (crate::h5::attrs::read(&ct, &name).unwrap().unwrap(), name))
                .collect();
            drop(ct);
            images.unlink("CT").unwrap();
            let ct = crate::h5::unwritten(&images, "CT", crate::array::DType::I16, &[4, 1 << 62, 1 << 62]);
            for (value, name) in kept {
                crate::h5::attrs::write(&ct, &name, &value).unwrap();
            }
            drop(images);
            file.close().unwrap();
        }
        let err = stats_for(&path, &StatsOptions::default()).unwrap_err();
        assert!(err.to_string().contains("more than this process can hold"), "{err}");
    }

    /// An index's counts are any `int64` its file states: three of
    /// `2**63 - 1` wrapped a class's sum, as two grids of `2**63 - 1` voxels
    /// did the cohort's.  A sum past 64 bits saturates.
    #[test]
    fn s11_3_a_count_past_the_integers_saturates() {
        let mut a = DatasetStats { total_voxels: u64::MAX - 1, ..DatasetStats::default() };
        a.classes.insert(1, ClassStats { voxels: i64::MAX as u64 * 2, ..ClassStats::new(1) });
        let b = a.clone();
        a.merge(&b).unwrap();
        assert_eq!((a.total_voxels, a.classes[&1].voxels), (u64::MAX, u64::MAX));
    }

    #[test]
    fn the_merge_is_exact_for_split_blocks() {
        let values: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.37 - 4.0).collect();
        let mut whole = Moments::default();
        whole.update(&values);
        let mut parts = Moments::default();
        parts.update(&values[..17]);
        parts.update(&values[17..]);
        assert_eq!(whole.count, parts.count);
        assert!((whole.mean - parts.mean).abs() < 1e-12);
        assert!((whole.std() - parts.std()).abs() < 1e-12);
        assert_eq!(parts.minimum, -4.0);
    }

    #[test]
    fn unseen_classes_get_no_weight() {
        let mut stats = DatasetStats::default();
        let class = |class_id, voxels, present_in| ClassStats {
            voxels,
            present_in,
            examined_in: 1,
            ..ClassStats::new(class_id)
        };
        stats.classes.insert(1, class(1, 10, 1));
        stats.classes.insert(2, class(2, 0, 0));
        stats.classes.insert(3, class(3, 30, 1));
        let weights = stats.class_weights("inverse_frequency").unwrap();
        assert_eq!(weights.keys().copied().collect::<Vec<_>>(), vec![1, 3]);
        assert!((weights[&1] + weights[&3] - 2.0).abs() < 1e-12);
        assert!(stats.class_weights_warning().unwrap().contains("[2]"));
    }

    /// A NaN made every moment NaN, and the merge kept or dropped it by
    /// which file came first: `[1, 2, 3]` then `[NaN, 0]` lost the 0.
    #[test]
    fn non_finite_values_are_counted_and_the_merge_does_not_depend_on_order() {
        let (mut finite, mut odd) = (Moments::default(), Moments::default());
        finite.update(&[1.0, 2.0, 3.0]);
        odd.update(&[f64::NAN, 0.0, f64::INFINITY]);
        let (mut forward, mut backward) = (finite.clone(), odd.clone());
        forward.merge(&odd);
        backward.merge(&finite);
        let mut whole = Moments::default();
        whole.update(&[1.0, 2.0, 3.0, 0.0]);
        for merged in [&forward, &backward] {
            assert_eq!((merged.count, merged.nonfinite), (4, 2));
            assert_eq!((merged.minimum, merged.maximum), (0.0, 3.0));
            assert!((merged.mean - 1.5).abs() < 1e-12 && (merged.std() - whole.std()).abs() < 1e-12);
        }
        // Finite values fold exactly as they did.
        let mut only = Moments::default();
        only.update(&[0.0]);
        assert_eq!(only, Moments { nonfinite: 0, ..odd.clone() });
        let mut empty = Moments::default();
        empty.update(&[f64::NAN]);
        assert_eq!((empty.count, empty.nonfinite), (0, 1));
    }

    /// `null` --- a NaN moment as JSON writes it --- reloaded as a mean of 0
    /// and a standard deviation of 0, so normalising used (0, 1).
    #[test]
    fn a_moment_written_as_null_is_refused_on_reload() {
        let mut m = Moments::default();
        m.update(&[1.0, f64::NAN, 4.0]);
        assert_eq!(Moments::from_json(&m.to_json()).unwrap().nonfinite, 1);
        assert_eq!(Moments::from_json(&m.to_json()).unwrap().mean, 2.5);
        for field in ["mean", "std", "min", "max"] {
            let mut doc = m.to_json();
            doc[field] = Value::Null;
            let err = Moments::from_json(&doc).unwrap_err();
            assert!(err.to_string().contains(&format!("no {field} (null)")), "{err}");
        }
        let doc = json!({"images": {"CT": {"count": 3, "mean": null, "std": null, "min": 1.0, "max": 4.0}}});
        let err = DatasetStats::from_json(&doc).unwrap_err();
        assert!(err.to_string().starts_with("image 'CT': statistics over 3 value(s)"), "{err}");
        // Nothing measured holds no moments to lose.
        let none = json!({"count": 0, "mean": 0.0, "std": 0.0, "min": null, "max": null});
        assert_eq!(Moments::from_json(&none).unwrap(), Moments::default());
    }

    /// One class id under two keys --- 1 is "liver" here and "spleen"
    /// there --- was one class, whose voxels fed one weight.
    #[test]
    fn s5_1_a_class_id_with_two_keys_is_refused_and_an_unknown_key_is_not_a_clash() {
        let with = |key: Option<&str>| {
            let mut s = DatasetStats { samples: 1, ..Default::default() };
            s.classes.insert(1, ClassStats { key: key.map(str::to_string), voxels: 8, ..ClassStats::new(1) });
            s
        };
        let mut liver = with(Some("liver"));
        let err = liver.merge(&with(Some("spleen"))).unwrap_err();
        assert!(err.to_string().contains("class 1 is 'liver' in these statistics and 'spleen'"), "{err}");
        assert_eq!(liver, with(Some("liver")), "a refused merge changes nothing");
        for (first, second) in [(None, Some("liver")), (Some("liver"), None)] {
            let mut merged = with(first);
            merged.merge(&with(second)).unwrap();
            assert_eq!((merged.classes[&1].key.as_deref(), merged.classes[&1].voxels), (Some("liver"), 16));
        }
        let back = DatasetStats::from_json(&liver.to_json()).unwrap();
        assert_eq!(back.classes[&1].key.as_deref(), Some("liver"));
    }
}
