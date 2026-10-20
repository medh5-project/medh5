//! `medh5.sampling`: where to read --- patch windows and visit pairs (spec
//! §14.3).
//!
//! The decisions are the engine's: which annotation and grid a draw is
//! measured in, foreground from the index or a scan, class weighting, window
//! placement --- and the draws, from the engine's generator, which `rng=`
//! seeds ([`rng_arg`]).  The result types (`Patch`, `TimepointPair`) stay the
//! 1.x dataclasses; this module hands back their fields.

use std::sync::Arc;

use indexmap::IndexMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyString, PyTuple};

use medh5::rng::Rng;
use medh5::sample::Sample as EngineSample;
use medh5::sampling::{self as engine, ClassWeights, PatchSize};

use crate::convert::class_keys;
use crate::errors::R;
use crate::exclusive::Exclusive;
use crate::reader::sample_arg;
use crate::rng::rng_arg;

/// A patch size: one length for every axis, or one per axis.
fn patch_size_arg(obj: &Bound<'_, PyAny>) -> PyResult<PatchSize> {
    if let Ok(v) = obj.extract::<i64>() {
        return Ok(PatchSize::Scalar(v));
    }
    let mut axes = Vec::new();
    for item in obj.try_iter()? {
        axes.push(item?.extract::<i64>()?);
    }
    Ok(PatchSize::Axes(axes))
}

fn class_weights_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<ClassWeights> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(ClassWeights::Named("uniform".into()));
    };
    if let Ok(name) = obj.cast::<PyString>() {
        return Ok(ClassWeights::Named(name.to_string()));
    }
    let mut map = IndexMap::new();
    for item in obj.call_method0("items")?.try_iter()? {
        let (k, v): (i64, f64) = item?.extract()?;
        map.insert(k, v);
    }
    Ok(ClassWeights::Explicit(map))
}

/// `slice(start, stop)`, step `None` as 1.x built them.
fn slice_of<'py>(py: Python<'py>, (start, stop): (i64, i64)) -> PyResult<Bound<'py, PyAny>> {
    py.get_type::<PySlice>().call1((start, stop))
}

fn slices_of<'py>(py: Python<'py>, slices: &[(i64, i64)]) -> PyResult<Bound<'py, PyTuple>> {
    let items = slices.iter().map(|s| slice_of(py, *s)).collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, items)
}

/// A `Patch`'s fields, as its constructor takes them.
fn patch_fields<'py>(py: Python<'py>, p: &engine::Patch) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("slices", slices_of(py, &p.slices)?)?;
    out.set_item("pad", PyTuple::new(py, p.pad.iter().copied())?)?;
    out.set_item("center", PyTuple::new(py, p.center.iter().copied())?)?;
    out.set_item("strategy", &p.strategy)?;
    out.set_item("class_id", p.class_id)?;
    out.set_item("used_index", p.used_index)?;
    out.set_item("grid_id", &p.grid_id)?;
    Ok(out)
}

/// The engine half of `PatchSampler`: its configuration, validated, and the
/// draws.
#[pyclass(module = "medh5._core", name = "PatchSamplerHandle", frozen)]
pub struct PatchSamplerHandle {
    inner: engine::PatchSampler,
}

#[pymethods]
impl PatchSamplerHandle {
    #[new]
    #[pyo3(signature = (patch_size, *, strategy="balanced", foreground_prob=0.5, foreground_classes=None,
        class_weights=None))]
    fn new(
        patch_size: &Bound<'_, PyAny>,
        strategy: &str,
        foreground_prob: f64,
        foreground_classes: Option<&Bound<'_, PyAny>>,
        class_weights: Option<&Bound<'_, PyAny>>,
    ) -> R<Self> {
        let classes = foreground_classes.filter(|c| !c.is_none()).map(class_keys).transpose()?;
        let inner = engine::PatchSampler::new(
            patch_size_arg(patch_size)?,
            strategy,
            foreground_prob,
            classes,
            class_weights_arg(class_weights)?,
        )?;
        Ok(PatchSamplerHandle { inner })
    }

    fn repr(&self) -> String {
        self.inner.repr()
    }

    /// One draw: the fields of a `Patch`.
    #[pyo3(signature = (sample, annotation=None, rng=None, *, grid=None))]
    fn draw<'py>(
        &self,
        py: Python<'py>,
        sample: &Bound<'py, PyAny>,
        annotation: Option<String>,
        rng: Option<Bound<'py, PyAny>>,
        grid: Option<String>,
    ) -> R<Bound<'py, PyDict>> {
        let sample = sample_arg(sample)?;
        let mut rng = rng_arg(rng.as_ref())?;
        let patch = py.detach(|| self.inner.draw(&sample, annotation.as_deref(), &mut rng, grid.as_deref()))?;
        Ok(patch_fields(py, &patch)?)
    }

    /// `n` draws from one generator: the fields of each `Patch`.
    #[pyo3(signature = (sample, annotation=None, n=1, rng=None, *, grid=None))]
    fn draws<'py>(
        &self,
        py: Python<'py>,
        sample: &Bound<'py, PyAny>,
        annotation: Option<String>,
        n: usize,
        rng: Option<Bound<'py, PyAny>>,
        grid: Option<String>,
    ) -> R<Vec<Bound<'py, PyDict>>> {
        let sample = sample_arg(sample)?;
        let mut rng = rng_arg(rng.as_ref())?;
        let patches =
            py.detach(|| self.inner.draws_in(&sample, annotation.as_deref(), n, &mut rng, grid.as_deref()))?;
        Ok(patches.iter().map(|p| patch_fields(py, p)).collect::<PyResult<Vec<_>>>()?)
    }

    /// The draws of [`draws`](Self::draws), one per `next()`.  The annotation
    /// and the grid are checked here, before the first patch.
    #[pyo3(signature = (sample, annotation=None, n=1, rng=None, *, grid=None))]
    fn iter_draws<'py>(
        &self,
        py: Python<'py>,
        sample: &Bound<'py, PyAny>,
        annotation: Option<String>,
        n: usize,
        rng: Option<Bound<'py, PyAny>>,
        grid: Option<String>,
    ) -> R<PatchDraws> {
        let sample = sample_arg(sample)?;
        let rng = rng_arg(rng.as_ref())?;
        py.detach(|| {
            let found = self.inner.annotation(&sample, annotation.as_deref(), grid.as_deref())?;
            let grid_id = self.inner.window_grid(&sample, found.as_deref(), grid.as_deref())?;
            sample.grid(&grid_id).map(|_| ())
        })?;
        Ok(PatchDraws {
            sampler: self.inner.clone(),
            sample,
            annotation,
            grid,
            state: Exclusive::new("a PatchDraws", Draws { rng, remaining: n }),
        })
    }

    /// The annotation a draw takes foreground from, auto-selected if `None`.
    #[pyo3(signature = (sample, annotation=None, grid=None))]
    fn annotation(
        &self,
        sample: &Bound<'_, PyAny>,
        annotation: Option<String>,
        grid: Option<String>,
    ) -> R<Option<String>> {
        let py = sample.py();
        let sample = sample_arg(sample)?;
        Ok(crate::h5(py, || self.inner.annotation(&sample, annotation.as_deref(), grid.as_deref()))?)
    }

    /// The grid a draw is measured in.
    #[pyo3(signature = (sample, annotation=None, grid=None))]
    fn window_grid(&self, sample: &Bound<'_, PyAny>, annotation: Option<String>, grid: Option<String>) -> R<String> {
        let py = sample.py();
        let sample = sample_arg(sample)?;
        Ok(crate::h5(py, || self.inner.window_grid(&sample, annotation.as_deref(), grid.as_deref()))?)
    }

    /// A class to sample from, weighted as configured; `None` when no class
    /// has foreground.
    #[pyo3(signature = (counts, rng=None))]
    fn pick_class<'py>(&self, counts: &Bound<'py, PyAny>, rng: Option<Bound<'py, PyAny>>) -> R<Option<i64>> {
        let mut map = IndexMap::new();
        for item in counts.call_method0("items")?.try_iter()? {
            let (k, v): (i64, i64) = item?.extract()?;
            map.insert(k, v);
        }
        let mut rng = rng_arg(rng.as_ref())?;
        Ok(self.inner.pick_class(&map, &mut rng)?)
    }
}

/// `n` patch draws from one generator, drawn one per `next()`: nothing is
/// held but the generator and the count, so a long run costs one patch at a
/// time.  Drawn by one thread at a time (E4): a `next()` while another runs
/// is refused, as two threads sharing one generator would draw nothing
/// either could repeat.
#[pyclass(module = "medh5._core", name = "PatchDraws", frozen)]
pub struct PatchDraws {
    sampler: engine::PatchSampler,
    sample: Arc<EngineSample>,
    annotation: Option<String>,
    grid: Option<String>,
    state: Exclusive<Draws>,
}

/// What each draw changes: the generator, and the draws still to come.
struct Draws {
    rng: Rng,
    remaining: usize,
}

#[pymethods]
impl PatchDraws {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// The next draw: the fields of a `Patch`.
    fn __next__<'py>(&self, py: Python<'py>) -> R<Option<Bound<'py, PyDict>>> {
        let mut state = self.state.write()?;
        if state.remaining == 0 {
            return Ok(None);
        }
        let draws = &mut *state;
        let patch = py.detach(|| {
            self.sampler.draw(&self.sample, self.annotation.as_deref(), &mut draws.rng, self.grid.as_deref())
        })?;
        draws.remaining -= 1;
        drop(state);
        Ok(Some(patch_fields(py, &patch)?))
    }

    /// The draws still to come.
    fn __length_hint__(&self) -> R<usize> {
        Ok(self.state.read()?.remaining)
    }
}

/// Broadcast a patch size across `ndim` spatial axes.
#[pyfunction]
fn sampling_coerce_patch_size<'py>(
    py: Python<'py>,
    patch_size: &Bound<'py, PyAny>,
    ndim: usize,
) -> R<Bound<'py, PyTuple>> {
    let size = engine::coerce_patch_size(&patch_size_arg(patch_size)?, ndim)?;
    Ok(PyTuple::new(py, size)?)
}

/// Refuse a volume shape with a negative extent: the windows over it would
/// run backwards (`slice(0, -5)`), which reads nothing.
fn check_shape(shape: &[i64]) -> R<()> {
    if shape.iter().any(|n| *n < 0) {
        return Err(medh5::Error::invalid(format!(
            "shape {} must be non-negative on every axis",
            medh5::json::repr_int_tuple(shape)
        ))
        .into());
    }
    Ok(())
}

/// `(slices, pad)` covering `patch` voxels around `center`.
#[pyfunction]
fn sampling_window_around<'py>(
    py: Python<'py>,
    center: Vec<i64>,
    patch: Vec<i64>,
    shape: Vec<i64>,
) -> R<(Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
    if center.len() != patch.len() || shape.len() != patch.len() {
        return Err(medh5::Error::invalid(format!(
            "center, patch and shape must have one entry per axis, not {}, {} and {}",
            center.len(),
            patch.len(),
            shape.len()
        ))
        .into());
    }
    if patch.iter().any(|v| *v <= 0) {
        return Err(medh5::Error::invalid(format!(
            "patch {} must be positive on every axis",
            medh5::json::repr_int_tuple(&patch)
        ))
        .into());
    }
    check_shape(&shape)?;
    let (slices, pad) = engine::window_around(&center, &patch, &shape);
    Ok((slices_of(py, &slices)?, PyTuple::new(py, pad)?))
}

/// The sliding-window cover of a volume: a list of `Patch` field dicts.
#[pyfunction]
#[pyo3(signature = (shape, patch_size, *, overlap=0, grid_id=None))]
fn sampling_grid_patches<'py>(
    py: Python<'py>,
    shape: Vec<i64>,
    patch_size: &Bound<'py, PyAny>,
    overlap: i64,
    grid_id: Option<String>,
) -> R<Bound<'py, PyList>> {
    check_shape(&shape)?;
    let patches = engine::grid_patches(&shape, &patch_size_arg(patch_size)?, overlap, grid_id.as_deref())?;
    let out = PyList::empty(py);
    for p in &patches {
        out.append(patch_fields(py, p)?)?;
    }
    Ok(out)
}

/// Refuse an unknown pair mode.
#[pyfunction]
fn sampling_check_pair_mode(mode: &str) -> R<()> {
    engine::TimepointPairSampler::new(mode)?;
    Ok(())
}

/// The visit pairs of a sample in `mode`: `(first, second, interval_days,
/// label)` each.
#[pyfunction]
fn sampling_pairs<'py>(py: Python<'py>, mode: &str, sample: &Bound<'py, PyAny>) -> R<Bound<'py, PyList>> {
    let sample = sample_arg(sample)?;
    let sampler = engine::TimepointPairSampler::new(mode)?;
    let pairs = crate::h5(py, || sampler.pairs(&sample))?;
    let out = PyList::empty(py);
    for p in pairs {
        out.append((p.first, p.second, p.interval_days, p.label))?;
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PatchSamplerHandle>()?;
    m.add_class::<PatchDraws>()?;
    for f in [
        wrap_pyfunction!(sampling_coerce_patch_size, m)?,
        wrap_pyfunction!(sampling_window_around, m)?,
        wrap_pyfunction!(sampling_grid_patches, m)?,
        wrap_pyfunction!(sampling_check_pair_mode, m)?,
        wrap_pyfunction!(sampling_pairs, m)?,
    ] {
        m.add_function(f)?;
    }
    let py = m.py();
    m.add("SAMPLING_STRATEGIES", PyTuple::new(py, engine::STRATEGIES)?)?;
    m.add("SAMPLING_PAIR_MODES", PyTuple::new(py, engine::PAIR_MODES)?)?;
    Ok(())
}
