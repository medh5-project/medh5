//! `medh5.collection` and `medh5.dataset`: collections (spec §2.2), cohort
//! manifests, splits, statistics and cohort checks (§12).
//!
//! The dataset types cross as their JSON form: `medh5/dataset/*.py` keeps
//! the 1.x dataclasses, and every computation on them --- grouping keys, the
//! membership digest, splitting, merging statistics, the cohort checks ---
//! runs here, on the engine's types rebuilt from that JSON.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use indexmap::IndexMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use medh5::collection as coll;
use medh5::dataset::{check, manifest, split, stats};

use crate::convert::{f64_vec, json_to_py, opt_strings, py_to_json};
use crate::errors::{path_error, paths_error, R};
use crate::nodes::Group;
use crate::reader::SampleHandle;

// -- collections ----------------------------------------------------------------------------

/// One open collection shard.  Members are samples sharing its file.
#[pyclass(module = "medh5._core", name = "CollectionHandle", frozen)]
pub struct CollectionHandle {
    inner: Mutex<Option<coll::Collection>>,
    path: Option<String>,
}

impl CollectionHandle {
    fn wrap(c: coll::Collection) -> Self {
        let path = c.path.as_ref().map(|p| p.to_string_lossy().into_owned());
        CollectionHandle { inner: Mutex::new(Some(c)), path }
    }

    /// `f` of the open collection, detached (`crate::h5`): the lock is
    /// taken and let go there too, so no thread waits for it attached.
    fn with<T: Send>(&self, py: Python<'_>, f: impl FnOnce(&coll::Collection) -> medh5::Result<T> + Send) -> R<T> {
        crate::h5(py, || {
            let guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            let c = guard.as_ref().ok_or_else(|| medh5::Error::File("this collection has been closed".into()))?;
            Ok(f(c)?)
        })
    }
}

#[pymethods]
impl CollectionHandle {
    #[getter]
    fn path(&self) -> Option<String> {
        self.path.clone()
    }
    /// Closing waits, detached, for a call another thread is making through
    /// the collection, then closes it with the lock let go.
    fn close(&self, py: Python<'_>) {
        crate::h5(py, || {
            let taken = self.inner.lock().unwrap_or_else(PoisonError::into_inner).take();
            if let Some(mut c) = taken {
                c.close();
            }
        })
    }
    #[getter]
    fn is_open(&self, py: Python<'_>) -> bool {
        crate::h5(py, || self.inner.lock().unwrap_or_else(PoisonError::into_inner).is_some())
    }
    fn keys(&self, py: Python<'_>) -> R<Vec<String>> {
        self.with(py, |c| c.keys())
    }
    fn __len__(&self, py: Python<'_>) -> R<usize> {
        self.with(py, |c| c.len())
    }
    fn contains(&self, py: Python<'_>, key: &str) -> R<bool> {
        self.with(py, |c| c.contains(key))
    }
    /// One member, as a sample handle.
    fn get(&self, py: Python<'_>, key: &str) -> R<SampleHandle> {
        Ok(SampleHandle::wrap(self.with(py, |c| c.get(key))?))
    }
    #[getter]
    fn version(&self, py: Python<'_>) -> R<String> {
        self.with(py, |c| c.version())
    }
    #[getter]
    fn kind(&self, py: Python<'_>) -> R<String> {
        self.with(py, |c| c.kind())
    }
    fn repr(&self, py: Python<'_>) -> R<String> {
        self.with(py, |c| c.repr())
    }
    fn summary<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        let doc = self.with(py, |c| c.summary())?;
        Ok(json_to_py(py, &doc)?)
    }
    /// `{sample_key: subject_id}`.
    fn subject_ids<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let pairs = self.with(py, |c| c.subject_ids())?;
        let out = PyDict::new(py);
        for (k, v) in pairs {
            out.set_item(k, v)?;
        }
        Ok(out)
    }
    /// The root group.
    #[getter]
    fn root(&self, py: Python<'_>) -> R<Group> {
        self.with(py, |c| Ok(Group::wrap(c.root.clone())))
    }
    /// The `samples` group.
    #[getter]
    fn group(&self, py: Python<'_>) -> R<Group> {
        Ok(Group::wrap(self.with(py, |c| c.group())?))
    }
}

#[pyfunction]
fn open_collection(py: Python<'_>, path: PathBuf) -> R<CollectionHandle> {
    let at = path.clone();
    Ok(CollectionHandle::wrap(py.detach(move || coll::open_collection(&path)).map_err(|e| path_error(e, &at))?))
}

/// A file whatever its kind: a sample handle, or a collection handle (a
/// member's sample handle when `key` is given).
#[pyfunction]
#[pyo3(signature = (path, *, key=None))]
fn open_any<'py>(py: Python<'py>, path: PathBuf, key: Option<String>) -> R<Bound<'py, PyAny>> {
    let at = path.clone();
    let found = py.detach(move || coll::open_any(&path, key.as_deref())).map_err(|e| path_error(e, &at))?;
    Ok(match found {
        coll::AnyFile::Sample(s) => Bound::new(py, SampleHandle::wrap(s))?.into_any(),
        coll::AnyFile::Collection(c) => Bound::new(py, CollectionHandle::wrap(c))?.into_any(),
    })
}

/// Whether a root declares itself a collection: a `Group`, a `Sample`, a
/// collection, or a path.
#[pyfunction]
fn is_collection(obj: &Bound<'_, PyAny>) -> R<bool> {
    let py = obj.py();
    if let Ok(g) = obj.cast::<Group>() {
        let group = &g.get().group;
        return Ok(crate::h5(py, || coll::is_collection(group))?);
    }
    if let Ok(c) = obj.cast::<CollectionHandle>() {
        return c.get().with(obj.py(), |c| coll::is_collection(&c.root));
    }
    if let Ok(root) = obj.getattr("root") {
        if let Ok(g) = root.cast::<Group>() {
            let group = &g.get().group;
            return Ok(crate::h5(py, || coll::is_collection(group))?);
        }
    }
    let path: PathBuf = obj.extract()?;
    // Opened, read and closed detached: each may wait for HDF5's lock.
    crate::h5(py, || {
        let file = medh5::h5::file::open_read(&path).map_err(|e| path_error(e, &path))?;
        Ok(coll::is_collection(&file)?)
    })
}

#[pyfunction]
fn default_key(path: PathBuf) -> R<String> {
    Ok(coll::default_key(&path)?)
}

#[pyfunction]
#[pyo3(signature = (sources, out, *, keys=None))]
fn pack(py: Python<'_>, sources: Vec<PathBuf>, out: PathBuf, keys: Option<&Bound<'_, PyAny>>) -> R<String> {
    let keys = opt_strings(keys)?;
    let named = sources.clone();
    let written = py
        .detach(move || {
            let refs: Vec<&Path> = sources.iter().map(PathBuf::as_path).collect();
            coll::pack(&refs, &out, keys.as_deref())
        })
        .map_err(|e| paths_error(e, &named))?;
    Ok(written.to_string_lossy().into_owned())
}

#[pyfunction]
#[pyo3(signature = (path, outdir, *, keys=None, suffix=".medh5".to_string()))]
fn unpack(
    py: Python<'_>,
    path: PathBuf,
    outdir: PathBuf,
    keys: Option<&Bound<'_, PyAny>>,
    suffix: String,
) -> R<Vec<String>> {
    let keys = opt_strings(keys)?;
    let written = py.detach(move || coll::unpack(&path, &outdir, keys.as_deref(), &suffix))?;
    Ok(written.into_iter().map(|p| p.to_string_lossy().into_owned()).collect())
}

#[pyfunction]
fn extract(py: Python<'_>, path: PathBuf, key: String, out: PathBuf) -> R<String> {
    let written = py.detach(move || coll::extract(&path, &key, &out))?;
    Ok(written.to_string_lossy().into_owned())
}

// -- manifests ---------------------------------------------------------------------------------

fn manifest_arg(doc: &Bound<'_, PyAny>) -> R<manifest::Manifest> {
    Ok(manifest::Manifest::from_json(&py_to_json(doc)?)?)
}

fn suffix_list(suffixes: Option<Vec<String>>) -> Vec<String> {
    suffixes.unwrap_or_else(|| manifest::SUFFIXES.iter().map(|s| s.to_string()).collect())
}

#[pyfunction]
#[pyo3(signature = (root, *, suffixes=None))]
fn dataset_find(py: Python<'_>, root: PathBuf, suffixes: Option<Vec<String>>) -> R<Vec<String>> {
    let suffixes = suffix_list(suffixes);
    let found = py.detach(move || {
        let refs: Vec<&str> = suffixes.iter().map(String::as_str).collect();
        manifest::find_with(&root, &refs)
    })?;
    Ok(found.into_iter().map(|p| p.to_string_lossy().into_owned()).collect())
}

/// `(manifest JSON, failures)`; `on_error="raise"` re-raises the first.
#[pyfunction]
#[pyo3(signature = (root, *, suffixes=None, on_error="warn"))]
fn dataset_scan<'py>(
    py: Python<'py>,
    root: PathBuf,
    suffixes: Option<Vec<String>>,
    on_error: &str,
) -> R<Bound<'py, PyTuple>> {
    let suffixes = suffix_list(suffixes);
    let strict = match on_error {
        "raise" => true,
        "warn" => false,
        other => {
            return Err(medh5::Error::invalid(format!(
                "on_error must be 'warn' or 'raise', not {}",
                medh5::json::repr_str(other)
            ))
            .into())
        }
    };
    let (found, failures) = py.detach(move || {
        let refs: Vec<&str> = suffixes.iter().map(String::as_str).collect();
        manifest::scan_with(&root, &refs, strict)
    })?;
    Ok(PyTuple::new(py, [json_to_py(py, &found.to_json())?, PyTuple::new(py, failures)?.into_any()])?)
}

#[pyfunction]
fn dataset_entries_for<'py>(py: Python<'py>, path: PathBuf) -> R<Bound<'py, PyList>> {
    let entries = py.detach(move || manifest::entries_for(&path))?;
    let out = PyList::empty(py);
    for e in entries {
        out.append(json_to_py(py, &e.to_json())?)?;
    }
    Ok(out)
}

/// An entry's JSON as the engine writes it (key order, omitted `None`s).
#[pyfunction]
fn dataset_entry_json<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &manifest::Entry::from_json(&py_to_json(doc)?)?.to_json())?)
}

/// A field by its dotted name (`cohort.site_id`), refused when not a field.
#[pyfunction]
fn dataset_entry_field<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>, dotted: &str) -> R<Bound<'py, PyAny>> {
    let entry = manifest::Entry::from_json(&py_to_json(doc)?)?;
    Ok(json_to_py(py, &entry.field(dotted)?)?)
}

/// The grouping key of each entry, in order (`str(entry.field(by))`).
#[pyfunction]
fn dataset_group_keys(entries: &Bound<'_, PyAny>, by: &str) -> R<Vec<String>> {
    let mut out = Vec::new();
    for item in entries.try_iter()? {
        let entry = manifest::Entry::from_json(&py_to_json(&item?)?)?;
        out.push(entry.field_str(by)?);
    }
    Ok(out)
}

#[pyfunction]
fn dataset_counts<'py>(py: Python<'py>, entries: &Bound<'py, PyAny>, by: &str) -> R<Bound<'py, PyDict>> {
    let mut parsed = Vec::new();
    for item in entries.try_iter()? {
        parsed.push(manifest::Entry::from_json(&py_to_json(&item?)?)?);
    }
    let out = PyDict::new(py);
    for (k, v) in manifest::counts(&parsed, by)? {
        out.set_item(k, v)?;
    }
    Ok(out)
}

/// The full manifest JSON (with its `sha256`).
#[pyfunction]
fn dataset_manifest_json<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &manifest_arg(doc)?.to_json())?)
}

#[pyfunction]
fn dataset_manifest_sha256(doc: &Bound<'_, PyAny>) -> R<String> {
    Ok(manifest_arg(doc)?.sha256())
}

/// Written with paths as `paths` says: `"keep"`, `"relative"` or `"absolute"`.
#[pyfunction]
#[pyo3(signature = (doc, path, *, paths="keep"))]
fn dataset_manifest_save(doc: &Bound<'_, PyAny>, path: PathBuf, paths: &str) -> R<String> {
    let style = manifest::PathStyle::parse(paths)?;
    Ok(manifest_arg(doc)?.save_with(&path, style)?.to_string_lossy().into_owned())
}

/// `(manifest JSON, legacy and read otherwise than 3.0 will read it)`.
#[pyfunction]
#[pyo3(signature = (path, *, base=None))]
fn dataset_manifest_load<'py>(py: Python<'py>, path: PathBuf, base: Option<PathBuf>) -> R<(Bound<'py, PyAny>, bool)> {
    let (found, differs) = manifest::Manifest::load_noting(&path, base.as_deref())?;
    Ok((json_to_py(py, &found.to_json())?, differs))
}

/// Whether a manifest is legacy (no `schema`); an unknown schema is refused.
#[pyfunction]
fn dataset_manifest_legacy(doc: &Bound<'_, PyAny>) -> R<bool> {
    Ok(manifest::check_schema(&py_to_json(doc)?)?)
}

#[pyfunction]
fn dataset_manifest_stale(py: Python<'_>, doc: &Bound<'_, PyAny>) -> R<Vec<String>> {
    let found = manifest_arg(doc)?;
    Ok(py.detach(move || found.stale()))
}

// -- splits ------------------------------------------------------------------------------------

fn split_arg(doc: &Bound<'_, PyAny>) -> R<split::Split> {
    Ok(split::Split::from_json(&py_to_json(doc)?)?)
}

fn entry_arg(doc: &Bound<'_, PyAny>) -> R<manifest::Entry> {
    Ok(manifest::Entry::from_json(&py_to_json(doc)?)?)
}

#[pyfunction]
fn dataset_default_ratios<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (k, v) in split::default_ratios() {
        out.set_item(k, v)?;
    }
    Ok(out)
}

/// `(split JSON, groups medh5.split/2 places elsewhere)`: the count only when
/// `algorithm` is left out and there is no `previous` split, else `None`.
#[pyfunction]
#[pyo3(signature = (manifest, *, set_id="default".to_string(), group_by="group_id".to_string(), stratify_by=None,
    ratios=None, k_folds=None, seed=0, algorithm=None, previous=None))]
#[allow(clippy::too_many_arguments)]
fn dataset_make_splits<'py>(
    py: Python<'py>,
    manifest: &Bound<'py, PyAny>,
    set_id: String,
    group_by: String,
    stratify_by: Option<String>,
    ratios: Option<&Bound<'py, PyAny>>,
    k_folds: Option<i64>,
    seed: i64,
    algorithm: Option<String>,
    previous: Option<&Bound<'py, PyAny>>,
) -> R<(Bound<'py, PyAny>, Option<usize>)> {
    let found = manifest_arg(manifest)?;
    let ratios = match ratios.filter(|r| !r.is_none()) {
        None => None,
        Some(r) => {
            let mut map = IndexMap::new();
            for item in r.call_method0("items")?.try_iter()? {
                let (k, v): (String, f64) = item?.extract()?;
                map.insert(k, v);
            }
            Some(map)
        }
    };
    let previous = match previous.filter(|p| !p.is_none()) {
        None => None,
        Some(doc) => Some(split::SplitRecord::from_json(&py_to_json(doc)?)?),
    };
    let algorithm = algorithm.as_deref().map(split::SplitAlgorithm::parse).transpose()?;
    let options = split::SplitOptions { set_id, group_by, stratify_by, ratios, k_folds, seed };
    let request = split::SplitRequest::from(options).with_previous(previous);
    let (record, moved) = split::make_splits_asked(&found, &request, algorithm)?;
    Ok((json_to_py(py, &record.to_json())?, moved))
}

/// The groups `doc` places elsewhere than `other` does: `{group: (was, now)}`.
#[pyfunction]
fn dataset_split_moved_from<'py>(
    py: Python<'py>,
    doc: &Bound<'py, PyAny>,
    other: &Bound<'py, PyAny>,
) -> R<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (group, (was, now)) in split_arg(doc)?.moved_from(&split_arg(other)?) {
        out.set_item(group, (was, now))?;
    }
    Ok(out)
}

#[pyfunction]
fn dataset_split_json<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &split::SplitRecord::from_json(&py_to_json(doc)?)?.to_json())?)
}

#[pyfunction]
fn dataset_split_partition_of(doc: &Bound<'_, PyAny>, entry: &Bound<'_, PyAny>) -> R<Option<String>> {
    let s = split_arg(doc)?;
    Ok(s.partition_of(&entry_arg(entry)?)?.map(str::to_string))
}

#[pyfunction]
fn dataset_split_fold_of(doc: &Bound<'_, PyAny>, entry: &Bound<'_, PyAny>) -> R<Option<i64>> {
    Ok(split_arg(doc)?.fold_of(&entry_arg(entry)?)?)
}

#[pyfunction]
fn dataset_split_paths(doc: &Bound<'_, PyAny>, partition: &str) -> R<Vec<String>> {
    Ok(split_arg(doc)?.paths(partition))
}

#[pyfunction]
fn dataset_split_counts<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (k, v) in split_arg(doc)?.counts() {
        out.set_item(k, v)?;
    }
    Ok(out)
}

#[pyfunction]
fn dataset_split_balance<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (k, v) in split_arg(doc)?.balance() {
        let inner = PyDict::new(py);
        for (a, b) in v {
            inner.set_item(a, b)?;
        }
        out.set_item(k, inner)?;
    }
    Ok(out)
}

#[pyfunction]
fn dataset_split_empty_folds(doc: &Bound<'_, PyAny>) -> R<Vec<i64>> {
    Ok(split_arg(doc)?.empty_folds())
}

#[pyfunction]
fn dataset_split_underfilled(doc: &Bound<'_, PyAny>) -> R<Vec<String>> {
    Ok(split_arg(doc)?.underfilled())
}

#[pyfunction]
fn dataset_split_leaks(doc: &Bound<'_, PyAny>) -> R<Vec<String>> {
    Ok(split_arg(doc)?.leaks())
}

#[pyfunction]
#[pyo3(signature = (split, manifest, *, assigned_by=None, fold=None))]
fn dataset_write_claims(
    py: Python<'_>,
    split: &Bound<'_, PyAny>,
    manifest: &Bound<'_, PyAny>,
    assigned_by: Option<String>,
    fold: Option<i64>,
) -> R<Vec<String>> {
    let (s, m) = (split_arg(split)?, manifest_arg(manifest)?);
    Ok(py.detach(move || split::write_claims(&s, &m, assigned_by.as_deref(), fold))?)
}

#[pyfunction]
fn dataset_split_load<'py>(py: Python<'py>, path: PathBuf) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &split::SplitRecord::load(&path)?.to_json())?)
}

// -- statistics ----------------------------------------------------------------------------------
//
// Statistics cross as their exact state (count, mean, M2, ...), read from the
// Python dataclasses' fields: the JSON form carries `std`, and rebuilding M2
// from it would not be exact.

type RawMoments = (u64, f64, f64, f64, f64, u64);

fn moments_of(obj: &Bound<'_, PyAny>) -> PyResult<stats::Moments> {
    Ok(stats::Moments {
        count: obj.getattr("count")?.extract()?,
        mean: obj.getattr("mean")?.extract()?,
        m2: obj.getattr("m2")?.extract()?,
        minimum: obj.getattr("minimum")?.extract()?,
        maximum: obj.getattr("maximum")?.extract()?,
        // A `Moments` pickled before the field existed has none.
        nonfinite: match obj.getattr("nonfinite") {
            Ok(v) => v.extract()?,
            Err(_) => 0,
        },
    })
}

fn raw_moments(m: &stats::Moments) -> RawMoments {
    (m.count, m.mean, m.m2, m.minimum, m.maximum, m.nonfinite)
}

/// The engine's `ClassStats` from a Python `ClassStats`' attributes.
fn class_stats_of(v: &Bound<'_, PyAny>) -> PyResult<stats::ClassStats> {
    Ok(stats::ClassStats {
        class_id: v.getattr("class_id")?.extract()?,
        key: match v.getattr("key") {
            Ok(key) => key.extract()?,
            Err(_) => None,
        },
        voxels: v.getattr("voxels")?.extract()?,
        present_in: v.getattr("present_in")?.extract()?,
        examined_in: v.getattr("examined_in")?.extract()?,
    })
}

/// `ClassStats.to_json`: the engine's JSON for one class.
#[pyfunction]
fn dataset_class_stats_json<'py>(py: Python<'py>, s: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &class_stats_of(s)?.to_json())?)
}

fn stats_of(obj: &Bound<'_, PyAny>) -> PyResult<stats::DatasetStats> {
    let mut images = IndexMap::new();
    for item in obj.getattr("images")?.call_method0("items")?.try_iter()? {
        let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
        images.insert(k, moments_of(&v)?);
    }
    let mut classes = IndexMap::new();
    for item in obj.getattr("classes")?.call_method0("items")?.try_iter()? {
        let (k, v): (i64, Bound<'_, PyAny>) = item?.extract()?;
        classes.insert(k, class_stats_of(&v)?);
    }
    Ok(stats::DatasetStats {
        samples: obj.getattr("samples")?.extract()?,
        images,
        classes,
        total_voxels: obj.getattr("total_voxels")?.extract()?,
        failures: obj.getattr("failures")?.extract()?,
        physical: obj.getattr("physical")?.extract()?,
    })
}

/// A statistics state as `{samples, physical, total_voxels, failures,
/// images: {key: (count, mean, m2, min, max, nonfinite)}, classes: {id:
/// (voxels, present_in, examined_in, key)}}`.
fn raw_stats<'py>(py: Python<'py>, s: &stats::DatasetStats) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("samples", s.samples)?;
    out.set_item("physical", s.physical)?;
    out.set_item("total_voxels", s.total_voxels)?;
    out.set_item("failures", PyTuple::new(py, &s.failures)?)?;
    let images = PyDict::new(py);
    for (k, m) in &s.images {
        images.set_item(k, raw_moments(m))?;
    }
    out.set_item("images", images)?;
    let classes = PyDict::new(py);
    for (k, c) in &s.classes {
        classes.set_item(k, (c.voxels, c.present_in, c.examined_in, c.key.as_deref()))?;
    }
    out.set_item("classes", classes)?;
    Ok(out)
}

/// `Moments.update`: the state after folding in `values`.
#[pyfunction]
fn dataset_moments_update(m: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>) -> R<RawMoments> {
    let mut found = moments_of(m)?;
    found.update(&f64_vec(values)?);
    Ok(raw_moments(&found))
}

/// `Moments.merge` (Chan--Golub--LeVeque): the merged state.
#[pyfunction]
fn dataset_moments_merge(m: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> R<RawMoments> {
    let mut found = moments_of(m)?;
    found.merge(&moments_of(other)?);
    Ok(raw_moments(&found))
}

#[pyfunction]
fn dataset_moments_std(m: &Bound<'_, PyAny>) -> R<f64> {
    Ok(moments_of(m)?.std())
}

#[pyfunction]
fn dataset_moments_json<'py>(py: Python<'py>, m: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &moments_of(m)?.to_json())?)
}

/// `Moments.from_json`: the state the JSON form describes.
#[pyfunction]
fn dataset_moments_from_json(doc: &Bound<'_, PyAny>) -> R<RawMoments> {
    Ok(raw_moments(&stats::Moments::from_json(&py_to_json(doc)?)?))
}

#[pyfunction]
fn dataset_stats_merge<'py>(
    py: Python<'py>,
    s: &Bound<'py, PyAny>,
    other: &Bound<'py, PyAny>,
) -> R<Bound<'py, PyDict>> {
    let mut found = stats_of(s)?;
    found.merge(&stats_of(other)?)?;
    Ok(raw_stats(py, &found)?)
}

#[pyfunction]
fn dataset_stats_json<'py>(py: Python<'py>, s: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &stats_of(s)?.to_json())?)
}

/// `DatasetStats.from_json`: the state the JSON form describes.
#[pyfunction]
fn dataset_stats_from_json<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyDict>> {
    Ok(raw_stats(py, &stats::DatasetStats::from_json(&py_to_json(doc)?)?)?)
}

#[pyfunction]
fn dataset_stats_normalization(s: &Bound<'_, PyAny>, image_key: &str) -> R<(f64, f64)> {
    Ok(stats_of(s)?.normalization(image_key))
}

#[pyfunction]
#[pyo3(signature = (s, *, scheme="inverse_frequency"))]
fn dataset_stats_class_weights<'py>(py: Python<'py>, s: &Bound<'py, PyAny>, scheme: &str) -> R<Bound<'py, PyDict>> {
    let found = stats_of(s)?;
    if let Some(message) = found.class_weights_warning() {
        let warnings = py.import("warnings")?;
        warnings.call_method1("warn", (message, py.get_type::<pyo3::exceptions::PyUserWarning>(), 3))?;
    }
    let out = PyDict::new(py);
    for (k, v) in found.class_weights(scheme)? {
        out.set_item(k, v)?;
    }
    Ok(out)
}

fn stats_options(
    images: Option<&Bound<'_, PyAny>>,
    annotations: Option<&Bound<'_, PyAny>>,
    sample_stride: i64,
    physical: bool,
) -> R<stats::StatsOptions> {
    // A stride of 0 read as 1, and a negative one raised `OverflowError`.
    let Some(sample_stride) = usize::try_from(sample_stride).ok().filter(|s| *s >= 1) else {
        return Err(medh5::Error::invalid(format!(
            "sample_stride must be at least 1, not {sample_stride}: it reads every Nth plane"
        ))
        .into());
    };
    Ok(stats::StatsOptions {
        images: opt_strings(images)?,
        annotations: opt_strings(annotations)?,
        sample_stride,
        physical,
    })
}

#[pyfunction]
#[pyo3(signature = (path, *, images=None, annotations=None, sample_stride=1, physical=true))]
fn dataset_stats_for<'py>(
    py: Python<'py>,
    path: PathBuf,
    images: Option<&Bound<'py, PyAny>>,
    annotations: Option<&Bound<'py, PyAny>>,
    sample_stride: i64,
    physical: bool,
) -> R<Bound<'py, PyDict>> {
    let options = stats_options(images, annotations, sample_stride, physical)?;
    let found = py.detach(move || stats::stats_for(&path, &options))?;
    Ok(raw_stats(py, &found)?)
}

#[pyfunction]
#[pyo3(signature = (paths, *, images=None, annotations=None, workers=1, sample_stride=1, physical=true))]
fn dataset_compute_stats<'py>(
    py: Python<'py>,
    paths: Vec<PathBuf>,
    images: Option<&Bound<'py, PyAny>>,
    annotations: Option<&Bound<'py, PyAny>>,
    workers: usize,
    sample_stride: i64,
    physical: bool,
) -> R<Bound<'py, PyDict>> {
    let options = stats_options(images, annotations, sample_stride, physical)?;
    let found = py.detach(move || stats::compute_stats(&paths, &options, workers))?;
    Ok(raw_stats(py, &found)?)
}

// -- cohort checks -------------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (manifest, *, set_id=None, deep=false))]
fn dataset_check<'py>(
    py: Python<'py>,
    manifest: &Bound<'py, PyAny>,
    set_id: Option<String>,
    deep: bool,
) -> R<Bound<'py, PyAny>> {
    let m = manifest_arg(manifest)?;
    let report = py.detach(move || check::check(&m, set_id.as_deref(), deep));
    Ok(json_to_py(py, &report.to_json())?)
}

#[pyfunction]
fn dataset_check_format(report: &Bound<'_, PyAny>) -> R<String> {
    Ok(check::CohortReport::from_json(&py_to_json(report)?).format())
}

#[pyfunction]
fn dataset_finding_line(finding: &Bound<'_, PyAny>) -> R<String> {
    Ok(check::Finding::from_json(&py_to_json(finding)?).to_string())
}

#[pyfunction]
fn dataset_finding_json<'py>(py: Python<'py>, finding: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &check::Finding::from_json(&py_to_json(finding)?).to_json())?)
}

#[pyfunction]
fn dataset_check_json<'py>(py: Python<'py>, report: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &check::CohortReport::from_json(&py_to_json(report)?).to_json())?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CollectionHandle>()?;
    for f in [
        wrap_pyfunction!(open_collection, m)?,
        wrap_pyfunction!(open_any, m)?,
        wrap_pyfunction!(is_collection, m)?,
        wrap_pyfunction!(default_key, m)?,
        wrap_pyfunction!(pack, m)?,
        wrap_pyfunction!(unpack, m)?,
        wrap_pyfunction!(extract, m)?,
        wrap_pyfunction!(dataset_find, m)?,
        wrap_pyfunction!(dataset_scan, m)?,
        wrap_pyfunction!(dataset_entries_for, m)?,
        wrap_pyfunction!(dataset_entry_json, m)?,
        wrap_pyfunction!(dataset_entry_field, m)?,
        wrap_pyfunction!(dataset_group_keys, m)?,
        wrap_pyfunction!(dataset_counts, m)?,
        wrap_pyfunction!(dataset_manifest_json, m)?,
        wrap_pyfunction!(dataset_manifest_sha256, m)?,
        wrap_pyfunction!(dataset_manifest_save, m)?,
        wrap_pyfunction!(dataset_manifest_load, m)?,
        wrap_pyfunction!(dataset_manifest_legacy, m)?,
        wrap_pyfunction!(dataset_manifest_stale, m)?,
        wrap_pyfunction!(dataset_default_ratios, m)?,
        wrap_pyfunction!(dataset_make_splits, m)?,
        wrap_pyfunction!(dataset_split_json, m)?,
        wrap_pyfunction!(dataset_split_partition_of, m)?,
        wrap_pyfunction!(dataset_split_fold_of, m)?,
        wrap_pyfunction!(dataset_split_paths, m)?,
        wrap_pyfunction!(dataset_split_counts, m)?,
        wrap_pyfunction!(dataset_split_balance, m)?,
        wrap_pyfunction!(dataset_split_empty_folds, m)?,
        wrap_pyfunction!(dataset_split_underfilled, m)?,
        wrap_pyfunction!(dataset_split_leaks, m)?,
        wrap_pyfunction!(dataset_split_moved_from, m)?,
        wrap_pyfunction!(dataset_write_claims, m)?,
        wrap_pyfunction!(dataset_split_load, m)?,
        wrap_pyfunction!(dataset_moments_update, m)?,
        wrap_pyfunction!(dataset_moments_merge, m)?,
        wrap_pyfunction!(dataset_moments_json, m)?,
        wrap_pyfunction!(dataset_moments_std, m)?,
        wrap_pyfunction!(dataset_moments_from_json, m)?,
        wrap_pyfunction!(dataset_stats_from_json, m)?,
        wrap_pyfunction!(dataset_stats_merge, m)?,
        wrap_pyfunction!(dataset_stats_json, m)?,
        wrap_pyfunction!(dataset_class_stats_json, m)?,
        wrap_pyfunction!(dataset_stats_normalization, m)?,
        wrap_pyfunction!(dataset_stats_class_weights, m)?,
        wrap_pyfunction!(dataset_stats_for, m)?,
        wrap_pyfunction!(dataset_compute_stats, m)?,
        wrap_pyfunction!(dataset_check, m)?,
        wrap_pyfunction!(dataset_check_format, m)?,
        wrap_pyfunction!(dataset_finding_line, m)?,
        wrap_pyfunction!(dataset_finding_json, m)?,
        wrap_pyfunction!(dataset_check_json, m)?,
    ] {
        m.add_function(f)?;
    }
    let py = m.py();
    m.add("COLLECTION_SUFFIX", coll::SUFFIX)?;
    m.add("SAMPLES_GROUP", coll::SAMPLES_GROUP)?;
    m.add("MANIFEST_SUFFIXES", PyTuple::new(py, manifest::SUFFIXES)?)?;
    m.add("GROUPABLE", PyTuple::new(py, manifest::GROUPABLE)?)?;
    m.add("MANIFEST_SCHEMA", manifest::MANIFEST_SCHEMA)?;
    m.add("CHECK_SEVERITIES", PyTuple::new(py, check::SEVERITIES)?)?;
    let codes = PyDict::new(py);
    for (code, summary) in check::CHECK_CODES {
        codes.set_item(code, summary)?;
    }
    m.add("CHECK_CODES", codes)?;
    let severity = PyDict::new(py);
    for (code, base) in check::CHECK_SEVERITY {
        severity.set_item(code, base)?;
    }
    m.add("CHECK_CODE_SEVERITY", severity)?;
    Ok(())
}
