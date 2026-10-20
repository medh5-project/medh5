//! `medh5.integrity`: digests, `content_id`, verification and repair (§13).
//!
//! `diagnose` and `fix` take a path, and the engine opens the file itself;
//! the digests and verification take the views a caller already holds
//! (`Dataset`, `Group`, or a `Sample` for its root).

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use medh5::integrity::{digest, repair, verify};

use crate::convert::{json_to_py, py_to_nd, strings};
use crate::errors::{path_error, R};

/// A verification result as the keyword arguments of `VerifyResult`.
pub fn verify_result_to_py<'py>(py: Python<'py>, r: &verify::VerifyResult) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);
    out.set_item("checked", PyTuple::new(py, &r.checked)?)?;
    out.set_item("mismatched", PyTuple::new(py, &r.mismatched)?)?;
    out.set_item("undigested", PyTuple::new(py, &r.undigested)?)?;
    out.set_item("malformed", PyTuple::new(py, &r.malformed)?)?;
    out.set_item("content_id_declared", &r.content_id_declared)?;
    out.set_item("content_id_computed", &r.content_id_computed)?;
    out.set_item("stale_index", PyTuple::new(py, &r.stale_index)?)?;
    out.set_item("unattested", PyTuple::new(py, &r.unattested)?)?;
    out.set_item("unverifiable", PyTuple::new(py, &r.unverifiable)?)?;
    Ok(out.into_any())
}

/// Return the `"<algo>:<hex>"` digest of an in-memory array stored at `path` (§13.1).
///
/// The digest covers the canonical byte stream --- the path, the dtype, the
/// shape and the little-endian bytes --- so it equals the digest of a dataset
/// holding the same array, however that is compressed.
#[pyfunction]
#[pyo3(signature = (path, array, algo="sha256"))]
fn array_digest(py: Python<'_>, path: &str, array: &Bound<'_, PyAny>, algo: &str) -> R<String> {
    let array = py_to_nd(array)?;
    Ok(py.detach(|| digest::array_digest(path, &array, algo))?)
}

/// Return the `"<algo>:<hex>"` digest of a byte string.
///
/// `algo` is one of `DIGEST_ALGOS`; another is refused.
#[pyfunction]
#[pyo3(signature = (payload, algo="sha256"))]
fn digest_bytes(py: Python<'_>, payload: &[u8], algo: &str) -> R<String> {
    // `bytes` are immutable: other threads may run while they are read.
    Ok(py.detach(|| medh5::digest::digest_bytes(payload, algo))?)
}

/// Split a `"<algo>:<hex>"` digest into `(algo, hex)`, raising E703 when it is malformed (§13.1).
#[pyfunction]
fn parse_digest(value: &str) -> R<(String, String)> {
    Ok(medh5::digest::parse_digest(value)?)
}

#[pyfunction]
fn diagnose<'py>(py: Python<'py>, path: PathBuf) -> R<Bound<'py, PyAny>> {
    let at = path.clone();
    let found = py.detach(move || repair::diagnose(&path)).map_err(|e| path_error(e, &at))?;
    Ok(json_to_py(py, &found.to_json())?)
}

#[pyfunction]
#[pyo3(signature = (path, *, rebuild_index=false, rewrite_digests=false, reason=None, performed_by=None, max_coords=None))]
fn fix<'py>(
    py: Python<'py>,
    path: PathBuf,
    rebuild_index: bool,
    rewrite_digests: bool,
    reason: Option<String>,
    performed_by: Option<String>,
    max_coords: Option<usize>,
) -> R<Bound<'py, PyAny>> {
    let options = repair::FixOptions { rebuild_index, rewrite_digests, reason, performed_by, max_coords };
    let at = path.clone();
    let done = py.detach(move || repair::fix(&path, &options)).map_err(|e| path_error(e, &at))?;
    Ok(json_to_py(py, &done.to_json())?)
}

// -- functions over stored objects (`Dataset`/`Group` proxies) ---------------------------

use crate::nodes::{Dataset, Group};

/// A dataset or group proxy as an HDF5 location.
enum Loc {
    Dataset(medh5::hdf5::Dataset),
    Group(medh5::hdf5::Group),
}

fn loc_of(obj: &Bound<'_, PyAny>) -> PyResult<Loc> {
    if let Ok(d) = obj.cast::<Dataset>() {
        return Ok(Loc::Dataset(d.get().ds().clone()));
    }
    if let Ok(g) = obj.cast::<Group>() {
        return Ok(Loc::Group(g.get().group.clone()));
    }
    // A Sample (or anything else carrying a root).
    let root = obj.getattr("root")?;
    Ok(Loc::Group(root.cast::<Group>()?.get().group.clone()))
}

/// The group a `Group` proxy or a `Sample` (its root) stands for.
pub(crate) fn group_of(obj: &Bound<'_, PyAny>) -> PyResult<medh5::hdf5::Group> {
    match loc_of(obj)? {
        Loc::Group(g) => Ok(g),
        Loc::Dataset(_) => Err(pyo3::exceptions::PyTypeError::new_err("expected a group, not a dataset")),
    }
}

fn file_root(g: &medh5::hdf5::Group) -> R<medh5::hdf5::Group> {
    Ok(g.file()?.as_group()?)
}

/// Spec attribute names as the engine's `'static` table entries (names a
/// caller supplies outside the tables are interned once).
fn static_names(names: Vec<String>) -> Vec<&'static str> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static INTERNED: Mutex<Option<HashMap<String, &'static str>>> = Mutex::new(None);
    let mut guard = INTERNED.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let table = guard.get_or_insert_with(HashMap::new);
    names.into_iter().map(|n| *table.entry(n.clone()).or_insert_with(|| Box::leak(n.into_boxed_str()))).collect()
}

/// The names a caller gave, or `None` for the root's own: those are read
/// from the file, by [`attr_names_of`], where the reading is detached.
fn attr_names_arg(attr_names: Option<&Bound<'_, PyAny>>) -> PyResult<Option<medh5::integrity::AttrNameMap>> {
    match attr_names.filter(|a| !a.is_none()) {
        None => Ok(None),
        Some(map) => {
            let mut out = medh5::integrity::AttrNameMap::new();
            for item in map.call_method0("items")?.try_iter()? {
                let (k, v): (String, Vec<String>) = item?.extract()?;
                out.insert(k, static_names(v));
            }
            Ok(Some(out))
        }
    }
}

fn attr_names_of(
    root: &medh5::hdf5::Group,
    given: Option<medh5::integrity::AttrNameMap>,
) -> medh5::Result<medh5::integrity::AttrNameMap> {
    match given {
        Some(names) => Ok(names),
        None => medh5::sample::attr_name_map_of(root),
    }
}

/// The digest of a stored dataset over its decompressed content (§13.1).
#[pyfunction]
#[pyo3(signature = (dataset, path=None, algo="sha256"))]
fn dataset_digest(py: Python<'_>, dataset: &Bound<'_, PyAny>, path: Option<String>, algo: &str) -> R<String> {
    let ds = crate::nodes::dataset_arg(dataset)?.get().ds().clone();
    Ok(py.detach(move || {
        let path = path.unwrap_or_else(|| ds.name().trim_start_matches('/').to_string());
        digest::dataset_digest(&ds, &path, algo)
    })?)
}

/// Canonical JSON over an object's named attributes (§13.2).
#[pyfunction]
fn canonical_attrs(obj: &Bound<'_, PyAny>, names: Vec<String>) -> R<String> {
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let loc = loc_of(obj)?;
    Ok(crate::h5(obj.py(), || match &loc {
        Loc::Dataset(d) => digest::canonical_attrs(d, &refs),
        Loc::Group(g) => digest::canonical_attrs(g, &refs),
    })?)
}

/// Return the digest of the canonical JSON of an object's named attributes (§13.2).
///
/// `obj` is a `Dataset`, a `Group` or a `Sample` (its root). Only those of
/// `names` the object carries are hashed, in sorted order.
#[pyfunction]
#[pyo3(signature = (obj, names, algo="sha256"))]
fn attrs_digest(obj: &Bound<'_, PyAny>, names: Vec<String>, algo: &str) -> R<String> {
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let loc = loc_of(obj)?;
    Ok(crate::h5(obj.py(), || match &loc {
        Loc::Dataset(d) => digest::attrs_digest(d, &refs, algo),
        Loc::Group(g) => digest::attrs_digest(g, &refs, algo),
    })?)
}

/// An object's path relative to a sample root (the file root by default).
#[pyfunction]
#[pyo3(signature = (node, root=None))]
fn relative_path(py: Python<'_>, node: &Bound<'_, PyAny>, root: Option<&Bound<'_, PyAny>>) -> R<String> {
    let loc = loc_of(node)?;
    let given = root.filter(|r| !r.is_none()).map(group_of).transpose()?;
    crate::h5(py, move || -> R<String> {
        let (name, own_root) = match &loc {
            Loc::Dataset(d) => (d.name(), file_root(&d.file()?.as_group()?)?),
            Loc::Group(g) => (g.name(), file_root(g)?),
        };
        Ok(digest::relative_path(&name, &given.unwrap_or(own_root)))
    })
}

/// The digest of an object group: its datasets and canonical attributes.
#[pyfunction]
#[pyo3(signature = (group, algo="sha256", *, root=None))]
fn group_digest(py: Python<'_>, group: &Bound<'_, PyAny>, algo: &str, root: Option<&Bound<'_, PyAny>>) -> R<String> {
    let g = group_of(group)?;
    let given = root.filter(|r| !r.is_none()).map(group_of).transpose()?;
    py.detach(move || -> R<String> {
        let root = match given {
            Some(root) => root,
            None => file_root(&g)?,
        };
        Ok(digest::group_digest(&g, &root, algo)?)
    })
}

/// The Merkle root over stored digests, `meta` and canonical attributes.
#[pyfunction]
#[pyo3(signature = (root, attr_names=None, *, algo=None))]
fn compute_content_id(
    py: Python<'_>,
    root: &Bound<'_, PyAny>,
    attr_names: Option<&Bound<'_, PyAny>>,
    algo: Option<String>,
) -> R<String> {
    let g = group_of(root)?;
    let given = attr_names_arg(attr_names)?;
    Ok(py.detach(move || -> medh5::Result<String> {
        let names = attr_names_of(&g, given)?;
        let algo = match algo {
            Some(a) => a,
            None => digest::root_algo(&g)?,
        };
        digest::compute_content_id(&g, &names, &algo, None)
    })?)
}

/// Whether the dataset at `path` under `root` matches its stored `digest` attribute (§13.1).
///
/// The digest is recomputed over the decompressed content. A dataset without a
/// `digest` is `False`; a malformed one raises (E703).
#[pyfunction]
fn verify_object(py: Python<'_>, root: &Bound<'_, PyAny>, path: &str) -> R<bool> {
    let g = group_of(root)?;
    Ok(py.detach(move || verify::verify_object(&g, path))?)
}

/// Return the names of the sampling index entries whose `source_digest` no longer matches (§13.3).
///
/// An entry is stale when it declares no `source_digest`, when its annotation
/// is gone, or when the annotation's digest has changed. A stale entry is not a
/// file error: readers ignore it and rebuild.
#[pyfunction]
fn stale_index_entries<'py>(py: Python<'py>, root: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
    let g = group_of(root)?;
    Ok(PyTuple::new(py, py.detach(move || verify::stale_index_entries(&g))?)?)
}

#[pyfunction]
#[pyo3(signature = (root, attr_names=None, *, partial=None, check_content_id=true))]
fn verify_root<'py>(
    py: Python<'py>,
    root: &Bound<'py, PyAny>,
    attr_names: Option<&Bound<'py, PyAny>>,
    partial: Option<Vec<String>>,
    check_content_id: bool,
) -> R<Bound<'py, PyAny>> {
    let g = group_of(root)?;
    let given = attr_names_arg(attr_names)?;
    let result = py.detach(move || -> medh5::Result<_> {
        let names = attr_names_of(&g, given)?;
        verify::verify_root(&g, Some(&names), partial.as_deref(), check_content_id)
    })?;
    Ok(verify_result_to_py(py, &result)?)
}

/// Return every stored chunk of a dataset as `bytes`, exactly as it sits on disk.
///
/// The chunks are still compressed, in storage order; a dataset that is not
/// chunked gives an empty list.
#[pyfunction]
fn raw_chunks<'py>(py: Python<'py>, dataset: &Bound<'py, PyAny>) -> R<Bound<'py, PyList>> {
    let ds = crate::nodes::dataset_arg(dataset)?.get().ds().clone();
    let chunks = py.detach(move || verify::raw_chunks(&ds))?;
    Ok(PyList::new(py, chunks.iter().map(|c| PyBytes::new(py, c)))?)
}

/// Return the paths at which two group trees differ byte for byte; empty means a pure copy.
///
/// Structure, attributes and *stored* chunk bytes are compared, so a
/// recompressed copy differs. Links are compared as a copy keeps them: a soft
/// link by the path it holds, not followed.
#[pyfunction]
fn subtrees_identical<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
    let (a, b) = (group_of(a)?, group_of(b)?);
    Ok(PyTuple::new(py, py.detach(move || verify::subtrees_identical(&a, &b))?)?)
}

#[pyfunction]
#[pyo3(signature = (paths, *, rebuild_index=false, rewrite_digests=false, reason=None, performed_by=None, max_coords=None))]
fn fix_paths<'py>(
    py: Python<'py>,
    paths: Vec<PathBuf>,
    rebuild_index: bool,
    rewrite_digests: bool,
    reason: Option<String>,
    performed_by: Option<String>,
    max_coords: Option<usize>,
) -> R<Bound<'py, PyList>> {
    let options = repair::FixOptions { rebuild_index, rewrite_digests, reason, performed_by, max_coords };
    let done = py.detach(move || {
        let refs: Vec<&std::path::Path> = paths.iter().map(PathBuf::as_path).collect();
        repair::fix_paths(&refs, &options)
    })?;
    let out = PyList::empty(py);
    for r in done {
        out.append(json_to_py(py, &r.to_json())?)?;
    }
    Ok(out)
}

/// The `digest` of every dataset under a sample root that carries one.
#[pyfunction]
#[pyo3(signature = (root, skip=vec!["index".to_string()]))]
fn collect_digests<'py>(py: Python<'py>, root: &Bound<'py, PyAny>, skip: Vec<String>) -> R<Bound<'py, PyDict>> {
    let g = group_of(root)?;
    let found = py.detach(move || {
        let skip: Vec<&str> = skip.iter().map(String::as_str).collect();
        medh5::integrity::digest::collect_digests(&g, &skip)
    })?;
    let out = PyDict::new(py);
    for (path, digest) in found {
        out.set_item(path, digest)?;
    }
    Ok(out)
}

// -- what the result types say of themselves ------------------------------------------

fn opt_text(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    obj.getattr(name)?.extract()
}

/// The engine value of a `VerifyResult` dataclass.
fn verify_result_arg(obj: &Bound<'_, PyAny>) -> PyResult<verify::VerifyResult> {
    let list = |name: &str| strings(&obj.getattr(name)?);
    Ok(verify::VerifyResult {
        checked: list("checked")?,
        mismatched: list("mismatched")?,
        undigested: list("undigested")?,
        malformed: list("malformed")?,
        content_id_declared: opt_text(obj, "content_id_declared")?,
        content_id_computed: opt_text(obj, "content_id_computed")?,
        stale_index: list("stale_index")?,
        unattested: list("unattested")?,
        unverifiable: list("unverifiable")?,
    })
}

/// The engine value of a `Diagnosis` dataclass.
fn diagnosis_arg(obj: &Bound<'_, PyAny>) -> PyResult<repair::Diagnosis> {
    let list = |name: &str| strings(&obj.getattr(name)?);
    Ok(repair::Diagnosis {
        path: obj.getattr("path")?.extract()?,
        mismatched: list("mismatched")?,
        malformed: list("malformed")?,
        undigested: list("undigested")?,
        unattested: list("unattested")?,
        stale_index: list("stale_index")?,
        missing_index: list("missing_index")?,
        content_id_ok: obj.getattr("content_id_ok")?.extract()?,
    })
}

/// The engine value of a `Repair` dataclass.
fn repair_arg(obj: &Bound<'_, PyAny>) -> PyResult<repair::Repair> {
    Ok(repair::Repair {
        path: obj.getattr("path")?.extract()?,
        diagnosis: diagnosis_arg(&obj.getattr("diagnosis")?)?,
        rebuilt_index: strings(&obj.getattr("rebuilt_index")?)?,
        rewrote_digests: obj.getattr("rewrote_digests")?.is_truthy()?,
        content_id: opt_text(obj, "content_id")?,
        notes: strings(&obj.getattr("notes")?)?,
    })
}

/// A verification's summary: its verdict, and what failed.
#[pyfunction]
fn integrity_verify_json<'py>(py: Python<'py>, result: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &verify_result_arg(result)?.summary())
}

/// A diagnosis as `diagnose --json` prints it.
#[pyfunction]
fn integrity_diagnosis_json<'py>(py: Python<'py>, diagnosis: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &diagnosis_arg(diagnosis)?.to_json())
}

/// A repair as `fix --json` prints it.
#[pyfunction]
fn integrity_repair_json<'py>(py: Python<'py>, repair: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &repair_arg(repair)?.to_json())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(collect_digests, m)?)?;
    m.add_function(wrap_pyfunction!(integrity_verify_json, m)?)?;
    m.add_function(wrap_pyfunction!(integrity_diagnosis_json, m)?)?;
    m.add_function(wrap_pyfunction!(integrity_repair_json, m)?)?;
    for f in [
        wrap_pyfunction!(array_digest, m)?,
        wrap_pyfunction!(digest_bytes, m)?,
        wrap_pyfunction!(parse_digest, m)?,
        wrap_pyfunction!(diagnose, m)?,
        wrap_pyfunction!(fix, m)?,
        wrap_pyfunction!(fix_paths, m)?,
        wrap_pyfunction!(dataset_digest, m)?,
        wrap_pyfunction!(canonical_attrs, m)?,
        wrap_pyfunction!(attrs_digest, m)?,
        wrap_pyfunction!(relative_path, m)?,
        wrap_pyfunction!(group_digest, m)?,
        wrap_pyfunction!(compute_content_id, m)?,
        wrap_pyfunction!(verify_object, m)?,
        wrap_pyfunction!(stale_index_entries, m)?,
        wrap_pyfunction!(verify_root, m)?,
        wrap_pyfunction!(raw_chunks, m)?,
        wrap_pyfunction!(subtrees_identical, m)?,
    ] {
        m.add_function(f)?;
    }
    m.add("ATTESTED_GROUPS", PyTuple::new(m.py(), verify::ATTESTED_GROUPS)?)?;
    m.add("DEFAULT_ALGO", medh5::digest::DEFAULT_ALGO)?;
    m.add("DIGEST_ALGOS", PyTuple::new(m.py(), medh5::digest::DIGEST_ALGOS)?)?;
    m.add("STREAM_BYTES", digest::STREAM_BYTES)?;
    Ok(())
}
