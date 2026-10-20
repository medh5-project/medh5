//! Stored objects: `Dataset`, `Group` and their `attrs`.
//!
//! These stand where 1.x handed out `h5py` objects --- `Image.dataset`, the
//! return value of `SampleWriter.add_image`, `SampleWriter.handle` --- and
//! answer the same questions: shape, dtype, chunks, filters, attributes and
//! `node[...]`.  They read through the engine's HDF5 library, so there is one
//! library instance in the process and no second set of handles to close.

use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyEllipsis, PyList, PyTuple};

use medh5::array::{Index, Slice};
use medh5::h5::{attrs, data, ops};

use crate::convert::{attr_to_py, dtype_to_py, index_item, nd_to_py, py_to_attr};
use crate::errors::R;
use crate::h5;

/// A group or a dataset: what `attrs` reads and writes.
#[derive(Clone)]
pub enum Node {
    Group(medh5::hdf5::Group),
    Dataset(medh5::hdf5::Dataset),
}

impl Node {
    fn location(&self) -> &medh5::hdf5::Location {
        match self {
            Node::Group(g) => g,
            Node::Dataset(d) => d,
        }
    }

    fn name(&self) -> String {
        match self {
            Node::Group(g) => g.name(),
            Node::Dataset(d) => d.name(),
        }
    }

    /// The location, refused once its file has been closed.
    fn live(&self) -> R<&medh5::hdf5::Location> {
        let location = self.location();
        medh5::h5::alive(location)?;
        Ok(location)
    }
}

// -- attrs ----------------------------------------------------------------------------

/// An object's attributes, as a mutable mapping.
///
/// Writes go straight to the file, so they only succeed on a file open for
/// writing --- `SampleWriter.handle`, in practice.
#[pyclass(module = "medh5.nodes", name = "Attrs", frozen, mapping)]
pub struct Attrs {
    node: Node,
}

impl Attrs {
    fn names(&self, py: Python<'_>) -> R<Vec<String>> {
        Ok(h5(py, || attrs::names(self.node.location()))?)
    }

    /// Every attribute by name, read in one detached call.
    fn read_all(&self, py: Python<'_>) -> R<Vec<(String, medh5::h5::attrs::AttrValue)>> {
        Ok(h5(py, || -> medh5::Result<_> {
            let mut out = Vec::new();
            for name in attrs::names(self.node.location())? {
                if let Some(v) = attrs::read(self.node.location(), &name)? {
                    out.push((name, v));
                }
            }
            Ok(out)
        })?)
    }
}

fn missing(name: &str) -> PyErr {
    PyKeyError::new_err(format!("Unable to locate attribute {}", medh5::json::repr_str(name)))
}

#[pymethods]
impl Attrs {
    fn __getitem__<'py>(&self, py: Python<'py>, name: &str) -> R<Bound<'py, PyAny>> {
        match h5(py, || attrs::read(self.node.location(), name))? {
            Some(value) => Ok(attr_to_py(py, &value)?),
            None => Err(missing(name).into()),
        }
    }
    fn __setitem__(&self, name: &str, value: &Bound<'_, PyAny>) -> R<()> {
        let py = value.py();
        let value = py_to_attr(value)?;
        h5(py, || Ok(attrs::write(self.node.live()?, name, &value)?))
    }
    fn __delitem__(&self, py: Python<'_>, name: &str) -> R<()> {
        h5(py, || {
            let location = self.node.live()?;
            if !attrs::has(location, name) {
                return Err(missing(name).into());
            }
            Ok(attrs::delete(location, name)?)
        })
    }
    fn __contains__(&self, name: &Bound<'_, PyAny>) -> R<bool> {
        let py = name.py();
        let name = name.extract::<String>().ok();
        h5(py, || {
            let location = self.node.live()?;
            Ok(name.is_some_and(|n| attrs::has(location, &n)))
        })
    }
    fn __len__(&self, py: Python<'_>) -> R<usize> {
        Ok(self.names(py)?.len())
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(PyList::new(py, self.names(py)?)?.as_any().try_iter()?.into_any())
    }
    fn keys(&self, py: Python<'_>) -> R<Vec<String>> {
        self.names(py)
    }
    fn values<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for (_, v) in self.read_all(py)? {
            out.append(attr_to_py(py, &v)?)?;
        }
        Ok(out)
    }
    fn items<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for (name, v) in self.read_all(py)? {
            out.append((name, attr_to_py(py, &v)?))?;
        }
        Ok(out)
    }
    #[pyo3(signature = (name, default=None))]
    fn get<'py>(&self, py: Python<'py>, name: &str, default: Option<Bound<'py, PyAny>>) -> R<Bound<'py, PyAny>> {
        match h5(py, || attrs::read(self.node.location(), name))? {
            Some(value) => Ok(attr_to_py(py, &value)?),
            None => Ok(default.unwrap_or_else(|| py.None().into_bound(py))),
        }
    }
    /// The attributes as a plain `dict`.
    fn to_dict<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (name, v) in self.read_all(py)? {
            out.set_item(&name, attr_to_py(py, &v)?)?;
        }
        Ok(out)
    }
    fn __repr__(&self, py: Python<'_>) -> String {
        h5(py, || {
            if self.node.location().is_valid() {
                format!("<Attributes of {}>", medh5::json::repr_str(&self.node.name()))
            } else {
                "<Attributes of a closed object>".into()
            }
        })
    }
}

// -- Dataset ------------------------------------------------------------------------------

/// A `Dataset` argument, refusing anything else --- an `h5py.Dataset` above
/// all --- with what to pass instead.
pub fn dataset_arg<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Dataset>> {
    if let Ok(ds) = obj.cast::<Dataset>() {
        return Ok(ds.clone());
    }
    let ty = obj.get_type();
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "expected a medh5 Dataset (`Sample.root[...]`, `Image.dataset`, or what a writer's `add_*` returns), not \
         {}.{}; medh5 reads files through its own engine, not h5py",
        ty.module()?,
        ty.qualname()?
    )))
}

/// A stored dataset: shape, dtype, chunks, filters, attributes and `ds[...]`.
#[pyclass(module = "medh5.nodes", name = "Dataset", frozen)]
pub struct Dataset {
    /// `None` only once dropped: see `Drop`.
    handle: Option<medh5::hdf5::Dataset>,
}

impl Dataset {
    pub fn wrap(ds: medh5::hdf5::Dataset) -> Self {
        Dataset { handle: Some(ds) }
    }

    pub fn ds(&self) -> &medh5::hdf5::Dataset {
        self.handle.as_ref().expect("a dataset's handle is taken only when it is dropped")
    }

    /// The dataset, refused once its file has been closed.
    fn live(&self) -> R<&medh5::hdf5::Dataset> {
        medh5::h5::alive(self.ds())?;
        Ok(self.ds())
    }

    fn shape_of(&self, py: Python<'_>) -> R<Vec<usize>> {
        h5(py, || Ok(self.live()?.shape()))
    }
}

impl Drop for Dataset {
    /// Closing a dataset's last handle flushes its chunk cache --- compression
    /// and writes, tens of milliseconds for a volume --- so it runs with the
    /// GIL released, as the write that filled the cache did.
    fn drop(&mut self) {
        if let Some(ds) = self.handle.take() {
            Python::attach(|py| py.detach(move || drop(ds)));
        }
    }
}

/// A string dataset as an object array of `str` in its stored shape (0-d for
/// a scalar): what `read()` and `numpy.asarray(ds)` return and `ds[key]`
/// indexes, so a key means what it means to NumPy, as it did to `h5py` ---
/// `ds[0]` of a 2-D dataset is its first row, and a scalar takes `()` or `...`.
fn strings_array<'py>(py: Python<'py>, ds: &medh5::hdf5::Dataset) -> R<Bound<'py, PyAny>> {
    let source = ds.clone();
    let (values, shape) = py.detach(move || data::read_strings(&source).map(|v| (v, source.shape())))?;
    let flat = crate::convert::numpy(py)?.call_method1("empty", (values.len(), "O"))?;
    flat.set_item(PyEllipsis::get(py), PyList::new(py, values)?)?;
    Ok(flat.call_method1("reshape", (PyTuple::new(py, shape)?,))?)
}

/// Expand a NumPy-style key against `ndim` axes.
fn dataset_key(key: &Bound<'_, PyAny>, ndim: usize) -> PyResult<Vec<Index>> {
    let items: Vec<Bound<'_, PyAny>> =
        if let Ok(t) = key.cast::<PyTuple>() { t.iter().collect() } else { vec![key.clone()] };
    let ellipsis = PyEllipsis::get(key.py());
    let ellipses = items.iter().filter(|i| i.is(ellipsis)).count();
    if ellipses > 1 {
        return Err(PyIndexError::new_err("an index can only have a single ellipsis ('...')"));
    }
    let explicit = items.len() - ellipses;
    if explicit > ndim {
        return Err(PyIndexError::new_err(format!(
            "too many indices for dataset: dataset is {ndim}-dimensional, but {explicit} were indexed"
        )));
    }
    let mut out = Vec::with_capacity(ndim);
    for item in &items {
        if item.is(ellipsis) {
            for _ in 0..(ndim - explicit) {
                out.push(Index::Slice(Slice::full()));
            }
        } else {
            out.push(index_item(item)?);
        }
    }
    while out.len() < ndim {
        out.push(Index::Slice(Slice::full()));
    }
    Ok(out)
}

#[pymethods]
impl Dataset {
    #[getter]
    fn name(&self, py: Python<'_>) -> R<String> {
        h5(py, || Ok(self.live()?.name()))
    }
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.shape_of(py)?)?)
    }
    #[getter]
    fn ndim(&self, py: Python<'_>) -> R<usize> {
        Ok(self.shape_of(py)?.len())
    }
    #[getter]
    fn size(&self, py: Python<'_>) -> R<usize> {
        Ok(self.shape_of(py)?.iter().product())
    }
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        match h5(py, || -> R<_> { Ok((!data::is_strings(self.live()?)).then(|| data::dtype(self.ds())).transpose()?) })?
        {
            None => Ok(crate::convert::numpy(py)?.call_method1("dtype", ("O",))?),
            Some(dtype) => Ok(dtype_to_py(py, dtype)?),
        }
    }
    #[getter]
    fn nbytes(&self, py: Python<'_>) -> R<usize> {
        h5(py, || Ok(data::nbytes(self.live()?)?))
    }
    #[getter]
    fn chunks<'py>(&self, py: Python<'py>) -> R<Option<Bound<'py, PyTuple>>> {
        let chunks = h5(py, || -> R<_> { Ok(data::chunks(self.live()?)) })?;
        Ok(chunks.map(|c| PyTuple::new(py, c)).transpose()?)
    }
    /// The filter pipeline, as `(filter id, client data)` pairs.
    #[getter]
    fn filters<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for (id, values) in h5(py, || -> R<_> { Ok(data::filters(self.live()?)?) })? {
            out.append((id, values))?;
        }
        Ok(out)
    }
    /// Bytes on disk (after compression).
    #[getter]
    fn storage_size(&self, py: Python<'_>) -> R<u64> {
        h5(py, || Ok(self.live()?.storage_size()))
    }
    #[getter]
    fn attrs(&self) -> Attrs {
        Attrs { node: Node::Dataset(self.ds().clone()) }
    }
    fn __len__(&self, py: Python<'_>) -> R<usize> {
        Ok(self.shape_of(py)?.first().copied().ok_or_else(|| PyTypeError::new_err("len() of unsized object"))?)
    }
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
        let (strings, ndim) = h5(py, || -> R<_> { Ok((data::is_strings(self.live()?), self.ds().shape().len())) })?;
        if strings {
            return Ok(strings_array(py, self.ds())?.get_item(key)?);
        }
        let ds = self.ds().clone();
        if ndim == 0 {
            return Ok(nd_to_py(py, py.detach(move || data::read(&ds))?).call_method1("__getitem__", (key,))?);
        }
        let index = dataset_key(key, ndim)?;
        let block = py.detach(move || data::read_region(&ds, &index))?;
        Ok(nd_to_py(py, block))
    }
    /// The whole dataset as an array: a numeric dataset as its dtype, a
    /// string dataset as an object array of `str`.
    fn read<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        if h5(py, || -> R<_> { Ok(data::is_strings(self.live()?)) })? {
            return strings_array(py, self.ds());
        }
        let ds = self.ds().clone();
        Ok(nd_to_py(py, py.detach(move || data::read(&ds))?))
    }
    /// `numpy.asarray(ds)`: the whole dataset, read.  A read always makes a
    /// new array, so `copy=False` is refused (`ValueError`), as `h5py`
    /// refuses it; a `dtype` converts the array read, without a second copy
    /// where it is already of that type.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<&Bound<'py, PyAny>>,
    ) -> R<Bound<'py, PyAny>> {
        if let Some(c) = copy.filter(|c| !c.is_none()) {
            if !c.is_truthy()? {
                return Err(PyValueError::new_err(
                    "a Dataset's array is read from the file, so it cannot be had without a copy (copy=False)",
                )
                .into());
            }
        }
        let array = self.read(py)?;
        match dtype {
            Some(d) if !d.is_none() => {
                let kwargs = PyDict::new(py);
                kwargs.set_item("copy", false)?;
                Ok(array.call_method("astype", (d,), Some(&kwargs))?)
            }
            _ => Ok(array),
        }
    }
    fn __repr__(&self, py: Python<'_>) -> String {
        h5(py, || {
            if self.ds().is_valid() {
                format!("<medh5 Dataset {} shape {:?}>", medh5::json::repr_str(&self.ds().name()), self.ds().shape())
            } else {
                "<closed medh5 Dataset>".into()
            }
        })
    }
}

// -- Group --------------------------------------------------------------------------------

/// A stored group: members by name (or path), and attributes.
#[pyclass(module = "medh5.nodes", name = "Group", frozen, mapping)]
pub struct Group {
    pub group: medh5::hdf5::Group,
}

impl Group {
    pub fn wrap(group: medh5::hdf5::Group) -> Self {
        Group { group }
    }

    /// The member at a `/`-separated path below this group.
    fn resolve<'py>(&self, py: Python<'py>, path: &str) -> R<Option<Bound<'py, PyAny>>> {
        Ok(match h5(py, || self.member(path))? {
            Some(Node::Group(g)) => Some(Bound::new(py, Group::wrap(g))?.into_any()),
            Some(Node::Dataset(d)) => Some(Bound::new(py, Dataset::wrap(d))?.into_any()),
            None => None,
        })
    }

    fn member(&self, path: &str) -> R<Option<Node>> {
        medh5::h5::alive(&self.group)?;
        let trimmed = path.trim_matches('/');
        if trimmed.is_empty() {
            return Ok(Some(Node::Group(self.group.clone())));
        }
        let (parent, leaf) = match trimmed.rsplit_once('/') {
            Some((p, l)) => match ops::child_group(&self.group, p) {
                Some(g) => (g, l),
                None => return Ok(None),
            },
            None => (self.group.clone(), trimmed),
        };
        if let Some(g) = ops::child_group(&parent, leaf) {
            return Ok(Some(Node::Group(g)));
        }
        Ok(ops::child_dataset(&parent, leaf).map(Node::Dataset))
    }

    fn members(&self, py: Python<'_>) -> R<Vec<String>> {
        Ok(h5(py, || ops::members(&self.group))?)
    }
}

#[pymethods]
impl Group {
    #[getter]
    fn name(&self, py: Python<'_>) -> R<String> {
        h5(py, || {
            medh5::h5::alive(&self.group)?;
            Ok(self.group.name())
        })
    }
    #[getter]
    fn attrs(&self) -> Attrs {
        Attrs { node: Node::Group(self.group.clone()) }
    }
    fn keys(&self, py: Python<'_>) -> R<Vec<String>> {
        self.members(py)
    }
    fn values<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for name in self.members(py)? {
            if let Some(item) = self.resolve(py, &name)? {
                out.append(item)?;
            }
        }
        Ok(out)
    }
    fn items<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for name in self.members(py)? {
            if let Some(item) = self.resolve(py, &name)? {
                out.append((name, item))?;
            }
        }
        Ok(out)
    }
    fn __getitem__<'py>(&self, py: Python<'py>, path: &str) -> R<Bound<'py, PyAny>> {
        match self.resolve(py, path)? {
            Some(found) => Ok(found),
            None => Err(PyKeyError::new_err(format!(
                "no object {} in {}",
                medh5::json::repr_str(path),
                medh5::json::repr_str(&h5(py, || self.group.name()))
            ))
            .into()),
        }
    }
    #[pyo3(signature = (path, default=None))]
    fn get<'py>(&self, py: Python<'py>, path: &str, default: Option<Bound<'py, PyAny>>) -> R<Bound<'py, PyAny>> {
        Ok(self.resolve(py, path)?.unwrap_or_else(|| default.unwrap_or_else(|| py.None().into_bound(py))))
    }
    fn __contains__(&self, py: Python<'_>, path: &Bound<'_, PyAny>) -> R<bool> {
        match path.extract::<String>() {
            Ok(p) => Ok(h5(py, || self.member(&p))?.is_some()),
            Err(_) => Ok(false),
        }
    }
    fn __delitem__(&self, py: Python<'_>, path: &str) -> R<()> {
        h5(py, || {
            medh5::h5::alive(&self.group)?;
            let trimmed = path.trim_matches('/');
            let (parent, leaf) = match trimmed.rsplit_once('/') {
                Some((p, l)) => (self.group.group(p)?, l.to_string()),
                None => (self.group.clone(), trimmed.to_string()),
            };
            if !ops::exists(&parent, &leaf) {
                return Err(PyKeyError::new_err(medh5::json::repr_str(path)).into());
            }
            Ok(ops::unlink(&parent, &leaf)?)
        })
    }
    fn __len__(&self, py: Python<'_>) -> R<usize> {
        Ok(self.members(py)?.len())
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(PyList::new(py, self.members(py)?)?.as_any().try_iter()?.into_any())
    }
    fn __repr__(&self, py: Python<'_>) -> String {
        h5(py, || {
            if self.group.is_valid() {
                format!("<medh5 Group {}>", medh5::json::repr_str(&self.group.name()))
            } else {
                "<closed medh5 Group>".into()
            }
        })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Attrs>()?;
    m.add_class::<Dataset>()?;
    m.add_class::<Group>()?;
    Ok(())
}
