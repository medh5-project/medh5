//! The engine's one way of replacing a file (1.0 §14.4), for what Python
//! writes: a document written whole, and a file a foreign writer (nibabel,
//! pydicom) writes to a temporary the engine then renames over its target.
//! Either way a symbolic link at the target is kept and the file it names
//! replaced, that file's mode kept, and the data flushed.

use std::path::{Path, PathBuf};

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

use medh5::h5::file::StagedFile;

use crate::errors::R;
use crate::exclusive::Exclusive;

/// `data` as `path`'s whole content: a `str` as UTF-8, its `\n` kept on
/// every platform, `bytes` as given.
#[pyfunction]
fn write_atomic(py: Python<'_>, path: PathBuf, data: &Bound<'_, PyAny>) -> R<()> {
    let bytes: Vec<u8> = if let Ok(text) = data.cast::<PyString>() {
        text.to_str()?.as_bytes().to_vec()
    } else if let Ok(raw) = data.cast::<PyBytes>() {
        raw.as_bytes().to_vec()
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "write_atomic writes str or bytes, not {}",
            data.get_type().name()?
        ))
        .into());
    };
    Ok(py.detach(move || medh5::h5::file::write_atomic(&path, &bytes))?)
}

/// A replacement of a file being written by someone else: write `path`,
/// then `commit()`.  As a context manager it commits when the block ends
/// without an exception, and aborts otherwise.  Used by one thread at a time
/// (E4): a commit or abort while another runs is refused.
#[pyclass(module = "medh5._core", name = "StagedFileHandle", frozen)]
pub struct StagedFileHandle {
    inner: Exclusive<Option<StagedFile>>,
    path: PathBuf,
    target: PathBuf,
}

impl StagedFileHandle {
    /// Commit, holding the handle across the rename (as a cache writer's
    /// commit does): a commit or abort meanwhile is refused, not told the
    /// file is finished before it is.  `None` when it already was.
    fn commit_held(&self, py: Python<'_>) -> R<Option<()>> {
        let mut held = self.inner.write()?;
        let Some(staged) = held.take() else { return Ok(None) };
        let committed = py.detach(move || staged.commit());
        drop(held);
        Ok(Some(committed?))
    }
}

#[pymethods]
impl StagedFileHandle {
    /// The temporary, which the foreign writer writes.
    #[getter]
    fn path(&self) -> PathBuf {
        self.path.clone()
    }

    /// The file the commit replaces: the target, its links followed.
    #[getter]
    fn target(&self) -> PathBuf {
        self.target.clone()
    }

    /// Rename the written temporary over the target.
    fn commit(&self, py: Python<'_>) -> R<()> {
        self.commit_held(py)?
            .ok_or_else(|| medh5::Error::File("this staged file has already been committed or aborted".into()).into())
    }

    /// Remove the temporary, leaving the target as it was.  Aborting a
    /// committed or aborted file does nothing.
    fn abort(&self) -> R<()> {
        if let Some(staged) = self.inner.write()?.take() {
            staged.abort();
        }
        Ok(())
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (exc_type=None, _exc=None, _tb=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> R<bool> {
        if exc_type.is_some_and(|t| !t.is_none()) {
            self.abort()?;
            return Ok(false);
        }
        self.commit_held(py)?;
        Ok(false)
    }
}

/// Stage a replacement of `path`: a new, empty temporary beside the file it
/// names, keeping its suffix.
#[pyfunction]
fn stage_file(path: PathBuf) -> R<StagedFileHandle> {
    let staged = StagedFile::create(&path)?;
    let (tmp, target) = (staged.path().to_path_buf(), staged.target().to_path_buf());
    Ok(StagedFileHandle { inner: Exclusive::new("a StagedFileHandle", Some(staged)), path: tmp, target })
}

/// Whether two paths name one existing file, through `..`, a symbolic or a
/// hard link.
#[pyfunction]
fn same_file(a: PathBuf, b: PathBuf) -> bool {
    medh5::h5::file::same_file(Path::new(&a), Path::new(&b))
}

/// An ITK transform file's entries, each `(type, parameters, fixed
/// parameters)`.
type ItkEntries = Vec<(String, Vec<f64>, Vec<f64>)>;

/// The transforms of an ITK HDF5 transform file, in its order, each as
/// `(type, parameters, fixed parameters)`.
#[pyfunction]
fn read_itk_h5(py: Python<'_>, path: PathBuf) -> R<ItkEntries> {
    let entries = py.detach(move || medh5::convert::read_itk_h5(&path))?;
    Ok(entries.into_iter().map(|e| (e.name, e.parameters, e.fixed_parameters)).collect())
}

/// Write `entries`, each `(type, parameters, fixed parameters)`, as the ITK
/// HDF5 transform file `path`, replacing it atomically.
#[pyfunction]
fn write_itk_h5(py: Python<'_>, path: PathBuf, entries: ItkEntries) -> R<()> {
    let entries: Vec<_> = entries.into_iter().map(|(n, p, f)| medh5::convert::ItkEntry::new(n, p, f)).collect();
    Ok(py.detach(move || medh5::convert::write_itk_h5(&path, &entries))?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StagedFileHandle>()?;
    m.add_function(wrap_pyfunction!(read_itk_h5, m)?)?;
    m.add_function(wrap_pyfunction!(write_itk_h5, m)?)?;
    m.add_function(wrap_pyfunction!(write_atomic, m)?)?;
    m.add_function(wrap_pyfunction!(stage_file, m)?)?;
    m.add_function(wrap_pyfunction!(same_file, m)?)?;
    Ok(())
}
