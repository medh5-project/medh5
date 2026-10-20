//! Engine errors as the Python exceptions 1.x raised.
//!
//! The `MEDH5Error` family lives in `medh5/errors.py` --- ordinary Python
//! classes, so they pickle, subclass and print like any exception --- and is
//! looked up here once.  Every other variant maps onto the built-in of the
//! same meaning (`KeyError`, `ValueError`, ...).

use pyo3::exceptions::{
    PyIndexError, PyKeyError, PyNotImplementedError, PyOSError, PyRuntimeError, PyTypeError, PyValueError,
};
use std::path::Path;

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

/// Why a binding call failed: the engine refused, or Python did.
pub enum BindError {
    Engine(medh5::Error),
    Py(PyErr),
}

/// The binding layer's result type; converts into a raised exception.
pub type R<T> = Result<T, BindError>;

impl From<medh5::Error> for BindError {
    fn from(e: medh5::Error) -> Self {
        BindError::Engine(e)
    }
}

impl From<PyErr> for BindError {
    fn from(e: PyErr) -> Self {
        BindError::Py(e)
    }
}

impl From<pyo3::pycell::PyBorrowError> for BindError {
    fn from(e: pyo3::pycell::PyBorrowError) -> Self {
        BindError::Py(e.into())
    }
}

impl From<pyo3::pycell::PyBorrowMutError> for BindError {
    fn from(e: pyo3::pycell::PyBorrowMutError) -> Self {
        BindError::Py(e.into())
    }
}

impl From<std::io::Error> for BindError {
    fn from(e: std::io::Error) -> Self {
        BindError::Engine(e.into())
    }
}

impl From<medh5::hdf5::Error> for BindError {
    fn from(e: medh5::hdf5::Error) -> Self {
        BindError::Engine(e.into())
    }
}

impl From<std::convert::Infallible> for BindError {
    fn from(e: std::convert::Infallible) -> Self {
        match e {}
    }
}

impl From<ndarray::ShapeError> for BindError {
    fn from(e: ndarray::ShapeError) -> Self {
        BindError::Engine(e.into())
    }
}

impl From<BindError> for PyErr {
    fn from(e: BindError) -> Self {
        match e {
            BindError::Py(err) => err,
            BindError::Engine(err) => engine_error(err),
        }
    }
}

fn medh5_class(py: Python<'_>, name: &str) -> PyResult<Py<PyType>> {
    static ERRORS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    let module = ERRORS.get_or_try_init(py, || py.import("medh5.errors").map(|m| m.unbind()))?;
    Ok(module.bind(py).getattr(name)?.cast_into::<PyType>()?.unbind())
}

/// A `MEDH5Error` itself, for a refusal no subclass names: an object used by
/// one thread at a time, found in use.
pub fn medh5_error(py: Python<'_>, message: String) -> PyErr {
    match medh5_class(py, "MEDH5Error").and_then(|class| class.bind(py).call1((message,))) {
        Ok(instance) => PyErr::from_value(instance),
        Err(e) => e,
    }
}

/// The exception a 1.x caller saw for this engine error.
pub fn engine_error(err: medh5::Error) -> PyErr {
    use medh5::Error as E;
    match err {
        E::Key(m) => PyKeyError::new_err(m),
        E::Value(m) => PyValueError::new_err(m),
        E::Type(m) => PyTypeError::new_err(m),
        E::Index(m) => PyIndexError::new_err(m),
        E::Io(m) => io_error(m),
        E::NotImplemented(m) => PyNotImplementedError::new_err(m),
        E::Runtime(m) => PyRuntimeError::new_err(m),
        family => Python::attach(|py| {
            let (name, args): (&str, (String, Option<String>)) = match family {
                E::File(m) => ("MEDH5FileError", (m, None)),
                E::Version(m) => ("MEDH5VersionError", (m, None)),
                E::Schema(m) => ("MEDH5SchemaError", (m, None)),
                E::Integrity(m) => ("MEDH5IntegrityError", (m, None)),
                E::Validation { message, code } => ("MEDH5ValidationError", (message, code)),
                _ => unreachable!("handled above"),
            };
            let class = match medh5_class(py, name) {
                Ok(c) => c,
                Err(e) => return e,
            };
            let made = if name == "MEDH5ValidationError" {
                class.bind(py).call1(args)
            } else {
                class.bind(py).call1((args.0,))
            };
            match made {
                Ok(instance) => PyErr::from_value(instance),
                Err(e) => e,
            }
        }),
    }
}

/// An engine `Io` error as Python would have raised it: an `OSError` with its
/// `errno` (and `filename`; on Windows, its `winerror`), which Python makes
/// the subclass of that errno,
/// where the message is the engine's `[Errno N] reason: 'path'` or the
/// system's `reason (os error N)`; otherwise a failure of HDF5's,
/// `MEDH5IOError`.
fn io_error(message: String) -> PyErr {
    if let Some((errno, reason, filename)) = errno_message(&message) {
        return Python::attach(|py| {
            let filename = filename.and_then(|literal| {
                py.import("ast").and_then(|ast| ast.call_method1("literal_eval", (literal,))).ok().map(Bound::unbind)
            });
            // Windows numbers its errors otherwise: a code the system worded
            // is a Win32 error, which Python maps to its errno (and subclass)
            // when it is given as `winerror`, as Python's own `OSError` there
            // carries it.  A reason the system did not word is Python's own
            // (`[Errno 2] No such file or directory`, as `open` raises it
            // there too), and its code is already an errno.
            let win32 = cfg!(windows) && system_worded(errno, &reason);
            match filename {
                Some(f) if win32 => PyOSError::new_err((errno, reason, f, errno)),
                None if win32 => PyOSError::new_err((errno, reason, py.None(), errno)),
                Some(f) => PyOSError::new_err((errno, reason, f)),
                None => PyOSError::new_err((errno, reason)),
            }
        });
    }
    Python::attach(|py| match medh5_class(py, "MEDH5IOError") {
        Ok(class) => match class.bind(py).call1((message,)) {
            Ok(instance) => PyErr::from_value(instance),
            Err(e) => e,
        },
        Err(e) => e,
    })
}

/// Whether `reason` is the system's own wording of error `code`, as
/// [`medh5::error::os_error`] and `std::io::Error` give it.
fn system_worded(code: i32, reason: &str) -> bool {
    let full = std::io::Error::from_raw_os_error(code).to_string();
    full.strip_suffix(&format!(" (os error {code})")).unwrap_or(&full) == reason
}

/// `(errno, reason, the path as a Python literal)` from `[Errno N] reason:
/// 'path'` ([`medh5::error::os_error`]'s words), or `(errno, reason, None)`
/// from `reason (os error N)` (an `std::io::Error`'s).
pub(crate) fn errno_message(message: &str) -> Option<(i32, String, Option<String>)> {
    if let Some(rest) = message.strip_prefix("[Errno ") {
        let (number, rest) = rest.split_once("] ")?;
        let errno = number.parse().ok()?;
        let at = rest.find(": '").or_else(|| rest.find(": \""));
        return Some(match at {
            Some(at) => (errno, rest[..at].to_string(), Some(rest[at + 2..].to_string())),
            None => (errno, rest.to_string(), None),
        });
    }
    let (reason, tail) = message.rsplit_once(" (os error ")?;
    Some((tail.strip_suffix(')')?.parse().ok()?, reason.to_string(), None))
}

/// `err`, from working on `path`: a `MEDH5FileError` asks the system again
/// why the file failed, and becomes the subclass that names it
/// (`MEDH5FileNotFoundError`, `MEDH5IsADirectoryError`,
/// `MEDH5PermissionError`) with its `errno` and `filename`.  Only the error
/// path asks; the message is the engine's, free of the errno.
pub fn path_error(err: medh5::Error, path: &Path) -> BindError {
    let medh5::Error::File(message) = err else { return err.into() };
    BindError::Py(Python::attach(|py| file_error(py, message, path).unwrap_or_else(|e| e)))
}

/// [`path_error`] for a call over several paths: the first that the system
/// says is missing, a directory or unreadable names the error.
pub fn paths_error(err: medh5::Error, paths: &[impl AsRef<Path>]) -> BindError {
    match paths.iter().find(|p| why(p.as_ref()).0 != Why::Other) {
        Some(path) => path_error(err, path.as_ref()),
        None => err.into(),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Why {
    NotFound,
    Directory,
    Permission,
    Other,
}

/// Why `path` cannot be opened, as the system says, and the system's errno
/// where it gave one.
fn why(path: &Path) -> (Why, Option<i32>) {
    use std::io::ErrorKind;
    let of = |e: &std::io::Error| match e.kind() {
        ErrorKind::NotFound => Why::NotFound,
        ErrorKind::PermissionDenied => Why::Permission,
        _ => Why::Other,
    };
    match std::fs::metadata(path) {
        Err(e) => (of(&e), e.raw_os_error()),
        Ok(meta) if meta.is_dir() => (Why::Directory, None),
        Ok(_) => match std::fs::File::open(path) {
            Err(e) => (of(&e), e.raw_os_error()),
            Ok(_) => (Why::Other, None),
        },
    }
}

fn file_error(py: Python<'_>, message: String, path: &Path) -> PyResult<PyErr> {
    let (why, raw) = why(path);
    let (class, constant) = match why {
        Why::NotFound => ("MEDH5FileNotFoundError", Some("ENOENT")),
        Why::Directory => ("MEDH5IsADirectoryError", Some("EISDIR")),
        Why::Permission => ("MEDH5PermissionError", Some("EACCES")),
        Why::Other => ("MEDH5FileError", None),
    };
    // POSIX's own number; elsewhere (Windows numbers its errors otherwise)
    // Python's for the class.
    let errno: Option<i32> = match constant {
        None => None,
        Some(_) if cfg!(unix) && why != Why::Directory && raw.is_some() => raw,
        Some(name) => Some(py.import("errno")?.getattr(name)?.extract()?),
    };
    let filename = path.to_string_lossy().into_owned();
    let instance = medh5_class(py, class)?.bind(py).call1((message, errno, filename))?;
    Ok(PyErr::from_value(instance))
}
