//! `medh5._core`: the format engine as a Python extension module.
//!
//! Everything format-native --- the data model and its rules, HDF5 I/O,
//! encodings, geometry, transforms, validation, integrity, the dataset and
//! curation tools, the conformance suite and the command line --- is the Rust
//! engine (`medh5` crate).  This module only translates: Python values in,
//! engine calls, Python values out.  The `medh5` Python package is built on
//! these names: each module re-exports what it serves as is (`medh5.geometry`,
//! `medh5.labels`, ...) and wraps the rest --- the `*Handle` classes behind
//! `Sample`, `Image`, `Annotation` and `Transform`.  A class's `__module__`
//! names the module that exports it; a handle, which none does, says
//! `medh5._core`.

use pyo3::prelude::*;

pub mod annotations;
pub mod cli;
pub mod clinical;
pub mod conformance;
pub mod convert;
pub mod curation;
pub mod curation_tools;
pub mod dataset;
mod deprecation;
pub mod document;
pub mod errors;
pub mod exclusive;
pub mod files;
pub mod geometry;
pub mod integrity;
pub mod io;
pub mod labels;
pub mod misc;
pub mod nifti;
pub mod nodes;
pub mod preflight;
pub mod reader;
pub mod records;
pub mod rng;
pub mod sampling;
pub mod storage;
pub mod tracking;
pub mod transforms;
pub mod validate;
pub mod values;
pub mod writer;

// `gil_used = false`: a build for free-threaded CPython (3.14t) declares the
// module `Py_MOD_GIL_NOT_USED`, and the interpreter keeps the GIL off as it
// imports it.  What that rests on: every class is `frozen`, and one that
// changes as it is used keeps its state behind a lock of its own and refuses
// a second thread (`exclusive`); the Python handle cache and registries are
// locked; a thread waits for HDF5's lock detached (`wait_detached`); and a
// fork waits for the engine's locks a detached thread may hold (`h5::fork`).
// Said out loud, as `true` was until this was audited, so the choice is read
// here rather than inferred from PyO3's default.
#[pymodule(gil_used = false)]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(Py_GIL_DISABLED)]
    medh5::h5::on_wait(wait_detached);
    // A fork waits for any HDF5 call another thread is in, so the child
    // never inherits HDF5's lock held by a thread it does not have.
    #[cfg(unix)]
    medh5::h5::fork::install();
    m.add("__version__", medh5::VERSION)?;
    m.add("__format_version__", medh5::FORMAT_VERSION)?;
    labels::register(m)?;
    geometry::register(m)?;
    curation::register(m)?;
    document::register(m)?;
    nodes::register(m)?;
    reader::register(m)?;
    writer::register(m)?;
    annotations::register(m)?;
    integrity::register(m)?;
    tracking::register(m)?;
    validate::register(m)?;
    misc::register(m)?;
    transforms::register(m)?;
    storage::register(m)?;
    cli::register(m)?;
    conformance::register(m)?;
    dataset::register(m)?;
    curation_tools::register(m)?;
    sampling::register(m)?;
    io::register(m)?;
    nifti::register(m)?;
    clinical::register(m)?;
    preflight::register(m)?;
    files::register(m)?;
    Ok(())
}

/// How a thread waits for HDF5's lock another thread holds, on free-threaded
/// CPython: detached from the interpreter, for the wait and the calls it was
/// waiting to make.  Attached, it would hold up every stop-the-world pause ---
/// a collection, say --- until the other thread's HDF5 call ended.  The engine
/// asks only a thread that does not hold the lock already, so no thread ever
/// waits to take the interpreter back while holding HDF5's lock.  With the
/// GIL, the waiting thread holds nothing another needs: the holder of HDF5's
/// lock runs detached, and lets it go before it attaches again.
#[cfg(Py_GIL_DISABLED)]
fn wait_detached(run: &mut dyn FnMut()) {
    // Detached already (a long call lets go of the interpreter first).
    // SAFETY: reads this thread's state only, and never fails.
    if unsafe { pyo3::ffi::PyThreadState_GetUnchecked() }.is_null() {
        return run();
    }
    struct Calls<'a>(&'a mut dyn FnMut());
    // SAFETY: what runs is the engine's HDF5 calls, which touch nothing of
    // Python's, on this same thread.
    unsafe impl Send for Calls<'_> {}
    let calls = Calls(run);
    // SAFETY: the check above says this thread is attached.
    let py = unsafe { Python::assume_attached() };
    py.detach(move || {
        let calls = calls;
        (calls.0)()
    });
}

/// Engine calls that read or write a file, made from a binding that holds
/// the interpreter: on free-threaded CPython, detached.  Every HDF5 call
/// takes HDF5's one lock, and `hdf5`'s own methods take it themselves, past
/// the engine's wait hook (`wait_detached`): a thread that met another's
/// read there while attached held up every stop-the-world pause --- a
/// collection --- until that read ended.  Detaching is not holding: `f`
/// takes HDF5's lock call by call, as it would attached, so an engine cache
/// filled on first use never waits for a thread that holds HDF5's lock.
/// With the GIL, `f` runs where it is, as it always has.
#[inline]
pub(crate) fn h5<T: Send>(py: Python<'_>, f: impl FnOnce() -> T + Send) -> T {
    #[cfg(Py_GIL_DISABLED)]
    return py.detach(f);
    #[cfg(not(Py_GIL_DISABLED))]
    {
        let _ = py;
        f()
    }
}
