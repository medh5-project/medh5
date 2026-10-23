//! Small rule helpers the Python modules use: collection member keys (§2.2),
//! image value types (§4.2) and the checks converters ask before writing ---
//! and the engine's read counts, which only the tests ask for.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::py_to_nd;
use crate::errors::R;

/// A collection member key (§2.2), or E003.
#[pyfunction]
fn validate_sample_key(name: &str) -> R<String> {
    Ok(medh5::ids::validate_sample_key(name)?.to_string())
}

#[pyfunction]
fn check_value_type(value_type: &str) -> R<String> {
    medh5::sample::image::check_value_type(value_type)?;
    Ok(value_type.to_string())
}

/// Whether a float array would survive `int16` storage unchanged (W907).
///
/// A scan of a whole volume, so other threads run while it does.
#[pyfunction]
fn lossless_as_int16(py: Python<'_>, array: &Bound<'_, PyAny>) -> R<bool> {
    let array = py_to_nd(array)?;
    Ok(py.detach(move || medh5::sample::image::lossless_as_int16(&array)))
}

/// Whether every value lies in `[0, 1]` (and there is at least one).
///
/// A scan of a whole volume, so other threads run while it does.
#[pyfunction]
fn is_probability(py: Python<'_>, array: &Bound<'_, PyAny>) -> R<bool> {
    let array = py_to_nd(array)?;
    Ok(py.detach(move || medh5::sample::image::is_probability(&array)))
}

/// The engine's dataset reads by kind (`whole`, `window`, `hyperslab`,
/// `rows`, `strings`) since the last call, and the counts set back to zero.
///
/// Private, for the tests that hold a read path to its number of reads: the
/// reads are the engine's, so a count of `h5py` calls sees none of them.
#[pyfunction(name = "_read_stats")]
fn read_stats(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    for (read, n) in medh5::h5::data::read_stats() {
        out.set_item(read.name(), n)?;
    }
    Ok(out)
}

/// Let one call use up to ``n`` threads for its work outside HDF5, from now
/// on and in every thread of this process; ``0`` is refused.
///
/// The default is 1, which runs every call on its caller's thread and starts
/// none.  It is set over ``MEDH5_THREADS``, and the command line's
/// ``--threads`` is set over it for the one command.
#[pyfunction]
fn set_threads(n: i64) -> R<()> {
    if n < 1 {
        return Err(medh5::Error::Value(format!("the thread count must be at least 1, not {n}")).into());
    }
    Ok(medh5::parallel::set_threads(n as usize)?)
}

/// How many threads one call may use for its work outside HDF5: what
/// :func:`set_threads` set, else ``MEDH5_THREADS``, else 1.
///
/// ``MEDH5_THREADS`` is read once, the first time it is needed; a value that
/// is not a positive integer raises ``ValueError``, naming it.
#[pyfunction]
fn get_threads() -> R<usize> {
    Ok(medh5::parallel::threads()?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_threads, m)?)?;
    m.add_function(wrap_pyfunction!(validate_sample_key, m)?)?;
    m.add_function(wrap_pyfunction!(check_value_type, m)?)?;
    m.add_function(wrap_pyfunction!(lossless_as_int16, m)?)?;
    m.add_function(wrap_pyfunction!(is_probability, m)?)?;
    m.add_function(wrap_pyfunction!(read_stats, m)?)?;
    Ok(())
}
