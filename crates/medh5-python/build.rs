//! The interpreter's configuration as `cfg`s: `Py_GIL_DISABLED` marks a build
//! for free-threaded CPython, the one where a thread waiting for HDF5's lock
//! detaches from the interpreter first (`lib.rs`, `wait_detached`).

fn main() {
    pyo3_build_config::use_pyo3_cfgs();
}
