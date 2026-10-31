//! The tests of HDF5's lock, which sit beside it in
//! `vendor/hdf5-metno-sys/src/lock.rs`. `cargo test --workspace` does not reach
//! a crate that `[patch]` brings in, so they run here, in a build of the lock
//! on its own, whichever lock the rest of this build has.
#![cfg(unix)]

#[allow(dead_code)]
#[path = "../../../vendor/hdf5-metno-sys/src/lock.rs"]
mod lock;
