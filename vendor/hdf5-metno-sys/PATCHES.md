# Local changes to hdf5-metno-sys

This directory is `hdf5-metno-sys` 0.12.5 as crates.io publishes it, the
bindings to HDF5 and the lock that serialises calls to it, with the change
below, marked `medh5:`. The workspace builds it in place of the published crate
(`[patch.crates-io]` in the root `Cargo.toml`), so every wheel, the sdist and
every standalone binary have this lock. A crate built from crates.io
(`cargo install medh5-cli`, or a crate that depends on `medh5`) does not: a
`[patch]` reaches only the workspace that declares it. So `medh5-sys` works with
either lock, and its tests with either. Keep the change when the crate is
updated, until `hdf5-metno-sys` has a lock of its own like it.

1. **HDF5's lock on Unix** (`src/lock.rs`, new; `LOCK` in `src/lib.rs`; the
   `lock_api` dependency in `Cargo.toml`; the metadata in `build.rs`). `LOCK`
   was `parking_lot::ReentrantMutex`, which keeps the threads waiting for it in
   a table shared across the process. A thread starting to wait holds a lock
   of the table's for a moment, and a `fork` in that moment leaves the child
   that lock held, by a thread it does not have. The child hung in its fork
   handler, about once in 500 forks made while another thread recompressed
   (medh5#106): parking_lot cannot hold or reset its table across a `fork`.

   The lock is now a `lock_api::ReentrantMutex` around `RawLock`, whose queue
   of waiters is two counts in one word of its own, so that a child sets it
   right with one write (`RawLock::forget_waiters`). It behaves as parking_lot's
   does: a thread spins for a moment, then takes a number and sleeps; letting go
   wakes the thread that has waited longest, which a running thread may
   overtake; and every half a millisecond the lock is handed to that thread, so
   that none starves. Threads wake by number because the kernels do not promise
   an order: threads asleep on one address on macOS wake out of order, and with
   one address for all of them a thread could wait for seconds under load.

   A thread sleeps on a futex on Linux and Android, on `__ulock_wait` on Apple
   platforms, which is private API that libc++'s `std::atomic::wait` uses too,
   and in short naps on other Unix. Windows has no `fork` and keeps
   parking_lot's lock.

   The build script reports `lock_in_own_word` on Unix, which `medh5-sys` reads
   as `DEP_HDF5_LOCK_IN_OWN_WORD`: what a child does about the waiters depends
   on the lock, and the published crate reports nothing.
