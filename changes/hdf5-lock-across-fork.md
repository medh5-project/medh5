<!-- area: platform -->
### Fixed

- **A forked child no longer hangs because another thread was starting to wait
  for HDF5.** A thread that met HDF5's lock at the moment of a `fork` could
  leave the child stuck in its fork handler, before any of its own code ran,
  about once in 500 forks made while another thread recompressed (#106). The
  lock kept its waiters in a table shared across the process, which a `fork`
  can copy half-changed. It now keeps its queue in one word of its own, which
  the child resets (`vendor/hdf5-metno-sys/PATCHES.md`). The wheels, the sdist
  and the release binaries have this lock; a build from crates.io keeps the
  published one, and the concurrency page says what that means.
