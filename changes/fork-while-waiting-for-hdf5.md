<!-- area: documentation -->
### Documentation

- **The concurrency page says when a forked child can still hang.** A thread
  that starts waiting for HDF5's lock at the moment of a `fork` can leave the
  child waiting forever, about once in 500 forks made while another thread
  recompresses (#106). A `DataLoader` meets it only if another thread is
  calling medh5 as its iterator is made; forking while no thread is calling
  medh5, or `spawn` and `forkserver`, rule it out.
