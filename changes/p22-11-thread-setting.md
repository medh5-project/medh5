<!-- area: python-api -->
### Added

- **`medh5.set_threads(n)` and `medh5.get_threads()` say how many threads one
  call may use for its work outside HDF5.** The count is what `set_threads`
  set, else the `MEDH5_THREADS` environment variable, else 1, which starts no
  thread; a value that is not a positive integer is refused, never guessed.
  Rust has the same as `medh5::parallel::set_threads` and `threads`. New in
  a patch release by the maintainer's decision, as the one setting 2.2.3's
  parallel work needs.
  `medh5.torch.worker_init_fn` sets one thread in each `DataLoader` worker.
  On macOS, the thread that forked a child runs each call by itself in the
  child: it would wait for threads on a Mach port made before the fork, which
  the child does not inherit.
