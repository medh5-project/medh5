<!-- area: platform -->
### Fixed

- **A thread waiting for HDF5 on free-threaded CPython lets go of the
  interpreter.** Waiting attached for another thread's HDF5 call held up every
  stop-the-world pause, a garbage collection among them, until that call ended.
  A read or write through medh5 now runs detached throughout. Closing a sample
  no longer holds up a thread asking for it, and a lock left by a panic is taken
  as it stands instead of raising again. A payload's dicts, edited by another
  thread while it is written or described, no longer panic the read.
- **A forked child no longer hangs at its first HDF5 call.** A child forked
  while another thread was in the middle of a read inherited HDF5's lock held
  by a thread it does not have. A fork now waits for the call to end, and the
  child takes the lock over cleanly.
