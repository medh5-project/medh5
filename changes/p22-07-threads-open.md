<!-- area: files -->
### Performance

- **Opening samples from several threads no longer slows down on the
  free-threaded build.** In 2.2.0, four threads opening samples on Python
  3.14t took 3.5 times as long as one thread, twice 2.1.0's time. Two
  changes fix it. A value read on first use (a grid, an image's metadata, a
  document) is now read under one take of HDF5's lock, and a thread no longer
  waits on the value while another thread waits for the lock. An open also
  takes the lock once for all of its calls. 256 opens from four threads now
  take 0.25 s instead of 0.52 s, on 2.1.0's level, and 0.19 s instead of
  0.26 s on the GIL build. One thread is unchanged.
