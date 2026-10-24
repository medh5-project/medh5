<!-- area: writer -->
### Performance

- **`recompress` and `resample` compress the chunks they write on the
  threads `--threads` (or `medh5.set_threads`) allows, outside HDF5's
  lock.** A worker compresses one whole chunk with the Blosc2 filter's own
  code, and the calling thread stores it with `H5Dwrite_chunk`, in order, so
  the file holds the bytes HDF5's write stores, at any thread count; a
  resample's workers also map and interpolate the chunk. On a 576 MiB sample
  with four threads, `recompress --profile archive` took 29.5 s instead of
  116.3 s, and a linear `resample` 29.1 s instead of 43.4 s. What the writer
  cannot store exactly as HDF5 would (`portable`, boolean masks, a dataset
  stored without chunks) goes through HDF5's write as before. A row of
  chunks past the 64 MiB a recompression reads at once is read and written
  a band of chunk columns at a time.
- **Blosc2 no longer starts threads for work it does not share out.** The
  frame around every chunk compressed its index with a context of four
  threads, started and stopped for each chunk; Blosc2 now starts threads only
  when it hands blocks to them, and a worker of the chunk writer compresses
  with one, whatever `BLOSC_NTHREADS` says.
