<!-- area: files -->
### Performance

- **Digests are hashed on the threads `--threads` (or `medh5.set_threads`)
  allows, outside HDF5's lock.** A commit, `verify` and `validate --level
  integrity` read each dataset's stored chunks on the calling thread and
  decompress and hash them on the others, several datasets at once. Each
  digest is still one hash over its dataset's bytes in order, so it is the
  one a single thread computes, and every report line, finding and refusal
  comes out in the same order and words. On a 576 MiB sample with four
  threads, `verify` took 0.8 s instead of 2.9 s, an amendment 2.1 s instead
  of 6.6 s, and a nearest `resample` 6.5 s instead of 9.7 s.
