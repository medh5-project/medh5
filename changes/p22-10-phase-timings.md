<!-- area: documentation -->
### Documentation

- **The performance guide says where each heavy command's time goes.**
  `docs/examples/bench_phases.py` runs `resample`, `convert from-nifti`, an
  amendment, `recompress`, `verify` and `validate` through a native CLI built
  with `--cfg medh5_phase_timings`, and reports the seconds inside HDF5's lock
  and outside it. The cfg is for measuring only: no profile sets it, and a
  build without it is unchanged.
