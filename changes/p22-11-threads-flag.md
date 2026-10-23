<!-- area: cli -->
### Added

- **`--threads N`, before or after any command name, sets the thread count
  for the one command.** Every process the command starts, a `--workers`
  process or a Python converter, is given it as `MEDH5_THREADS`. With it, a
  long option of `agree` shortened to `--th`, `--thr` or `--thre` names two
  options; spell `--threshold` out.
