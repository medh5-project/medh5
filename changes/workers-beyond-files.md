<!-- area: cli -->
### Added

- **`migrate`, `convert to-nnunet`, `conformance run`, `task preflight` and
  `clinical augment-cohort` take `--workers N`.** Each runs its units in up
  to N processes and prints what one process prints, byte for byte: a
  subject's 0.x files for `migrate`, a case for `to-nnunet` and `conformance
  run`, a subject for the other two. What the run shares (the label set, the
  case names, the one pass over the source) is worked out first, and units
  that touch one file go to one process. Over 32 files on four cores, with
  four workers: `migrate` 3.7 times faster, `conformance run` 3.5,
  `task preflight` 3.6, `to-nnunet` 2.4, `augment-cohort` 1.6. A process
  that dies fails only its unit. `scrub`, `splits`, `labels check` and
  `dataset check --deep` take milliseconds there, and take no `--workers`;
  the performance guide gives their numbers.
