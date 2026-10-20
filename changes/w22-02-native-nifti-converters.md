<!-- area: converters -->
### Added

- **The engine writes NIfTI and adds NIfTI masks to a sample.**
  `medh5::convert::nifti` gains `to_nifti`, `write_nifti`, `for_export`,
  `write_field` and `import_seg_nifti`, 2.1.0's converter ported rule for
  rule and held to the same parity corpus as the reader: every one of its 348
  cases gives 2.1.0's objects, `/meta`, report and refusal.

### Behaviour changes

- **NIfTI conversion no longer needs nibabel.** `medh5.io.nifti`, the ITK
  displacement fields of `medh5.io.itk` and the nnU-Net exporter read and
  write NIfTI through the engine, with their signatures unchanged. nibabel is
  neither installed nor imported; the `nifti` extra is kept, empty, so
  `pip install "medh5[nifti]"` still works, and `require_nibabel` says medh5
  no longer needs it rather than naming that extra.
- **A header NIfTI cannot hold is a `ValueError`.** It was nibabel's
  `HeaderDataError`, a `ValueError` subclass, with the same message.
- **A `.nii.bz2` or `.nii.zst` file is refused.** nibabel read them through
  Python's codecs; the engine inflates gzip only, and says to decompress the
  file first.
- **`convert to-nifti` keeps its overlap warning in the report.** Flattening
  overlapping classes into one labelmap was a Python `UserWarning` on stderr
  from the command line; it is now a warning in the report, as the other
  notes are. `to_nifti` in Python still warns, as it did.

### Fixed

- **A scaled complex-valued NIfTI mask keeps its imaginary voxels.** Under
  `scl_slope`, 2.1.0 tested only each voxel's real part, so a voxel such as
  `0+1i` was dropped from the mask; a voxel is now in the mask when either
  physical part is non-zero, as it already was unscaled.
- **Frame times one volume states bind the volumes after it.** When the
  first volume assumed its frame indices as times and a second stated those
  same times, a third stating other times replaced them, putting the second
  on a timeline it disagreed with; it is now refused, as two volumes stating
  different times always were.
- **`write_nifti` writes a fourth axis of no frames given its empty
  `time_values`.** 2.1.0 failed reading the first of no frame times
  (`IndexError`); there is no first time to state, so the header states none.
