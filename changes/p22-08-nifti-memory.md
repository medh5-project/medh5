<!-- area: converters -->
### Performance

- **A NIfTI import holds one volume at a time.** `from-nifti` read every input
  before it wrote the first, and held each twice while decoding it. It now
  reads every header first, so a refusal still comes before anything is
  written, then decodes each image into its array a megabyte at a time and
  writes it, one image after another. Four 512×512×256 inputs (two `int16`
  CTs, a `float32` PET and a mask) peaked at 721 MiB and now at 301 MiB, the
  PET and the writer's buffers; one CT at 272 MiB, now 158 MiB.
- **A NIfTI export writes its voxels a megabyte at a time**, rather than
  copying the volume into Fortran order whole first: one 512×512×256 `int16`
  image peaked at 275 MiB, now 157 MiB.
