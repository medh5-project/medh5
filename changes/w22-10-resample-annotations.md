<!-- area: geometry -->
### Added

- **Annotations follow a resample.** `medh5 resample` and
  `medh5.transforms.resample` carry the annotations on the source grids onto
  the target, each naming its source in `derived_from` and keeping its classes
  and coverage. Voxel encodings move nearest in their own encoding, instances
  object by object under their own ids (an object with no voxel left is
  refused, never dropped), probability maps linearly; points and keypoints
  map exactly, by the inverse of the path the image is sampled through, and
  meshes along an affine path. A box is carried only where the
  mapping keeps the axes and a contour only where its plane stays a grid
  plane; anything else is refused rather than approximated. Target voxels
  outside the source become an ignore region (§7.7). `--no-annotations`
  (`annotations=False`) moves the images alone, and the report lists what was
  carried (`ResampledAnnotation`) and what was not (`not_carried`).
