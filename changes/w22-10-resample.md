<!-- area: geometry -->
### Added

- **Images resample onto another grid as a recorded derivation.**
  `medh5.transforms.resample(path, image_ids, onto=…)` and
  `medh5 resample PATH --image ID --onto GRID` map images onto an existing
  grid: through the affines when the two grids share a comparable frame, and
  through the sample's transforms otherwise (§10.2), evaluated at the target's
  voxels. `spacing=` with `grid_id=` (`--spacing`, `--grid-id`) makes a new
  grid in the image's frame that keeps its voxel edges. A pair no transform
  relates, a frame that cannot be compared (`W917`) and a grid naming no frame
  are refused rather than assumed aligned. Interpolation is `linear`,
  `nearest` or `cubic`; voxels outside the source take `fill` and the output's
  `valid_mask` marks the rest. One `resample` activity records the inputs, the
  target, the transform path and the interpolation, and every object written
  names it. The work goes a target chunk at a time, so memory follows the
  chunk, not the volume. The refusals that send you to resample now name the
  command: the torch datasets' patch across two grids, the MONAI reader's
  channel stack across two grids, and agreement between annotations on
  different grids (`E101`) each say `medh5 resample PATH --image ID --onto
  GRID`; the NIfTI import's volumes on two grids, its masks drawn on another
  grid, and the nnU-Net export's labels off the channels' grid name it too.
