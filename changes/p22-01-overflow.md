<!-- area: validation -->
### Fixed

- **Arithmetic on what a file states no longer overflows.** A box from one
  end of the integers to the other, extents past 64 bits, a link span of
  `2**64 - 1` and counts summed past 64 bits panicked in a debug build and
  wrapped in a release one: the layers overlap check took four layers on a
  grid of `2**62` voxels for empty and passed them unread, a box-only
  object counted -1 voxels, and an index on a grid wider than `2**31` voxels
  wrapped its coordinates. Each is now exact, saturated where that is still
  the answer, or refused through the code that already covers the field.
- **An ITK B-spline whose lattice overflows is refused.** NumPy's product of
  `2**32` control points an axis wrapped to 0, and an empty parameter list
  matched it.
