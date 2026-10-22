<!-- area: geometry -->
### Performance

- **`resample` does less work a voxel, and writes the same content.** Whether
  any target voxel lies outside the source is now read from the target's
  corners for an affine mapping, rather than by mapping every voxel once more;
  a block's indices are counted up rather than divided out of the row number;
  `linear` and `cubic` find each point's taps once an axis, rather than for
  every term through a full index; and no floor is called where a cast is the
  same. Resampling a 512×512×256 `int16` CT to 512×512×512 on one thread took
  39 s (nearest), 72 s (nearest, with an 8-class annotation), 107 s (linear,
  with it) and 321 s (cubic, with it) on the machine measured, and now takes
  27, 35, 52 and 68 s. Every object it writes has the digest 2.2.0's had.
