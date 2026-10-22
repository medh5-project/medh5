<!-- area: geometry -->
### Performance

- **`resample` onto a grid on the source's own axes maps each axis once, not
  each voxel.** Respacing, or any target whose axes run along the source's,
  gives every axis its source coordinates as a short table, worked out by the
  same arithmetic a voxel's row was; nearest, linear and cubic, and a carried
  annotation, then read them from the tables. Resampling a 512×512×256 `int16`
  CT to 512×512×512 on one thread now takes 14 s (nearest), 18 s (nearest,
  with an 8-class annotation), 34 s (linear, with it) and 49 s (cubic, with
  it), from 27, 36, 52 and 74 s. Every object it writes has the digest 2.2.0's
  had.
