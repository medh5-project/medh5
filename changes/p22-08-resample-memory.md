<!-- area: geometry -->
### Performance

- **`resample` holds one target chunk, not one target volume.** The
  `valid_mask`, a carried mask or label map, and a new ignore mask were built
  whole on the target grid before they were written; each is now written a
  chunk at a time, as the image already was, with the same content. Resampling
  a 512×512×256 `int16` CT to 512×512×512 with an annotation peaked at 304 MiB
  (nearest), 318 MiB (linear) and 498 MiB (nearest, past the source's edge),
  and now at 109, 109 and 114 MiB. A probability map is still built whole, and
  `cubic` still prefilters the whole source; the resampling guide says what
  each costs.
- **A digest hashes a block a piece at a time.** It copied a whole slab into
  little-endian bytes before hashing it, up to 64 MiB, and now copies 1 MiB at
  a time.
