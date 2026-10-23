/* Decompress part of one stored chunk: the blocks of a B2ND frame that a
 * window covers.
 *
 * The HDF5-Blosc2 filter stores every chunk of rank > 1 as a B2ND frame, a
 * grid of independently compressed blocks.  HDF5's filter pipeline can only
 * return whole chunks; this reads the window's part and leaves the other
 * blocks compressed.  It is C because the checks need the frame's own rank,
 * shape and item size, which B2ND keeps in a struct Rust should not mirror.
 */

#include <stdbool.h>
#include <stdint.h>

#include "b2nd.h"
#include "blosc2.h"

/* 0 on success.  1 when the frame is not a B2ND frame of rank `ndim`, shape
 * `chunk_shape` and item size `typesize`, or the slice is not inside it: the
 * caller then reads through HDF5.  A negative Blosc2 error code when the
 * blocks do not decompress.
 *
 * `frame` is read in place and not modified; `out` receives the slice
 * `[start, stop)` in C order and must hold `out_len` bytes. */
int medh5_b2nd_read_slice(const uint8_t *frame, int64_t frame_len, int8_t ndim,
                          const int64_t *chunk_shape, int32_t typesize,
                          const int64_t *start, const int64_t *stop,
                          void *out, int64_t out_len) {
  if (ndim < 1 || ndim > B2ND_MAX_DIM || typesize < 1) {
    return 1;
  }
  int64_t shape[B2ND_MAX_DIM];
  int64_t need = typesize;
  for (int i = 0; i < ndim; i++) {
    if (start[i] < 0 || stop[i] > chunk_shape[i] || start[i] >= stop[i]) {
      return 1;
    }
    shape[i] = stop[i] - start[i];
    need *= shape[i];
  }
  if (need > out_len) {
    return 1;
  }

  /* `copy = false`: the super-chunk reads `frame` where it is, and freeing
   * the super-chunk leaves `frame` alone. */
  blosc2_schunk *schunk = blosc2_schunk_from_buffer((uint8_t *) frame, frame_len, false);
  if (schunk == NULL) {
    return 1;
  }
  b2nd_array_t *array = NULL;
  if (b2nd_from_schunk(schunk, &array) < 0) {
    /* Not a B2ND frame (a plain Blosc2 one, say): the super-chunk is still
     * ours to free, since only a built array takes it over. */
    blosc2_schunk_free(schunk);
    return 1;
  }

  int rc = 1;
  if (array->ndim != ndim || array->sc->typesize != typesize) {
    goto done;
  }
  for (int i = 0; i < ndim; i++) {
    if (array->shape[i] != chunk_shape[i]) {
      goto done;
    }
  }
  rc = b2nd_get_slice_cbuffer(array, start, stop, out, shape, out_len);
  if (rc > 0) {
    rc = -1;
  }

done:
  b2nd_free(array); /* and the super-chunk with it */
  return rc;
}

/* 0 when `frame` is a B2ND frame of rank `ndim`, shape `chunk_shape` and item
 * size `typesize`, which `medh5_b2nd_read_slice` reads unless its blocks do
 * not decompress; 1 otherwise.  Parses the frame's metadata and decompresses
 * nothing.  `frame` is read in place and not modified. */
int medh5_b2nd_frame_fits(const uint8_t *frame, int64_t frame_len, int8_t ndim,
                          const int64_t *chunk_shape, int32_t typesize) {
  if (ndim < 1 || ndim > B2ND_MAX_DIM || typesize < 1) {
    return 1;
  }
  blosc2_schunk *schunk = blosc2_schunk_from_buffer((uint8_t *) frame, frame_len, false);
  if (schunk == NULL) {
    return 1;
  }
  b2nd_array_t *array = NULL;
  if (b2nd_from_schunk(schunk, &array) < 0) {
    blosc2_schunk_free(schunk);
    return 1;
  }
  int rc = array->ndim != ndim || array->sc->typesize != typesize;
  for (int i = 0; rc == 0 && i < ndim; i++) {
    rc = array->shape[i] != chunk_shape[i];
  }
  b2nd_free(array); /* and the super-chunk with it */
  return rc;
}
