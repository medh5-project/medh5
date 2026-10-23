# Local changes to the HDF5-Blosc2 filter

`blosc2_filter.h` and `blosc2_filter.c` are upstream's filter as `hdf5plugin`
7.1.0 ships it in `lib/HDF5-Blosc2/src/` (HDF5-Blosc2 3.0.1): the header byte
for byte, the source with the changes below, each marked `medh5:`. They turn
assertions on values read from a file into the filter's ordinary error return,
and check what upstream trusted, so a damaged or hostile file is an HDF5 read
error rather than an aborted process or an unbounded allocation. They are
offered upstream; until they land there, keep them when the filter is updated.

Defining `NDEBUG` is not a substitute. It removes the assertion and leaves the
condition unchecked, so a file whose stored chunk size is smaller than its
chunk would go on to be decompressed into the wrong buffer.

1. **Decompression (B2ND).** `assert(outbuf_size >= size)` aborted the process
   (SIGABRT) when the chunk-size filter value (`cd_values[3]`) was smaller than
   the chunk it describes. The size is now computed with overflow
   checks from the stored array's shape, the array's type size must be the
   filter's, and the buffer must hold the array --- exactly, when the filter
   values name the chunk shape, since HDF5 expects the whole chunk back.
   Otherwise the filter pushes an HDF5 error and fails.
2. **Filter values.** A type size of 0 and a chunk dimension below 1, above
   `INT32_MAX`, or whose product overflows `size_t` are refused before either
   direction uses them.
3. **Compression.** `compute_b2nd_block_shape` returned through assertions on
   its arguments (a type size of 0 divided by zero); it now returns `-1`, and
   the caller fails the filter.
4. **Decompressed size.** Both branches allocated what the file states the
   chunk holds --- the plain branch the size in the chunk's header, without
   checking that the header parsed (`blosc2_cbuffer_sizes`'s return), the B2ND
   branch the chunk size in the filter values --- however little HDF5 expects
   back. A chunk of special values states 2 GB in a few bytes. The size must
   now parse, and is capped before anything is allocated at twice the buffer
   HDF5 hands the filter and 64 KiB more: the cap the Zstandard filter applies
   (`src/zstd_filter.rs`), which leaves room for a filter applied before this
   one, such as a Fletcher32 checksum.
5. **The failure path.** Upstream's exit path called `blosc2_destroy()` on
   every chunk that failed, freeing Blosc2's global context and destroying its
   mutexes while a window read (`src/b2nd_slice.c`) may be decompressing on
   another thread, outside HDF5's lock. Removed: nothing the filter allocates
   outlives the call.
6. **Uninitialised locals.** The plain branch's exit path frees `chunk` when
   `needs_free` says so, and a super-chunk of no chunks failed before either
   was set. Both start initialised.
7. **Compression as an entry point.** The compress branch is now
   `medh5_blosc2_compress`, which the filter calls and the engine's chunk
   writer (`crates/medh5/src/h5/chunks.rs`) calls on its workers, outside
   HDF5's lock, before storing the result with `H5Dwrite_chunk`. It checks the
   filter values as the filter did and returns a negative code where the
   filter pushed an error; the filter maps each code back to upstream's
   message. With `one_thread`, the super-chunk's own compression and
   decompression contexts are set to one thread, whatever `BLOSC_NTHREADS`
   says, so a worker never starts threads of its own. The B2ND path is
   `b2nd_from_cbuffer` written out (`b2nd_empty`, then
   `b2nd_set_slice_cbuffer`) so the contexts can be set between the two. The
   bytes it stores are upstream's: a test pins the stored chunks of every
   profile and type against the filter before this patch. Upstream's
   `compute_blosc2_blocksize` kept its scratch array in a `static`, which two
   workers would share; it is on the stack.

Patches 4 to 6 are held by tests in `src/lib.rs` that call the filter as HDF5
does, and patch 7 by the chunk writer's tests in
`crates/medh5/src/h5/chunks.rs`.

C-Blosc2 itself is unchanged. `src/lib.rs` hands it a threads callback
(`blosc2_set_threads_callback`) before `blosc2_init`, so a context of more
than one thread starts threads only when it has work to share out, and joins
them before it returns. Without it, the frame around every chunk compressed
its index with a context of four threads, started and stopped with every
chunk for an index of a few bytes.

The window reads that skip HDF5's pipeline (`src/b2nd_slice.c`) are this
crate's own code, not upstream's, and check the same conditions.
