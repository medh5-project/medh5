# Local changes to hdf5-metno-src

This directory is `hdf5-metno-src` 0.10.4 as crates.io publishes it, which
packages the source of HDF5 2.2.0 under `ext/hdf5/`, with the changes below,
each marked `medh5:`. The workspace builds it in place of the published crate
(`[patch.crates-io]` in the root `Cargo.toml`), so every wheel, the sdist and
every standalone binary link this HDF5. A crate built from crates.io
(`cargo install medh5-cli`, or a crate that depends on `medh5`) does not: a
`[patch]` reaches only the workspace that declares it. Until the first change
ships in HDF5 and in an `hdf5-metno-src` release, keep both when the crate is
updated.

1. **Global heap objects** (`ext/hdf5/src/H5HGcache.c`,
   `H5HG__cache_heap_deserialize`). An object of a global heap collection,
   free space included, holds at least its own header. One that states less
   is refused as a damaged heap, as HDF5 refuses one that overruns its
   collection.
2. **The build script** (`build.rs`) reruns when anything under `ext/hdf5`
   changes, not only when the script does, so a build cache never links an
   HDF5 compiled before a change here.
3. **Driver info blocks** (`ext/hdf5/src/H5Fsuper_cache.c`,
   `H5F__drvrinfo_prefix_decode`). A driver info block lies within the file.
   One whose stated size runs past the file's end is refused before the
   metadata cache reads it, rather than read to that size and refused after.
4. **Chunks copied from the chunk cache** (`ext/hdf5/src/H5Dchunk.c`,
   `H5D__chunk_copy_cb`). `H5Ocopy` takes a chunk that a handle open in the
   process has cached from the cache, unfiltered and at its full size, into
   the buffer it sized for the chunk as stored, which the output pipeline of
   the chunk before may have left as small as that chunk compressed. The
   buffer grows to the full size first, as it does for a chunk read from the
   file, rather than taking the chunk past its end. HDF5's `develop` has the
   same code.

The first and third changes are held by the
`global-heap-object-smaller-than-its-header` and
`driver-info-block-past-the-end` fixtures of the robustness corpus
(`crates/medh5/src/conformance/robustness.rs`), which CI reports through every
wheel and every binary it builds. The fourth is held by
`s14_4_a_dataset_a_reader_holds_open_copies_whole`
(`crates/medh5/src/h5/graph.rs`), whose copy writes past its buffer without it.
