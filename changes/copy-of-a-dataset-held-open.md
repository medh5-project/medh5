<!-- area: writer -->
### Fixed

- **Copying a dataset the process holds open no longer corrupts memory.** An
  amendment, pack, unpack, extract or repack copies datasets as stored, and a
  recompression the ones it leaves as they are. HDF5 took the chunks an open
  handle had cached into a buffer sized for a compressed chunk, at their full
  size: with a `Dataset` from `Sample.root` kept, or the file read on another
  thread, the copy wrote past the buffer's end, and the process crashed or ran
  on with its heap damaged. HDF5 is now built with a fix
  (`vendor/hdf5-metno-src/PATCHES.md`); the wheels and the release binaries
  carry it, and a build from crates.io does not until HDF5 releases the fix.
