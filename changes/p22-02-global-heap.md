<!-- area: validation -->
### Security

- **A crafted file no longer hangs opening or validating it.** One damaged
  structure inside HDF5 made the library loop without end on the first read
  of a string, so `medh5 validate`, opening the file from Python, and every
  other read never returned. HDF5 is now built with a fix
  (`vendor/hdf5-metno-src/PATCHES.md`), and the file is reported as
  unreadable (`E001`). The wheels and the release binaries carry it; a build
  from crates.io does not until HDF5 releases the fix (`SECURITY.md`).
