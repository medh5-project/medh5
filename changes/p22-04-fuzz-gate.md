<!-- area: converters -->
### Fixed

- **Writing an ITK HDF5 transform file refuses a type name holding a NUL.**
  HDF5 ends a stored string at its first NUL, so the file kept a shorter name
  than the one given, and read back as another transform type.
