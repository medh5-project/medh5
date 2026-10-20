<!-- area: cli -->
### Added

- **The standalone `medh5` binary converts NIfTI itself.** `convert
  from-nifti`, `import-seg-nifti` and `to-nifti` run in the engine, with the
  checks, messages and reports the Python package gave them, so they work
  with no Python installed. The other converters are still handed to Python.
  An error the package let escape as a traceback, such as a header NIfTI
  cannot hold, is now `medh5: <why>` with exit code 1.
