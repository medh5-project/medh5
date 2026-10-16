<!-- area: platform -->
### Added

- **medh5 runs on free-threaded CPython without re-enabling the GIL.** The
  cp314t wheel declares the extension safe to run with no GIL, so
  `import medh5` leaves it off; 2.1.0 asked for it, and CPython turned it on
  with a warning. The test suite runs on 3.14t with the GIL off, many threads
  at a time.
