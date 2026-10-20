<!-- area: validation -->
### Fixed

- **A box whose finite bounds lie past the integers no longer overflows the
  validator.** An `instances` object's extent was its box's two slice ends
  subtracted, and a box from one end of the `i64` range to the other
  overflowed: a build with overflow checks panicked (the `open_validate`
  fuzzer found it), and a release build compared the crop with a negative
  extent. The extent now saturates, and the crop is reported as `E405`.
