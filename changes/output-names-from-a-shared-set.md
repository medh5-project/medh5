<!-- area: converters -->
### Fixed

- **Threads naming outputs from one shared set no longer panic.** The first
  read of the set in a process could let go of the GIL partway through, and a
  name another thread added then ended the call with a `PanicException`
  (`Set changed size during iteration`). `output_name` now copies the set in
  one step.
