<!-- area: torch -->
### Fixed

- **A forked child takes over every handle cache before its first call.** An
  `os.register_at_fork` hook abandons the parent's handles and gives each
  cache a new lock as the child starts: two threads of a child that both
  noticed the fork replaced the cache's lock one after the other, and then
  held different locks over the same handles. Registering a vocabulary
  (`medh5.labels.register`) and editing a document's live views from several
  threads are locked too, and no code of the caller's runs under that lock
  (an index's `__index__`, an item's `__eq__`, a key's `__hash__`). A key
  JSON cannot spell is refused as it is given, leaving the member editable.
  `medh5.labels.describe()` no longer raises E305 for a vocabulary another
  thread unregisters meanwhile, and a document's label set and timeline are
  read with every class and timepoint as it stood at one moment.
