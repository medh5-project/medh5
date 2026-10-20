<!-- area: python-api -->
### Behaviour changes

- **A writer, document, payload, provenance, cache writer, staged file or
  patch draw used by a second thread at once raises `MEDH5Error`.** The
  second call is refused, says the object "is used by one thread at a time",
  and changes nothing; it raised `RuntimeError: Already borrowed` before.
  Calls that only read share the object: they are refused only while another
  thread is changing it.
