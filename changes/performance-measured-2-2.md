<!-- area: documentation -->
### Documentation

- **The performance guide's numbers are 2.2.0's, measured beside 2.1.0's on
  one machine.** 2.2.0 is within the run-to-run spread of 2.1.0 on every
  `medh5 bench` row. A new example, `docs/examples/bench_threads.py`, times
  the same reads from one, two and four threads, and the guide gives its
  table: opening samples from four threads takes 1.9 times as long as from
  one, and 3.8 times on the free-threaded build.
