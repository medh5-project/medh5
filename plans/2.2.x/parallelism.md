# medh5 2.2.3 and 2.2.4: parallelism

Two patch releases about parallelism: threads for the work inside one file
that runs outside HDF5's lock, processes for the work across files, and the
cost of digests at commit, and nothing else ([D2](#decisions-the-maintainer)).
2.2.3 measures; 2.2.4 adds one thread setting and the paths that use it. Read
[plans/README.md](../README.md) and the [2.2.x plan](README.md) first. Their
rules hold here, with `p22/` branches and `P22-` titles, and with one
exception to what a patch may change ([D6](#decisions-the-maintainer)).

- **Baseline.** v2.2.2, released from `main` at 0e5e61c. The "State at 2.2.x"
  claims below were first checked against `main` at 5a8a2b4, and the line
  numbers again at 0e5e61c, after 2.2.2's resample changes (#99, #101). A
  workstream checks each claim against `main` when it starts.
- **Where the reasons come from.**
  - [What 2.2.0 taught](what-2.2.0-taught.md), cited as "lessons §n".
  - The 2.2.x measurements: #92's tables (P22-06), and lane B's phase timings
    of the 512³ resample, summarised in
    [The numbers](#the-numbers-this-plan-starts-from).
- **What these patches may change.** The [2.2.x rules](README.md), with one
  exception:
  - **The exception (2.2.4 only).** The thread setting of P22-11 is public
    API and a command-line flag, which a patch otherwise never adds. The
    maintainer allowed it once, because it is a small change
    ([D6](#decisions-the-maintainer)). It covers exactly the names P22-11
    lists, and `--threads` on the commands P22-12 and P22-13 make use it.
    Nothing else in 2.2.3 or 2.2.4 adds API or a flag.
  - The format stays 1.2, and the diagnostic code table keeps its codes.
  - **No thread count changes an output.** Every path this plan parallelises
    writes the same `content_id`, the same stored chunk bytes and the same
    command output at any thread or worker count, and the same as 2.2.2 at
    one thread. Tests hold that for each path (see each item's Tests).
- **The API gates.** `API_RELEASE_TYPE` stays `patch`. One list of allowed
  additions, `.github/api-exceptions.txt`, names each excepted Rust path and
  Python name and the decision that allowed it, and CI holds both languages
  to it:
  - **Python.** P22-11 has `public_api.py compare` read the list: under
    `patch`, an addition not on it still fails.
  - **Rust.** `cargo semver-checks` has no lint for an added item, so today
    nothing fails a new Rust name. P22-11 adds a check of additions to the
    `semver` job: each crate's public items against the baseline's (a pinned
    `cargo public-api` diff, or the rustdoc JSON `cargo semver-checks`
    already builds), failing under `patch` on an item not on the list.
  - **Everything else stays private.** The engine's new code (the chunk
    writer in `h5/`, the hashing in `integrity/`, the timings) is
    `pub(crate)`, and `medh5-sys` gains no Rust name: the C entry point
    P22-12 adds is declared in a private `extern` block inside `medh5`
    (the symbol links in with `medh5-sys`'s static library). The list holds
    P22-11's names, P22-14's flags if D7 takes them, and nothing else.
  - The list stays through 2.2.4's release pull request, whose gate still
    compares against the published 2.2.3, and is emptied by the first pull
    request after v2.2.4 is published, when those names are the baseline.

## The numbers this plan starts from

- **HDF5 runs one call at a time in a process, and its filters run inside
  that call** (`docs/explanation/concurrency.md`). Threads help only with
  work done outside an HDF5 call.
- **Threads in one process measured slower where the work is HDF5's.**
  - 2.1.0's preflight thread pool: 3.6 s on one thread, 6.9 s on four
    (`docs/explanation/design-rationale.md`).
  - 256 opens: 1.8–1.9 times as long on four threads as on one, on every
    build, and still slower than one thread after P22-07 (#92, #95).
  - `portable` (gzip) windows: 1.08–1.10 times one thread (#92).
- **Threads measured faster where the work is outside HDF5.** Blosc2
  `dense()` windows decompress outside the lock and took 0.30 of one thread's
  time on four (#92).
- **Processes scale across files.** `medh5 verify` over 32 files took 10.1 s
  in one process and 2.7 s in four (`docs/guides/performance.md`).
- **The 512³ resample, one thread** (lane B, a timing build, before 2.2.2's
  axis-aligned path, which changes the mapping row):
  - nearest, no annotation, 26.5 s: mapping 14.2 s, **digests at commit
    8.2 s**, writing and compressing 2.1 s, reading 0.4 s, sampling 0.3 s;
  - linear with the annotation, 50.6 s: mapping 12.6 s, interpolating
    12.2 s, **digests 11.7 s**, carrying the label map 7.7 s, writing 4.9 s.
  - Of the digest time, 576 MiB is the source's untouched images, which the
    commit hashes again
    ([P22-13](#p22-13-digests-in-parallel-outside-the-lock-224)).
  - The machine (a 4-vCPU Linux VM) has no SHA instructions. Apple Silicon
    and recent x86_64 have them, so digests cost much less there.

## How this plan answers what came before

| Before | So in 2.2.3 and 2.2.4 |
|---|---|
| 2.2.x left threads out: a thread pool is a new behaviour and needs a setting (P22-09). | The maintainer allowed the setting in a patch, once (D6). It defaults to one thread, today's code path. |
| 2.1.0 added a preflight thread pool and it ran slower. Nothing was measured before it was built. | P22-10 measures how much of each command runs outside HDF5's lock before any thread code is written, ships alone in 2.2.3, and drops an item that cannot pay. |
| P22-06's table set the work of P22-07 to P22-09, and each of those recorded its numbers (lessons §9). | Every item records 1, 2 and 4 threads before and after, on one machine, in its pull request. |
| P22-08 and P22-09 changed hot paths with every digest pinned to 2.2.0's. | Every parallel path is tested for the same `content_id` and the same stored chunks at 1 and 4 threads. |
| Arming auto-merge on #96 was stopped by the session's own permission check, and waited for a card. | Each lane's brief says whether the maintainer allows it to arm auto-merge on its own pull requests ([Before the lanes start](#before-the-lanes-start-the-maintainer)). |
| The weekly limit set 2.2.0's release date (lessons §1). | Two lanes, with the measurement and the enabler first. |

## Releases

| Release | Holds | Tagged when |
|---|---|---|
| 2.2.3 | P22-10, the measurement | its pull request has merged and the release checklist passes |
| 2.2.4 | P22-11 to P22-14: the thread setting, chunk writes, digests, and processes across files | their pull requests have merged and the release checklist passes |

2.2.3 changes no behaviour, so it ships alone and does not wait on 2.2.4.
Patches are tagged from `main` in order, so no 2.2.4 item (P22-11 to P22-14)
merges before v2.2.3 is tagged. A 2.2.4 pull request may open and pass
review earlier, and waits to be queued.

## Work items

### P22-10 Measure what can run outside the lock (2.2.3, start at once)

- **State at 2.2.x.**
  - `docs/examples/bench_threads.py` (#92) times reads from one, two and
    four threads. Nothing times the phases of a command: lane B's resample
    phases came from a local timing build that is not in the repository.
  - `medh5 bench` and the guide's tables time reads, conversions and
    resampling as a whole.
- **Change.** No behaviour changes.
  - Add phase timings behind a `cfg` the release profile never sets
    (`--cfg medh5_phase_timings` in `RUSTFLAGS`), so they are not a Cargo
    feature and not public API. Each phase is one of: under the HDF5 lock
    (open, metadata, read, write, filters), and outside it (decompression a
    window does itself, mapping, interpolation, hashing, encoding).
  - Profile, on one large sample, the native CLI's `resample` (nearest and
    linear, with the annotation), `convert from-nifti`, a no-op amendment
    (`splits` or an added label), `recompress --profile archive`, `verify`
    and `validate --level integrity`.
  - Measure the throughput of one core for Blosc2 at each profile's settings
    and for each digest algorithm. Record which `sha2` backend each release
    platform gets (SHA extensions on x86_64 and aarch64, or software).
  - From those, give each command's ceiling: the share of its time outside the
    lock, and so the most four threads could give.
- **Output.**
  - A table in the pull request: each command's phases, in and outside the
    lock, and its ceiling at four threads.
  - A ranked list that sets the scope of P22-12 and P22-13. A path whose
    ceiling at four threads is below 1.3 times is dropped, and the plan says
    so.
  - If the maintainer runs the script on Apple Silicon, a second column.
- **Changelog.** `<!-- no entry: ... -->` unless the timings build is
  documented for users.
- **Done when.** The table covers every row above, on 2.2.2's code.

### P22-11 One thread setting (2.2.4, the enabler)

Lands alone and first: P22-12 and P22-13 build on it. It is the API the
exception allows ([D6](#decisions-the-maintainer)).

- **State at 2.2.x.**
  - `--workers N` runs a batch command's files in up to *N* processes:
    `validate`, `verify`, `fix`, `index build` and `recompress`
    (`crates/medh5-cli/src/workers.rs`). The output is the one-process run's,
    byte for byte.
  - `dataset stats --workers N` is the one command whose workers are threads
    (`dataset/stats.rs` `compute_stats`, `std::thread::scope`), because it only
    reads. `docs/explanation/concurrency.md` calls it the exception.
  - Nothing else in the engine starts a thread, and there is no thread
    setting anywhere. The workspace has no thread-pool dependency.
  - Blosc2 has its own thread count, and the vendored filter compresses and
    decompresses with one thread
    (`crates/medh5-sys/vendor/hdf5-blosc2/blosc2_filter.c`).
- **Change.** Settles [D3 and D4](#decisions-the-maintainer).
  - **The setting.** How many threads one call may use for its work outside
    HDF5. The default is 1, which runs today's code and starts no thread.
    - Rust: `medh5::parallel::set_threads(n)` and `medh5::parallel::threads()`.
      No options type gains a field: a call reads the setting.
    - Python: `medh5.set_threads(n)` and `medh5.get_threads()`.
    - The environment: `MEDH5_THREADS`, read once, at the first call that
      asks. A value that is not a positive integer is refused with an error
      naming the variable, never guessed.
    - The command line: `--threads N` on each command that uses it, added by
      the item that makes the command use it. `--workers` keeps its meaning
      (processes, across files).
    - Precedence: a command's flag, then `set_threads`, then the
      environment, then 1. `0` is refused: there is no "auto".
  - **These names are the whole exception.** P22-11 lists them in
    `.github/api-exceptions.txt` with D6, and makes both API jobs read that
    list (see [The API gates](#medh5-223-and-224-parallelism)). A test holds
    the list to these names.
  - **Threads live inside one call.** Each call that uses threads starts them
    (`std::thread::scope`) and joins them before it returns, so no thread
    outlives a call and nothing is left running across a `fork`. There is no
    persistent pool and no new dependency.
  - **Only the calling thread talks to HDF5.** Workers get bytes and return
    bytes. The caller does every HDF5 call, under `h5::locked`, in a fixed
    order, so the file's layout does not depend on which worker finished
    first.
  - **Training loaders stay at one.** `medh5.torch.worker_init_fn` sets one
    thread in each `DataLoader` worker, so a parent's setting is not
    multiplied by the worker count.
- **Tests.** Precedence; the refusals; the environment read once; one thread
  starts no thread (a Rust test counts them); `worker_init_fn` resets it;
  `public_api.py compare` under `patch` passes a listed addition and fails an
  unlisted one.
- **Documentation.** `docs/explanation/concurrency.md` gains "Threads inside
  one call": what the setting covers, why loaders keep one, and `--threads`
  beside `--workers`.
- **Changelog.** Added, area `python-api` (and `cli` for the flags, in the
  items that add them). The entry says the setting is new in a patch by the
  maintainer's decision.
- **Done when.** The setting exists, is documented, and no path uses it yet.

### P22-12 Compress outside the lock, with direct chunk writes (2.2.4)

- **State at 2.2.x.**
  - Chunks are written through `data::write_region` (`h5/data.rs:908`), so
    HDF5 runs the Blosc2 filter inside the write, under its lock.
  - `H5Dwrite_chunk` appears only in tests (`h5/mod.rs:512`,
    `storage/recompress.rs:498`, `conformance/robustness.rs:140`).
  - The read side already bypasses the filter: `h5/window.rs` `read_window`
    reads a stored Blosc2 chunk with `H5Dread_chunk` and decompresses it
    outside the lock, or answers `None` and the caller reads through HDF5.
  - `resample` already works one target chunk at a time, in C order
    (`transforms/resample.rs` `blocks`, line 1146).
  - Three of the four codec profiles (`training`, `balanced`, `archive`)
    store each chunk through Blosc2 alone; `portable` is shuffle and deflate
    (`storage/codecs.rs` `profiles`).
- **Change.**
  - **A chunk writer, the mirror of the window reader.** For a dataset whose
    one filter is Blosc2 (the test `window.rs` `blosc2_chunk` makes), workers
    compress whole chunks with the vendored filter's own compression code and
    the dataset's filter values, outside the lock. The caller stores each with
    `H5Dwrite_chunk`, in chunk order. The same code compresses, so the stored
    bytes are the filter's.
  - `blosc2_filter_function` reports a failure with `H5Epush`, an HDF5 call,
    so it cannot run outside the lock as it is. Its compression is split into
    an entry point that calls no HDF5 function and returns an error code,
    which both the filter and the chunk writer call. `medh5` declares it in
    a private `extern` block, so it is no public Rust name in either crate.
    The change is a patch to
    the vendored filter, listed in its `PATCHES.md`
    (`crates/medh5-sys/vendor/hdf5-blosc2/`) as the others are.
  - **C-Blosc2's process state is touched only by the caller.**
    `blosc2_init` is not safe from two threads at once. The engine already
    starts it once: `medh5_sys::register_blosc2_filter`, which `h5/mod.rs`
    calls before any Blosc2 dataset is opened or made, runs the private
    `start_blosc2` (a `std::sync::Once`). The caller makes sure that has
    run before any worker starts, so no new Rust name is needed for it, and
    the entry point calls no `blosc2_init`.
    Each call compresses with its own context (`blosc2_create_cctx`, as the
    filter does), and never through the global context or a `blosc1_*`
    setter.
  - **One Blosc2 thread per worker, whatever the environment says.**
    `blosc2_create_cctx` and `blosc2_create_dctx` take `BLOSC_NTHREADS`
    over the thread count they are given, so a user's setting would start
    Blosc2's own threads inside every worker. The entry point, and the
    decompression P22-13's workers do through the window reader's path,
    set their context back to one thread after creating it. Other
    `BLOSC_*` settings reach the entry point exactly as they reach the
    filter, since it is the filter's code, so the stored bytes still match.
  - **Tests.** Compressing from four threads as the first Blosc2 use in a
    fresh process; with `BLOSC_NTHREADS=8`, a four-thread call starts no
    more than four threads (a Rust test counts them), and stores the same
    bytes as one thread.
  - Anything else goes through `H5Dwrite` as now: `portable`, a second
    filter, a type HDF5 would convert, a write that covers part of a chunk.
    "HDF5's bytes or HDF5's write", as the reader's invariant says.
  - **The unit is always one whole chunk of the dataset being written**, in
    its own layout. A worker is handed that chunk's values, never a source's
    chunk. A source's stored chunk stands in for it only where the source is
    chunked with exactly the target's chunk shape; anything else is read in
    slabs, as today, cut on the target's chunk boundaries, and each target
    chunk is cut from them.
  - Edge chunks store what `H5Dwrite` stores past the extent; the test
    compares them.
  - **Who uses it**, in the order P22-10 ranks them:
    - `recompress`: where the source's chunk shape is the new layout's,
      read each stored chunk raw, decompress, recompress and write it
      directly, all but the HDF5 calls on workers. The raw read takes what
      `read_window` takes (P22-13 gives the rule); a converted type, an
      unwritten chunk or one stored with Blosc2 skipped is handled as
      `recompress` handles it today, through HDF5. Where the shapes differ
      (`--rechunk`, or a contiguous source given chunks), the slabs of
      `storage/recompress.rs:387` stay, cut on the new chunks' boundaries,
      and only the compression moves to workers;
    - `resample`: workers map, interpolate and compress whole target chunks,
      on 2.2.2's axis-aligned path;
    - `convert from-nifti`: decode a slab of whole chunks, compress on
      workers;
    - an amendment's rewritten datasets (a scrub's, a re-encoded annotation).
  - Each command that uses it gains `--threads N` (D6).
- **Tests.**
  - For each profile and each user, at 1 and 4 threads: the same
    `content_id`, every chunk's stored bytes and filter mask the same
    (`H5Dread_chunk` over every chunk), and the same as 2.2.2's.
  - A chunk written directly reads back through HDF5 and through
    `read_window` alike (Rust).
  - `portable`, a two-filter dataset and a big-endian dataset fall back to
    `H5Dwrite`, at any thread count.
  - `recompress` of a big-endian dataset, of a sparse dataset (unwritten
    chunks) and of one with a filter-skipped chunk gives what it gives
    today.
  - `recompress --rechunk` of a source whose chunks are not the canonical
    ones (smaller, larger, and crossing the new boundaries), and of a
    contiguous source, gives what it gives today, at 1 and 4 threads.
  - A `fork` from another thread during a threaded write leaves a child that
    opens and reads files.
- **Changelog.** Added (`--threads`), area `cli`; Performance, area `writer`.
- **Done when.**
  - P22-10's rows for these commands are measured again at 1, 2 and 4
    threads, before and after, on one machine.
  - One thread is no slower than 2.2.2 by more than noise.

### P22-13 Digests in parallel, outside the lock (2.2.4)

- **State at 2.2.x.**
  - `SampleWriter::commit(true)` recomputes every dataset's digest
    (`sample/writer.rs:1849`, through `stamp_in` and `stamp_digests`), one
    dataset after another. Each streams 32 MiB at a time through HDF5 reads
    (`integrity/digest.rs`, `STREAM_BYTES`), which decompress under the lock.
    `resample` commits with `true` (`transforms/resample.rs:352`), as the
    NIfTI import does.
  - An amendment copies the source's objects raw (`writer.rs` `inherit`,
    `GraphCopy` with `Raw`), then the commit hashes those copies again. That
    is how it refuses a source whose dataset no longer matches its digest
    (E701, `refuse_restamping`) instead of restamping it.
  - 1.0 §14.4 requires the cheaper check: an amendment refuses a file whose
    `content_id` does not recompute from its *stored* digests (E702). The
    E701 refusal is the reference writer's own guarantee, pinned by
    `s13_1_an_amendment_never_restamps_a_digest_its_source_failed`.
  - `verify` (`integrity/verify.rs` `verify_root`) and `validate --level
    integrity` hash the same way, in sequence.
- **Change.** Under [D1](#decisions-the-maintainer), as settled, the
  guarantee stays and its cost falls.
  - Hash datasets on workers, one dataset per worker (a hash is sequential
    within one stream). The caller reads stored chunks raw under the lock,
    and workers decompress them (the Blosc2 path of `window.rs`).
  - **The raw path takes exactly what `read_window` takes, through the same
    checks, and everything else reads through HDF5.** One rule, not a list
    kept by hand: a dataset that is not Blosc2 alone, a stored type HDF5
    would convert on the way out (a big-endian dataset, whose raw bytes are
    not §13.1's little-endian values), strings, and, per slab, a chunk never
    written (it reads as the fill value) or stored with Blosc2 skipped. A
    slab with any such chunk is read through HDF5 whole, so the bytes hashed
    are always HDF5's. The tests hash, against the sequential digest, a
    big-endian dataset, a sparse dataset with unwritten chunks and one with
    a filter-skipped chunk.
  - **The hash sees §13.1's C-order stream, never chunk order.** Where a row
    spans chunks, chunk order is not C order: a 4×4 array in 2×2 chunks
    holds the left halves of two rows before their right halves. So the
    chunks covering one slab, as `STREAM_BYTES` cuts it today, are
    decompressed and placed into that slab in C order (as `window.rs`
    `fill` and `place` assemble a window), and the slab is hashed whole,
    slab after slab. A test hashes a dataset whose chunks split every axis,
    and one whose slabs end inside a chunk, against the sequential digest.
  - One large dataset is pipelined: the next slab's chunks are read and
    decompressed while this slab is hashed, so the hash never waits on the
    lock.
  - **Overlap stays inside one call** (D3: no thread outlives the call that
    started it). A command that amends and commits in one call
    (`resample`, `scrub`, `fix`, `recompress`, an import) hashes what it
    inherited on workers beside its own work, and joins them before it
    returns. A writer used across calls (Python's `amend()`, then edits,
    then `commit()`) hashes what it inherited inside `commit`, in parallel,
    and nothing runs between the calls.
  - Users: `commit`, `verify`, `validate --level integrity`,
    `refuse_failed_digests` and `recompress`'s output check. `verify` and
    `validate` gain `--threads N` (D6).
- **Tests.**
  - The same digests at 1 and 4 threads, for every dtype, strings, compound
    types, and datasets whose slabs fall on and across chunk edges.
  - The E701 refusal at any thread count.
  - `verify`'s report the same, line for line, at 1 and 4 threads.
- **Documentation.** The performance guide's digest row per platform, with
  the `sha2` backend each gets.
- **Changelog.** Performance, area `files`.
- **Done when.** Commit and `verify` on P22-10's sample are measured again at
  1, 2 and 4 threads.

### P22-14 Processes across files (2.2.4)

Its flags wait on [D7](#decisions-the-maintainer). As proposed, 2.2.4 measures
and documents, and adds no flag.

- **State at 2.2.x.**
  - `validate`, `verify`, `fix`, `index build` and `recompress` take
    `--workers N` (processes). A worker costs a process start, about 0.1 s
    from the Python console script, so `validate` is slower with workers
    (`docs/guides/performance.md`).
  - Commands that take many files, or a cohort, without it: `scrub`,
    `splits`, `labels check`, `migrate`, `convert to-nnunet`,
    `dataset check --deep`, `clinical augment-cohort`, `task preflight` and
    `conformance run`.
  - The Python API has no worker count outside `medh5.dataset` statistics.
- **Change.**
  - Time each command above over 32 files, in one process and in four, and
    record the numbers in the pull request and the performance guide: which
    commands would be at least 1.3 times faster on four.
  - Python keeps no worker count. `docs/explanation/concurrency.md` gains a
    `ProcessPoolExecutor` recipe, one file per task.
  - **If D7 takes the flags:** add `--workers` to each command that measured
    at least 1.3 times faster, and list them in `.github/api-exceptions.txt`
    beside P22-11's names. Keep `workers.rs`'s rules: output in input order,
    byte for byte the one-process run's; a writing command's paths that name
    one file go to one worker; a worker that dies fails only its unit.
  - **A worker gets a whole unit of the command, never a bare path.** The
    unit is what the command computes without looking at another, and
    anything shared across the cohort is computed once, by the parent,
    before any worker starts, and handed to each worker as is. The
    measurement in this item names, for each command, its unit and its
    shared state, from the code:
    - `migrate`: a subject's files, which become one sample, with the
      cohort's one label set (`convert/legacy.rs` `migrate_paths`);
    - `convert to-nnunet`: a case, with the dataset's label map and case
      numbering fixed first; `dataset.json` is written by the parent;
    - `task preflight`: a subject, keeping task-cache-1's output order, and
      never threads;
    - `splits`, `labels check`, `dataset check --deep` and `clinical
      augment-cohort`: a file's reading, where the cohort-wide answer
      (the assignment, the comparison, the cohort checks) is then made by
      the parent from every worker's result;
    - `scrub` and `conformance run`: a file, and a case.

    A command whose unit cannot be cut this way from its code does not gain
    the flag, and the pull request says why.
- **Tests.** The recipe runs (`test_docs_python.py`). If D7 takes the flags:
  for each command that gains one, the text, `--json` and the exit code the
  same at 1 and 4 workers, as the existing commands' tests do, on a cohort
  where a unit spans several files (a subject's legacy files for `migrate`)
  and the shared state matters (labels a file lacks, an uneven split).
- **Changelog.** Documentation, or Added, area `cli`, if D7 takes the flags.
- **Done when.** Every command listed has a number and a decision.

## Lanes

Two sessions (lessons §1; at most three, as in 2.2.x). Branch names differ
from 2.2.1's `p22/measure` and `p22/threads`, which are taken.

| Lane | Sequence | Branches |
|---|---|---|
| A, threads in a file | P22-10 → P22-11 → P22-12 | `p22/phase-timings`, `p22/thread-setting`, `p22/chunk-writes` |
| B, across files and digests | P22-14 → P22-13 | `p22/workers`, `p22/digests` |

- **Edges.**
  - P22-10 starts at once, and v2.2.3 is tagged when it merges.
  - Every 2.2.4 item merges after v2.2.3 is tagged, P22-14 included.
  - P22-11 lands alone, before P22-12 and P22-13.
  - P22-13 waits on P22-11. Lane B does P22-14 first, which needs neither
    to build, and holds it for the v2.2.3 tag.
  - P22-12 and P22-13 both read P22-10's ranked list.
- **Critical path.** Lane A: the measurement, the enabler, then the chunk
  writer.
- **Two sessions.** The releases need no third; a third would only wait on
  P22-11.

### Shared files (2.2.3 and 2.2.4)

| File | Owner | Others touch it for |
|---|---|---|
| `crates/medh5/src/parallel.rs` (new), `medh5/__init__.py`'s thread setting, `.github/api-exceptions.txt` (new), `.github/scripts/public_api.py` | P22-11 | P22-14 (its flags, if D7 takes them) |
| `crates/medh5/src/h5/**` (a chunk writer beside `window.rs`) | P22-12 | P22-13 (raw chunk reads for hashing) |
| `crates/medh5/src/transforms/resample*`, `storage/recompress.rs`, `convert/nifti/**` | P22-12 | — |
| `crates/medh5/src/sample/writer.rs` | P22-12 (rewritten datasets) | P22-13 (`commit`, `stamp_in`) |
| `crates/medh5/src/integrity/**` | P22-13 | — |
| `crates/medh5-cli/src/workers.rs` | P22-14 | — |
| `docs/explanation/concurrency.md`, `docs/guides/performance.md` | P22-11, P22-10 | every item (its rows) |
| `tests/project/python_api.txt`, `medh5/_core.pyi`, `docs/reference/cli.md`, `crates/medh5-cli/src/app.rs` | no owner | plans/README.md, "Shared files" |

## Out of scope

| Item | Why |
|---|---|
| A default above one thread, or "auto" | These patches measure first. A later minor may change the default with their numbers, announced a minor ahead. |
| A persistent thread pool, or a thread-pool dependency | Threads that outlive a call cross a `fork` (D3). |
| Threads in preflight or opening | Measured slower: the work is HDF5's (2.1.0; #92). |
| Any other new API or flag | The exception covers P22-11's names and `--threads` only (D6), and P22-14's flags if D7 takes them. |
| MEDS export, then FHIR and OMOP | A later minor (D2). The review of this plan settled two points for whoever plans it: the caller names each sample's MEDS `subject_id` (the importer hashes it into every event id, so none can be minted), and the round trip is promised by rule: export is a fixed point, and an event's id comes back only where its import dropped nothing. A time the importer read inside a daylight-saving fold holds both instants, and its export must keep both. OMOP needs a vocabulary mapping the user supplies. |
| Clinical extension columns in a format minor | Needs the projection study first; candidate for 1.3. |
| `rle` | Only for a measured case `instances` serves badly. |
| `open(swmr=True)` | Small, when someone needs it. SWMR writes are out by design. |
| `medh5.cache/2` | A recorded limit with no demand. |
| Costs out of proportion left by #92: DICOM import holds the volume about 2.3 times, a carried probability map is built whole | Patch material for 2.2.x, planned on their own. |
| HDF5's global-heap fix in crates.io builds | Waits on an upstream HDF5 and `hdf5-metno-src` release (P22-02). |
| Python 3.10 | Stays, unless 2.2.0 E5's trigger fires. |
| The 3.0 ledger | 3.0. |

## Before the lanes start (the maintainer)

1. Keep the repository variable `API_RELEASE_TYPE` at `patch`. P22-11's
   allowed list carries the exception, for Python and Rust alike.
2. Say in each lane's brief whether the lane may arm auto-merge on its own
   pull requests once CI is green and Codex is clean. Without that, it asks
   once, on a card.
3. Optionally, run P22-10's script on an Apple Silicon Mac, for the column
   with SHA extensions.

## Release checklist (the coordinator)

As 2.2.x's, plus:

1. For 2.2.3: no public name and no flag added; `docs/guides/performance.md`
   names the version it measured.
2. For 2.2.4: the API jobs show no addition but the names in
   `.github/api-exceptions.txt`. The release pull request keeps the list;
   the first pull request after v2.2.4 is published empties it.
3. Every thread-identity and worker-identity test passes on every CI
   platform, including the free-threaded wheels.
4. `docs/guides/performance.md` gives 2.2.4's rows at 1, 2 and 4 threads and
   names the machine.
5. `docs/explanation/concurrency.md` describes `--threads` beside
   `--workers`, and `worker_init_fn`'s reset.
6. No changelog sentence promises a later default. If D3's default is to
   change in a later minor, the maintainer has settled it, and the sentence
   names that decision.

## Decisions (the maintainer)

Proposed by this plan unless marked settled. Each is settled when the
maintainer answers it, and the plan is updated with the answer.

- **D1 The digest at commit.** Settled by the maintainer on 2026-10-22: an
  amendment keeps hashing the datasets it copied unchanged, so it still
  refuses a source whose dataset fails its digest (E701), and P22-13 makes
  that hashing parallel, overlapped with the edit inside commands that amend
  and commit in one call. Not chosen: keeping the stored digests of
  unchanged copies (a damaged source dataset would be carried rather than
  refused), and an opt-out flag.
- **D2 Scope.** Settled by the maintainer on 2026-10-22: parallelism and
  digests only. The MEDS export moves to a later minor (Out of scope).
- **D3 The thread setting.** Proposed: one process-wide count, default 1, no
  "auto"; threads started and joined inside each call, with no persistent
  pool and no new dependency; only the calling thread calls HDF5; loaders'
  workers reset to one.
- **D4 Names.** Proposed: `--threads N` for threads inside one call,
  `--workers N` for processes across files. `dataset stats --workers` keeps
  its meaning (threads) and is documented as the exception, as now.
- **D6 Patches, not a minor.** Settled by the maintainer on 2026-10-22: the
  work ships as 2.2.3 and 2.2.4, not 2.3.0, and the patch rule is relaxed
  this once for the thread setting, a small API change. First planned as
  2.3.0 (D5, when its lanes start, fell away with it).
- **D7 `--workers` on more commands.** Open, on a card: whether the
  exception also covers P22-14's `--workers` flags. Proposed: no; 2.2.4
  measures them and adds the recipe, and the flags wait for the next minor.
