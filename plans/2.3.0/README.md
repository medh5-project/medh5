# medh5 2.3.0

A minor release about parallelism: threads for the work inside one file that
runs outside HDF5's lock, processes for the work across files, and the cost of
digests at commit. It also takes the MEDS export that 2.2.0 named a candidate
for 2.3 ([D2](#decisions-the-maintainer)). Read
[plans/README.md](../README.md) first. Its rules hold here, with `w23/`
branches and `W23-` titles.

- **Baseline.** v2.2.1 plus #96 and #98. Every "State at 2.2.x" claim below
  was checked against `main` at 5a8a2b4, with that commit's line numbers.
  2.2.2 is still in progress (P22-09, #99, and lane B's axis-aligned resample
  path), so a workstream checks each claim against `main` when it starts.
- **Where the reasons come from.**
  - [What 2.2.0 taught](../2.2.x/what-2.2.0-taught.md), cited as "lessons §n".
  - The 2.2.x measurements: #92's tables (P22-06), and lane B's phase timings
    of the 512³ resample, summarised in
    [The numbers](#the-numbers-this-plan-starts-from).
- **What the release may change.**
  - New API, additive, under plans/README.md's Rust and Python rules. Every
    new knob defaults to what 2.2.x does.
  - The format stays 1.2, and the diagnostic code table keeps its codes.
    Nothing here needs a format change.
  - **No thread count changes an output.** Every path this plan parallelises
    writes the same `content_id`, the same stored chunk bytes and the same
    command output at any thread or worker count. Tests hold that for each
    path (see each item's Tests).

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
- **The 512³ resample, one thread** (lane B, a timing build, before the
  axis-aligned path; 2.2.2 changes the mapping row):
  - nearest, no annotation, 26.5 s: mapping 14.2 s, **digests at commit
    8.2 s**, writing and compressing 2.1 s, reading 0.4 s, sampling 0.3 s;
  - linear with the annotation, 50.6 s: mapping 12.6 s, interpolating
    12.2 s, **digests 11.7 s**, carrying the label map 7.7 s, writing 4.9 s.
  - Of the digest time, 576 MiB is the source's untouched images, which the
    commit hashes again
    ([W23-04](#w23-04-digests-in-parallel-outside-the-lock)).
  - The machine (a 4-vCPU Linux VM) has no SHA instructions. Apple Silicon
    and recent x86_64 have them, so digests cost much less there.

## How this plan answers what came before

| Before | So in 2.3.0 |
|---|---|
| 2.2.x left threads out: a thread pool is a new behaviour and needs a setting (P22-09). | 2.3.0 is the minor that can add the setting. It defaults to one thread, today's code path. |
| 2.1.0 added a preflight thread pool and it ran slower. Nothing was measured before it was built. | W23-01 measures how much of each command runs outside HDF5's lock before any thread code is written, and drops an item that cannot pay. |
| P22-06's table set the work of P22-07 to P22-09, and each of those recorded its numbers (lessons §9). | Every item records 1, 2 and 4 threads before and after, on one machine, in its pull request. |
| P22-08 and P22-09 changed hot paths with every digest pinned to 2.2.0's. | Every parallel path is tested for the same `content_id` and the same stored chunks at 1 and 4 threads. |
| Arming auto-merge on #96 was stopped by the session's own permission check, and waited for a card. | Each lane's brief says whether the maintainer allows it to arm auto-merge on its own pull requests ([Before the lanes start](#before-the-lanes-start-the-maintainer)). |
| The weekly limit set 2.2.0's release date (lessons §1). | Three lanes at most, with the enabler first, as in 2.2.x. |

## Work items

### W23-01 Measure what can run outside the lock (start at once)

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
  - A ranked list that sets the scope of W23-03 and W23-04. A path whose
    ceiling at four threads is below 1.3 times is dropped, and the plan says
    so.
  - If the maintainer runs the script on Apple Silicon, a second column.
- **Done when.** The table covers every row above, after 2.2.2's resample
  changes have merged.

### W23-02 One thread setting (the enabler)

Lands alone and first: W23-03 and W23-04 build on it.

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
      A new options type that takes a count is `#[non_exhaustive]` with a
      `with_threads` method.
    - Python: `medh5.set_threads(n)` and `medh5.get_threads()`.
    - The environment: `MEDH5_THREADS`, read once, at the first call that
      asks. A value that is not a positive integer is refused with an error
      naming the variable, never guessed.
    - The command line: `--threads N` on each command that uses it, added by
      the item that makes the command use it. `--workers` keeps its meaning
      (processes, across files).
    - Precedence: a command's flag, then `set_threads`, then the
      environment, then 1. `0` is refused: there is no "auto" in 2.3.0.
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
  starts no thread (a Rust test counts them); `worker_init_fn` resets it.
- **Documentation.** `docs/explanation/concurrency.md` gains "Threads inside
  one call": what the setting covers, why loaders keep one, and `--threads`
  beside `--workers`.
- **Changelog.** Added, area `python-api` (and `cli` for the flags, in the
  items that add them).
- **Done when.** The setting exists, is documented, and no path uses it yet.

### W23-03 Compress outside the lock, with direct chunk writes

- **State at 2.2.x.**
  - Chunks are written through `data::write_region` (`h5/data.rs:908`), so
    HDF5 runs the Blosc2 filter inside the write, under its lock.
  - `H5Dwrite_chunk` appears only in tests (`h5/mod.rs:512`,
    `storage/recompress.rs:498`, `conformance/robustness.rs:100`).
  - The read side already bypasses the filter: `h5/window.rs` `read_window`
    reads a stored Blosc2 chunk with `H5Dread_chunk` and decompresses it
    outside the lock, or answers `None` and the caller reads through HDF5.
  - `resample` already works one target chunk at a time, in C order
    (`transforms/resample.rs` `blocks`, line 977).
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
    which both the filter and the chunk writer call. The change is a patch to
    the vendored filter, listed in its `PATCHES.md`
    (`crates/medh5-sys/vendor/hdf5-blosc2/`) as the others are.
  - Anything else goes through `H5Dwrite` as now: `portable`, a second
    filter, a type HDF5 would convert, a write that covers part of a chunk.
    "HDF5's bytes or HDF5's write", as the reader's invariant says.
  - Edge chunks store what `H5Dwrite` stores past the extent; the test
    compares them.
  - **Who uses it**, in the order W23-01 ranks them:
    - `recompress`: read each stored chunk raw, decompress, recompress and
      write it directly, all but the HDF5 calls on workers;
    - `resample`: workers map, interpolate and compress whole target chunks
      (after 2.2.2's axis-aligned path);
    - `convert from-nifti`: decode a slab of whole chunks, compress on
      workers;
    - an amendment's rewritten datasets (a scrub's, a re-encoded annotation).
  - Each command that uses it gains `--threads N`.
- **Tests.**
  - For each profile and each user, at 1 and 4 threads: the same
    `content_id`, every chunk's stored bytes and filter mask the same
    (`H5Dread_chunk` over every chunk), and the same as `main` before the
    change.
  - A chunk written directly reads back through HDF5 and through
    `read_window` alike (Rust).
  - `portable` and a two-filter dataset fall back, at any thread count.
  - A `fork` from another thread during a threaded write leaves a child that
    opens and reads files.
- **Changelog.** Added (`--threads`), area `cli`; Performance, area `writer`.
- **Done when.**
  - W23-01's rows for these commands are measured again at 1, 2 and 4
    threads, before and after, on one machine.
  - One thread is no slower than 2.2.2 by more than noise.

### W23-04 Digests in parallel, outside the lock

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
- **Change.** Under [D1](#decisions-the-maintainer) as proposed, the
  guarantee stays and its cost falls.
  - Hash datasets on workers, one dataset per worker (a hash is sequential
    within one stream). The caller reads stored chunks raw under the lock,
    and workers decompress them (the Blosc2 path of `window.rs`). A dataset
    that is not Blosc2 alone is read through HDF5, as now.
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
    `refuse_failed_digests` and `recompress`'s output check.
  - If the maintainer picks option 2 of D1 instead, the commit keeps the
    stored digest of every dataset it copied unchanged and hashes only what
    it wrote. That is a behaviour change with its own changelog entry, and
    the test above is rewritten to pin the new rule.
- **Tests.**
  - The same digests at 1 and 4 threads, for every dtype, strings, compound
    types, and datasets whose slabs fall on and across chunk edges.
  - The E701 refusal at any thread count.
  - `verify`'s report the same, line for line, at 1 and 4 threads.
- **Documentation.** The performance guide's digest row per platform, with
  the `sha2` backend each gets.
- **Changelog.** Performance, area `files`.
- **Done when.** Commit and `verify` on W23-01's sample are measured again at
  1, 2 and 4 threads.

### W23-05 `--workers` on every batch command that gains from it

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
  - Time each command above over 32 files, in one process and in four,
    first. Add `--workers` to each one that is at least 1.3 times faster on
    four, and list the rest with their numbers in the pull request.
  - Keep `workers.rs`'s rules: output in input order, byte for byte the
    one-process run's; a writing command's paths that name one file go to
    one worker; a worker that dies fails only its file.
  - `task preflight` splits by subject and keeps task-cache-1's output order.
    It never uses threads (lessons above).
  - Python keeps no worker count. `docs/explanation/concurrency.md` gains a
    `ProcessPoolExecutor` recipe, one file per task.
- **Tests.** For each command that gains the flag, the text, `--json` and the
  exit code the same at 1 and 4 workers, as the existing commands' tests do.
- **Changelog.** Added, area `cli`.
- **Done when.** Every command listed has a number and a decision.

### W23-06 MEDS export

Under [D2](#decisions-the-maintainer) as proposed.

- **State at 2.2.x.**
  - `medh5/io/meds.py` imports MEDS 0.3 (`read_meds`, `augment_from_meds`,
    `augment_cohort_from_meds`), mapping rows to clinical events and inventing
    nothing MEDS does not say. The `meds` extra is `pyarrow>=14`.
  - Nothing exports. The 2.2.0 plan's Out of scope named the export as a
    candidate for 2.3, MEDS first, because it round-trips with the importer.
- **Change.**
  - `medh5.io.meds.write_meds(paths, out, …)` and
    `medh5 clinical export-meds`, writing a MEDS 0.3 dataset (`data/` shards
    and `metadata/dataset.json`) from the samples' clinical histories.
  - The mapping is the importer's, inverted, and the pull request gives it as
    a table: a `static` event to a null `time`, a code system back to its
    declared prefix, `value_num` and `value_text`, the declared `unit` and
    `available` columns.
  - What MEDS cannot hold is never dropped silently. An interval's end, a
    second time pair, a revision chain: each goes into a declared extension
    column, or is listed in the report as not exported. Which is the lane's
    call, recorded in the pull request.
  - Documents' text is exported only when asked (`texts=True`), as on import.
- **Tests.** Import, export and import again gives the same events (ids,
  codes, values, both time pairs) for every MEDS fixture the importer's tests
  use. A history MEDS cannot hold reports what it left out.
- **Documentation.** The clinical guide gains "Export to MEDS".
- **Changelog.** Added, area `clinical`.
- **Done when.** The round trip passes, and the CLI reference lists the
  command.

## Lanes

At most three sessions at once (lessons §1).

| Lane | Sequence | Branches |
|---|---|---|
| A, threads in a file | W23-01 → W23-02 → W23-03 | `w23/measure`, `w23/threads`, `w23/chunk-writes` |
| B, across files and digests | W23-05 → W23-04 | `w23/workers`, `w23/digests` |
| C, clinical | W23-06 | `w23/meds-export` |

- **Edges.**
  - W23-01 waits on 2.2.2's resample pull requests, which change what it
    measures. It is the only item that may open before 2.2.2 is tagged,
    because it changes no behaviour.
  - W23-02 lands alone, before W23-03 and W23-04.
  - W23-04 waits on W23-02. Lane B does W23-05 first, which needs neither.
  - W23-03 and W23-04 both read W23-01's ranked list.
  - W23-06 waits on nothing.
- **Critical path.** Lane A: the measurement, the enabler, then the chunk
  writer.
- **With two sessions.** Lane C's work follows lane B's.

### Shared files (2.3.0)

| File | Owner | Others touch it for |
|---|---|---|
| `crates/medh5/src/parallel.rs` (new), `medh5/__init__.py`'s thread setting | W23-02 | — |
| `crates/medh5/src/h5/**` (a chunk writer beside `window.rs`) | W23-03 | W23-04 (raw chunk reads for hashing) |
| `crates/medh5/src/transforms/resample*`, `storage/recompress.rs`, `convert/nifti/**` | W23-03 | — |
| `crates/medh5/src/sample/writer.rs` | W23-03 (rewritten datasets) | W23-04 (`commit`, `stamp_in`) |
| `crates/medh5/src/integrity/**` | W23-04 | — |
| `crates/medh5-cli/src/workers.rs` | W23-05 | — |
| `medh5/io/meds.py` | W23-06 | — |
| `docs/explanation/concurrency.md`, `docs/guides/performance.md` | W23-02, W23-01 | every item (its rows) |
| `tests/project/python_api.txt`, `medh5/_core.pyi`, `docs/reference/cli.md`, `crates/medh5-cli/src/app.rs` | no owner | plans/README.md, "Shared files" |

## Out of scope

| Item | Why |
|---|---|
| A default above one thread, or "auto" | 2.3.0 measures first. A later minor may change the default with 2.3.0's numbers, announced a minor ahead. |
| A persistent thread pool, or a thread-pool dependency | Threads that outlive a call cross a `fork` (D3). |
| Threads in preflight or opening | Measured slower: the work is HDF5's (2.1.0; #92). |
| FHIR or OMOP export | After MEDS, if wanted (D2). OMOP needs a vocabulary mapping the user supplies. |
| Clinical extension columns in a format minor | Needs the projection study first; candidate for 1.3. |
| `rle` | Only for a measured case `instances` serves badly. |
| `open(swmr=True)` | Small, when someone needs it. SWMR writes are out by design. |
| `medh5.cache/2` | A recorded limit with no demand. |
| Costs out of proportion left by #92: DICOM import holds the volume about 2.3 times, a carried probability map is built whole | Patch material for 2.2.x, not features. |
| HDF5's global-heap fix in crates.io builds | Waits on an upstream HDF5 and `hdf5-metno-src` release (P22-02). |
| Python 3.10 | Stays, unless 2.2.0 E5's trigger fires. |
| The 3.0 ledger | 3.0. |

## Before the lanes start (the maintainer)

1. Tag v2.2.2. Until then `main` is the patch line, and only W23-01 opens.
2. Clear the repository variable `API_RELEASE_TYPE`, so the API jobs check
   `main` as a minor. A 2.2.3, if one is needed after that, is cut on
   `release/2.2` from `v2.2.2`.
3. Say in each lane's brief whether the lane may arm auto-merge on its own
   pull requests once CI is green and Codex is clean. Without that, it asks
   once, on a card.
4. Optionally, run W23-01's script on an Apple Silicon Mac, for the column
   with SHA extensions.

## Release checklist (the coordinator)

As 2.2.x's, plus:

1. Every thread-identity and worker-identity test passes on every CI
   platform, including the free-threaded wheels.
2. `docs/guides/performance.md` gives 2.3.0's rows at 1, 2 and 4 threads and
   names the machine.
3. `docs/explanation/concurrency.md` describes `--threads` beside
   `--workers`, and `worker_init_fn`'s reset.
4. No changelog sentence promises a later default. If D3's default is to
   change in a later minor, the maintainer has settled it, and the sentence
   names that decision.

## Decisions (the maintainer)

Proposed by this plan. Each is settled when the maintainer answers it, and
the plan is updated with the answer.

- **D1 The digest at commit.** Does an amendment keep hashing the datasets it
  copied unchanged? Proposed: yes, and W23-04 makes it parallel, overlapped
  with the edit inside commands that amend and commit in one call. The
  alternatives are to keep the stored digests of unchanged
  copies (cost follows the edit, but a damaged source dataset is carried with
  its failing digest rather than refused; `verify` still finds it), or to
  keep the check by default with an opt-out flag.
- **D2 Scope.** Proposed: parallelism and digests, plus the MEDS export. The
  alternatives are parallelism alone, or MEDS and FHIR export.
- **D3 The thread setting.** Proposed: one process-wide count, default 1, no
  "auto"; threads started and joined inside each call, with no persistent
  pool and no new dependency; only the calling thread calls HDF5; loaders'
  workers reset to one.
- **D4 Names.** Proposed: `--threads N` for threads inside one call,
  `--workers N` for processes across files. `dataset stats --workers` keeps
  its meaning (threads) and is documented as the exception, as now.
- **D5 When.** Proposed: the lanes start once v2.2.2 is tagged; W23-01 may
  start before.
