# medh5 2.2.x

A series of patch releases on 2.2.0: fixes, performance and hardening, and no
new features. No 2.3.0 is planned for now. Read [plans/README.md](../README.md)
first. Its rules hold here, with `p22/` branches and `P22-` titles; the
changes summarised under [Process](#process) were made to it with this plan.
The reasons are in [What 2.2.0 taught](what-2.2.0-taught.md), cited below as
"lessons §n".

- **Baseline.** v2.2.0, released 2026-10-20 from `main` at ea7b8fa. Every
  "State at 2.2.0" claim below was checked against `main` at 5249d79 (v2.2.0
  plus #86).
- **Line.**
  - `main` is the 2.2.x line while no 2.3.0 is planned, and each patch is
    tagged from it (`v2.2.1`, `v2.2.2`, …); the maintainer settled this on
    2026-10-21.
  - If a feature must land before the series ends, `release/2.2` is cut from
    the last 2.2.x tag first, and the rest of the series moves there, as
    [the 2.0.x line](../README.md#the-20x-line) does.
- **What a patch may change.**
  - It may fix a wrong answer, a crash, a hang, a cost out of proportion to
    the file, the documentation, or CI.
  - It adds no public API, in Rust or Python, and no command-line flag.
    `cargo semver-checks` and the API snapshot hold the first. Review holds
    the rest: a new name or flag fails review.
  - The format stays 1.2, and the diagnostic code table keeps its 100 codes.
    A new bound reports through an existing code.
  - A performance change writes the same `content_id` as 2.2.0 for the same
    input. The conformance corpus and a before/after check in the pull
    request hold that.
- **Older lines.** 2.0.x and 2.1.x are closed (the maintainer, 2026-10-21):
  every fix lands on 2.2.x only, and nothing is backported. P22-05 makes the
  repository say so. The release-day overflow (lessons §4) would not have
  needed a backport anyway: release builds gave a wrong number in a message,
  not a crash.

## What 2.2.0 taught, and how this plan answers it

| In 2.2.0 | So in 2.2.x |
|---|---|
| Three lanes stopped for four days on the weekly limit. Lane F's uncommitted work was lost when its environment was reset. | At most three lanes at once, each starting on what the release needs first. Work is committed and pushed before a turn ends (Process). |
| Codex left 148 findings in classes, and fixing only the named instance meant another round. 69 % of pull-request CI runs were cancelled by the next push. | A finding is swept as a class, with the sweep listed in the reply, and each round is one push (Process). P22-01 does that for the overflow class across the engine. |
| The fuzz job found a real overflow on release day, and only because the release session looked. Red had become normal. The global-heap hang is excluded in the fuzz target but still hangs a user. | P22-02 stops the hang in the engine and lifts the exclusion. P22-04 makes a red scheduled run something the coordinator sees and the release cannot pass. |
| Stale docs (`CLAUDE.md`'s format version, `SECURITY.md`'s dependencies, the PyPI links) and a two-day-old red lock refresh were found on release day. | P22-05 adds tests that hold those to the code. The coordinator's sweep covers bot pull requests (Process). |
| Nothing in 2.2.0 was measured: not free-threading's cost on one thread or several, nor the NIfTI engine. Several paths hold whole volumes. | 2.2.2 measures first (P22-06), then fixes what the measurement shows (P22-07, P22-08) and refreshes the published numbers. |
| About ten Codex findings were declined with reasons and never revisited. | P22-03 turns each decline into a test that pins the decision, or a fix. |
| Sessions could not delete branches or reach Read the Docs. | Two settings for the maintainer ([Before the lanes start](#before-the-lanes-start-the-maintainer)). |

## Releases

| Release | Holds | Tagged when |
|---|---|---|
| 2.2.1 | P22-01 to P22-05: hostile input, declined findings, fuzz as a gate, docs held to code | its five pull requests have merged and the release checklist passes |
| 2.2.2 | P22-06 to P22-08: performance, measured first | its three pull requests have merged |

2.2.1 does not wait for the performance work. If P22-06's measurement finds a
regression against 2.1.0 that is a bug (a slower answer, not a missed
speedup), its fix may go into 2.2.1 instead.

## Work items

### P22-01 The overflow class, swept once (2.2.1)

- **State at 2.2.0.**
  - #85 fixed one subtraction in `annotations/layout.rs`.
  - `geometry::affine::box_to_slices` turns a box's float bounds into
    integer indices. Outside tests it has nine callers: eight in the engine,
    in six files, and one in the Python binding:
    - `annotations/layout.rs` (the one #85 fixed);
    - `annotations/encode.rs`, two, one of which subtracts the bounds
      unchecked (`hi - lo`);
    - `annotations/read.rs`, two;
    - `annotations/read_geometric.rs`;
    - `curation/tracking.rs`, which subtracts unchecked and multiplies
      (`(b - a)` then `product()`);
    - `transforms/resample/annotations.rs`;
    - `crates/medh5-python/src/geometry.rs` (`box_to_slices`).
  - These nine are where the sweep starts, not where it ends: the same
    arithmetic on stated values appears without that function.
  - The NIfTI reader already checks its own arithmetic (`convert/nifti/read.rs`,
    `header.rs`).
  - Release builds have `overflow-checks` off, so an overflow there is silent.
- **Change.**
  - Audit every integer computation on a value the file states: box bounds,
    extents, offsets, shapes, chunk counts, grid sizes in `resample`, and ITK
    transform layouts.
  - Make each one checked, or saturating where a saturated value still
    gives the right answer.
  - A file whose arithmetic does not fit is a finding through the code that
    already covers the field, never a panic or a wrapped number.
  - List every site examined in the pull request, including the ones that were
    already safe.
- **Tests.** One test per site that could overflow, named for its clause, as
  `s7_4_a_box_past_the_integers_is_a_finding_not_a_panic` is.
- **Question (decided by P22-06's numbers).** Should the release profile turn
  `overflow-checks` on? That would turn a wrapped number that was missed into
  a panic. PyO3 raises a panic as `PanicException`, which is loud rather than
  wrong. Do it only if the benchmark shows it costs under 2 %.
- **Changelog.** Fixed, area `validation` (or the area hit).
- **Done when.** The sweep list is complete, every site that could overflow
  has a test, and `open_validate` runs its scheduled 40 minutes green.

### P22-02 The global-heap hang, stopped in the engine (2.2.1)

- **State at 2.2.0.**
  - `fuzz/src/lib.rs` (`endless_global_heap`) keeps `open_validate` off a
    global-heap collection that HDF5 2.2.0 (through hdf5-metno-src 0.10.4)
    steps through without end.
  - The engine has no such guard, so `medh5 validate` on such a file hangs.
  - The details of the HDF5 defect stay out of the public plan and the pull
    request, as W22-06 required, and the maintainer decides when to report it
    upstream.
- **Where HDF5 is built.** `medh5-sys` does not build HDF5:
  `hdf5-metno-src` builds it for `hdf5-metno-sys`, before `medh5-sys`'s
  build script runs (its `build.rs` says so). So a patch shipped inside
  `medh5-sys` would not reach the linked library. A patch reaches what the
  workspace builds only through the crate that builds HDF5.
- **Change.** Patch HDF5 (the maintainer chose this over checking files
  before HDF5 reads them, 2026-10-21).
  - Vendor `hdf5-metno-src` with a minimal patch to HDF5's global-heap
    decoder, and point the workspace at it with `[patch.crates-io]`. The
    patch is listed as the Blosc2 filter's patches are
    (`vendor/hdf5-blosc2/PATCHES.md`), and kept until it lands upstream.
  - That covers everything the release workflow builds: the wheels, the
    sdist (which must carry the vendored crate), the standalone binaries
    and the Homebrew formula, which installs those binaries.
  - It does not cover a crates.io build (`cargo install medh5-cli`, or a
    crate depending on `medh5`): `[patch]` does not reach dependents. Those
    get the fix from an upstream HDF5 and `hdf5-metno-src` release, so the
    defect is reported upstream with the patch.
    `SECURITY.md` says which builds are patched until then.
  - HDF5's row in `SECURITY.md`'s native inventory says it is patched, and
    the test that compares the inventory with the lock follows.
  - The fuzz exclusion (`endless_global_heap`) is removed, so the job fuzzes
    that path again.
  - The maintainer decides when the defect is reported upstream, and offers
    the patch with it.
- **Tests.** A built file that hung 2.2.0 is refused in bounded time. The
  test builds the file in the test; no input is committed. The release
  workflow's smoke tests run it against the wheels and the standalone
  binary.
- **Changelog.** Security, area `validation`.
- **Done when.** The exclusion is gone and the scheduled `open_validate` run is
  green.

### P22-03 Declined findings, settled for good (2.2.1)

- **State at 2.2.0.** About ten Codex findings were declined with a reason
  (lessons §2):
  - three on #75, about E3 parity;
  - three or four on #84: the `valid_mask` report, path-qualified
    `ignore_mask`, reserved IDs where only the message changed, and the
    `ResampleReport` order;
  - two on #76: the dict values snapshot and the registry fork barrier;
  - one on #74, which did not reproduce;
  - one on #82, about ITK's B-spline naming.
- **Change.**
  - Re-read each one against `main`.
  - A decline that still holds gets a test that pins the behaviour it chose,
    named for the reason.
  - A decline that no longer holds, or that was a correct finding put off, is
    fixed, as long as the fix needs no new API.
  - The pull request lists all of them with their outcome.
- **Done when.** Every declined finding has a test or a fix.

### P22-04 Fuzzing as a gate, and the engine's other readers fuzzed (2.2.1)

- **State at 2.2.0.**
  - `fuzz.yml` runs on Tuesdays and Fridays and on pull requests that touch
    `fuzz/`.
  - It is not part of `ci-ok`, and a red scheduled run notifies no one in the
    project.
  - The engine reads ITK HDF5 transform files (`convert/itk.rs`, W22-08), and
    no target fuzzes them.
- **Change.**
  - Add an `itk_transform` target that reads any file as an ITK transform file,
    with the same limits as `open_validate`.
  - Add a pull-request smoke run whenever `crates/medh5/src/convert/` or
    `annotations/` change, not only `fuzz/`.
  - Keep crash inputs out of artifacts, as now. The workflow's comment says
    why.
- **Process side (plans/README.md).**
  - The coordinator's sweep looks at the last scheduled fuzz run of every
    target.
  - The release checklist needs a green run of every target on the release
    pull request's head (`workflow_dispatch`).
- **Done when.** The new target has run 40 minutes green, and the release
  checklist names the fuzz run.

### P22-05 Documentation held to the code, and the older lines closed (2.2.1)

- **State at 2.2.0.** These were wrong until release day, and no test caught
  them:
  - `CLAUDE.md`'s "Format X, package Y" line;
  - the runtime dependencies `SECURITY.md` names;
  - the PyPI project links to the normative specifications;
  - a changelog fragment's promise about "next".
- **Change.** A test for each:
  - `CLAUDE.md`'s format and package versions equal `version::LATEST_VERSION`
    and `[workspace.package] version`, major and minor;
  - `SECURITY.md`'s runtime dependencies equal `pyproject.toml`'s;
  - every `docs/spec/medh5-1.*.md` has a project URL in `pyproject.toml`;
  - `changelog.py check` flags a fragment that says "next release", "later
    release" or "will", for the release checklist to confirm.
- **Closing 2.0.x and 2.1.x.** The maintainer closed both lines (2026-10-21).
  - `advisories.yml`'s weekly matrix audits `main` and `release/2.2` instead
    of `main` and `release/2.0`; a line not cut yet still ends its job at
    the first step. `test_packaging.py`'s
    `test_the_weekly_audit_reads_every_line` follows.
  - `SECURITY.md` gains "Supported versions": the newest 2.2.x gets fixes,
    and earlier lines do not.
  - The `release/2.0` branch stays, as the record of 2.0.3.
- **Changelog.** Documentation, area `release`: only the newest 2.2.x
  receives fixes.
- **Done when.** Each test fails on 2.2.0's text before the fix. That is shown
  in the pull request by reverting one line.

### P22-06 Measure 2.2.0 (2.2.2, start at once)

- **State at 2.2.0.**
  - `docs/guides/performance.md`'s numbers were measured with 2.0.0.
  - No 2.2.0 pull request recorded a timing.
  - The only published thread figure is 2.1.0's preflight thread pool (3.6 s
    on one thread, 6.9 s on four), which is why preflight runs on one
    thread. No benchmark reads from several threads: `bench_preflight.py`
    runs preflight once, with no thread count.
- **Change.** Measure, and change no code:
  - `medh5 bench` and `docs/examples/bench_*.py`, on 2.1.0 and 2.2.0, on one
    machine;
  - NIfTI import and export through the engine against 2.1.0's nibabel path;
  - reading from one, two and four threads, on the GIL build and the
    free-threaded build, through medh5 (`bench_io.py` times h5py): a new
    `docs/examples/bench_threads.py` on the pattern
    `docs/explanation/concurrency.md` shows, one handle per thread.
    Preflight stays on one thread by design and is not a thread row;
  - the cost of detaching on every binding call, against 2.1.0, on one thread;
  - peak memory (RSS) of `resample` (linear and cubic), `from_nifti` with
    several volumes, and an enhanced DICOM import.
- **Output.**
  - A table in the pull request.
  - Refreshed "The numbers" in `docs/guides/performance.md`, with the machine
    and version.
  - A short list of regressions and of costs out of proportion, ranked. That
    list sets P22-07 and P22-08.
  - Optionally, a scheduled, non-gating benchmark job that reports a change
    of more than 20 %. Runner timings are too noisy to gate on.
- **Done when.** The table covers every row above and the page is refreshed.

### P22-07 What free-threading costs, removed where it is excess (2.2.2)

- **State at 2.2.0.**
  - HDF5 runs one call at a time (one global lock).
  - Every binding call detaches (`crate::h5`), and live views take an `RLock`
    (`medh5/_live.py`).
  - What that costs is unknown until P22-06's thread rows exist. If they
    show neither problem below, P22-07 is dropped and the plan says so.
- **Change.**
  - Find where the time goes (lock hand-offs, detach and attach per HDF5
    call, the views' locks) and remove the excess. For example, take the
    HDF5 lock once per batch of calls, not once per call. Keep 2.2.0's
    guarantees (`docs/explanation/concurrency.md`) as written.
  - No speedup is promised: HDF5 still takes turns.
- **Tests.** The free-threaded stress job still passes. A timing is a
  benchmark row, not a test.
- **Done when.**
  - Four threads take no longer than one, within noise, on P22-06's thread
    read benchmark.
  - One thread is no slower than 2.1.0 by more than noise.

### P22-08 Memory follows a chunk, not a volume (2.2.2)

- **State at 2.2.0.**
  - `resample` works a block of the target at a time, but `valid_mask` and
    carried annotations are built whole, because `add_mask` and
    `write_annotation` take whole arrays.
  - `from_nifti` holds every input volume until the write begins.
  - A NIfTI image is held twice during decode.
  - Cubic resampling prefilters the whole source, by design.
- **Change.**
  - Write `valid_mask` and carried masks block by block, through a
    `pub(crate)` path in the writer. The public signatures do not change.
  - Write each NIfTI volume before decoding the next.
  - Decode into the output buffer.
  - Document cubic's whole-source cost on the resampling guide; it is
    unchanged.
- **Tests.** For each path, a Rust test of the slab budget, as the window reader
  has. A Python test checks the same `content_id` as 2.2.0 for the same input.
- **Done when.** Peak memory on P22-06's rows is about one image plus chunk
  buffers, and the measurement is in the pull request.

## Lanes

At most three sessions at once (lessons §1).

| Lane | Sequence | Branches |
|---|---|---|
| A, hostile input | P22-01 → P22-02 → P22-04 | `p22/overflow`, `p22/global-heap`, `p22/fuzz-gate` |
| B, performance | P22-06 → P22-07 → P22-08 | `p22/measure`, `p22/threads`, `p22/memory` |
| C, settled and documented | P22-05 → P22-03 | `p22/docs-held`, `p22/declined` |

- **Edges.**
  - P22-04's new target runs after P22-01, so that it does not rediscover
    what P22-01 fixes.
  - P22-07 and P22-08 wait on P22-06's list.
  - Nothing else waits.
- **Critical paths.** Lane A is 2.2.1's critical path. Lane B is 2.2.2's.
- **With two sessions.** Merge C into A after A's first pull request.

### Shared files (2.2.x)

| File | Owner | Others touch it for |
|---|---|---|
| `crates/medh5/src/annotations/**`, `geometry/affine.rs` | P22-01 | P22-03 (tests only) |
| `crates/medh5-sys/**`, the vendored `hdf5-metno-src` and the workspace `[patch.crates-io]`, `fuzz/**`, `.github/workflows/fuzz.yml` | P22-02, then P22-04 | — |
| `crates/medh5/src/transforms/resample*`, `sample/writer*.rs`, `convert/nifti/**` | P22-08 | P22-01 (checked arithmetic), P22-03 (#84's declines) |
| `crates/medh5/src/h5/**`, `crates/medh5-python/src/**`, `medh5/_live.py` | P22-07 | P22-02 (if the guard sits in `h5/`) |
| `docs/guides/performance.md` | P22-06 | P22-07, P22-08 (their rows) |
| `CLAUDE.md`, `CONTRIBUTING.md`, `SECURITY.md`, `pyproject.toml`'s URLs, `.github/workflows/advisories.yml`, `tests/project/test_packaging.py` | P22-05 | P22-02 (`SECURITY.md`'s HDF5 row) |

## Process

Made to plans/README.md with this plan, and summarised here.

- **Branches.**
  - "Branches and pull requests" gains the 2.2.x line: `p22/<name>`, `P22-`
    titles, and patches tagged from `main` while no minor is planned.
  - A lane uses the branch names its brief gives. Where the session's branch
    carries a suffix, the brief records the name it has.
  - One branch per pull request, never reused for the next.
- **Committing.** Work is committed and pushed before a turn ends. When the
  pull request is open, the branch is still pushed, because a cancelled CI run
  costs less than lost work. 2.2.0 lost lane F's annotation carry to an
  environment reset.
- **Review.**
  - On a Codex finding, find every instance of its class, fix them together,
    and list the sweep in the reply.
  - Push once per round, not once per finding.
- **The coordinator's sweep.** Besides lane pull requests, it covers:
  - bot pull requests (the weekly lock refresh);
  - the last scheduled fuzz run of each target;
  - `main`'s last CI.
- **Lanes.** At most three at once (lessons §1). "Waiting and waking" says why.
- **Approvals.** A relay never says the maintainer reviewed unless the
  maintainer wrote so. A lane never asks for an approval, because the queue
  needs none.
- **Patch rules.** A short "Patch releases" section:
  - no public API and no flag;
  - same `content_id`;
  - a performance claim is measured in its pull request;
  - fixes for the newest line only (2.0.x and 2.1.x are closed).

## Before the lanes start (the maintainer)

1. **Branch deletion.** Turn on "Automatically delete head branches" in the
   repository settings, so merged branches go without a session. Sessions
   cannot delete them.
2. **Read the Docs.** Optionally, add `readthedocs.org` to the environment's
   network policy, so the release session can check the published versions
   itself.
3. **The queue timeout.** Consider raising it to 150 minutes. #74 passed in
   126 and was dropped at 120.

## Release checklist (the coordinator)

As 2.2.0's, plus:

1. Every fuzz target has a green run on the release pull request's head
   (P22-04).
2. No bot pull request is open and red; the lock refresh is merged or closed.
3. The documentation tests of P22-05 pass. Every changelog sentence they flag
   is confirmed by the maintainer.
4. `cargo semver-checks` and the Python API job show no addition, not only no
   break.
5. For 2.2.2: `docs/guides/performance.md` names the version it measured.

## Decisions (the maintainer)

Settled on 2026-10-21:

- **The line.** Patches are tagged from `main`, and `main` takes only what a
  patch may change until a 2.3.0 is planned. `release/2.2` is cut only if a
  feature must land first.
- **The global heap (P22-02).** HDF5 is patched, not guarded by a check
  before HDF5 reads.
- **Older lines.** 2.0.x and 2.1.x are closed, with fixes only on 2.2.x
  (P22-05 says so in the repository).
