# Release plans

These plans are written so that a Claude Code session, or a person, can take
one workstream and implement it without the conversation that produced it.
They are not documentation: nothing here is published, tested or normative.
The specification stays in `docs/spec/`; a plan quotes the text a workstream
is to put there.

| Release | Kind | Plan | Status |
|---|---|---|---|
| 2.1.0 | minor: features, errata, deprecations | [2.1.0/README.md](2.1.0/README.md) | released 2026-10-15 |
| 2.2.0 | minor: format 1.2, native converters, platforms, resampling | [2.2.0/README.md](2.2.0/README.md) | released 2026-10-20 |
| 2.2.x | patch releases on `main`: fixes, performance, hardening | [2.2.x/README.md](2.2.x/README.md) | planned |
| 2.3.0 | minor: threads inside a file, workers across files, digest cost | [2.3.0/README.md](2.3.0/README.md) | proposed; starts after 2.2.2 |
| 2.0.x | patch line, cut on demand | this file, [the 2.0.x line](#the-20x-line) | 2.0.3 released; closed 2026-10-21 |

## How to use a plan

1. Read this file in full: its rules hold for every workstream of every release.
2. Read the release's `README.md` (scope, schedule, dependencies) and its
   `decisions.md` (the specification and API decisions, with the text to write).
3. Read your workstream's file. It names its branch, what must land before it,
   the files it owns, its pull requests in order, and for each item the state
   at 2.0.2, the change, the API, the tests, the documentation, the changelog
   fragment and the acceptance criteria. Its first section is a prompt to start
   a session with.
4. Before you change anything, check each "State at 2.0.2" claim against
   current `main`. The claims were verified at 2.0.2, with 2.0.2's line numbers;
   code moves. Every citation also names its function, so it can be followed.
   If a claim no longer holds, adapt, and say so in the pull request.
5. A decision a plan does not settle is not yours to invent: specification
   text, a new diagnostic code, a format change or an API break. Stop and ask
   the coordinator (the maintainer, or the session they appoint to merge).
   Everything else, use your judgement and say what you chose.

## Ground rules

### Branches and pull requests

- `main` is the line of the next minor release. A patch line (`release/2.0`)
  exists only once a patch is needed (see [the 2.0.x line](#the-20x-line)).
  While no minor is planned, `main` is the 2.2.x patch line instead (see
  [Patch releases](#patch-releases)).
- A workstream works on `w21/<name>` (2.1.0), `w22/<name>` (2.2.0),
  `p22/<name>` (2.2.x) or `w23/<name>` (2.3.0), cut from the latest `main`;
  the pull request's title starts with the workstream id (`W21-02: …`,
  `P22-01: …`, `W23-01: …`). A workstream that
  lands as several pull requests pushes each from its own branch
  (`w22/<name>`, `w22/<name>-2`, …), so the next one can open while the last
  sits in the queue. A lane's brief names every branch its
  workstreams need, and the session may push all of them. A branch carries one
  pull request and is never reused for the next (in 2.2.0, one branch carried
  three and ran 41 CI runs). Where the session's branch carries a suffix the
  brief did not give, the brief records the name it has. (In 2.1.0 each
  session had one assigned branch; two lanes had to ask for a second, and a
  third held a finished workstream back until its first merged.)
- From 2.1.0's W21-01 on, a workstream lands as **one pull request**
  holding every item its file lists, in that order, at least one commit per
  item. A workstream file's "PR n" is item n of that pull request, and an edge
  naming "W21-05 PR 1" means the pull request that holds that item. The
  exceptions are named in the release's README: an *enabler*, a first item
  that other lanes wait on, lands alone, ahead of the rest of its workstream;
  and a workstream whose items another lane waits on, or too large to review
  as one, lands in the groups the README gives. W21-00 (phase 0) kept its
  eight pull requests. Every item still writes its own fragment, and an item
  that may be backported stays one commit, so it can be cherry-picked from the
  branch.
- The pull request's description lists its items, each with what it changed
  and the "Done when" it meets, so a reviewer can take it an item at a time.
- `main` takes pull requests through GitHub's merge queue, which squash-merges
  each after testing it on top of `main` and of the pull requests ahead of it,
  so the merged tree is the tested tree. It needs `ci-ok`, not an approving
  review: every 2.1.0 pull request merged without one, so a session never
  tells the maintainer a pull request "needs your approval". Nor does a relay
  say the maintainer reviewed anything the maintainer did not write about. A
  workstream never merges another workstream's branch into its own.
- **Review before the queue.** Open the pull request ready for review, not as
  a draft, once its items pass the local gate: Codex reviews a draft only when
  it is marked ready (in 2.1.0, five and eight hours late). Then:
  1. Wait for Codex's review of the head commit, and check it completed. In
     2.1.0 it failed silently on six pull requests and never ran on six more,
     so twelve merged with no automated review. If it failed, ask again with
     an `@codex review` comment.
  2. Fix or answer every finding, then arm auto-merge, which enqueues the
     pull request once CI is green. A finding names one instance of a class
     (an overflow, a path collision, a lock taken out of order): find every
     instance in the change, fix them together, and list the sweep in the
     reply. Push once per round, not once per finding: in 2.2.0, 148 findings
     came in classes, a fix of only the named instance drew the next round,
     and 69 % of pull-request CI runs were cancelled by the next push.
  3. A queued pull request cannot be pushed to, and only the maintainer can
     remove it from the queue. A finding that arrives while it is queued goes
     into a follow-up pull request at once (`W22-07 review: …`, as #63 was),
     unless the queue drops it first.
- **When the queue drops a pull request** (a conflict, or red CI on the merged
  tree), fix it in the same turn: bring `main` in (merge it, or rebase a branch
  you created), regenerate the generated files instead of merging them by hand
  (see [Generated files](#generated-files-and-stated-counts)), run the local
  gate, push and re-arm auto-merge. A pull request is not finished until it
  has merged.

### Waiting and waking

In 2.1.0 lanes sat idle for hours with a dropped pull request or a merged edge
and nothing to wake them: W21-06 (#55) from 07:32 to 15:47, W21-16 from 16:10
until a relay said W21-09 had merged. A session cannot count on a GitHub
event arriving, so:

- While it owns an open pull request, or waits on another lane's, a session
  keeps a check-in scheduled (about every 30 to 45 minutes) and on each one
  looks at the pull request's queue state, CI and conflicts, and acts.
- A session waiting on an edge subscribes to the pull request that holds it,
  so its merge wakes the session, and codes ahead meanwhile.
- The coordinator runs an hourly sweep over every open pull request and every
  lane: a dropped pull request, a red head, or a lane quiet for more than an
  hour with work open gets a nudge, and a merged edge is relayed to the lanes
  waiting on it. The sweep also covers the bots' pull requests (the weekly
  lock refresh), the last scheduled run of each fuzz target, and `main`'s last
  CI: in 2.2.0 a red lock refresh and a red fuzz run were each found days
  late, on release day.
- Lanes running at once share one account's usage limit. In 2.1.0 six lanes
  stopped together for two to three and a half hours; in 2.2.0 the weekly
  limit stopped three for four days. Run at most three lanes at once, and
  start each on the pull request other lanes wait on.
- Commit and push work before a turn ends, even with the pull request open: a
  cancelled CI run costs less than lost work. In 2.2.0 an environment reset
  during a pause lost a lane's uncommitted work, which was written again.

### The local gate

Run the chain in `CLAUDE.md` ("Pre-commit checks") before every push. CI takes
about 20 minutes to reach its first test, and a red push costs a cycle. While
iterating, run the affected tests; before pushing, run the whole chain.

The chain grows during 2.1.0: W21-00 adds `python .github/scripts/changelog.py
check`, and W21-16 adds `python .github/scripts/api_index.py --check`. The
chain in `CLAUDE.md` and in `CONTRIBUTING.md` must stay equal (a test compares
them): a pull request that adds a step adds it to both.

What the local gate does not reach, check another way:

- **Old dependency floors.** CI's `test-min-deps` job installs the floors
  (NumPy 1.24, torch 2.0, pydicom and highdicom floors, MONAI 1.3). 2.0.2's
  release pull request failed there twice (`operator.index` taking a NumPy
  `bool_` before NumPy 2.3; highdicom 0.23 writing no slice spacing). If you
  touch code that handles NumPy scalars, dtypes or a converter's DICOM, install
  the floors from `ci.yml`'s `test-min-deps` step into a second virtual
  environment and run the affected tests there.
- **Without an optional dependency.** A test that needs one must
  `pytest.importorskip` it. To check that, run the affected tests with the
  dependency blocked (a `sys.meta_path` finder that refuses `highdicom`,
  `torch` or `monai` is enough).
- **Windows-only code.** Type-check a `#[cfg(windows)]` function by copying it
  into a scratch crate and running `cargo check --target
  x86_64-pc-windows-msvc` (`rustup target add x86_64-pc-windows-msvc`); the
  medh5 crates cannot cross-build without MSVC. The `rust-platforms` and
  `test-windows` jobs are the real check.
- **Paths on Windows.** A message that embeds a path embeds its `repr` on
  Windows (`'C:\\x'`); compare with `repr(str(path))`, not an f-string.

### Rust API

The crates follow Cargo's semver rules against the newest *published* release.
From W21-00 PR 8, `cargo semver-checks` gates every pull request; until then
the rules hold anyway, by review. 2.0.2 shows why: three structs gained fields
in a patch release.

- Never add a field to an existing `pub` struct, a variant to an existing `pub`
  enum, or change a `pub fn`'s signature, bounds or return type; never put
  `#[non_exhaustive]` on an existing type (that too is a break).
- New behaviour is a new function (`thing_with(…, &ThingOptions)`, the old
  function delegating to it with defaults) or a new type.
- A new type that may grow is `#[non_exhaustive]` and is built through
  `Default` and `with_*` methods, or a `new` constructor. A new enum that may
  grow is `#[non_exhaustive]`.
- A new constant table is a slice, `pub const X: &[T]`, never an array
  `[T; N]`: an array's length is part of its type, so growing it is a break
  (2.0.2 grew two). An existing array is replaced by a function or slice and
  deprecated, never resized.
- A helper only a frontend needs is `#[doc(hidden)] pub`; a helper only the
  crate needs is `pub(crate)`.
- New error conditions use the existing `medh5::Error` variants (`invalid`,
  `coded`, `File`, `Io`, …); `Error` gains no variant before 3.0.

### Python API

From W21-00 PR 8, a CI job compares the public API with the newest published
release and fails on a break; `tests/project/python_api.txt` is the committed
snapshot.

- Add parameters only as keyword-only parameters with defaults, at the end.
- Add dataclass fields only at the end, with defaults, so positional
  construction keeps working.
- Keep `medh5/_core.pyi` in step with the bindings (`stubtest` runs in
  `tests/project/test_typing.py`), and a class's `#[pyclass(module = …)]` naming
  the module that exports it.
- A name that is deprecated stays importable (see below) and stays in the API
  snapshot as deprecated until 3.0.

### Deprecations

There was no mechanism before 2.1.0; W21-00 PR 5 adds it, and every
deprecation uses it.

- **Python.** `medh5/_deprecation.py` provides `warn(message, *, category,
  stacklevel)`. Every message starts with `medh5 3.0 ` and says what to do
  instead: `"medh5 3.0 removes dataset_kwargs: use dataset_layout"`.
  - A removed or renamed name raises `DeprecationWarning`.
  - A default whose *result* changes in 3.0 (a split's algorithm, a default
    sample id) raises `FutureWarning`, which Python shows to end users too.
    Raise it only where the answer will actually differ, so correct code never
    sees it.
  - A deprecated module attribute, a function as much as a constant, moves
    into a module-level `_DEPRECATED = {name: (value, message)}` served by the
    module's `__getattr__`, and leaves `__all__`; the API snapshot lists it as
    deprecated. Reaching the name warns, so a deprecated function needs no
    second warning in its body. A deprecated method, parameter or attribute of
    a class warns where it is used.
  - A warning raised in Rust goes through
    `crates/medh5-python/src/deprecation.rs` (`PyErr::warn` with the same
    prefix), for the classes `_core` binds directly.
  - `pyproject.toml` sets `filterwarnings = ["error:medh5 3\\.0 "]`, so a test
    that triggers a deprecation without `pytest.warns(…, match="^medh5 3.0 ")`
    fails, and medh5 never uses what it deprecates.
- **Rust.** `#[deprecated(since = "2.1.0", note = "<what to use>; removed in
  3.0")]` on items only: functions, methods, types, constants, fields. Never on
  a `pub mod` (it warns at every use of every item inside, and `clippy -D
  warnings` fails) or a `pub use` (ignored). A deprecated item still used inside
  the workspace gets `#[allow(deprecated)]` at that use, with a comment saying
  why.
- **Changelog.** Every deprecation has an entry under `### Deprecated`, and a
  line in the [3.0 ledger](#the-30-ledger) below, in the same pull request.

### Specification changes

`docs/spec/medh5-1.0.md`, `docs/spec/medh5-1.1.md` and
`docs/spec/task-cache-1.md` are normative. A plan quotes the text to write;
write it as given unless implementing it shows it wrong, and then ask.

- **An erratum** (a clause corrected because it was unimplementable, ambiguous
  or wrong) changes the text and adds a row: 1.0 Appendix C.1, whose sentence
  "Seventy-seven clauses have been corrected — …" a test counts against the
  rows (add a group "N in the development of 2.1" and keep the total right);
  1.1 Appendix A; task-cache-1 Appendix A (created by W21-11 PR 1). The row
  says what changed and why, ending "Found by …" when a review or a test found
  it.
- **A diagnostic code** is allocated in `crates/medh5/data/codes.json` and in
  the spec's code table, as the code-table revision rule (2.1.0 decision D1)
  says, and needs a conformance case (a test fails until it has one). Codes are
  never redefined, renumbered or reused.
- **A format version** changes only with content a reader must know about
  (2.1.0 adds none; 2.2.0 adds 1.2).
- **Companion contracts** (`medh5.task/1`, `medh5.cache/1`) take errata that
  add no field and change no normal form, fingerprint or default. An erratum
  may add a requirement on what an implementation opens, reads or reports (as
  2.1.0's D19 does for hostile sources), never on a manifest's or a cache's
  content: whatever was valid stays valid. Anything else is a `/2`.

### Changelog fragments

From W21-00 PR 4, each item adds `changes/<slug>.md` (the format is in
`changes/README.md`), so a workstream's pull request adds one fragment per
item; `CHANGELOG.md` is assembled from them at release. Before that pull
request lands, add entries under `[Unreleased]`; PR 4 converts them. An item
with nothing user-visible adds a fragment holding only `<!-- no entry:
<reason> -->`.

Classify each entry as 2.0.2 did:

- **Security**: closes a way hostile input or a shared file harms the user.
- **Fixed**: a crash, a refusal of valid input, a silently wrong answer
  corrected, or data loss prevented (a data-loss fix made by a new refusal
  stays here).
- **Behaviour changes**: anything that changes what existing code produces or
  accepts, stated so a reader can tell whether their pipeline is affected. An
  item that is both a fix and a change goes here only.
- **Added**, **Deprecated**, **Performance**, **Documentation** as usual;
  **Removed** only in a major release.

Each entry opens with a bold sentence that states the change as a fact, then
one to three sentences. No `---` (write `—`), lines of at most 80 columns.

An entry that promises something about a later release ("will require",
"the last line to support", "removed in") states a decision the maintainer
settled, and names it; a plan's own proposal is not enough. 2.1.0's D36 put
"medh5 2.2.0 will require Python 3.11" into the release pull request's
changelog, README and install page before anyone had decided it, and it was
withdrawn an hour before the tag. The release checklist lists every such
sentence for the maintainer to confirm.

### Tests and documentation

`CLAUDE.md` ("Testing patterns", "Documentation") holds. In addition:

- Every new behaviour has a test that fails without the change. Run it red
  first, and say in the pull request that you did.
- A test that holds a clause names it (`test_S13_1_…`, `s14_4_…`); a clause of
  a document whose section numbers would read as 1.0's names the document
  (`test_v1_2_S3_…` for 1.2 §3). A test of behaviour no clause states (an
  integration, a command's output, an error class) has a plain name, as the
  existing ones do.
- Never skip, disable or loosen a test to get green. A test pinning behaviour
  a plan changes is rewritten to pin the new behaviour, and the pull request
  says which.
- A claim the documentation used to make wrongly goes into `STALE_CLAIMS`
  (`tests/project/test_docs_examples.py`) when it is corrected.
- A documented `python` block runs; `<!-- illustrative -->` only where it
  cannot, and fewer than half of all blocks may be.

### Generated files and stated counts

Never merge these by hand. After a rebase, regenerate them and commit the
result:

| File | Regenerate with |
|---|---|
| `THIRD_PARTY_NOTICES` | `python .github/scripts/third_party_notices.py` |
| `tests/project/python_api.txt` (from W21-00 PR 8) | `python .github/scripts/public_api.py write` |
| `docs/reference/api-index.md` (from W21-16 PR 3) | `python .github/scripts/api_index.py` |
| corpus counts stated in the docs | read the failure of `TestStatedCounts` and correct each sentence |
| 1.0 Appendix C.1's count sentence | count the rows |

### Shared files

Most conflicts come from a few files. Each has an owner; another workstream
touches it only for the smallest change its item needs (a call site, a
registration), and says so in the pull request. The table below is 2.1.0's;
a release's README gives its own.

Four files are touched by nearly every pull request that adds an API or a
command, and in 2.1.0 they caused most of the merge queue's drops:
`tests/project/python_api.txt` and `medh5/_core.pyi` (14 of 24 pull
requests each), `docs/reference/cli.md` (14) and `crates/medh5-cli/src/app.rs`
(13). They have no owner. After bringing `main` in, regenerate the snapshot,
re-merge the other three by hand keeping both sides' entries in order, and run
the tests that compare them (`test_typing.py`, `test_docs_examples.py`)
before pushing.

| File | Owner (2.1.0) | Others touch it for |
|---|---|---|
| `crates/medh5/src/h5/file.rs` | W21-05 files | W21-00 PR 6 (Windows mark), W21-03 PR 1 (create time tracking), W21-04 PR 1 (snapshot at commit), W21-11 PR 1 (FIFO refusal) |
| `crates/medh5/src/sample/writer.rs`, `writer_*.rs` | W21-04 writer | W21-02 PR 3 (a crate-private `digest_algo` for a corpus case), W21-02 PR 4 (the pyramid refusal, D11), W21-03 PR 1 (create time tracking), W21-13 PR 8 (`labelmap=`), W21-10 PR 3 (de-identified derivation) |
| `crates/medh5/src/dataset/split.rs` | W21-08 cohorts | W21-04 PR 1 (the claim writer's pre-check calls the E702 check) |
| `crates/medh5-cli/src/app.rs`, `medh5/cli.py` | each command's owner | registering a command or flag |
| `crates/medh5/data/codes.json`, the §15.2 table | W21-02 integrity | W21-13 PR 2 (W919, E404's case) |
| `crates/medh5/src/conformance/**` | W21-03 conformance | each new code's case; W21-11 PR 1 (task fixture) |
| `crates/medh5/src/conformance/robustness.rs` | W21-01 platform | — (W21-03 does not touch it) |
| `medh5/errors.py`, `medh5/validate.py` | W21-06 python | W21-02 PR 1 (`Code`'s fields, `codes_revision`) |
| `medh5/sampling.py` | W21-06 python | W21-14 PR 3 (adds the aggregator, a new class) |
| `crates/medh5-python/src/convert.rs` (values across the boundary) | W21-06 python | W21-13 PR 8 (the mask copy), W21-15 PR 1 (`typed()` without the GIL) |
| `docs/spec/*.md` | the workstream whose decision it is | — |
| `medh5/_core.pyi` | each binding's owner | — |

### Things never to do

- Skip, disable or quarantine a test; push an empty commit to re-run CI;
  rewrite history on a branch you did not create.
- Set `HDF5_USE_FILE_LOCKING` or turn HDF5's locking off implicitly (1.0 §14.4).
- Invent geometry, a unit, a code, a clock conversion or a date (the format's
  rule, and every converter's).
- Write a file in place: every write goes through the engine's replace
  routine (`AtomicFile`, `write_atomic`, and from W21-05 `stage_file`).
- Put a model identifier, a session transcript or a scratch path into a commit,
  a pull request or the repository.

## Patch releases

A patch fixes a wrong answer, a crash, a hang, a cost out of proportion to the
file, the documentation or CI. It adds no public API, in Rust or Python, and
no command-line flag; the format and the diagnostic code table stay as they
are, and a new bound reports through an existing code. A performance change
writes the same `content_id` for the same input, and its pull request records
the measurement it claims. Only the newest line receives fixes: the
maintainer closed 2.0.x and 2.1.x on 2026-10-21, and an older line reopens
only by the maintainer's decision.

While no minor is planned, `main` is the patch line and each patch is tagged
from it ([2.2.x](2.2.x/README.md)), and the repository variable
`API_RELEASE_TYPE` is `patch`, so the API jobs check `main` as a patch: by
its version alone, `main` reads as the next minor's work. If a feature must
land first, a `release/X.Y` branch is cut from the last patch tag and the
series moves there, as [the 2.0.x line](#the-20x-line) does.

Fuzzing is a gate, not a report. A red scheduled run of any target
(`fuzz.yml`) is a finding the coordinator's sweep raises the hour it is seen,
and a patch is not tagged until every target has a green `workflow_dispatch`
run on the release pull request's head: the release checklist names that run.
In 2.2.0 a red run had become normal, and a real overflow was found on release
day only because the release session looked.

## The 2.0.x line

Closed on 2026-10-21, with 2.1.x: fixes land on 2.2.x only. What follows is
the record of how 2.0.3 was cut, and the pattern a `release/2.2` would
follow.

`release/2.0` is created from the `v2.0.2` tag only when a 2.0.3 is wanted.
A fix lands on `main` first; a cherry-pick pull request to `release/2.0`
carries the same fragment file. 2.0.3 is cut there (`changelog.py build 2.0.3`,
tag `v2.0.3`); the release workflow marks it not-latest once 2.1.0 exists, and
a pull request to `main` then copies the `[2.0.3]` section into `CHANGELOG.md`
and deletes the fragments that shipped.

Candidates for a 2.0.3, each marked "backport" in its workstream. A backport
corrects a wrong answer or closes a hazard; a change to how a correct answer
is spelled (the MONAI metadata's `"none"` for `None`, D30), or anything a
deprecation announces, waits for 2.1.0.

| Item | Workstream |
|---|---|
| A Windows child process no longer inherits HDF5's handle (amends fail, files stay locked) | W21-00 PR 6 |
| A hostile task manifest reads and reports nothing its pins do not name; a FIFO source no longer hangs | W21-11 PR 1 |
| A procedure that asserts its own link attests its mask (1.1 §7.3) | W21-11 PR 1 |
| The `pydicom` floor is 3.0.1, which highdicom requires | W21-00 PR 2 |
| `cargo install medh5-cli` documents the release profile | W21-01 |

## The 3.0 ledger

What 2.x deprecates and 3.0 does. A 2.x workstream never does these early; it
adds its line here when it adds the deprecation.

| 3.0 change | Deprecated in | Workstream |
|---|---|---|
| `medh5::Error` becomes `#[non_exhaustive]` and gains `Os { kind, path, message }`; `MEDH5IOError` derives from `MEDH5FileError` | 2.1.0 | W21-06 |
| `SampleWriter::add_mask` returns its `Group` in Rust (Python does from 2.1.0) | 2.1.0 | W21-04 |
| The default `sample_id` and pack key are the file stem; `subject_id` defaults to it only when the stem has no dot | 2.1.0 | W21-04 |
| `storage.dataset_kwargs` is removed (use `dataset_layout`) | 2.1.0 | W21-06 |
| `annotations::header::{ANNOTATION_KINDS, GEOMETRIC_KINDS}` and `version::FORMAT_VERSIONS` arrays are removed (use the functions) | 2.1.0 | W21-02 |
| `Tracking::{state_at, states, is_new, is_resolved}` (class-level) are removed (use `locate_tracks`) | 2.1.0 | W21-09 |
| `ClassStats.prevalence` is `float \| None` (`Option<f64>`) | 2.1.0 | W21-08 |
| `make_splits` defaults to `medh5.split/2` | 2.1.0 | W21-08 |
| A legacy cohort manifest (no `schema`) whose two readings differ is refused unless `base=` is given | 2.1.0 | W21-08 |
| A scrub withholds the date shift by default; UID pseudonyms need a salt | 2.1.0 | W21-10 |
| `scrub_paths` (Rust) is removed (use `scrub_each`); `Deidentification.extra` is removed | 2.1.0 | W21-10, W21-06 |
| `curation::scrub::STRICT_RULES` is removed (use `strict_rules()`) | 2.1.0 | W21-10 |
| A loaded task manifest's sources are confined to its directory by default | 2.1.0 | W21-11 |
| `Target(negative=…)` without `min_follow_up_us` is refused by the builders | 2.1.0 | W21-11 |
| `medh5.task.STATUSES` and `LABELS` are removed (use `ROW_STATUSES`, `TARGET_STATUSES`) | 2.1.0 | W21-11 |
| The engine functions with no caller deprecated by W21-06 are removed | 2.1.0 | W21-06 |
| `h5`, `json`, `pyval` move under `medh5::__private`; `numeric` becomes `pub(crate)`; whether to wrap `hdf5::Group`/`Dataset` is decided | — | 3.0 design |

Not in the ledger, because a minor does it: dropping an interpreter. One is
dropped in a minor, after one minor's notice, once a dependency medh5 needs no
longer supports it (2.2.0 E5); that is the only removal a 2.x minor makes.
Python 3.10 stays supported in 2.2.0: 2.1.0 had planned to drop it there
(D36), and the maintainer kept it before 2.1.0 was tagged.
