# Contributing to medh5

medh5 is two things kept in step: a **format** — the normative
specifications ([1.0](docs/spec/medh5-1.0.md), and [1.1](docs/spec/medh5-1.1.md)
for the clinical profile), their JSON Schemas and their conformance corpus, with
the separately versioned [task and cache contract](docs/spec/task-cache-1.md)
beside them — and an **implementation**: one format engine in Rust, with three
frontends over it (the Rust crate, the Python package and the `medh5` command
line). Most of what follows is about keeping the format, the engine, its
frontends and the documentation that describes them from drifting apart.

## Setting up

You need Rust, a C compiler and CMake — the engine builds HDF5 and C-Blosc2
from source and links them statically — and Python 3.10 or later. With rustup,
a build in the checkout uses the Rust that `rust-toolchain.toml` pins, and
installs it the first time: the version CI and the release build with, bumped
on purpose in a pull request of its own. The crates themselves build with any
Rust from `rust-version` in `Cargo.toml` on, which CI checks too.

```bash
git clone https://github.com/medh5-project/medh5.git
cd medh5
# The Python package, with the engine compiled into it by maturin (release
# profile). Re-run after changing Rust.
pip install -e ".[dev,torch,nifti,dicom,dicomseg]"
pip install -r docs/requirements.txt        # to build the documentation site
pip install pre-commit && pre-commit install   # optional: the hooks CI runs, on every commit
```

MONAI is left out of that line on purpose: it is heavy and only a handful of
tests need it. They skip without it, and CI runs them in a job of their own:

```bash
pip install -e ".[monai]"
pytest tests/ -k "Monai or monai"
```

## The checks

Every change has to pass all of these. CI runs each of them, and more (below).

```bash
cargo fmt --all -- --check && cargo clippy --workspace --all-targets -- -D warnings \
  && cargo test --workspace \
  && python .github/scripts/third_party_notices.py --check \
  && python .github/scripts/changelog.py check \
  && ruff check . && ruff format --check . && mypy medh5 \
  && python .github/scripts/api_index.py --check \
  && pytest tests/ --cov=medh5 --cov-fail-under=90 \
  && medh5 conformance run /tmp/corpus \
  && cargo run --release -p medh5-cli -- conformance run /tmp/corpus-native \
  && mkdocs build --strict
```

- **Rust** — `rustfmt` (`rustfmt.toml`), `clippy` with `-D warnings`, and the
  engine's tests: unit tests beside the code, the conformance corpus in
  `crates/medh5/tests/`, and the crate README's example as a doctest.
- **Notices** — `THIRD_PARTY_NOTICES` is generated from `Cargo.lock`, and
  `--check` fails while it is stale (see [below](#changelog-and-releases)).
- **Lint and format** — `ruff`, pinned in the `dev` extra and in
  `.pre-commit-config.yaml`; bump the two together. CI also runs the commit
  hooks over every file (`pre-commit run --all-files`), which add trailing
  whitespace, final newlines and YAML syntax to what `ruff` checks.
- **Types** — `mypy --strict` over `medh5/`. The engine's Python types are
  `medh5/_core.pyi`; `tests/project/test_typing.py` runs `stubtest` against the
  built module, so a binding change without its stub fails.
- **Tests** — `tests/`, with a 90 % coverage floor: `format/`, `tools/`,
  `integrations/` and `project/`, each module named for what it tests.
- **Conformance** — the corpus is a shipped artifact that third-party
  implementations run, not a test fixture. It must stay green through both
  command lines.
- **Documentation** — `mkdocs build --strict` fails on any broken link or
  anchor, including links into the specification by clause.

CI also builds the wheel and the native binary for every platform the project
ships — Linux x86_64 and aarch64 (manylinux_2_28 and musllinux_1_2), macOS
arm64, Windows x64 — and runs the conformance corpus through each, and each
binary's hand-over of the converters to Python. It runs the suite against every
wheel: on Python 3.10–3.14 on Linux x86_64, on macOS arm64 (for the `spawn`
start method), on Windows (for the atomic-replace paths, `medh5.torch`'s handle
cache among them), on Linux aarch64, on musl (x86_64 and aarch64), and at every
floor `pyproject.toml` declares (NumPy's, each extra's, and the test suite's
HDF5 tools'); and on the next Python (3.15) for information, not as a gate,
until what the suite needs publishes for it. It reads every valid corpus case,
and its samples under each Blosc2 profile, with HDF5 1.10, 1.12 and 1.14, since
the specification promises files HDF5 1.10 reads. It runs the engine's Rust
tests on Windows and macOS as well as Linux; checks the minimum Rust version;
packages the crates as crates.io will receive them; builds the sdist from
source; and runs the reference writer in `docs/examples/`, then validates what
it writes with the built wheel and binary, to keep the specification's Appendix
C.2 honest.

## Writing tests

- Name a test after the clause it holds: `test_S8_1_boxes_shift_by_half_a_voxel`,
  `s14_4_the_source_is_closed_before_the_replace`.
- Put a test with what it tests, not with the release that fixed it: the module
  for that part of the spec, tool or integration. A regression test is named,
  like any other, for the clause it holds; the changelog says which release
  fixed it.
- Build fixtures with the **public writer**, so every reader test is also a
  writer test: shared builders are in `tests/helpers.py`, fixtures in
  `tests/conftest.py`, and the small samples the regression tests use in
  `tests/kits.py`. Plant defects with `h5py`, a test dependency only.
- Engine internals with no Python door are held by Rust tests beside the code;
  the Python test keeps the observable outcome.
- Guard an optional dependency with `pytest.importorskip` — and make sure some
  CI job installs it. A skipped test is not a passing test.
- A bug fix comes with the test that would have caught it.

## Changing the format

`docs/spec/medh5-1.0.md` and `docs/spec/medh5-1.1.md` are **normative**: the
code implements them, and when the two disagree one of them is a bug. A new
code is allocated in `crates/medh5/data/codes.json` *and* the spec's code table
of the version that adds it (1.0 §15.2, 1.1 §11.2); a test compares the two. Decide which before changing either. Engine
modules map onto specification sections (`crates/medh5/README.md` lists them),
so a clause has one obvious home in the code.

- **A clause that turned out to be unimplementable**, ambiguous, or more than
  the implementation can honestly promise is corrected in the text *and*
  recorded as a row in Appendix C.1. A test counts the rows against the sentence
  that states how many there are.
- **Diagnostic codes are stable API.** They live in one table,
  `crates/medh5/data/codes.json`, which the validator, every frontend and the
  documentation read; a test asserts that it and the §15.2 table list the same
  codes. A code's meaning never changes, and a retired code is never reused. A
  new code needs a conformance case, in `crates/medh5/src/conformance/build.rs`
  — or `clinical.rs` beside it, for a 1.1 code; a test fails until it has one.
- **Versioning follows §16**: a minor format version may add objects, kinds,
  profiles and codes; it may not change what an existing one means.
- The JSON Schema lives in one place, `crates/medh5/data/`, beside the code
  table: the engine embeds it, §2.4 names it, the documentation site publishes
  it, and `medh5 conformance publish` writes it into the suite.

## Writing documentation

The site is [MkDocs](https://www.mkdocs.org/) with the Material theme, laid out
as **tutorials** (learning), **how-to guides** (tasks), **reference** (what
everything is), **explanation** (why) and the **specification**. Put a page
where its reader will look for it, and link rather than repeat.

Documentation here is checked against the code, not proofread:

- **Every `python` block runs.** `tests/project/test_docs_python.py` executes
  each one against a real sample, with the names the pages use by convention
  (`s`, `w`, `ann`, `paths`, …) already bound. A block that genuinely cannot
  run — it needs a DICOM series, or it elides arguments with `...` — carries an
  `<!-- illustrative -->` comment on the line above its fence, where readers
  see it too. Most blocks must stay runnable.
- **Every documented CLI flag exists.** `tests/project/test_docs_examples.py`
  parses each `medh5 …` line in a fenced block against the command line's
  grammar.
- **Corrected claims stay corrected.** The same file keeps a list of statements
  that were once wrong in the documentation; add to it when you fix a claim
  that could plausibly come back.
- **A dash is `—`.** GitHub, PyPI, crates.io and the site all print `---` as
  three hyphens, so the pages, the READMEs, `CHANGELOG.md` and this file write
  the character; the same file checks each of them.
- **Counts are checked.** Where a page states the size of the conformance
  corpus or its breakdown, a test compares it with the corpus.
- **Measurements say how they were taken.** A number on a page comes from a
  stated run on a stated machine; re-measure rather than carry one forward, and
  read a new window for every patch — the same window twice is HDF5's chunk
  cache, not a read.
- **Some tables are generated.** The diagnostic codes, the cohort check codes
  and the sample-document schema are rendered at build time from
  `crates/medh5/data/codes.json`, the `CHECK_CODES` table in
  `crates/medh5/src/dataset/check.rs` and the JSON Schema by
  `hooks/mkdocs_hooks.py`. Edit the source, not the page; the page carries only
  a marker comment. The task-and-cache finding codes are written out in
  task-cache-1 §9, so the raw text shows them, and a test holds that table to
  the `CODES` table in `crates/medh5/src/companion/mod.rs`: edit both.
- **The Rust example is tested twice.** The engine crate's README is its crate
  documentation, so its example is a doctest; `docs/reference/rust.md` shows
  the same block, and a test compares the two.
- **Moving a page breaks links people already have.** Add the old path to
  `REDIRECTS` in `hooks/mkdocs_hooks.py`.
- `CHANGELOG.md` lives at the repository root and is pulled into the site by
  the same hook; link into the docs from it as `docs/…`, which is rewritten for
  the site.

Preview with `mkdocs serve`, which live-reloads on port 8000.

## Deprecating something

Nothing is removed or renamed within a major version: it is deprecated, keeps
working, and goes in 3.0.

- **Python.** Warn with `medh5._deprecation.warn(message)`. Every message
  starts with `medh5 3.0 ` and says what to use instead: `"medh5 3.0 removes
  dataset_kwargs: use dataset_layout"`. A removed or renamed name raises
  `DeprecationWarning`; a default whose *result* changes in 3.0 raises
  `FutureWarning` (`category=FutureWarning`), only where the answer will
  actually differ, so correct code never sees it.
- **A module attribute**, a function as much as a constant, leaves `__all__`
  and moves into the module's `_DEPRECATED = {name: (value, message)}` table,
  served by `__getattr__ = module_getattr(__name__, _DEPRECATED)`. Reaching
  the name warns, so a deprecated function needs no second warning in its
  body. A deprecated method, parameter or attribute warns where it is used.
- **The bindings.** A class or function `_core` binds directly warns through
  `crates/medh5-python/src/deprecation.rs`, with the same prefix.
- **Rust.** `#[deprecated(since = "2.1.0", note = "<what to use>; removed in
  3.0")]` on items only: functions, methods, types, constants, fields. Never on
  a `pub mod` (every use of every item inside warns, and `clippy -D warnings`
  fails) or a `pub use` (ignored). A deprecated item still used inside the
  workspace gets `#[allow(deprecated)]` at that use, with a comment saying why.
- **Tests.** The suite turns any warning starting `medh5 3.0 ` into an error
  (`filterwarnings` in `pyproject.toml`), so medh5 never uses what it
  deprecates, and a test that means to trigger one says so with
  `pytest.warns(DeprecationWarning, match="^medh5 3.0 ")`.
  `tests/project/test_deprecations.py` checks every message and every
  `#[deprecated]`.
- **Changelog.** Every deprecation has an entry under `### Deprecated`.
- **The API gates.** CI compares the public API with the newest release at or
  below the tree's version, and fails a minor or patch that breaks it:
  `cargo semver-checks` the crates (the `semver` job), and
  `.github/scripts/public_api.py` the Python package and, with `--cli`, the
  command line's commands and options (`python-api`). A
  deprecated Python name prints as `deprecated` there, which is not a break;
  its removal is, until 3.0. `tests/project/python_api.txt` is the Python
  API as text: a change to it regenerates the file
  (`python .github/scripts/public_api.py write`), so the diff shows reviewers
  what changed.
- **The API index.** `docs/reference/api-index.md` lists every public name
  with its docstring's first sentence and the section of the docs that covers
  it; `python .github/scripts/api_index.py` regenerates it, and `--check`
  fails while it is stale. A new public name gets a docstring and a mention in
  the reference page that covers it, in code; it is not added to
  `tests/project/undocumented_api.txt`, which lists the names still waiting
  for one and may only shrink.

## Changelog and releases

Each pull request records its change in a fragment, `changes/<slug>.md`, and
leaves `CHANGELOG.md` alone; [`changes/README.md`](changes/README.md) gives
the format, and CI's `lint` job fails a pull request that adds none. A pull
request with nothing a user would notice adds a fragment holding only
`<!-- no entry: <reason> -->`. Each entry opens with a bold sentence stating the
change as a fact, then one to three sentences, and goes in one section
([Keep a Changelog](https://keepachangelog.com/)):

- **Security**: closes a way hostile input or a shared file harms the user.
- **Fixed**: a crash, a refusal of valid input, a silently wrong answer
  corrected, or data loss prevented (a data-loss fix made by a new refusal
  stays here).
- **Behaviour changes**: anything that can alter what existing code produces
  or accepts, stated so a reader can decide whether their pipeline is
  affected. An item that is both a fix and a change goes here only.
- **Added**, **Deprecated**, **Performance** and **Documentation** as usual;
  **Removed** only in a major release.

`python .github/scripts/changelog.py preview` prints what the next release's
section will hold, and the documentation site shows it under [Unreleased].
An entry states what its own release does. `changelog.py check` lists every
line of a fragment that promises something about a later one ("will", "next",
"a later minor", "removed in", "until 3.0", "planned for", "deprecated", "the
last line to support"), and the release waits until the maintainer has
confirmed each.

Only the newest line receives fixes (`SECURITY.md`, "Supported versions").
While no minor release is planned, `main` is that line and each patch is
tagged from it: a patch adds no public API and no command-line flag, and the
repository variable `API_RELEASE_TYPE=patch` has the API gates fail an
addition as well as a break.

The version lives in `[workspace.package] version` in `Cargo.toml`: every
crate inherits it, maturin stamps it on the wheel, and the engine writes it into
every file's `generator`. The crates also name each other at exactly that
version — the `=X.Y.Z` pins of `medh5-sys`, `medh5` and `medh5-cli` under
`[workspace.dependencies]`, which crates.io needs because it publishes each
crate on its own — and a test holds the pins to the version. To release, set
the version and the three pins (and `Cargo.lock` with `cargo update -w`), write
the release's bold summary in `changes/lead.md`, assemble the fragments with
`python .github/scripts/changelog.py build X.Y.Z --date YYYY-MM-DD` (which
deletes them), and push a `vX.Y.Z` tag on a commit whose CI passed.
`.github/workflows/release.yml` then:

1. runs the full CI on the tagged commit, which builds every wheel, the sdist
   and every binary, and the advisory checks (`advisories.yml`: `cargo audit`
   over `Cargo.lock`, `pip-audit` over the package's requirement and over the
   extras), which fail on an advisory against a crate or the requirement that
   `SECURITY.md` has not dispositioned; an advisory against what an extra
   installs is reported in the run's summary, to disposition in `SECURITY.md`,
   and does not stop the release;
2. checks the tag against the workspace version, every platform's wheel and
   binary against the tag (each carrying `LICENSE` and `THIRD_PARTY_NOTICES`),
   and extracts the version's section of `CHANGELOG.md` — a tag with no
   section, a CHANGELOG with entries still under `[Unreleased]`, or a fragment
   left in `changes/`, fails here, before anything is published;
3. publishes the wheels and the sdist to PyPI through Trusted Publishing, and
   the crates to crates.io through trusted publishing too: every crate is
   first built from its package, with no token in existence, and a failure
   there publishes nothing; then the job exchanges its identity for a
   short-lived token and uploads `medh5-sys`, `medh5` and `medh5-cli`, in
   that order, each in the index before the next one, which resolves against
   it. Beside these, every archive, wheel and sdist is attested (`gh
   attestation verify`);
4. creates the GitHub Release, with the changelog section as its notes (each
   paragraph on one line, since a release page breaks a line wherever its
   source does) and the distributions, the binaries and their `SHA256SUMS`
   attached. A backport, a version below the newest final release already
   tagged (a 2.0.x patch after 2.1.0), is not marked "Latest";
5. pushes the Homebrew formula for the binaries to the tap
   (`<owner>/homebrew-medh5`, or the `HOMEBREW_TAP` variable) with the
   `HOMEBREW_TAP_TOKEN` secret. Pre-releases and backports skip this step, so
   the tap stays on the newest release;
6. checks that PyPI, crates.io and the release page have the version, and the
   tap too when it took it.

Unlike what the registries hold, a release's notes can be corrected after it:
fix the version's section, then `python .github/scripts/release_notes.py vX.Y.Z`
writes them again and `gh release edit vX.Y.Z --notes-file release-notes.md`
puts them on the release page.

In this repository a crates.io exchange that issues no token, or a missing
`HOMEBREW_TAP_TOKEN`, fails the step that needs it; in a fork the step warns
and does nothing, so a fork can release to PyPI alone. The format version
(`medh5.FORMAT_VERSION`) is separate and changes only with the specification.

`THIRD_PARTY_NOTICES` holds the licences of everything the wheel and the binary
compile in, generated from `Cargo.lock`. After changing `Cargo.lock`, run
`python .github/scripts/third_party_notices.py`; CI fails while it is stale.
Every Monday `lock-refresh.yml` refreshes the lock, regenerates the notices
and opens a pull request on `deps/cargo-lock`; it holds back the crates of the
native inventory below and lists their bumps in its body, for a person to make
with the inventory's row. If the repository has neither the `LOCK_REFRESH_APP_ID`
app nor the `LOCK_REFRESH_TOKEN` token, that pull request runs no CI until it is
closed and reopened.
A library under a licence the others do not already name goes into
`pyproject.toml`'s `license` expression too, and the Homebrew formula's
(`.github/scripts/homebrew_formula.py`); neither can be corrected for a
version once it is published.

`SECURITY.md` inventories the C libraries the wheel and the binary link
statically, which no advisory database maps to a crate, and keeps a
disposition for each advisory against them. A `-sys` crate bump or a vendored
update changes a row (a test compares the inventory with `Cargo.lock` and the
vendored sources); read every advisory against the new version. A finding of
`advisories.yml` that does not apply is passed over there by its id, with the
reason in `SECURITY.md`.

### Before the first release to a registry

What a release needs from a registry is set up once, outside the repository,
and a gap in it shows only when the tag runs — by which time PyPI may have the
version. Before the first tag that publishes there:

- **crates.io**: each crate (`medh5-sys`, `medh5`, `medh5-cli`) has a trusted
  publisher naming this repository, `release.yml` and the `crates-io`
  environment. Trusted publishing can only update a crate that exists, so a
  new crate's first version is published by hand by an owner.
- **PyPI**: the project's Trusted Publisher names this repository,
  `release.yml` and the `pypi` environment, as it has since 1.x; renaming
  either here means renaming it there.
- **Homebrew**: the tap (`medh5-project/homebrew-medh5`) exists, and
  `HOMEBREW_TAP_TOKEN` can push to it — a fine-grained token with read and
  write access to that repository's contents.

After it, add a second owner to each crate (`cargo owner --add`), so that
publishing does not hang on one account.

### If a release fails partway

Nothing published can be taken back, and nothing has to be: every publishing
step skips what is already published *and identical* to the run's build,
publishes what is missing, and stops on anything published from something
else (`.github/scripts/release.py`).

1. Read the failed job's log, and fix the cause outside the repository if it is
   there: a secret, a token's scope, the tap's permissions, a registry outage.
2. **Re-run the failed jobs** of the same workflow run (not "Re-run all jobs",
   and not a new tag). Re-running reuses that run's artifacts, so what is
   published next is byte-for-byte what was published first.
   - PyPI skips the files it has, then checks that each one has this run's
     digest.
   - crates.io skips a crate whose version exists only if its contents equal
     `cargo package` of the tagged commit, then publishes the rest in order.
     A yanked version stops the job, since crates.io never takes a version
     twice: undo the yank (`cargo yank --undo`) if it holds this release, or
     bump the version.
   - The GitHub Release is created if missing; otherwise the missing files are
     attached, and a file of the same name with other content stops the job.
   - The tap is committed to only if its formula changed.
3. The `verify` job names anything still missing. It runs however the jobs
   before it ended, so a partial release is never reported as green.

If a fix needs a code change, the version cannot be reused on any registry that
already has it: bump the version, add a CHANGELOG section, and tag again. A
version that reached PyPI but not crates.io (or the reverse) is then a gap to
note in the new version's notes.

GitHub Actions are pinned by commit SHA; Dependabot proposes the updates as pull
requests.
