# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# The Python package, with the extension built from crates/ (maturin, release
# profile) and the extras the test suite needs.  Re-run after changing Rust.
pip install -e ".[dev,torch,nifti,dicom,dicomseg]"

# The engine and the native CLI
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace            # unit tests, the corpus, the README doctest
# THIRD_PARTY_NOTICES is generated from Cargo.lock: after changing the lock,
# run it without --check to regenerate the file, or CI fails
python .github/scripts/third_party_notices.py --check

# Full Python suite (90% coverage floor)
pytest tests/ --cov=medh5 --cov-report=term-missing --cov-fail-under=90

# A single file or test
pytest tests/format/test_sample.py -v
pytest tests/tools/test_dataset.py::TestSplits -v
cargo test -p medh5 --lib h5::file

# Lint, format, types
ruff check . && ruff format --check . && mypy medh5

# The conformance corpus must stay green --- through both command lines
medh5 conformance run /tmp/corpus
cargo run --release -p medh5-cli -- conformance run /tmp/corpus-native

# MONAI stays out of the line above --- it is heavy, and only a handful of
# tests need it. Those tests `importorskip`, so they skip silently without it
# and CI runs them in a dedicated job.
pip install -e ".[monai]" && pytest tests/ -k "Monai or monai"

# The documentation site. `--strict` makes an unresolved cross-reference a
# build failure, which is the gate CI runs; `serve` live-reloads on :8000.
pip install -r docs/requirements.txt
mkdocs build --strict
mkdocs serve
```

## Pre-commit checks

All of these must pass before committing:

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

CONTRIBUTING.md has the same chain; keep the two equal (a test compares them).

## The model

**Format 1.2, package 2.2.** A `.medh5` file is **one subject at one or more
timepoints**, with every image, annotation, transform and curation record about
them. Not one scan — one subject. Most of the design follows from that: splitting
by file is subject-safe, a change annotation has a referent, and registration
between visits is an object in the file rather than a convention between
filenames. 1.1 adds one optional profile, `clinical` — the subject's history on
one clock, every event with when it happened *and* when it became known — and
1.2 one annotation kind, `polylines` (open curves), and nothing else; a sample
is written at the lowest version its content needs, so imaging alone is still
1.0, and removing the last polylines annotation lowers the version again.

`docs/spec/medh5-1.0.md`, `docs/spec/medh5-1.1.md` and `docs/spec/medh5-1.2.md`
are **normative**. Code implements them; when they disagree, one of them is a
bug. 1.0 Appendix C records the clauses corrected because implementing them
showed the text was not implementable — including the four the Rust
re-implementation found and the one 1.1 found; 1.1 Appendix A records the
changes to the reviewed 1.1 draft, and 1.2 Appendix A those to its planned text.
`docs/spec/task-cache-1.md` is the **separately versioned** task-and-cache
contract (`medh5.task/1`, `medh5.cache/1`): nothing in it is a payload
requirement.

## Architecture

**One format engine, three frontends.** The engine is Rust; the Python package
and the CLI are layers over it, and all three read and write the same bytes.

- **`crates/medh5`** — the engine (crates.io `medh5`). Modules map onto
  specification sections, so a spec change has one obvious home: `sample`
  (§2, §4, §14.4), `geometry` (§3), `annotations` (§6–§9), `transforms` (§10),
  `labels` (§5), `curation` (§11–§12), `integrity` (§13), `storage` (§14),
  `validate` (§15; its rules one module per domain of the code table, E0xx …
  E8xx), `collection`, `dataset`, `sampling`, `conformance`; and for 1.1,
  `version` (1.1 §2: full, projection, unsupported), `clinical` (1.1 §3–§10:
  model, columns, tables, checks, schema, selection, augmentation) and
  `companion` (task-cache-1: tasks, source pins, preflight row views, caches);
  `convert` holds the conversion report and grouping, the 0.x migration, the
  NIfTI converter and the layout of ITK's HDF5 transform files.
  `data/codes.json` is the §15.2 diagnostic code table (a test asserts the
  table and the spec agree); `data/` also holds the schema and vocabularies,
  embedded at compile time.
- **`crates/medh5-sys`** — HDF5 (static, built from source), C-Blosc2
  (vendored) and the Blosc2 (32026) and Zstandard (32015) HDF5 filters.
- **`crates/medh5-cli`** — the `medh5` command line (crates.io `medh5-cli`).
  The Python console script runs the same code through `_core.cli_main`;
  NIfTI converts in the engine (`convert::nifti`), the other converters and
  the dataloader benchmark are Python, and the standalone binary runs those
  through `Host`: `-m medh5.cli` under `MEDH5_PYTHON`, else `python3`
  (`python` on Windows).
- **`crates/medh5-python`** — the PyO3 extension `medh5._core` (not published;
  maturin builds it into the wheel). Engine handles live in a facade's
  `._handle`; result types are rebuilt as the 1.x Python classes.
- **`medh5/`** — the Python package: one module per area (`geometry`,
  `labels`, `storage`, `transforms`, `validate`, …) re-exporting `_core` and
  holding the facades, plus what is Python's — `torch`, `monai`, `io`
  converters. A class's `#[pyclass(module = …)]` must name the module that
  exports it, or `pickle` cannot find it. `medh5/_core.pyi` types the
  extension; keep it in step with the bindings
  (`tests/project/test_typing.py` runs `stubtest` against the build).
  `tests/project/python_api.txt` is the public Python API as text: a change
  to the API regenerates it (`python .github/scripts/public_api.py write`),
  and CI's `semver` and `python-api` jobs fail a minor or patch that breaks
  the newest release's API.
  A new public name also gets a docstring and a mention in its reference
  page, and `python .github/scripts/api_index.py` regenerates
  `docs/reference/api-index.md` (`tests/project/undocumented_api.txt` only
  shrinks).
- **`conformance`** — the corpus is a *shipped artifact*, not a test fixture:
  third-party implementations run it. The engine builds it
  (`crates/medh5/src/conformance/build.rs`).

The version has one source, `[workspace.package] version` in `Cargo.toml`:
every crate inherits it, maturin stamps it on the wheel, and the engine writes
it into every file's `generator`. The crates' `=X.Y.Z` pins on each other in
`[workspace.dependencies]` must follow it (a test checks); a bump sets both.

## Invariants that are easy to break

- **Coverage.** `class_ids` is what an annotation *can express*;
  `annotated_class_ids` is what was *looked for*. A class examined and absent is
  a usable negative; a class nobody examined is not. Never collapse the two.
- **Boxes sit at voxel edges**, indices at voxel centres. `[a, b]` is the slice
  `a+0.5 : b+0.5`. Every off-by-one in detection lives here.
- **Digests cover decompressed content**, so recompression changes every stored
  byte and no digest. `content_id` is a Merkle root over *stored digests*, so an
  edited dataset breaks its object digest and leaves the root matching — verify
  per object, never only the root.
- **Canonical JSON is defined to the byte** (§5.1): sorted keys, no
  whitespace, UTF-8, Python's float spelling. A NaN in an *attribute* hashes as
  `NaN`, as 1.x hashed it; a NaN in a *document* is refused (JSON has none).
  `content_id` must not depend on which version wrote a file.
- **Geometry is never invented.** Converters refuse rather than resample, guess a
  grid, or fabricate a transform. `transform_between` returns `None` when no path
  exists.
- **HDF5 handles must not cross `fork`.** The torch handle cache is PID-keyed and
  a forked child *abandons* the parent's handles (`SampleHandle.abandon`) rather
  than closing them.
- **Closing a file closes what was opened through it** (`close_everything`), as
  `h5py`'s `File.close()` does; a view used afterwards raises `MEDH5FileError`.
- **A window read is HDF5's bytes or HDF5's read.** `h5/window.rs` decompresses
  only the Blosc2 blocks a window covers, from the stored chunk; anything it
  cannot read exactly as HDF5 would (another filter, a converted type, an
  unwritten chunk, a strided selection) returns `None` and `read_region` falls
  back. Widen what it accepts only with a test against `read_hyperslab`.
- **Two times, never one.** A clinical event's *effective* time (when it
  happened) and *available* time (when it became known) are separate pairs of
  inclusive bounds; unknown stays null and strict selection never uses it. A
  revision is a new immutable event linked by `supersedes`, never an edit.
- **Inputs only through attestation.** A row reads a payload only when an
  admitted event owns it (structurally), or an eligible version asserts it (as
  `asserted_by_event_id`); crop centres come from eligible annotations only.
  Supervision, like the target, may come from after the cutoff — and never
  enters an input.
- **A later minor is a projection**: read, validated as what is known, unknowns
  `W913`, and never amended, recompressed or packed. An unknown profile refuses
  an amendment. Known profiles are re-derived from content, as in 1.0.
- **Caches pin `content_id`**: a changed source makes an entry *stale* (T403),
  told apart from a cache whose own bytes are wrong (T401/T402).
- **`amend` is copy-on-write** and replaces the file, so anything holding an open
  handle across it keeps reading the old inode. A rewrite in place closes its
  source before the rename (Windows cannot replace an open file). That is why a
  training lease takes the row's pinned `content_id` (`HandleCache.lease`): a
  stale cached handle is reopened, a changed source refused (T302).
- **Document text is read on demand.** The `text` column is read deferred
  (`read_table_deferring`): offsets only, checked against the buffer's stored
  length. Opening, selecting, preflight and the validator never decode the
  column whole --- a document's bytes are read (and checked as UTF-8) when it is
  asked for, and the validator scans in bounded slabs (`TextColumn::scan`).
  Only `records()` --- an export --- reads every document.
- **Rows index, never copy.** A preflight row's selection holds positions in its
  subject's merged history (`Selected::index`), which every row of the subject
  shares; Python receives columns (`crate::preflight`, `medh5.task.RowView`).
  Copying records per row is what made 720 rows cost 5 GB. Whatever reports a
  whole cohort takes it a subject at a time (`preflight_each`): the command
  line's `--json` (`write_preflight`), once built as one tree, took 12 GB at
  200 subjects.
- **The native inventory follows the lock.** RustSec sees crates, not the C
  libraries they compile (zlib 1.3.2 ships as `libz-sys` 1.1.29). `SECURITY.md`
  lists each statically linked library and dispositions every advisory against
  it, with a VEX statement in `vex/`; a `-sys` bump or vendored update changes
  its row (a test compares) and re-reads its advisories. `advisories.yml`
  (`cargo audit`, `pip-audit`) runs weekly and on the release tag, before
  anything is published.

## Linting & style

- **ruff** is pinned in the `dev` extra and in `.pre-commit-config.yaml` (bump
  both). **rustfmt** (`rustfmt.toml`) and **clippy** with `-D warnings` for Rust.
  CI also runs the commit hooks over every file (`pre-commit run --all-files`):
  trailing whitespace, final newlines, YAML syntax. Vendored C is excluded.
- **Rust is pinned** in `rust-toolchain.toml`, which rustup reads in a checkout
  and every CI and release job installs; bump it on purpose, in a pull request
  of its own. `rust-version` stays the crates' minimum (the `msrv` job). The
  tools that build the artifacts --- maturin, CMake, rustup-init --- are pinned
  in `.github/scripts/build-dist.sh`.

## Testing patterns

- Python tests are grouped by purpose under `tests/`: `format/` (one module
  per area of the spec, through the public API), `tools/` (collections, cohorts,
  curation, repair, sampling, the CLI), `integrations/` (torch, MONAI, the
  converters) and `project/` (docs, packaging, types). A test lives with what it
  tests, not with the release that added it. Rust tests sit beside the code
  (`#[cfg(test)]`) and in `crates/medh5/tests/`. Optional Python deps are
  guarded with `pytest.importorskip` --- per test, or at the top of a module
  whose every test needs one.
- Test names cite the clause they hold: `test_S8_1_boxes_shift_by_half_a_voxel`,
  `s14_4_the_source_is_closed_before_the_replace`.
- Fixtures are built by the **public writer**, so every reader test is also a
  writer test. Shared builders are in `tests/helpers.py`, fixtures in
  `tests/conftest.py`, and the small samples of the regression tests in
  `tests/kits.py` (one class per kit). Tests plant defects with `h5py` (a test
  dependency only).
- Engine internals with no Python door (slab budgets, temporary names, the
  rename order) are held by Rust tests; the Python test keeps the observable
  outcome.
- **A skipped test is not a passing test.** Anything guarded by
  `pytest.importorskip` needs a CI job that installs the dependency, or it
  reports coverage it does not have.

## Documentation

`docs/` is the MkDocs site: `tutorials/`, `guides/` (how-to), `reference/`,
`explanation/`, `spec/`, and `examples/` (standalone scripts). `CHANGELOG.md`
stays at the root and `hooks/mkdocs_hooks.py` pulls it in, with the fragments in
`changes/` under [Unreleased]. A pull request never edits `CHANGELOG.md`: it adds
`changes/<slug>.md` (`changes/README.md`), or one holding only `<!-- no entry:
<reason> -->`, and the release assembles them. The same hook renders
the diagnostic-code, cohort-code and schema tables from their sources (edit
`crates/medh5/data/codes.json`, the `CHECK_CODES` table in
`crates/medh5/src/dataset/check.rs`, or the schema — never the page) and emits
redirects for moved pages (`REDIRECTS`). task-cache-1 §9 writes its code table
out, so a reader of the raw text sees it; a test holds it to the `CODES` table
in `crates/medh5/src/companion/mod.rs`, so edit both. There are no separate design records:
the reasoning lives in `docs/explanation/design-rationale.md`.

The docs are tested, not proofread:

- `tests/project/test_docs_python.py` executes every `python` block against a
  real sample. A block that cannot run gets `<!-- illustrative -->` on the line
  above its fence; fewer than half may.
- `tests/project/test_docs_examples.py` checks every documented `medh5 …` flag
  against the parser, and keeps `STALE_CLAIMS` — statements once wrong in the
  docs that must not come back. It also refuses `---` as a dash in the pages,
  the READMEs, `CHANGELOG.md` and `CONTRIBUTING.md`: GitHub, PyPI and the site
  print it as three hyphens, so write `—` there.
- `TestStatedCounts` in `tests/format/test_conformance.py` compares every stated
  corpus size and breakdown with the corpus.
- The engine crate's README is its crate documentation, so its Rust example is a
  doctest; `docs/reference/rust.md` shows the same block (a test compares them).

`CONTRIBUTING.md` is the human-facing version of this file.
