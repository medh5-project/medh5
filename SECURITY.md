# Security

## Reporting a vulnerability

Report a suspected vulnerability privately, through the repository's
[private vulnerability reporting](https://github.com/medh5-project/medh5/security/advisories/new)
(**Security**, then **Report a vulnerability**), not in a public issue. If that
form is unavailable, open an issue that asks for a private contact and leaves
out the details.

medh5 reads files that other people wrote, so a file that makes it crash, hang,
exhaust memory or run code is a vulnerability, as is anything that leaks what
`medh5 scrub` was asked to remove.

## Supported versions

Only the newest 2.2.x release receives fixes, security fixes among them. The
2.0.x and 2.1.x lines are closed (2026-10-21): nothing is backported to them,
and a finding against them is fixed by upgrading to the newest 2.2.x. The 2.2.x
patches are tagged from `main` while no minor release is planned, or from
`release/2.2` once a minor's work needs `main`.

## What is checked, and when

`.github/workflows/advisories.yml` runs every week, on a pull request that
changes `Cargo.lock` or `pyproject.toml`, and on every release tag before
anything is published:

- **The crates** every wheel and binary compiles in, the ones `Cargo.lock`
  names, against the [RustSec](https://rustsec.org/) database, with
  `cargo audit --deny warnings`. An unmaintained, unsound or yanked crate fails
  as a vulnerable one does.
- **The Python packages** that `pip install medh5` installs (NumPy) and that
  each extra installs, as they resolve today, against PyPI's advisories, with
  `pip-audit`.

Neither can see the C libraries the crates build from source and link
statically: a C library is not a crate, and an advisory names the library's
version, not the crate's. They are inventoried below, with a disposition for
every advisory against a version the project ships.

A finding that does not apply is passed over in the workflow by its id, and
recorded under [Advisories](#advisories) with the reason. A release ships no
finding against a crate or against what `pip install medh5` installs that has
neither a fix nor a disposition here.

An extra installs projects medh5 neither ships nor pins (PyTorch, MONAI,
pydicom, highdicom, PyArrow, h5py, hdf5plugin), and an advisory against one may
have no fix for weeks. Such a finding is reported, in the weekly run and in the
release run's summary, and dispositioned here, but it does not hold a medh5
release: the release would not change what that extra installs.

## Opening a file you did not write

A `.medh5` file is HDF5, and HDF5 is a rich format: a file can name filters,
nest groups, link to other names and state sizes that its bytes do not hold.
What `medh5.open`, `medh5 validate` and the readers built on them guarantee for
a file from someone else:

- **No code is loaded from the file's say-so.** A dataset names its filters by
  number, and HDF5 would load a filter it has not registered from the shared
  libraries in its plugin search path. medh5 turns HDF5's plugins off: Blosc2
  and Zstandard are compiled in, and a dataset whose filter is neither those nor
  one of HDF5's own fails to read, with HDF5's "filter plugins disabled". Only
  `HDF5_PLUGIN_PATH` (load from those directories) or `MEDH5_ALLOW_PLUGINS`
  (set, and not `0`: load from HDF5's standard directory) turns them back on,
  each read once per process ([Install](docs/install.md)).
- **Bounded depth.** Opening walks the file a group at a time, and a file nested
  more than 64 groups deep is refused (`E001`), not followed.
- **Bounded decompression.** A chunk decompresses to at most twice the size of
  its dataset's chunk plus 64 KiB; a Zstandard frame or Blosc2 header that
  states more is refused before anything is allocated, and a Blosc2 header that
  does not parse is refused. A feature cache's entry whose stored dtype or shape
  differs from the cache's declaration is `T402`, with nothing read.
- **A refusal, not a crash.** A malformed file is reported: an error from
  `open`, or diagnostic codes from `validate`. The structural fixtures of
  `damaged.py --structural` (deep nesting, an unregistered filter, chunks that
  overstate their size) are checked against every release binary.
- **A rewrite does not follow a link it should not.** `amend`, `scrub --apply`,
  `recompress` and `fix` replace the file a symbolic link names and keep the
  link, but refuse a link in a sticky, world-writable directory that another
  user owns.

Not guaranteed: that a file within these bounds is cheap. The bounds hold per
chunk and per group, not per file. A file can honestly state a large dataset,
many chunks or many objects, and reading or validating it at `--level strict`
costs time and memory in proportion to what it states. Open files from a source
you do not trust with the limits you would give any other untrusted input
(a container, `ulimit`, a timeout). What `medh5 scrub` cannot read is reported
as `unreadable` or `not_checked`, never passed as clean
([De-identify](docs/guides/deidentify.md)).

A task manifest chooses which files are opened; see the next section.

## Running a task manifest you did not write

A task manifest names its sources by path, so whoever wrote it chooses what
`medh5 task preflight`, `task reconcile` and `cache validate` open: any path
the running user can read, relative to the manifest's directory or absolute.
What medh5 guarantees is what the pins cover. A source that is not a regular
file — a named pipe, a device — is refused before it is opened, and a source
that is not the version its reference pins is read no further: its finding
(`T302`) names the pin and nothing the file holds, and the file adds no
identity, clock, event or document to the report. A file that matches its pin
is reported in full, and whether a path exists can still be told from its
finding (`T301`).

To run a manifest someone else wrote, confine it: `--source-root DIR` on the
command line, `confine=True` (the manifest's directory) or `confine=DIR` in
Python. A URI outside the directory, its `.` and `..` resolved as text, is
refused (`T301`) without the path being examined, the same whether or not it
exists, so a manifest can neither open nor probe anything outside it. Without
either, 2.1 opens wherever the URIs point and says so on stderr (a
`FutureWarning` in Python) when one lies outside the manifest's directory;
3.0 confines by default.

One limit remains: whether a source is a regular file is checked before HDF5
opens it, by path, and HDF5 opens only by path. A process that can write to a
source's directory can swap a named pipe in between, and the open then blocks.
Confine sources to a directory nobody else can write to.

## Native inventory

What every wheel and every `medh5` binary links statically. The crate column is
what `Cargo.lock` locks, and a test holds the two together; the version column
is the library's own, which is what advisories and scanners name.

| Library | Version | Built from | Used for |
|---|---|---|---|
| HDF5 | 2.2.0 | `hdf5-metno-src` 0.10.4 | the container |
| zlib | 1.3.2 | `libz-sys` 1.1.29 | HDF5's deflate filter, C-Blosc2's zlib codec |
| C-Blosc2 | 3.3.5 | vendored: `crates/medh5-sys/vendor/c-blosc2` | the Blosc2 codecs |
| LZ4 | 1.10.0 | `lz4-sys` 1.11.1+lz4-1.10.0 | C-Blosc2's LZ4 codecs |
| Zstandard | 1.5.7 | `zstd-sys` 2.1.0+zstd.1.5.7 | C-Blosc2's Zstd codec, the Zstandard filter (32015) |
| HDF5-Blosc2 filter | 3.0.1 | vendored: `crates/medh5-sys/vendor/hdf5-blosc2` | the Blosc2 filter (32026) |

The HDF5-Blosc2 filter is the source `hdf5plugin` 7.1.0 ships, with the local
changes `crates/medh5-sys/vendor/hdf5-blosc2/PATCHES.md` lists.

When a library changes (a `-sys` crate is bumped, or a vendored source
updated), update its row, then read every advisory against the new version.

## Advisories

| Advisory | Library | Status | Reason |
|---|---|---|---|
| [CVE-2026-85091](#cve-2026-85091) | zlib 1.3.2 | not affected | the vulnerable code is not linked |

### CVE-2026-85091

A heap overflow in zlib's gzip file interface (CWE-787). After a non-blocking
`gzwrite()` stalls, `gz_vacate()` leaves stale buffer pointers, and a later
`gzprintf()` or `gzvprintf()` writes past the input buffer. Published on
2026-09-03 against zlib 1.3.1.2 through 1.3.2, so the bundled 1.3.2 is in
range. No zlib release fixes it yet.

**Not affected:** the vulnerable code is not present (VEX justification
`vulnerable_code_not_present`). Nothing medh5 compiles in uses the gzip file
interface. HDF5's deflate filter and C-Blosc2's zlib codec call
`compress2`, `uncompress`, `deflate` and `inflate` on memory buffers.
`libz-sys` compiles `gzwrite.c` into its static archive, but a linker takes an
archive member only to resolve a reference, and nothing refers to one. `nm` on
2.0.1's published Linux wheels, x86_64 and aarch64, finds zlib's `adler32`,
`compress2`, `crc32`, `deflate`, `deflateInit_`, `inflate`, `inflateInit_` and
`uncompress`, and no `gz*` symbol; so does a build of this `Cargo.lock`. From
2.0.2 the release profile strips symbols, so repeat the check on a build with
`CARGO_PROFILE_RELEASE_STRIP=none`.

Nor will anything gain a reason to. From 2.2.0 the NIfTI converter is the
engine's, and a `.nii.gz` file is inflated in Rust (`flate2` over
`miniz_oxide`), never by zlib, and never past the bytes its header implies.
CI's `rust` job builds the native binary and the extension with their symbol
tables and runs `nm` on both: it fails on any `gz*` symbol, and on finding no
zlib symbol at all, which would mean it looked at nothing.

A scanner that finds zlib 1.3.2 in the binary (its version string is there,
for `deflateInit_` to check) or in a bill of materials built from `Cargo.lock`
still reports it. [`vex/medh5.openvex.json`](vex/medh5.openvex.json) states
this disposition in OpenVEX, which Trivy and Grype read with `--vex`.

Once `libz-sys` ships a zlib that fixes it, bump the crate, and record the bump
in the CHANGELOG.
