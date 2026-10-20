# Install

Most people need one command:

```bash
pip install medh5
```

That installs the Python package and the `medh5` command line: everything you
need to create, read, validate and inspect `.medh5` files. HDF5 and its
compression filters are built into the package, so NumPy is the only other
dependency and there is nothing else to set up.

Check that it worked:

```bash
medh5 --version
```

The rest of this page is for when you need more: an extra for PyTorch, MONAI
or DICOM, the command line without Python, the Rust crate, or a system that
pip has no ready-built package for.

## Add what you connect it to

The package works on its own. Extras add the libraries that a few of its
modules use to talk to other tools; install the ones you use.

| To… | Install | Adds |
|---|---|---|
| train with PyTorch (`medh5.torch`) | `pip install "medh5[torch]"` | PyTorch |
| load samples into MONAI (`medh5.monai`) | `pip install "medh5[monai]"` | MONAI, PyTorch |
| import DICOM series, DICOM SEG and RTSTRUCT; export RTSTRUCT | `pip install "medh5[dicom]"` | pydicom |
| export DICOM SEG | `pip install "medh5[dicomseg]"` | highdicom, pydicom |
| open files directly with `h5py` | `pip install "medh5[h5py]"` | h5py, hdf5plugin |

Extras combine in one command:

```bash
pip install "medh5[torch,dicom]"
```

If you use a module without its extra, the error names the one to install.

NIfTI files and nnU-Net datasets need no extra: since 2.2.0 the engine reads
and writes NIfTI itself, and nibabel is no longer installed or imported. The
`nifti` extra is kept, empty, so `pip install "medh5[nifti]"` still works.

## Where it runs

pip downloads a ready-built package (a *wheel*) for Python 3.10 and later on:

| System | Requirement |
|---|---|
| Linux, x86_64 or ARM64 | glibc 2.28 or later: Ubuntu 20.04, Debian 10, RHEL 8 or newer |
| Linux with musl, x86_64 or ARM64 | musl 1.2 or later: Alpine 3.13 or newer |
| macOS, Apple silicon | macOS 11 or later |
| Windows, x64 | Windows 10 or later |

Each system has a second wheel for free-threaded CPython 3.14 (`python3.14t`),
which cannot load the first. medh5 is not yet audited for running without the
GIL, so it asks for it: importing medh5 turns the GIL back on, and Python says
so once with a `RuntimeWarning`. Everything works as it does on a regular
CPython; nothing runs in parallel that would not there. Running without the GIL
is planned for 2.2; see [Threads, processes and open files](explanation/concurrency.md).

Anywhere else, pip builds medh5 from its source package instead. That takes
several minutes and needs:

- Rust 1.85.1 or later, from [rustup.rs](https://rustup.rs);
- a C compiler and CMake, because HDF5 is compiled as part of the build;
- network access to crates.io, where the build fetches its Rust dependencies.

This covers macOS on Intel (2.2.0 dropped its wheel; 2.1 was the last with
one), Linux with a glibc older than 2.28 (Ubuntu 18.04, CentOS 7) or a musl
older than 1.2, Windows on ARM, PyPy 3.11 and free-threaded
CPython 3.15 or later. Free-threaded CPython 3.13 and PyPy 3.10 cannot build it
at all: PyO3, the library that binds the engine to Python, does not support
them. If pip prints `Building wheel for medh5`, it found no wheel for your
system and needs these three.

Python 3.10 reaches its end of life in October 2026. medh5 still supports it,
and nothing warns at run time; dropping it will be announced one minor release
ahead.

## Only the command line

`pip install medh5` already puts `medh5` on your path. To have the command line
on a machine without Python, such as a server, a CI image or a shared
workstation, install the standalone binary instead. It is the same program,
compiled ahead of time, with the same commands and output.

**Homebrew**, on macOS or Linux:

```bash
brew install medh5-project/medh5/medh5
```

**A download.** Every [GitHub Release](https://github.com/medh5-project/medh5/releases/latest)
has an archive for Linux (x86_64 or ARM64, glibc 2.28 or later), macOS (Apple
silicon) and Windows (x64), and a `SHA256SUMS` file to check it against. The
Linux archives named `…-linux-musl` hold a static binary, which needs no C
library at all and so runs on any Linux of its architecture, Alpine and
distroless images among them. Unpack it and put `medh5` on your path.

Each archive, like each wheel, also carries a signed attestation that the
project's release workflow built it from the tagged commit. With the
[GitHub CLI](https://cli.github.com), check one before you unpack it:

```bash
gh attestation verify medh5-2.1.0-x86_64-unknown-linux-gnu.tar.gz --repo medh5-project/medh5
```

**From source**, with Rust 1.85.1 or later, a C compiler and CMake:

```bash
cargo install --locked medh5-cli
```

On Windows, set one variable first; without it the build stops and names it:

```powershell
$env:CFLAGS_x86_64_pc_windows_msvc = "-DNDEBUG"
cargo install --locked medh5-cli
```

Prefer the Homebrew formula or a download where one fits your system: a crate
on crates.io carries no build profile, so `cargo install` builds without the
link-time optimisation the release binaries have, and keeps the symbol table,
for a larger and slower binary. To build it as the releases are, give cargo
the project's release profile:

```bash
CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 \
  CARGO_PROFILE_RELEASE_STRIP=symbols cargo install --locked medh5-cli
```

In PowerShell:

```powershell
$env:CARGO_PROFILE_RELEASE_LTO = "thin"
$env:CARGO_PROFILE_RELEASE_CODEGEN_UNITS = "4"
$env:CARGO_PROFILE_RELEASE_STRIP = "symbols"
cargo install --locked medh5-cli
```

Or, in any shell, as cargo options:

```bash
cargo install --locked medh5-cli --config 'profile.release.lto="thin"' \
  --config profile.release.codegen-units=4 --config 'profile.release.strip="symbols"'
```

The binary does everything itself, NIfTI conversion included, except the
other converters (`medh5 convert …` for DICOM, DICOM SEG, RTSTRUCT, ITK and
nnU-Net) and the dataloader row of `medh5 bench`, which are built on Python
libraries. For those it runs a Python that has medh5 installed:
`python3` (`python` on Windows), or the interpreter named by the
`MEDH5_PYTHON` environment variable. When there is none, it says what to
install.

## In a Rust project

```bash
cargo add medh5
```

The [`medh5` crate](reference/rust.md) is the format engine itself, the code the
Python package and the command line run. It compiles HDF5 from source and links
it in, so a program built with it needs nothing installed; building needs Rust
1.85.1 or later, a C compiler and CMake. For an MSVC target, give HDF5's build
`NDEBUG` as above (`CFLAGS_<target>=-DNDEBUG`); [the crate's page](reference/rust.md)
explains why.

## The 1.x line

medh5 1.x is pure Python over h5py, and writes format 1.0:

```bash
pip install "medh5<2"
```

Use it where 2.x has no wheel and cannot be built. It does not have the
`clinical` profile of format 1.1. A sample without clinical history is format
1.0 whichever line writes it, so 1.x reads what 2.x writes for imaging alone.
Moving code from 1.x to 2.x? 2.0 changed the Python API where it meets HDF5;
[the changelog](changelog.md) lists every change.

## If something goes wrong

**pip is compiling medh5, or fails because `cargo` or `cmake` is missing.**
There is no wheel for your system or your Python version; see
[Where it runs](#where-it-runs).

**An `ImportError` names an extra.** Install it, for example
`pip install "medh5[dicom]"` for DICOM conversion; see
[the table of extras](#add-what-you-connect-it-to).

**`medh5 convert` says the converters are part of the Python package.** You
are running the standalone binary, and it found no Python with medh5 installed.
Install the package with the extras you need into a Python on your path, or
point `MEDH5_PYTHON` at an interpreter that has them.

**h5py opens a file but cannot read its images.** The default storage profile
compresses with Blosc2, which h5py reads only once `hdf5plugin` is imported:
install `medh5[h5py]` and `import hdf5plugin` before opening the file. To hand
files to someone with a plain HDF5 build, write them with
`medh5 recompress --profile portable`, which uses gzip; see
[Storage](reference/storage.md).

**A read fails with "filter plugins disabled".** The dataset names a
compression filter medh5 does not carry. Blosc2 and Zstandard are compiled in
and HDF5 brings gzip, shuffle and Fletcher32, which covers every storage
profile, so medh5 does not load HDF5's filter plugins unless asked: HDF5 would
load whatever shared library its search path holds, as whoever opened the file,
and on Windows that path includes a directory any local user can create. To
read such a file, set `HDF5_PLUGIN_PATH` to the directory holding the filter,
or `MEDH5_ALLOW_PLUGINS=1` to load from HDF5's standard directory
(`/usr/local/hdf5/lib/plugin`, or `%ALLUSERSPROFILE%\hdf5\lib\plugin` on
Windows). Both are read once, before the process opens its first file.
