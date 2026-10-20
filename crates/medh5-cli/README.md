# medh5-cli

The native `medh5` command line: inspect, validate, verify, curate, pack and
benchmark MEDH5 samples; select from a subject's clinical history as of a
cutoff; and check task manifests and feature caches against the samples they
name.  It is a frontend over the
[`medh5`](https://crates.io/crates/medh5) format engine — the same code the
Python package's `medh5` command runs — so both produce the same output and
the same exit codes: `0` success, `1` a handled error, `2` a usage error.

```bash
cargo install --locked medh5-cli # or a binary from the GitHub Release, or Homebrew
medh5 info case.medh5            # grids, images, annotations, coverage
medh5 validate case.medh5 --level integrity
medh5 verify case.medh5          # digests and content_id
medh5 conformance run corpus/    # the 175-case conformance corpus
```

HDF5 is linked statically: the binary needs nothing installed. Building it
(`cargo install`) needs a C compiler and CMake, which HDF5's build uses.

**Prefer the release binaries** (the GitHub Release, or Homebrew) where one
fits your system. A crate on crates.io carries no build profile, so
`cargo install` builds without the link-time optimisation the releases use,
and keeps the symbol table: a larger binary, and a slower one. To build it as
the releases are, give cargo the workspace's release profile:

```bash
CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4 \
  CARGO_PROFILE_RELEASE_STRIP=symbols cargo install --locked medh5-cli
```

In PowerShell, set the same three first:

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

**On Windows (MSVC)**, give HDF5's C build `NDEBUG` before installing:

```powershell
$env:CFLAGS_x86_64_pc_windows_msvc = "-DNDEBUG"
cargo install --locked medh5-cli
```

cmake-rs, which builds HDF5, drops CMake's release flags under the Visual
Studio generator, `/DNDEBUG` with them; HDF5 would keep its assertions, and a
damaged file would abort the process instead of being reported. The build stops
and says so when the flag is missing. The release binaries are built with it.

## Converters

The NIfTI converters are native: the binary reads and writes NIfTI itself,
with no Python.

```bash
medh5 convert from-nifti case.medh5 --image CT=ct.nii.gz
```

The other converters (DICOM, DICOM SEG, RTSTRUCT, ITK, nnU-Net) are
integrations with Python libraries — pydicom, highdicom — and live in the
Python package, as does the dataloader row of `medh5 bench` (PyTorch). The
binary hands them to a Python interpreter that has the package installed
(`python3` — `python` on Windows — or `MEDH5_PYTHON`), and says what to install
when there is none:

```bash
pip install "medh5[dicom]"
medh5 convert from-dicom series/ case.medh5
```

`medh5 migrate`, which reads 0.x files, is native.

The full command reference is at
[medh5.readthedocs.io](https://medh5.readthedocs.io/en/stable/reference/cli/).
