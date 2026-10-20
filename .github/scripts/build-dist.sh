#!/usr/bin/env bash
# Build the Python wheel and the `medh5` binary for one target:
#
#     .github/scripts/build-dist.sh TARGET
#
# The wheels go to dist/ --- one abi3 wheel serves CPython 3.10 and later, and
# a second serves free-threaded CPython 3.14, which the stable ABI does not
# cover --- and the binary, archived with its licence, the notices of what it
# links and its README, to cli/.  CI runs this on every
# platform the project ships --- on Linux inside a manylinux_2_28 container, so
# both run on any glibc from 2.28, and a musllinux_1_2 one, for musl and a
# static binary --- and the release publishes what CI built.
set -euo pipefail

target="$1"

# What builds the artifacts is pinned, as the Rust is (rust-toolchain.toml): a
# new release of any of them reaches what the project publishes only through a
# pull request that bumps it here.  These are the newest maturin (within the
# range pyproject.toml's build-system requires) and CMake of the day 2.0.1 was
# built, so the ones that built it; a hosted runner's CMake is its image's.
maturin_version=1.15.0
cmake_version=4.4.4
# rustup-init for the manylinux and musllinux containers, which have no Rust:
# this release, checked against the digests static.rust-lang.org publishes
# beside it.
rustup_version=1.28.2
rustup_sha256_x86_64=20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c
rustup_sha256_aarch64=e3853c5a252fca15252d07cb23a1bdd9377a8c6f3efa01531109281ae47f841c
rustup_sha256_x86_64_musl=e6599a1c7be58a2d8eaca66a80e0dc006d87bbcf780a58b7343d6e14c1605cb2
rustup_sha256_aarch64_musl=a97c8f56d7462908695348dd8c71ea6740c138ce303715793a690503a94fc9a9

if [[ -x /opt/python/cp312-cp312/bin/python ]]; then
  # A manylinux or musllinux container: its own Python, with CMake and Rust
  # installed here.  A musl target is built in a musllinux container, whose
  # Rust is musl's too, so the binary links musl statically.
  python=/opt/python/cp312-cp312/bin/python
  MEDH5_FT_PYTHON=/opt/python/cp314-cp314t/bin/python
  "$python" -m pip install --quiet "maturin==$maturin_version" "cmake==$cmake_version"
  export PATH="$(dirname "$python"):$HOME/.cargo/bin:$PATH"
  case "$target" in
    *-linux-musl) libc=musl; compatibility=(--compatibility musllinux_1_2) ;;
    *) libc=gnu; compatibility=(--compatibility manylinux_2_28) ;;
  esac
  if ! command -v rustup >/dev/null; then
    arch=$(uname -m)
    case "$arch-$libc" in
      x86_64-gnu) digest=$rustup_sha256_x86_64 ;;
      aarch64-gnu) digest=$rustup_sha256_aarch64 ;;
      x86_64-musl) digest=$rustup_sha256_x86_64_musl ;;
      aarch64-musl) digest=$rustup_sha256_aarch64_musl ;;
      *) echo "no rustup-init digest for $arch-$libc" >&2; exit 1 ;;
    esac
    # Under this name: rustup is one binary that acts as the name it runs as.
    init="$(mktemp -d)/rustup-init"
    curl --proto '=https' --tlsv1.2 -sSf -o "$init" \
      "https://static.rust-lang.org/rustup/archive/$rustup_version/$arch-unknown-linux-$libc/rustup-init"
    # Compared here, not by `sha256sum --check`: BusyBox's, in the musllinux
    # images, takes none of GNU's long options.
    actual=$(sha256sum "$init" | cut -d " " -f 1)
    if [[ "$actual" != "$digest" ]]; then
      echo "rustup-init's digest is $actual, not $digest" >&2
      exit 1
    fi
    chmod +x "$init"
    "$init" -y --no-modify-path --profile minimal --default-toolchain none
    rm -r "$(dirname "$init")"
  fi
else
  # A hosted runner: Python from setup-python, CMake and rustup preinstalled;
  # build.yml names its free-threaded Python in MEDH5_FT_PYTHON.
  python=python
  "$python" -m pip install --quiet "maturin==$maturin_version"
  compatibility=()
fi
# The Rust rust-toolchain.toml pins, which rustup reads from the checkout, as
# every cargo below does, with the standard library of the target (a no-op for
# the host's own).
rustup toolchain install --no-self-update
rustup target add "$target"

# (The `+` form: an empty array is "unbound" to the bash 3.2 macOS ships.)
# Every build names its target, so every target's output is in
# target/<target>/release, a cross-build's as much as a native one's.
maturin build --release --locked --target "$target" --out dist --interpreter "$python" ${compatibility[@]+"${compatibility[@]}"}
# Free-threaded CPython cannot load an abi3 extension, so it gets one built for
# it: `-cp314-cp314t-<platform>.whl`.  Only PyO3 and the extension compile
# again; the engine and HDF5 are the build above's.  The module runs without
# the GIL (`gil_used = false`), so CPython leaves it off as it imports it.
if [[ -z "${MEDH5_FT_PYTHON:-}" ]]; then
  echo "MEDH5_FT_PYTHON names no free-threaded Python to build the cp314t wheel with" >&2
  exit 1
fi
maturin build --release --locked --target "$target" --out dist --interpreter "$MEDH5_FT_PYTHON" ${compatibility[@]+"${compatibility[@]}"}
# The binary, from the same checkout and lock as the wheel.  Cargo reuses what
# maturin compiled only where the two builds agree on target and flags; where
# they do not, the graph compiles again, HDF5 included --- CI time, nothing else.
cargo build --release --locked --target "$target" -p medh5-cli

exe="target/$target/release/medh5"
if [[ -f "$exe.exe" ]]; then
  exe="$exe.exe"
fi
# Apple silicon runs x86_64 code too (Rosetta), so running what was built
# proves nothing about its architecture: the binary's and each wheel's
# extension's are read instead.
if [[ "$target" == *-apple-darwin ]]; then
  want="${target%%-*}"
  if [[ "$want" == aarch64 ]]; then want=arm64; fi
  unpacked=$(mktemp -d)
  for wheel in dist/*.whl; do
    "$python" -m zipfile -e "$wheel" "$unpacked/$(basename "$wheel")"
  done
  for built in "$exe" "$unpacked"/*/medh5/_core*.so; do
    arch=$(lipo -archs "$built")
    if [[ "$arch" != "$want" ]]; then
      echo "$built is built for $arch, not $want" >&2
      exit 1
    fi
  done
  rm -r "$unpacked"
fi
# The musl binary is the one for any Linux, whatever its C library or none:
# it must need no shared library at all.
if [[ "$target" == *-linux-musl ]] && readelf -d "$exe" | grep -q NEEDED; then
  readelf -d "$exe" >&2
  echo "$exe needs a shared library; it should link musl statically" >&2
  exit 1
fi
"$exe" --version

version=$("$python" -c 'import tomllib; print(tomllib.load(open("Cargo.toml", "rb"))["workspace"]["package"]["version"])')
name="medh5-$version-$target"
mkdir -p "cli/$name"
cp "$exe" LICENSE THIRD_PARTY_NOTICES "cli/$name/"
cp crates/medh5-cli/README.md "cli/$name/README.md"
cd cli
if [[ "$target" == *windows* ]]; then
  "$python" -m zipfile -c "$name.zip" "$name"
else
  "$python" -m tarfile -c "$name.tar.gz" "$name"
fi
rm -rf "$name"
ls -l . ../dist
