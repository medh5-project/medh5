"""Refresh Cargo.lock, holding back the crates that build the native libraries.

    python .github/scripts/lock_refresh.py [--summary FILE]

`.github/workflows/lock-refresh.yml` runs this every week and opens a pull
request with the result; Dependabot cannot, since THIRD_PARTY_NOTICES must be
regenerated with the lock.  `cargo update` moves every crate to the newest
version the crates' `rust-version` allows.  Then each crate that builds a
library of `SECURITY.md`'s native inventory (`hdf5-metno-src`, `libz-sys`, ...)
is put back where it was: its row, the advisories read against it and the VEX
statements follow its version, and a person updates those.  The bumps held
back are listed in the summary, the pull request's body, for that person.
Last, the notices are regenerated.

Standard library only.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def native_crates(security: str) -> list[str]:
    """The crates the native inventory names in its "Built from" column, as
    `crate` version (a vendored source names none)."""
    table = security.split("\n## Native inventory\n", 1)[1].split("\n## ", 1)[0]
    return sorted(
        set(re.findall(r"^\|[^|]+\|[^|]+\| `([\w-]+)` [^\s,|]+[^|]* \|", table, re.M))
    )


def locked(lock: str) -> dict[str, list[str]]:
    """Each crate Cargo.lock names, with its versions (a crate can be locked at
    more than one)."""
    found: dict[str, list[str]] = {}
    for name, version in re.findall(
        r'^name = "([^"]+)"\nversion = "([^"]+)"', lock, re.M
    ):
        found.setdefault(name, []).append(version)
    return found


def held_back(
    before: dict[str, list[str]], after: dict[str, list[str]], native: list[str]
) -> list[tuple[str, str, str]]:
    """`(crate, old, new)` for each native crate the update moved.  A crate
    locked at one version before and after is all the inventory holds."""
    moves = []
    for name in native:
        old, new = before.get(name, []), after.get(name, [])
        if len(old) == 1 and len(new) == 1 and old != new:
            moves.append((name, old[0], new[0]))
        elif sorted(old) != sorted(new):
            raise SystemExit(
                f"{name} was locked at {old or 'nothing'} and is now at "
                f"{new or 'nothing'}: the native inventory holds one version"
            )
    return moves


def summary(moves: list[tuple[str, str, str]]) -> str:
    lines = [
        "The weekly refresh of `Cargo.lock` (`.github/workflows/lock-refresh.yml`),",
        "with `THIRD_PARTY_NOTICES` regenerated from it.",
        "",
    ]
    if moves:
        lines += [
            "Held back, because each builds a library of `SECURITY.md`'s native",
            "inventory: a bump updates its row, re-reads its advisories and the",
            "VEX statements in `vex/`, by hand, in a pull request of its own.",
            "",
            "| Crate | Locked | Available |",
            "|---|---|---|",
            *(f"| `{name}` | {old} | {new} |" for name, old, new in moves),
        ]
    else:
        lines.append("No crate of the native inventory had a newer version.")
    return "\n".join(lines) + "\n"


def cargo(*args: str) -> None:
    print("+ cargo", " ".join(args), flush=True)
    # Within the crates' `rust-version`, as the `msrv` job requires.
    env = {**os.environ, "CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS": "fallback"}
    subprocess.run(["cargo", *args], cwd=ROOT, check=True, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, help="write the pull request body")
    args = parser.parse_args(argv)
    lock = ROOT / "Cargo.lock"
    native = native_crates((ROOT / "SECURITY.md").read_text(encoding="utf-8"))
    before = locked(lock.read_text(encoding="utf-8"))
    cargo("update")
    moves = held_back(before, locked(lock.read_text(encoding="utf-8")), native)
    for name, old, new in moves:
        cargo("update", "-p", f"{name}@{new}", "--precise", old)
    still = held_back(before, locked(lock.read_text(encoding="utf-8")), native)
    if still:
        raise SystemExit(f"could not hold back {still}")
    subprocess.run(
        [sys.executable, str(ROOT / ".github/scripts/third_party_notices.py")],
        cwd=ROOT,
        check=True,
    )
    text = summary(moves)
    print(text)
    if args.summary:
        args.summary.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
