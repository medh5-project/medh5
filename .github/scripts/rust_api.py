"""The public Rust items a pull request adds, against the release before it.

    python .github/scripts/rust_api.py compare OLD.json NEW.json [...] \
        --release-type TYPE
    python .github/scripts/rust_api.py compare --semver-checks target/semver-checks \
        --baseline VERSION --crate medh5 [--crate ...] --release-type TYPE

`cargo semver-checks` fails what breaks a caller, and has no lint for an
added item: under `patch` nothing failed a new public name.  This reads the
rustdoc JSON of a crate at the baseline release and at this tree (the files
`cargo semver-checks` builds, or any two built by the same rustdoc) and lists
every public path the new one has and the old one lacks: modules, functions,
types and their fields, variants and inherent methods, traits and their
items, constants, statics and macros, reached from the crate root through
public modules and re-exports, as a caller names them.  Items rustdoc hides
(`#[doc(hidden)]`) are not in the JSON and are not API.  Under `patch` an
addition that `.github/api-exceptions.txt` does not list (kind `rust`) fails;
under `minor` and `major` the additions are listed only.  Standard library
only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from public_api import EXCEPTIONS, RELEASE_TYPES, ROOT, exceptions

MEMBERS = ("struct", "enum", "union", "trait")


def items(doc: dict[str, Any]) -> dict[str, str]:
    """Every public path of the crate `doc` documents, and its kind."""
    index: dict[str, Any] = doc["index"]
    found: dict[str, str] = {}

    def add(path: list[str], kind: str) -> bool:
        name = "::".join(path)
        if name in found:
            return False
        found[name] = kind
        return True

    def visit(id_: Any, path: list[str], name: str | None = None) -> None:
        item = index.get(str(id_))
        # A re-export cycle (`pub use super::*`) would name paths forever.
        if item is None or len(path) > 16:
            return
        (kind, inner), *_ = item["inner"].items()
        if kind == "use":
            if item.get("visibility") != "public":
                return
            target = inner.get("id")
            if inner.get("is_glob"):
                module = index.get(str(target))
                if module is not None and "module" in module["inner"]:
                    for child in module["inner"]["module"]["items"]:
                        visit(child, path)
                return
            if target is None or str(target) not in index:
                # A re-export of another crate's item, or one rustdoc did
                # not document: the name is the API.
                add([*path, inner["name"]], "use")
                return
            visit(target, path, inner["name"])
            return
        if kind == "impl":
            return
        name = name or item.get("name")
        if not name or item.get("visibility") != "public":
            return
        here = [*path, name]
        if not add(here, kind):
            return
        if kind == "module":
            for child in inner["items"]:
                visit(child, here)
        elif kind in MEMBERS:
            members(item, kind, inner, here)

    def members(
        item: dict[str, Any], kind: str, inner: dict[str, Any], here: list[str]
    ) -> None:
        children: list[Any] = []
        if kind == "struct":
            shape = inner.get("kind")
            if isinstance(shape, dict):
                if "plain" in shape:
                    children += shape["plain"].get("fields", [])
                elif "tuple" in shape:
                    children += [f for f in shape["tuple"] if f is not None]
        elif kind == "union":
            children += inner.get("fields", [])
        elif kind == "enum":
            children += inner.get("variants", [])
        elif kind == "trait":
            children += inner.get("items", [])
        for impl_id in inner.get("impls", []):
            impl = index.get(str(impl_id))
            if impl is None:
                continue
            body = impl["inner"].get("impl", {})
            # A trait's items are the trait's API, not the type's.
            if body.get("trait") is None and not body.get("blanket_impl"):
                children += body.get("items", [])
        for child_id in children:
            child = index.get(str(child_id))
            if child is None or not child.get("name"):
                continue
            (child_kind, _), *_ = child["inner"].items()
            public = child.get("visibility") == "public" or kind in ("enum", "trait")
            if public:
                add([*here, child["name"]], child_kind)

    root = index[str(doc["root"])]
    crate = root["name"]
    found[crate] = "module"
    for child in root["inner"]["module"]["items"]:
        visit(child, [crate])
    return found


def additions(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """What `new` adds, each new item once: an added module's or type's
    members come with it."""
    fresh = sorted(set(new) - set(old))
    return [
        f"{path}: new {new[path]}"
        for path in fresh
        if path.rpartition("::")[0] not in fresh
    ]


def compare(
    pairs: list[tuple[str, str]],
    release_type: str,
    allowed: dict[str, str] | None = None,
) -> tuple[int, str]:
    """The exit status and a Markdown report, for each crate's (old, new)
    rustdoc JSON text."""
    allowed = allowed or {}
    lines = [f"## Rust API additions ({release_type})", ""]
    unlisted: list[str] = []
    listed: list[str] = []
    for old_text, new_text in pairs:
        for line in additions(items(json.loads(old_text)), items(json.loads(new_text))):
            path = line.rpartition(": new ")[0]
            if path in allowed:
                listed.append(f"- `{line}` ({allowed[path]})")
            else:
                unlisted.append(f"- `{line}`")
    if not listed and not unlisted:
        lines.append("No public item added.")
    if listed:
        lines += [f"{len(listed)} addition(s), each excepted:", "", *listed, ""]
    if unlisted:
        verdict = (
            f"not allowed in a patch"
            f" ({EXCEPTIONS.relative_to(ROOT).as_posix()} does not list them)"
            if release_type == "patch"
            else "allowed"
        )
        lines += [f"{len(unlisted)} addition(s), {verdict}:", "", *unlisted, ""]
    status = 1 if unlisted and release_type == "patch" else 0
    return status, "\n".join(lines).rstrip("\n") + "\n"


def locate(directory: Path, crate: str, baseline: str) -> tuple[Path, Path]:
    """The baseline's rustdoc JSON of `crate` and this tree's, as
    `cargo semver-checks` left them under `directory`: a registry release in
    its cache (`cache/`), the local project in its build directory
    (`local-*/`)."""
    name = crate.replace("-", "_")
    old: list[Path] = []
    new: list[Path] = []
    for path in sorted(directory.rglob("*.json")):
        if path.name.endswith(".metadata.json"):
            continue
        local = any(
            part.startswith("local-") for part in path.relative_to(directory).parts
        )
        if local and path.name != f"{name}.json":
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            root = doc["index"][str(doc["root"])]["name"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if root != name:
            continue
        if local:
            new.append(path)
        elif doc.get("crate_version") == baseline:
            old.append(path)
    if len(old) != 1 or len(new) != 1:
        raise SystemExit(
            f"rust_api.py: {len(old)} baseline and {len(new)} local rustdoc JSON"
            f" files of {crate} {baseline} under {directory}, not one of each"
        )
    return old[0], new[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "compare", help="fail on an addition a patch may not make"
    )
    command.add_argument(
        "pairs",
        nargs="*",
        type=Path,
        metavar="OLD NEW",
        help="each crate's baseline rustdoc JSON, then this tree's",
    )
    command.add_argument(
        "--semver-checks",
        type=Path,
        metavar="DIR",
        help="find the files where cargo semver-checks left them",
    )
    command.add_argument("--baseline", metavar="VERSION", help="with --semver-checks")
    command.add_argument(
        "--crate", action="append", default=[], help="with --semver-checks"
    )
    command.add_argument("--release-type", choices=RELEASE_TYPES, required=True)
    command.add_argument("--exceptions", type=Path, default=EXCEPTIONS)
    args = parser.parse_args(argv)
    paths: list[Path] = list(args.pairs)
    if args.semver_checks:
        if paths or not args.baseline or not args.crate:
            parser.error("--semver-checks takes --baseline and --crate, and no files")
        for crate in args.crate:
            paths += locate(args.semver_checks, crate, args.baseline)
    if not paths or len(paths) % 2:
        parser.error("give each crate's two files: OLD NEW [OLD NEW ...]")
    texts = [p.read_text(encoding="utf-8") for p in paths]
    status, report = compare(
        list(zip(texts[::2], texts[1::2], strict=True)),
        args.release_type,
        exceptions(args.exceptions.read_text(encoding="utf-8"))["rust"],
    )
    sys.stdout.write(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(report)
    return status


if __name__ == "__main__":
    sys.exit(main())
