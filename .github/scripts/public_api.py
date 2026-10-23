"""The public Python API as text, and whether a change to it breaks callers.

    python .github/scripts/public_api.py dump [--output FILE]
    python .github/scripts/public_api.py write
    python .github/scripts/public_api.py compare OLD NEW --release-type TYPE

`dump` prints the API of the *installed* `medh5`: the package and every module
path without a `_`-prefixed part (`medh5._core` is stubtest's), one line per
public name, per class member and per module, sorted:

    medh5.sample.Sample<TAB>class<TAB>(path, mode='r') bases=
    medh5.sample.Sample.close<TAB>method<TAB>()

Signatures carry parameter names, kinds and defaults, never annotations: a
literal default prints its repr, an enum member `Class.NAME`, a dataclass
factory `<factory>` and anything else `<value>`; a leading `self` or `cls` is
dropped and addresses are stripped, so the text is the same on every Python
the package supports and with or without its optional dependencies.
Constants print their type only. A name in a module's `_DEPRECATED` table
prints with kind `deprecated`.

`write` regenerates `tests/project/python_api.txt`, which a test holds equal to
the installed package's dump. `compare` reads two dumps and lists what breaks a
caller of OLD: a removed name or member, a change of kind (except to
`deprecated`), a positional parameter renamed or moved, a keyword, `*args` or
`**kwargs` removed, a default removed, a new required parameter, a constant's
type or a class's base changed. It fails on any of them unless TYPE is `major`;
under `patch` it also fails an addition that `.github/api-exceptions.txt` does
not list, and lists the ones it does. Standard library only: CI runs it in a
virtual environment holding nothing but a released wheel.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import enum
import importlib
import inspect
import os
import pkgutil
import re
import sys
import typing
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "tests" / "project" / "python_api.txt"
PACKAGE = "medh5"
RELEASE_TYPES = ("major", "minor", "patch")
EXCEPTIONS = ROOT / ".github" / "api-exceptions.txt"
EXCEPTION_KINDS = ("rust", "python", "cli")

UNKNOWN = "(?)"
"""A signature `inspect` cannot read (a PyO3 exception type)."""

PROTOCOL = frozenset(
    {
        "__call__",
        "__contains__",
        "__enter__",
        "__exit__",
        "__fspath__",
        "__getitem__",
        "__iter__",
        "__len__",
        "__next__",
    }
)
"""The dunders that are API: what a caller writes `x[i]`, `len(x)` or `with x`
for. The rest (`__eq__`, `__repr__`, `__replace__`, …) differ between Python
versions in what a dataclass or PyO3 generates, and are not listed."""

_ADDRESS = re.compile(r" at 0x[0-9A-Fa-f]+")


# --- dump ---


def _ours(obj: Any) -> bool:
    module = getattr(obj, "__module__", None) or ""
    return module == PACKAGE or module.startswith(PACKAGE + ".")


def _type_name(cls: type) -> str:
    if _ours(cls):
        return f"{cls.__module__}.{cls.__qualname__}"
    if cls.__module__ == "builtins":
        return cls.__qualname__
    return "<external>"


def _literal(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return not isinstance(value, enum.Enum)
    if isinstance(value, (tuple, list)):
        return all(_literal(item) for item in value)
    if isinstance(value, dict):
        return all(_literal(k) and _literal(v) for k, v in value.items())
    return False


def _default(value: Any) -> str:
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if repr(value) == "<factory>":
        return "<factory>"
    if _literal(value):
        return repr(value)
    return "<value>"


def _signature(obj: Any, *, bound: bool = False) -> str:
    """`(a, b=1, *, c)`; `bound` drops the leading `self` or `cls`."""
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return UNKNOWN
    parameters = list(signature.parameters.values())
    if bound and parameters and parameters[0].name in ("self", "cls"):
        parameters = parameters[1:]
    parts: list[str] = []
    slash = star = False
    kinds = inspect.Parameter
    for parameter in parameters:
        if parameter.kind is kinds.POSITIONAL_ONLY:
            slash = True
        elif slash:
            parts.append("/")
            slash = False
        if parameter.kind is kinds.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        if parameter.kind is kinds.VAR_POSITIONAL:
            star = True
            text = f"*{parameter.name}"
        elif parameter.kind is kinds.VAR_KEYWORD:
            text = f"**{parameter.name}"
        else:
            text = parameter.name
        if parameter.default is not inspect.Parameter.empty:
            text += f"={_default(parameter.default)}"
        parts.append(text)
    if slash:
        parts.append("/")
    return f"({', '.join(parts)})"


def _constructor(cls: type) -> str:
    """The class's call signature, or `(?)` when no class of ours defines one:
    what Python reports for `Exception` or `object` differs between versions,
    as it does for an enum, whose call is a lookup."""
    if issubclass(cls, enum.Enum):
        return UNKNOWN
    for klass in cls.__mro__:
        if "__init__" in vars(klass) or "__new__" in vars(klass):
            if not _ours(klass):
                return UNKNOWN
            break
    return _signature(cls)


def _bases(cls: type) -> str:
    """The bases a caller can name: builtins and our public classes."""
    names = [_type_name(base) for base in cls.__bases__]
    return ",".join(
        name
        for name in names
        if name not in ("object", "<external>")
        and not any(part.startswith("_") for part in name.split("."))
    )


def _member(cls: type, name: str, raw: Any) -> tuple[str, str]:
    if isinstance(raw, staticmethod):
        return "staticmethod", _signature(raw.__func__)
    if isinstance(raw, classmethod) or type(raw).__name__ == "classmethod_descriptor":
        return "classmethod", _signature(getattr(cls, name), bound=True)
    if isinstance(raw, property) or type(raw).__name__ == "getset_descriptor":
        return "property", ""
    if type(raw).__name__ in ("cached_property", "member_descriptor"):
        return "property", ""
    if inspect.isclass(raw):
        return "class", ""
    if callable(raw):
        return "method", _signature(raw, bound=True)
    return "attribute", _type_name(type(raw))


def _class(path: str, cls: type) -> Iterator[tuple[str, str, str]]:
    yield path, "class", f"{_constructor(cls)} bases={_bases(cls)}"
    fields = set()
    if dataclasses.is_dataclass(cls):
        fields = {field.name for field in dataclasses.fields(cls)}
        for name in fields:
            if not name.startswith("_"):
                yield f"{path}.{name}", "field", ""
    members = dict(cls.__members__) if issubclass(cls, enum.Enum) else {}
    for name, member in members.items():
        value = member.value
        yield f"{path}.{name}", "member", repr(value) if _literal(value) else ""
    seen: set[str] = set()
    for klass in cls.__mro__:
        if not _ours(klass):
            continue
        for name, raw in vars(klass).items():
            if name in seen or name in fields or name in members:
                continue
            seen.add(name)
            if name.startswith("_") and name not in PROTOCOL:
                continue
            if inspect.isclass(raw) and _ours(raw):
                yield from _class(f"{path}.{name}", raw)
            else:
                yield (f"{path}.{name}", *_member(cls, name, raw))


def _modules() -> list[str]:
    package = importlib.import_module(PACKAGE)
    names = [PACKAGE]
    for info in pkgutil.walk_packages(package.__path__, PACKAGE + "."):
        if not any(part.startswith("_") for part in info.name.split(".")):
            names.append(info.name)
    return sorted(names)


def entries() -> Iterator[tuple[str, str, str]]:
    """Every line of the dump, unsorted."""
    for module_name in _modules():
        module = importlib.import_module(module_name)
        yield module_name, "module", ""
        for name in vars(module).get("_DEPRECATED", {}):
            yield f"{module_name}.{name}", "deprecated", ""
        for name in module.__all__:
            path = f"{module_name}.{name}"
            value = getattr(module, name)
            if inspect.ismodule(value):
                yield path, "module", ""
            elif typing.get_origin(value) is not None:
                # `tuple[int, int]` is a class on one Python and not on
                # another, and spells itself differently on each.
                yield path, "alias", ""
            elif inspect.isclass(value):
                yield from _class(path, value)
            elif callable(value):
                yield path, "function", _signature(value)
            else:
                yield path, "constant", _type_name(type(value))


def line(entry: tuple[str, str, str]) -> str:
    """One line of the dump, without the addresses a repr may carry."""
    return _ADDRESS.sub("", "\t".join(entry).rstrip("\t"))


def dump() -> str:
    # Reaching a deprecated name warns; the dump reads the table instead, and
    # an import's own warning is not the dump's business.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lines = {line(entry) for entry in entries()}
    return "".join(f"{text}\n" for text in sorted(lines))


# --- compare ---


@dataclasses.dataclass(frozen=True)
class Entry:
    kind: str
    detail: str


def parse(text: str) -> dict[str, Entry]:
    api: dict[str, Entry] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        path, kind, *rest = line.split("\t")
        if len(rest) > 1:
            raise ValueError(f"line {number}: more than three fields: {line!r}")
        api[path] = Entry(kind, rest[0] if rest else "")
    return api


@dataclasses.dataclass(frozen=True)
class Parameters:
    positional: list[tuple[str, bool, bool]]
    """Each positional parameter: its name, whether it has a default, and
    whether it may be passed by keyword."""
    keyword: dict[str, bool]
    """Every parameter a caller may pass by keyword, and whether it has a default."""
    args: bool
    kwargs: bool


def parameters(signature: str) -> Parameters | None:
    """The parameters of a dumped signature, or `None` when it is `(?)`."""
    if signature == UNKNOWN:
        return None
    source = re.sub(r"=<(value|factory)>", "=_", signature)
    function = ast.parse(f"def f{source}: pass").body[0]
    assert isinstance(function, ast.FunctionDef)
    spec = function.args
    positional = [*spec.posonlyargs, *spec.args]
    first_default = len(positional) - len(spec.defaults)
    rows = [
        (arg.arg, index >= first_default, index >= len(spec.posonlyargs))
        for index, arg in enumerate(positional)
    ]
    keyword = {name: default for name, default, by_keyword in rows if by_keyword}
    for arg, default in zip(spec.kwonlyargs, spec.kw_defaults, strict=True):
        keyword[arg.arg] = default is not None
    return Parameters(rows, keyword, spec.vararg is not None, spec.kwarg is not None)


def _signature_breaks(old: str, new: str) -> list[str]:
    before, after = parameters(old), parameters(new)
    if before is None or after is None:
        return []
    breaks = []
    for index, (name, default, by_keyword) in enumerate(before.positional):
        if index >= len(after.positional):
            breaks.append(f"positional parameter {name!r} is no longer positional")
            continue
        new_name, new_default, new_by_keyword = after.positional[index]
        if by_keyword and new_name != name:
            breaks.append(
                f"positional parameter {index + 1} was {name!r}, is {new_name!r}"
            )
        elif by_keyword and not new_by_keyword:
            breaks.append(f"parameter {name!r} became positional-only")
        elif default and not new_default:
            breaks.append(f"parameter {name!r} lost its default")
    old_positional = {row[0] for row in before.positional}
    new_positional = {row[0] for row in after.positional}
    for name, default in before.keyword.items():
        if name not in after.keyword:
            if not after.kwargs:
                breaks.append(f"keyword {name!r} was removed")
        elif default and not after.keyword[name] and name not in old_positional:
            breaks.append(f"parameter {name!r} lost its default")
    for index, (name, default, _) in enumerate(after.positional):
        if not default and index >= len(before.positional):
            breaks.append(f"new required positional parameter {name!r}")
    known = set(before.keyword) | old_positional | new_positional
    for name, default in after.keyword.items():
        if not default and name not in known:
            breaks.append(f"new required keyword {name!r}")
    if before.args and not after.args:
        breaks.append("*args was removed")
    if before.kwargs and not after.kwargs:
        breaks.append("**kwargs was removed")
    return breaks


def _split_class(detail: str) -> tuple[str, set[str]]:
    signature, _, bases = detail.rpartition(" bases=")
    return signature, set(filter(None, bases.split(",")))


def changes(
    old: dict[str, Entry], new: dict[str, Entry]
) -> tuple[list[str], list[str]]:
    """What breaks a caller of `old`, and what `new` adds."""
    breaks: list[str] = []
    gone: list[str] = []
    for path in sorted(old):
        if any(path.startswith(parent + ".") for parent in gone):
            continue
        before = old[path]
        after = new.get(path)
        if after is None:
            gone.append(path)
            breaks.append(f"{path}: the {before.kind} was removed")
            continue
        if after.kind != before.kind:
            if after.kind != "deprecated":
                breaks.append(f"{path}: was a {before.kind}, is a {after.kind}")
            gone.append(path)
            continue
        if before.kind == "class":
            old_signature, old_bases = _split_class(before.detail)
            new_signature, new_bases = _split_class(after.detail)
            for base in sorted(old_bases - new_bases):
                breaks.append(f"{path}: no longer a subclass of {base}")
            reasons = _signature_breaks(old_signature, new_signature)
        elif before.kind in ("function", "method", "classmethod", "staticmethod"):
            reasons = _signature_breaks(before.detail, after.detail)
        elif before.kind in ("constant", "attribute", "member"):
            reasons = (
                [f"was {before.detail or 'a value'}, is {after.detail or 'a value'}"]
                if before.detail != after.detail
                else []
            )
        else:
            reasons = []
        breaks.extend(f"{path}: {reason}" for reason in reasons)
    fresh = sorted(set(new) - set(old))
    added = [
        f"{path}: new {new[path].kind}"
        for path in fresh
        if path.rpartition(".")[0] not in fresh
    ]
    return breaks, added


def exceptions(text: str) -> dict[str, dict[str, str]]:
    """`.github/api-exceptions.txt`: for each kind, each name a patch may add
    and the decision that allowed it."""
    allowed: dict[str, dict[str, str]] = {kind: {} for kind in EXCEPTION_KINDS}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3 or not all(f.strip() == f and f for f in fields):
            raise ValueError(f"line {number}: not KIND<TAB>NAME<TAB>DECISION: {line!r}")
        kind, name, decision = fields
        if kind not in allowed:
            raise ValueError(
                f"line {number}: kind {kind!r} is not one of"
                f" {', '.join(EXCEPTION_KINDS)}"
            )
        if name in allowed[kind]:
            raise ValueError(f"line {number}: {name} is listed twice")
        allowed[kind][name] = decision
    return allowed


def compare(
    old_text: str,
    new_text: str,
    release_type: str,
    allowed: dict[str, str] | None = None,
) -> tuple[int, str]:
    """The exit status and a Markdown report.  `allowed` is the Python names
    a patch may add, with the decision that allowed each."""
    allowed = allowed or {}
    breaks, added = changes(parse(old_text), parse(new_text))
    lines = [f"## Python API ({release_type})", ""]
    if breaks:
        verdict = "allowed in a major" if release_type == "major" else "not allowed"
        lines += [f"{len(breaks)} breaking change(s), {verdict}:", ""]
        lines += [f"- `{line}`" for line in breaks]
    else:
        lines.append("No breaking change.")
    unlisted: list[str] = []
    if added and release_type == "patch":
        listed = [line for line in added if line.partition(":")[0] in allowed]
        unlisted = [line for line in added if line.partition(":")[0] not in allowed]
        if listed:
            lines += [
                "",
                f"{len(listed)} addition(s) in a patch release, each excepted:",
                "",
            ]
            lines += [
                f"- `{line}` ({allowed[line.partition(':')[0]]})" for line in listed
            ]
        if unlisted:
            lines += [
                "",
                f"{len(unlisted)} addition(s) in a patch release, not allowed"
                f" ({EXCEPTIONS.relative_to(ROOT).as_posix()} does not list them):",
                "",
            ]
            lines += [f"- `{line}`" for line in unlisted]
    status = 1 if (breaks and release_type != "major") or unlisted else 0
    return status, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("dump", help="print the installed package's API")
    command.add_argument("--output", type=Path)
    commands.add_parser("write", help=f"regenerate {SNAPSHOT.relative_to(ROOT)}")
    command = commands.add_parser("compare", help="fail on a breaking change")
    command.add_argument("old", type=Path)
    command.add_argument("new", type=Path)
    command.add_argument("--release-type", choices=RELEASE_TYPES, required=True)
    command.add_argument(
        "--exceptions",
        type=Path,
        default=EXCEPTIONS,
        help="the additions a patch may make (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if args.command == "dump":
        text = dump()
        if args.output:
            args.output.write_text(text, encoding="utf-8", newline="\n")
        else:
            sys.stdout.write(text)
        return 0
    if args.command == "write":
        SNAPSHOT.write_text(dump(), encoding="utf-8", newline="\n")
        print(f"wrote {SNAPSHOT.relative_to(ROOT)}")
        return 0
    status, report = compare(
        args.old.read_text(encoding="utf-8"),
        args.new.read_text(encoding="utf-8"),
        args.release_type,
        exceptions(args.exceptions.read_text(encoding="utf-8"))["python"],
    )
    sys.stdout.write(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(report)
    return status


if __name__ == "__main__":
    sys.exit(main())
