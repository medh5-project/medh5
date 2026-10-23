"""The public API is compared with the last release before it ships.

2.0.2's "no public API change" was checked by hand, and the patch still changed
Rust structs.  CI now compares both APIs with the newest release at or below
this tree: `cargo semver-checks` the crates, `.github/scripts/public_api.py`
the Python package.  These hold the script and its snapshot,
`tests/project/python_api.txt`, which says what a pull request changed in the
Python API where a reviewer reads it.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

from tests.helpers import ROOT

SCRIPTS = ROOT / ".github" / "scripts"


def _script(name: str) -> ModuleType:
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # `dataclasses` looks the module up while it builds the script's classes.
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def api() -> ModuleType:
    return _script("public_api.py")


def test_the_installed_api_is_the_snapshot(api: ModuleType) -> None:
    """A change to the Python API changes the snapshot in the same pull
    request, where a reviewer sees it."""
    snapshot = api.SNAPSHOT.read_text(encoding="utf-8")
    assert api.dump() == snapshot, (
        "the public Python API differs from tests/project/python_api.txt; if "
        "the change is meant, run `python .github/scripts/public_api.py write`"
    )


# --- dump ---


class Mode(enum.Enum):
    FAST = "fast"


@dataclasses.dataclass
class Settings:
    name: str
    tags: list[str] = dataclasses.field(default_factory=list)


class Thing:
    def method(self, a, /, b, *, c=Mode.FAST, d=object()): ...  # noqa: B008

    @classmethod
    def build(cls, size=(1, 2), **options): ...


def test_signatures_carry_names_kinds_and_literal_defaults(api: ModuleType) -> None:
    """No annotation and no repr that differs between Pythons: an enum member
    prints `Class.NAME`, a dataclass factory `<factory>`, any other value that
    is not a literal `<value>`, and the receiver is dropped."""
    assert api._signature(Thing.method, bound=True) == (
        "(a, /, b, *, c=Mode.FAST, d=<value>)"
    )
    assert api._signature(Thing.build, bound=True) == "(size=(1, 2), **options)"
    assert api._signature(Settings) == "(name, tags=<factory>)"
    assert api._signature(lambda *args, key: None) == "(*args, key)"
    assert api._default(...) == "<value>"  # PyO3's spelling of an unknown default
    assert api._default(float("nan")) == "nan"


def test_a_dump_line_carries_no_address(api: ModuleType) -> None:
    entry = ("medh5.X", "attribute", "<object object at 0x7f3a9c0b2d50>")
    assert api.line(entry) == "medh5.X\tattribute\t<object object>"
    assert api.line(("medh5", "module", "")) == "medh5\tmodule"


PACKAGE = {
    "__init__.py": """
        import dataclasses
        import enum
        from typing import Optional

        from fakeapi import sub

        __all__ = ["LIMIT", "Box", "Error", "Kind", "Maybe", "measure", "sub"]

        LIMIT = 3
        Maybe = Optional[int]

        class Error(Exception):
            pass

        class Kind(enum.Enum):
            A = 1

        class _Base:
            def shared(self, x):
                pass

            def __len__(self):
                return 0

            def __repr__(self):
                return "Box()"

        @dataclasses.dataclass
        class Box(_Base):
            size: int
            label: str = "x"

            @property
            def area(self):
                return self.size

            class Inner:
                def go(self):
                    pass

        def measure(box, *, unit="mm"):
            pass

        def _old():
            pass

        _DEPRECATED = {"old": (_old, "medh5 3.0 removes old: use measure")}
        """,
    "sub.py": '__all__ = ["ONE"]\nONE = 1.0\n',
    "_private.py": '__all__ = ["HIDDEN"]\nHIDDEN = 1\n',
}


def test_the_dump_lists_names_members_and_kinds(
    api: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every public module and name; members of our classes and of their
    bases of ours, with the protocol dunders but not the generated ones; a
    deprecated name from the table; no private module or base."""
    package = tmp_path / "fakeapi"
    package.mkdir()
    for name, source in PACKAGE.items():
        (package / name).write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(api, "PACKAGE", "fakeapi")
    assert api.dump().splitlines() == [
        "fakeapi\tmodule",
        "fakeapi.Box\tclass\t(size, label='x') bases=",
        "fakeapi.Box.Inner\tclass\t(?) bases=",
        "fakeapi.Box.Inner.go\tmethod\t()",
        "fakeapi.Box.__len__\tmethod\t()",
        "fakeapi.Box.area\tproperty",
        "fakeapi.Box.label\tfield",
        "fakeapi.Box.shared\tmethod\t(x)",
        "fakeapi.Box.size\tfield",
        "fakeapi.Error\tclass\t(?) bases=Exception",
        "fakeapi.Kind\tclass\t(?) bases=",
        "fakeapi.Kind.A\tmember\t1",
        "fakeapi.LIMIT\tconstant\tint",
        "fakeapi.Maybe\talias",
        "fakeapi.measure\tfunction\t(box, *, unit='mm')",
        "fakeapi.old\tdeprecated",
        "fakeapi.sub\tmodule",
        "fakeapi.sub.ONE\tconstant\tfloat",
    ]


# --- compare ---

BEFORE = """\
medh5\tmodule
medh5.Error\tclass\t(?) bases=Exception
medh5.Writer\tclass\t(path, *, codec='balanced') bases=
medh5.Writer.add\tmethod\t(ann_id, boxes, *, scores=None)
medh5.Writer.close\tmethod\t()
medh5.load\tfunction\t(path, mode='r', **options)
medh5.LIMIT\tconstant\tint
medh5.old\tfunction\t(x)
"""


def _changes(api: ModuleType, edit: dict[str, str | None]) -> tuple[list, list]:
    """BEFORE against BEFORE with lines replaced (None removes one)."""
    after = []
    for text in BEFORE.splitlines():
        path = text.split("\t", 1)[0]
        if path not in edit:
            after.append(text)
    after += [value for value in edit.values() if value is not None]
    return api.changes(api.parse(BEFORE), api.parse("\n".join(after)))


def test_the_same_api_changes_nothing(api: ModuleType) -> None:
    assert api.changes(api.parse(BEFORE), api.parse(BEFORE)) == ([], [])


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (
            {"medh5.Writer.close": None},
            "medh5.Writer.close: the method was removed",
        ),
        (
            {"medh5.load": "medh5.load\tfunction\t(mode='r', path=None, **options)"},
            "medh5.load: positional parameter 1 was 'path', is 'mode'",
        ),
        (
            {"medh5.load": "medh5.load\tfunction\t(path, mode, **options)"},
            "medh5.load: parameter 'mode' lost its default",
        ),
        (
            {"medh5.Writer.add": "medh5.Writer.add\tmethod\t(ann_id, boxes)"},
            "medh5.Writer.add: keyword 'scores' was removed",
        ),
        (
            {
                "medh5.Writer.add": (
                    "medh5.Writer.add\tmethod\t(ann_id, boxes, *, scores)"
                )
            },
            "medh5.Writer.add: parameter 'scores' lost its default",
        ),
        (
            {
                "medh5.Writer": (
                    "medh5.Writer\tclass\t(path, *, codec='balanced', root) bases="
                )
            },
            "medh5.Writer: new required keyword 'root'",
        ),
        (
            {"medh5.load": "medh5.load\tfunction\t(path, mode='r', *, strict=False)"},
            "medh5.load: **kwargs was removed",
        ),
        (
            {"medh5.load": "medh5.load\tfunction\t(path, /, mode='r', **options)"},
            "medh5.load: parameter 'path' became positional-only",
        ),
        (
            {"medh5.Error": "medh5.Error\tclass\t(?) bases=ValueError"},
            "medh5.Error: no longer a subclass of Exception",
        ),
        (
            {"medh5.LIMIT": "medh5.LIMIT\tconstant\tstr"},
            "medh5.LIMIT: was int, is str",
        ),
        (
            {"medh5.load": "medh5.load\tclass\t(path, mode='r', **options) bases="},
            "medh5.load: was a function, is a class",
        ),
    ],
)
def test_a_break_is_named(
    api: ModuleType, edit: dict[str, str | None], expected: str
) -> None:
    breaks, _ = _changes(api, edit)
    assert expected in breaks


def test_a_removed_class_is_one_break_not_one_per_member(api: ModuleType) -> None:
    breaks, _ = _changes(
        api,
        {"medh5.Writer": None, "medh5.Writer.add": None, "medh5.Writer.close": None},
    )
    assert breaks == ["medh5.Writer: the class was removed"]


def test_what_a_minor_may_add(api: ModuleType) -> None:
    """A new name, a new keyword with a default, a new optional positional
    after the others, a new base, and a name moving to `deprecated`."""
    breaks, added = _changes(
        api,
        {
            "medh5.Writer.add": (
                "medh5.Writer.add\tmethod\t(ann_id, boxes, extra=None, *, "
                "scores=None, quality=None)"
            ),
            "medh5.Error": "medh5.Error\tclass\t(?) bases=Exception,medh5.Base",
            "medh5.old": "medh5.old\tdeprecated",
            "medh5.Writer.flush": "medh5.Writer.flush\tmethod\t()",
            "medh5.new": "medh5.new\tfunction\t(x, *, y=1)",
            "medh5.Reader": "medh5.Reader\tclass\t(path) bases=",
            "medh5.Reader.read": "medh5.Reader.read\tmethod\t()",
        },
    )
    assert breaks == []
    # A new class's members come with it.
    assert added == [
        "medh5.Reader: new class",
        "medh5.Writer.flush: new method",
        "medh5.new: new function",
    ]


def test_an_unreadable_signature_is_not_compared(api: ModuleType) -> None:
    breaks, _ = _changes(api, {"medh5.Writer": "medh5.Writer\tclass\t(?) bases="})
    assert breaks == []


def test_a_deprecated_name_leaves_only_in_a_major(api: ModuleType) -> None:
    deprecated = BEFORE.replace("medh5.old\tfunction\t(x)", "medh5.old\tdeprecated")
    removed = deprecated.replace("medh5.old\tdeprecated\n", "")
    status, report = api.compare(deprecated, removed, "minor")
    assert status == 1
    assert "`medh5.old: the deprecated was removed`" in report
    status, report = api.compare(deprecated, removed, "major")
    assert status == 0
    assert "1 breaking change(s), allowed in a major" in report


def test_a_patch_adds_only_what_the_exceptions_list_allows(
    api: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A patch adds no API without a decision: an addition the list does not
    name fails, and a listed one goes to the job's summary with its decision,
    where the release's author reads it."""
    old, new, summary = tmp_path / "old.txt", tmp_path / "new.txt", tmp_path / "s.md"
    listed, empty = tmp_path / "listed.txt", tmp_path / "empty.txt"
    old.write_text(BEFORE, encoding="utf-8")
    new.write_text(BEFORE + "medh5.extra\tfunction\t()\n", encoding="utf-8")
    listed.write_text("# a comment\n\npython\tmedh5.extra\tD6\n", encoding="utf-8")
    empty.write_text("rust\tmedh5::extra\tD6\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    compare = ["compare", str(old), str(new), "--release-type"]
    assert api.main([*compare, "patch", "--exceptions", str(listed)]) == 0
    text = summary.read_text(encoding="utf-8")
    assert "No breaking change." in text
    assert "- `medh5.extra: new function` (D6)" in text
    # The list's Rust names are not Python's.
    assert api.main([*compare, "patch", "--exceptions", str(empty)]) == 1
    assert "1 addition(s) in a patch release, not allowed" in summary.read_text(
        encoding="utf-8"
    )
    assert api.main([*compare, "minor", "--exceptions", str(empty)]) == 0
    assert api.main(["compare", str(new), str(old), "--release-type", "patch"]) == 1


def test_a_patch_allowing_a_class_allows_none_of_its_members(api: ModuleType) -> None:
    """A listed class's members are additions of their own: listing a class
    is no way to add methods unreviewed."""
    after = BEFORE + "medh5.Reader\tclass\t(path) bases=\n"
    after += "medh5.Reader.read\tmethod\t()\n"
    assert api.compare(BEFORE, after, "patch")[0] == 1
    status, report = api.compare(BEFORE, after, "patch", {"medh5.Reader": "D6"})
    assert status == 1
    assert "- `medh5.Reader: new class` (D6)" in report
    assert "- `medh5.Reader.read: new method`\n" in report
    allowed = {"medh5.Reader": "D6", "medh5.Reader.read": "D6"}
    assert api.compare(BEFORE, after, "patch", allowed)[0] == 0


def _tree(options: list[str], commands: dict | None = None) -> dict:
    return {"options": options, "positionals": [], "commands": commands or {}}


def test_the_command_line_dump_lists_commands_and_their_own_options(
    api: ModuleType,
) -> None:
    """clap copies a global option into every command; it prints once, under
    `medh5`, so a new global option is one addition and not one per command.
    An option only `medh5` has is not inherited: a command's own option by
    the same spelling prints under the command."""
    tree = _tree(
        ["--threads", "--version", "--help"],
        {
            "info": _tree(
                ["--json", "--version", "--threads", "--help"],
                {"x": _tree(["--threads", "--help"])},
            ),
            "show": _tree(["--threads", "--help"]),
        },
    )
    assert list(api.cli_entries(tree)) == [
        ("medh5", "command", ""),
        ("medh5 --threads", "option", ""),
        ("medh5 --version", "option", ""),
        ("medh5 --help", "option", ""),
        ("medh5 info", "command", ""),
        ("medh5 info --json", "option", ""),
        ("medh5 info --version", "option", ""),
        ("medh5 info x", "command", ""),
        ("medh5 show", "command", ""),
    ]
    dumped = api.parse(api.dump_cli())
    assert dumped["medh5 --threads"].kind == "option"
    assert dumped["medh5 --version"].kind == "option"
    assert "medh5 info --threads" not in dumped
    assert "medh5 info --help" not in dumped
    assert dumped["medh5 convert from-nifti"].kind == "command"


def test_a_patch_adds_only_the_flags_the_exceptions_list_allows(
    api: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new flag is an addition like a new name: a patch fails one the
    list's `cli` names do not hold, and a removed flag is a break."""
    old, new = tmp_path / "old.txt", tmp_path / "new.txt"
    listed = tmp_path / "listed.txt"
    before = "medh5\tcommand\nmedh5 info\tcommand\nmedh5 info --json\toption\n"
    old.write_text(before, encoding="utf-8")
    new.write_text(before + "medh5 info --key\toption\n", encoding="utf-8")
    listed.write_text("cli\tmedh5 info --key\tD6\n", encoding="utf-8")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    compare = ["compare", "--cli", str(old), str(new), "--release-type"]
    # The list's Python names are not the command line's.
    python = tmp_path / "python.txt"
    python.write_text("python\tmedh5 info --key\tD6\n", encoding="utf-8")
    assert api.main([*compare, "patch", "--exceptions", str(python)]) == 1
    assert api.main([*compare, "patch", "--exceptions", str(listed)]) == 0
    assert api.main([*compare, "minor", "--exceptions", str(python)]) == 0
    status, report = api.compare(before, new.read_text(), "patch", {}, "Command line")
    assert status == 1 and report.startswith("## Command line (patch)")
    assert "- `medh5 info --key: new option`" in report
    removed = api.compare(before, "medh5\tcommand\n", "minor", {}, "Command line")
    assert removed[0] == 1
    assert "- `medh5 info: the command was removed`" in removed[1]


@pytest.mark.parametrize(
    "line",
    [
        "python\tmedh5.x",
        "go\tmedh5.x\tD6",
        "python\tmedh5.x \tD6",
        "python\tmedh5.x\tD6\textra",
    ],
)
def test_an_exceptions_line_is_kind_name_and_decision(
    api: ModuleType, line: str
) -> None:
    with pytest.raises(ValueError, match="line 1"):
        api.exceptions(line + "\n")
    with pytest.raises(ValueError, match="listed twice"):
        api.exceptions("python\tmedh5.x\tD6\npython\tmedh5.x\tD7\n")


def test_the_exceptions_are_the_thread_setting_and_the_workers_flags(
    api: ModuleType,
) -> None:
    """The list holds the names the maintainer's decisions allow in 2.2.3 and
    nothing else: the thread setting (D6), and command line flags (D6, D7),
    each of which the parser has."""
    from medh5._core import cli_command_tree

    allowed = api.exceptions(api.EXCEPTIONS.read_text(encoding="utf-8"))
    assert allowed["rust"] == {
        "medh5::parallel": "D6",
        "medh5::parallel::set_threads": "D6",
        "medh5::parallel::threads": "D6",
    }
    assert allowed["python"] == {"medh5.set_threads": "D6", "medh5.get_threads": "D6"}
    snapshot = api.parse(
        (ROOT / "tests" / "project" / "python_api.txt").read_text(encoding="utf-8")
    )
    assert set(allowed["python"]) <= set(snapshot)
    tree = cli_command_tree()
    for flag, decision in allowed["cli"].items():
        assert decision in ("D6", "D7"), flag
        program, *words, option = flag.split(" ")
        assert program == "medh5" and option.startswith("--"), flag
        node = tree
        for word in words:
            node = node["commands"][word]
        assert option in node["options"], flag


# --- Rust additions ---


@pytest.fixture(scope="module")
def rust_api() -> ModuleType:
    return _script("rust_api.py")


def _rustdoc(extra: bool = False, version: str = "2.2.2") -> dict:
    """A crate's rustdoc JSON, as much of it as the walk reads: modules, a
    function, a struct with a field and an inherent method, a trait impl, a
    private item, a re-export of another crate and one of its own."""

    def item(name, kind, inner, visibility="public"):
        return {"name": name, "visibility": visibility, "inner": {kind: inner}}

    index = {
        "0": item("medh5", "module", {"items": ["1", "2", "5", "6", "7"]}),
        "1": item("geometry", "module", {"items": ["3", "4"]}),
        "2": item(
            "hdf5",
            "use",
            {"source": "hdf5", "name": "hdf5", "id": "99", "is_glob": False},
        ),
        "3": item(
            "Grid",
            "struct",
            {"kind": {"plain": {"fields": ["8", "9"]}}, "impls": ["10", "11"]},
        ),
        "4": item("private", "function", {}, visibility="crate"),
        "5": item(
            "Grid",
            "use",
            {"source": "geometry::Grid", "name": "Grid", "id": "3", "is_glob": False},
        ),
        "6": item(None, "impl", {"items": [], "trait": None}),
        "7": item("VERSION", "constant", {}),
        "8": item("shape", "struct_field", {}),
        "9": item("hidden", "struct_field", {}, visibility="default"),
        "10": item(None, "impl", {"items": ["12"], "trait": None}),
        "11": item(None, "impl", {"items": ["13"], "trait": {"path": "Clone"}}),
        "12": item("ndim", "function", {}),
        "13": item("clone", "function", {}, visibility="default"),
    }
    if extra:
        index["0"]["inner"]["module"]["items"].append("20")
        index["20"] = item("parallel", "module", {"items": ["21", "22", "24"]})
        index["21"] = item("threads", "function", {})
        index["22"] = item("set_threads", "function", {})
        index["10"]["inner"]["impl"]["items"] += ["23", "25"]
        index["23"] = item("spacing", "function", {})
        # Hidden, as the JSON `cargo semver-checks` builds still holds them:
        # an attribute is `{"other": text}` in newer formats, the text in older.
        index["24"] = item("for_this_run", "function", {})
        index["24"]["attrs"] = [{"other": "#[doc(hidden)]"}]
        index["25"] = item("internal", "function", {})
        index["25"]["attrs"] = ['#[doc(alias = "x", hidden)]']
    return {"root": "0", "crate_version": version, "index": index}


def test_a_glob_of_another_crate_is_an_addition(rust_api: ModuleType) -> None:
    """A `pub use dependency::*` names a module the JSON does not hold, so the
    names it brings cannot be listed: the glob is recorded as one item, and a
    new one fails a patch rather than adding nothing."""
    old = _rustdoc()
    new = _rustdoc()
    new["index"]["0"]["inner"]["module"]["items"].append("30")
    new["index"]["30"] = {
        "name": "ndarray",
        "visibility": "public",
        "inner": {
            "use": {"source": "ndarray", "name": "ndarray", "id": "98", "is_glob": True}
        },
    }
    assert rust_api.items(new)["medh5::ndarray::*"] == "use"
    status, report = rust_api.compare([(json.dumps(old), json.dumps(new))], "patch")
    assert status == 1 and "- `medh5::ndarray::*: new use`" in report


def test_a_deep_module_is_walked_and_a_glob_cycle_ends(rust_api: ModuleType) -> None:
    """Depth is no reason to stop: a function twenty modules down is an item.
    A cycle is, and only a cycle: `pub use super::*` names its parent's items
    once more under the child, and not the child again; two modules that
    glob each other name each other's items once."""

    def item(name, kind, inner):
        return {"name": name, "visibility": "public", "inner": {kind: inner}}

    index = {"0": item("medh5", "module", {"items": ["1"]})}
    for level in range(1, 21):
        index[str(level)] = item(f"m{level}", "module", {"items": [str(level + 1)]})
    index["21"] = item("deep", "function", {})
    index["0"]["inner"]["module"]["items"].append("30")
    index["30"] = item("a", "module", {"items": ["31", "32"]})
    index["31"] = item("f", "function", {})
    index["32"] = item("b", "module", {"items": ["33"]})
    index["33"] = item(
        None, "use", {"source": "super", "name": "super", "id": "30", "is_glob": True}
    )
    # And two modules that glob each other.
    index["0"]["inner"]["module"]["items"] += ["40", "41"]
    index["40"] = item("x", "module", {"items": ["42"]})
    index["41"] = item("y", "module", {"items": ["43", "44"]})
    index["42"] = item(
        None, "use", {"source": "y", "name": "y", "id": "41", "is_glob": True}
    )
    index["43"] = item(
        None, "use", {"source": "x", "name": "x", "id": "40", "is_glob": True}
    )
    index["44"] = item("g", "function", {})
    found = rust_api.items({"root": "0", "index": index})
    assert {path for path in found if path.startswith(("medh5::x", "medh5::y"))} == {
        "medh5::x",
        "medh5::x::g",
        "medh5::y",
        "medh5::y::g",
    }
    deep = "::".join(["medh5", *(f"m{level}" for level in range(1, 21)), "deep"])
    assert found[deep] == "function"
    assert {path for path in found if path.startswith("medh5::a")} == {
        "medh5::a",
        "medh5::a::f",
        "medh5::a::b",
        "medh5::a::b::f",
    }


def test_rust_items_are_the_paths_a_caller_names(rust_api: ModuleType) -> None:
    assert rust_api.items(_rustdoc()) == {
        "medh5": "module",
        "medh5::geometry": "module",
        "medh5::geometry::Grid": "struct",
        "medh5::geometry::Grid::shape": "struct_field",
        "medh5::geometry::Grid::ndim": "function",
        "medh5::hdf5": "use",
        "medh5::Grid": "struct",
        "medh5::Grid::shape": "struct_field",
        "medh5::Grid::ndim": "function",
        "medh5::VERSION": "constant",
    }


def test_a_patch_adds_only_the_rust_items_the_exceptions_list_allows(
    rust_api: ModuleType,
) -> None:
    old, new = json.dumps(_rustdoc()), json.dumps(_rustdoc(extra=True))
    status, report = rust_api.compare([(old, new)], "patch")
    assert status == 1
    # A new module's items come with it; a method added to a type at both
    # of its paths is two additions.
    assert "- `medh5::parallel: new module`" in report
    assert "medh5::parallel::threads" not in report
    # What `#[doc(hidden)]` marks is not API.
    assert "for_this_run" not in report and "internal" not in report
    assert "- `medh5::Grid::spacing: new function`" in report
    assert "- `medh5::geometry::Grid::spacing: new function`" in report
    allowed = {
        "medh5::parallel": "D6",
        "medh5::Grid::spacing": "D9",
        "medh5::geometry::Grid::spacing": "D9",
    }
    # Allowing a module allows the module, not what it holds.
    status, report = rust_api.compare([(old, new)], "patch", allowed)
    assert status == 1
    assert "- `medh5::parallel: new module` (D6)" in report
    assert "- `medh5::parallel::threads: new function`\n" in report
    assert "- `medh5::parallel::set_threads: new function`\n" in report
    allowed |= {"medh5::parallel::threads": "D6", "medh5::parallel::set_threads": "D6"}
    status, report = rust_api.compare([(old, new)], "patch", allowed)
    assert status == 0
    assert "- `medh5::parallel::threads: new function` (D6)" in report
    assert rust_api.compare([(old, new)], "minor")[0] == 0
    assert rust_api.compare([(old, old)], "patch") == (
        0,
        "## Rust API additions (patch)\n\nNo public item added.\n",
    )


def test_the_rust_files_are_found_where_semver_checks_left_them(
    rust_api: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checks = tmp_path / "semver-checks"
    cache = checks / "cache"
    local = checks / "local-medh5-2_2_2-x86_64-unknown-linux-gnu-abc" / "target" / "doc"
    cache.mkdir(parents=True)
    local.mkdir(parents=True)
    (cache / "medh5-2_2_2-x86_64-unknown-linux-gnu-abc.json").write_text(
        json.dumps(_rustdoc())
    )
    (cache / "medh5-2_2_2-x86_64-unknown-linux-gnu-abc.metadata.json").write_text("{}")
    (cache / "medh5-2_2_1-x86_64-unknown-linux-gnu-abc.json").write_text(
        json.dumps(_rustdoc(version="2.2.1"))
    )
    (local / "medh5.json").write_text(json.dumps(_rustdoc(extra=True)))
    (local / "hdf5.json").write_text(json.dumps(_rustdoc()))
    exceptions = tmp_path / "exceptions.txt"
    exceptions.write_text("rust\tmedh5::parallel\tD6\n")
    argv = [
        "compare",
        "--semver-checks",
        str(checks),
        "--crate",
        "medh5",
        "--release-type",
        "patch",
    ]
    assert (
        rust_api.main([*argv, "--baseline", "2.2.2", "--exceptions", str(exceptions)])
        == 1
    )
    out = capsys.readouterr().out
    assert "- `medh5::parallel: new module` (D6)" in out
    assert "4 addition(s), not allowed in a patch" in out
    with pytest.raises(SystemExit, match="0 baseline and 1 local"):
        rust_api.main([*argv, "--baseline", "2.2.0"])


def test_ci_checks_rust_additions_after_semver_checks() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = ci[ci.index("\n  semver:") : ci.index("\n  python-api:")]
    assert job.index("cargo semver-checks") < job.index("rust_api.py compare")
    for crate in ("medh5-sys", "medh5", "medh5-cli"):
        assert f"--crate {crate}" in job


# --- the baseline ---


@pytest.mark.parametrize(
    ("current", "branch", "override", "expected"),
    [
        # The tree's version is the newest release's: the next release's work.
        ("2.0.3", "main", "", ("2.0.3", "minor")),
        ("2.0.3", "release/2.0", "", ("2.0.3", "patch")),
        # A merge group names its base as a ref.
        ("2.0.3", "refs/heads/release/2.0", "", ("2.0.3", "patch")),
        # A bumped tree is compared with the release before it.
        ("2.1.0", "main", "", ("2.0.3", "minor")),
        ("2.0.4", "release/2.0", "", ("2.0.3", "patch")),
        ("3.0.0", "main", "", ("2.0.3", "major")),
        # A release candidate is below its final, and never a baseline.
        ("2.1.0-rc.2", "main", "", ("2.0.3", "minor")),
        # 3.0's work says so outright.
        ("2.0.3", "main", "major", ("2.0.3", "major")),
    ],
)
def test_the_baseline_is_the_newest_final_release_at_or_below(
    current: str, branch: str, override: str, expected: tuple[str, str]
) -> None:
    release = _script("release.py")
    published = ["1.4.4", "2.0.0", "2.0.1", "2.0.3", "2.1.0-rc.1", "2.0.2", "x"]
    assert (
        release.api_baseline(current, published, branch=branch, override=override)
        == expected
    )


def test_the_baseline_refuses_what_it_cannot_compare() -> None:
    release = _script("release.py")
    with pytest.raises(SystemExit, match="no final release at or below 1.0.0"):
        release.api_baseline("1.0.0", ["2.0.0", "1.0.0-rc.1"], branch="main")
    with pytest.raises(SystemExit, match="is not one of major, minor, patch"):
        release.api_baseline("2.0.3", ["2.0.3"], branch="main", override="huge")
    with pytest.raises(SystemExit, match="is not X.Y.Z"):
        release.api_baseline("2.0", ["2.0.3"], branch="main")
