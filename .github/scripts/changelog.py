"""The changelog's fragments: one file per pull request, assembled at release.

    python .github/scripts/changelog.py check
    python .github/scripts/changelog.py preview
    python .github/scripts/changelog.py build VERSION --date YYYY-MM-DD

Each pull request adds `changes/<slug>.md` (the format is in
`changes/README.md`), written while the change is fresh, instead of editing
`CHANGELOG.md`, which then never conflicts between pull requests.  `check`
holds every fragment to the format, and `CHANGELOG.md`'s [Unreleased] section
empty, and lists every line that promises something about a later release, for
the maintainer to confirm before the release; `preview` prints what the next
release's section will hold, and the documentation site shows it under
[Unreleased]; `build` writes the section into `CHANGELOG.md` and deletes the
fragments, in the release pull request.

Standard library only: the lint job runs it before anything is installed.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHANGES = ROOT / "changes"
CHANGELOG = ROOT / "CHANGELOG.md"

# In the order a release's section prints them.  Removed is for a major only.
SECTIONS = (
    "Security",
    "Added",
    "Deprecated",
    "Behaviour changes",
    "Removed",
    "Fixed",
    "Performance",
    "Documentation",
)

# In the order a section's entries print: the file before what is in it, the
# format before what reads it, the library before its frontends.
AREAS = (
    "files",
    "writer",
    "python-api",
    "annotations",
    "geometry",
    "validation",
    "specification",
    "curation",
    "deidentification",
    "cohorts",
    "clinical",
    "converters",
    "cli",
    "torch",
    "task-cache",
    "platform",
    "release",
    "documentation",
)

# Files in `changes/` that are not fragments.
NOT_FRAGMENTS = ("README.md", "lead.md")

WIDTH = 80

_AREA = re.compile(r"<!-- area: ([a-z-]+) -->")
_NO_ENTRY = re.compile(r"<!-- no entry: \S.*? -->")
_DASH = re.compile(r"(?<!-)---(?!-)")
_CODE = re.compile(r"`+[^`\n]*`+")
# Words that promise something about a later release.  An entry states what
# its release does; a promise about the next one is the maintainer's to make,
# and 2.2.0's said the converters move onto the engine "next" after they had.
# The phrases are plans/README.md's ("will require", "the last line to
# support", "removed in") and the forms the changelog has used: a later or
# coming release of any kind ("a later minor"), a version something waits for
# ("until 3.0", "planned for 2.2"), and a deprecation, which promises a
# removal.  An entry wraps at 80 columns, so a phrase may span two lines.
_RELEASE = r"(?:releases?|versions?|minors?|majors?|patch(?:es)?|lines?)"
_PROMISE = re.compile(
    r"\b(?:will|next|future|upcoming|deprecat\w*"
    rf"|(?:later|subsequent|coming|forthcoming)\s+{_RELEASE}"
    r"|removed\s+in|until\s+v?\d|(?:planned|scheduled|slated)\s+for"
    r"|last\s+(?:line|release|version|minor)\s+to\s+support)\b",
    re.I,
)


class FragmentError(Exception):
    """A fragment that does not follow `changes/README.md`."""


@dataclasses.dataclass(frozen=True)
class Entry:
    section: str
    area: str
    name: str
    text: str  # the entry's lines, from `- **` to its last line, no newline


def fragments(directory: Path = CHANGES) -> list[Path]:
    """The fragments in `directory`, by file name."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.glob("*.md")
        if path.name not in NOT_FRAGMENTS and path.is_file()
    )


def parse(path: Path) -> list[Entry]:
    """The entries of one fragment ([] for a `no entry` fragment).

    Raises `FragmentError` naming the file and line of the first problem.
    """
    name = path.name
    text = path.read_text(encoding="utf-8")

    def fail(line: int, why: str) -> FragmentError:
        return FragmentError(f"changes/{name}:{line}: {why}")

    if not text.endswith("\n") or text.endswith("\n\n"):
        raise fail(text.count("\n") + 1, "end the file with one newline")
    lines = text[:-1].split("\n")
    for number, line in enumerate(lines, 1):
        if len(line) > WIDTH:
            raise fail(number, f"{len(line)} columns; wrap at {WIDTH}")
        if line != line.rstrip():
            raise fail(number, "trailing whitespace")
        if _DASH.search(_CODE.sub("", line)):
            raise fail(number, "write a dash as `—`, not `---`")
    if _NO_ENTRY.fullmatch(lines[0]):
        if len(lines) != 1:
            raise fail(2, "a `no entry` fragment holds that line only")
        return []
    area = _AREA.fullmatch(lines[0])
    if area is None:
        raise fail(
            1,
            "start with `<!-- area: NAME -->` or `<!-- no entry: REASON -->`",
        )
    if area[1] not in AREAS:
        raise fail(1, f"unknown area {area[1]!r}; one of {', '.join(AREAS)}")
    entries: list[Entry] = []
    section: str | None = None
    seen: set[str] = set()
    current: list[str] = []
    expect_blank = False
    need_heading = False

    def close() -> None:
        if current:
            assert section is not None
            entries.append(Entry(section, area[1], name, "\n".join(current)))
            current.clear()

    for number, line in enumerate(lines[1:], 2):
        if expect_blank:
            if line:
                raise fail(number, "leave a blank line after a heading")
            expect_blank = False
            continue
        if need_heading and not line.startswith("#"):
            # A blank line would end the list: entries follow one another.
            raise fail(number, "no blank line between entries")
        need_heading = False
        if line.startswith("#"):
            close()
            heading = re.fullmatch(r"### (.+)", line)
            if heading is None:
                raise fail(number, "a section heading is `### Name`")
            if heading[1] not in SECTIONS:
                raise fail(
                    number,
                    f"unknown section {heading[1]!r}; one of {', '.join(SECTIONS)}",
                )
            if section is not None and not any(e.section == section for e in entries):
                raise fail(number, f"section {section!r} has no entry")
            if heading[1] in seen:
                raise fail(number, f"section {heading[1]!r} appears twice")
            seen.add(heading[1])
            section = heading[1]
            expect_blank = True
        elif not line:
            if section is None:
                raise fail(number, "a blank line before the first heading")
            if not current:
                raise fail(number, "one blank line between sections")
            close()
            need_heading = True
            continue
        elif section is None:
            raise fail(number, "an entry before the first `### Section`")
        elif line.startswith("- "):
            close()
            if not line.startswith("- **"):
                raise fail(number, "an entry opens with `- **`, its bold summary")
            current.append(line)
        elif line.startswith("  ") and current:
            current.append(line)
        else:
            raise fail(number, "continue an entry indented two spaces")
    close()
    if section is None:
        raise fail(len(lines), "no `### Section` heading")
    if not any(e.section == section for e in entries):
        raise fail(len(lines), f"section {section!r} has no entry")
    return entries


def collect(directory: Path = CHANGES) -> list[Entry]:
    """Every entry of every fragment in `directory`."""
    return [entry for path in fragments(directory) for entry in parse(path)]


def assemble(entries: list[Entry]) -> str:
    """The sections, in SECTIONS order, each with its entries sorted by area
    (AREAS order) and then file name; '' when there is none."""
    blocks = []
    for section in SECTIONS:
        chosen = sorted(
            (e for e in entries if e.section == section),
            key=lambda e: (AREAS.index(e.area), e.name),
        )
        if chosen:
            body = "\n".join(e.text for e in chosen)
            blocks.append(f"### {section}\n\n{body}")
    return "\n\n".join(blocks)


def promises(directory: Path = CHANGES) -> list[str]:
    """Every line of a fragment, or of `lead.md`, that promises something about
    a later release, as `changes/NAME:LINE: TEXT`, code spans aside.

    Not an error: the release checklist has the maintainer confirm each one.
    """
    lead = directory / "lead.md"
    paths = [*fragments(directory), *([lead] if lead.is_file() else [])]
    found = []
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        # Searched as one text, a line per line, so that a phrase wrapped
        # onto the next line is found, and is reported at its first.
        prose = "\n".join(
            "" if line.startswith("<!--") else _CODE.sub("", line) for line in lines
        )
        numbers = {prose.count("\n", 0, m.start()) for m in _PROMISE.finditer(prose)}
        found += [
            f"changes/{path.name}:{n + 1}: {lines[n].strip()}" for n in sorted(numbers)
        ]
    return found


def unreleased(changelog: str) -> str:
    """What `CHANGELOG.md` has under [Unreleased], stripped."""
    found = re.search(
        r"^## \[Unreleased\][^\n]*\n(.*?)(?=^## |\Z)", changelog, re.M | re.S
    )
    if found is None:
        raise FragmentError("CHANGELOG.md has no `## [Unreleased]` section")
    return found[1].strip()


def splice(changelog: str, preview: str) -> str:
    """`changelog` with `preview` under its [Unreleased] heading."""
    if not preview:
        return changelog
    heading = "## [Unreleased]\n\n"
    if heading not in changelog:
        raise FragmentError("CHANGELOG.md has no `## [Unreleased]` section")
    return changelog.replace(heading, f"{heading}{preview}\n\n", 1)


def check(directory: Path = CHANGES, changelog: Path = CHANGELOG) -> int:
    """Every fragment parses, and [Unreleased] is empty; the number of entries."""
    entries = collect(directory)
    if unreleased(changelog.read_text(encoding="utf-8")):
        raise FragmentError(
            "CHANGELOG.md has entries under [Unreleased]: move them into "
            "changes/<name>.md (see changes/README.md)"
        )
    return len(entries)


def build(
    version: str, date: str, directory: Path = CHANGES, changelog: Path = CHANGELOG
) -> None:
    """Write `version`'s section into `changelog` and delete the fragments."""
    # As tagged, in Cargo's spelling: a pre-release is `2.1.0-rc.1`.
    if not re.fullmatch(r"\d+\.\d+\.\d+(-[0-9A-Za-z]+(\.[0-9A-Za-z]+)*)?", version):
        raise FragmentError(f"version {version!r} is not X.Y.Z or X.Y.Z-PRE")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise FragmentError(f"date {date!r} is not YYYY-MM-DD")
    check(directory, changelog)
    paths = fragments(directory)
    body = assemble(collect(directory))
    lead_path = directory / "lead.md"
    lead = lead_path.read_text(encoding="utf-8").strip() if lead_path.is_file() else ""
    if not lead:
        raise FragmentError(
            "write changes/lead.md, the release's bold summary, before building"
        )
    section = f"## [{version}] — {date}\n\n{lead}\n"
    if body:
        section += f"\n{body}\n"
    text = changelog.read_text(encoding="utf-8")
    if f"## [{version}]" in text:
        raise FragmentError(f"CHANGELOG.md already has a section for {version}")
    heading = "## [Unreleased]\n\n"
    # Checked before anything is written or deleted: a heading spelt otherwise
    # would leave the changelog unchanged and the fragments gone.
    if heading not in text:
        raise FragmentError("CHANGELOG.md has no `## [Unreleased]` line, blank after")
    # `\n` on every platform: Windows would otherwise write the file as CRLF.
    changelog.write_text(
        text.replace(heading, heading + section + "\n", 1), "utf-8", newline="\n"
    )
    for path in [*paths, lead_path]:
        path.unlink()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="changelog.py", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "check", help="check every fragment, and [Unreleased]; list promises"
    )
    commands.add_parser("preview", help="print the assembled entries")
    building = commands.add_parser("build", help="write a release's section")
    building.add_argument("version")
    building.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            count = check(CHANGES, CHANGELOG)
            print(f"changelog: {len(fragments(CHANGES))} fragments, {count} entries")
            for line in promises(CHANGES):
                print(f"changelog: a promise for the maintainer to confirm: {line}")
        elif args.command == "preview":
            print(assemble(collect(CHANGES)))
        else:
            build(args.version, args.date, CHANGES, CHANGELOG)
            print(f"changelog: wrote [{args.version}] into CHANGELOG.md")
    except FragmentError as error:
        print(f"changelog: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
