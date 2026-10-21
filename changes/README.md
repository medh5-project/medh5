# Changelog fragments

Every pull request adds one file here, `changes/<slug>.md`, named for its
change (`w907-nan-advice.md`), instead of editing `CHANGELOG.md`. The release
pull request assembles the fragments into the release's section with
`python .github/scripts/changelog.py build VERSION --date YYYY-MM-DD`, which
deletes them. Until then the documentation site shows them under
[Unreleased], and `python .github/scripts/changelog.py preview` prints them.
A pull request that changes only `plans/` needs no fragment: nothing in it
ships, and CI skips its checks.

## The format

```markdown
<!-- area: validation -->
### Fixed

- **`W907` no longer advises `int16` for a float image holding NaN.** Only the
  finite values were tested, so the advice lost every NaN on conversion.
```

- The first line names the fragment's area, one of: `files`, `writer`,
  `python-api`, `annotations`, `geometry`, `validation`, `specification`,
  `curation`, `deidentification`, `cohorts`, `clinical`, `converters`, `cli`,
  `torch`, `task-cache`, `platform`, `release`, `documentation`. Within a
  section, entries are sorted in that order, then by file name.
- Then one or more `### Section` blocks, each followed by a blank line and its
  entries, with one blank line between sections. The sections, in the order
  a release prints them: Security, Added, Deprecated, Behaviour changes,
  Removed (a major release only), Fixed, Performance, Documentation.
- Each entry opens with `- **`, a bold sentence saying what changed for the
  reader, and continues indented two spaces. Entries follow one another with
  no blank line between them.
- Lines are at most 80 columns. Write a dash as `—`, never `---`.

A pull request with nothing a user would notice (CI, tests, refactoring) adds a
fragment holding only

```markdown
<!-- no entry: CI only -->
```

It is a file rather than a label because CI re-runs a pull request with the
event it started with, and a label change would re-run all of it.

Which section an entry belongs in is said in `CONTRIBUTING.md`, under
"Changelog and releases".

`lead.md` is the exception: the release's bold summary paragraph, written in
the release pull request and consumed by `build`.

`python .github/scripts/changelog.py check` checks every fragment, and that
`CHANGELOG.md` has nothing under [Unreleased]; CI's `lint` job runs it, and
fails a pull request that adds no fragment. It also lists every line that
promises something about a later release ("will", "next", "a later release",
"removed in", "the last line to support"): an entry says what its own release
does, and a promise is the maintainer's to confirm before the release.
