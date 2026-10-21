# What 2.2.0 taught

Checked on 2026-10-21 against `main` at 5249d79 (v2.2.0 plus #86). Sources:
the lane A–F threads and the release thread; pull requests #68–#86; the 21
merge-queue runs and the 124 pull-request CI runs after the lanes started;
every Codex review thread on #69–#85; and the `fuzz.yml` and
`lock-refresh.yml` runs of 16–20 October.

## What happened

The lanes started at 01:13 on 16 October. By 14:09 the same day, 13 of the 15
workstream pull requests had merged. Then lanes C, D and F and the coordinator
hit the weekly usage limit. The last two pull requests waited four days:
W22-04 (#76) merged at 21:42 on 20 October and W22-10 PR 2 (#84) at 22:35. The
release pull request (#85) took 2.3 hours. The maintainer tagged v2.2.0 at
about 23:39, and #86 marked the plan released.

All 15 workstream pull requests landed as planned, with no workstream cut or
split further. A pull request took **5.8 hours** from open to merge at the
median (2.1 h to 10.2 h, and 114 h for #76). In 2.1.0 the median was 1.8 hours.

## Findings

### 1. The weekly limit set the release date, not the work

- Lanes A, B and E were done by 11:03–12:02 on the first day. Lanes C, D and F
  stopped between 13:41 and 14:09 and resumed about 11:00 on 20 October.
- **#76** was mid-review when it stopped. It needed 21 more Codex rounds,
  about 10 hours of work, on 20 October.
- **#84** had not been opened. The session's environment was reset during the
  pause, and lane F's uncommitted annotation-carry work was lost and written
  again. #84 opened at 16:49 on 20 October.
- 2.1.0's limit stopped six lanes for 2–3.5 hours. This one stopped three for
  four days. Six parallel lanes spend a weekly limit in under a day.

### 2. Codex reviewed every pull request, and its rounds became the long pole

- **What it found.** Codex reviewed all of #69–#85, against 12 pull requests
  it missed in 2.1.0. It left **148 findings: 78 P1 and 70 P2**. In 2.1.0 it
  left 31.

  | PR | Workstream | Findings | P1 | P2 | Rounds |
  |---|---|---|---|---|---|
  | #76 | W22-04 free-threading | 38 | 18 | 20 | 21 |
  | #81 | W22-03 enhanced DICOM | 26 | 20 | 6 | about 20 pushes |
  | #84 | W22-10 PR 2 | 24 | 13 | 11 | about 12 |
  | #75 | W22-02 PR 2 | 17 | 9 | 8 | |
  | #80 | W22-09 clinical cohorts | 13 | 8 | 5 | 7 |
  | the other ten | | 30 | 10 | 20 | |

- **Declines.** About 10 findings were declined, each with a reason on its
  thread:
  - #75: 3, on E3's parity with 2.1.0, one of them settled by the maintainer on
    a decision card.
  - #84: 3–4, among them the `valid_mask` report, path-qualified
    `ignore_mask`, and the order of `ResampleReport`.
  - #76: 2, the dict values snapshot and the registry fork barrier.
  - #74: 1, which did not reproduce.
  - #82: 1, where Codex had ITK's B-spline naming wrong.
- **Findings came in classes.** The findings grouped into five kinds:
  - the concurrency surface;
  - DICOM stack and concatenation semantics;
  - resample geometry and annotation memory;
  - NIfTI overflow and parity;
  - path and collision safety.

  A lane that fixed only the instance Codex named got the next instance on the
  next round. Lane D sometimes swept the whole class after the first finding,
  which is the habit to keep.
- **The cost.** Every round was a push and a full CI run. 124 pull-request CI
  runs started after the lanes did, and **86 (69 %) were cancelled** by the
  next push. Lane F's one branch alone ran 41.

### 3. The queue was clean, but slower than in 2.1.0

- All 21 merge-queue runs passed on their first attempt.
- On the first morning (#70 to #80), a run took **72–126 minutes**, against
  43–74 in 2.1.0. Late runs, alone, took 25–31 minutes.
- **#74's run passed in 126 minutes** and was dropped anyway, because the
  timeout is 120.
- #69 and #73 each ran twice, and #79 dropped on a conflict with #71.
- The 2.2.0 plan suggested lowering the queue's build concurrency to 2–3.
  Whether it was is a repository setting, which this record cannot show.

### 4. The fuzz job did its job, and nobody was watching it

- **It found a real bug.** W22-06 (#78) bounded the 4 GB string case and kept
  `open_validate` off the global heap that HDF5 cannot leave. The next
  scheduled run (Tuesday 20 October, started about five hours after its cron
  time) found something new:
  - an `i64` overflow in `mask_problems` on finite box bounds such as ±1e300;
  - present since 2.0 (`layout.rs:402` at v2.0.3 and at v2.1.0);
  - fixed in #85 with `saturating_sub` and
    `s7_4_a_box_past_the_integers_is_a_finding_not_a_panic`.
- **How serious it was.** Release builds have no overflow checks, so the
  shipped wheels and binaries wrapped the subtraction. They reported the file
  as invalid, correctly, with a wrong number in the E405 message. A build with
  overflow checks panicked. That is why no backport is needed.
- **Only luck found it in time.** The fuzz job is not part of `ci-ok`, and a
  scheduled failure notifies no one in the project. The release session found
  the red run because it went looking.
- **Red had become normal.** Scheduled fuzz was red on 16 October (the known
  HDF5 cases, before #78), and the pull-request smoke runs on #75 and #77 were
  red too.
- **The global-heap hang is excluded only in the fuzz target.** The engine
  still hangs on such a file, so a user who opens it is not protected.

### 5. Things days old were found on release day

- **Stale documentation**, which no test held to the code:
  - `CLAUDE.md` still said format 1.1;
  - `SECURITY.md` still named nibabel as a dependency;
  - the PyPI project links lacked the 1.2 specification;
  - a changelog entry still said NIfTI "moves onto it next".
- **The lock refresh.**
  - The weekly lock-refresh pull request #83 opened on 19 October and failed
    `cargo deny` on the Unicode-3.0 licence.
  - Nobody looked for two days, because the coordinator's sweep covered lane
    pull requests, not the bot's.
  - At the release, the maintainer allowed the licence. #85 took
    unicode-normalization 0.1.26 and changed `deny.toml`, the wheel's licence
    expression, the Homebrew formula and `THIRD_PARTY_NOTICES`.
  - #83 was then out of date, and was closed unmerged.

### 6. Sessions cannot do the repository's housekeeping

- A session can push only its own branch, so it cannot delete one. The
  maintainer deleted `w22/resample-th8hex` and `deps/cargo-lock` by hand.
- Branches carried a random suffix, not the plan's names.
- Lanes B, E and F reused one branch for successive pull requests, despite the
  plan's one branch per pull request. Lane C used its three.
- Read the Docs is outside the environment's network policy, so the maintainer
  checked the published versions from a screenshot.

### 7. Decisions made mid-lane cost work; measured studies saved it

- **The Intel decision.** The maintainer decided to drop Intel Macs (E12)
  about two hours after #70, the Intel cross-build, had merged. #79 then
  removed what #70 had added.
- **The B-spline study.** The plan's sampled-field approach to B-splines missed
  SimpleITK by **32.6 mm**. Lane B measured that before building on it. It
  stored ITK's coefficients as the existing `bspline` kind instead, which
  agrees to 1.8e-13 mm.
- **Quick decisions.** Decision cards that came early cost little:
  - "Publish both" HDF5 workarounds (lane A);
  - extension skip, which kept 2.1.0's behaviour (lane C).

### 8. Free-threading surfaced old bugs, and its cost was never measured

- **A gate hang.** Lane D's gate hung under threads. Python's after-fork hooks
  let a reader block while holding the GIL.
  - The fork barrier moved into `pthread_atfork` (`medh5::h5::fork`).
  - The HDF5 lock's waiters are now cleared in a forked child.
- **A rare segfault.** A C-library test segfaulted 24 times in 1,500 runs, on
  `main` too. It was fixed.
- **Threads were slower than one.**
  - Preflight took **3.6 s on one thread and 6.9 s on four**.
  - HDF5 has one global lock, so no speedup was expected. Being twice as slow
    was not expected.
  - Nobody measured, against 2.1.0, what detaching on every binding call and
    the live views' locks cost a single thread.

### 9. Performance was not measured at all

- No pull request in 2.2.0 records a timing or a peak memory. That includes
  #76, the free-threading change, and #75/#77, where NIfTI moved from nibabel
  into the engine.
- "The numbers" in `docs/guides/performance.md` were measured with 2.0.0.
- No benchmark runs in CI. `medh5 bench` and the `bench_*.py` examples exist,
  and nobody ran them.
- Known memory costs left as they were:
  - `valid_mask` and carried annotations are built whole, because `add_mask`
    and `write_annotation` take whole arrays;
  - cubic resampling prefilters the whole source (by design, as SciPy does);
  - `from_nifti` holds every input volume until the write begins;
  - a NIfTI image is held twice during decode.

### 10. Waking worked; approvals and losses still slipped

- **What worked.** The coordinator's hourly sweep, new in 2.2.0, kept lane A
  moving through its waits.
- **Approvals.**
  - A coordinator relay to lane D wrongly said the maintainer had reviewed.
  - Lane D's last message still hedged "if the repo requires an approval".
    Approvals are not required.
- **Lost work and space.**
  - Workers were lost mid-turn twice (B, C).
  - Lane B's gate ran out of disk.

### 11. The release itself went smoothly

- #85 opened at 21:15 and merged at 23:35, with every job of the 47-job release
  run green.
- It published 13 files to PyPI, along with crates.io, Homebrew and GitHub
  Releases.
- No changelog sentence promised a later release, and the maintainer
  confirmed that.

## What changes for 2.2.x

The [2.2.x plan](README.md) takes these up:

- **Hostile input first** (§4):
  - sweep the overflow class across the engine, not one site;
  - stop the global-heap hang in the engine, not only in the fuzz target;
  - make a red fuzz run visible.
- **Measure before changing performance** (§8, §9): 2.1.0 against 2.2.0, one
  thread against four, and peak memory. Then fix the slow paths found, and
  refresh the published numbers.
- **Tests for the stale documentation** (§5): `CLAUDE.md`'s version line,
  `SECURITY.md`'s dependencies and the PyPI links fail CI when they drift.
- **Process** (§1, §2, §6):
  - at most three lanes at once;
  - work committed and pushed before a turn ends;
  - a Codex finding swept as a class, with one push per round;
  - bot pull requests and scheduled fuzz in the coordinator's sweep;
  - housekeeping that only the maintainer can do, listed once.
