# Clinical history beside the images

Format 1.1 lets a sample carry the subject's clinical history next to the images: laboratory values,
reports and their revisions, diagnoses, medications, assessments — each with **when it happened**
and **when it became known**. This page shows how to write that history, read it back, ask what was
known at a moment, and add it to files you already have. The rules are [MEDH5 1.1](../spec/medh5-1.1.md);
the cohort used below is the one
[`docs/examples/clinical_longitudinal.py`](../examples/clinical_longitudinal.py) writes. Write it
once, from a checkout of the repository:

```bash
python docs/examples/clinical_longitudinal.py cohort
```

That writes six subjects, `cohort/P-01.medh5` to `cohort/P-06.medh5`, and the task
[Train on clinical tasks](clinical-training.md) loads, `cohort/progression.task.json`, which pins
each file's `content_id`. The script ends by loading a training batch, so it needs PyTorch
(`pip install "medh5[torch]"`). The examples below amend only copies, so the cohort stays as the
task pins it.

Nothing changes for imaging-only data: a sample without clinical records is still written as 1.0,
or as 1.2 if it holds open curves.

## Write a history

Times are signed microseconds on one **subject clock**, and every time is a pair of inclusive bounds:
an exact instant is one integer, a date known to the day is the day's bounds, and an unknown time is
`None` — never a guess. Declare the clock first; then add events, documents and the links between
them and the imaging objects.

```python
import numpy as np
import medh5
from medh5.clinical import DAY, HOUR, Clock, Document, Event, Link

with medh5.create("history.medh5", sample_id="P-07", subject_id="P-07") as w:
    w.add_grid("ct", shape=(8, 16, 16), spacing=(2.0, 0.8, 0.8))
    w.add_image("CT", np.zeros((8, 16, 16), np.int16), grid="ct", modality="CT")

    w.set_clock(Clock.relative("clock", "acquisition start of the baseline CT"))
    # The scan, available an hour after it was acquired.
    w.add_event(Event("ct0", "ct0", "imaging", "point", "final",
                      effective_start_us=0, available_us=HOUR, timepoint_id="tp0"))
    w.add_link(Link.between(("event", "ct0"), "describes", ("image", "CT")))
    # A creatinine drawn two days before the scan: no imaging timepoint needed.
    w.add_event(Event("lab0", "lab0", "observation", "point", "final",
                      effective_start_us=-2 * DAY, available_us=-2 * DAY + 3 * HOUR,
                      code_system="http://loinc.org", code="2160-0",
                      value_num=1.1, unit="mg/dL"))
    # A report: the event owns its timing, the document its text.
    w.add_event(Event("rep_v1", "rep", "document", "point", "preliminary",
                      effective_start_us=0, available_us=4 * HOUR))
    w.add_document(Document("rep_text_v1", "Preliminary read: 14 mm nodule."))
    w.add_link(Link.between(("event", "rep_v1"), "describes", ("document", "rep_text_v1")))

with medh5.open("history.medh5") as s:
    assert s.version == "1.1" and "clinical" in s.profiles
    assert [e.event_id for e in s.clinical.events] == ["lab0", "ct0", "rep_v1"]
```

The writer checks every record as it is added and the whole history at `commit`, as it does for
images: a `point` without a start, an observation with a value but no code, a link to an image that
does not exist — each is refused with its diagnostic code rather than written.

A few rules carry most of the meaning:

- **A version is immutable.** A revised report is a new event with the same `record_id` and a new
  document, linked to the old version by `supersedes` — never an edit of the old row. A measurement
  corrected later, or an interval whose end was learned later, is a new version too. So is what a
  version attests: an amendment refuses to give a version the file already held a payload or a
  relationship it did not have, or to replace one it did (`E809`) — a grounding drawn later is
  asserted by a new event, available when it was drawn, and a corrected mask is a new annotation.
- **Unknown stays unknown.** Leave `available_us` as `None` when the source does not say; never
  substitute the time the data was ingested. Strict selection will then never use the event, which is
  the honest outcome.
- **A missing result is not a negative.** Record an expected result that is absent with
  `missing_reason`, and no value.
- **Clinical codes are strings.** They never share the segmentation label set's `uint16` class ids,
  however many concepts a cohort has.

## Revise a record

```python
import shutil

from medh5.clinical import HOUR, Document, Event, Link

shutil.copyfile("cohort/P-03.medh5", "P-03-revised.medh5")   # the task pins the original
with medh5.amend("P-03-revised.medh5") as w:
    w.add_event(Event("report0_v3", "report0", "document", "point", "amended",
                      effective_start_us=0, available_us=72 * HOUR))
    w.add_document(Document("report0_text_v3", "Addendum: nodule unchanged on review."))
    w.add_link(Link.between(("event", "report0_v3"), "describes",
                            ("document", "report0_text_v3")))
    w.add_link(Link.between(("event", "report0_v3"), "supersedes", ("event", "report0_v2")))
```

The amendment is copy-on-write like any other: images and annotations are copied as stored, their
digests unchanged, and the sample gets a new `content_id` because its content changed.

## Ask what was known at a moment

`select` answers with the engine's strict prospective rules (1.1 §9): only versions known available
by the cutoff, the newest of each record, and only the images, documents and annotations those
versions attest.

```python
from medh5.clinical import HOUR

with medh5.open("cohort/P-03.medh5") as s:
    at_day_one = s.clinical.select(24 * HOUR)
    assert at_day_one.certified
    assert at_day_one.event_ids == ["lab0", "ct0", "report0_v1"]   # not v2, not ct1
    assert at_day_one.admits("document", "report0_text_v1")
    assert not at_day_one.admits("image", "CT_tp1")
    text = s.clinical.text("report0_text_v1")      # read when asked for
```

A later revision whose availability is unknown — or straddles the cutoff — makes the selection
**uncertifiable** rather than silently handing back the older version: whether the older version was
still current at the cutoff is exactly what is unknown. `latest_provable` is the one named
alternative, and says so in its status:

```python
from medh5.clinical import HOUR

with medh5.open("cohort/P-03.medh5") as s:
    provable = s.clinical.select(24 * HOUR, policy="latest_provable")
    assert provable.status == "provable"
```

A **policy** narrows what a selection reads: a context window before the cutoff, the event kinds
admitted, plans, a limit on events ([task-cache-1](../spec/task-cache-1.md) §3.4). The window
bounds each event's order time from below; one day before the cutoff leaves out the creatinine
drawn two days before the scan:

```python
from medh5.clinical import HOUR, SelectionPolicy

with medh5.open("cohort/P-03.medh5") as s:
    last_day = s.clinical.select(24 * HOUR, SelectionPolicy(context_us=24 * HOUR))
    assert last_day.event_ids == ["ct0", "report0_v1"]      # lab0, at -48 h, is outside
    assert last_day.excluded == {"after_cutoff": 6, "outside_context": 1}
```

A **plan** — an event `planned`, or one whose start is not definitely at or before the cutoff —
is left out unless the policy admits plans, and then it is marked as one, never read as an
outcome. `select` takes records not read from a file, too:

```python
from medh5.clinical import DAY, HOUR, Event, SelectionPolicy, select

order = Event("ct1_order", "ct1_order", "procedure", "point", "planned",
              effective_start_us=90 * DAY, available_us=2 * HOUR)
assert select([order], [], 24 * HOUR).excluded == {"plan": 1}
with_plans = select([order], [], 24 * HOUR, SelectionPolicy(plans=True))
assert with_plans.event_ids == ["ct1_order"] and with_plans.events[0].plan
```

A task carries its policy for every row (`TaskManifest.new(..., policy=...)`). The same answers
from the command line, where `--policy-file` takes a whole policy as JSON:

```bash
medh5 clinical show cohort/P-03.medh5
medh5 clinical select cohort/P-03.medh5 --cutoff-hours 24
medh5 clinical select cohort/P-03.medh5 --cutoff-hours 24 --context-us 86400000000
```

## Add a history to a 1.0 file

`augment` copies a sample's objects as stored — their digests do not change — adds the records,
and writes a 1.1 sample with a new identity. It reports what the records leave unknown, and refuses
a `clinical` group it did not write rather than reinterpret it.

```python
import medh5.clinical as clinical
from medh5.clinical import HOUR, Clock, ClinicalRecords, Event, Link

records = ClinicalRecords(
    Clock.relative("clock", "acquisition start of CT_tp0"),
    events=(Event("ct0", "ct0", "imaging", "point", "final",
                  effective_start_us=0, available_us=HOUR, timepoint_id="tp0"),),
    links=(Link.between(("event", "ct0"), "describes", ("image", "CT_tp0")),),
)
report = clinical.augment("case.medh5", records, out="case-1.1.medh5")
assert report["version_after"] == "1.1" and report["unchanged_digests"] > 0
```

Visit dates you already have can seed day-precision imaging events; their availability stays
unknown, and the notes say so:

```python
import medh5.clinical as clinical

events, links, notes = clinical.imaging_events_from_timepoints("case.medh5")
clock = clinical.baseline_day_clock("clock")
```

The command line takes the same records as a logical-record bundle — the JSON
`ClinicalRecords.to_json()` returns and `medh5 clinical export` writes, checked against
`medh5-clinical-1.schema.json`:

```bash
medh5 clinical export cohort/P-03.medh5 --out P-03-records.json
medh5 clinical augment case.medh5 case-records.json --out case-1.1.medh5
```

## Remove it again

Removing the clinical profile is the imaging projection (1.1 §2.1, §10), and only ever a requested
operation: the result is a different sample, with its own `content_id`, written at the lowest
version what remains needs — 1.0 for imaging alone, 1.2 with open curves. `medh5 clinical strip`
writes it to a new file and reports the loss:

```bash
medh5 clinical strip cohort/P-03.medh5 --out P-03-imaging.medh5
```

`drop_clinical()` in an amendment writes it in place of its source — here a copy, since the
training task pins the original:

```python
import shutil

shutil.copyfile("cohort/P-03.medh5", "P-03-dropped.medh5")
with medh5.amend("P-03-dropped.medh5") as w:
    w.drop_clinical()

with medh5.open("P-03-dropped.medh5") as s:
    assert s.version == "1.0"
```

## Share it de-identified

De-identifying a history is another derivation (1.1 §10). `medh5 scrub --apply` moves every time on
a `utc` clock by the one shift `/meta`'s dates take — effective and available bounds alike, so a
cutoff moved by the shift admits the same versions — makes the clock `shifted_utc`, and re-mints the
clock, event, record, document and encounter ids from the salt, so a subject's fragments scrubbed
with one salt still share a clock. The text keeps its bytes, and so every span; it is
de-identified by a tool that handles free text, and attested with `--attest-text`:

<!-- illustrative -->
```bash
medh5 scrub cohort/P-03.medh5 --apply --salt "$SALT" --date-shift-range=-400:-100 \
    --attest-text "philter 2.1" --out shared/P-03.medh5
```

[De-identify and publish](deidentify.md#4-a-clinical-sample) has the rules it applies and what
blocks it.

## Check it

```bash
medh5 validate cohort/P-03.medh5 --level integrity
```

Clinical tables are checked like everything else: the column encoding, the records, and — at
`integrity` — every clinical dataset's digest, recomputed from the bytes, so an edit to a report
is found even when the stored `content_id` was left alone. A file of a **later** minor version than
this package implements is read as a projection: what is known is validated, what is not is reported
as `W913`, and the file is never amended.

## Related

- **[Train on clinical tasks](clinical-training.md)** — a task over this cohort, its preflight and
  its batches.
- **[Python API](../reference/python-api.md#clinical-history-format-11)** — the clinical records,
  `select` and the writer's methods.
- **[MEDH5 1.1](../spec/medh5-1.1.md)** — the profile, and strict selection (§9).
