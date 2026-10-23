# Runnable examples

Eleven scripts. Four are standalone: the benchmarks behind the measured claims
in the [specification](../spec/medh5-1.0.md) (§7.0, §14.2, §14.3, §14.5) and the
[design rationale](../explanation/design-rationale.md), and a reference writer
that builds a complete file from the specification alone. None of those imports
`medh5` — they measure and exercise the *format*, with h5py and nothing between
the reader and the bytes.

The other seven run through the package's public API, end to end, and the test
suite runs each of them. Five exercise [format 1.1](../spec/medh5-1.1.md) and
the [task and cache contract](../spec/task-cache-1.md)
(`tests/integrations/test_clinical_training.py`); the sixth times reading from
threads, and the seventh where a native command's time goes
(`tests/tools/test_bench.py`):

| Script | Does |
|---|---|
| [`clinical_longitudinal.py`](clinical_longitudinal.py) | The worked example of 1.1 §9.3 as a cohort: two CT visits, an intervening lab, a report and its revision, follow-up responses. Writes, reopens and validates every file, preflights a progression task, and prints a training batch whose rows see different visits, missing modalities and histories of different lengths |
| [`clinical_collection.py`](clinical_collection.py) | A subject split across two members of a `.medh5c` shard: refused until its duplicated event is reconciled, then trained on through worker processes |
| [`clinical_cache.py`](clinical_cache.py) | Event- and patient-level feature caches, then every way one stops being usable — a source amended (stale), bytes rotted (corrupt), a whole-history embedding (inadmissible), statistics fitted on the wrong split — and the explicit re-pin |
| [`bench_clinical.py`](bench_clinical.py) | Write, select, read, batch, cache and amend costs on a synthetic cohort ([results below](#clinical-paths-format-11)) |
| [`bench_preflight.py`](bench_preflight.py) | A task's preflight, dataset and items on a large multi-site cohort — many events, documents, fragments and cutoffs — each measured in its own interpreter: time, peak memory, bytes read ([results below](#preflight-at-cohort-scale)) |
| [`bench_threads.py`](bench_threads.py) | Image windows (Blosc2 and gzip), annotation windows and opens, the same work from one, two and four threads with a handle each, and the cost of the smallest engine call ([results](../guides/performance.md#the-numbers)) |
| [`bench_phases.py`](bench_phases.py) | `resample`, `convert from-nifti`, an amendment, `recompress`, `verify` and `validate` on a 512×512×256 sample, through a native CLI built with phase timings: the seconds inside HDF5's lock and outside it, and the most four threads could give ([results](../guides/performance.md#where-a-commands-time-goes)) |

```bash
pip install "medh5[torch,h5py]"   # h5py: clinical_cache.py rots a cache's bytes with it
python docs/examples/clinical_longitudinal.py out/   # ~5 s
python docs/examples/clinical_collection.py out2/    # ~5 s
python docs/examples/clinical_cache.py out3/         # ~5 s
python docs/examples/bench_clinical.py --json bench.json   # ~1 min at the defaults
python docs/examples/bench_preflight.py --json pre.json    # ~1 min at the defaults
python docs/examples/bench_threads.py --json threads.json  # ~3 min at the defaults
# A native CLI built with the timings first (a cfg, not a Cargo feature):
RUSTFLAGS="--cfg medh5_phase_timings" CARGO_TARGET_DIR=target/timings \
    cargo build --release -p medh5-cli
python docs/examples/bench_phases.py --medh5 target/timings/release/medh5  # ~15 min
```

| Script | Produces |
|---|---|
| [`bench_encodings.py`](bench_encodings.py) | Multi-label voxel encodings on a 160³, 200-class phantom: size, write time, all-class patch read (spec §7.0) |
| [`bench_query.py`](bench_query.py) | Codec-matched single-class versus all-class patch reads across per-class, `layers` and `bitmask` (spec §7.0) |
| [`bench_io.py`](bench_io.py) | Codec profiles, `int16 + rescale` versus `float32`, and the foreground index versus `argwhere` (spec §14.2, §14.3) |
| [`reference_writer.py`](reference_writer.py) | A complete MEDH5 1.0 file exercising `core+seg+det+cls+reg+curation+training+longitudinal` — one subject at two timepoints — then validates `/meta` against the JSON Schema and runs the §15 semantic and integrity checks |

```bash
pip install numpy h5py hdf5plugin jsonschema
cd docs/examples
python bench_encodings.py     # ~15 s; run first, bench_io.py reads its files
python bench_query.py         # ~30 s
python bench_io.py            # ~20 s
python reference_writer.py    # ~10 s, writes case_0001.medh5 to the current directory
```

The benchmarks write their scratch `*.h5` files and JSON results beside
themselves; those are ignored by git and by the documentation build, as is the
`case_0001.medh5` the reference writer leaves here.

## Recorded results

macOS on Apple silicon, Python 3.12, h5py 3.16, hdf5plugin 6.0, local SSD,
medians over 10–50 repetitions. Absolute timings are hardware-dependent; the
ratios between encodings and codecs are what the decisions rest on.

### Multi-label encodings

160³ phantom, 200 classes (24 mutually exclusive organs and 176 overlapping
structures, 0.25 labels per voxel), greedy colouring `L = 5` layers, `P = 4`
bitmask planes:

| Codec | Encoding | Size | One-class 64³ read | All-class 64³ read |
|---|---|---|---|---|
| lz4 L1 + shuffle | per-class `bool` (0.x) | 3.57 MiB | 0.51 ms | 116.89 ms |
| lz4 L1 + shuffle | **`layers`** | **0.55 MiB** | **0.09 ms** | **6.69 ms** |
| lz4 L1 + shuffle | `bitmask` | 0.68 MiB | 2.79 ms | 10.14 ms |
| zstd L5 + bitshuffle | per-class `bool` (0.x) | 3.01 MiB | 0.33 ms | 139.27 ms |
| zstd L5 + bitshuffle | **`layers`** | **0.20 MiB** | 0.18 ms | 13.48 ms |
| zstd L5 + bitshuffle | `bitmask` | **0.15 MiB** | 10.31 ms | 37.06 ms |

`instances` on the same data is 0.08 MiB — 45× smaller than per-class dense,
because its cost tracks object volume rather than image volume.

### Codec profiles

192×256×256 `int16` CT, 32×64×64 chunks, read with h5py, a new 64³ window
each read:

| Codec | Write | Size | Ratio | 64³ read | Full read |
|---|---|---|---|---|---|
| lz4 L1 (`training`) | 0.05 s | 12.80 MiB | 1.9× | 1.0 ms | 0.02 s |
| zstd L3 (`balanced`) | 0.09 s | 9.16 MiB | 2.6× | 2.3 ms | 0.03 s |
| lz4hc L8 (the 0.x default) | 0.48 s | 12.33 MiB | 1.9× | 0.8 ms | 0.02 s |
| zstd L9 + bitshuffle (`archive`) | 3.36 s | 9.53 MiB | 2.5× | 1.6 ms | 0.02 s |
| gzip L4 (`portable`) | 0.42 s | 9.72 MiB | 2.5× | 8.6 ms | 0.08 s |

The same volume stored as `float32` rather than `int16` plus a rescale is
36.75 MiB against 12.33 MiB at lz4hc L8 — three times the disk for no
information, which is why spec §4.2 recommends `int16` HU and `W907` flags the
alternative.

### Foreground sampling

160³, one class, 33 533 foreground voxels:

| Path | Time | Resident memory |
|---|---|---|
| 0.x: full mask and `np.argwhere`, cached per file and class | 9.2 ms | 0.8 MiB, O(volume) |
| 1.0: read a 4096-coordinate index | **0.52 ms** | **48 KiB**, O(1) |

The package's own sampler now reads one coordinate per draw rather than the
whole subsample: see [Tune performance](../guides/performance.md#the-numbers)
for its current numbers, and `medh5 bench` to reproduce them.

### Access pattern

`d[k][roi]` against `d[(k, *roi)]` on a 160³ `uint64` bit plane: **32.8 ms
against 0.8 ms**. The first form materialises the whole plane before slicing,
which is why spec §14.5 requires one-call slicing of the reference reader.

### Clinical paths (format 1.1)

`bench_clinical.py` at its defaults: 24 subjects, each with two `64×128×128`
`int16` CT visits (stored in `64×32×32` chunks, `training` codec), an MR at
follow-up for every second subject, 500 coded laboratory values over four
years, a report and its revision (≈2 KiB of text each) and two responses; a
task of 48 rows (two cutoffs per subject) with a CT slot of `32³` and an MR slot
of `16³`. Linux x86_64 container, 4 vCPUs, Python 3.13, NumPy 2.5, local disk;
medians of 20 repetitions where a measurement repeats. One machine, a
synthetic cohort: these validate the path and show where its costs are, and are
not clinical-scale performance.

| Measurement | Result |
|---|---|
| Write one sample, 1.1 with its history / 1.0 without | 116 ms / 87 ms |
| Bytes per sample, 1.1 / 1.0 | 2.88 MB / 2.76 MB (+4 %) |
| Open, read the clinical tables, select at a cutoff (cold) | 4.6 ms |
| Select again on the open sample (507 events) | 1.5 ms |
| One document's text | 0.02 ms |
| A `32³` slot window: from the 1.1 file / from its 1.0 imaging projection | 0.06 ms / 0.08 ms |
| Preflight of the whole task (24 sources, 48 rows; every pin and clinical digest re-verified) | 0.86 s |
| Build / validate the event-level report cache (48 entries) | 0.47 s / 0.27 s |
| Batches of 8, documents encoded on the fly: 0 / 2 workers | 171 / 158 rows/s |
| Batches of 8, documents from the cache: 0 / 2 workers | 505 / 187 rows/s |
| Amend one sample by one event (copy-on-write of the whole file) | 54 ms |
| Peak resident memory of the measuring process | 604 MiB |

The imaging path is unchanged: a window reads the same chunks from a 1.1 file
as from its 1.0 projection, and opening a sample reads no clinical table until
`Sample.clinical` is asked for. A history adds a few percent to a sample whose
images dominate it. Selection is metadata-only and milliseconds per subject;
preflight is dominated by re-verifying every pinned sample's clinical digests,
which is what lets it detect an edit under an unchanged stored root. At this
size two workers do not beat one process — each worker pays its own opens for
48 small rows — so measure your own shard and worker layout before choosing.

### Preflight at cohort scale

`bench_preflight.py` builds a two-site cohort: each subject split across two
files on one clock, with coded laboratory values (one in ten revised later, one
in a hundred of unknown availability), medication orders and courses,
day-precision diagnoses, static demographics held by both fragments (so
preflight reconciles them), notes of ≈2 KiB each with every tenth revised, a
small CT per visit and an MR at the second site; rows monthly from the first
visit, a CT and an MR slot, a progression target and a one-year context window.
Each measurement runs in its own interpreter, so its peak resident size and the
bytes it read (`rchar`) are its own. Same container as above. "Before" is the
first 1.1 implementation (commit `eeb8c33`), on the same files.

| 60 subjects × 2 fragments, 2 000 events and 100 notes each; 720 rows (43 MiB on disk) | Before | After |
|---|---|---|
| `task.preflight()`, end to end | 45–61 s (three runs) | 3.4 s |
| Peak resident memory of that process | 5 158 MiB | 131 MiB |
| Bytes read by it | 114 MiB | 78 MiB |
| The engine alone (`medh5 task preflight`, native) | 10.0 s | 3.7–4.2 s |
| `ClinicalTaskDataset` pickled, as each spawned worker receives it | 79 MiB, in 10–11 s | 34 MiB, in 0.06 s |
| Items per second, ≈520 events each, notes encoded on the fly | 24 | 41 |

| 200 subjects × 2 fragments, 3 000 events and 150 notes each; 4 800 rows (201 MiB on disk) | After |
|---|---|
| `task.preflight()`, end to end | 18.8 s |
| Peak resident memory of that process | 481 MiB |
| Bytes read by it | 374 MiB |
| `ClinicalTaskDataset` built / pickled | 1.5 s / 219 MiB in 0.38 s |
| Items per second, ≈785 events each, notes encoded on the fly | 30 |

The native command line on the same 200 subjects, before and after it took the
subjects one at a time too (commit `3921e74` → now):

| `medh5 task preflight`, 200 subjects | Before | After |
|---|---|---|
| The table | 18.6 s, 742 MiB | 13.5 s, 34 MiB |
| `--json` (1.6 GB written) | 130 s, 12.3 GB | 70 s, 315 MiB |

Where the time went, and what changed: rows had copied their subject's event
records (every row of every subject, through JSON, into Python objects — 5 GB
for 720 rows), every row rebuilt the revision chains, every event was digested
to find the few held twice, each source's tables were read twice and its text
decompressed three times. Now a subject's history is prepared once and shared by
its rows as indices, only duplicates are digested, a source is walked once and
its tables read once, text is read only to verify it, and the result crosses
into Python as columns built a subject at a time. What remains is the guarantee
itself (a profile of the engine, by instructions): checking each pin — one
walk of the file and a hash of every clinical byte — is ≈40 % of the work,
validating each source's clinical tables ≈30 %, opening the sources ≈8 %, and
selecting at every cutoff and building every row ≈12 %. The hash is SHA-256 in
software here (this CPU has no SHA extensions; the `sha2` crate uses them where
they exist), ≈14 % on its own. Reading the clinical tables twice — hashed,
then validated — is not where the time goes: every HDF5 read with its
decompression, in both passes, is 6.5 %, so one pass would save 3–4 % and tie
the digest code to clinical parsing. Threads do not help — every HDF5 call
holds one lock; four threads ran slower than one
([why](../explanation/concurrency.md)). Items with notes are bound by
the deliberately naive `HashingTextEncoder`: without documents the same items
build at ≈130 per second.

## The reference writer

`reference_writer.py` is the executable proof that the specification is
self-consistent: it follows the spec literally, with `h5py` and no `medh5`, and
its output passes JSON Schema validation, cross-reference checks (E1xx–E6xx),
per-object digests, `content_id` and `source_digest` (§13), plus reader-side
round trips for the affine, the box↔slice convention, instance mask decoding and
lossless `layers ↔ bitmask` transcoding. Each check raises when it fails. CI
runs it on every push, then validates the file it wrote at integrity level with
the wheel and the native binary — its own checks are its own reading of the
text, and for a while that reading computed a `content_id` no implementation
agreed with.

The sample it writes is longitudinal: baseline CT and PET sharing one frame of
reference, a follow-up CT on its own grid with shorter z coverage and its own
frame, organ and lesion annotations at both visits, a RECIST response label
spanning them, and the registration relating the two. It therefore also
exercises the timepoint rules (§3.7 — E106/E107/E108), cross-timepoint instance
tracking (§7.4 — persisted / resolved / new), change labels (§9 — E409) and the
longitudinal warnings W909–W911.

> **Note.** Reading a Blosc2-compressed MEDH5 file requires `import hdf5plugin`
> before the read, or HDF5 raises a confusing plugin-path error rather than a
> missing-filter error. This is the reason the spec defines the `portable` codec
> profile (§14.2).
