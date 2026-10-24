# Tune performance

Four levers decide how fast a training loop reads: the **sampling index**, the
**chunk shape**, the **codec profile** and the **worker count**. This page is
what each is worth, how to tell which one you are missing, and how to reproduce
the numbers on your own hardware.

The short version, in payoff order:

| Lever | Worth | Do it when |
|---|---|---|
| **Build a sampling index** | foreground sampling goes from O(volume) to O(1) — 210 ms → 0.03 ms on a 12.6 Mvox volume, 2.4 s → 0.03 ms at 512³ | always, unless you only sample uniformly |
| **Set `patch_hint` on the grid** | sizes chunks to what you will actually read | at write time, if you know your patch size |
| **`--profile training`** | decompresses fastest | the cohort is read far more often than written |
| **`num_workers > 0`** | ~590 → 880–980 patches/s (64³) with four workers on four cores | always, with `worker_init_fn` |
| **`FileGroupedSampler`** | one open per file per epoch instead of one per item | shuffled training over more files than the handle cache holds |

## Build the index first

This is the one that changes the complexity class rather than the constant.

```bash
medh5 index build cohort/*.medh5 --max-coords 4096
```

Per voxel annotation, per class, it stores a bounded sample of foreground
coordinates (at most `--max-coords`, which must be at least 1), the exact voxel
count and a tight bounding box. Foreground patch sampling becomes a lookup
instead of a scan, and `medh5 dataset stats` gets its class counts for a few
hundred bytes rather than a decompression pass.

**How to tell you are missing it.** Nothing fails — the sampler falls back to
scanning the labels and records what it did:

```python
item["meta"]["patch"]["used_index"]     # True, False, or None
batch["meta"]["patch"]["used_index"]    # after collate: a list, one per item
```

Three states, because there are three answers. `True` means the index answered
the foreground query; `False` means the labels had to be scanned instead,
because the annotation has no current index or its index lacks a class the draw
could pick; **`None` means the question did not arise** — a uniform draw
consults no index at all. A class in `foreground_classes` that an annotation
cannot express is not one the draw could pick: it has no voxels there, so one
sampler configured for a whole cohort keeps every file's index.

`False` is the one that costs you. Count it per batch:

```python
scanned = sum(used is False for used in batch["meta"]["patch"]["used_index"])
```

An index carries the digest of the annotation it derives from, so it goes
**stale** when the annotation changes. Readers must then ignore it, and the
validator raises `W905`:

```bash
medh5 fix cohort/*.medh5 --rebuild-index
```

A stale index is not a file error. It is a cache that needs rebuilding, and the
format says so rather than making the file invalid.

## Size the chunks to the patch

The chunk is the unit a read fetches: reading one voxel reads a whole stored
chunk. Blosc2 compresses a chunk as a grid of blocks, and under the Blosc2
profiles medh5 decompresses only the blocks a window covers, so a chunk larger
than the patch costs I/O rather than decompression; under `portable` it costs
both. Two forces pull against each other — sizing to the L3 cache keeps a patch
in cache after decompression, sizing to the training patch keeps read
amplification low — and the optimiser resolves them by starting at the patch,
growing toward the cache budget, and stopping before the chunk is much larger
than the patch.

<!-- illustrative -->
```python
w.add_grid("ct", shape=..., spacing=..., patch_hint=(96, 96, 96))
```

`patch_hint` is how you say what you will read; without one the optimiser
assumes a 64-voxel patch on each axis. The cache budget is 80 % of one CPU's
share of the L3 (its size over the CPUs sharing it), or of ~1.375 MiB where
the platform reports none, held between 512 KiB and 4 MiB. A chunk starts at
the patch, each axis rounded up to a power of two, and no computed chunk is
larger than 4 MiB: a patch whose rounding would pass that starts at the patch
itself, halved until it fits.

For an existing cohort, `--rechunk` re-derives the chunk shape as well as the
codec, by the writer's own rule — from each grid's `patch_hint`, one plane per
chunk for `layers`, `bitmask` and `probmap` — so a re-chunked file is chunked
the way a fresh write would be:

```bash
medh5 recompress cohort/*.medh5 --profile training --rechunk
```

(Before 1.4.2 it let h5py choose, and h5py's guess spanned the stacked axis the
spec keeps at 1, which the validator reported as `W902`.)

## Choose a codec profile

Storage is a training parameter. `training` decompresses fastest; `archive` is
smallest. The full table is in [Storage](../reference/storage.md#codec-profiles).

```bash
medh5 recompress cohort/*.medh5 --profile training
```

Every stored byte changes and no `content_id` does — the digest is over content,
not over its encoding — so a recompressed cohort is still verifiably the same
data.

## Use workers

```python
loader = DataLoader(dataset, batch_size=2, num_workers=8,
                    worker_init_fn=worker_init_fn, collate_fn=collate)
```

`worker_init_fn` drops handles inherited across a `fork`. It is recommended
rather than required — the cache is PID-keyed and resets on first use in a
forked worker either way. See [PyTorch and MONAI](../reference/torch.md#the-dataloader).

## Check a batch in several processes

[Threads, processes and open files](../explanation/concurrency.md) says when
threads help and when processes do.

One lock covers the HDF5 library in a process, so files checked in one process
take turns however many threads ask. The batch commands take `--workers N` and
run their work in up to *N* processes:

```
$ medh5 verify --workers 4 data/*.medh5
```

The output — the text, the `--json` document and the exit code — is the
one-process run's, byte for byte, printed in input order. Each command's unit
is the work it can do without looking at another: a file for `validate`,
`verify`, `fix`, `index build` and `recompress`; a subject's 0.x files for
`migrate`; a case for `convert to-nnunet` and `conformance run`; a subject for
`clinical augment-cohort` and `task preflight`. What a run shares — the
cohort's label set, the dataset's case names, the one pass over a MEDS dataset
or a FHIR export — is worked out first, in one process, so it is not repeated
per worker, and is the part `--workers` does not shorten. Over 32 files of
96×192×192 (two per subject, and a clinical cohort of 32 subjects) on four
cores, the median of three runs:

| Command | Unit | 1 process | 4 workers | console script: 1 | 4 |
|---|---|---|---|---|---|
| `verify` | file | 3.4 s | 0.9 s | 3.5 s | 1.4 s |
| `index build` | file | 4.6 s | 1.2 s | 4.5 s | 1.6 s |
| `recompress --profile archive` | file | 183 s | 48 s | — | — |
| `migrate` | subject | 10.1 s | 2.8 s | 10.1 s | 3.3 s |
| `convert to-nnunet` | case | 18.7 s | 7.7 s | 21.2 s | 8.6 s |
| `conformance run` | case | 4.0 s | 1.1 s | 4.5 s | 1.7 s |
| `task preflight` | subject | 0.72 s | 0.20 s | 0.93 s | 0.67 s |
| `clinical augment-cohort` | subject | 2.4 s | 1.5 s | 2.4 s | 1.5 s |
| `validate` | file | 0.06 s | 0.04 s | 0.20 s | 0.47 s |

A worker costs a process start, which the Python console script pays as an
interpreter start, about 0.1 s each: `validate`, which reads structure rather
than data, finishes before that pays off, and is slower there with workers.
The default, 1, runs in one process. `scrub`, `splits`, `labels check` and
`dataset check --deep` take no `--workers`: over the same 32 files each takes
milliseconds, and four processes at once, each over a quarter, would be the
most a flag could give:

| Command | 1 process | four at once | console script: 1 | four at once |
|---|---|---|---|---|
| `scrub` | 0.04 s | 0.02 s | 0.19 s | 0.32 s |
| `splits` | 0.02 s | 0.01 s | 0.18 s | 0.26 s |
| `labels check` | 0.02 s | 0.01 s | 0.16 s | 0.26 s |
| `dataset check --deep` | 0.015 s | 0.011 s | 0.17 s | 0.25 s |

`docs/examples/bench_workers.py` builds the cohort and times every command
above, with the native binary (`--medh5 target/release/medh5`) or the console
script.

Within one Python process, the calls that work over large arrays — copying a
mask into the engine's layout, `curation.agreement.dice` and `iou`, the
geometric encoders and `to_world` — release the GIL while they work, so other
Python threads keep running.

## Write a label volume as it is

A segmentation that is already one integer volume — a model's argmax, an
nnU-Net or NIfTI label file — goes to `add_segmentation(labelmap=volume)`,
not through a mask per class. The masks cost the volume's voxels once per
class, and the writer copied each again: at 104 classes of 384³ the masks
route peaked 11.2 GB above the image write, and `labelmap=` nothing above it,
for the same bytes on disk. See
[From a label volume](../reference/annotations.md#from-a-label-volume).

## Keep a file's patches together

Each worker keeps its 32 most recently used files open. `shuffle=True`
permutes *items*, so with `samples_per_volume` items per file and more files
than that, consecutive items rarely share a file and most of them open it
again: at 100 small files and 4 items each, about 295 opens an epoch and
1.1 ms per item.

`FileGroupedSampler` shuffles files and yields each file's items together —
100 opens, one per file, and 0.52 ms per item on the same run:

```python
from medh5.torch import FileGroupedSampler

loader = DataLoader(ds, batch_size=4, sampler=FileGroupedSampler(ds, seed=0),
                    num_workers=8, worker_init_fn=worker_init_fn, collate_fn=collate)
for epoch in range(epochs):
    ds.set_epoch(epoch)     # new patches, and a new file order
```

With workers, make `samples_per_volume` a multiple of `batch_size`: a batch is
read by one worker, so a file whose items straddle two batches is opened by
both. `set_cache_size(n)` is the other lever — raise it toward the number of
files a worker cycles through, lower it when descriptors are scarce. Under DDP
the same sampler gives each rank a contiguous share of the files
(`num_replicas`, `rank`), where `DistributedSampler` would have every rank
open every file. A kept handle also keeps its transforms' cubic coefficients:
a `cubic` displacement field, alone or inside a composite, is read and
prefiltered once per handle, within a 256 MiB budget per process.

Stitching sliding-window predictions back (`GridStitcher`) keeps one
accumulator per file in flight: `channels` volumes of the output `dtype`
(float32 unless you say otherwise) and one of weights. A 512 × 512 × 300
volume with 14 output channels is 15 × 300 MB = 4.4 GB while its windows are
outstanding. Iterate files in order — `shuffle=False`, or `FileGroupedSampler`
— so a file completes, and its accumulator is released, before the next is
begun; a fully shuffled loader keeps every file open until its last window
arrives.

## On a network filesystem

HDF5's file locking is unreliable on NFS, Lustre and GPFS, where training
clusters usually keep their data, and a lock that hangs or fails there looks
like a slow or broken file. Reading is safe without it, so turn it off for the
training job:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

Set it in the job's environment before Python starts, so every `DataLoader`
worker inherits it. medh5 never sets it for you (§14.4): it also disables the
lock that keeps two writers apart, and whether that is safe is a fact about your
cluster, not about the file.

## Starting programs while files are open

A file medh5 opens is kept out of every program the process starts, on every
platform: a child does not inherit it, so it neither keeps the file locked nor
blocks the rename an amendment ends with. Two cases stay open. A program
started by *other* code on another thread in the instant between HDF5's open
and medh5 marking the file can still inherit it; medh5's own children wait for
that instant to pass. And a `fork` without `exec` shares every open file, which
is why a forked `DataLoader` worker drops the handles it inherited rather than
using them (§14.4).

## The numbers

Measured with 2.2.0 and 2.1.0, the released wheels, on one machine: a Linux
x86_64 container with four vCPUs, Python 3.14, on a 192×256×256 synthetic CT
with eight classes, reading a new window for every patch. Each figure is the
median of ten `medh5 bench` runs (throughput: six runs of 512 batches), and
2.2.0 is within the run-to-run spread of 2.1.0 on every row. The 2.0.0 column
is the table this page published before, measured on another four-core
machine; it is the one the other pages quote. 1.4.4, measured beside an earlier
build of 2.0, was as fast or slower on every row. That comparison is the
[changelog](../changelog.md)'s, under 2.0.0. The 0.x column is the layout 1.0
replaced, as measured then.

| Metric | Target | 0.x | 2.0.0 | 2.1.0 | 2.2.0 |
|---|---|---|---|---|---|
| 64³ patch, multi-class labels only | ≤ 10 ms | 117 ms | 2.3 ms | 3.6 ms | **3.5 ms** |
| Foreground centre sampling *(indexed)* | ≤ 1 ms, O(1) memory | 9.2 ms, O(volume) | 0.03 ms | 0.05 ms | **0.05 ms** |
| … at 63 classes | ≤ 1 ms | — | 0.04 ms | 0.06 ms | **0.06 ms** |
| Metadata-only read | ≤ 2 ms | ~1.5 ms | 0.13 ms | 0.20 ms | **0.22 ms** |
| Full `open()` → first patch | ≤ 15 ms | ~120 ms | 1.6 ms | 2.1 ms | **2.2 ms** |
| Sustained 64³ throughput, one process | ≥ 400 patches/s | ~60 | ~590 | 430 | **440** |
| … with four workers | — | — | 880–980 | 450 | **510** |

```
$ medh5 bench                       # builds a synthetic sample and measures
$ medh5 bench case.medh5 --patch 64 --repeats 20 --workers 4 --json
```

Two things to know before quoting these.

**The sampling rows need an index.** `bench` calls `build_index()` on the
samples it builds, so 0.05 ms is the indexed path — the one you get after
`medh5 index build`, not the one you get by default. Unindexed, the same draw
scans the labels: 285 ms on this volume here (210 ms on the 2.0.0 machine, and
2.4 s at 512³ there), growing with the volume while the indexed draw stays
flat. `used_index` in the batch metadata, and in
the row's `detail` under `bench --json`, says which you measured. The 63-class
row exists because the class count is the other axis: before 1.4.2 each draw
re-read the index's class table once per class, 0.9 ms at eight classes and
7.6 ms at 63.

**`bench` does not check the throughput target.** The rows with a target are
verified and reported against; throughput depends on worker count, so it is
measured and printed without one, and *all targets met* is a statement about
the checked rows. `medh5 bench` with no `--workers` runs single-process; pass
`--workers 4` for the four-worker row. It is the row that depends most on the
machine and the patch: the four workers share the four cores with the main
process, which gained 1.5–1.7× from them on the 2.0.0 machine and 1.2× on this
one,
and at 96³ the 2.0.0 machine's loader moved about 490 patches/s. `bench` times
63 patches after the first batch, and on this machine its four-worker figure
ran from 120 to 670 patches/s between runs: for a number to compare, time more
batches (`medh5.bench.throughput(paths, batches=512, workers=4)`).

**Threads.** [`bench_threads.py`](../examples/bench_threads.py) deals one
fixed list of work over one, two and four threads, each with its own handles,
on four 2.2.0 samples like the one above. Seconds for the whole of it, median
of four processes, on the same machine:

| Work | GIL build: 1 thread | 4 threads | Free-threaded: 1 thread | 4 threads |
|---|---|---|---|---|
| 512 windows of a `training` (Blosc2) CT | 0.79 s | 0.74 s | 1.26 s | 0.76 s |
| 512 windows of a `portable` (gzip) CT | 6.16 s | 6.86 s | 6.20 s | 6.64 s |
| 512 windows of the annotation, `dense()` | 2.40 s | 0.71 s | 2.34 s | 0.73 s |
| 256 opens, each reading the grid and image metadata | 0.13 s | 0.25 s | 0.14 s | 0.52 s |
| … the same opens on 2.2.1 | 0.14 s | 0.19 s | 0.14 s | 0.25 s |

Windows that decompress outside HDF5's lock (Blosc2, and the label planes'
decoding) gain from threads; gzip, which HDF5 decompresses under its lock,
loses about a tenth to the hand-offs, and opening loses most, because an open
is many small HDF5 calls and each one changes hands. In 2.2.0 the
free-threaded build (Python 3.14t) lost the most there: 2.1.0's free-threaded
wheel ran with the GIL, and took 0.24 s at four threads. 2.2.1 reads each value
an open needs under one take of HDF5's lock, which brings four threads back to
2.1.0's time on the free-threaded build and below it on the GIL build (the last
row, built from source and timed beside 2.2.0 on the same machine). One thread
is no slower than 2.1.0 on either build. The one-thread Blosc2 row moves between about 0.8 and 1.3 s
with the process rather than the build: timed side by side in one process,
the two builds read those windows in the same time.
[Threads or processes](../explanation/concurrency.md#threads-or-processes)
says which to choose.



Three decisions are behind the label-read number: each stacked plane is
chunked separately, so one layer reads without the others; a multi-class
`dense()` reads **by plane rather than by class**, so a 200-class annotation
packed into four layers is four reads and not two hundred; and a window
decompresses only the Blosc2 blocks it covers, not every chunk it touches.

### Where a command's time goes

HDF5 runs one call at a time, its filters included, so threads can only
shorten the part of a command that runs outside an HDF5 call.
[`bench_phases.py`](../examples/bench_phases.py) times that part through a
native CLI built with phase timings (a `cfg`, `--cfg medh5_phase_timings`
in `RUSTFLAGS`, which no release sets). Seconds, median of three, on 2.2.2's
code: a Linux x86_64 container with four vCPUs and no SHA extensions, a
256×512×512 sample of two CTs, a PET and an eight-label mask (576 MiB). The
wall in brackets is an ordinary build's, run beside it.

| Command | Wall | Under the lock | Outside it | Largest phases |
|---|---|---|---|---|
| `resample`, nearest, no annotation | 16.7 (14.9) | 3.5 | 9.8 | hash 5.5, window 2.6, compress 2.1, map 1.4 |
| `resample`, linear, carrying the annotation | 39.9 (40.4) | 10.4 | 24.2 | hash 8.5, interpolate 8.1, compress 6.9, window 4.4, carry 2.6 |
| `convert from-nifti` | 16.1 (16.0) | 4.7 | 8.1 | hash 3.3, compress 2.9, inflate 2.8, write 1.6 |
| an amendment (`dataset split --write-claims`) | 12.3 (12.2) | 1.8 | 8.6 | hash 6.3, window 2.3, copy 1.3 |
| `recompress --profile archive` | 130.2 (129.8) | 122.5 | 5.9 | compress 119.9 |
| `recompress --profile training` | 10.1 (10.3) | 3.5 | 5.4 | hash 3.3, window 2.1, write 1.4, compress 1.2 |
| `verify` | 5.7 (5.4) | 0.2 | 4.7 | hash 3.3, window 1.4 |
| `validate --level integrity` | 5.7 (5.8) | 0.2 | 4.7 | hash 3.4, window 1.3 |

*Under the lock* is opening, metadata, copying, reading and writing, and
the Blosc2 or Zstandard filter running inside a read or write (`compress`,
`decompress`). *Outside it* is the Blosc2 blocks a window decompresses itself
(`window`), hashing, the resampler's mapping and interpolation, carrying an
annotation, and NIfTI's gzip and decoding. What neither claims (argument
parsing, validation's rules, allocation) is the rest of the wall.

What that allows, by Amdahl's bound at four threads: with every phase
outside the lock spread over four threads; with only what a chunk writer's
workers would do (compressing whole chunks, and the mapping, interpolation
and decoding that make them); with only what the digests' workers would do
(decompressing and hashing, inside the stages that hash); and with both.

| Command | Outside, now | Chunk writes | Digests | Both |
|---|---|---|---|---|
| `resample`, nearest | 1.79× | 1.21× | 1.51× | 2.04× |
| `resample`, linear | 1.83× | 1.42× | 1.30× | 2.11× |
| `convert from-nifti` | 1.61× | 1.20× | 1.27× | 1.61× |
| an amendment | 2.11× | — | 1.35× | 1.35× |
| `recompress --profile archive` | 1.04× | 3.23× | 1.03× | 3.54× |
| `recompress --profile training` | 1.67× | 1.10× | 1.45× | 1.67× |
| `verify`, `validate --level integrity` | 2.60× | — | 2.60× | 2.60× |

Four things follow. The digests are the largest single cost of every
command that writes or checks a sample: SHA-256 runs at about 175 MiB/s on
one core here (`digest_bytes`: SHA-512 435 MiB/s, BLAKE2b 690 MiB/s), and a
commit hashes every dataset it keeps. Reading for a digest already
decompresses outside the lock, through the window path. Compression is the
other: the archive profile's Zstandard compresses this sample at about
7 MiB/s, all of it inside HDF5's write, so `recompress --profile archive` is
the one command that only a chunk writer can shorten. And the gzip of a
NIfTI file is one stream, inflated in order.

The SHA-256 above is software: this machine has no SHA extensions. `sha2`
0.10 uses them by run-time detection on x86_64 (Linux and Windows wheels and
binaries, on a CPU that has them), and on aarch64 (Linux and Apple Silicon)
only with its `asm` feature, which medh5 does not enable, so every aarch64
build hashes in software. On a CPU that hashes several times faster the
digests' share, and so what threads give them, is smaller.

### Threads in `recompress` and `resample`

With `--threads N`, `recompress` and `resample` compress the chunks they
write on *N* threads, and `resample` maps and interpolates each chunk on the
thread that compresses it
([Which calls use them](../explanation/concurrency.md#which-calls-use-them)).
The same sample and machine as above, an ordinary release build, seconds,
median of three:

| Command | 2.2.2 | 1 thread | 2 threads | 4 threads |
|---|---|---|---|---|
| `recompress --profile archive` | 116.3 | 90.5 | 48.2 | 29.5 |
| `recompress --profile training` | 8.5 | 7.9 | 8.2 | 7.5 |
| `resample`, nearest, no annotation | 15.7 | 15.3 | 15.4 | 14.1 |
| `resample`, linear, carrying the annotation | 43.4 | 40.0 | 33.4 | 29.1 |

Compression is nearly all of an archive recompression, so it is the command
threads shorten most. One thread is faster than 2.2.2 there too: 2.2.2 with
`BLOSC_NTHREADS=1` took 104.6 s, so about 12 s of its time was Blosc2
starting threads it never used, which it no longer does. A `training`
recompression and a nearest resample spend little time compressing, and most
of the rest is digests: these figures were taken before the digests used
threads, which the next section measures. Every file written at four threads
holds the bytes one thread writes.

### Threads in the digests

With `--threads N`, every command that commits a sample, `verify` and
`validate --level integrity` decompress and hash on *N* threads, several
datasets at once, while the calling thread reads the stored chunks
([Threads inside one call](../explanation/concurrency.md#threads-inside-one-call)).
The same sample on another Linux x86_64 container with four vCPUs, this one
*with* SHA extensions (`digest_bytes`: SHA-256 1,266 MiB/s), an ordinary
release build, seconds, median of three; one thread runs the digests as
2.2.2 does:

| Command | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| `resample`, nearest, no annotation | 9.7 | 6.8 | 6.5 |
| `convert from-nifti` | 12.3 | 10.4 | 10.3 |
| an amendment (`dataset split --write-claims`) | 6.6 | 2.9 | 2.1 |
| `recompress --profile training` | 5.9 | 4.2 | 4.1 |
| `verify` | 2.9 | 1.2 | 0.8 |
| `validate --level integrity` | 2.8 | 1.1 | 0.8 |

Two threads gain more than twice on the checks because the calling thread
keeps reading while both hash. A command that writes gains less: its
digests run after it writes, and what it writes, compresses and, for a
conversion, inflates is outside them. Every digest is the one a single
thread computes.

A dataset is one hash, on one worker at a time, so the largest one a
command keeps bounds its digests however many threads there are; the
largest are begun first. A slab of a dataset whose rows are long lies
across a band of chunks that holds several times its bytes: a 26-row slab
of a 512×513×513 `float32` image touches all of its band's 128-row chunks.
Each band of the slab's next axis is then read and decompressed on its
own, on the workers. Before that, HDF5 read such a slab under its lock,
decompressing every chunk of the band once for each slab across it. On a
machine like the first (four vCPUs, no SHA extensions), a linear `resample`
carrying the annotation took 14.9 s with four threads and 21.2 s with two,
against 18.3 s and 24.5 s reading so, median of five; the other rows did
not change.

## Related

- **[Storage](../reference/storage.md)** — codec profiles, chunking and the index in full.
- **[PyTorch and MONAI](../reference/torch.md)** — the datasets and samplers these levers act on.
- **[`medh5 bench`](../reference/cli.md#medh5-bench)** — every flag.
