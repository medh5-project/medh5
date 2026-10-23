# Threads, processes and open files

Read this before you add threads or processes to code that uses medh5. Most of what follows comes
from one fact: the HDF5 library underneath medh5 is not thread-safe, so every call into it in a
process takes one lock. Everything else is about working with that lock, or around it.

## One lock per process

Each process has one HDF5 lock, and every call into HDF5 holds it: opening a file, reading
metadata, reading a dataset, and decompressing that dataset's chunks through HDF5's filter
pipeline. The filters run *under* the lock, so two threads reading two different files in one
process take turns, however many cores the machine has.

One path runs outside it. A window read of a Blosc2-compressed image or label (the patch a sampler
asks for, a region a `roi` names) reads the stored chunk under the lock, then decompresses only the
Blosc2 blocks the window covers *after* releasing it. Threads reading patches of Blosc2 data
therefore decompress at once: four threads reading random 64³ patches measured about 1.2 ms per
read, against about 7.2 ms when every chunk went through HDF5. A whole-dataset read, and any data
compressed another way, decompresses under the lock. See
[Size the chunks to the patch](../guides/performance.md#size-the-chunks-to-the-patch) for why
windows are the common case in training.

The Python GIL is a separate lock. medh5's reads and writes, and its heavy array work (copying an
array of 1 MiB or more into the engine's layout, the annotation encoders, digests, the agreement
metrics), let go of the GIL while they work, so other Python threads keep running while one waits
for HDF5 or decompresses. Holding the GIL does not get a thread past the HDF5 lock, and releasing
it does not let two threads into HDF5 together.

## Threads or processes

**Processes** scale; threads mostly do not. Each process has its own HDF5 lock, so work spread
over processes runs side by side. The figures below are approximate, from one four-core machine,
and come from [Tune performance](../guides/performance.md#the-numbers) and the
[design rationale](design-rationale.md#clinical-context-and-prediction-tasks-11):

| Work | One | Four |
|---|---|---|
| `medh5 verify` over 32 files | 3.4 s | 0.9 s (processes) |
| A training loader, 64³ patches | about 590 patches/s | 880–980 patches/s (`DataLoader` workers) |
| A task's preflight | 3.6 s | 6.9 s (threads: slower) |

**Threads** are worth it when the work is mostly outside HDF5: Blosc2 window reads, your own NumPy
or PyTorch work between reads, or waiting on a GPU. A thread pool reading files is correct, only
not much faster:

```python
from concurrent.futures import ThreadPoolExecutor

import medh5


def mean_intensity(path):
    with medh5.open(path) as sample:            # one handle per thread
        return float(sample.images["CT"].read((slice(0, 8),) * 3).mean())


with ThreadPoolExecutor(max_workers=2) as pool:
    means = list(pool.map(mean_intensity, ["case.medh5", "case2.medh5"]))
assert len(means) == 2
```

Open a sample in the thread that reads it, as above. The training datasets' handle cache is shared by a process's threads and locked,
and an item holds its handle until it is done, so a thread-based loader never has a file closed
under it by another thread's eviction.

**Processes** take a file each. The Python API keeps no worker count of its own: a
`ProcessPoolExecutor` with one file per task is the whole recipe. Each task opens its file in the
worker and returns plain values, so no handle crosses between processes, and the pool's `spawn`
start makes each worker a fresh interpreter rather than a fork of one that may hold files open:

```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from medh5.validate import validate_file

paths = ["case.medh5", "case2.medh5"]
spawn = multiprocessing.get_context("spawn")
with ProcessPoolExecutor(max_workers=2, mp_context=spawn) as pool:
    reports = list(pool.map(validate_file, paths))   # in the order of paths
assert [report.ok for report in reports] == [True, True]
```

The function a pool runs crosses to the worker by name, so it is one a module defines (here
`validate_file`; your own goes in a module of yours, not in a notebook cell). A worker costs an
interpreter start, about 0.1 s, so a pool pays off on work that takes longer than that per file:
reading data, not only structure.

## Fork and exec

A `DataLoader` with `num_workers > 0` starts its workers with `fork` on Linux. An HDF5 handle
must not cross a `fork`: parent and child would share the library's state and the open file
descriptors under it. The torch handle cache is keyed by process id, so a worker that finds its
parent's handles *abandons* them, never closing what the parent still uses, and opens its own.
`worker_init_fn` does that as each worker starts rather than at its first read:

```python
from torch.utils.data import DataLoader

import medh5.torch
from medh5.sampling import PatchSampler

dataset = medh5.torch.PatchDataset(
    ["case.medh5", "case2.medh5"],
    PatchSampler((8, 8, 8), strategy="uniform"),
    images=["CT"],
    samples_per_volume=2,
)
loader = DataLoader(dataset, batch_size=2, num_workers=2,
                    worker_init_fn=medh5.torch.worker_init_fn,
                    collate_fn=medh5.torch.collate)
batch = next(iter(loader))
assert tuple(batch["images"]["CT"].shape) == (2, 8, 8, 8)
```

The same holds for your own code: open samples in the child, after the fork. A sample opened in the
parent and read in a forked child is the case the cache exists to avoid.

A fork waits for any HDF5 call another thread is in the middle of. A forked child has only the
thread that forked, so a lock another thread held at that moment would stay held in the child, and
its first read would wait for it forever. medh5 has `fork` itself take HDF5's lock as it begins
(`pthread_atfork`) and let it go on both sides as it returns, and the locks medh5 keeps in Python
(the handle caches, the vocabulary registry, documents' live views) are made anew in the child.

`exec` is the other half. HDF5 opens files without close-on-exec, so a program started while a
sample was open used to inherit the descriptor, and with it HDF5's file lock. medh5 now marks each
file it opens close-on-exec, on every platform, so a child process neither keeps the file locked
nor blocks the rename an amendment ends with.
[Starting programs while files are open](../guides/performance.md#starting-programs-while-files-are-open)
gives the two cases that remain.

## Writers

**A writer belongs to one thread.** A second thread calling the same writer while the first is in
a call is refused with a `MEDH5Error` saying the writer "is used by one thread at a time", rather
than made to wait, and the refused call changes nothing. Two writers on two files, one per thread,
are fine, and take turns at the HDF5 lock. The same holds for every object that changes as it is
used: a document, an annotation payload, a provenance record, a cache writer, a staged file and a
sampler's draws. Calls that only read one of them share it; they are refused only while another
thread is changing it. The dicts they hand out (a document's `extra`, a payload's `datasets`) are
yours to edit from any thread: a write or a `describe()` takes each as it stands at one moment.

**`amend` replaces the file.** An amendment is copy-on-write: it writes the new version beside the
file and renames it over the old one. Anything holding the file open across that keeps reading the
old version, because it still holds the old file. Open it again to read the new one.

**A training row reads what it pins.** A task's row pins its source's `content_id`, and a
dataset over the task takes its handle through a lease on that pin: a cached handle to a file that
was amended since is reopened, and a source that is now another version is refused (`T302`). The
other datasets lease without a pin, so an epoch that spans an amendment can read both versions.
See [the DataLoader](../reference/torch.md#the-dataloader).

**Two amendments of one file do not both win.** An amendment, and a `repack` or `recompress` in
place, refuses to commit when its source changed after it read it (`MEDH5FileError`: "changed after
this amendment read it; nothing was written"). That detects a lost update; it does not lock one
out, so serialise the writers of one file yourself
([why](design-rationale.md#amendments-detect-a-lost-update-rather-than-lock-it-out)).

HDF5's own file locking keeps writers apart on a local disk, and is unreliable on NFS, Lustre and
GPFS; see [On a network filesystem](../guides/performance.md#on-a-network-filesystem).

## Parallel commands

The command line follows the same rule. `validate`, `verify`, `fix`, `index build` and
`recompress` take `--workers N` and run the files in up to *N* processes, printing what one process
would, byte for byte. So do the commands whose unit of work is not one file: `migrate` (a subject's
0.x files, which become one sample), `convert to-nnunet` (a case), `clinical augment-cohort` (a
mapped subject), `task preflight` (a subject) and `conformance run` (a case). What the whole run
shares — the cohort's label set, the dataset's case names, the index over a bulk export — is
worked out once, before any worker starts, and units that touch one file go to one worker, in
order. `medh5 dataset stats --workers` is the exception: threads in one process, because it only
reads.

```bash
medh5 verify --workers 4 case.medh5 case2.medh5
medh5 dataset stats cohort.json --workers 4 -o stats.json
```

## Free-threaded CPython 3.14

CPython 3.14's free-threaded build (`python3.14t`) installs its own medh5 wheel (`cp314t`), and
runs it without the GIL: importing medh5 leaves the GIL off. Threads then run Python at once, and
medh5's own work beside them, but the HDF5 lock is the same one lock per process, so threads
reading files still take turns inside HDF5. What changes is everything around it: your NumPy and
PyTorch work between reads, Blosc2 window decompression, and the parts of medh5 written in Python.

A call that reads or writes a file lets go of the interpreter for as long as it runs, waiting for
the HDF5 lock included, so a garbage collection, which stops every thread, does not wait for one
thread's read to end because another is queued behind it. A few short steps still wait for the
lock attached, such as taking a second reference to an open dataset; a pause that meets one waits
for the read ahead of it, and no longer. An object
that changes as it is used (see [Writers](#writers)) is still used by one thread at a time, and a
second thread is refused rather than made to wait. A sample read from several threads at once
reads correctly, since every call takes the HDF5 lock, though one handle per thread, as above, is
still the simpler pattern. The training datasets' handle cache and the vocabulary registry are
locked.
