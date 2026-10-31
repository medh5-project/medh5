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
Threads waiting for HDF5's lock at that moment are not the child's either, and the child forgets
them.

That last step depends on how medh5 was built. The wheels, the sdist and the release binaries have a
lock that keeps its queue of waiters in one word of its own, which the child resets with one write.
A build from crates.io (`cargo install medh5-cli`, or a crate that depends on `medh5`) has the lock
the published `hdf5-metno-sys` crate provides, which queues its waiters in a table shared across the
process. There, if another thread starts waiting for HDF5's lock at the very moment of the fork, it
can be partway into the table when the fork copies it, and the child hangs before any of its own
code runs ([#106](https://github.com/medh5-project/medh5/issues/106)). Measured with one thread
recompressing at four threads while another forked over and over, that came about once in 500 forks.
To rule it out in such a build, fork while no other thread is calling medh5.

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

## Threads inside one call

One setting says how many threads a single medh5 call may use for the work it does outside HDF5:
compressing chunks before HDF5 stores them, hashing bytes it has already read. HDF5's own calls stay
on the calling thread, one at a time, in a fixed order, so a file written with four threads holds
the same bytes as one written with one. The default is 1, which starts no thread at all.

The digests (§13.1) are the threaded work every command shares. A commit, `verify` and
`validate --level integrity` read each dataset's stored chunks on the calling thread, a slab at a
time, and decompress and hash them on the others, several datasets at once; each dataset's
digest is still one hash over its bytes in order, so it is the one a single thread computes, and
every report, finding and refusal comes out in the same order with the same words. A chunk that is
not Blosc2, or that the workers cannot decode, is read by HDF5 on the calling thread, as it is at
one thread. Each thread holds up to two slabs of 32 MiB in flight.

```python
import medh5

medh5.set_threads(4)              # every call in this process, from now on
assert medh5.get_threads() == 4
medh5.set_threads(1)              # the default: no thread is started
```

The count comes from `medh5.set_threads(n)` (Rust: `medh5::parallel::set_threads`), else the
`MEDH5_THREADS` environment variable, read once at the first call that needs it, else 1. A value
that is not a positive integer is refused with an error naming the variable; `0` is refused too,
since there is no automatic count. The threads are started inside the call and joined before it
returns: there is no pool, and nothing runs between calls. A `fork` waits for a call's threads to
finish, so a child never inherits a lock held by a thread it does not have.

On macOS, the thread that forked runs every call by itself in the child, whatever the count. A
call waits for its threads by parking, and Rust parks a thread there on a libdispatch semaphore,
a Mach port the thread makes the first time it parks; a forked child does not inherit Mach ports,
so a thread that had parked in the parent would wait on one that is not there, and the child would
be stopped with `SIGTRAP`. Threads the child starts itself make their own and use the count.

Training loaders keep one thread per call. `medh5.torch.worker_init_fn` sets one in each
`DataLoader` worker, so a parent's count is not multiplied by the number of workers: the workers
are the parallelism there, and they already fill the cores.

The command line takes `--threads N` before or after any command name, and passes the count to
every process it starts. It sits beside `--workers`: `--workers M` runs files in *M* processes, and
`--threads N` lets each call in each of them use *N* threads, so `--workers 4 --threads 2` can keep
eight threads busy. A command whose work has no threaded path runs as it did. The count is the
command's own: a process it starts is told it, but a child forked from it, such as a `DataLoader`
worker of `medh5 bench`, is outside the command and takes its count as any process does, so
`worker_init_fn`'s one thread holds there too.

### Which calls use them

`recompress` and `resample` compress the chunks they write on the threads. A worker is handed one
whole chunk of the dataset being written, compresses it with the Blosc2 filter's own code, and
hands it back; the calling thread stores each with `H5Dwrite_chunk`, in chunk order, so the file
holds the bytes HDF5's own write stores. `resample` also maps and interpolates each chunk on the
worker that compresses it. That covers images and label maps written under `training`, `balanced`
and `archive`. Everything the writer cannot store exactly as HDF5 would goes through HDF5's write
on the calling thread, as before: `portable` (gzip), a dataset under the compression threshold
(stored without chunks), boolean masks, and a chunk only partly covered by a write.

Blosc2's own `BLOSC_NTHREADS` is separate. A worker compresses with one Blosc2 thread, whatever
it says, so four threads never become thirty-two. Where HDF5's write runs the filter, the variable
still spreads a large chunk's blocks over that many threads, as in 2.2.2; those threads store the
blocks in the order they finish, so the stored bytes can differ from run to run, while the values
and every digest do not.

```bash
medh5 resample case.medh5 --image CT --spacing 1,1,1 --grid-id iso -o iso.medh5 --threads 4
medh5 verify --workers 2 --threads 2 case.medh5 case2.medh5
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
