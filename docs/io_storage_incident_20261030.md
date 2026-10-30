# Server I/O incident and storage cleanup — 30 October 2026

The user stopped the Luna experiment after the server became difficult to use,
including SSH. Experiments remain stopped. No simulation or model episode was
restarted during this investigation. The runner now rejects this study's
explicit user-stop marker instead of resuming its remaining schedule.

The current report is
[`watch.html`](../results/luna-pipeline-validation-20261030/watch.html).
The complete cleanup accounting and deletion manifests are under
[`io-storage-cleanup-20261030`](../results/io-storage-cleanup-20261030/summary.json).

## What was established

- The project is on NFS (`/common/users`), and the user's home is also on NFS.
  `/tmp` is RAM-backed tmpfs, not local SSD storage.
- After the stop, Linux I/O pressure still showed a five-minute average of
  47.69% `some` and 37.83% `full`. Later measurements returned to approximately
  zero. These are host-wide measurements, not per-process attribution.
- The user service journal records the stop at 10:59:56 a.m. Eastern and a
  reported peak memory charge of `41.5G` for the job, including its children
  and charged cache. This is not isolated Luna memory or GPU memory.
- The live reporter rebuilt the full study every 15 seconds, rereading completed
  logs and timing artifacts and embedding large candidate/stage arrays. Its
  latest HTML was 37,004,807 bytes. Rewriting it also forces Live Preview to
  handle a large changing file.
- Simulator captures write uncompressed arrays on the control path. One
  retained full capture was 72,364,870 bytes. A separate public sensor export is
  also written, and public exports were retained across many archived episodes.
  The results inventory alone contained 188.9 GB and 254,261 files before cleanup.
- Six stopped actor workspaces remained in RAM-backed `/tmp`, totaling
  2,629,149,845 bytes, including copied numerical-library payloads.

These establish substantial avoidable I/O and storage accumulation. They do
not identify the exact blocked system call or prove that this job alone caused
the entire SSH outage: the processes were already gone, so their I/O counters
and blocked stacks could not be sampled. No matching kernel NFS/OOM message
was returned by the available journal query. A future load test needs
per-process memory/I/O measurements as well as host pressure.

## Changes and checks

The dashboard links full evidence instead of embedding large arrays, caches
unchanged event logs and timing summaries, and skips rendering when its inputs
have not changed. The same report fell to 1,378,828 bytes. An unchanged poll
took about 0.0004 seconds and did not rewrite the page; the initial rebuild
took about 0.68 seconds. The stopped page does not automatically refresh.

Finished actor archival now applies depth retention to the public workspace
before copying it to persistent storage. It keeps the first/latest exports,
explicit measurement/reference inputs and identifiable custom-analysis inputs.
Unresolved custom dependencies keep all depth. RGB images, calibration,
proprioception, logs, notes, trajectories and results are retained.

The supplied image viewer hashes PNG files. That `read_bytes` use must not be
mistaken for arbitrary raw-depth access. Only an exact source match to the
known viewer receives that exemption; modified/custom code stays conservative.

Validation: 37 report/storage/dependency tests passed, followed by one focused
test proving that an explicitly stopped study cannot launch an actor. The
remaining 1,448 previously archived protected depth snapshots were checked
again after deletion. Numerical planning, controller settings and physical
execution were not changed or exercised.

## Cleanup result

| Category | Logical file bytes removed |
|---|---:|
| 2,139 old intermediate public sensor exports | 54.73 GB |
| Pip/UV download caches and compressed Conda download archives | 10.70 GB |

After accounting for hard links and preserving about 0.5 GB of newly archived
interrupted-run evidence, the estimated net allocated space reclaimed on the
project filesystem is **61.06 GB**. The six temporary workspaces were archived
before removal, releasing **2.63 GB from RAM-backed `/tmp`**. An additional
0.94 GB of unneeded sensor exports from those newly archived workspaces was
pruned under the same dependency rules.

Deletion ran at a limited rate; temporary evidence transfers were limited to
8 MiB/s. No videos, RGB images, learner memories, task outcomes, model weights,
installed environments or source files were deleted. Download caches can be
regenerated. Pruned intermediate depth cannot be replayed directly; the exact
removed paths and retained-input policy are recorded in the cleanup manifests.

The audit of selected large roots measured about 327.1 GB before cleanup; it was
not a complete inventory of every hidden project directory. The Qwen 9B and
27B weight downloads alone still occupy about 74.9 GB and were preserved.

## Before another long experiment

Do not automatically restart the interrupted frozen study. Five baseline
episodes completed; practice and held-out transfer did not run. Preserve that
record, including all interrupted attempts. The reporting/storage repair ends
the original source freeze; a subsequent experiment needs a declared version.

Active bulk capture files should be staged on actual node-local disk, with
compact archival to NFS. `/var/tmp` was on local NVMe with about 58 GB available
at inspection; `/tmp` consumes shared RAM. This staging change is not yet
implemented. Do not move an unbounded study into either location.

A compression-only probe reduced one 72.4 MB capture to 32.0 MB but cost about
1.97 seconds of CPU time. Capture compression was therefore not silently
changed during this cleanup. The next measured design should distinguish
capture computation, serialization, local writes, NFS archival and memory
pressure, without restarting a full batch to obtain that first measurement.
