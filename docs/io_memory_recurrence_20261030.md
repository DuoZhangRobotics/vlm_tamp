# Repeated server I/O outage: native cache growth and active writes

The user stopped the rerun after SSH access degraded again. Labmates could log
in again after it stopped. The experiments remain stopped; practice and transfer
were not reached. Three new baseline episodes completed: opening 37 failed at
600 controls, opening 13 succeeded at 422, and closing 17 failed at 600. Closing
49 was interrupted and is not counted as a completed task failure.

## Confirmed memory defect

`NativeScene` cached a complete native collision index for every distinct
observed/predicted point cloud. The dictionary had no eviction. Closing a query
released its clone but left the large shared index owned by the global cache.
Only `finish_attempt()` cleared the cache.

In closing-49, one 219.35-second push changed job memory from approximately
5.30 GiB to 56.99 GiB. The job peaked at 57.47 GiB. The last recorded full I/O
pressure average was 77.63%, and full memory pressure reached 13.51%. Earlier,
the robot-service process alone had about 32 GiB RSS while the report process
had about 136 MiB. This was host RAM, not Luna model weights.

The shared `user.slice` has a memory-high boundary of 175,877,652,480 bytes and
a maximum of 183,163,158,528 bytes. It covers user workloads and sessions. The
cache growth and observed memory/I/O pressure are consistent with reclaim
pressure affecting interactive access. The available records do not isolate
every contribution from other workloads or identify a particular blocked SSH
system call.

The four native adapters now use a shared two-entry cache: current geometry and
an immediate prediction can be reused; old indexes release their cache
ownership. Native clones retain shared ownership, so eviction does not remove
geometry from a live query. A finalizer also releases abandoned query handles.
No collision geometry, tolerance, controller setting or motion limit was
changed by this repair.

An offline test used the same eight updates of a subsampled saved point cloud,
without model calls, simulation or robot motion. With the old retention policy,
RSS grew from 0.40 GB to 2.50 GB. With two retained indexes it leveled off around
0.75 GB. All eight collision results matched. A native regression also verifies
that an obstacle remains blocking after its index is evicted from the cache
and the cache is cleared while a query still holds a reference.

## Why the watchdog did not protect SSH

Its last sampling interval stretched to 174 seconds. Detailed `/proc/PID`
inspection was on the critical path, and requiring three 15-second samples
was ineffective when sampling itself stopped progressing. The trace does not
identify exactly which inspection or scheduling delay caused that gap.

The critical loop now reads only local pressure/cgroup counters every two
seconds. Process details run separately and are never awaited by that loop;
target-process command-line memory is no longer inspected.
The launcher additionally requires an actual kernel memory limit of at most
32 GiB, zero swap, and whole-group OOM killing. A systemd launch must therefore
include `MemoryMax=32G`, `MemorySwapMax=0`, and `OOMPolicy=kill`.

An isolated 64 MiB kernel-limit test deliberately attempted a larger allocation.
It verified `memory.max=67108864` and `memory.oom.group=1` inside the process,
then ended in the expected kernel `oom-kill`. This did not run the robot or Luna.

## What was written to storage

The last interrupted run retained the following output. These are file sizes,
not a measurement of all traffic including reads and rewrites.

| Output | Count | Bytes | Location |
|---|---:|---:|---|
| Raw RGB-D arrays | 35 | 1,031,198,408 | NFS project storage |
| Full observation metadata | 35 | 344,966,214 | NFS project storage |
| Video frames | 411 | 12,934,025 | NFS project storage |
| Public sensor exports | 33 | 532,210,043 | RAM-backed `/tmp` |
| Actor and tool JSONL logs | 2 | 1,120,395 | NFS project storage |

The runtime still writes raw observations and metadata to NFS. Shrinking the
HTML did not remove that path. Node-local working storage with compact,
rate-limited archival remains a change to make before another long batch;
it has **not** been implemented in this repair.

After shutdown, a five-second audit found no experiment/simulator writer or
active dump handler. Whole-node NFS writes were approximately 0.137 MB/s on
`/common/users` and 0.0175 MB/s on `/common/home`, with sub-millisecond mean
write RPC times and zero current I/O pressure. The largest measured writer
under this account was Codex session/state logging, about 0.09 MB/s of storage
writes in that sample. These short samples do not prove all future activity
will remain quiet and do not attribute other users' traffic to this project.

## Stable inspection and all videos

The VS Code workspace disables Live Preview auto-reload. The viewer starts with
display updates paused; experiments and display state are independent. Status
refreshes keep selected details and existing video elements intact. Expanded
issues and paused playback survive updates; session state restores inspection
across an explicit page reload.

The main HTML is 3.6 KB plus 10.5 KB of JavaScript. The status index lists every
available episode video, so the gallery does not need to download detailed logs
or media to populate itself. The current rerun has four recordings: three
completed episodes and a partial closing-49 recording encoded from its 411
existing frames. No new physics was executed to make that partial recording.

A real Chrome test verified all four gallery entries, no video download before
Play, no navigation during status updates, preserved paused video position,
preserved selected run and expanded issue, and preserved expansion after a
manual detail update. The preview needs one manual reload to load this version.

## Evidence and remaining validation

See `results/io-memory-recurrence-20261030/summary.json`, the tool/memory
timeline, resource samples, old/bounded cache probe logs, kernel-limit test,
current I/O audits, and browser test. Relevant unit/native checks passed
(43 tests, plus the strengthened live-query eviction check).

No full Luna episode has been run with this repair. Preserve the interrupted
studies rather than silently continuing their frozen-source comparisons.
The next qualification should use the corrected cache and enforced kernel
limits, and account for the remaining NFS capture path before a long rerun.
