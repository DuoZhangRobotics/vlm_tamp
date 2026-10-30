# Cache-repair canary and conditional batch — 30 October 2026

The user authorized one fresh Luna trial with close memory/I/O monitoring and
continuation of the batch if stable. The new study is
`results/luna-cache-repair-validation-20261030`. Earlier interrupted studies
remain archived and are not pooled into this version.

The first trial is the declared baseline entry
`baseline-r1-close_microwave-state049-600`, the scene that exposed the runaway
cache. It counts as one of the twelve baseline trials; it is not repeated when
the remaining schedule runs. The case-order override is recorded before the
trial starts. Luna receives no old evaluator notes or successful coordinates.

Settings remain Luna/high, GPU 3 for CUDA/EGL, 600 controls and 1,800 seconds per
episode. Collision geometry, controller limits and task success criteria are
unchanged. This version includes the two-entry native scene cache and abandoned
query-handle cleanup from the resource repair.

## Host protection and admission

- This launch uses a **16 GiB** hard job memory limit, zero swap and whole-group
  OOM killing. The supervisor refuses unbounded or incorrectly configured jobs.
- Before each actor starts, admission checks host available memory and shared
  `user.slice` headroom. Clean inactive file cache is considered reclaimable;
  tmpfs/shared memory and dirty file data are not treated as free capacity.
- The check allows the job to grow to its enforced limit while leaving 2 GiB
  additional headroom. Existing charged anonymous/tmpfs memory is accounted for
  without counting reclaimable file cache twice.
- Waiting consumes no episode control or wall-time budget. A small waiting
  record updates the dashboard at most once per minute.
- During an admitted trial, the critical watchdog samples pressure and cgroup
  memory every two seconds. Detailed process and whole-node NFS READ/WRITE/
  COMMIT counters are sampled separately every 15 seconds. They cannot block
  the critical loop. NFS traffic is not attributed solely to this job.
- Three consecutive full-I/O pressure readings at least 10%, or full-memory
  pressure readings at least 5% accompanied by actual host/shared-pool shortage
  or new hits on the job's memory limit, stop the job. Infrastructure interruptions are
  separate from task failures, and are never silently retried.

## Qualification and continuation

`--stage qualify-and-continue --first-run baseline-r1-close_microwave-state049-600`
runs the canary first. Continuation requires normal driver/episode completion,
no resource stop, at least three active resource samples, and no sampling gap
over ten seconds. Robot task success is reported separately and is not required
to qualify the reporting/resource repair.

If qualified, the remaining eleven baseline episodes and the six-episode clean
practice session proceed sequentially under the same guard and admission check.
The later eight held-out evaluations still follow the existing exact-memory
scope review. That review is an analyst task, not another approval request.

Raw capture storage is still the existing NFS path. This authorized monitored
trial measures that path with the cache fix and records NFS traffic. It does
not claim that node-local capture staging has already been implemented.

Source and native-binary hashes are frozen at the first run. The dashboard
remains manual-refresh by default, with all available videos in the gallery.
Use `study-manifest.json`, `resource-qualification.json`, `resource-monitor.json`
and per-run terminal records to continue work without guessing from stale status.

## Monitoring-only amendment after the first three completions

The canary completed normally with 11.04 GiB peak memory and no sampling gap
over 2.004 seconds. Opening 37 then succeeded at 305 controls and opening 13
at 444. The original host-wide memory-pressure rule interrupted closing 17 at
its last returned state of 500 controls. At that point the job used about
12 GiB (14.02 GiB lifetime peak), I/O pressure was zero, host available memory
was about 34.5 GiB, and shared effective headroom was about 23.1 GiB. The
monitor remained responsive. Those readings did not establish renewed cache
growth, a job-limit hit, or a host memory shortage.

The monitor now records job-local memory pressure and limit-event counters.
A host-wide pressure signal with available capacity is no longer sufficient
to abort an episode. The 16 GiB kernel limit, whole-group OOM kill and I/O stop
remain. The original partial attempt is kept at its original path; a fresh
replacement uses a distinct directory and maps to the same scheduled case.
It is not a task failure or an extra scored repetition.

This amendment changes monitoring/orchestration and observer display only.
The study records the exact before/after hashes. Numerical tools, prompts,
controller settings, case schedule and native binaries remain pinned. No
evaluator notebooks or interrupted-agent context are supplied to the retry.
