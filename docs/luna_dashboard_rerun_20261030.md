# Small live dashboard and authorized rerun — 30 October 2026

This version was subsequently stopped after the repeated resource outage.
The [resource repair handoff](io_memory_recurrence_20261030.md) supersedes its
monitoring and UI behavior. Experiments remain stopped.

The user authorized a rerun after making `watch.html` smaller and writing it
less frequently. The new study is
`results/luna-pipeline-validation-rerun-20261030`; its `watch.html` is the current
Live Preview entrypoint. The interrupted earlier study remains archived and its
five completed outcomes are not pooled into the new comparison.

## Reporting

- The main HTML is static, under 10 KB, and is written only when its template
  changes. It fetches `watch-status.json` once per minute while the tab is visible.
  Phase/episode boundaries publish immediately; ordinary tool updates do not.
- Run details live in separate `trials/<run>/watch-data.json` files and load for
  the selected run only. Complete logs stay available through links.
- Video source URLs are assigned only when the user presses **Play selected
  video**. `preload="none"` alone was insufficient in the browser test.
  Contact overlays load on expansion. The selected active run's recorded frame
  remains available without downloading the full video.
- Timing, feedback counts, selected skill/mode, requested/measured speed,
  measured object progress and recent numerical errors remain visible. Nested
  timings are explicitly labeled; outside-tool time is not pure model reasoning.
- Completed event logs and timing summaries are reused. Finished episode
  reports summarize large arrays and link to the complete tool log.
- A missing live process is not labeled running merely because an old status
  file said so. The browser flags a running status that has not been refreshed
  for more than 150 seconds.

Validation uses focused report/retention/resource tests, a real headless Chrome
request audit, and a short server-managed resource-monitor probe. Preflight
evidence is retained under the new study's `preflight/` directory.

## Experiment and resource protocol

The model, prompts, numerical tools and physical settings are unchanged:
Luna/high, 600 controls and 1,800 seconds per episode, GPU 3 for CUDA and EGL,
zero added collision padding, and no additional dense path validation.
The source and native binary are frozen before the first episode.

Run one complete opening-37 regression first and inspect its I/O record. Then
continue the declared twelve fresh baseline episodes (three each for opening
13/37 and closing 17/49), one clean persistent six-episode practice session,
and eight fresh held-out evaluations with/without the frozen learner notes.
The first episode counts as one of the twelve. No old evaluator notes are
supplied. The existing parent review of exact learner-authored transfer memory
is an analyst step, not another user-approval requirement.

The original `scripts/supervise_luna_dashboard_study.py` sampled its own systemd
job's cgroup memory and process I/O counters every 15 seconds. Logs are small local files;
the dashboard reads their latest sample. Process `rchar/wchar` include IPC,
and sums can fall when a process exits, so they are not treated as NFS bandwidth.
The host pressure measurements also include unrelated workloads.

Three consecutive samples with full I/O pressure at least 10%, or full memory
pressure at least 5%, stop the current process group and stop the batch. This
is recorded as an infrastructure interruption, never as a Luna task failure.
There are no automatic retries after an interruption. These thresholds protect
the shared server following the observed outage; they do not alter robot motion.

Active captures still use the existing storage path. This rerun measures
whether the report changes are enough; it does not assume that HTML was the
only source of the previous stall. If pressure returns, investigate the
recorded processes and stage bulk captures on node-local disk before retrying.

## Operation

Source `scripts/server_runtime_env.sh` and set
`CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3`. Use a user systemd service so
monitoring and trial execution survive chat interruptions:

```bash
python scripts/supervise_luna_dashboard_study.py \
  --study results/luna-pipeline-validation-rerun-20261030 --stage first
python scripts/supervise_luna_dashboard_study.py \
  --study results/luna-pipeline-validation-rerun-20261030 --stage remaining
```

After practice, review the exact guidelines, record `memory-scope-review.json`,
and use `--stage transfer`. Preserve every interrupted or completed attempt.
The monitor's local paths are recorded in `resource-monitor.json`; the usual
study manifest and normal episode terminal records remain authoritative for
task outcomes.
