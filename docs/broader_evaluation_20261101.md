# Frozen evaluation after the push/video repair

The user requested additional cases after the two successful closing reruns.
Run twelve independent Luna/high episodes on the same 4090: four additional
closing cases, four additional opening cases, and two more repeats each of
closing states 17 and 49. Keep the 600-control and 1800-second episode limits,
no inherited memories, no worked example, and no policy edits during this panel.

## Declared order

| Episode | Task | State | Group |
| --- | --- | --- | --- |
| 1 | close microwave | 30 | additional closing |
| 2 | open microwave | 36 | additional opening |
| 3 | close microwave | 17 | repeat |
| 4 | close microwave | 5 | additional closing |
| 5 | open microwave | 5 | additional opening |
| 6 | close microwave | 49 | repeat |
| 7 | close microwave | 3 | additional closing |
| 8 | open microwave | 6 | additional opening |
| 9 | close microwave | 17 | repeat |
| 10 | close microwave | 48 | additional closing |
| 11 | open microwave | 48 | additional opening |
| 12 | close microwave | 49 | repeat |

The additional cases were sampled with seed 20261101 after auditing 701 saved
evaluation, protocol and practice-case metadata files. Selection used recorded
task/state identifiers, not geometry or outcomes. The sampled task/state pairs
were absent from that audit; this does not assert global benchmark novelty.
The selection audit and exact input panel are in
`.runtime/4090-deployment-20261101/broader-selection-audit.json` and
`broader-cases-20261101.json`.

The existing runner now accepts an explicit `--cases-file` for evaluation-only
panels. Its actor invocation, tool contracts and numerical pipeline are
unchanged. The first case qualifies resource monitoring; ordinary task failure
does not stop the remaining cases. The existing watchdog stops infrastructure
failures without silently retrying a case. Later actors receive no earlier
findings or guidelines.

Launch with the existing supervisor's `baseline-rerun` stage and the explicit
case panel. Use 16 GiB RAM, no swap, GPU telemetry and a lightweight 60-second
mirror from `/home/duo/vlm_tamp` to this workspace. The new study is
`results/luna-broader-evaluation-20261101`. It is a flat study: its `trials/`
directory contains real run folders and must be included in the mirror.

Report additional closing, additional opening and repeated cases separately.
The repeat group plus the previous repaired pair gives three runs per familiar
closing state, while the additional cases test behavior outside those two
development states. Inspect the study manifest for actual progress and results.
