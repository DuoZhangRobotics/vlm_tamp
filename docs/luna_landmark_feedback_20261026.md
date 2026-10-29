# Luna landmark-feedback diagnostic — 26 October 2026

**Completed: measured landmark feedback improved motion discrimination from
4/8 to 7/8 in this small replay diagnostic.** This is a separate follow-up to the completed
[prediction-loop diagnostic](luna_evidence_loop_20261026.md).
[Open the report and exact Luna I/O](../results/luna-landmark-feedback-20261026/watch.html).

## Results

| Measure | Baseline | Landmark summary |
| --- | ---: | ---: |
| Valid output schemas | 8/8 | 8/8 |
| Correct motion/no-visible-motion decisions | 4/8 | 7/8 |
| Recognized stationary-door cases | 0/4 | 3/4 |
| Uncertain on stationary-door cases | 4/4 | 1/4 |
| Incorrectly asserted movement in stationary-door cases | 0/4 | 0/4 |
| Recognized moving-door controls | 4/4 | 4/4 |
| Static next-request checks passed | 7/8 | 8/8 |
| Mean actor wall time | 46.08 s | 41.55 s |

There are three treatment-only resolved pairs, four pairs where both are
correct, and one where both remain uncertain. The gain is resolving three
baseline uncertainties; neither condition made a false definite motion claim
in these follow-up samples. The preceding two-turn experiment had a different
sample and should not be pooled with these fresh baselines.

In treatment run 003, Luna read and cited the 88/90 near-stationary matches but
remained uncertain about which surface the matches represented. Run 013 on the
same checkpoint combined the counts with the views and reported no visible
door movement. This remaining variability is visible in the full report.
Baseline run 015 proposed `measure_tracking` without an active saved reference;
the other proposed requests passed the limited static checks. Passing those
checks does not prove a grounding selection or physical recovery would work.

All 16 sessions completed, with no timeouts or infrastructure failure. The
study runner took 766.08 s. The original 24-session experiment took 1,548.73 s;
40 fresh sessions were completed in total, without any physical robot episode.

Both follow-up conditions spent approximately 0.10 s per response in observed
shell-command intervals. The remaining 45.98/41.45 s includes model processing,
image tools, scheduling and transport. Sandbox preparation averaged 2.53/2.48 s;
archival work averaged 0.14/0.15 s. Planning, grasp search and IK are zero.
The derived evidence was prepared before these actor sessions. The exploratory
four-pair RGB-D replay took 25.75 s including input loading and feature tracking;
that preprocessing is not included in actor wall time. The 4.53-second actor
difference is descriptive and does not establish lower end-to-end latency.

The [final audit](../results/luna-landmark-feedback-20261026/audit.json) verifies
fresh contexts, Luna/high settings, exact original image delivery, immutable
inputs, private-data isolation and identical common input bytes within every
case. Only the derived JSON and its index differ. The original 547 runtime/guide
files remained unchanged, and all four original tracker fits reproduced exactly.

## Change being tested

The existing RGB-D hinge fitter removes correspondences with XY displacement
at or below 1.5 mm before fitting a moving part. Its public correspondence
count therefore omits near-stationary observations. In four replayed cases,
the original fits reproduced exactly from public saved RGB-D and actor-selected
geometry. The two stationary-door cases had 88/99 and 88/90 near-stationary
matches; the two moving-door cases had none.

The treatment adds a compact numerical summary: number of finite matches,
near-stationary count using the existing 1.5-mm threshold, XY displacement
quantiles, vertical-drift count and the number entering the unchanged hinge
fit. It contains no motion classification, native door angle, native contact,
correct hinge or prescribed next action. The explanation explicitly says that
selection can include background and correspondences can be wrong.

These are point measurements, not a validated semantic object-motion detector.
No acceptance rule, planner, controller, robot interface default or learner
memory is changed. The current artifact is an offline numerical-feedback
prototype. It has not been integrated into live practice.

The byte-identical summary implementation is now also available as the reusable
function `vlm_tamp.landmark_evidence.summarize_pairs(first, second)` in
[landmark_evidence.py](../src/vlm_tamp/landmark_evidence.py). It accepts two
N-by-3 arrays in world coordinates and reports descriptive statistics. Installing
this module after the experiments did not connect it to any live call path.

## Controlled comparison and limits

Four selected ordinary arc checkpoints receive two fresh samples per condition:
16 Luna sessions, one turn each. Both arms use `gpt-6-luna`, high reasoning,
the same exact prompt, eight original before/after images, complete public
prefixes, guide, manuals, analysis libraries, output schema and 240-second
wall budget. The sole input difference is a derived-evidence JSON file and its
index entry; baseline has an empty index. Sessions are serial and condition
order is counterbalanced. There are no automatic retries or teacher agents.

The primary endpoint is motion discrimination, with uncertainty reported
separately from false claims. Evidence support and static next-request
preconditions are reviewed separately. The previous experiment's responses
are not reused as this follow-up's baseline.

This is exploratory: the reused checkpoints motivated the summary, several
share episodes, and they come from one asset. Any improvement here needs new
scenes and physical practice validation before promoting a general lesson or
changing the default pipeline. One-turn wall times must not be compared
directly with the preceding two-turn experiment.

## Artifacts

- [Protocol](../results/luna-landmark-feedback-20261026/protocol.json)
- [Identical actor prompt](../results/luna-landmark-feedback-20261026/prompt.txt)
- [Summary implementation](../results/luna-landmark-feedback-20261026/landmark_summary.py)
- [Reusable module](../src/vlm_tamp/landmark_evidence.py)
- [Fresh-session runner](../results/luna-landmark-feedback-20261026/run_study.py)
- [Observer audit](../results/luna-landmark-feedback-20261026/audit_study.py)
- [Summary JSON](../results/luna-landmark-feedback-20261026/summary.json)
- [Timing CSV](../results/luna-landmark-feedback-20261026/timing.csv)

Three numerical-summary tests passed before model calls, covering stationary
points, moving points, vertical drift, nonfinite values, empty evidence and
invalid array shapes. The earlier 30 replay tests also passed.

The next validation should use new scenes and perturbations, with deliberately
imperfect target regions, occlusion and background contamination. Preserve the
separation between pose estimation, hinge estimation, motion verification and
contact verification. No task-specific coordinates or evaluator answers should
be promoted into Luna's guide from these reused checkpoints.
