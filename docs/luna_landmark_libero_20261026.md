# Luna landmark feedback in full LIBERO episodes — 26 October 2026

**Completed: landmark feedback succeeded in 4/6 episodes, versus 3/6 for
baseline.** All twelve full simulator episodes finished, and all study-owned
model/simulator/planner processes exited.
[Watch task outcomes, Luna I/O and timing](../results/luna-landmark-libero-20261026/watch.html).

The preceding [offline diagnostic](luna_landmark_feedback_20261026.md) improved
motion discrimination from 4/8 to 7/8 by resolving three uncertain answers.
Those were recorded evidence assessments. This experiment measures official
LIBERO task completion after Luna selects and executes its own actions.

## Completed results

| Task | Baseline | Landmark feedback |
| --- | ---: | ---: |
| Open microwave | 3/3 | 3/3 |
| Close microwave | 0/3 | 1/3 |
| Total | 3/6 | 4/6 |

Three pairs both succeeded, one succeeded only with feedback, and two pairs
both failed. This is one additional success in a small, single-sample paired
pilot on one asset family. It does not establish reliable general superiority.
Both arms share the VISTA-inspired visual helper and prediction/comparison
prompt; the measured difference is optional numerical landmark feedback.

| Task / state | Baseline: result / controls / actor seconds | Feedback: result / controls / actor seconds |
| --- | --- | --- |
| Open / 10 | Success / 348 / 419.05 | Success / 286 / 658.11 |
| Open / 13 | Success / 278 / 428.31 | Success / 278 / 344.99 |
| Open / 19 | Success / 472 / 471.02 | Success / 297 / 301.88 |
| Close / 9 | Failure / 600 / 1081.80 | Success / 502 / 1190.81 |
| Close / 17 | Failure / 600 / 930.98 | Failure / 600 / 727.90 |
| Close / 49 | Failure / 600 / 1294.06 | Failure / 600 / 1476.56 |

Mean actor time over all six cases was **770.87 s baseline / 783.38 s feedback**;
mean controls were **483 / 427.17**. On the three pairs where both succeeded,
feedback averaged 79 fewer controls and 4.46 seconds less actor time. The first
pair's substantial slowdown was not a consistent overall pattern. Failed
episodes remain in the overall means; success-only timing is reported as a
separate matched subset, not as a substitute for the primary outcome.

The complete scored runner took 9,456.73 seconds (157.61 minutes), excluding
the separately recorded infrastructure preflights.

## Feedback coverage and actual references

The six feedback-enabled episodes returned **31 landmark summaries across five
episodes**. All 31 contained at least one finite tracked pair; that does not
make every fit reliable. Closing state 17 received zero summaries because all
executed motions used surface-contact mode, which disables the current
articulated tracker.

The dashboard now shows per-episode and live-total counters for returned
summaries, summaries with tracked points, explicit public-note mentions, and
successful `visual.py inspect` calls. There were two public messages matching
the summary-specific term detector in the treatment. This is an observable
reference count, **not** a count of cognitive use: a message can repeat a
summary, silent use/paraphrases can be missed, and zero mentions does not prove
the data was ignored. Baseline produced no optional landmark summaries.
Successful visual inspections numbered 14 baseline and 5 treatment; the helper
was available in both conditions, and its use was self-selected.

The arithmetic for all 31 summaries totaled **0.01096 seconds**, already nested
inside RGB-D fitting. It was not the source of long episodes. Mean robot-service
time was 336.72/295.40 seconds; mean time outside that service was 434.14/487.97
seconds. The latter includes model processing, local/image tools, scheduling
and transport, not pure reasoning.

The report separates reset, grounding, grasp generation, grasp IK and posture
admission, arc-goal IK/projection, local/global path connections, execution,
image/depth capture, compression and tracking. For example, one failed baseline
state-17 grasp query spent 0.84 seconds in IK-worker calls but 43.79 seconds in
overall selection. All 381 tested grasp postures were rejected by the observed
collision model, mostly at arm links 5/6 against stationary points. The actual
initial approach posture was accepted. The label `start_collision` referred
to a candidate manipulation-start posture, not necessarily the robot's current
configuration. Remaining selection work is labelled as a residual rather than
being falsely attributed to IK.

## Why the closing cases failed

- **Baseline state 9:** the first arc closed the door by about 35 degrees. Luna
  misinterpreted the visual change as opening, reversed the model axis, and
  subsequent arcs reopened the door to about 120 degrees. The feedback run
  also lost contact but eventually closed the door with small surface pushes.
- **Feedback state 17:** the initial estimated hinge was about 25 cm from the
  native hinge line, near the opposite edge; the selected handle anchor was
  close to the actual hinge. Geometric fit acceptance did not validate part
  identity. No robot–microwave contacts were recorded, and the door angle was
  exactly unchanged throughout. The apparent wrong-way arm movements were
  unsuccessful approaches/retreats, not physical opening or closing. This
  episode also exposed the zero-feedback surface-mode coverage gap.
- **Baseline state 17:** no handle-grasp plan was admitted, and repeated pose
  and contact attempts produced only about 2 degrees of closing before the
  600-control cap. This was an effective-contact/planning failure.
- **State 49, both conditions:** both moved the door partway but could not
  recover useful contact and a closing path within budget. Feedback had an
  initially accurate hinge estimate (about 4.8 mm line error) and 11 summaries,
  yet ended about 61 degrees open after contact loss, rejected refits/poses,
  and late grasp-reserve rejections. Baseline ended about 45 degrees open and
  recognized that its final arc moved the hand with little panel motion.

These diagnoses use private native measurements after completion. None of the
native angles, contacts, hinge locations or interpretation notes were supplied
to Luna. No episode was replaced after observing an unfavorable result.

## Final validation and decision

All 67 focused runtime tests and two feedback-counter tests passed. The final
audit passed 312 per-trial native/controller/source/isolation checks. All 548
registered runtime source files and the guide stayed frozen during the scored
runs. Every trajectory completed or stopped on native success; no new execution
veto was introduced.

All six pairs match exactly on metric arrays, calibration, robot state and
common public inputs. **Exact RGB identity failed in three pairs:** open state
19 differs in 24 channel values, close state 9 in 26, and close state 49 in 85;
each difference is 1/255. The other three pairs match exactly. This deviation
is retained explicitly rather than hidden behind a tolerance.

The feedback feature remains **opt-in**, and no learner guide or memory was
promoted. The next technical priorities are validating part roles independently
of surface-fit residuals, preserving target-motion observation across contact
modes, and making candidate-posture and recovery-budget diagnostics explicit.
These require separate validation; the completed cohort stays unchanged.

## Registered cohort

| Official task | Suite | Initial states | Conditions |
| --- | --- | --- | --- |
| `KITCHEN_SCENE7_open_the_microwave` | `libero_90` | 10, 13, 19 | Baseline and landmark feedback |
| `KITCHEN_SCENE6_close_the_microwave` | `libero_90` | 9, 17, 49 | Baseline and landmark feedback |

There are **12 full episodes: two tasks × three states × two conditions**.
One fresh Luna context and one physical attempt are allowed per condition/case.
Each has 600 controls and 1,800 actor seconds. Trials run serially on GPU 0,
with tasks interleaved and condition order alternated within each task.

A retained-history audit read 1,810 ordinary/compressed metadata files and
included the documented older regression cohort. The selected task-specific
states had no prior use or reservation in that history. Deleted/unrecorded
runs cannot be ruled out. Both tasks use the same microwave asset family;
this is neither a full-suite benchmark nor new-object transfer. Drawer and
stove manipulation adapters are not part of the current Luna interface.

## Shared setup and intervention

Both conditions use `gpt-6-luna` with high reasoning, the same prompt and frozen
guide, the local visual-inspection helper, and a brief prediction/comparison
instruction. There is no reflection reviewer, stronger-model fallback,
fine-tuning, cross-episode memory update or automatic memory promotion. This
is a VISTA-inspired common harness, not a reproduction of the full VISTA system.

The only condition-specific capability is `--landmark-evidence`. It appends
measured displacement statistics from the same RGB-D correspondences **before**
the existing hinge fitter filters near-stationary points. The full and compact
responses retain these statistics for automatic post-motion fitting, explicit
tracking measurements and combined sequence stages. They do not classify
target motion or certify contact. Background selection and bad matches remain
possible.

The estimator, pose-transport choice, cuRobo v2 planner, collision checks and
controller are unchanged. Arc position tolerance remains 3 mm. Each requested
trajectory executes completely unless native success stops it. There is no
new execution-time veto. Summary arithmetic is timed inside the existing
RGB-D-fit cost, with a separate observer-only nested timer.

Two common interface changes apply to both arms: task selection now routes
the existing opening/closing motion mode, and explicit tracker documentation
states that `start_tracking` must precede the motion being measured. The
automatic execution tracker does not establish that explicit reference.
Neither change supplies target coordinates, native object geometry or an
episode-specific recovery strategy.

## Verification before model runs

- 67 focused tests passed, including preservation of fitting/execution behavior,
  summary reporting failures, compact sequence feedback and task direction.
- Four archived public RGB-D pairs reproduced their original fits exactly.
  Additional summary arithmetic took roughly 0.2–0.4 ms per pair in that check.
- Both tasks reset correctly on excluded state 0, with the official instruction,
  12 cameras, initial success false and the inherited 60-control gripper-opening
  procedure. These two infrastructure preflights are separate from scored runs.
- Both prepared conditions correctly expose the closing task and feature flag.

Sources, prompt, guide and case selection were frozen before the first scored
episode. Failed task attempts stay in the denominator. Infrastructure failure
stops the suite for inspection, with no automatic replacement.

## Results and logs

The report contains native success, controls, actor time, robot-service time,
grasp generation/selection, full-pose IK, local/global arc planning, observation
capture/compression, motion fitting and execution timings. Model-session gaps
also contain scheduling, image tools and transport; they are not pure inference.
Nested costs and residual subtraction buckets are labelled explicitly.

- [Protocol](../results/luna-landmark-libero-20261026/protocol.json)
- [State reservation and historical evidence](../results/luna-landmark-libero-20261026/state-reservation.json)
- [Episode results CSV](../results/luna-landmark-libero-20261026/trial-results.csv)
- [Timing CSV](../results/luna-landmark-libero-20261026/time-breakdown.csv)
- [Driver](../results/luna-landmark-libero-20261026/run_study.py)
- [Observer audit](../results/luna-landmark-libero-20261026/audit_study.py)
- [Live feedback tests](../tests/test_live_landmark_feedback.py)

Native axes, contacts and door angles are used only for private post-episode
diagnosis. They never enter Luna's inputs or guide. Exact sensor equality is
reported separately; small rendering differences will not silently be treated
as equal inputs.
