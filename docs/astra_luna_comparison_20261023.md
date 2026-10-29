# Astra–Luna diagnostic comparison after numerical repairs

Status: **complete — 20/20 trials. Astra 9/10 successes; Luna 6/10.**

[Interactive report](../results/astra-luna-comparison-20261023/watch.html) · [Paired results CSV](../results/astra-luna-comparison-20261023/paired-results.csv) · [Timing CSV](../results/astra-luna-comparison-20261023/timing-summary.csv) · [Full analysis](../results/astra-luna-comparison-20261023/comparison-analysis.json).

The sequential runner took **235.5 minutes**. Both models used high reasoning, the same frozen prior guide and tools, and fresh contexts. These were evaluations: no cross-trial learning or guide promotion occurred.

## Outcomes

| State | Repeat | Luna: outcome / controls / actor minutes | Astra: outcome / controls / actor minutes |
|---|---|---|---|
| 39 | 1 | Success / 296 / 5.72 | Success / 293 / 7.33 |
| 24 | 1 | Failure / 600 / 17.24 | Success / 575 / 23.89 |
| 35 | 1 | Success / 279 / 4.65 | Success / 271 / 11.50 |
| 20 | 1 | Success / 340 / 7.75 | Success / 284 / 8.62 |
| 49 | 1 | Success / 441 / 14.35 | Success / 303 / 8.10 |
| 39 | 2 | Failure / 600 / 8.81 | Success / 313 / 9.78 |
| 24 | 2 | Success / 285 / 5.41 | Success / 374 / 11.00 |
| 35 | 2 | Success / 382 / 8.08 | Success / 290 / 9.78 |
| 20 | 2 | Failure / 600 / 15.21 | Success / 347 / 11.74 |
| 49 | 2 | Failure / 600 / 12.51 | Failure / 599 / 30.00 |

Paired outcomes: 6 both-success, 3 Astra-only, 0 Luna-only, and 1 both-failure pairs. These are two actor samples per model on five reused diagnostic states of one asset, not twenty independent new environments.

Among the **6 pairs where both succeeded**, mean Astra-minus-Luna time was **103.71 seconds for the whole actor session** and **44.83 seconds to the first terminal tool reply**. Positive means Astra was slower. These are descriptive paired costs; quickly failed trials are not efficiency wins.

## Time breakdown

The following means include all ten trials per model, including failures. Units are **seconds per trial**. Robot calls are clipped to the actor-session interval for this additive table; the HTML/CSV also retain full service and launcher timing.

| Component | Luna | Astra |
|---|---|---|
| Model/session gaps (observer estimate) | 257.11 | 506.77 |
| Other observed tools outside robot service | 20.62 | 8.44 |
| Direct grasp planning | 25.05 | 26.05 |
| Requested arc planning | 107.39 | 88.92 |
| Execution/gripper, including observation/update | 146.81 | 116.61 |
| Reset and RGB-D grounding | 29.42 | 28.46 |
| Other robot calls | 11.93 | 15.17 |
| Actor-session total | 598.32 | 790.41 |

The model/session estimate includes inference, scheduling, delivery and unobserved overhead. CLI event receipts have approximately 100 ms resolution. This is not a direct model-server inference timer.

Nested observation and tracking costs below are subdivisions of service work, **not additional actor time**. Capture includes reset and standalone observation calls; motion-tracker timings cover trajectory issue calls.

| Nested component | Luna | Astra |
|---|---|---|
| Simulator observation RPC | 14.03 | 9.56 |
| Read/project observations | 1.26 | 0.88 |
| Prepare/export public PNGs | 11.62 | 8.04 |
| Compress public sensor arrays | 61.98 | 42.82 |
| Tracker initialization around motion | 49.24 | 47.31 |
| RGB-D motion fitting | 0.19 | 0.13 |
| Scene transport and records | 0.59 | 0.37 |

Nested arc planning costs, seconds per trial:

| Component | Luna | Astra |
|---|---|---|
| Scene / optimizer setup and frame checks | 5.91 | 5.23 |
| Native scene setup | 3.52 | 2.10 |
| Local cuRobo worker elapsed | 26.25 | 20.63 |
| CPU equality polish | 21.19 | 20.02 |
| Native local path connections | 16.88 | 10.27 |
| Fallback scene / optimizer setup | 1.15 | 1.74 |
| Global cuRobo worker elapsed | 8.38 | 12.64 |
| Native global path connections | 0.49 | 0.31 |
| Admission / IPC / final validation / other (subtraction) | 23.63 | 15.98 |

Worker elapsed time includes host overhead and is not GPU utilization. The interactive report and timing CSV contain grasp generation, full-pose IK, approach search/resampling, execution and per-query details. Raw CLI usage is retained; the timed-out Astra episode has incomplete usage reporting, so no complete token-cost or monetary-cost estimate is claimed.

## Failure evidence

The following reviews use observer-only native geometry/contact records. They were never supplied to either actor. Native success is the outcome metric; source/control integrity passing does not certify that a physical trajectory tracked its reference accurately.

**state020-r2-luna: incorrect mechanism hypothesis retained through recovery.** Handle-side hinge line initially 265.2 mm from the native line. First arc moved the door 3.8 degrees. Later groundings retained that side; re-established finger contact did not correct the model. Native opening ended near 16.4 degrees at 600 controls. This establishes a model/grounding error, not a universal claim about model capability.

**state024-r1-luna: incorrect mechanism hypothesis with repeated failed physical progress.** Handle-side hinge annotation placed the fitted line 266.5 mm from the native line. First 30-degree arc moved the door 2.9 degrees. Subsequent groundings kept the same side; native opening ended near 10.1 degrees at 600 controls. The wrong hypothesis is directly observed; the separate contributions of grasp slip and model error were not isolated by intervention.

**state039-r2-luna: degraded mechanism estimate, contact loss and exhausted control budget.** The initial hinge line error was 28.4 mm. A slow approach and 45-degree jaw-axis motion used 394 controls to reach about 39 degrees of native opening. Re-grounding from another camera then moved the fitted hinge line to 119.2 mm from the native line. Later arcs made limited physical progress, and finger contact was absent at sampled endpoints from step 506. The episode exhausted 600 controls at about 63.5 degrees opening. Tracking errors were small (maximum about 0.024 rad), so this is distinct from Astra’s large tracking-deviation case. The relative contributions of model revision, contact loss and control allocation were not isolated by intervention.

**state049-r2-astra: failed contact recovery with large execution tracking deviation.** Initial hinge line was on the correct side (about 23 mm error). Door reached about 66 degrees, then finger contact was lost. A recovery motion lasting 198 controls had 1.178 rad maximum reference error and ended 22.52 mm/58.62 degrees from its commanded pose; palm/environment contacts were recorded. A usable grasp was not restored; door ended near 60.9 degrees and actor timed out at 1800 seconds with 599 controls. Reference clipping obeyed the existing controller contract. Controller pacing, collision-model mismatch and contact effects require separate isolation; no cause is assumed solely from clipping.

**state049-r2-luna: incorrect hinge annotation and grasp execution choices.** Horizontal hinge_pixels across the panel produced a midpoint pivot 128.4 mm from the native hinge. The first grasp-approach execute omitted the gripper argument, invoking the default closing command. Finger contact was not retained in early arcs. Late recovery produced only about 18.2 degrees opening before 600 controls. The trace does not isolate how much each annotation/gripper choice contributed. Tracking error stayed small compared with the failed Astra recovery.

[Detailed failure review](../results/astra-luna-comparison-20261023/failure-review.json) · [Grounding histories](../results/astra-luna-comparison-20261023/grounding-histories-private.json) · [Motion/contact traces](../results/astra-luna-comparison-20261023/motion-traces-private.json).

![Astra state 49 repeat 2 recovery tracking](../results/astra-luna-comparison-20261023/astra-state49-r2-recovery.png)

## Next engineering and learning targets

Post-study decision, 2026-10-24: keep **Luna as the actor for practice and evaluation**, following the user's preference. The pilot runner, compact practice runner, and fresh/resumed actor helpers already select Luna by default. Future routine experiments should use Luna; this supersedes the earlier recommendation to retain Astra as the reference model.

The research objective is to make affordable practice produce useful, transferable memory and to assess whether the input/output interface supports that process. A stronger actor may compensate for ambiguous inputs or tool behavior, making interface weaknesses harder to identify. That is a methodological concern, not a causal result established by this comparison. Test interface changes individually with the same Luna configuration and matched budgets, then evaluate practice-derived memories on separate held-out cases. Preserve the completed comparison as historical evidence.

- Validate mechanism annotations against the declared model assumptions and make unverified hinge hypotheses explicit. Practice hypothesis revision using observed object motion, not repeated successful planning calls alone. Do not copy evaluation pixels or poses into memory.
- Investigate execution pacing and endpoint tracking with isolated control tests. Expose useful execution-cost and endpoint-error summaries before relying on a planned motion for contact recovery. The failed Astra recovery mixed large tracking deviation with physical contacts, so those contributions still need separation.
- Reduce measured CPU overhead in sensor-array compression and tracker initialization. The tracker currently performs full-image nearest-neighbor membership queries; a bounded-distance implementation is a candidate requiring output-equivalence validation.
- Keep the guide frozen for evaluation and test future memory changes on separate development/held-out cases. This comparison does not measure learning across its fresh contexts.

Integrity checks passed for all twenty trials: native outcomes/control counts, issued references, bounded-controller mapping, stop-on-success, frozen sources/inputs, correct model on resume, distinct threads and isolation. [Independent audit](../results/astra-luna-comparison-20261023/independent-audit.json) · [Timing audit](../results/astra-luna-comparison-20261023/timing-audit.json) · [Initial-image audit](../results/astra-luna-comparison-20261023/initial-image-pair-audit.json).

## Repairs and validation

The pinned cuRobo v0.8.0 `ik/particle_ik.yml` still names its costs `bound_cfg`
and `multi_link_pose_cfg`, with `TELEPORT` as its bound-cost type. The current
`RobotCostManagerCfg.create` recognizes `cspace_cfg` and `tool_pose_cfg`, and
the current CSpace enum uses `POSITION`. The old keys were silently ignored.
With environment and self-collision optimization disabled for full-pose IK,
MPPI consequently had no remaining costs to aggregate. Successful LM seeds
could exit before this broken fallback, explaining why simple IK still worked.

The adapter now translates these legacy keys in a copy of the configuration,
preserving weights and collision settings. It requires a pose objective in
each optimizer stage. No installed dependency file was modified. This repair
also restores the particle stage's pose objective in arc planning; it is a
numerical behavior change, not merely exception suppression.

An optimizer exception now releases partial CUDA graph state while preserving
the error for that request. It does not silently retry or label the exception
as geometric infeasibility. A subsequent request can initialize normally.

The grasp selector now checks its existing minimum control allowance before
generation, IK or geometry search. The 200-control manipulation reserve,
40-control closure reserve and 60-control approach hold policy are unchanged.
Per-candidate checks still account for the actual approach trajectory.

Validation before the source freeze:

- 70 focused tests passed; one OS-isolation integration test was deselected.
  Every live pilot separately runs its actual isolation preflight.
- A synthetic reachable → unreachable → unreachable → reachable sequence
  reproduced the original empty-cost error followed by `None.replay()`.
  The repaired version returned zero valid solutions for unreachable targets
  and solved subsequent reachable targets, including forced optimizer execution.
- An injected warmup exception verified that partial graph inputs are cleared,
  the original exception is reported, and the next forced optimizer call succeeds.
- The saved-observation arc lifecycle regression passed all checks, including
  native endpoint admission, reproducibility and stale/closed-scope rejection.
- The full development replay reached native success at **289 controls** in
  **207.14 seconds** overall. All its source hashes remained unchanged.
- Minimal ChatGPT-authenticated access checks returned READY from both models.

[Change record](../results/astra-luna-comparison-20261023/fixes.json),
[exact patch against the prior working files](../results/astra-luna-comparison-20261023/fixes-and-instrumentation.patch),
[IK before](../results/astra-luna-comparison-20261023/ik-before.json),
[IK after](../results/astra-luna-comparison-20261023/ik-after.json),
[injected-error recovery](../results/astra-luna-comparison-20261023/graph-recovery.json),
[development replay](../results/astra-luna-comparison-20261023/development-replay/summary.json).

## Registered comparison

Twenty episodes: states **39, 24, 35, 20 and 49**, two repetitions for each of
`gpt-6-luna` and `gpt-6-astra`. These are previously evaluated diagnostic states,
not new held-out scenes. Repetitions are nested in five distinct states of one
microwave task/asset; they do not constitute twenty independent environments.

Both models receive identical prompt and prior-guide bytes, the same abstract
example, compact interface, RGB-D and robot-state access, corrected numerical
stack, seed 0, declared high reasoning, 600 controls and 1,800 actor seconds.
The prior guide is `luna-scene-motion-v1`, SHA256
`65f9612d3f66380234d78f3e05c6342f784b3f137ac4df8032f7fb04ba0c13c2`.
The learned candidate guide is not used in this model comparison.

Each episode starts a fresh context. Model order alternates in repetition one;
repetition two reverses both state order and each state's model order. Runs are
serialized on GPU 2. Native success ends motion immediately; failures remain
in the results without retries. An incomplete infrastructure trial stops the
suite for diagnosis. Source, inputs and driver hashes are checked between trials.

There is no Reflection reviewer, cross-trial memory update, task-specific human
hint, external-model fallback, API-key authentication or automatic guide promotion.
Astra's traces are observer evidence and are not passed to Luna.

## Timing interpretation

The report retains additive launcher/actor/service totals and nested grasp,
IK, arc, native-search and execution profiles. New private receipts separate:

- Simulator observation RPC, including nested sensor capture/robot depth priors,
  robot metadata and lossless observation storage.
- Reading/projecting observations, public PNG export, and sensor-array compression.
- Tracker initialization, RGB-D motion fitting, hand-motion prediction, scene
  transport, scene evidence and terminal bookkeeping.

The parent observer timestamps CLI event arrivals at roughly 100 ms resolution.
Intervals between tool-start and tool-complete receipts are combined as unions,
then intersected with robot-service intervals to avoid double counting. The
remaining model/session gaps include inference, scheduling, delivery and
unobserved overhead; they are not direct server-side inference measurements.
Raw per-turn CLI usage is preserved without treating it as billing or peak context.

In the development replay's seven observation captures, public sensor compression
used **32.23 seconds**, versus 6.64 seconds in simulator observation RPCs and
6.01 seconds preparing/exporting PNGs. This is a development measurement, not
the model-comparison result. Observation behavior remains frozen during the comparison.
