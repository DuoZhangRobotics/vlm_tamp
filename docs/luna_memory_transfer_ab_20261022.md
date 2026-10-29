# Frozen previous-guide versus learned-guide evaluation

Status: **complete — 10/10 trials, previous guide 3/5 successes, learned guide 2/5**.
The user requested five unpracticed initial states, two frozen memory conditions,
and detailed timing. This follows the
[five-success adaptive practice study](luna_curobo_continuation_practice_20261022.md).

[Interactive results, timing, and raw evidence](../results/luna-memory-transfer-20261022/watch.html).
Open with VS Code Live Preview. Both study processes exited successfully.
The sequential runner took **116.3 minutes**; actor sessions totalled **114.5 minutes**.

## Results

Time below is actor-session wall time, including robot-service calls and agent work.
Every failed episode exhausted its 600-control budget. No trial was retried or dropped.

| Official state | Previous guide: outcome / controls / minutes | Learned guide: outcome / controls / minutes |
|---|---|---|
| 39 | Failure / 600 / 22.57 | Success / 285 / 5.19 |
| 24 | Success / 278 / 4.34 | Failure / 600 / 14.10 |
| 35 | Success / 360 / 5.76 | Failure / 600 / 19.33 |
| 20 | Failure / 600 / 14.34 | Success / 272 / 3.99 |
| 49 | Success / 316 / 6.84 | Failure / 600 / 18.04 |

The learned guide wins two pairs; the previous guide wins three. No pair succeeds
under both guides, so there is **no successful-pair efficiency comparison**.
The five-success practice result did not establish transferable reliability.
The candidate has not earned promotion; neither frozen guide was modified.
Five pairs with one stochastic actor sample per condition are insufficient for
a broad claim about model quality or a reliable estimate of the memory effect.

## Measured time

These are **seconds per trial**, averaged over all five trials in each condition,
including failures. They describe costs, not success-adjusted efficiency.

| Actor-session component | Previous | Learned |
|---|---:|---:|
| Luna/session work outside robot calls | 312.34 | 271.11 |
| Direct grasp-planning calls | 44.33 | 174.90 |
| Requested arc-planning calls | 90.76 | 85.80 |
| Execution and gripper calls, including observation/update | 141.34 | 148.37 |
| Reset/runtime startup | 23.00 | 20.74 |
| RGB-D grounding | 13.61 | 10.14 |
| Other robot calls, including sequence/pose/joint planning and observation | 20.99 | 16.67 |
| **Actor-session total** | **646.36** | **727.74** |
| Launcher startup/reporting/cleanup outside actor session | 10.08 | 9.21 |
| **Launcher total** | **656.44** | **736.95** |

“Outside robot calls” includes inference, local analysis, transport and scheduling;
it is **not pure reasoning time**. Rounded entries may differ slightly from totals.

Within direct grasp planning:

| Nested component, seconds per trial | Previous | Learned |
|---|---:|---:|
| GraspGen generation/cache boundary, including model startup | 15.80 | 13.00 |
| Closed-gripper scene preparation | 4.36 | 3.47 |
| Full-pose IK worker | 2.03 | 2.42 |
| IK transport and CPU admission | 0.07 | 0.58 |
| Native approach connections | 0.96 | 135.33 |
| Contact-arc preflight | 5.25 | 5.31 |
| Remaining selection and tool-boundary work | 15.85 | 14.78 |

The learned condition's **135.33 seconds of native approach work per trial** divides
into 7.22 seconds native search, **75.45 seconds dense resampling**, and 52.66 seconds
endpoint checks/process/I/O/remaining work. This subdivision belongs inside the
135.33-second parent, not on top of it. Additional grasp work nested in
`plan_sequence` is accounted separately under that operation in the HTML/CSV.

Within requested arc planning:

| Nested component, seconds per trial | Previous | Learned |
|---|---:|---:|
| Initial scene/optimizer setup and frame checks | 4.70 | 4.14 |
| Native scene setup | 2.09 | 2.55 |
| Local cuRobo worker | 20.40 | 19.07 |
| CPU equality polishing | 21.09 | 20.44 |
| Native local path connections | 10.06 | 12.27 |
| Global fallback setup | 1.49 | 0.83 |
| Global cuRobo worker | 10.91 | 5.98 |
| Native global connections | 0.24 | 0.00 |
| Admission/IPC/final validation/remaining work | 19.78 | 20.52 |

Execution/gripper calls divide into simulator trajectory work (**7.47 / 7.33 seconds**),
trajectory RPC/host overhead (**2.65 / 3.14 seconds**), and observation/scene-update/tool-I/O
work (**131.22 / 137.90 seconds**). The last interval is measured by subtraction;
rendering, tracking, scene updates and storage were not individually instrumented.

[Trial results CSV](../results/luna-memory-transfer-20261022/trial-results.csv),
[aggregate timing CSV](../results/luna-memory-transfer-20261022/timing-summary.csv),
[every timing receipt CSV](../results/luna-memory-transfer-20261022/time-breakdown.csv),
[machine-readable summary](../results/luna-memory-transfer-20261022/study-summary.json).

## What failed and why

**Wrong mechanism grounding:** learned-guide states 35 and 49 place hinge pixels
on the handle side. Their fitted hinge lines are 267.7 mm and 264.2 mm from the
native hinge line. The first requested 30-degree arcs move the real door only
3.0 and 2.4 degrees, with only one finger pad contacting the microwave at the
sampled endpoint. The corresponding previous-guide first arcs move the door
28.0 and 22.3 degrees. Both pairs have exactly identical initial images and joint
positions. These are demonstrated perception/model errors; successful planning
under the wrong model does not establish physical progress. They do not, by
themselves, prove that the learned guide caused the errors.

**Expensive searches that cannot satisfy the budget:** the grasp selector reserves
200 manipulation controls and 40 closure controls, and its approach schedule
contains 60 hold controls. After step 300, even those holds cannot fit the
remaining approach allowance. The check occurs after generation, IK and native
approach planning. Three late calls spend **740.48 seconds (12.34 minutes)** in
selection: 230.03 and 237.19 seconds on state 35; 273.26 seconds on state 49.
They begin at steps 320, 434 and 394. This is a confirmed avoidable search cost,
separate from whether the reservation policy should itself change.

**cuRobo recovery runtime failures:** four full-pose IK calls fail while MPPI
sums an empty list of costs/constraints during CUDA-graph warmup. Another
24 tool calls fail on `None.replay()`. The executor stores `_graph_input` before
warmup and creates `_graph` afterward; a warmup exception leaves the input
populated and the graph absent. A later call skips initialization and tries to
replay the missing graph. The stack traces and code support this chain.
The underlying reason for empty MPPI costs still needs an isolated reproducer.
Some later pose calls succeed, so the worker is not uniformly unusable.
**This corrects the provisional GraspGen attribution: the empty-tensor exception
originates in cuRobo IK, not grasp generation.**

**Goal admission and contact recovery:** previous-guide state 39 reaches about
59.5 degrees, then loses contact during recovery. Learned-guide state 24 and
previous-guide state 20 reach about 69.9 and 69.8 degrees but stall on subsequent
arc admission. For example, state 24 event 14 rejects 14 local candidates for
scene collisions and 50 outside the equality-polish capture range; all 64 global
proposals are outside that range. No native path connection runs in that request.
For GPU-rejected proposals, the capture gate is 3 mm in this configuration.
This locates the failure stage; finite search does not establish that a feasible
solution is absent. Geometry/model/optimizer contributions remain unresolved.
Recovery is further impaired by the runtime and grasp-budget defects above.

[Failure review and evidence links](../results/luna-memory-transfer-20261022/failure-review.json),
[grounding/contact diagnosis](../results/luna-memory-transfer-20261022/grounding-diagnostics-private.json).
Native hinge and contact values are observer-only and were never passed to actors.

## Validation and next work

All ten trials pass the independent native-outcome, control-count, issued-target,
bounded-controller, immediate-success-stop, frozen-source, guide-hash, isolation,
and no-human-hint checks. Each uses a distinct actor thread. All 509 registered
source/config files remain unchanged. API ledgers are unchanged. Paired initial
robot joints match exactly; RGB images are exact or differ in at most nine pixels
by one channel quantization level out of 255. Timing partitions are nonnegative;
JavaScript syntax and the report render were checked.

[Independent audit](../results/luna-memory-transfer-20261022/independent-audit.json),
[initial-image comparison](../results/luna-memory-transfer-20261022/initial-image-pair-audit.json),
[report accounting audit](../results/luna-memory-transfer-20261022/report-audit.json),
[final checks and artifact hashes](../results/luna-memory-transfer-20261022/final-audit.json).

Before another learning comparison, prioritize the cheap pre-search grasp-budget
check and a reproducible fix for the cuRobo empty-cost/partial-initialization
failure. Then practice generic hinge identification and observed-motion/contact
validation on separate development scenes. Preserve the evaluation cases as
evaluation evidence; do not copy their pixels, poses, geometry or successful
actions into the guide. No fixes or memory promotions were made during this study.

## Registration

The selected states are **39, 24, 35, 20, and 49**, drawn with seed 20261022 from
the conservatively unexposed pool. Historical manifests alone omitted some
pre-restoration cases, so the exposure audit also excludes states named in
older project documentation. Selection uses state identifiers, never outcomes
or held-out images. The audit supports no recorded exposure in the retained
project history, not absence from model pretraining or undocumented runs.

Each state receives one trial with the previous `luna-scene-motion-v1` guide
and one with `luna-curobo-continuation-state45-v1`. Guide files are copied
byte-for-byte and hashed. The prompt, abstract example, current numerical
tools, task, seed, and budgets are identical. Previous-guide-first and
learned-guide-first order alternates across pairs. Trials run sequentially on
GPU 2 to reduce direct resource contention.

Every trial has a fresh Luna/high context, one episode, 600 controls, and an
1,800-second actor budget. It stops on native success or an actual budget
limit. There is no Reflection reviewer, cross-trial memory update, or automatic
memory promotion. Trial-authored findings remain evaluation records. Task and
agent failures are retained without retries; a startup/infrastructure failure
that prevents a complete trial stops the suite for inspection.

[Frozen protocol](../results/luna-memory-transfer-20261022/protocol.json),
[exposure audit](../results/luna-memory-transfer-20261022/exposure-audit.json),
[identical prompt](../results/luna-memory-transfer-20261022/inputs/prompt.txt),
[previous guide](../results/luna-memory-transfer-20261022/inputs/previous-guide.json),
[learned guide](../results/luna-memory-transfer-20261022/inputs/learned-guide.json).

## Timing boundaries

The top-level account separates robot-service calls from actor-session work
outside those calls, plus launcher startup/reporting/cleanup. The outside-call
residual includes model inference, shell analysis, transport, and scheduling;
it is not a direct measure of pure reasoning time. Failed or quickly terminated
trials must not be called efficiency improvements merely because they are short.

Private timing receipts were added before the source freeze, with no change
to planner decisions or learner-facing tool replies. The existing focused
checks passed: 41 tests, three unavailable-fixture skips.

- Grasp generation: checkpoint/cache-key work, model initialization, generation
  call, cache reads/writes, and remaining boundary work. A cached proposal does
  not count its historical inference time as work in the current trial.
- Grasp selection: full-pose IK worker time, RPC overhead, CPU admission,
  all native approach calls including unsuccessful searches, contact-arc
  preflight, and residual geometry/check/storage work.
- Arc planning: scene and optimizer setup, local/global GPU-worker calls,
  CPU equality polishing, local/global native connections, fallback setup,
  and other admission/IPC/storage work. Native search and dense resampling
  appear as nested components, not additional top-level costs.
- Execution: simulator trajectory-worker time, trajectory RPC overhead, and
  observation/scene-update/storage time outside the trajectory RPC.

Worker wall time includes host overhead and is not GPU utilization. Nested
components are labelled to prevent double counting. Residual categories are
explicitly marked as subtraction. Source hashes and frozen inputs are verified
before and after every trial.

## Interpretation limits

This is an initial-state transfer pilot on the same microwave task and asset,
not a different-object or different-task study. Five pairs are too small for
broad reliability claims. Native success is primary; controls and time should
be compared with failures and budget censoring visible, and successful-pair
efficiency reported separately.

The native hinge anchor is identical in these five official initial states;
their initial images vary, but this is not a test of new hinge geometry.
A memory condition that avoids a defective recovery branch cannot by itself
establish that the recovery implementation is reliable.
