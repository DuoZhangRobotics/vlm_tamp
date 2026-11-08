# Compact transition feedback experiment

The completed pilot did not demonstrate a task-success improvement: both formats failed all
three live tasks. Compact feedback reduced JSON payload size, with a small difference in offline
judgments that did not extend to detecting wrong-direction motion.

The experiment tested a presentation change. It adds no tracker, motion gate, model upgrade,
task-specific action, geometry repair or inherited experience. Numerical backend files remain
identical to `b4f8db0`; both conditions use Luna/high, the two native LIBERO cameras, eight anchored
interaction memories, 600 controls and 1800 wall seconds.

The existing condition preserves current nested feedback. The compact condition replaces each
retained executed receipt with one transition containing the command, intent, achieved hand
motion/arrival, contact evidence, object-motion evidence, task status, phase boundaries and view
references. Repeated latest feedback points to that same transition. Missing measurements remain
unknown, not zero motion or lost contact. World-coordinate field names and selected contact
provenance remain explicit. Generic execute replies cannot acquire an unreported target pose.

Current observations, image attachments and their order, memory selection, non-motion replies,
tool contracts, action dispatch and termination are unchanged. Detailed original receipts remain
archived in decision results and tool logs. The compact presentation omits repetitive planner
implementation metadata; it is a summary of public evidence, not a new measurement source.

## Recorded-input test

Five predeclared development contexts cover microwave 17's wrong-direction stroke and later
partial closure, microwave 49's failed grasp approach, and the bowl episode's empty first grasp
and retained second grasp. Each receives two fresh judgments per format: 20 model calls total.
The images and their hashes are identical within each pair. Both formats receive the same
additional offline questions about object-goal change and contact retention. Returned actions
are never executed. Correct judgments, uncertain answers and false assertions are counted
separately, along with model/client time and text size. Private post-episode labels are read only
by the scorer and never enter the model's prompt. These selected cases cannot estimate general
accuracy or task success. Explicit questions may change behavior relative to an ordinary live
policy; that is why the live pilot follows separately.

## Paired live pilot

- Microwave closing, state 12.
- Top-drawer closing, state 1.
- Bowl placement, state 1.

Each receives one fresh episode per format, with condition order alternated between cases.
These IDs were absent from 106 indexed study manifests/configurations. They are new evaluation
states within familiar task families, not a claim of unseen geometry or comprehensive historical
holdout. There are six episodes total. The common seed, reset, controller and per-episode budgets
are fixed. No additional judgment question is inserted into the live actor. Live metrics are
native task success, control use, tool selections, wall/model time, prompt text size and observed
failure mechanisms. Recorded-input judgments measure detection separately.

All attempts remain in the report. Unknown evidence, unsuccessful tasks and infrastructure
interruptions are distinct outcomes. The existing 32 GiB job cap, zero-swap allowance and
memory/I/O/SSH watchdog remain active. A single outcome per format/state is a development pilot,
not a statistically established improvement. Retention of consulted tool schemas remains off
in both conditions. Compact feedback remains opt-in pending results.

Results are available in
[watch.html](/common/users/dz323/vlm_tamp/results/luna-compact-transition-20261108/watch.html),
with a separate linked recorded-judgment page. Refresh is manual; no camera or native geometry
is added to the policy observations.


## Completed recorded-input results

| Format | Correct claims | Uncertain | False assertions | Mean model/client seconds |
|---|---:|---:|---:|---:|
| Existing | 3/10 | 3 | 4 | 19.55 |
| Compact | 4/10 | 4 | 2 | 20.79 |

Neither format detected any of the four scored wrong-direction examples per format. On the two
partial-closure judgments, existing feedback had zero correct and compact had one. Both formats
identified the empty grasp twice and the retained grasp once in two repeats each. The lower false
assertion count partly reflects more uncertainty, not recovered motion measurements. These are
correlated repeats on five selected contexts, not ten independent robot trials. No latency gain
was demonstrated in this matched-image test.

[Recorded judgments, exact inputs and images](/common/users/dz323/vlm_tamp/results/luna-compact-transition-20261108/recorded-decisions/watch.html)
include every declared repeat. All 122 image/document links were checked.

## Completed live results

All six episodes ran with frozen revision `4a747d3`. The later fixes below were not present during
the comparison. Native success was false in every case.

| Task / state | Format | Controls | Actor wall s | Model/client s | Stop |
|---|---|---:|---:|---:|---|
| Microwave 12 | Existing | 240 | 1800.24 | 1555.43 | Wall budget |
| Microwave 12 | Compact | 600 | 254.01 | 185.91 | Control budget |
| Drawer 1 | Compact | 600 | 975.35 | 807.70 | Control budget |
| Drawer 1 | Existing | 248 | 1800.14 | 1666.22 | Wall budget |
| Bowl 1 | Existing | 600 | 741.45 | 639.42 | Control budget |
| Bowl 1 | Compact | 600 | 824.80 | 721.38 | Control budget |

Total actor time to failure was 72.36 minutes with existing feedback and 34.24 minutes with compact
feedback. This is not faster task completion. The compact microwave run consumed 434 controls on
its first executed call, before any compact motion receipt was delivered. Initial action choices
also differed between stochastic runs. A single pair per state cannot establish that formatting
caused the large wall-time difference. The compact bowl episode was slower than its baseline.

For the 105 compact live contexts, their original JSON payloads would total 6,104,198 bytes;
the presented versions total 4,531,246 bytes, a 25.8% reduction. This is JSON text size, not total
multimodal tokens or billed cost. The image-identity claim applies to each presentation transform
and to the paired recorded probes. Live trajectories diverged, so the two live policies did not
receive identical images after choosing different actions.

## What physically prevented completion

The dashboard contains each failed execution video, precise phase timestamps, planning budgets,
remaining time, contact evidence and next diagnostic. Operator-only native joint/contact labels
were never supplied to the policy.

- **Microwave / existing:** pixel [344,218] sampled the tabletop through the door window, at
  world height 0.90134 m with a nearly upward normal. Luna interpreted this as a door normal.
  The contact approach hit the table and stopped 40.15 mm short. Later motions remained blocked;
  the door changed only from 86.06 to 85.47 degrees open before wall time expired.
- **Microwave / compact:** the first acquisition opened and then partially closed the door while
  using 434 controls. The door peaked at 115.84 degrees open and ended 78.44 degrees open.
  Later strokes encountered housing contacts. Faster control exhaustion did not solve closure.
- **Drawer / compact:** one grasp approach reached within 0.21 mm, but later +Y strokes reopened
  the drawer and hit cabinet geometry. Public feedback reported `no_object_target_associated`.
  It started 148.51 mm open and ended 160.31 mm open.
- **Drawer / existing:** twenty precontact IK and twelve contact IK attempts returned no solution.
  The one acquisition that executed was blocked 12.16 mm short; the subsequent stroke was blocked
  30.28 mm short. The drawer ended 156.59 mm open, and wall time expired with 352 controls unused.
  Finite solver failures do not prove that every contact was infeasible.
- **Bowl / existing:** no gripper-closing command was issued. Two contact approaches stopped
  against the bowl/table. A late timing mismatch rejected a nominally feasible approach; after
  Luna explicitly corrected the timing, tracking waits still exhausted the last 43 controls.
- **Bowl / compact:** grasp, 12 cm lift and carry were physically achieved, and Luna recognized
  retention. A lower pose had no robot-valid IK result. Release left the bowl visibly offset over
  the plate without native success. A corrective contact attempt and last stroke used the rest
  of the controls. Exact object-body-center alignment was not exported, so the residual in the
  native placement criterion is not quantified.

These outcomes identify contact selection/acquisition, missing part-motion association and
placement alignment as remaining problems. Compact wording did not manufacture the missing
measurements or make an unreachable approach physically succeed.

## Post-study repairs and validation

The bowl baseline exposed a planning/execution parameter mismatch. `plan_grasp` accepted a path
estimated at 24 controls with 1 rad/s and zero hold. `execute` without explicit timing used
0.35 rad/s plus six holds, producing 74 nominal controls and rejecting it against the 43 remaining.
The executor now inherits stored grasp speed/hold when omitted. It also defaults to the declared
open gripper for a grasp approach, while preserving explicit overrides and existing contact-plan
aperture rules. Other plans keep their existing timing defaults. The existing budget error now
reports requested and remaining counts; no additional rejection rule was introduced.

A scheduling-only replay of the recorded path confirms 24 nominal controls after the repair.
It executes no robot motion and does not demonstrate task success. In the actual original run,
Luna eventually supplied the matching settings itself; the arm still needed tracking waits and
exhausted the controls before arrival. This timing fix is therefore not presented as the sole
cause or solution of that episode's failure.

A post-study preservation fix also carries optional generic scene-pose/motion-effect reports into
compact object evidence. None occurred in the six live trajectories, and replaying their input
transforms still matches all recorded contexts exactly. One older history entry in the offline
source corpus contained unresolved tracking diagnostics; the recorded probes retain the originally
tested format and were not rerun or retroactively replaced.

Validation:

- 23 focused tests passed before freezing the experiment; all 65 archived executed receipts from
  the preceding ten runs converted successfully.
- All 301 live contexts reconstruct from the archived public receipts with matching original
  payload hashes; all 300 distinct image paths counted per run have verified hashes. No policy
  code or unexpected agent tool execution appeared in the audited model event streams.
- After the parameter and preservation fixes, 89 focused tests pass; configured Ruff checks and
  `git diff --check` pass. The full repository suite was not run, and the post-study changes were
  not followed by another physical trial batch. No native code or dependency rebuild was needed.
- Browser validation passed for all six execution videos, 14 sampled decisions, 132 input images,
  94 document links and three diagnostic images, including the actual presented-feedback panel.
  The page also passed at a 360-pixel viewport.
- The service that schedules execution is now included explicitly in future numerical-reference
  comparisons, in addition to the full source freeze already used by this study.
- The run stayed under its 32 GiB job cap, with a 6.53 GiB peak in anonymous/shared memory,
  at least 67.32 GiB host-available RAM, zero OOM kills and 455/455 successful SSH probes.
  Total cgroup usage included substantial reclaimable file cache. Host I/O pressure briefly
  reached 11.98%; successful SSH probes continued. Workers stopped normally, and GPU 3 returned
  to its previous 59,567 MiB allocation by other workloads.

Compact feedback remains opt-in. This pilot supports a smaller, inspectable representation;
it does not establish improved task success, reliable wrong-direction detection or lower inference
latency. More episode memory alone is not a demonstrated remedy for the measured contact and
object-motion failures.
