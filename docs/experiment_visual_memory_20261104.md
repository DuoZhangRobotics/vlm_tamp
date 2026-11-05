# Luna visual interface and bounded memory pilot

The implementation and eight declared trials are complete. The enhanced condition succeeded
on one of four tasks; the matched RGB/recent condition succeeded on none. One initial state
per task does not establish a general improvement or reliability.

Open [watch.html](/common/users/dz323/vlm_tamp/results/luna-visual-memory-native-images-20261104/watch.html)
with VS Code Live Preview. It contains all eight videos, exact inputs and outputs, marked images,
operator diagnoses, tool choices, nested profiling, and resource records. Refresh is manual.

## What was implemented

Luna/high returns one JSON tool request per decision. The host dispatches literal arguments to
fixed robot tools. No learner-authored policy program is executed. Each decision receives a
fresh context assembled from current observations and selected episode memory; tool history
outside that context is not silently retained by the CLI.

Only native `agentview` and `wrist` RGB-D enter the interface. Native images are supplied
separately at their original 512×512 resolution. `scene_views` projects observed points onto
XY/YZ/XZ views; it does not render hidden geometry or add cameras. `mark_scene_point` resolves
an explicitly marked projection pixel to its original camera/pixel and measured 3D point.
Free-space marking explicitly takes coordinates from two projections without claiming surface
support. `preview_hand` shows known gripper geometry at a requested pose; successful plans
also receive automatic previews. Previews issue no motion and add no acceptance check.

Both conditions use the same structured actor and numerical backend. Twenty numerical files
remain byte-identical to reference commit `e2c9860`. Common adapter work added official task
identifiers, admitted the LIBERO spatial suite, and allowed the existing direct GraspGen path
on a selected surface without fabricating a hinge. Those changes are shared by both conditions.

The memory budget is eight positive-length before/request/receipt/after interactions. The
baseline retains the first and most recent interactions. The enhanced selector retains fixed
temporal anchors plus the latest interaction, deduplicating repeated anchors and marking gaps.
Sparse anchors can omit short gripper actions; eight is a maximum, not a promise to fill every
slot. Memory is within the current episode only: no demonstrations, inherited task lessons,
weight updates, or cross-episode RSI claims are involved.

This is an adaptation of [OpenETA’s visual tools](https://github.com/OpenMOSS/OpenETA/blob/openeta-for-codex/docs/openeta-light.md)
and [RoboICL’s retention method](https://arxiv.org/html/2609.34261v1#S3.SS3), not a reproduction
of either complete system. RoboICL’s zero-shot setting uses Astra and 25 memory slots with
camera triptychs. Here Luna uses eight slots and separate native images, with a 48-image ceiling.
Unlike the paper, old zero-motion exchanges remain only in an immediate bounded tail rather
than being copied into every retained execution chunk. Full records remain archived.

## Declared protocol and outcomes

`configs/study_visual_memory.json` fixed four cases, seed zero, fresh agents, and alternating
condition order. Every episode had 600 controls and 1800 seconds. Controls are simulator
commands at 20 Hz, including 60 reset/settling controls: 600 controls are 30 simulated seconds,
with 27 seconds available after reset. Model calls and planning consume wall time, not controls.
These are development cases, including the previously used microwave state 17; no held-out
population claim is made.

| Task / initial state | RGB + first/recent memory | Cloud views + anchored memory |
|---|---|---|
| Microwave closure · 17 | Failed; 600 controls; 23.37 min | Failed; 518 controls; 30.00 min |
| Drawer closure · 0 | Failed; 600 controls; 24.57 min | Success; 231 controls; 3.85 min |
| Plate pushing · 0 | Failed; 365 controls; 30.00 min | Failed; 600 controls; 26.34 min |
| Bowl placement · 0 | Failed; 600 controls; 29.40 min | Failed; 570 controls; 30.00 min |

The enhanced microwave, baseline plate, and enhanced bowl exhausted wall time with controls
unused. The other failed cases exhausted controls. Native success is the task outcome; a
successful tool request or reached robot pose is not counted as task completion.

## What the failures actually showed

- **Contact postures:** One saved plate contact-IK candidate matches its requested point but
  places the other finger about 38 mm below the observed tabletop. The following approach
  failed native feasibility. The rejected approach configuration and per-component feasibility
  witness were not retained, so its exact rejection cause is unresolved. The diagram is an
  **unexecuted operator diagnostic**, not a penetration observed during execution or an image
  supplied to Luna.
- **Direction grounding:** A baseline plate request described a destination above/right in the
  image but requested world `[+0.707,-0.707,0]`. Calibration projects a 4 cm displacement in that
  direction down/left by approximately `[-21.25,+16.08]` pixels. The selected pixel itself
  round-trips correctly. This identifies a direction-conversion error in that request.
- **Bowl state interpretation:** The enhanced actor grasped the correct bowl and lifted it
  roughly 12 cm. Recorded contacts and images support retention; decision 24 received the
  actual post-lift image. Luna then inferred that the bowl remained on the table and opened
  the gripper to retry. The second grasp also retained bowl contact during carrying, but no
  final release occurred before the wall deadline. All ten issued command batches completed.
- **Placement geometry:** The bowl baseline completed two grasp/transport/release attempts and
  all 15 issued batches, but native placement remained false. The existing LIBERO `On` rule
  requires contact, vertical ordering, and XY body-center distance below 3 cm. Terminal object
  poses were not exported, so this report does not quantify the residual alignment error.
  A TCP target at a plate point does not itself specify the held bowl’s center at that point.

An existing permission difference is also relevant: dedicated push free approaches check
fingers, whereas `surface_contact_tools.plan_contact_pose` uses `NativeScene` with
`check_fingers=False` along its generic path. It was not changed during this comparison.
The bowl examples demonstrate that pose execution can work precisely while task interpretation
and object-to-goal placement remain wrong. They do not justify blaming every failure on IK.

An operator-only replay of the enhanced bowl references followed by the remaining 30
open-gripper controls did not succeed. Its robot trace was close, but finger states and contact
history differed from the original. It is **not a matched counterfactual** and does not settle
whether release alone would have completed the original episode. No diagnostic action is
counted as a Luna success, and no operator evidence was supplied to the actors.

## Timing and actual feature use

Across the eight actor sessions, 90.03% of elapsed time was in model/client calls. There were
595 completed choices, including 179 help requests. These measurements include client and
formatting overhead; they are not isolated model reasoning time. Nested IK, grasp generation,
path planning, capture and execution timers overlap and must not be added to their parent totals.

| Task / condition | Session s | Model/client s | Robot service s | Choices / help |
|---|---:|---:|---:|---:|
| close_microwave / rgb_recent | 1402.23 | 1274.44 | 117.19 | 70 / 24 |
| close_microwave / cloud_anchors | 1800.19 | 1625.53 | 159.31 | 87 / 30 |
| close_top_drawer / cloud_anchors | 230.82 | 180.94 | 47.50 | 13 / 6 |
| close_top_drawer / rgb_recent | 1474.34 | 1314.28 | 148.41 | 69 / 18 |
| push_plate / rgb_recent | 1800.22 | 1643.37 | 142.05 | 88 / 27 |
| push_plate / cloud_anchors | 1580.13 | 1380.98 | 186.18 | 62 / 10 |
| place_bowl / cloud_anchors | 1800.19 | 1654.95 | 128.04 | 107 / 36 |
| place_bowl / rgb_recent | 1764.18 | 1596.46 | 150.59 | 99 / 28 |

All enhanced actors received initial observed-cloud projections. Voluntary `scene_views` calls
were 0/1/2/1 for microwave/drawer/plate/bowl. Point marking was attempted twice: one camera-name
error and one valid projection mark. No actor voluntarily called `preview_hand`; 12 distinct
hand previews were delivered automatically, mostly during bowl planning. The enhanced plate
retained 7 of 18 executed transitions and enhanced bowl 5 of 10. This combined-condition pilot
cannot isolate visual-interface effects from memory selection or ordinary rollout variation.

## Validation, resource use and provenance

- 25 focused tests pass, covering visual coordinate provenance, memory boundaries/gaps,
  structured dispatch, generic surface-grasp admission, and timing-report generation.
- Browser validation passed for all eight execution videos, the separate diagnostic video,
  four operator images, 227 sampled input images and 127 document links, including the main
  tool logs. The page also passed at a 360-pixel viewport.
- Real recorded projection marks preserve exact source samples from both cameras, with native
  reprojection errors below 0.001 pixels. These tests issue no robot commands.
- All 598 submitted model contexts have matching archived image hashes. The audit found no
  command-execution, file-edit, MCP, or web-search items and no known private-evaluation markers
  in the supplied payloads. Protocol isolation and source integrity records are retained.
- After the frozen study, invalid projection names now list the available IDs and coordinate
  convention; final `finish` requests use empty arguments and keep conclusions in `note`.
  A real Luna format-only check passed without a robot connection.
- A floor/inclusive-endpoint correction brings temporal boundary handling into agreement with
  the reference rule. Replaying all 271 enhanced input contexts gives **zero changed retained
  selections** on these recordings. Sparse retention itself was left unchanged.
- The earlier montage pilot was stopped for a 36-pixel header/coordinate ambiguity. Its completed
  baseline and interrupted enhanced recording remain separate and excluded from these eight
  outcomes. The corrected study also paused after its first completed episode for a missing
  timing artifact; the reporting fix and explicit resume preserved that episode and ran the
  remaining seven without repetition. The source revision is recorded in the manifest.

The corrected study and diagnostic used GPU 1, a 32 GiB job limit, zero swap allowance, and the
existing memory/I/O/SSH watchdog. Peak experiment-accounted RAM was 23.28 GiB; minimum host
available memory was 29.38 GiB. The monitored jobs recorded zero OOM kills, and all 784 SSH
probes succeeded. A host-wide I/O spike reached 65.5%; sampled job throughput was low during
inspection, but the spike’s system-wide cause was not established. GPU-device peak memory
was 39.32 GiB including other users’ jobs, not an isolated allocation for this experiment.
All experiment and diagnostic workers are stopped.

No numerical C++ code changed, so no planner rebuild was needed. A full repository test suite
was not run; validation targeted the modified interfaces and the real eight-case integration.
The strongest next work is generic contact/approach consistency and measured object/hand/goal
feedback. More task-specific directions or learned coordinate recipes are not supported by
these results.
