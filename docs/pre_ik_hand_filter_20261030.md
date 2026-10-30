# Filter hand/environment collisions before IK

The user correctly distinguished hand-pose collision from arm-posture
collision: changing IK cannot repair a collision of the fixed proposed hand
pose itself. The live sequence is now:

**Target-cloud GraspGen proposals → open-hand/full-scene filter → IK and
whole-robot admission → approach → physical closure feedback.**

The hand filter transforms the calibrated open gripper to the proposed TCP
pose and uses the same native scene indexes, convex geometry and contact
permissions as the planner. It does not solve IK or make an arm-reachability
claim. It introduces no collision margin or dense path-validation pass.
Returned IK configurations still undergo whole-robot scene checks, including
their actual FK residuals relative to the proposed hand pose.

The shared native function `gripper_scene_clear` now serves both this filter
and the planner's full-robot collision checks. `NativeScene.validate_hand_pose`
converts the world TCP into the calibrated native EEF frame. The selector calls
it before `ik.solve_many`; hand-rejected proposals issue no IK query.

## Eight-pose audit

The original saved open37/seed-1 batch was checked without generating new
grasps, running IK or moving the robot:

| Candidate | Hand-only result | Blocking hand part |
|---|---|---|
| 1 | Clear | — |
| 2 | Rejected | Palm / stationary scene |
| 3 | Rejected | Palm / stationary scene |
| 4 | Rejected | Palm / stationary scene |
| 5 | Rejected | Finger 2 body / stationary scene |
| 6 | Rejected | Finger 1 body / moving noncontact surface |
| 7 | Rejected | Finger 2 body / moving noncontact surface |
| 8 | Clear | — |

The first audit took **1.34 ms for all eight poses**, excluding shared scene
construction. The paired repeat gave the same outcomes at both the original
requested hand poses and the FK poses of the recorded IK solutions. Candidate
2–4's original full-robot reports stopped at arm collisions; the independent
hand query reveals their additional palm collisions. The old report's first
blocking pair did not enumerate every collision.

Candidate 8 remains hand-clear but is the previously diagnosed empty grasp.
This illustrates that scene clearance alone does not establish target capture.
The current score ordering selects candidate 1, so candidate 8 is not executed
in the successful run.

Audits are saved under `results/open-hand-filter-audit-20261030` and
`results/open-hand-filter-paired-audit-20261030`. The latter is reproducible with
`scripts/audit_grasp_hand_filter.py` and contains pose-file hashes and the paired
nominal/FK checks. No native object geometry is used in the hand filter.

## Physical repeat with the final filter

`results/pre-ik-hand-filter-20261030/watch.html` contains the video and timing.
The exact archived GraspGen poses were reused. Source-integrity checks passed.

- Candidate 1, score 0.9858, selected after one hand query, one IK request and
  one path query.
- Hand filter: **0.341 ms** for that candidate.
- Grasp call: **3.591 s**, including cached-proposal lookup/provenance and IK
  startup. Path call **0.128 s**; native RRTC solve **0.082 s**.
- Extra dense validation **0 s**.
- Pad gap after closure **21.757 mm**; measured hand rotation **9.939°** and
  observed door rotation **9.449°** in the requested 10-degree probe.
- This is a local component repeat, not a full-task or fresh Luna success rate.

The preceding [open-hand admission and search correction](open_hand_grasp_admission_20261030.md)
remains in place: no hypothetical closed-hand veto, contact-aware grasped
manipulation, and ordinary RRT-Connect exploration for free approaches.

## Verification

Native tests check that the filter detects a palm at a translated proposed
pose without IK, preserves selected-pad contact permissions, and can pass a
hand pose while the separate arm check rejects an arm/environment collision.
Selection tests verify that a rejected hand pose never calls IK. The physical
repeat verifies the full sequence. The native adapter was rebuilt. **130
focused tests passed**, including the broader contact, sequence, schema and
timing checks.
