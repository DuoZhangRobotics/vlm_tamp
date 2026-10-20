# MC-VAMP for drawers and stove controls

This experiment follows the seven-task regression in `results/all-prior-tasks-frozen-v3-20261015`. It compares the three previously tested drawer tasks and stove turning on official states 1, 11, 21, 31 and 41, seed 0. Every scheduled episode counts; native LIBERO success is the outcome. These states have already been inspected, so this is a regression rather than held-out generalization.

## Why the earlier pipeline failed

All 300 top-drawer-opening manipulation probes and all 117 stove probes were rejected for overlap between a fabricated convex moving-object envelope and stationary scene points at the initial pose. Their manifold residuals passed. The generic 'IK posture search exhausted' exception hid that distinction. The nearest-24 candidate restriction also preceded approach-direction filtering, sometimes eliminating every usable GraspGen pose.

## Changes

The new `mcvamp_mechanism` native adapter is separate from the frozen microwave adapter. It uses the same pinned upstream MC-VAMP revision and Panda calibration. The upstream library is unchanged.

For a drawer, the grasp point stays on an estimated line while its grasp orientation is preserved. Sliding progress is inferred from robot forward kinematics. The native collision callback translates the observed moving points by that progress. MC-VAMP C-RRTC searches the constrained joint path; cuRobo supplies terminal IK alternatives. VAMP RRTC searches the free approach through a standoff. This replaces the old straight interpolation / IK-continuation admission.

For a stove knob, the contact point and gripper orientation rotate together about the RGB-D-estimated control axis. Progress comes from orientation, so a grasp centered on the axis can still specify a turn. The visible footprint supplies a center estimate instead of treating a selected surface pixel as the mechanical axis. Protruding measured control points supply the grasp cloud. No asset axis, origin, segmentation or pose is a policy input.

The measured-surface collision model checks robot arm/palm against stationary and moving depth endpoints. Functional fingers may contact the RGB-D-selected handle/control and a bounded adjacent face. It creates no solid payload hull. **Mechanism-versus-environment collision is not certified or vetoed in this experiment**; simulator physics resolves drawer–cabinet, knob–base and other mechanism contacts. Unobserved space remains unmodeled. This matches the scope of the current microwave MC-VAMP adapter and is not a complete environment collision certificate.

The 24-candidate budget now applies after approach/closing-direction eligibility. Up to 96 IK postures are tested. GraspGen remains the Panda diffusion/discriminator model with 200 proposals and 100 retained scores, seed 0. Every episode performs fresh selection. Model proposal caches are private to each run and deleted after their exact outputs and checkpoint provenance have been saved in the episode.

The failed closing response with whole-sheet coordinates is repaired only when one uniform translation from the declared camera's sheet cell makes every coordinate valid. This is a layout conversion, not a manual object selection. Its original response and conversion are retained. Other invalid choices still fail.

## Physical execution and direction diagnosis

The official joint controller uses kp 300. Reference-time execution issues every reference of a complete trajectory, with no runtime collision, progress, retention, constraint or task-success veto. Actual actuator limits and simulator contacts remain active. Gripper closure creates all attachment; there is no weld or pose assignment.

Drawer segments are at most 2.5 cm. Closing includes a fixed 5 mm planned seating allowance, introduced during development after a complete 15 cm command left the drawer about 1 mm from its native closed condition. This allowance is the same across all closing states, and is not computed from native joint coordinates.

The stove first executes an 8-degree probe in the VLM-selected direction. Depth-template registration estimates control rotation. If it does not support motion in that direction, a complete planned opposite-direction probe follows. A supported observed turn determines the continuation direction. If both observations are ambiguous, the runner records that explicitly and continues a bounded direction trial; it does not read native success or joint state to choose a direction. Subsequent nominal segments are 8 degrees. A symmetric or insufficient surface observation is not treated as a measured angle.

The original 600-control-step cap and five settling actions remain. Every issued trajectory is checked against the remaining budget before it begins and then runs completely. Completed trajectories may physically fall short; that is measured as an execution outcome. An unavailable subsequent plan remains a planning failure.

## Comparison and retention

This changes geometry, candidate ranking, goal completion and planning together; it is not an isolated planner speed ablation. Development attempts are stored separately. The final comparison freezes source hashes and snapshots before its first episode. Frozen historical sources and previous results remain intact.

`run_mcvamp_mechanisms.py` runs the trials. `report_mcvamp_mechanisms.py` compares native outcomes and retained videos. `validate_mcvamp_mechanisms.py` independently matches saved references to native action logs and tracking, checks every movie, verifies native scores, source snapshots and the cumulative $2 API budget. Completed cached Luna High choices are reused; no API retry is automatic.

After every episode, workers exit, movies are verified and redundant raw observations/JPEG sequences are removed. Initial/latest observations, numeric constraints, native scene inputs, candidates, selected grasps, trajectories, contacts, commands, outcome logs and source provenance remain. Planning pauses are omitted from videos; 20 Hz previews show simulation time, not pipeline wall time.

## Measured outcomes

The frozen 20-episode comparison completed with 15 native successes, against 5 in the matched baseline subset:

| Task | Baseline | MC-VAMP adaptation |
|---|---:|---:|
| Open middle drawer | 5/5 | 5/5 |
| Open top drawer | 0/5 | 5/5 |
| Close top drawer | 0/5 | 2/5 |
| Turn on stove | 0/5 | 3/5 |

The five failures were: closing state 11 (finger versus stationary-point goal rejection), stove state 21 (finger versus stationary-point grasp rejection), closing state 31 (full requested stroke completed, roughly 7 mm left open), stove state 31 (reached about 18 degrees, then the next endpoint was rejected for finger versus stationary-point contact), and closing state 41 (missing selected-camera field in the sheet-coordinate repair path).

After both frozen partitions completed and their sources/actions/movies were audited, the camera bundle was corrected. Its regression failed before the fix and passed afterward. A separate state-41 retry reached numerical grounding and grasp search, but failed grasp/approach/probe admission. The original 15/20 score is unchanged; the additional physical attempt is recorded separately. `post-run-camera-repair.json` records the exact before/after source hashes. Frozen source snapshots remain intact.

The final 61 relevant tests passed. All 21 comparison/retry movies and every saved joint reference were verified; execution-time veto count was zero. No additional API request was charged; cumulative API cost remained $0.25178425. All 27 episodes including six development attempts completed cleanup, reclaiming 8,564,087,453 bytes of temporary data. The report and its final validation record contain the per-episode evidence.
