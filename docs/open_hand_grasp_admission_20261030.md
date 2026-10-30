# Admit the open approach, then establish contact physically

The final pipeline also [filters hand collisions before IK](pre_ik_hand_filter_20261030.md),
as requested in the user's follow-up. The same top grasp passed another local
physical probe with that filter enabled.

The user identified that candidate 1 should not be rejected because the
hypothetical closed hand contacts the handle. The live GraspGen selector now
checks the **open-hand endpoint and approach** against the observed scene.
It does not construct or veto a hypothetical closed-hand configuration.
Closure is executed physically, followed by measured gap and object-motion
feedback. No grasp pose was manually shifted or replaced.

For subsequent grasped manipulation, finger bodies and pads may contact the
selected handle patch. The arm and palm remain checked, as do finger contacts
with unrelated surfaces. The broader finger/handle permission is tied to a
commanded closure and an associated handle model; it is not a global collision
exception. Open approaches and push/hook contact policies retain their own
contact rules.

## Point-cloud inputs: what was and was not missing

The generator receives the selected target/handle cloud. The native checker
receives the broader observed workspace, split into stationary, moving and
selected-contact points. For this recorded case the native counts are 43,607,
2,692 and 1,067 respectively, after voxel sampling and removal of observed robot
surfaces. Scene context was not absent from IK/approach admission.

The [official GraspGen demo](https://github.com/NVlabs/GraspGen/blob/main/scripts/demo_collision_free_grasps.py)
uses the same broad architecture: proposals from the object cloud, then scene
collision filtering. Its [sampler](https://github.com/NVlabs/GraspGen/blob/main/grasp_gen/grasp_server.py)
takes object points; it does not take a complete robot arm configuration.
Supplying the whole scene as the object would change the generation target and
does not guarantee collision-free arm postures. Arm collision depends on the
IK configuration, so full-scene robot checking remains necessary after IK.

The network input was not changed in this experiment. The exact eight saved
GraspGen poses were reused to isolate the admission and search corrections.

## An additional search defect exposed by the corrected admission

The first replay accepted candidate 1's open endpoint, but RRTC did not find a
path. It reached 3,000 iterations in 39 ms with only 20 collision-state checks.
The selector eventually chose the same poor eighth candidate and closed empty.
That attempt is preserved at `results/open-hand-grasp-admission-20261030`.

Fixed-query, CPU-only comparisons retained the identical start, goal and scene:

| RRTC configuration | Iterations | State checks | Native solve | Outcome |
|---|---:|---:|---:|---|
| Dynamic-domain pruning on | 3,001 | 20 | 39 ms | No path found |
| Same pruning, larger limit | 100,001 | 20 | 44 ms | No path found |
| Same pruning, much larger limit | 1,000,001 | 20 | 64 ms | No path found |
| Dynamic-domain pruning off | 10 | 44 | 82 ms | Path found |

The failure was the exploration-pruning behavior for this near-contact query,
not merely the iteration limit. Free-motion queries now use ordinary
RRT-Connect exploration with that pruning disabled. The collision checker,
observed points, planner sampling resolution and returned-path policy remain
the same. Constrained CRRTC settings are unchanged. The extra dense collision
and projection pass remains removed.

Comparison artifacts are under `results/open-hand-approach-search-20261030`
and `results/open-hand-approach-no-pruning-20261030`. Their native-input records
include adapter source hashes. `scripts/probe_open_grasp_approach.py` reproduces
the saved query with the currently built adapter.

## Physical result

`results/open-hand-grasp-exploration-20261030/open_microwave-state037` uses the
same archived eight-pose batch and public region selections, with exact metric
input matching. Source-integrity verification passed.

- Selected **candidate 1**, discriminator score **0.9858**, closest admitted IK.
- One IK request and one path request.
- Grasp call **3.620 s**: cache/provenance **0.674 s**, scene preparation
  **0.018 s**, selection **2.920 s** including first-query IK setup.
- Path call **0.137 s**, including native RRTC solve **0.088 s**.
- Extra dense validation **0 s**.
- Approach endpoint tracking error **1.019 mm**.
- Pad gap after closure **21.757 mm**, versus observed bar diameter **19.031 mm**.
- Requested 10-degree probe: hand motion **9.939°**, observed part motion
  **9.449°**. The pad gap stayed approximately **21.8 mm** afterward.
- Offline native records show contacts from both finger pads to the microwave,
  with no arm/palm environment contacts in this trial.

This supports a successful local grasp-and-motion probe. It is one component
trial, not a full opening-task success, a fresh Luna evaluation, or a claim of
general repeatability. Native task success stayed false because the trial
stopped after the short probe. Runtime public evidence remains cautious about
certifying a pinch; the physical contact identities above were read only by
the offline report.

The earlier 0.479 mm hypothetical finger-body rejection did not justify
discarding this grasp. The physical test supports the user's correction.

## Code and validation

128 focused tests passed, including native contact-pair tests for both approach
and grasped modes, open-hand-only selection, the recorded non-direct RRTC query,
sequence execution, contact skills, timing and public schemas. The native adapter rebuilt successfully. The report is
`results/open-hand-grasp-exploration-20261030/watch.html` and is linked from the
earlier collision and empty-grasp watch pages.

Core changes are in `graspgen_selection.py`, `_select_contact` in
`serve_subscription_robot.py`, `measured_robot_scene` in `sequence_execution.py`,
the contact permissions in `mcvamp_contact_numerical.py`, and the free-motion
RRTC setting in `native/mcvamp_contact_numerical.cc`.
