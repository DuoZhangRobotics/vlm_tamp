# Bowl grasp geometry integration repair

Two reproduced integration problems contributed to the failed bowl episode. Missing closure
was not the initial failure: the baseline did not reach a usable grasp pose, while the compact
episode explicitly closed and lifted. Automatic closure is a separate interface improvement.

## GraspGen received the contact neighborhood instead of the selected target

Source: `luna-compact-transition-20261108/trials/03-place_bowl-s001-existing`, events 3–4.
Luna selected image region `[220,266,290,324]`, pixel `[256,294]`, contact radius 4 cm.
The backend used that radius to crop the cloud supplied to GraspGen. The selected region covered
the bowl, but the generator received only 864 points near the contact pixel. All eight learned
poses failed the open-hand scene check, on palm spheres, before any IK or path query.

In a matched diagnostic, both initial camera images were identical to the recording. Supplying
all 4,189 observed points from the same selected image region, with the same seed, eight-proposal
limit and collision scene, produced a reachable grasp on the third examined proposal. The robot
approached, automatically closed and completed a 12 cm upward retention probe.

Production change: `contact_snapshot` retains the full caller-selected region/mask for grasp
generation. The radius still limits the local contact patch and surface-normal fit. Every other
observed point remains in the collision world. The tool reply and public point archive distinguish
the grasp target from the contact patch. No automatic segmentation, target substitution, pose
adjustment, extra proposal batch or collision bypass was added.

A rectangle can include background: it is still Luna's selection, not guaranteed object identity.
An explicitly selected part mask remains supported. This test validates one input correction;
it does not establish general grasp reliability.

## Current-state cleanup disagreed with the planner's robot geometry

Source: the same baseline, grasp event 14 at control 243. Current-state point filtering used
physical robot meshes, while cuRobo and VAMP used their shared sphere model. A finger sphere
still overlapped 23 stationary-cloud points outside those meshes, so the measured start was
rejected at sphere 56. This was not an optimizer timeout or a failed self-collision check.

Production cleanup now also removes points inside the same current arm/palm/finger spheres
used by planning and IK. It uses measured arm joints and finger aperture, zero inflation,
and no future candidate pose. Raw sensor observations remain intact. The same recorded state
then passes, including the originally saved open-aperture planning model. The admission replay
executes no motion. This aligns the existing user-authorized current-state clearing policy;
it does not assert that the removed sensor points were physically false.

## Validation and evidence

Dashboard: `results/bowl-grasp-repair-20261108/watch.html`.

- `input-comparison/`: original contact-cloud request fails without task motion; complete-region
  input succeeds in acquisition and lift. All requests and the executed video are retained.
- `qualified-fix/`: the production code, without the diagnostic cloud override, executes the
  original first request successfully. Approach 60–147, closure 147–187, lift 187–268 controls.
  Jaw gap after closure: 4.09 mm; nine evaluator-only bowl contacts remain after the lift.
  Total wall time: 55.81 s. No placement was attempted in this component qualification.
- `current-start-diagnosis.json`: mesh-only rejection and repaired admission, including the
  saved original aperture. No motion and no privileged object geometry supplied to planning.
- `live/`: one fresh Luna/high bowl-state-1 evaluation with existing feedback, two native
  cameras, 600 controls, 1800 wall seconds, and no inherited memory or demonstrated coordinates.
  See its manifest for the measured task outcome; operator checks do not count as Luna success.

124 focused tests passed, including full-region preservation, native sphere admission, future
geometry retention, grasp selection, automatic closure, schemas and compact feedback. Scoped
Ruff and `git diff --check` passed. No native build was needed. Recorded test fixtures were updated
to complete current robot metadata because older reduced snapshots lacked calibrated finger-link
poses; no production fallback was introduced for missing metadata.

All physical tests use the existing 32 GiB job cap, zero swap, and memory/I/O/SSH watchdog.
Source hashes are frozen during each test. No other user's job is changed.

## Completed fresh Luna evaluation

The fresh episode at frozen revision `d28d710` still failed placement at 600 controls.
Its first grasp request succeeded, followed by measured approach arrival and 12 automatic
closure controls at control 165. Both fingers contacted the intended task bowl. This improves
acquisition in this episode; one stochastic rerun is not a success-rate comparison.

Luna then requested a finger's distal face at the measured plate point, using a point-only
contact-pose constraint. Three earlier placement queries had returned no IK solution. The
accepted robot-contact plan is not an object-placement plan: it does not specify the held bowl's
pose relative to the plate. During the resulting carry/descent:

- The gripper command stayed at hold; there was no release command.
- Opposing-finger contact with the bowl was first absent at control 202 (10.10 video seconds).
- Bowl–palm contacts appeared at 204 (10.20 s); finger–plate contacts appeared at 217 (10.85 s).
- The approach stopped at 266 (13.30 s), 15.731 mm short of its robot-contact target. Pad gap
  increased from 4.60 to 39.57 mm. The frame shows the bowl beside the plate under the palm.
- Luna noticed the failed arrival and widened gap at decision 18. A later reached grasp/closure
  ended without robot–object contacts at control 492. Subsequent recovery exhausted 600 controls.

The collision sequence and misplaced robot-contact target are confirmed; they do not prove every
alternative placement pose infeasible. The next deliberate interface experiment should express
the held object's goal relative to an observed support surface, preserving a measured hand–object
relation and using the existing planner. This was not silently added during the frozen evaluation.
Full placement and drawer manipulation remain unresolved; no success is inferred from the repairs.

The complete failed execution video is
`results/bowl-grasp-repair-20261108/live/trials/01-place_bowl-s001-existing/attempt01/initial/rollout.mp4`.
It has 601 frames at 20 Hz (30.05 s). No fault was injected. Placement planning had a 60 s budget
and solved; the physical carry stopped on the existing reference-arrival condition. The whole
episode stopped on its control limit, with about 847 wall seconds remaining after the final decision.

Actor wall time was 953.30 s, including 669.42 s model/client time. Reset took 96.45 s.
The first grasp tool took 86.49 s: GraspGen initialization 66.12 s, generation 6.33 s,
selection 11.32 s (including 10.76 s IK RPC), plus bookkeeping. Later grasp calls used the warm
model. These measurements do not demonstrate an end-to-end latency improvement.

Peak anonymous/shared job memory was 6.45 GiB; total cgroup memory reached the 32 GiB cap,
mostly cached model files. Host available RAM remained above 59.23 GiB, maximum sampled host
full-I/O pressure was 8.74%, and there were no OOM kills or failed sampled SSH probes. The service
exited successfully as an experiment runner, although the robot task failed. All workers stopped
and GPU 3 returned to its pre-run allocation of 59,567 MiB by other workloads.
