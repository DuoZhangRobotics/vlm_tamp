# What the current visual interface actually does

This describes the implementation used in the
[live LIBERO landmark-feedback comparison](luna_landmark_libero_20261026.md).
It borrows selected VISTA ideas; it is not a port of the full VISTA harness.

## Saved observations versus model input

At each captured observation, the service saves twelve configured camera views
at their captured 512×512 resolution as lossless PNGs. It also saves calibrated
depth/world-point arrays, camera calibration and robot state. Captures occur at
reset, explicit observation calls and after physical actions. They are not a
continuous twelve-camera RGB-D recording at every control step. Rollout video
is recorded separately for review.

Files live under `robot_data/attemptNN/obsNNN/` in the actor workspace. Observation
IDs correspond to service operation IDs, so there can be gaps between captures.
The compact reply lists two preview paths; the full reply lists all cameras.
Luna must open the desired images with its viewer. Keeping all files available
on disk does not place all their pixels in the model context. The saved history
can be revisited during the episode, but reading/viewing it still incurs model
and tool work. Rendering and storing the twelve views also has a real cost.

See [capture](../scripts/serve_subscription_robot.py) and
[compact observation display](../src/vlm_tamp/compact_robot_interface.py).

## `visual.py`

The public `visual.py` is a copy of
[visual_inspection.py](../src/vlm_tamp/visual_inspection.py). It has two commands:

- `history` lists saved observation IDs, control counts and camera names. The
  default page contains eight observations; a page is limited to 32.
- `inspect` selects one to four saved views and optionally crops them using
  original-pixel rectangles. It produces a labelled comparison image and
  metadata mapping displayed pixels back to the source coordinates.

The views can be different cameras at one moment or the same camera at different
moments. Enlarging a crop uses nearest-neighbor interpolation and adds no
observed detail. Original crop files are also retained. The returned comparison
path must be opened in a separate image-viewer call in the current implementation.

The helper performs no target detection, hinge estimation, motion measurement,
robot RPC or simulator-state read. It leaves depth/calibration and planning
geometry unchanged. It is convenient retrieval and presentation, not a
perception model. Actor-selected crops are not automatically correct targets.

In opening state 10 with landmark feedback, the four requests were one rejected
malformed request, one history request, and two successful inspections. Both
successful inspections used observation 15: a four-camera montage and a
two-camera crop comparison. This demonstrates multiview/crop use; it is not
evidence of an explicit before/after montage in that episode.

## Two different meanings of prediction

The prompt asks Luna to briefly state an expected observable action effect,
execute its chosen manipulation, compare the returned evidence and revise its
working assumption. For example, it may expect an open-finger approach to leave
the hand beside the handle with the fingers clear for closure. This is a short
public hypothesis, not a generated future image, physics rollout, hidden
reasoning transcript or separate reviewer. The written expectation does not
itself alter the planning geometry. There is no cross-episode memory update in
the current evaluation.

This is a prompt instruction, not an enforced validator. No separate model
checks the expectation before permitting motion. Luna can still misinterpret
the images or revise its hypothesis incorrectly, as the closing-state-9 baseline
demonstrated. The live prompt does not require VISTA-style `WORKING.md`/`GUIDE.md`
memory maintenance; the supplied guide is frozen for this comparison.

The separate numerical label `hand_motion_prediction` is an existing fallback
in the robot service. During contact manipulation it estimates a door transform
from hand motion around the estimated hinge, assuming contact retention. An
accepted RGB-D pose estimate takes precedence. If the visual fit is unavailable,
the fallback can update planning geometry even though actual object motion is
unconfirmed. Ordinary free motions without a contact assumption do not use
that fallback.

The landmark-feedback treatment exposes measured tracked-point displacements
before the hinge fitter filters near-stationary matches. This can challenge a
hand-predicted movement. It does not prove object identity or retained contact,
because the selected region can contain background or incorrect correspondences.
It also does not automatically veto execution or change the fallback policy.

## What the current comparison isolates

Both arms have saved observations, `visual.py`, the same brief prediction and
comparison prompt, Luna/high, and the same frozen guide and planner. Only the
optional numerical landmark summary and its availability documentation differ.
The experiment therefore measures adding that feedback within the shared
VISTA-inspired setup. It does not measure full VISTA versus an unmodified agent.

For the first matched opening case, both arms succeeded. Baseline used 348
controls and 419.05 actor seconds; feedback used 286 controls and 658.11 seconds.
The service-time difference was 10.29 seconds, while time outside the service
differed by 228.78 seconds. The latter includes model processing, local/image
tools, scheduling and transport, not pure reasoning. The six summary calculations
totaled 0.00177 seconds, nested inside the existing fit cost. This single pair
does not establish an overall control/time trade-off.
