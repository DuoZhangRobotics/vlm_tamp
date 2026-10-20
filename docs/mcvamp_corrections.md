# MC-VAMP microwave corrections, 15 October 2026

These are targeted development diagnostics. Preserve the original ten-episode
comparison in `results/mcvamp-microwave-frozen-20261015`; do not merge subsequent
state-11 trials into a new frozen benchmark success rate.

## Direct 90-degree planning

`run_mcvamp_microwave.py --full-task-only` requires a path to the complete
remaining task endpoint. Opening starts at 90 degrees. This disables the 20- and
10-degree fallback both before grasping and between completed trajectories, and
tries up to 24 initial grasp poses. Endpoint IK proposals are now scene-checked
before they consume the feasible-goal quota. Previously four rejected proposals
could prevent trying the rest of the wrist orientation menu.

The state-11 direct-90 trial failed before manipulation. Failure to find an
admissible terminal posture in this finite search is distinct from a native
MC-VAMP connecting-path search failure and does not prove infeasibility.

## Gripper collision geometry

The calibrated 12 mm grid-cell balls overhang the actual convex Panda palm.
The frozen postclosure state-11 rejection included static samples 9--17 mm
outside the palm. Balls now serve only as a CAPT broad phase. A hit is confirmed
against the configured convex halfspaces using a shared hashed point index.
Moving-door and selected-contact clouds are queried in their hinge reference
frame. The stationary cloud stays stationary.

Contact-rich trials use zero additional gripper padding. The closest static
sample in the rejected closed state is 0.64 mm outside the actual palm, so adding
2.5 mm padding still rejects that posture. The arm's native sphere checker keeps
the existing 2.5 mm point radius. Configured fingers may contact the selected
handle and bounded mount region. Palm and arm penetrations still reject plans;
unrelated stationary and moving points are not exempted.

The exact saved start now passes. Tests include a real palm penetration, an
unrelated moving obstacle, a palm obstacle transformed by the hinge, and the
existing finger-contact permissions. The old native library and build record
are preserved under `.runtime/mcvamp-microwave/versions/`.

## Planning and execution

MC-VAMP's adapter imposes a fixed hinge radius and height for the gripper contact
point, jaw alignment or bounded tilt, bounded progress, and free wrist twist.
The output is a joint path, sampled and checked during planning. A timed joint
PD controller executes every issued reference without execution-time collision,
constraint, tracking-error or progress vetoes. The simulator handles contacts.

A prescribed end-effector curve needs a path-progress coordinate in planning
and continuous IK or differential IK to track it in time. MC-VAMP is a geometric
planner, not a tracking controller. The current custom finite-difference
projection adapter does not preserve all of upstream's SIMD projection speedup.

The new unconstrained choice of IK branch is recorded separately from a paired
diagnostic that replays the original RGB-D-derived approach/grasp. That paired
diagnostic verifies initial RGB, proprioception, handle cloud and sampled path
admission before motion. It does not replay manipulation: postclosure contact
paths are planned afresh from observations. Both remain physical-grasp trials,
with no attachment, ground-truth object geometry, or execution veto.

## Luna at 12k

`retry_luna_12k_microwave.py` makes one authorized 12k/high request for each of
closing states 21, 31 and 41, preserving the exact original numbered RGB image
and prompt. The shared durable ledger enforces the cumulative $2 ceiling and
prevents automatic retries, including uncertain requests. Completed responses
are cached. All three calls completed, and the existing sensor-only numerical
handle and full-pad support checks passed. These are intermediate checks, not
physical success or verification of the true hinge location.

Output tokens were 4,252 / 7,879 / 6,759; reasoning tokens were 4,108 / 7,678 /
6,562. Additional conservative spend was $0.010162, cumulative $0.23347075,
remaining $1.76652925. The output cap includes hidden reasoning and visible
text; see https://developers.openai.com/api/docs/guides/reasoning.

Every physical run invokes the existing retention policy after worker shutdown.
Keep videos, action/outcome logs, source pins and deduplicated lossless numerical
planning inputs; remove intermediate sensor archives and duplicate image frames.
