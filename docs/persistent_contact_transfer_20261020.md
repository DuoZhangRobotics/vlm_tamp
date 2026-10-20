# Persistent episodes with general contact choices

The previous agent could call `finish` while native success was false and
physical steps remained. The service accepted incomplete/uncertain endings,
and a normal final response also ended the one-turn launcher.

The new opt-in persistent mode changes both layers. The service denies early
finish, reports actual remaining step/time budgets, and stops at native success
or the control-step limit. The launcher resumes the same ChatGPT-authenticated
Luna thread when it returns early. Automatic continuation messages contain only
budget/status reminders. No reset or new task-specific guidance is introduced.
Runtime/account failures are reported as infrastructure failures, not relabeled
as exhausted budgets.

The client also admits only one in-flight numerical request. Budget snapshots
remain available locally while a planner is busy, and `await_request` retrieves
an existing reply without issuing the operation again. An initial evaluation
exposed queued status requests and a duplicate plan; it was interrupted at 60
setup steps before task-manipulation commands. Its four-slot manifest, one
started attempt, three unstarted slots and failure evidence remain preserved
under `results/persistent-contact-transfer-20261020`. It is not pooled with the
fresh corrected cohort.

Limits remain one episode, 600 control steps, and 1,800 seconds of agent/tool
time per initial state. Already-issued requests are allowed to finish at the
wall deadline; no executing trajectory gains a collision/progress veto.
An endpoint can therefore include in-flight completion and cleanup time.

The new prompt describes general multi-stage manipulation and contact choices.
It does not prescribe a particular object surface, direction, orientation,
location, or recovery sequence. Previous agent-authored memory is retained
unchanged; its local successful recipe is explicitly a hypothesis.

## Generic contact interface

`select_surface_contact` uses current RGB-selected pixels and depth to choose
a bounded contact patch. All other observed points remain obstacles. Configured
finger contact with the patch is permitted; palm and arm checks remain. Robot
pad landmarks come from calibrated robot geometry. No handle cylinder, hinge,
grasp closure or virtual attachment is required.

`plan_pose` uses the existing cuRobo IK and MC-VAMP free-motion collision
checker to plan to a caller-selected TCP pose. It supplies no target direction
or task policy. The caller uses `execute` and chooses the gripper command.
After motion, the surface snapshot must be refreshed from current observations.
No hand-driven prediction of object motion is applied in this mode.

The selected patch is an observed surface model, not a guarantee of useful
contact, force closure, physical trajectory tracking or unseen-space clearance.
Existing geometry/contact limitations remain. The articulated handle interface
continues to use the shared implementation from the previous comparison.

## Experiment

`scripts/run_persistent_contact_transfer.py` runs initial states 1, 11, 21 and 31,
seed 0, one episode each, with no across-case memory updates. The new registered
profile points to the same prior method and guide, plus the general continuation
prompt. Persistence, prompt and surface-contact capabilities change together;
this experiment does not isolate their individual contributions.

The results are compared with the preserved shared-tool transfer. Native
simulator records determine success even if later postprocessing fails.
Started failures remain in the denominator. Each run retains observations,
plans, issued commands, videos, source hashes, prompts and continuation messages,
and removes temporary workspaces and redundant intermediate observations.

The corrected frozen cohort is
`results/persistent-contact-transfer-v2-20261020`. It achieved **3/4** native
successes: state 1 exhausted 600 steps without success; states 11, 21 and 31
succeeded at 339, 541 and 297 steps. There were no premature episode endings or
post-success commands. Each case used one reset and one agent reasoning turn;
automatic same-thread continuation was available but was not needed in these
four runs.

State 21's initial long sequence did not succeed. The agent continued with new
grounding, model revisions and smaller arcs, reaching success during its final
2-degree request. It explored surface-contact pose plans but did not execute
one in that case. State 1 selected surface patches 22 times and requested 29 pose
plans, executing six trajectories in surface mode. During one of those motions
the native door angle increased by approximately 7.6 degrees; that observation
does not certify a specific inside-push maneuver. The case still ended at only
about 64.9 degrees, below native completion. No specific surface or direction
was provided in the prompt.

The previous shared-tool transfer scored 1/4. This small bundled rerun is useful
development evidence, not an isolated causal attribution to persistence or
pushing, and not a general success rate on new assets. Backend numerical calls
are counted separately from local status/reply queries and model reasoning
turns; many numerical calls came from agent-written search loops.

Before physical evaluation, targeted tests check finish denial, genuine terminal
budgets, restricted same-thread resumption, bounded contact selection and native
free-motion planning. A separate two-turn subscription smoke test verifies
context retention without running a physical episode.
