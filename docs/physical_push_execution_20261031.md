# Physical push execution

The user explicitly authorized simulation pushes to be attempted rather than
rejected by uncertain scene/contact predictions, including 10 cm requests. This
supersedes the previous controlled-push collision/visual-admission behavior.
The simulator's physical contacts remain enabled; no object state or geometry
is supplied to the policy or the stroke planner.

## Current tools

`push_stroke(direction_world, distance_m)` moves the current hand while holding
its gripper aperture. It requires no object model, prior contact, image region,
hinge estimate, preview, or contact search. The default is one complete requested
stroke. `continue_push` reuses associated contact/target feedback when available
and otherwise behaves as a current-hand stroke. The initial targeted `push` still
provides approach selection when the hand needs to reach a selected region.

Controlled push insertion and strokes use robot-valid IK references with physical
execution feedback. There is no second dense interpolation/admission pass for
these strokes. Scene predictions and failed object tracking do not veto them.
Reachable prefixes can execute when the whole requested translation cannot be
generated. Existing joint bounds, the 1 rad/s reference limit, and the episode's
600-control/1800-second limits remain. No new geometric padding was added.

Requested, actually issued reference, and measured hand travel are distinct.
`stroke_executed` means the command was attempted; it does not certify that the
hand traveled the entire requested distance or moved the target. There is no
3 mm post-execution completion gate. Object motion is `unmeasured` without a
target association and can be `unresolved` with one. Native task success remains
the goal criterion.

During longer strokes, one second with less than 1 mm of measured progress along
the commanded direction, while references advance at least 1 cm, returns
`physical_stall`. This is post-action robot feedback, not a point-cloud veto.
The current-hand reference remains usable for recovery. A stall is neither task
success nor an infrastructure interruption; episode accounting records it as a
feedback-terminated trajectory separately from a fully issued trajectory.

Targeted free approaches, grasped manipulation, hooks and push-and-release retain
their existing planning policies. A failed targeted acquisition is not a
permission check for the current-hand primitive.

## Observation and display

The existing wrist RGB-D camera is included in push feedback and is visible under
“Wrist camera — latest contact view” in the dashboard. Current-hand strokes need
only the global and wrist views. No wrist movie is rendered at every control;
the existing global video and exact frame/control index remain available.

Associated contact tracking retains its original multi-view reference across
strokes and reports incremental transforms from successive absolute fits. It
selects feedback views by observed match counts. The old implementation repeatedly
reinitialized from reduced captures, so a full-view retry could not use the newly
captured cameras. On the reported state-17 event 28, the original 8-pair/7-inlier
rejection becomes an accepted 18-pair/15-inlier fit from the retained full reference.
Tracking still remains optional for direction-only physical execution.

The dashboard distinguishes hand execution, object measurement and native task
outcome. It shows every push/push_stroke/continue_push call, timing and resource
measurements. The HTML remains static; status publishes at most once a minute
or at lifecycle boundaries. Manual display refresh remains the default.

## Validation and reproduction

The previous frozen Luna cohort completed at **2/12 native successes**. Its state-17
second closing run reached contact at control 196, then rejected further motion.
Three physical replays of that recorded prefix, without initializing an object
model or retained contact, each closed the microwave at **control 406** using the
current-hand primitive. The world direction came from Luna's original request;
no diagnostic coordinates or action sequence is supplied to fresh actors.

Each replay attempted consecutive 10 cm requests. Measured hand-reference travel
was approximately 94.57, 94.35, 59.75 and 6.81 mm; native success interrupted the
last stroke. Warm tool calls took about 3.1–3.2 seconds; first-call IK startup and
success-time cleanup are reported separately. Peak sampled job memory was
3.57 GiB; maximum full I/O and memory pressure were zero. These are local physical
repeatability results, not fresh-policy success rates.

A deliberately blocked downward stroke executed 22 controls before returning a
measured directional stall: references advanced 29.85 mm over the final second,
while the hand advanced 0.84 mm along that direction. The hand also displaced
sideways; do not interpret directional stalling as zero total 3D movement.

Regression validation: 233 passed, three missing historical-fixture skips.
The final physical test records `sources_unchanged: true`. The reusable evaluator
is `scripts/validate_physical_push.py`. Run it on GPU/EGL device 3 in a systemd
job with MemoryMax=16G, MemorySwapMax=0 and OOMPolicy=kill, after sourcing
`scripts/server_runtime_env.sh`:

```text
python scripts/validate_physical_push.py \
  --source /common/users/dz323/vlm_tamp/results/luna-contact-ownership-rerun-20261031/trials/baseline-r2-close_microwave-state017-600 \
  --output results/physical-push-reproduction \
  --through-event 13 --direction-event 14 --distance .1 --strokes 5 --repeats 3 --expect success
```

Use `--direction 0 0 -1 --strokes 1 --repeats 1 --expect stall` for the blocked
diagnostic instead of `--direction-event`. Keep diagnostic inputs outside actor
workspaces. Fresh Luna actors receive generic tool documentation only, start
without inherited notebooks, and keep the existing episode limits.

Validation videos/timings: `results/physical-push-validation-20261031/index.html`.
Fresh monitored trials: `results/luna-physical-push-rerun-20261031/watch.html`.
