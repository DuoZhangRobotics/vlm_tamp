# Live Luna comparison: legacy grounding and the new motion toolkit

All six registered episodes completed. **Legacy succeeded in 2/3 cases;
motion-assisted succeeded in 1/3.** Luna used the new route in all three
treatment episodes, applying five estimates and issuing seven successful
execution requests for plans built under those measured models. Adoption worked;
this small development cohort did not establish a task-success improvement.

Open the same [watch.html](../results/luna-live-hinge-comparison-20261027/watch.html)
in VS Code and use Live Preview. It contains all six outcomes, videos, latest
observations/perception outputs, prompts, tool I/O, diagnoses, and nested timing.

| Task / initial state | Legacy result | Motion-assisted result |
| --- | --- | --- |
| Open / 13 | Success; 465 controls; 641.1 s | Success; 380 controls; 499.5 s |
| Close / 17 | Success; 535 controls; 750.4 s | Failure; 600 controls; 1494.9 s |
| Close / 49 | Failure; 581 controls; 1800.0 s | Failure; 600 controls; 1792.2 s |

Times are complete actor sessions, including robot-tool waits and final notes,
not pure model inference. Both arms had one episode, 600 controls and 1800 actor
seconds. Case 49 legacy reached the wall budget; the two motion-assisted closing
failures reached the control budget. No failure was replaced or silently rerun.

- [Registered protocol and exact prompts](../results/luna-live-hinge-comparison-20261027/protocol.json)
- [Outcome CSV](../results/luna-live-hinge-comparison-20261027/trial-results.csv)
- [Timing CSV](../results/luna-live-hinge-comparison-20261027/time-breakdown.csv)
- [Timing/outcome figure](../results/luna-live-hinge-comparison-20261027/outcome-timing.png)
- [Final source/input/controller audit](../results/luna-live-hinge-comparison-20261027/final-audit.json)

## What was compared

This is a comparison of complete agent/tool workflows. Both start from Luna's
visual hinge/contact grounding. The new condition adds independent part/motion
tools and generic encouragement to measure, check and apply a hinge hypothesis.
The legacy condition retains its existing RGB-D motion feedback and model
updates. It is **not** a permanently frozen visual guess: in opening state 13,
Luna copied the legacy motion estimate from event 19 into `set_model` at event
23. Its initial hinge error was 23.2 mm; that later estimate was 0.94 mm off.

Each episode used a fresh Luna context, the same cuRobo v2/controller stack,
3 mm arc-position tolerance, native-success stopping, landmark feedback and
surface-contact/visual-inspection capabilities. No worked example, prior guide,
teacher, reflection reviewer, task-specific pixels or corrective human hints
were supplied. Notes were retained but not promoted into another actor's memory.

The cases were already used in development and share a microwave asset family.
There is one episode per route/case. Actors chose different regions, grasps,
speeds, motion horizons and grasp constraints. This does not isolate the causal
effect of hinge coordinates alone or establish statistical/generalization gains.

## How Luna can use the new route

`--part-motion-tools` on `scripts/run_subscription_agent_pilot.py` supplies the
generic introduction at the beginning of `tools.txt`, individual help/schema
files, and these live operations:

1. `segment_parts` supplies masks from Luna's image box/clicks.
2. `associate_views` optionally checks/transfers the observed surface to another
   calibrated view.
3. `track_part_motion` handles saved RGB-D views. `track_video_motion` instead
   reads recorded agent-view JPEGs between the supplied depth timestamps,
   retaining up to 64 RGB frames and all requested depth anchors. It performs
   no new rendering and invents no intermediate depth.
4. `estimate_and_verify_joint` fits an upright hinge and optionally checks another
   motion interval. Its result is a registered estimate ID.
5. `apply_joint_estimate` installs that ID into the current articulated model
   and invalidates old plans. Luna then requests fresh plans and chooses actions.

Applying an estimate preserves the signed task direction and current contact
geometry. It requires current-episode evidence reaching the current control
step. Contradicted estimates are rejected. A usable but unverified/inconclusive
estimate requires an explicit `allow_unverified=true` and remains provisional.
It does not certify grasp retention or repair a stale contact model. Re-grounding
replaces the model. Manual hinge replacement clears measured provenance.

The introduction encourages this route when hinge geometry is uncertain or
predicted motion disagrees with observations. It supplies no successful regions,
angles or trajectories. Initial probing still needs a usable contact/bootstrap
hypothesis; this study did not test manipulation with no initial visual hinge.

The opt-in `--no-example` flag was added for this study so both conditions receive
tool contracts without a worked solution. It is independent of the motion-tools
flag. Existing worked-example behavior remains available when requested.

## What the live runs revealed

**Open 13:** Both succeeded; the new route was faster and used fewer controls.
However, its provisional and later motion-supported applied estimates had
60.8 and 36.8 mm native hinge errors. The latter passed a roughly 7.57-degree
measured-motion check with 2.54 mm RMS. The native axis did not drift. Passing
that consistency check therefore did not establish accurate hinge position.
Both actors also chose different motion strategies and speeds.

**Close 17, motion-assisted:** Initial visual geometry was already about 5.0 mm
accurate, and the applied provisional estimate was about 6.0 mm accurate.
The first 5-degree and 15-degree arcs produced approximately 5.1 and 15.3
degrees of actual door motion. Both pads remained in contact at the boundaries
of every executed arc. Planning nevertheless stalled before the new estimate
was applied; recovery later consumed the remaining budget. Only about 26.5
degrees of closing was achieved, leaving the door about 62.5 degrees open.

Event 34 gives a specific planner explanation. In point-contact mode, candidate
endpoints passed position-constraint projection, joint bounds and the observed
scene collision check. They were rejected solely for requiring 28.5–29.6 degrees
of hand reorientation while the request allowed only 8 degrees. At this posture,
q7 was about 4.7 degrees below its upper limit. These are valid endpoint checks,
not proof of a complete connecting path or continued physical grip. The
[recorded evidence](../results/luna-live-hinge-comparison-20261027/planner-stall-evidence.json)
retains the values. Other rigid/jaw-axis requests rejected unsuccessful IK
proposals before polishing, including some with 3.6–3.8 mm contact-position
error against a 3 mm capture cutoff. Other pose constraints might still fail.

The successful legacy state-17 run mostly used point-contact requests and later
the existing `refined` flag to increase wrist allowance. That flag does not
refine a hinge. Its marked hinge remained about 85.8 mm off. This reinforces
that task outcome depends on constraint/posture choices and feedback, rather
than hinge accuracy alone. It does not prove that any inaccurate hinge works.

**Close 49, legacy:** No robot/microwave contact or native door motion was
recorded. Grasp planning took about 641.8 seconds, and free-pose recovery
requests repeatedly failed observed-scene/path checks. No articulated arc ran.

**Close 49, motion-assisted:** The door moved from roughly 107.3 to 49.8 degrees
open. Applied estimates had 4.32 mm error (motion-supported) and 1.84 mm error
(provisional). Further planning/recovery did not finish within 600 controls.
Finger-pad contact was absent at the checked arc endpoints, so the progress
does not establish a retained grasp. The page preserves the contact and motion
diagnostics separately from native task success.

## Timing interpretation

Across the three episodes in each route:

| Actor-session component | Legacy total | Motion-assisted total |
| --- | ---: | ---: |
| Outside robot-tool calls | 1215.8 s | 2169.3 s |
| Inside robot-tool calls | 1975.8 s | 1617.2 s |
| Total actor-session time | 3191.5 s | 3786.6 s |

Outside-service time includes model inference, image viewing, local analysis,
scheduling and transport. Failure/recovery behavior and different numbers of
calls also affect it. These totals include failed episodes; they are not a
pure speed comparison among successful runs.

The webpage separates overall request times from nested GraspGen initialization
and generation, full-pose IK worker/transport/admission, approach search and
resampling, contact-probe/arc goals, fallback searches, execution, RGB-D capture,
compression, segmentation, CoTracker and hinge fitting. Residual subtraction
buckets are labeled. Sequence execution stages are included, and nested timers
must not be added to overall elapsed time again.

## Validation and next decision

The focused suite passed 74 tests. Three no-example interface variants and a
real recorded 58-RGB/five-depth wrapper check passed before the cohort.
All 558 registered runtime files and the perception weights remained unchanged.
All six trials passed isolation, feedback-field, no-hint, trajectory and timing
audits. Issued references matched recorded execution, no extra execution veto
was introduced, and no controls followed native success.

All three pairs matched exactly on initial metric arrays, calibration and robot
state. RGB matched exactly for state 49. Opening state 13 differed in 18 wrist
image channel values; closing state 17 differed in one arm-lower-left channel
value. Each difference was 1/255. These differences are retained explicitly
in the paired-input audit rather than declared exact.
The report was checked as an ordinary HTML file, including image links and a
narrow layout. Live Preview needs only the single `watch.html` file.

Keep the new toolbox available as an opt-in capability. The data do not support
promoting it as the default route yet. The concrete next investigations are
planner wrist/admission policies at the recorded stalled posture, and hinge
observability/validation that can distinguish a consistent short-motion fit
from an accurate axis. Preserve final contact tolerance and collision checks
when testing different planning allowances. Do not solve scored cases with
manual corrections or promote unsupported actor claims into learner memory.
