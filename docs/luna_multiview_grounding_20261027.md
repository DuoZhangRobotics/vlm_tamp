# Luna-selected multiview grounding: tool implementation and capability pilot

The previous [reviewed-geometry diagnostic](verified_grounding_planner_20261026.md)
used assistant-selected image annotations to isolate planning failures. It was
not an improvement in Luna's autonomous grounding. Camera calibration in both
studies is existing sensor/robot calibration; neither the assistant nor Luna
estimated it.

This follow-up implements a tool Luna can use directly and tests Luna's own
selections on three newly captured initial states. No human-selected pixels,
corrected examples, old guide, teacher, or native object coordinates entered
any actor session. The result is useful but not reliable grounding yet:
one selected estimate passed the registered hinge screen, one case remained
unresolved, and one numerically consistent estimate was physically wrong.

- [Dashboard: prompts, selections, overlays, replies and Luna's notes](../results/luna-multiview-grounding-20261027/watch.html)
- [Registered protocol](../results/luna-multiview-grounding-20261027/protocol.json)
- [Exact common prompt](../results/luna-multiview-grounding-20261027/actor-prompt.txt)
- [Post-run native scoring](../results/luna-multiview-grounding-20261027/evaluation-private.json)

## Tool and division of work

Enable the optional tool with `--multiview-grounding` on
`scripts/run_subscription_agent_pilot.py`. The flag reaches the generated
workspace documentation and numerical service. It is off by default.

`ground_multiview(observation_id, regions, comparison_regions, check_view)`
accepts Luna's handle/panel boxes and hinge-edge pixels in two current camera
views, plus its hinge-edge pixels in a third view. Original image coordinates
are required. The current observation ID prevents mixing captures around robot
motion. Luna selects the physical parts and judges the images; it does not
calculate camera calibration, triangulation, metric geometry or joint angles.

The tool uses existing calibration to triangulate an upright hinge line from
the first two views. It checks that line against the third view, compares
independent handle/door depth fits, and determines the task-motion axis sign
from observed geometry. These inherit the existing upright-door and vertical
handle assumptions. The implementation contains no task-state IDs or saved
answer annotations.

Replies preserve the acceptance status, failed checks, measured disagreements,
and paths to three overlays. Cyan shows Luna's edge, red the triangulated line,
orange the handle box, green the panel box and yellow the fitted contact anchor.
Full receipts remain in files. Observed compact replies were 1.5–3.0 kB; this
is a byte measurement, not a token/context-size claim. No point arrays or robot
joint vectors are dumped into the tool reply.

If consistency fails, the current planning model and epoch remain unchanged.
If it passes, the live service constructs a fresh collision scene, installs the
candidate, and invalidates previous plans through a new geometry epoch. It
does not retain role assignments from the old hypothesis. The call issues no
motion. **Accepted means numerically consistent, not a verified physical hinge.**
The reply explicitly retains `semantic_identity_verified=false`.

Implementation:

- [Shared geometry and overlay code](../src/vlm_tamp/multiview_grounding.py)
- [Schema and generic help](../src/vlm_tamp/robot_contracts.py)
- [Live service](../scripts/serve_subscription_robot.py)
- [Compact presentation](../src/vlm_tamp/compact_robot_interface.py)
- [Tests](../tests/test_multiview_grounding.py)

## What Luna actually did

Each session received raw RGB-D, existing calibration, four initial raw camera
images and generic tool help. Other raw views were available on demand. Each
fresh Luna session could make six geometry calls in 360 seconds. Luna supplied
every annotation, inspected feedback and overlays, and wrote its own final
decision and procedural notes. The same prompt was used in all three cases.
No human correction was sent between attempts.

| Initial state | Geometry calls | Final selection | Native hinge error | Session time | Numerical tool time, nested |
| --- | ---: | --- | ---: | ---: | ---: |
| 4 | 4 | candidate004 | 0.81 mm | 99.3 s | 25.4 s |
| 28 | 2 | unresolved | no selected model | 106.6 s | 17.4 s |
| 41 | 6 | candidate006 | 115.53 mm | 160.0 s | 40.6 s |

Native scoring occurred after all sessions and the integration check. The
registered screen required an actor-selected, tool-accepted model with hinge
line error at most 10 mm and the correct closing-axis sign. Only state 4 passed.
These are three states of one asset, not a broad success-rate estimate or an
end-to-end manipulation benchmark. Session time includes model inference,
image inspection, shell work and transport; it is not pure reasoning time.

In state 4, two early candidates failed the existing depth-direction check.
A third had inconsistent edge/projection measurements. Luna corrected its
selections and the fourth passed. Its notes identified the useful generic
lesson that a world-vertical edge can be slanted in an image. The successful
candidate used no assistant coordinates.

In state 28, the two depth fits selected locations about 263 mm apart and
opposing panel normals. Both submissions were rejected. Luna reported that it
could not match the parts consistently and selected no model. Private grading
also showed that the proposed contact was near the native hinge and its
closing-axis sign was wrong. Rejecting that disagreement is useful feedback,
but the tool did not repair the semantic selection for Luna.

State 41 exposes a false positive. Candidate005 and candidate006 had the same
115.5 mm hinge error. Luna revised the third-view annotation to make it agree
with the projected line, so candidate006 passed the numerical checks without
improving hinge accuracy. A checking view is excluded from the numerical fit,
but its annotations cease to be independent validation when repeatedly revised
using the displayed prediction. Static geometric self-consistency cannot
certify which physical edge is the joint.

Luna's state-41 notes describe candidate006 as supported by the images. The
post-run result contradicts that claim. The original notes and all failures
are preserved, and none were promoted into the learner guide.

## Integration and verification

The first actor-selected accepted candidate, state 4/candidate004, was chosen
for integration before native grading. A fresh simulator reset produced
identical metric sensor arrays. The real `RobotService.ground_multiview` call
installed exactly Luna's proposed hinge and built its planning scene. It
remained at the 60 reset controls; no manipulation trajectory was planned or
executed. This checks that the feature is usable by the actual robot interface,
not just a standalone notebook.

Fifty-four focused tests passed, covering triangulation, wrong-view feedback,
request validation, compact output, stale-capture rejection, failure preserving
the old scene, successful scene replacement, and existing sequence/pose-update
behavior. The final source/provenance audit checked 550 runtime files. Read-only
sensor/tool inputs remained unchanged, and each actor's isolation preflight
blocked the project implementation, old manual annotations, experiment/private
files and credentials. See the [audit](../results/luna-multiview-grounding-20261027/audit.json)
and [integration receipt](../results/luna-multiview-grounding-20261027/integration.json).

## Implications for tools and learning

This is actual Luna tool use and within-session revision, not assistant
annotation presented as learning. It does not demonstrate reliable grounding
or transferred procedural memory. The opt-in tool supplies useful geometry and
rejection evidence, while the false positive establishes a remaining limitation.

The next tools should make correspondence and articulation evidence explicit:
show where each fitted contact projects in other views, distinguish an observed
attachment feature from an inferred hidden axis, and use measured motion when
static images cannot resolve that distinction. Keep numerical consistency,
semantic identity, physical contact and task completion separate. Do not feed
native coordinates back to Luna or turn a consistency pass into an unconditional
memory saying that the hinge was correct.

The preserved `findings.txt` files are Luna-authored candidate lessons with
provenance. They are available for future controlled practice; this pilot did
not change the global learner guide, planner defaults, wrist constraints,
controller, or grasp/execution timing.
