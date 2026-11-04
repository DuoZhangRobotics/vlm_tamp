# Live push/contact diagnostic — 2026-11-04

Both live probes failed to implement the intended closing push. Both fresh Luna/high actors
chose CW about nearly upward canonical hinge axes. The simulator door opened in both cases.
This is a test of prospective action grounding and live feedback, separate from the earlier
recorded-clip motion-classification result.

Open `results/live-push-contact-20261104/watch.html` in VS Code Live Preview. It includes the
actual execution videos, all Molmo2 proposals, exact intent and outcome files, contact overlays,
preview evidence, phase-specific native changes, IK/arrival diagnostics, timing and resource audit.
The HTML is about 13 KB and the separate report data about 130 KB. It does not refresh itself.

## Protocol and changes before execution

- One targeted push request per state, 17 and 49; fresh Luna/high with no inherited memory,
  examples, closing sign, successful coordinates, motion-circle hint or native geometry.
- One Molmo2 worker remained loaded across both cases on GPU 1. Each request had fresh model
  context. Luna chose its query, camera and whether to accept or revise the proposal.
- 600-control ceiling and 1,800-second actor session. Voluntary finish after this diagnostic
  was enabled; these are not full-task runs forced to consume the remaining budget.
- A planned wrist-camera viewing motion was allowed, but neither actor used one. Neither
  substituted a grasp, hook or second executed push.
- Automatic action comparisons now include both native views and explicitly label camera
  motion. The earlier helper omitted the moving wrist camera. Approach and stroke comparisons
  remain separate; image displacements are not automatically labelled opening/closing.
- Retention now preserves the depth exports at those comparison boundaries. It does not
  capture additional images or preserve every intermediate export indiscriminately.
- No planner, IK, controller, collision policy, motion threshold or fallback algorithm changed.
  Runtime sources were frozen and verified across both cases: 647 files checked per run.

## Physical results

| State | Actual phase | Controls | Change in native opening |
|---|---|---|---:|
| 17 | Free approach | 60 → 128 | +1.1235° |
| 17 | First push increment | 128 → 135 | +1.7231° |
| 49 | Free approach | 60 → 115 | +13.0416° |
| 49 | Requested stroke | Not issued | Not applicable |

State 17 ended 88.9203° → 91.7669° open; state 49 ended 107.2647° → 120.3063° open.
Neither reached native task success. The first 60 controls are reset/warm-up. No fault was
injected. Native joint/contact records were read only after each actor had finished.

### State 17: a real push in the wrong semantic direction

Luna initially selected the fixed housing, then corrected the identity after inspecting images
and surface measurements. A later Molmo2 proposal landed in the window; Luna explicitly revised
it to the solid frame. The first lower-frame preview at `[274,248]` returned no solved IK:
19 constraint-feasible seeds, zero solver successes, and minimum reported position error
43.426 mm after 6.17 seconds. This is not proof of global infeasibility.

Luna then selected `[274,173]` on the upper-left frame. The preview returned 20 IK solutions;
the successful preview's IK and approach planning took 0.015 and 0.092 seconds respectively.
It predicted CW rotational tendency and motion entering the selected face, while explicitly
leaving opening/closing semantics unverified. Luna chose CW 25° as its closing hypothesis.

The approach reached its target within 1.84 mm. A roughly 1 cm IK push increment followed;
measured hand travel along the request was 7.31 mm. Both phases opened the door. Native
control-boundary samples recorded finger-2 pad contact with the solid door collider at controls
129–135, peak normal force 0.774 N. The selected landmark was on finger 1; contact at that
specific landmark was not demonstrated. Absence of a contact in a control-boundary sample
does not exclude a transient contact between sampled times.

The tool stopped with `motion_feedback_unresolved`: the joint-target mode needed measured
object progress before continuing its incremental motion, and the tracker returned zero usable
matches. It did not stop on a collision prediction or exhausted budget. There were 465 controls
remaining and approximately 1,793 seconds left in the separately requested skill budget.

Luna labelled the approach unchanged and the whole call/stroke unclear because of occlusion.
Its `intent.json` contains a stale normal in the nested draft push request; the tool log and
`outcome.json` retain the actual executed upper-patch normal. No tool silently replaced it.

### State 49: depth behind the window was used as the door contact

Luna accepted Molmo2's `[327,207]` window pixel. The tool lifted it to
`[-0.377234, 0.143750, 0.901376]` m and measured an almost vertical normal `[0,0,1]`.
This is a horizontal background surface visible through the window, not a measured vertical
door face. The planar fit's small residual did not establish semantic surface identity.

Evaluator-only asset inspection confirms the distinction: the visual window material has
alpha 0.25, while `microdoorroot` has a solid box collision geometry spanning the door.
See `third_party/LIBERO/libero/libero/assets/articulated_objects/microwave.xml`, lines 16 and
44–56. No native geometry from that inspection was supplied to Luna or planning.

The contact preview solved IK and an approach. During that approach the finger hit the solid
door and opened it 13.04°. Native contacts with `microwave_1_g17` begin at control 92; peak
normal force was 56.16 N. Contact near the end was at approximately z=1.108 m, well above the
requested background contact at z=0.901 m.

The controller stopped at intermediate reference 28: 34.8 mm reference-position error,
0.1608 rad joint lead versus the existing 0.05 rad lead allowance, after the existing wait
allowance. The final goal was still 249.9 mm away. This is measured execution failure with
physical door contact, not an IK/scene rejection or demonstrated global infeasibility.
No requested push stroke ran. There were 485 controls and 116.67 skill seconds remaining.
Luna incorrectly labelled the approach as closing.

## Integration and reporting findings

The live `ContactTracker` still wraps `RGBDHingeTracker` optical flow. CoTracker3 is available
through explicit part-motion tools, but neither actor invoked it. The previous offline fused
CoTracker result was therefore **not** tested as the automatic push feedback backend here.
Both cameras were supplied for visual comparisons and existing live feedback.

The recorded tool response also incorrectly stated `object_progress_is_execution_gate=false`
for a joint-target push, although the existing implementation stops joint continuation when
object tracking is unresolved. After both runs completed, only reporting/help was corrected:

- The flag now distinguishes direct hand-distance strokes from joint-progress requests.
- A separate `joint_progress_requires_object_tracking` field and the tracker class are exposed
  in compact feedback.
- The schema, compact index and introduction disclose the existing 1 cm joint increments and
  tracking dependency. Direction/distance strokes retain their existing behavior.

No stopping rule or execution algorithm was changed in that correction. No physical rerun with
the revised metadata/help is claimed. Original prompts, responses and source snapshots remain
under each run; the study manifest pins the code that actually executed.

The contact/direction failures must not become successful procedural memory. Actor findings
remain separate from the operator-only native diagnosis.

## Timing, resources and validation

| Measured time | State 17 | State 49 |
|---|---:|---:|
| Actor session including tools | 540.42 s | 242.86 s |
| All robot/tool service calls | 78.37 s | 34.51 s |
| Actor, transport and analysis outside service | 462.05 s | 208.34 s |
| Reset/startup | 36.12 s | 17.37 s |
| Molmo2 calls | 8.03 s | 1.82 s |
| Segmentation calls | 9.87 s | 4.64 s |
| Executed push call, including approach/captures | 7.18 s | 3.34 s |

Time outside service is not claimed to be pure model inference. No grasp planning ran.

The existing 32 GiB cgroup, zero-swap and whole-group OOM policy were used. Peak accounted job
memory reached 32 GiB, mostly model-file cache; peak anonymous/shared memory was 5.56 GiB.
Minimum available host memory was 109.10 GiB. Peak GPU-1 memory was 23.27 GiB. All 56 SSH probes
succeeded, zero OOM kills occurred, and maximum sampled full I/O pressure average was 4.06%.
All workers exited and GPU 1 returned to approximately 3 MiB used. Other users' jobs were left
running. The GPU monitor was corrected to follow the explicitly selected GPU rather than
always reading GPU 0.

142 targeted tests passed after the reporting correction, and Ruff/diff checks passed. Browser
checks verified six preview/execution records, three executed phases, 14 document links,
both videos decoding/seeking, and 360 px layout. No CMake build was needed.

Entry points are `scripts/run_push_contact_probe.py`, `prompts/push_contact_probe.txt`,
`scripts/report_push_contact_probe.py` and `templates/push_contact_probe.html`. The monitored
launcher is under `.runtime/live-push-contact-20261104/`.

## Next boundary

The next implementation decision is how to explicitly associate window clicks with a plane
measured on opaque door material, instead of using background depth as the contact surface.
That must be a declared contact model, not an automatic point substitution. For simple physical
probes, keep desired door rotation separate from a direction/distance hand stroke. Integrating
CoTracker feedback into live action assessment also remains outstanding. More control budget
does not establish a correct goal direction or repair a wrongly associated depth point.
