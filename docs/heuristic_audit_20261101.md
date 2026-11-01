# Heuristic removal and remaining assumptions — 2026-11-01

This audit covers the active Luna robot service and its grounding, IK, contact, arc-planning, execution and feedback helpers. Historical offline backends are identified separately. Passing implementation tests is **not evidence that a heuristic improves task performance**. Unless stated otherwise below, no controlled necessity ablation has been established.

## Removed in this change

1. **Adjacent-depth task-direction guess and conflict veto.** Removed from both single-view and multiview grounding. Nearby points no longer choose which sign means opening/closing, and cannot reject the actor's image-motion arrow. A resolved arrow supplies the sign. An absent/ambiguous arrow leaves task direction unknown; +Z is explicitly only a coordinate convention, independent of the task's open/close label. Geometry is still available, and signed motion remains the caller's choice.
2. **The point-mode jaw-tilt/alignment constraint.** Removed from both Python endpoint geometry and the compiled MC-VAMP task constraint. Point mode now constrains the grip-point circle and requested arc extent, with no jaw or wrist orientation restriction. The explicitly requested `rigid` and `jaw_axis` modes remain different attachment models.
3. **Incorrect direction-preservation reporting.** An explicit fitted-axis sign reversal no longer returns `direction_preserved=true`. Explicit axis updates invalidate the old image-direction provenance rather than retaining a misleading description.
4. **Non-default full-goal IK movement penalty.** A controlled comparison of weight 1 versus 0 produced the same saved-case success counts and rigid-IK position errors. It did not explain that failure and showed no benefit there. The custom override is removed; full-goal IK uses the upstream optimizer configuration. Local continuation settings remain separate and are audited below. Physical execution speed limits are unchanged.
5. **Opaque IK error reporting.** Failed GPU endpoint replies now retain the actual endpoint error and its 1 mm IK accuracy setting, rather than omitting the limit or confusing it with the 6 mm path band.

No replacement task-direction heuristic, collision margin, fallback planner or successful task coordinates were added.

## Door and hand motion: the actual distinction

The native collision callback already rotates the tagged moving door geometry. It performs the equivalent inverse transform on robot sphere queries to reuse the observed point-cloud index. Stationary points do not rotate. The angle is inferred from the candidate grip point around the configured hinge.

In `point` mode, the grip point follows that circle, but the wrist is not required to co-rotate with the door. Other arm links are also not rigidly attached to the door. Consequently a palm or arm can enter door geometry while the grip point stays on the requested circle. This is a different model from carrying the entire hand rigidly with a firmly grasped object.

The user's rigid-motion argument is correct: applying one rigid transform to the hand and the door preserves their mutual distances for **any** chosen hinge. A wrong hinge alone cannot create a hand-versus-door collision under that common rigid transform. It can change their relationship to stationary obstacles and to the real physical articulation. Wrong point ownership can also leave actual door points in the stationary partition; that is a separate observation-model limitation, not permission to clear obstacles under hypothetical poses.

Saved opening-state-36/control-584 evidence, with no new GPU query or robot motion:

| Candidate | Grip-point error relative to door motion | Wrist rotation relative to door | Palm-sphere departure from rigid carriage | Initial / rigid-carriage clearance | Actual candidate clearance |
|---|---:|---:|---:|---:|---:|
| 15°, palm sphere 45 | 0.024 mm | 143.35° | 117.67 mm | 31.65 / 31.65 mm | −8.95 mm |
| 6°, palm sphere 51 | 0.037 mm | 10.17° | 8.00 mm | 1.46 / 1.46 mm | −0.90 mm |

The former 30° jaw rule did not enforce rigid hand-door motion. Removing it does not make point mode a rigid grasp. The existing `rigid` mode requests that relationship; it is not being silently enabled as a new fallback.

[Numerical transport audit](/common/users/dz323/vlm_tamp/results/door-hand-transport-audit-20261101/audit.json) · [Palm comparison](/common/users/dz323/vlm_tamp/results/door-hand-transport-audit-20261101/palm-relative-motion.png)

## Remaining choices exposed by the audit

| Choice | What it actually does | Basis / status |
|---|---|---|
| Point, jaw-axis and rigid attachment modes | Different relationships between hand and moving part. `plan_arc` defaults to point; `plan_sequence` defaults to jaw-axis. | These define the requested kinematic problem, but their defaults are design choices. No evidence establishes one default as best. Tool descriptions now explicitly distinguish point motion from rigid carriage. |
| Upright door and vertical hinge/bar fitting | Single-view grounding fits a vertical plane; hinge direction is vertical. The handle fitter assumes an upright cylinder. | Active domain assumptions, not general articulated-object recognition. Not learned by Luna. They must be exposed as tool scope, not claimed universally valid. |
| Handle radius/height thresholds | Cylinder fit uses 5–18 mm radius bounds and at least 65 mm observed height, with residual/support thresholds. | Restricts the handle fitter. No necessity ablation established. Failure can return door-only geometry; it does not substitute a different handle. |
| Moving/static point ownership | Masks, projected depth agreement, panel bands, handle proximity and a small connected mount region assign point roles. | Association heuristics, not object identities. Current door points may be mislabeled stationary. Neighboring depth is still used for geometric thickness/association; its use for task direction is removed. |
| Panel band and mount thresholds | Uses measured quantiles with fixed floors/caps; the mount association assumes a finite bar-to-panel gap and local connectivity. | Shape/association assumptions with no task-success ablation. They tag existing observations; live planar completion remains disabled. |
| Eight full-angle goals and 0.12-rad deduplication | Offers up to eight distinct goal postures from an already-returned cuRobo batch; near joint configurations can be discarded as duplicates. | Search heuristics, not physical constraints. Similar postures can have different collision outcomes. No necessity evidence. The saved 15° audit checked all successful returns and found no unused collision-free winner, but that does not justify these limits generally. |
| Full-pose IK failed-flag rescue and clipping | The older full-pose adapter can accept GPU-failed results after FK/self checks and clips to joint bounds. Arc endpoints are preserved unchanged and require GPU success. | Remaining inconsistency. No evidence that the rescue is needed in the current pipeline; the preceding 12-case batch recorded zero rescued full-pose candidates. Not changed silently in this audit. |
| Full-pose posture ranking | Adds a joint-limit-distance preference to distance from the current posture; some selectors re-sort by current-posture distance. | Optimization preference with no demonstrated necessity. It can influence which posture reaches path search first. |
| cuRobo point-mode soft preferences | Point IK retains a small orientation cost; local arc/contact IK retains movement regularization. These do not enforce rigid hand-door attachment. Full-goal IK now uses upstream defaults. | No isolated necessity ablation of the remaining soft weights is established. A small soft preference did not prevent the recorded 143° relative wrist rotation. |
| Local IK time step | Local arc continuation supplies a time step derived from 3° / 15° per second; controlled contact IK supplies 0.05 s. cuRobo documents this as enabling velocity-aware clamping. | An implicit restriction during IK, distinct from the later execution scheduler. There is a regression test for the combined one-seed push implementation, but no isolated necessity ablation of this clamp. It remains a candidate for testing/removal. |
| Contact point/orientation/standoff menus | Targeted contact acquisition samples up to five observed points plus calibrated contact landmarks and wrist/standoff alternatives. | Hand-written proposal generation. Distinct from the learned GraspGen grasp route. Search exhaustion is not proof that no contact is possible. No claim of optimality or ablated superiority. |
| Near-contact reuse | A supported landmark within 6 mm can skip approach search; an unchanged preview can be reused. | Computational reuse, not proof of engagement. Simulator/observed motion determines whether contact worked. |
| Missing retained contact in `continue_push` | Substitutes the documented current-hand `push_stroke` behavior with no associated object model. | Previously user-authorized simulation behavior, explicitly reported. It does not assume grasp/contact success. |
| Extra-view retry | A failed motion fit may capture missing reference views and retry once at the same control state. | Automatic sensing policy. Cost is separately profiled; no necessity ablation established. It took 14.84 s in the last state-49 run. |
| Tracking/fit acceptance thresholds | Feature visibility, measured depth, rigid-fit residuals, span, angle and bootstrap spread filter or qualify estimates. | Numerical data-quality choices. Some nondegeneracy requirements are mathematical; the particular thresholds are not proven task-optimal. They are not calibrated guarantees of hinge accuracy. Provisional application is explicit. |
| Gripper-gap and ineffective-motion stops | Uses measured aperture/object motion to return after likely loss of engagement or repeated ineffectiveness. | Requested feedback behavior; not force sensing or a grasp certificate. Threshold values remain implementation choices. |
| Automatic sequence stops before closure | If target tracking is unresolved after approach, the sequence may return without closing the gripper. | A recovery-policy choice, not proof that closing cannot work. No necessity ablation established. The explicit gripper operation remains available. |
| Arm correction during a gripper command | A measured joint-limit overshoot can trigger an inward arm reference, nominally one 0.05-rad controller increment, while opening/closing the gripper. | Automatic recovery behavior that can move the arm during a gripper call. It is reported, not a simulator teleport; whether this belongs in the gripper tool has not been established by an ablation. |
| Physical push stall | Returns after one second with at least 10 mm commanded travel but less than 1 mm measured directional hand travel. | Physical-feedback stopping policy, not an observed-scene collision veto. Its exact threshold has not been established optimal. |
| Large-angle reference segmentation | Requests above 90° are planned in chunks to avoid the current atan2 progress representation wrapping. | Numerical coordinate choice. The 90° threshold itself is not proven necessary; 15° requests are attempted whole first. No robot execution occurs between planning chunks. |
| Arc-progress interval | Current point/coupled constraints allow progress from −0.03 rad to requested angle +0.03 rad. | Encodes a bounded requested sweep, with an existing numerical allowance. It is separate from collision padding and is not an object joint-limit estimate. |
| Optional wrist-excursion parameter | Currently filters endpoint postures, not every point along the returned path. Default is no extra cap. | Caller-selected option with an implementation-scope limitation; help now states that limitation. It is not an automatic “short motion” restriction. |
| Sequence calibration | Initial small physical calibration motions only occur when `calibrate_joint=true`. | Opt-in since the preceding cleanup. No automatic semantic direction selection is restored. |
| `pull_axis_world` legacy name | In the door grounder it aliases the camera-facing panel normal. | A normal is not a verified push/pull direction. The active push interface requires explicit direction; this legacy name remains misleading and is listed for cleanup rather than treated as an action recommendation. |
| Native hinge-constraint callback | A custom circle residual and finite-difference projection callback supplies the task to upstream CRRTC. | A task constraint is required; this finite-difference implementation is one choice, not an established necessity. It is not the upstream generated analytic SIMD constraint. Its cost/alternatives need measurement. No post-plan validation is added. |
| Repeated robot-frame verification | Startup and scene/arc construction compare calibrated FK and sphere frames. | Checks calibration consistency rather than task feasibility. Repeating identical robot checks may be avoidable; no timing ablation establishes the need for their current frequency. |
| Same-episode actor continuation | An actor that ends its turn before native success or budget exhaustion is resumed with a budget/status reminder. | Implements the requested persistent episode. It supplies no strategy or extra controls. Runtime errors are not silently replaced with another model. |

## Retained contracts and requested behavior

Physical robot joint bounds, upstream VAMP self/environment sphere checks, finger/environment exemptions, zero collision inflation, calibrated robot/TCP geometry, and explicit point/rigid/jaw-axis targets remain. Current measured-robot point filtering is observation preprocessing previously requested by the user; it is not a convex candidate collision pass. Native-success termination, 600 controls, 1800 wall seconds, 1 rad/s joint-reference limit, and the existing memory/I/O watchdog remain.

Invalid array shapes, nonfinite values, undefined camera rays and stale pixel references are input/data validity issues. They should be reported precisely; none establishes physical impossibility. Search budgets also bound computation, not feasibility.

The default learned grasp route still uses eight unmodified GraspGen proposals. The old frontal/oblique/depth-sweep menu, CPU palm-clearance objective, SciPy arc proposer, legacy convex planners and historical scripted recovery pipelines are not fallback routes of the active Luna evaluation.

## Evaluation policy

Tests first verify removal, input/output semantics, and shared-transform geometry. Matched saved-data checks then compare the recorded request with and without the removed rules. Point and the existing explicitly requested rigid mode are diagnosed separately; a rigid result does not silently change point mode. Fresh Luna trials use no inherited memory or successful coordinates. A failed trial is retained and explained rather than repaired with an unreported strategy.

The targeted implementation suite passed **238 tests**, with two historical-fixture skips. This verifies behavior, not the necessity of the remaining heuristics. The fresh regression study is `results/luna-heuristic-removal-20261101`, closing states 17 and 49 once each, under the existing 600-control/1800-second limits and 16 GiB/no-swap resource guard.

## Recorded checks and fresh outcomes

The original state-49 image arrow now returns +Z without changing the estimated hinge location. With a missing arrow the same geometry returns an unknown task direction, not a replacement closing guess. Evidence: [grounding replay](/common/users/dz323/vlm_tamp/results/no-heuristics-grounding-replay-20261101/result.json).

| Saved opening-state-36 request | Result without the jaw rule | Wall time |
|---|---|---:|
| Point-mode 5° | Complete path found through the declared continuation fallback | 1.84 s |
| Point-mode 15° | Endpoint collisions; continuation stopped at 6° | 0.97 s |
| Explicit rigid-mode 15° | No IK endpoint met 1 mm accuracy; best full-goal position error was 46.0 mm | 1.25 s |

The explicit rigid-mode comparison does not establish that the robot can realize the ideal rigid hand/door motion shown in the geometric audit. Its IK search failed; no robot motion was executed and no physical infeasibility claim is made. Repeating the 15° point/rigid checks with global IK movement-penalty weight 0 instead of 1 left the outcomes and rigid errors unchanged. The prior explanation that this penalty might cause the failure was therefore not supported by this test. The custom full-goal override was subsequently removed in favor of upstream defaults. [Comparison](/common/users/dz323/vlm_tamp/results/ik-default-regularization-ablation-20261101/summary.json).

Fresh physical trials used frozen commit `1ab6e7d`:

| Closing state | Native success | Controls | Actor wall time | Robot service | Luna/transport/local analysis outside service |
|---|---|---:|---:|---:|---:|
| 17 | Yes | 505 | 256.7 s | 93.9 s | 162.8 s |
| 49 | Yes | 591 | 703.3 s | 145.7 s | 557.6 s |

State 49 initially supplied an image arrow that mapped to −Z. Later the actor changed its arrow and the tool returned +Z; both choices came from the actor, with no depth-direction veto. These are two trials, not a statistical claim of improved reliability. Neither actor called `plan_arc`, so the arc changes are supported by the separate numerical tests, not inferred from these pushing outcomes.

Peak RAM was 9.28 GiB and sampled GPU memory was 2211 MiB. Across 497 watchdog samples, maximum full I/O pressure was 0.18%, memory pressure was zero, and no OOM/resource interruption occurred. After this frozen batch, the full-goal IK default and error-reporting cleanup passed 25 targeted checks; those changes did not affect the already-recorded trial inputs.

## Source map

- Direction and geometry: [single-view grounding](/common/users/dz323/vlm_tamp/src/vlm_tamp/handle_contact.py), [multiview grounding](/common/users/dz323/vlm_tamp/src/vlm_tamp/multiview_grounding.py), [point ownership](/common/users/dz323/vlm_tamp/src/vlm_tamp/coupled_scene.py), [public contracts](/common/users/dz323/vlm_tamp/src/vlm_tamp/robot_contracts.py).
- Planning and search: [arc endpoint/search policy](/common/users/dz323/vlm_tamp/src/vlm_tamp/curobo_v2_arc.py), [native constraint and collision callback](/common/users/dz323/vlm_tamp/src/vlm_tamp/native/mcvamp_contact_numerical.cc), [cuRobo settings](/common/users/dz323/vlm_tamp/src/vlm_tamp/rekep_libero/curobo_v2_worker.py), [full-pose IK postprocessing](/common/users/dz323/vlm_tamp/src/vlm_tamp/rekep_libero/task_ik_curobo.py), [large-angle coordinates](/common/users/dz323/vlm_tamp/src/vlm_tamp/signed_contact_motion.py).
- Contact and execution: [contact proposals](/common/users/dz323/vlm_tamp/src/vlm_tamp/contact_skills.py), [contact feedback/recovery](/common/users/dz323/vlm_tamp/src/vlm_tamp/contact_skill_service.py), [sequence policy](/common/users/dz323/vlm_tamp/src/vlm_tamp/sequence_execution.py), [timing and gripper arm correction](/common/users/dz323/vlm_tamp/src/vlm_tamp/timed_execution.py), [measured stall](/common/users/dz323/vlm_tamp/src/vlm_tamp/push_execution_feedback.py).
- Observation and persistence: [motion estimation](/common/users/dz323/vlm_tamp/src/vlm_tamp/part_motion.py), [tracking](/common/users/dz323/vlm_tamp/src/vlm_tamp/rigid_hinge_tracking.py), [service/model updates](/common/users/dz323/vlm_tamp/scripts/serve_subscription_robot.py), [same-episode continuation](/common/users/dz323/vlm_tamp/src/vlm_tamp/episode_persistence.py).
