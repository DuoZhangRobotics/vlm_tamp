# Push approach and contact phases

Targeted `push` now separates free travel from contact acquisition. Luna still selects the
observed point, robot contact face and desired motion. No task identifier, known direction,
successful coordinate, object mesh or scripted task sequence enters the tool.

1. The requested surface normal defines the outward side. Point-only requests use the
   opposite of their explicitly requested push direction for approach, without inventing a normal.
2. cuRobo first solves the selected contact point with contact permitted. For each solved
   hand pose, the minimum translation along the outward normal places every hand sphere
   outside the reference plane. No full-hand-length stand-off, minimum distance or margin
   is added. A hand already outside the plane needs zero translation.
3. cuRobo solves that precontact goal with arm, palm and finger scene collisions enabled.
   VAMP plans the free-space route using the same observed points and sphere model.
4. A straight cuRobo continuation approaches the original selected point at 3 cm/s. Finger
   contact is allowed; arm/palm permissions follow the existing contact policy. The observed
   target remains stationary during this approach. No CPU IK repair or backup branch is used.
5. Existing measured arrival and push execution follow. Physics resolves pushing contact.

The four upstream finger spheres retain their original radii. Their positions use current
robot finger-link transforms, including measured aperture. Link transforms avoid MuJoCo's
internal mesh-centering transformation. Three existing palm spheres supply a rigid coordinate
frame, so native SIMD and GPU environment queries use existing sphere FK. GPU coefficient and
permission tensors update in place for warm CUDA graphs. Original VAMP self-collision remains
unchanged. Contact-phase exemptions do not leak into the next free approach.

`approach_plan` and two-camera `approach_images` expose the precontact point, side, distance
and planned pad trajectory. `free-approach`, `contact-approach` and `push-stroke` remain distinct
executed phases with before/after captures, control timestamps and timing. The dashboard
labels planned overlays separately from execution video and evaluator-only object motion.

The numerical planner still sees only observed surfaces. Missing or misassociated RGB-D
geometry can still produce a physically incorrect plan. A correct approach also does not
establish whether Luna chose the correct face or task direction.

Validation and fresh state-17/state-49 results are recorded in
`results/luna-approach-minimal-20261104/watch.html`. The full-episode protocol holds the
600-control limit, Luna/high, two native cameras and fresh memory condition. The existing
32 GiB cgroup, zero swap and RAM/I/O/SSH monitoring remain active. This is a two-state
engineering evaluation, not evidence of transfer to new task families.

The initial whole-hand-extent version is preserved at
`results/luna-approach-collision-20261104/watch.html`. It consumed 332 controls in two
24.8 cm contact approaches in state 49. That unnecessary travel motivated the geometric-minimum
translation above; no successful coordinates or direction from that evaluation enter new actors.

## Final fresh evaluations

The final numerical implementation ran at commit `c1dbdb1`. Both actors used Luna/high,
600 controls, two native cameras and no inherited memory. Both ended at the control limit.
The later reporting-only change reads the already-recorded trajectory, capture and comparison
clocks separately; it does not change either execution.

| State | Native opening, start → finish | Actor session | Tool service | Outside tool service |
|---|---|---:|---:|---:|
| 17 | 88.9203° → 88.9203° | 498.74 s | 104.73 s | 394.01 s |
| 49 | 107.2647° → 120.0865° | 691.67 s | 105.66 s | 586.01 s |

Outside-service time includes model, transport and analysis; it is not pure inference time.
600 controls correspond to 30 simulated seconds, including the initial 60 reset controls.

State 17 selected pixel `[459,318]` on the fixed housing. Two targeted free approaches and
7 mm contact approaches reached their goals within 2.64 mm. Contact approach took five controls
(0.25 simulated seconds) each. The door never moved. The later +Y stroke pressed the housing:
operator-only samples show finger-pad contact with `microwave_1_g8`, and the stroke stopped
50.54 mm from its hand goal. The episode exhausted controls with 1358.7 wall seconds remaining.
This is a semantic part-selection failure; successful robot positioning did not demonstrate
contact with the moving door. Video: `trials/push-close_microwave-state017/attempt01/initial/rollout.mp4`.

State 49 used point-only contact. The first 32.408 mm contact approach stopped 8.19 mm short
at control 188, before issuing its requested stroke, with 412 controls still available. Native
samples recorded finger-2 pad and palm contact with the door, although finger 1 was selected.
The door had opened from 107.26° to 120.48°. Its later 41.526 mm contact approach stopped
15.85 mm short even after the 60 additional wait controls Luna requested. The selected +Y
motion has an opening moment about the evaluator's hinge at that contact; the door was already
at its approximately 120° opening limit. Finger-1 pad contact reached 106.26 N. The final direct
strokes did not close it, and controls expired with 1189.2 wall seconds remaining. Video:
`trials/push-close_microwave-state049/attempt01/initial/rollout.mp4`.

One physical discrepancy remains explicit: state 49's first free approach opened the door
4.25°, although its planned path used finger checks. The change starts at control 136. The
control-boundary contact samples contain no contact during that free phase; at control 136,
measured TCP position differed from its reference by 9.08 mm. This does not establish the cause.
A targeted follow-up should inspect substep contacts and compare executed/reference geometry.
Do not claim physical collision-free execution from planner approval. No added margin, collision
recheck, native geometry input or fallback was introduced to hide this discrepancy.

All native angles, limits and forces above are evaluator-only, read after the respective
actor finished. No fault was injected. Failed IK searches and controller arrival failures are
separate from demonstrated physical infeasibility. The dashboard preserves exact search
budgets, residuals, per-phase controls, robot/part selections, images and videos.

## Validation and resources

The native adapter was rebuilt through `make -j32`. The initial implementation passed 191 CPU
and 21 GPU tests. The minimum-travel correction passed 125 focused tests, overlapping the
initial set. Tests cover phase permissions, measured aperture, native/GPU parity, warm CUDA
graph updates and minimal hand-plane displacement. Two historical calibration-fixture tests
were unavailable. Ruff, JavaScript syntax and diff checks passed.

The completed final batch stayed within the 32 GiB cgroup with zero swap allowance. Peak
accounted memory was 27.85 GiB, peak anonymous/shared memory 5.49 GiB, and available host memory
never fell below 65.37 GiB. There were zero OOM kills; all 81 SSH probes succeeded. Maximum
sampled full I/O pressure was 2.25%. GPU-1 total residency peaked at 51.74 GiB, including roughly
29.54 GiB already occupied by another job; this is a device total, not our process allocation.
Both experiment workers exited. Browser QA checks videos decoding/seeking, approach overlays,
phase counts, log links and a 360 px viewport.
