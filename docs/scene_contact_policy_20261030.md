# Environment collision checking restored for live robot tools

User instruction: check the arm and palm against the environment, allow
intended finger-pad contact, and keep the extra dense path validation removed.

The earlier `advisory` default in `configs/articulated_libero.json` and the
shared tool configuration caused `NativeScene` to pass **empty stationary,
moving and contact point clouds** to its native checker. Those results were
not MC-VAMP approval of the actual environment collisions. This was an
integration/configuration bypass of environment checks.

The live schemas now permit only `collision_policy="veto"`. Defaults in the
configuration, grasp/free/arc planning, sequences, push and hook tools agree.
Offline code retains explicit advisory support for historical comparisons;
Luna cannot request it through the live callable schema. Documentation and the
compact tool index were updated. Frozen learner notebooks were not rewritten.

MC-VAMP's `ObservedPanda::fkcc` receives the observed scene clouds and checks
configured arm/palm/finger geometry during its own planning. Finger contact is
limited to distal pad geometry and overlapping fingertip geometry against the
selected patch. Proximal fingers and unrelated surfaces remain checked. An
explicit palm-contact skill still permits only its selected palm/target pair;
ordinary grasp planning never grants that permission. The old automatic
submillimetre palm-overlap exception is disabled by default.

The grasp approach now allows intended pad/target contact while the gripper is
open. Both open and predicted closed apertures are checked when selecting IK.
For each of eight original learned poses, one GPU query returns parallel IK
proposals. We pick the closest returned branch that passes both scene checks,
then issue at most one path query for that pose. No geometric grasp fallback,
pose displacement menu, extra collision padding or dense path pass was added.

## Replays and timings

Report: `results/scene-contact-policy-20261030/watch.html`.
The previous watch pages link to it. Use VS Code Live Preview.

The three matched cases reused the exact archived eight-pose GraspGen arrays.
All geometry inputs matched, and source-integrity checks passed.

| Case | Grasp call | Outcome | Path queries |
|---|---:|---|---:|
| open37, original eight | 6.83 s | 6 poses blocked by scene checks; 2 without robot-valid IK | 0 |
| open13, original eight | 5.90 s | 6 poses blocked by scene checks; 2 without robot-valid IK | 0 |
| close49, original eight | 7.87 s | 7 poses blocked by scene checks; 1 without robot-valid IK | 0 |

For open37, the native context now contains 43,607 stationary points, 2,692
moving noncontact points and 1,067 selected contact points. Previously the
advisory mode supplied zero in all three categories. Rejected endpoint checks
identify proximal-finger/door or contact-patch overlap, palm/stationary-scene
overlap and arm-link-5/stationary-scene overlap. All returned robot-valid IK
branches were tested in distance order. These are collision predictions and a
finite candidate search, not proof that every possible grasp is infeasible.

No grasp approach was executed in those three rejected batches. Reset's
existing 60-control initialization is separate.

One additional fresh eight-pose batch (seed 1, open37) found an admissible
eighth-ranked pose, using the same rules:

- 8 IK queries, 1 path query; native RRTC solving **0.094 s**, complete path
  call **0.253 s**, extra dense validation **0 s**.
- Whole grasp call **20.13 s**, including generation/model startup **12.25 s**,
  scene preparation **3.80 s**, selection **4.05 s**.
- Measured approach endpoint error **0.321 mm**.
- Native contact logs contain **no arm or palm environment contacts**. The
  recorded scene contacts involve one finger pad and the microwave.
- The fingers nevertheless closed empty: pad gap approximately zero against
  the selected bar's roughly 19 mm diameter. The 10-degree hand probe did not
  verify object following; native task success remained false.

The collision fix therefore admitted and physically executed an unobstructed
approach, but **did not establish a successful grasp or task**. The accepted
learned pose had discriminator score 0.206 (a score, not a calibrated success
probability). At arrival the pad midpoint was approximately 22.0 mm from the
observed fitted handle axis, whose radius was 9.5 mm. This is consistent with
the empty closure despite accurate robot arrival. Grasp proposal/contact
placement remains the next problem; the prior 8–9 cm blocked approach is a
different failure. No manually adjusted pose was introduced.

Native object contacts were read only after execution for this report and
were never planner/actor inputs. These were numerical component experiments,
not fresh Luna policy evaluations or full-task success trials.

## Retained attempts and replay harness

- `results/small-graspgen-scene-contact-20261029`: matched opening cases and the
  first closing reset, which stopped before planning.
- `results/small-graspgen-scene-contact-close49-20261029`: completed matched
  closing case.
- `results/small-graspgen-scene-contact-seed1-20261029`: first seed-1 reset,
  stopped before planning.
- `results/small-graspgen-scene-contact-seed1-metric-20261029`: completed new
  seed-1 batch, executed approach, closure and probe.

The early resets stopped because the harness required bitwise RGB equality.
The differences were only 85 one-level RGB values in an unused view (closing)
and 15 one-level RGB values in the selected view (seed 1). All depth, world
point and robot-depth arrays were exactly equal. The numerical replay supplies
fixed public pixel regions and consumes metric arrays, not RGB intensities.
It now requires exact metric equality and records RGB differences separately.
This is not a relaxation of collision checking and does not apply to a fresh
visual actor evaluation. Both stopped attempts remain in the report.

## Validation

115 focused tests passed; 3 optional fixture tests skipped. Native integration
tests actually exercised pad/target allowance and arm/palm/unrelated-surface
rejection. The live schema rejects advisory requests. Selection tests verify
that a scene-colliding nearest IK branch does not exclude a farther valid
branch, and only one path query follows selection. Source-integrity checks
passed for all completed experiment drivers.

The previously rebuilt native planner still returns its original checked path;
the removed 0.006-radian dense collision/projection pass remains absent.
Collision guarantees remain limited to observed RGB-D geometry and the
planner's own discrete checking resolution, as requested.
