# Explicit surface-contact orientation

The point-only contact tool left wrist orientation free. A numerically correct
point could therefore correspond to an unsuitable physical hand posture. The
user approved an explicit point-plus-surface-normal model on November 1/2.
This is an optional model, not an automatic fallback or a task-specific action.

## Contract

`push`, `hook_pull`, and `plan_contact_pose` accept
`contact_constraint="surface_normal"` and a signed outward
`surface_normal_world`. The robot face comes from the selected calibrated
`contact_name`. cuRobo aligns that face against the supplied surface normal and
leaves twist free. The implementation creates a contact frame whose local Z is
the robot face normal, then uses native goal-frame weights `[1,1,1,1,1,0]`.
Existing numerical tolerances and returned joint solutions are unchanged.
Acquisition and continuation solvers remain cached separately for each model.

The normal selects a face and orientation. It does not select an opening/closing
sign or a motion direction. A world direction or explicitly associated joint
increment is still required. A joint-linked stroke rotates its normal using the
same declared joint increment while its contact-point waypoints remain a straight
chord. Accepted RGB-D motion transports the same material face; it is never
re-flipped toward a new camera. Unresolved tracking remains unresolved.

Completion matches the chosen model: point-only approaches check the selected
contact point; normal-constrained approaches/strokes check that point and its
normal, with the existing 4 mm / 0.05 rad controller tolerances. Twist is not an
extra completion condition. Retargeting responds to measured target movement;
it does not run another IK search to polish residual error on an unchanged target.
Contact-pose plans default to holding their planned gripper aperture. Explicit
aperture changes require replanning because the contact offset would change.

No CPU IK correction, analytical Jacobian, posture menu, collision buffer, dense
returned-path validation, or automatic normal-to-point fallback was added.
Free approaches use VAMP RRTC through the existing MC-VAMP adapter. Push strokes
remain direct cuRobo IK. Grasped arcs retain MC-VAMP CRRTC.

## Communication

Previews identify the robot face role (inner jaw, distal pad, body-mesh reference,
or palm), the chosen normal, initial robot contact's signed plane offset, and the
motion component into/away from that face. Candidate camera sides are geometric
view information, not visibility or reachability certificates. The normal and
motion arrows are rendered together. Measured orientation reports carry control
steps and surface-pose freshness.

`measure_surface` supplies a camera-oriented normal without replacing the active
scene. `select_surface_contact` installs a surface scene. Local normal support,
spatial spread and fit residual are exposed; no whole-mask completeness gate is
added. A partial mask can support a plane, while an incorrect depth association
can still describe background geometry instead of the intended moving part.

Opening/closing requires a goal relationship to the fixed structure. Unsigned
plane angles are 0–90 degrees and fold larger angles; camera-facing normal angles
also depend on which material faces were selected. Neither alone certifies task
progress. The existing material-motion report now exposes the observed point
translation, with its measurement interval, instead of requiring the actor to
extract it from a 4x4 transform. No hand-motion substitute is used.

Compact contact-search failures now preserve native endpoint rejection causes,
including robot part, observed surface and representative sphere coordinates.
Previously the wrapper displayed `unknown` for these nested native reasons.

## Evidence before Luna trials

- GPU-only checks: 6–30 returned solutions across current/opposite/arbitrary
  signed faces; maximum observed normal error 0.323 degrees; maximum position
  error 1.65 mm under the existing 3 mm numerical tolerance. Warm calls took
  about 6–11 ms. Initial acquisition was 2.30 s; initial local continuation
  1.17 s. No robot motion was requested. See
  `results/contact-normal-ik-check-20261102`.
- Same old Luna-selected inner-pad inputs: state 17's normal-model endpoints
  were rejected by native palm/arm collision checks before motion; state 49's
  normal approach stopped 52.2 mm short during physical contact. Normal alignment
  alone did not make those selections usable. See
  `results/contact-normal-matched-20261102`.
- State 49, observed front face/end pad: approach plus a 2 cm inward stroke
  repeated 3/3, moving the door from 107.26 to 111.91 degrees open. This validates
  contact execution, **not closing**. See `results/contact-normal-face-checks-20261102`.
- State 17, observed outer handle face: approach reached in 3/3 and the push
  interval closed from 87.65 to 84.49 degrees (initial scene 88.92 degrees).
  The requested 2 cm stroke stopped short. This is repeatable partial closing
  motion, **not full task success**. See `results/contact-normal-handle-face-check-20261102`.
- Rejected closing-side candidates are visualized in
  `results/contact-normal-rejections-20261102`. A 24 mm palm sphere overlaps
  observed handle points by 3.43 mm for the end-pad candidate; the higher inner-pad
  candidate overlaps observed top-edge points by 5.47 mm. These are native
  spherical-model overlaps, not proof of exact physical palm penetration.

All diagnostic pixels/settings above were operator inputs, excluded from Luna
prompts and memory. Native object angles/contacts are evaluation-only evidence.

The preceding frozen point-model study completed four clean practice episodes
and four fresh evaluations: 0/4 practice success; 0/2 with learner notes and 0/2
without on held-out states 8 and 35. Notes were frozen verbatim and explicitly
acknowledge no successful method. See
`results/luna-contact-handoff-practice-v2-20261101/watch.html`. These results do
not evaluate the new surface-normal model.
