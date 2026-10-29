# Restriction audit and recovery proposal — 2026-10-29

This is an audit of the active Luna service, contact/grasp/arc planners,
perception gates, controller and experiment budgets. Runtime policy was not
changed. The machine-readable inventory has 20 findings and hashes of 19 source
files in `results/restriction-audit-20261029/audit.json`. Selected older helpers
are identified separately; finding a limit in an unused helper does not show
that it caused a current failure.

## Main findings

| Finding | Current implementation | Proposed change |
| --- | --- | --- |
| Extra joint reserves | IK excludes 0.025 rad near each true bound; native planning independently excludes 0.005 rad. | Compare zero additional reserve against the baseline. Keep true joint limits and explicit measured-state recovery. |
| Grasp ordering favors reserve | `select_grasp` re-sorts IK branches by decreasing joint reserve, then returns the first complete feasible plan. | Rank feasible postures by scheduled approach and total control cost. Keep the best feasible plan if the search budget expires. |
| Slow approach defaults | New contact approaches hardcode 0.35 rad/s; legacy grasp defaults also use 0.35, although callers can override them. | Expose approach speed consistently. Physically test 0.6 and 1.0 rad/s with identical recorded paths before choosing a new default. |
| Inconsistent IK tolerances | Full-pose solver: 0.5 mm / 0.005 rad; CPU admission: 1 mm / 0.01 rad. Arc position tolerance: 6 mm by default. | Use request-level tolerances consistently through proposal generation and admission; test 3 mm contact goals. Keep separate known-robot calibration checks precise. |
| Solver flag used as a veto | Full-pose IK discards GPU-failed proposals before independent FK admission. The arc implementation already admits/refines such proposals. | Check actual residuals, real joint bounds and self geometry; use the optimizer flag as diagnostic information. |
| Collision policy differs by tool | New push/hook support advisory predictions; legacy grasp/arc/free-pose planning still veto predicted environment collisions. | Make the simulation policy consistent and explicit. Prefer observed clear routes, retain advisory fallback, and keep physical contacts enabled. |
| Arbitrary angle API limits | Legacy arc/sequence requests are positive and at most 90°; contact skills allow signed increments only within ±90°. | Support signed larger motions through internal increments and correct progress bookkeeping. Unwrapping must be implemented before permitting arbitrary rotations. |
| Mandatory approach pattern | Legacy grasp always goes through a 70 mm standoff. Common defaults include 6/8 approach holds and 40 closure controls. | Test shorter/local approaches and measured settling. These timings are choices, not physical laws. |
| Predicted grasp treated as execution geometry | Sequence manipulation is planned before closure using an aperture derived from fitted handle radius + 1 mm. There is no post-closure replanning. | Replan from the measured aperture and current part pose. Detect lost engagement and offer local regrasp or push recovery. |
| Joint model bootstrap depends on handle grounding | Applying an independently estimated joint still needs a grounded articulated contact/panel scene. Legacy ground assumes a vertical cylinder with at least 65 mm support. | Decouple an observed part/joint model from the optional handle model, allowing joint-linked pushing without a vertical-handle requirement. |

Relevant implementation locations:

- `src/vlm_tamp/hinged_contact_tools.py`: grasp ordering, fixed standoff, budget
  pricing and first-feasible selection.
- `src/vlm_tamp/rekep_libero/task_ik_curobo.py`: joint reserve, solver flag and
  independent residual admission.
- `src/vlm_tamp/rekep_libero/curobo_v2_worker.py`: internal solver tolerances.
- `src/vlm_tamp/mcvamp_contact_numerical.py`: independent native joint inset.
- `scripts/serve_subscription_robot.py`: predicted closure geometry,
  preplanned sequence execution, angle limits and episode termination.
- `src/vlm_tamp/contact_skill_service.py`: skill budgets and feedback returns.

## Checks that need a different treatment

The rigid-motion fit requires at least eight pairs, 60% inliers within 4 mm,
and 25 mm spatial span. Removing these conditions would let weak or incorrect
correspondences masquerade as measured object motion. The current-v4 practice
and evaluation logs contain 12 contact calls returned for unresolved motion.
The improvement should be better reacquisition/current-surface recovery and
clear uncertainty reporting, rather than treating hand motion as object motion.

The fixed-hinge consistency check (6 mm) protects a model update, not physical
motion. A conflict should support a separately versioned revised hypothesis or
a fresh surface action. It should not silently move a supposedly fixed hinge.

The 1 cm / 3 cm/s controlled strokes, 3 cm release pulses, 120 s skill budget,
30 s initial search and 5 s subsequent planning limits came from the approved
experiment plan. They should be explicit configurable experimental settings.
A timeout is not proof of physical infeasibility. A different limit should be
tested and reported rather than introduced as an undocumented rule.

The current controller clips each joint reference increment to 0.05 rad at
20 Hz. Keep its 1 rad/s reference limit for this comparison; increasing only a
requested speed beyond that would not proportionally increase actual motion.

Finite inputs, correct coordinate frames/TCP conventions, true joint bounds,
observation/model freshness and known-robot self checks remain meaningful.
Strict robot-model calibration tolerances are not sensor/contact tolerances.
Some `1e-6` values are normalized residual thresholds, not micrometre clearance.

No added collision padding was found in the active native convex checks.
CAPT point radii and enclosing cell balls are broad-phase search geometry;
candidate hits are confirmed against configured convex boundaries with zero
padding. Removing broad-phase coverage would create missed collisions, not
remove an unnecessary clearance buffer. Old configuration prose about 1 mm
clearance does not describe this active narrow phase.

## Give recovery more controls

Start with **1,200 controls per learning/recovery episode** and retain the
30-minute wall limit. At 20 Hz, 600 controls are 30 simulated seconds and 1,200
are 60; the current initialization already consumes 60 controls. Reasoning and
image-processing time do not advance simulator physics.

State 49 still had **1,224.13 wall seconds available** when control 600 ended
the episode. Increasing the wall timeout alone would therefore not address
this failure.

The limit must be configured once per run and propagated through the simulator
workers, clients, service status/execution, grasp/contact budget checks, prompts,
protocol metadata and reports. It is currently duplicated across these layers.
Do not merely change `maximum_control_steps` in the JSON configuration.

For extended runs, retain success-at-600 as a milestone and report first success
step, total controls, wall time and recovery count. Keep 600-control runs as a
separate baseline. A longer budget creates recovery opportunities; it does not
demonstrate that the existing policy became more efficient.

## Measured-path timing calculation

Using the saved state-49 joint paths and the actual scheduler, with the same
six endpoint-hold controls and 40 closure controls:

| Reference speed | Approach | First arc | Second arc | Total including closure |
| --- | ---: | ---: | ---: | ---: |
| 0.35 rad/s | 315 | 51 | 90 | 496 |
| 0.60 rad/s | 186 | 33 | 55 | 314 |
| 1.00 rad/s | 114 | 22 | 36 | 212 |

The faster rows are **scheduling calculations only**, not successful physical
replays. Tracking lag, contacts, grasp retention and task success may change.
For approach-only acceleration, preserve contact-phase timing and measure
whether the shorter schedule actually reaches a usable grasp.

## Proposed implementation and validation order

1. Centralize the episode control limit and all reported constraint values.
   Add a 1,200-control condition without changing physics or native success.
2. Replace reserve-first/first-feasible grasp selection with control-cost
   selection, remove extra hard joint insets in the comparison condition,
   and harmonize full-pose proposal/admission tolerances and solver-flag handling.
3. Test the same saved approach at 0.35, 0.6 and 1.0 rad/s. Measure actual robot
   tracking and contact retention; select a speed from outcomes.
4. Replan manipulation after real closure and provide local contact recovery.
   Preserve distinct pinch, hook and push evidence. Improve current-surface
   reacquisition when the motion fit is unresolved.
5. Test budget-only, planner-changes-only and combined conditions while keeping
   Luna and its frozen memory constant. Score success at 600 and at 1,200.
   A separate physically restored state-49 checkpoint can test recovery
   feasibility, but must not be counted as a fresh full-task success.
6. Then practice across varied starting configurations and freeze new
   learner-authored memory for a new evaluation split. The already inspected
   state 49 becomes a development/regression case; it is not a new held-out case.

Keep all failed runs, report the exact rejected quantity and limit, and preserve
the distinction between an incomplete search, an unsupported perception model,
an unreachable candidate and a physically executed failure. Do not add new
collision padding, task-index triggers or successful coordinates to Luna.
