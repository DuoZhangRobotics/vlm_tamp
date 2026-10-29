# 3 mm arc tolerance and a working cuRobo continuation prototype

Follow-up: [continuation is integrated and the fresh simulator replay succeeds](curobo_v2_continuation_20261022.md).
The prototype results below remain as the preceding controlled experiment.

The user requested a realistic positional tolerance and a concrete cuRobo fix
for the failed second pull. The production arc-position acceptance is now
**3 mm**. An isolated cuRobo continuation prototype now plans the complete
previously failing 30-degree arc. The continuation method is not yet integrated
into the normal service, and no fresh physical replay is claimed.

[Open the report](../results/curobo-v2-tolerance-20261022/report.html).
[Previous failure diagnosis](curobo_v2_failure_diagnosis_20261022.md).

## Positional acceptance

The old 1-micrometre criterion came from the numerical constraint solver, not
from measured perception or grasp accuracy. Tight numerical optimization can
still aim at the model's centerline; it should not impose an unjustified
physical acceptance threshold.

`arc_position_tolerance_m = 0.003` is shared by native arc projection, native
dense-path admission, CPU endpoint acceptance, and cuRobo arc IK success
tolerance. The failed-GPU-candidate polish capture range is at least this
tolerance. Grasp IK retains its separately configured criterion.

Native positional error is the Euclidean norm of radial and axial error to
the **original fixed estimated hinge circle**. It is not a fresh 3-mm allowance
accumulated at each intermediate step. CPU endpoint positional error is the
distance to the requested point on that circle; goal progress and wrist limits
remain separate checks. Collision permission/padding is unchanged.

The native residual previously mixed positional error with weighted angular
terms. Replacing every `1e-6` with `.003` would also relax angular constraints
and upstream joint-update stopping rules. Instead the native convergence metric
is now explicitly scaled dimensionless:

```
1e-6 * max(circle_position_error / position_tolerance_m,
           weighted_angular_error / 1e-6)
```

The upstream threshold remains an implementation detail. Logs expose physical
`position_error_m`, `position_tolerance_m`, and angular error independently;
`residual` is labelled `scaled_dimensionless_acceptance_ratio`. Do not read
that field as metres. The final appended endpoint now must satisfy the same
constraint, in addition to collision admission.

Validation:

- Ten native integration checks passed: 2.5-mm radial/height offsets are
  accepted unchanged; 3.5-mm and combined 2.5+2.5-mm offsets require correction;
  angular and palm-collision rejection remain active.
- Five previously successful GPU planning cases still solve. Independent
  audits using the final native build found maximum positional errors of
  2.993, 2.811, 2.988, 2.764, and 1.698 mm. All sampled constraints, collision
  checks, wrist checks, and endpoint replans passed.
- Focused Python checks: 17 passed, five skipped because old fixtures are
  absent.
- Changing only the positional tolerance does **not** solve the newer failed
  state: its original endpoint planner still exhausts the budget in 22.32 s.

Evidence: [boundary checks](../results/curobo-v2-tolerance-20261022/tolerance-checks003/checks.json),
[five saved paths](../results/curobo-v2-tolerance-20261022/saved-path-audit.json),
[Python checks](../results/curobo-v2-tolerance-20261022/tests.log),
[failed-state control](../results/curobo-v2-tolerance-20261022/failed-state-3mm/results.json).

## Concrete cuRobo correction: local continuation

The existing global endpoint formulation repeatedly finds distant postures,
then struggles to connect them while holding the handle. Supplying more local
seeds to that global solve did not repair it. The prototype changes the solve
formulation:

1. Plan the complete arc as ten **3-degree intermediate targets**, always
   computed from the same original arc model. Nothing executes during this
   process.
2. Seed each target from the preceding admitted posture, with nearby seeds
   only. Disable global LM reseeding through the configuration field actually
   read by the pinned solver: `solver.config.use_lm_seed = False`.
3. Assign a planning sample duration from the desired arc speed: 3 degrees at
   15 degrees/second gives 0.2 s. This loads the correct current posture into
   cuRobo's velocity-related objective and bounds. It is not an arbitrary
   timestep inserted into untimed global IK. LM seed velocity weight is zero;
   the active local rollout velocity regularization uses weight .001.
4. Move the observed moving-surface map to each intermediate angle. Use current
   measured moving points and a 2-mm moving mesh; stationary mesh remains 5 mm.
5. Admit each candidate and connect it to the preceding posture using native
   constraints, collision checks, and the full-arc wrist limit. Assemble and
   validate the whole path before any possible execution.
6. Reuse the local optimizer **within this one arc**. Start a fresh solver for
   a new arc/recovery request so failed or unrelated global queries cannot
   silently alter the next planning problem.

The pinned public IK API exposes no independent untimed posture-target
argument. Its internal position cost supports one, but using it would require
an adapter/custom-cost extension. The timed local formulation above is
supported and has now been tested. The prototype retains native wrist/jaw
admission; it does not pretend those inequalities are already encoded as
matching hard GPU constraints. A future all-cuRobo constraint formulation must
implement and validate those terms explicitly. Final execution timing would
still use the trajectory scheduler.

## Measured results

Same failed saved state, observations, calibration, and 30-degree request:

| Variant | Result | Planning increments | Worker + scene initialization included |
| --- | --- | ---: | ---: |
| Original global endpoint solver, 3 mm tolerance | Budget exhausted | Not applicable | Benchmark wall 22.32 s excludes worker initialization |
| Local continuation, fresh solver each increment | 10/10 connected | 28.645 s | 36.441 s |
| Local continuation, solver reused within arc | 10/10 connected | **14.014 s** | **23.339 s** |

Optimizer reuse reduces GPU-call wall time from 19.377 to 3.843 s. Native
connections take 0.671 and 0.662 s respectively; remaining increment time
includes CPU candidate polishing/admission and diagnostics. The final wrist
excursions are about 28.8 and 28.7 degrees, below the 63-degree full-arc limit.
Both paths reach 30 degrees in the planning model. All issued numerical path
checks pass. These are development planning results, not physical success or
held-out generalization measurements.

An independent whole-path audit found 83 and 78 samples, respectively, with
maximum positional deviations of **0.180 mm and 0.142 mm**. Thus the successful
prototype stays well inside the allowed 3-mm tube. All sampled constraints,
collision checks, and whole-path wrist checks pass in the final native build.
See [whole-path audit](../results/curobo-v2-tolerance-20261022/continuation-path-audit.json).

Evidence: [fresh-per-increment run](../results/curobo-v2-tolerance-20261022/continuation-local002/result.json),
[reuse-within-arc run](../results/curobo-v2-tolerance-20261022/continuation-reuse001/result.json),
[GPU requests/results](../results/curobo-v2-tolerance-20261022/continuation-reuse001/curobo-v2-queries.json),
[prototype code](../results/curobo-v2-tolerance-20261022/probe_continuation.py).
`continuation-local001` stopped at the source/build hash guard before planning
because the native build was being finalized; it is not an IK failure.

## Integration next

Promote the local continuation mode through a deliberate arc-scoped worker API,
and give it a reserved portion of the search budget. Do not wait until several
five-second global connection attempts have exhausted the budget. Keep the
fixed anchor and full-path native checks. Validate solver reset between arcs,
the five old positive cases, the previously failed state, then fresh physics
with the same decision sequence. Adaptive Luna practice follows that numerical
and physical validation; no change to Luna prompts or task-specific memory is
needed for this fix.

Production changes in this turn are the positional tolerance and its logging,
plus native endpoint admission. Local continuation, corrected local objective,
and changed moving-mesh setup remain in the isolated prototype.
