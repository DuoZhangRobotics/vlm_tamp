# Success-focused LIBERO pipeline

The goal of this work is to recover physical task failures before further speed work. Native LIBERO success remains the task outcome. Existing source snapshots and historical scores are preserved. Every episode scheduled in a manifest stays visible, including grounding, planning and pre-motion failures.

## Reproducible entry point

Use `scripts/run_success_pipeline.py` for the seven previously evaluated tasks. `--describe` prints the task-family policies without starting models or simulation. The runner uses the existing research and simulator environments and creates a new output directory.

```bash
PYTHONPATH=src:scripts:.runtime/mc-vamp-deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 \
.runtime/envs/rekep-libero/bin/python scripts/run_success_pipeline.py \
  --output results/success-regression-NEW
```

The official states are 1, 11, 21, 31 and 41, seed 0, five settling actions, and at most 600 control steps. `--tasks` and `--states` select a subset. `--resume` requires the exact saved source hashes and configuration and never reruns a completed episode. Grounding reuses RGB/depth-validated Luna High responses; any needed request uses the existing cumulative $2 ledger, the 12,000-token ceiling and zero automatic retries. These experiments required no new API requests.

## Diagnoses and corrections

### Drawers

The failed closing grasps were rejected by points on the selected drawer front that the RGB mask had left in the stationary map. The offline audit found these points within approximately 0–1.4 mm of the fitted drawer plane. The new scene builder reconciles measured coplanar points within 65 mm of the selected handle and within a 4 mm plane band. They move with the drawer, and functional fingers can contact this bounded local face. They remain obstacles for the palm and arm.

The old fixed 15 cm closing command plus 5 mm allowance left state 31 about 7 mm open. The new target comes from the depth of the dominant parallel cabinet face below the selected drawer, plus a fixed 10 mm seating allowance. This is a depth estimate, not native joint-coordinate feedback. All issued trajectories still complete in full.

### Stove

Some Luna boxes included the burner below the knob. Fitting a pivot to the full box shifted the estimated axis by several centimeters and sometimes generated grasps on the wrong surface. The new estimator selects a raised connected depth component within a bounded expansion of the VLM box, fuses nearby observed endpoints from the calibrated views, and estimates the control axis from that component. The existing bounded direction probes remain; the native stove angle is evaluation-only.

### Bowl

The source-view-only GraspGen input failed state 41. Feeding the fused observed bowl body recovered grasp and transport, but using that input everywhere changed another previously working grasp. The final proposal policy first runs the original source-view search with its 24-candidate/96-posture budget. Only if that search produces no admitted grasp does it generate a second 200-proposal set from the fused observed body, with a larger search budget. Both pools and their rejection evidence are retained. This fallback happens before motion and uses no success oracle or scene reset.

During placement, functional fingers may contact the RGB-D-selected plate. The permission is restricted to the selected support surface and placement/release phases. The palm, arm, table and unrelated objects retain their checks. No geometric path constraint was added to ordinary pick-and-place.

An intermediate disconnected-surface ownership experiment did not fully fix a lift regression; it is recorded separately and is not the final bowl policy.

### Microwave

A failed closing response selected pixels on a mug, and the original cylinder fit selected a 2 mm-high tabletop patch. The revised grounding validates vertical extent and fit residual, then searches for a supported continuous vertical cylinder within the already selected door region when the initial contact fit is inconsistent. The fit is refined on the central bar, excluding mounting regions. No native handle mesh, pose or identity is used.

The old 0.60 rad/s approach could outrun the delta-position servo. Its configured delta range is 0.05 rad, with kp 300 and kd 34.64. Saved failures reached the end of the scheduled approach with substantial lag, then closed while still catching up. Longer open-gripper endpoint settling fixed one opening case. A slower 0.30 rad/s complete approach reduced a failed closing arrival from roughly 9 cm error to under 0.1 mm.

When closing reaches within 3 degrees of the observed closed plane, the runner requests one complete motion covering the remaining angle plus a fixed 2-degree seating allowance. It finishes after that planned segment rather than requiring the estimated plane residual to converge through repeated tiny motions. Physical hard-stop contact is resolved by the simulator; native LIBERO success remains the independent outcome.

The final selection policy reserves room for manipulation within the same 600-step limit. It evaluates complete approach paths at their actual execution schedule, rejects approaches that leave fewer than 200 manipulation steps after closure, and considers up to eight IK alternatives. Joint reserve remains a selection criterion because a nearby root can leave a poor configuration for the later door arc. This is planning-time candidate selection, not an execution veto.

For arm-versus-scene checking, the microwave adapter uses configured convex robot volumes transported by the calibrated seven-joint kinematic chain. This replaces oversized arm-sphere scene checks; upstream Panda self-collision checks remain. Current measured depth is still the scene representation. A 6 mm association allowance is bounded to the observed handle-mount region for functional finger contact.

An existing shallow palm–selected-handle overlap can otherwise prevent any new plan. One recorded rejection was a single point 0.073 mm inside the palm shape. A separate numerical-contact adapter keeps strict checking for ordinary starts, but may permit an existing contact shallower than 0.45 mm with at most 0.05 mm additional tolerance, capped at 0.5 mm. It applies only to the configured palm and current selected contact points. Deeper overlap, arm collisions and unrelated surfaces remain blocking. The new unit checks exercise all three cases.

## Execution and interpretation

Planning prepares a complete joint-reference sequence. The unchanged official joint controller executes every reference; there is no execution-time collision, retention, residual, progress or task-success veto. Observations and subsequent planning occur at segment boundaries. Grasp attachment arises only from physical closure and retention; no weld, object-pose assignment or assisted attachment is used.

These changes address perception, contact-role assignment, grasp selection and execution timing together. They are a pipeline regression, not an isolated planner or model benchmark. Previously inspected states and validated cached VLM outputs cannot establish held-out generalization or fresh-VLM reliability. Concurrent runs are used for success testing; their wall times are not a speed benchmark.

## Evidence and retention

`results/success-first-report/index.html` links the measured outcomes, all development attempts, independent validation and simulation-time previews. The full drawer/stove regression is 20/20 versus 15/20 previously. The final bowl fallback regression is 5/5 versus 4/5 previously. The final microwave closing regression is also 5/5 after the bounded seating change. The final microwave opening regression is 5/5. Together, the final task-family regressions achieved 35/35 native successes, with the native goal also true at every final recorded state. All 35 command traces and movies passed independent verification; 130 focused software tests passed. The scope is the same five previously inspected states per task with cached, validated Luna grounding.

After each episode, workers close, movies are verified, redundant raw observations and image sequences are removed, proposal caches are deleted, and retained observation arrays are losslessly recompressed. Initial/latest RGB-D, constraints, proposals, selected grasps, collision inputs, exact reference/action logs, native outcomes, compact videos and source provenance remain.

The unified runner also releases obsolete native point-cloud indices when a new observation is captured between segments. It retains reuse within a planning stage. This bounds memory across long short-motion episodes; it does not change scene geometry, constraints, or execution references. During development, one superseded opening configuration was stopped between completed motions to reduce RAM pressure; its native outcome, complete command trace and administrative reason are retained.
