# MC-VAMP microwave experiment

This experiment compares the frozen Atlas pipeline with MC-VAMP planning and timed execution on LIBERO microwave opening and closing. The baseline and its source snapshots remain unchanged.

## Planning and execution boundary

The input is the baseline's observation-validated RGB grounding, metric depth, calibrated robot model and cached GraspGen candidates. Existing frontal grasp refinements and cuRobo multi-IK remain. No new VLM requests are made, and simulator object states, identities, meshes and articulation parameters remain evaluation-only.

MC-VAMP plans the approach and maintained-contact motion. It first tries the remaining task goal. If that cannot be planned, it uses a complete 20-degree or 10-degree contact trajectory, following the baseline's horizon policy. The entire selected trajectory is issued before observing and planning again. The simulator pauses while a new plan is computed.

`timed_execution.execute` accepts joint references and a simulator client. It has no scene checker, constraint object, progress gate, contact gate or success-triggered early exit. It sends every scheduled reference through the official Robosuite joint-position controller (20 Hz simulated control, kp 300). The driver retains finite-command and configured joint-limit handling and its 0.05-radian joint-delta scale. MuJoCo contacts, dynamics and physical gripper closure remain enabled; there are no welded attachments or object state assignments.

Trajectory timing uses cumulative joint path length, not the number of dense planning samples. The approach uses 0.60 rad/s maximum joint speed and a 20-control endpoint hold. Contact trajectories use 0.35 rad/s and a 12-control endpoint hold. A complete trajectory must fit the remaining 600-control episode budget before it starts. Tracking error and native contacts are recorded for evaluation and cannot interrupt the trajectory.

RGB-D tracking, upright hinge fitting and scene rebuilding occur only between completed contact trajectories. A subsequent planning failure can end the run, but it cannot retroactively change a native task success into failure. The report separates native task outcome, final native outcome, trajectory completion and pipeline termination.

## Native adapter

The upstream checkout is `third_party/mc-vamp`, branch `constrained_planner`, revision `24e0f8adef0649a957fec77bf33f8c2f04515a18`. Upstream sources were not edited for this experiment. The bridge is `src/vlm_tamp/native/mcvamp_microwave.cc`; its binary and source hashes are saved in `.runtime/mcvamp-microwave/build.json` and each run protocol.

The adapter uses upstream constrained RRT-Connect, composable projection, Panda FK/self collision and CAPT point-cloud queries. For environment collision queries it uses native arm spheres 1–36 and a sphere cover of the calibrated gripper's configured convex volumes. This replaces the bundled fixed-open hand/finger geometry for environment checks. The upstream self-collision model remains unchanged. Gripper cover cells are at most 12 mm wide and are an approximation of the configured robot geometry.

Observed points are tagged as stationary, moving door, or permitted contact region. Only fingers may contact the observed selected bar and its bounded local mount region; palm and arm still query these points. The policy matches the baseline's bar association (28 mm) and local panel/mount association (65 mm, bounded by measured panel band and handle height). It does not delete the handle from the environment. These approximate contact pairs do not provide a penetration certificate.

Static clouds use 7.5 mm voxels, moving noncontact clouds 5 mm, and contact clouds 3 mm, with a 2.5 mm point radius. CAPT indices use tight 6–81 mm query-radius bounds and are reused between candidates. The observed door cloud moves about the perceived hinge; query spheres are transformed into the reference-door frame. The inferred solid-panel completion used in the baseline is disabled. Unobserved space and door-versus-environment collision are not certified.

The seven native coordinates are the seven arm joints. Door progress is derived from the configured finger-pad midpoint:

    p(q) = TCP_position(q) + TCP_rotation(q) * local_pad_midpoint
    theta(q) = signed angle of the pad's radial vector about the perceived hinge

The pad must remain at its initial axial height and hinge radius. Aligned mode additionally keeps jaw closure perpendicular to the vertical bar. The alternative permits jaw tilt up to 60 degrees. This represents the baseline's point-on-hinge-arc constraint while eliminating its explicit eighth door-angle coordinate. The up-axis cone has the same 90-degree allowance as the baseline; gripper twist remains free.

The custom projection step currently uses scalar finite differences and damped least squares inside upstream MC-VAMP's composition. Consequently this is an MC-VAMP adaptation, not a reproduction of the paper's fully vectorized projection timings. Native solves run in isolated CPU workers with bounded lifetimes. The bridge explicitly aligns AVX inputs and stores; worker failures become recorded planning failures.

## Protocol and reproduction

The frozen evaluation uses official initial-state indices **1, 11, 21, 31, 41**, seed 0, five settling actions and at most 600 control steps. The same frozen grounding is checked against the new rendered observations. Missing/incomplete baseline VLM outputs remain recorded input failures and stay in the denominator. There are no paid retries. Task success is LIBERO's actual predicate, recorded independently of the policy.

From the project directory:

```bash
.runtime/envs/rekep-libero/bin/python scripts/build_mcvamp_microwave.py

PYTHONPATH=src:scripts:.runtime/mc-vamp-deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.runtime/envs/rekep-libero/bin/python scripts/run_mcvamp_microwave.py \
  --output results/mcvamp-microwave-new-run --states 1 11 21 31 41

PYTHONPATH=src:scripts:.runtime/mc-vamp-deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.runtime/envs/rekep-libero/bin/python scripts/report_mcvamp_microwave.py \
  --runs results/mcvamp-microwave-new-run \
  --output results/mcvamp-microwave-new-report
```

The protected LIBERO Conda environment is not modified. Source/configuration snapshots, the native build revision, controller settings and thread settings are recorded before the first evaluation episode. Source hashes are rechecked before every episode.

The combined comparison changes planner backend, collision approximation and execution policy. It therefore cannot attribute a success or timing difference solely to MC-VAMP. `scripts/benchmark_mcvamp_saved_microwave.py` separately replays saved baseline planning queries and reports native search, dense resampling, worker overhead and CAPT construction.

## Storage

Each episode's `finally` block closes workers, saves actions/outcomes, compacts its video, removes duplicate raw observations and frames, compresses large diagnostic JSON and clears native geometry caches. The retained evidence includes summaries, commands, native evaluations, video, selected keypoints, numeric paths, deduplicated scene clouds and planning inputs. Superseded development runs retain compact diagnostics and videos; their duplicate raw sensor archives can be pruned after the frozen comparison is complete.

## Artifacts

- Frozen evaluation: `results/mcvamp-microwave-frozen-20261015`
- Paired report: `results/mcvamp-microwave-comparison-report/index.html`
- Saved-query component benchmark: `results/mcvamp-microwave-saved-query-benchmark`
- Focused checks: `tests/test_mcvamp_microwave.py`, `tests/test_timed_execution.py`, `tests/test_coupled_storage.py`

The development opening episode passed native task success after five completed trajectories. The development closing episode completed two trajectories but lost handle contact and stopped before closing. These are development observations; the frozen report contains the paired evaluation outcomes.

## Frozen results

| Task | Frozen Atlas baseline | MC-VAMP plus timed execution |
| --- | --- | --- |
| Open microwave | 0/5 | 3/5 |
| Close microwave | 1/5 | 0/5 |

All 20 issued contact trajectories completed, with zero execution vetoes. Native final success agreed with success-ever in all ten episodes. Opening states 1, 31 and 41 finished at 77.96, 75.03 and 75.64 degrees respectively and passed LIBERO's actual predicate. Opening states 11 and 21 stopped at 4.07 and 66.77 degrees when the native planner could not produce a further path.

Closing state 1 executed two contact trajectories but stopped at 42.30 degrees open. Native evaluation recorded one-sided handle contact and zero controls with both finger pads contacting the handle; final handle contact was absent. The remaining closing failures were one invalid frozen handle region/depth support and three incomplete frozen Luna outputs. These four input failures remain in the denominator.

The median preparation time for the 20 executed contact plans was 2.54 seconds (range 2.09–4.11), including endpoint proposals and native scene setup but excluding fresh perception. The saved-query component benchmark solved 12/15 queries; successful queries had median native search 3.47 ms and 37.22 ms including dense resampling/worker overhead. Median cold CAPT construction was 0.45 seconds. Different collision approximations prevent a planner-only speedup claim.

The comparison made zero API calls and left the cumulative $2 ledger unchanged at $0.22330875 spent. Automatic per-episode retention reclaimed about 1.20 GiB during this frozen run. Additional pruning removed redundant development searches and sensor copies while retaining their outcomes, videos, commands, numeric paths and representative failure geometry.

The post-run audit passed all ten cases, checked source snapshots/baseline preservation, and verified complete reference counts, native outcomes, zero execution vetoes and retention. After evaluation, a synthetic cache regression exposed a possible collision between different static/contact array partitions. The current runner adds role and shape tags to its cache key. All 32 scene contexts used by the frozen evaluation were audited and had **zero partition aliases**. The measured source version remains in `source-snapshot`; `post-evaluation-hardening.json` records the later changes. The frozen version passed 18 focused checks before evaluation; the hardened current version passes 20.

Two WSL dumps produced by early native adapter crashes were identified by project executable and recorded process IDs and removed, reclaiming **21.64 GiB outside the project**. The current native worker disables core dumps and cleans only its own matching WSL dumps after a worker failure. The cleanup regression checks that other process IDs and executable names are preserved. Compact failure logs and source snapshots remain available.
