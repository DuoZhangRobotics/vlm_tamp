# Full-pipeline timing and lower-overhead execution

The implementation covers steps 1–2 of the requested performance work. It adds fresh DINO/SAM and Luna timing and reduces execution/communication overhead. It preserves the original planner, numeric constraints, contact permissions, controller gains, collision margins, task horizons and 600-step limit. No asynchronous planning or longer trajectories are introduced.

## Entry points

The updated runner is `scripts/profile_mechanism_pipeline.py`. Existing frozen runners remain available as historical baselines. Use the same research and simulator environments; no dependencies were changed.

```bash
PYTHONPATH=src:scripts:.runtime/mc-vamp-deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2 \
.runtime/envs/rekep-libero/bin/python scripts/profile_mechanism_pipeline.py \
  --output results/profile-NEW --backend fast --grounding fresh \
  --tasks open_top_drawer turn_on_stove --states 1
```

`--grounding fresh` runs DINOv2 ViT-S/14 and MobileSAM on the two selected views and makes one real Luna High request per task, with the existing prompt and a 12,000-token ceiling. The returned choices drive the actual pipeline. A request metadata tag separates the explicitly requested timing trial from the durable response cache without changing the model input. Server prompt caching may still apply; usage is retained. Requests keep the shared $2 budget and zero automatic retries.

`--grounding cached` measures subsequent planning/execution using observation-validated grounding. `--backend baseline` uses the original per-command protocol and recording path, with timing instrumentation. Use a new output directory for each run.

## Changes and behavior preservation

`ProfiledWorker` inherits the frozen simulator/controller implementation. A batch invokes the original `CoupledWorker.joint_action` for every reference, preserving target normalization, controller feedback, physical stepping, contacts, native success checks and command order. Only per-command full metadata responses and accumulated-log writes are suppressed; they are emitted at the segment boundary. Tracking errors and task-success flags do not stop execution.

Automatic camera observables are disabled between explicit captures. The unchanged calibrated capture routine still returns all 12 RGB-D views and the same robot-only depth priors. Every joint command retains a video frame. Intermediate images remain in the lossless observation arrays, while redundant PNG and robot-mesh diagnostic exports are omitted. Initial/opened/final PNGs needed by grounding and review remain.

Observation arrays are stored without compression on the critical path and recompressed after worker shutdown and cleanup, with numerical equality checked. Robot frame calibration is reused only for byte-equivalent input records. Point-cloud voxel deduplication uses a monotone mixed-radix integer key with an overflow fallback. It returns exactly the same selected points in the same order as the original lexicographic sort. This changes neither scene geometry nor sampling density.

The instrumentation temporarily wraps engine boundaries and restores them afterward. Original planning source files are unchanged. Source snapshots and hashes identify each run. GPU vision stages synchronize before and after measurement. Timings are nested/inclusive and must not all be summed together.

## Timing scopes

The timeline records process imports; vision/grasp model loading; simulator startup and reset; initial opening/capture; DINO, SAM, keypoint processing; request encoding and the entire Luna round trip; parsing/contact binding; metric constraint construction; GraspGen and candidate selection; IK initialization/queries; scene updates; native planning; execution; finalization and cleanup.

First-motion processing time starts when the opened RGB-D observation is available and ends when the first planned task trajectory is requested. It includes fresh vision and Luna when enabled. A separate episode-to-first-motion interval includes simulator setup, settling, initial gripper opening and acquisition. Model loading is reported separately. The direct monotonic start-to-start interval between manipulation segments measures the actual update cycle.

Execution throughput is commands per wall-clock second while a trajectory is being delivered. Commands retain their original 20 Hz simulated-time schedule. This throughput is distinct from the slower observation/replanning frequency and does not establish hard real-time robot deadlines.

## Measured results

Four task types passed exact-reference replay: middle-drawer opening, top-drawer opening, top-drawer closing and stove turning. Every controller command, measured arm/gripper state and native success flag matched the saved source exactly. Initial and final sensor arrays matched byte for byte. The paired original/fast top-drawer and stove runs also matched every intermediate observation boundary.

The standalone paired top-drawer execution improved from 50.19 s to 6.52 s (7.7x); stove execution improved from 88.92 s to 12.93 s (6.9x). Whole replay time improved approximately 4.3–4.4x. Fast execution throughput ranged from 23.6 to 34.2 commands/s across the four tasks.

Final full-pipeline cached-grounding measurements gave a 2.624 s median manipulation cycle (0.381 Hz) and a 2.038 s pause between completed segments. The earlier 8.944 s cycle was a retrospective file-timestamp estimate; the new value uses explicit monotonic timing. The sub-second full update target is not yet met.

The two final fresh tasks measured 2.96–3.60 s for complete two-view vision preparation, including 0.365–0.413 s for DINO features and transfer. Actual Luna round trips were 16.03 s and 47.03 s. Metric constraint construction took 0.028–0.044 s. Fresh-observation-to-first-motion processing took 30.19 s and 57.43 s, after vision/grasp weights were loaded; model loading itself took 7.24 s. IK-worker initialization remains included in first-motion timing.

Fresh v1 profiling had one success and one planning failure; final fresh v2 profiling likewise had one success and one planning failure. Both cached v2 cases succeeded. These small latency samples are not a new success-rate benchmark: fresh stochastic Luna outputs can change grounding. The saved-reference replays isolate I/O correctness and performance, and are never presented as fresh-grasp success evaluations.

## Verification and retention

70 focused tests passed. An independent audit checks all 13 runs, exact command delivery, native scores, videos, source snapshots, fresh GPU inference, four actual Luna requests and cleanup. Existing planning sources and parameters remain unchanged.

Every run was compacted after worker exit. Necessary observations, selected geometry, generated outputs, trajectories, commands, contacts, timings and verified videos remain. Temporary data reclaimed totaled 7,652,720,265 bytes. The four new Luna requests cost $0.00531875 conservatively; cumulative API cost is $0.257103 of the $2 cap.

The report is `results/pipeline-overhead-report/index.html`. Its `evaluation.json`, `final-validation.json` and per-run monotonic traces provide the evidence. Videos show simulation time and omit planning pauses; they should not be used to infer wall-clock speedup.
