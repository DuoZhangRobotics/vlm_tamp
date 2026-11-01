# VAMP spheres and full-arc planning — 2026-11-01

This records the earlier sphere/full-arc cleanup. The subsequent [heuristic removal and audit](/common/users/dz323/vlm_tamp/docs/heuristic_audit_20261101.md) removes the depth-direction rule and point-mode jaw restriction and records fresh trials.

Implemented from the user's correction: use the existing fast planner, permit finger contact with every environment surface, and attempt the whole requested arc before discretized continuation.

## Current design

- Native planning uses upstream VAMP Panda sphere FK and CAPT sphere/point-cloud queries. Environment queries are SIMD-vectorized across planner states. VAMP's own self-collision implementation remains authoritative; no custom convex collision pass remains in the active Luna planner.
- The pinned sphere order is verified against `panda_spherized.urdf`: arm 1–36, palm 37–54, fingers 55–58. Fixed base sphere 0 remains omitted from environment checks. All finger/environment pairs are exempt, including stationary points and proximal fingers. Arm and palm remain checked. Explicit palm contact permits the palm on the selected contact region.
- Point radius is zero. The previous 2.5 mm radius only served the old broad phase before a zero-padding convex test; retaining it as final sphere padding would introduce an inflation the user did not request.
- cuRobo receives the same upstream arm/palm spheres for robot-frame calibration, but **does not build or check a second environment mesh**. It generates endpoint IK. MC-VAMP owns the observed collision world and path checking.
- For `plan_arc(15)`, the first IK query targets 15 degrees directly. MC-VAMP CRRTC searches the full constrained path to an offered endpoint. Only failure of that attempt permits local continuation. The existing deadline is shared; no time is reserved by a new buffer. A fallback is explicitly recorded under `continuation_fallback`.
- The current endpoint query returns up to eight distinct postures from one cuRobo batch. MC-VAMP can reject a bad endpoint before searching its path. “IK goal offered” does not mean collision-free.
- No SciPy endpoint re-solve, extra endpoint projection gate, or post-plan Python collision/constraint/wrist scan occurs in the active cuRobo/MC-VAMP arc route. The native planner's returned path is used unchanged and time-parameterized for the controller.
- The old global IK zero-joint seed penalty is removed. The previous “local first, global retry” route is removed from the active adapter.
- Sequence hinge calibration is now opt-in (`calibrate_joint=false` by default). It does not silently insert initial 3-degree calibration movements. An explicitly requested sequence feedback interval still controls when that tool observes and replans.
- Existing controlled push execution remains driven by simulator physics and measured feedback. It is distinct from a planned grasped arc.

The adapter still translates the observed scene and supplies the task's hinge-circle constraint to upstream CRRTC. Its existing task-specific constraint residual/projection callback remains; this revision does not replace it with a generated analytic constraint. That callback runs inside planning, not as a second check after planning. The simulator, controller, 600-control budget and 1800-second episode limit are unchanged.

## Validation

- Targeted suite: 149 passed, 3 skipped (historical calibration fixtures). Native tests exercise real sphere collision queries on arm, palm and finger points in static, moving and contact partitions, plus parity with upstream self-collision checking.
- Search-order tests require one full-angle IK query on the successful primary route, continuation only after failure, and unchanged native returned paths.
- Robot sphere coordinates agreed between cuRobo and VAMP within 0.00024 mm in the saved-scene GPU checks.
- The pinned CAPT SIMD query assumes a root split even for a singleton cloud. The binding repeats the identical point for that case; the represented geometry is unchanged. This is covered by the one-point collision tests.

Saved opening-state-36/control-584 diagnostic, without simulator execution:

| Request | Result after assigning collision ownership to MC-VAMP | Planning wall time |
|---|---|---:|
| 5° | Full-angle endpoint attempts rejected; continuation then found a complete path | 1.87 s |
| 15° | Full-angle endpoint attempts rejected; continuation also failed | 1.03 s |

The initial numerical check retaining cuRobo's separate mesh world failed both requests before native planning. That evidence is preserved in `results/vamp-spheres-full-arc-check-20261101`. The final single-world architecture is tested in `results/vamp-spheres-one-world-check-20261101`. These are numerical planning tests, not physical success claims. The surviving rejections concern upstream arm/palm spheres, not finger or convex tests.

## Fresh evaluation

Study: `results/luna-vamp-spheres-full-arc-20261101`.
Two sequential fresh Luna/high evaluations: closing state 17 and closing state 49. No inherited notebooks, no successful coordinates or task-specific script. The tool introduction describes the changed generic policy. The study freezes source hashes and native binaries. RAM is capped at 16 GiB with no swap; the existing two-second RAM/I/O watchdog and GPU monitoring are retained. Dashboard publishing/mirroring remains throttled to 60 seconds and does not reload the page automatically.

The frozen evaluation used implementation commit `256e7ef`:

| Closing state | Native task success | Controls | Total actor wall time | Robot service | Luna/transport/local analysis outside service |
|---|---|---:|---:|---:|---:|
| 17 | Yes | 538 | 270.1 s | 112.0 s | 158.1 s |
| 49 | No | 600 | 469.7 s | 112.9 s | 356.8 s |

Both actors selected pushes, with no `plan_arc` calls. These physical results therefore do not establish improved arc performance; arc integration was tested separately above. Contact search/stroke planning totaled 2.10/1.89 seconds for state 17 and 5.93/3.56 seconds for state 49. Observation/execution and actor work account for most wall time.

Peak job RAM: 7.36 GiB; sampled GPU use: 1985 MiB. Across 384 watchdog samples, maximum full I/O pressure was 0.05%, memory pressure was zero, and there were no OOM kills or resource interruptions.

State 49's door went from 107.26° to 119.98° open. The first two approach/insertion sequences opened it to 110.31° and 120.08°; their subsequent push strokes ended at 110.33° and 119.98°. Later strokes mostly left it at 119.98°. Native contact logs show unsuccessful engagement during the strokes. No contact-search-exhausted stop occurred. This is an ineffective contact-acquisition/recovery failure, not the removed finger-collision veto.

A separate grounding error is confirmed: the initial caller hint selected +Z, which matches the native closing axis. The adjacent-depth prior selected −Z and rejected the disagreement; the actor then explicitly omitted its hint and installed the prior's −Z. However, executed pushes used the constant world direction `[0.91917, -0.39386, 0]`, not a joint-linked command, so the wrong axis alone is not demonstrated to cause the executed opening. This heuristic has not been changed as part of the planner cleanup.

After the frozen trials, stale explanatory contact-permission strings were corrected to match the already-active finger exemption. Original run logs are preserved. No physical collision policy changed in that reporting correction.

Results are in the study manifest, per-trial timing breakdowns and `watch.html`. A repeated run of these two known states is a regression check, not evidence of broad generalization.

## Scope

This replaces the active `mcvamp_contact_numerical` route imported by Luna's shared contact tools. Old offline `mcvamp_microwave`, `mcvamp_mechanism` and historical SciPy arc implementations remain in the repository; they are not fallback routes used by this evaluation. The previous audit is a historical snapshot, not a description of the new architecture.
