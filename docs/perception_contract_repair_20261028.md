# Perception and state-estimation contracts — 2026-10-28

The scene contains useful information, but the pipeline did not consistently use it or preserve what it had learned. The failures cannot fairly be attributed solely to Luna's visual ability. Specific software errors have been repaired; **the full task is still not reliably solved**.

Use the existing [watch.html](../results/luna-live-hinge-comparison-20261027/watch.html) in VS Code Live Preview. Its October 28 section contains the images, both fresh validation episodes, profiling and input/output logs. All robot-actor runs requested **gpt-6-luna, high reasoning**. No teacher, reflection reviewer, worked solution, native hinge/contact feedback, or fine-tuning was supplied.

## Evidence and repairs

| Finding | Evidence | Implemented change |
|---|---|---|
| Useful views were buried | The two default previews mostly show the back of the open door; other recorded cameras clearly show the handle and attachment. Twelve cameras already existed. | One labeled camera overview at reset and on unresolved-motion/contact-loss recovery. Open originals for pixel annotations. Multiview grounding is available with the motion toolkit. |
| The tool changed part identity | The original contact box had a **10.77 mm** cylinder residual, above **3.5 mm**. Recovery moved the box from `[414,248,457,357]` to `[410,178,443,260]`, selecting another vertical feature. | Report the failed fit and explicitly unconfirmed candidates. Do not install another feature as the handle. Known robot-depth samples are excluded from panel/handle fitting. |
| The direction attribution was wrong | In the failed run, **Luna's image hint implied +Z**. The depth heuristic overrode it with −Z. In a different recorded grounding, the hint was wrong and the depth prior was right. | A conflict leaves grounding unresolved. Neither source silently wins. The caller rechecks or explicitly delegates an unknown image hint to the depth prior. |
| Stationarity was discarded | After an ineffective action, **49/53** tracked points were stationary; displacement median and P90 were **0 mm**. The hinge fitter discarded stationary points and failed, then the scene moved with the hand prediction. | Stationary selected geometry can yield an observed identity pose without a hinge estimate. RGB-D mode no longer substitutes hand motion for missing object motion. Stale model geometry cannot seed current background tracking. |
| Parameters and changing geometry were mixed | A **5.50 mm-error** motion-supported hinge was replaced by a later visual guess with **48.06 mm error** when the actor refreshed geometry. The native hinge did not move. | `refresh_contact` updates visible handle/panel geometry while retaining the fixed joint. Historical joint reuse is explicit. Joint-only updates preserve point roles and part pose instead of rerunning old image selections on new frames. |
| Measured motion lacked a clean application path | The new part tracker measured motion, but the old `apply_tracking` belonged to another tracker and could fail before initialization. | `apply_part_motion` accepts a registered track and matching reference/current captures; it preserves the hinge and rejects inconsistent pose updates. Missing legacy tracking now produces a clear precondition error. |
| Grasp failure was not prominent | Pad gap changed from **24.59 mm to essentially zero** during a 30° move. In the second run, a combined sequence closed to **0.16 mm** for a modeled **19.52 mm** bar. | `grasp_evidence` reports calibrated pad gap, bar diameter and gap collapse/near-closed evidence. This is proprioceptive evidence, not a tactile contact certificate. Combined-sequence compaction now retains stage evidence and timestamps. |

The last sequence-display and near-closed labeling fixes were made **after** the second frozen episode. They passed recorded-response and regression tests; they have not had another full Luna run. Single-view grounding now explicitly states that plane/cylinder residuals do not verify hinge accuracy.

## Why a small fit residual was insufficient

An old held-out interval rotated only about 3°. A 2.5 mm point residual over such a small rotation permits a large range of pivots. The two independently fitted pivots differed by **41.85 mm**, even though the axis-prediction residual met the old consistency threshold. Bootstrap spread is not a calibrated accuracy guarantee.

In the first repaired run, a later **25.86°** interval gave **1.54 mm** prediction RMS and **1.96 mm** disagreement between independently fitted pivots. This is stronger evidence, not a universal guarantee of millimetre accuracy.

A separate late pose fit looked precise at **1.60 mm RMS**, but applying its free rigid transform would displace the retained fixed hinge by **17.18 mm** and put **96.4%** of contact samples in multi-view observed free space. That transform made a numerical 8° query solve, demonstrating why a planner pass alone is insufficient. It was **not executed** and is **not counted as a successful repair**.

The actual `apply_part_motion` API rejects that update without changing the model. It accepts an earlier measured update with **4.73 mm** fixed-hinge displacement and about **4.6%** contact-depth conflict. Pose admission checks the configured position allowance and rejects more than 10% contact samples contradicted by at least two views at 5 mm depth tolerance. These are declared model checks, not probabilities or a physical grasp guarantee.

An experimental soft hinge prior for choosing the largest image plane did not reliably distinguish the cabinet from the door. It was removed. Exploratory joint-constrained registration diagnostics also remain experiments, not deployed fixes.

## Collision policy: pads versus finger bodies

The recent perception and tolerance repairs did **not** change the contact exemption. The tracked baseline already had the same rule:

- Configured fingers, including pad geometries, may contact the selected handle and bounded nearby mount region.
- Fingers/pads are still checked against stationary and other non-contact geometry. Zero finger clearance means no extra margin; it does not disable penetration checks.
- Palm and arm geometry remain checked.
- The cuRobo proposal model omits fingers from its collision proxy. Final native admission still performs the checks above.

The late query rejected `gripper0_finger1_collision` and `gripper0_finger2_collision`, not merely the separately named pad geometries. A replay located nine stationary samples inside finger 2's convex body by **0.14–5.16 mm**. All project inside the selected contact box, but their nearest selected-contact samples are **22–27 mm** away and the fitted panel plane is **117–128 mm** away. That is a concrete image/3D-association mismatch; image-box membership alone does not establish that every sample should be exempted.

Two of the nine witnesses also overlap a pad geometry; seven do not. Disabling only pad checks therefore would not remove the finger-body rejections. These are overlaps in the observed model at a proposed configuration, not measurements of physical penetration. No obstacle points were deleted to force a planning pass, and no global pad-contact policy was silently introduced.

## Live validation results

These are separate fresh contexts on one known development case, not a statistical comparison or generalization benchmark. Frozen sources and the original 600-control / 1,800-second limits were retained for each run.

| Run | Initial hinge error | Final opening | Outcome |
|---|---:|---:|---|
| Previous planner-repair follow-up | 254.49 mm | 119.98° | Failure; started 88.92° open and opened farther |
| Perception repair v1 | 5.12 mm | 22.89° | Failure at 600 controls; substantial closing, later grasp/model problems |
| Perception repair v2 | 36.99 mm | 65.79° | Failure at the wall limit, 540 controls |

The second run shows the remaining inconsistency: richer observations and better contracts do not ensure that Luna independently checks a single-view hinge. It did not use multiview grounding or the new measured-pose application operation. Four grasp/sequence-planning calls consumed **1,165.80 s**. The actor spent **421.50 s outside robot calls**; this includes inference, image inspection, analysis and transport. A final 20° sequence was found only after about **495 s** of search, too late for the actor to execute it. Its predicted 60 controls would have exhausted the remaining physical budget and would not by itself guarantee closing the remaining door angle.

Current limits are therefore **inconsistent grounding validation, physical grasp recovery, and expensive grasp search**. They are not explained by insufficient GPU memory or an accidentally selected Qwen/Astra actor. Do not describe the full pipeline as solved or promote either failed episode as a successful learned method.

## Durable implementation rules

1. Keep fixed joint parameters separate from changing part pose and contact geometry. Preserve their evidence and timestamps.
2. A robot motion, close command, IK solution or planner success is not proof that the object followed or that a grasp exists.
3. Stationary evidence is useful for pose estimation even when it is uninformative for hinge estimation.
4. Never silently replace a selected semantic part or override conflicting direction evidence.
5. Preserve decisive negative evidence through every compact response, including nested action sequences.
6. Keep native success/geometry/contact grading separate from actor inputs. Use only public RGB-D, calibration and robot proprioception for the tools.
7. Report failure as module → measured evidence → allowed range/assumption → consequence. Distinguish collision geometry labels from anatomical pad/finger regions.

## Artifacts

Root: [perception-contract-repair-20261028](../results/perception-contract-repair-20261028/).

- `source-before/`, `source-final/`, `v2/source-final/`: preserved code stages and frozen actor runtimes.
- `grounding-replay.json`, `motion-replay.json`: original-input regressions, including four stationary intervals.
- `robot-contamination-before.json`: 73/2,618 contact samples in one intermediate fit matched robot depth; none did in the late 49th grounding. This was not proven to cause the late failure.
- `collision-witness.json`, `pad-versus-finger-body-check.json`: specific rejection geometry.
- `pose-api-validation.json`: real API accepts supported early motion and rejects the contradicted late update atomically.
- `contact-refresh-replay.log`: fixed-joint/contact separation before the rejected plane-ranking experiment. `rejected-plane-prior-*` preserve the unsuccessful experiment.
- `luna-summary.json`, `luna-completed-analysis.json`, and their `v2/` counterparts: complete trial records and offline grading.
- `post-run-display-validation.json`, `tests-final.log`: final reporting/evidence regressions.
- Main viewing entry remains the existing `watch.html` linked above; no new viewer setup is required.
