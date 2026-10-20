# Drawer insertion: generation, independent checks and correction — 2026-10-11

The VLM generated a structured task specification and manipulation constraints for an existing drawer-insertion scene. An independent simulator verifier measured the mechanism and tested static placement samples. A second VLM call received those scoped facts and correctly supplied the measured world axis and signed joint limits. This mechanical subset required no human expert annotations. **Robot motion and full task execution remain unverified.**

The trial used `libero_goal-03-state001-seed0`, the same published instruction “open the top drawer and put the bowl inside,” and two calls to `gpt-5.4-mini-2026-03-17`. Both responses passed schema/input validation without structural repair. No retries or additional calls were made. Estimated new usage was **$0.01332975**; project totals are **114 calls / $0.35803960**.

## What the VLM produced

The proposal declares the cabinet, top-drawer region and bowl; final `Open(drawer)` and `In(bowl, drawer)` predicates; opening and placement stages; a prismatic `follow_joint` constraint; and a placement/contact declaration. Initial numerical axis and range parameters were unresolved. All model-generated constraints retained `verified=false`.

After verifier feedback, the model supplied:

- Joint mechanism: **prismatic**.
- World-axis direction: **[0, -1, 0]**, checked sign-invariantly against the actual model and joint sweep.
- Native joint limits: **[-0.16, 0.01] m**, in the joint-coordinate frame.

Those are measured simulator facts supplied through explicitly oracle-assisted feedback, not geometry inferred from RGB or a claim of calibrated model confidence. The generated goals did not change. The model's additional final `Open` predicate differs from the published BDDL containment-only goal, so exact benchmark agreement remained false in both passes. Mechanics cannot decide whether the user wants the drawer to remain open afterward; ordinary desired-outcome clarification is enough for that issue.

## Independently checked claims

| Pass | Raw/linked schema valid | Verified checks | Contradicted | Unresolved |
|---|---|---:|---:|---:|
| Proposal | Yes / Yes | 5 | 0 | 7 |
| Simulator-feedback correction | Yes / Yes | 7 | 0 | 5 |

These are **scoped checks**, not percentages of correct complete constraints. They include structural validity, joint-state satisfiability, sampled predicate/collision consistency, joint type and forward kinematics. Correction resolved two numerical claims, axis and limits. Required/conditional strategy declarations are not certified merely because their mechanics are correct.

Three deliberately corrupted copies of the corrected proposal—wrong mechanism, orthogonal world axis and wrong signed limits—were all contradicted by the same checker. These are diagnostic controls, not new model responses. See `results/drawer-verifier-trial/negative-controls.json`.

## Static placement evidence and its limits

The verifier loads the exact public environment/initial state, checks source hashes and re-render equality, measures all three eligible drawer joints without goal filtering, performs nine-value joint sweeps, and directly repositions the actual bowl collision geometry at 125 sampled poses per drawer state. It restores the original scene afterward; the restored RGB is identical.

| Drawer state | Joint q | Samples | Official `In` true | Pass ≤0.5 mm penetration tolerance | Also all collision-box vertices inside semantic site |
|---|---:|---:|---:|---:|---:|
| Closed | 0.01 m | 125 | 125 | 3 | 0 |
| Open | -0.16 m | 125 | 125 | 36 | 0 |

The official site `In` predicate checks an object's reference point, and site contact returns true. Thus all 250 point-membership successes do not prove collision-free or whole-object placement. Additional checks use the actual 40 box collision geometries, contact penetration distances, and an exact vertex test in the rotated semantic-site box.

There are sampled static poses satisfying official point containment plus the stated collision tolerance. **None tested certifies that all bowl collision-box vertices fit inside the semantic site box.** This conservative proxy is not an exact physical drawer-volume model, and a finite fixed-orientation search cannot prove impossibility. Neither static pose existence nor its absence proves a reachable insertion path. Floating/nonpenetrating poses also do not establish grasp stability or stable support.

Probe images are labelled as directly repositioned static tests. They are not demonstrations of robot execution. Full containment, access-path necessity, contact strategy, dynamic stability and executable robot motion remain unresolved.

## Independence and reproducibility

The first request contains only public instruction, RGB and candidate inventory. Correction receives the original model proposal, scoped status/reasons and measured entity/joint/body/axis/range facts. It receives no benchmark goal answers, expert labels, private code, task IDs, filenames or initial-state vectors. The environment is loaded from the official task file, but its benchmark success function is not used by the physical probe. Benchmark goal comparison runs only after both model passes and never enters feedback.

`configs/frozen_verification_trial_protocol.json` records prompt/schema, evidence and implementation hashes, maximum two attempts, no retries, a 2,000-token output cap and $0.08 admission budget. `results/payload-audit-verification-trial.json` records the public-data boundary. Model temperature was 0, reasoning `none`, high image detail. Both returned model IDs matched the snapshot.

**103 tests passed**, including real scene collection, independent drawer probing/restoration, wrong type/axis/limit controls, frame/unit handling, answer-compartment exclusion, paid guards and a local fake two-pass API test. Formatting/static/dependency checks passed. The pinned simulator stack retains 20 deprecation warnings. No human-reviewed labels or motion-planner integration were created.

| Accounting | Value |
|---|---:|
| New API attempts | 2 |
| Input tokens | 8,227 |
| Output tokens | 1,591 |
| Estimated usage USD | 0.01332975 |

Prices were checked on the official [GPT-5.4 Mini page](https://developers.openai.com/api/docs/models/gpt-5.4-mini) on 2026-10-11. Input usage is conservatively priced at the uncached rate; provider invoices remain authoritative. Credentials were used only for authentication.

## Outputs

- `results/drawer-verifier-trial/report.html`: nontechnical claim/status report with original and static-probe images.
- `results/drawer-verifier-trial/phases.json`: both complete model outputs, scoped checks, exact correction payload and final-only benchmark assessment.
- `results/drawer-verifier-trial/grounded-specification.json`: candidate specification plus verified mechanical bindings; `ready_for_robot_execution=false`.
- `results/drawer-physics/facts.json`: independent state/source hashes, joint sweeps and every placement sample.
- `results/drawer-verifier-trial/summary.json`: exact accounting and scope-limited outcomes.

The next verification work is an actual drawer-volume model and checked access/robot trajectories, using the existing planner when that integration is authorized. The current trial already establishes an automatic generator/checker/correction loop for mechanical facts; it does not replace missing motion or task-intent evidence with model-generated labels.
