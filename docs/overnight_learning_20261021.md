# Scene repair and Luna learning study, 2026-10-21

Later runtime audit: the isolated learner lacked the promised NumPy/SciPy
environment, and several runs logged NumPy import failures. The physical
outcomes below are retained, with this additional limitation on interpreting
learner capability. See [the runtime defect and repair](public_analysis_runtime_20261021.md).

Updated 2026-10-21T09:11:35.932902+00:00. Study phase: **complete**.

The scene-update defect is repaired and tested. Luna achieved 4/5 practice successes. On the four paired transfer states, frozen memory achieved 3/4 native successes versus 1/4 without the new memory. This is a small same-asset pilot, not a reliable generalization claim.

## Repair and validation

The service now observes rigid object motion after every grounded articulated motion, including approach and closure. An accepted RGB-D estimate updates moving geometry. An unconfirmed ordinary motion leaves the estimate unchanged and reports uncertainty. The existing hand-motion fallback remains restricted to contact arcs and is labeled as a prediction. Collision rules, bounded contact permissions, controller, physics and native success remain unchanged.

| Matched saved observations | Baseline | Repair |
| --- | --- | --- |
| State 44, 5° point continuation | Stationary-point collision | Solved |
| State 44, 10° point continuation | Stationary-point collision | Original collision cleared; search budget exhausted |
| State 2, 10° rigid continuation | Solved | Solved |
| State 2, 5° point continuation | Solved | Solved |

The three previously offending handle samples remain in the repaired moving cloud; they were not deleted. Both variants consumed identical saved sensor arrays. Native object geometry was excluded from their inputs.

Physical checks replayed the exact saved prefixes and executed one new continuation each. State 44 advanced 3.997° over 16 steps; state 2 advanced 7.934° over 30 steps, with robot–object contact throughout both. These are development checks, not fresh task-success trials. The focused suite passed **98 tests**, with **8 missing-fixture skips**. The Luna/high subscription and filesystem-isolation preflight passed.

[Matched results](../results/tool-repair-learning-20261021/matched-comparison.json) · [Tool freeze](../results/tool-repair-learning-20261021/tool-freeze.json) · [Validated patch](../results/tool-repair-learning-20261021/validated-changes.patch) · [Tests](../results/tool-repair-learning-20261021/tool-validation-tests.log)

## Learning protocol

Luna uses high reasoning through the existing ChatGPT subscription, with no separately billed model API. Practice uses state 1, one continuing context, five initial attempts with extensions up to twenty, and a three-hour session cap. Each episode has at most 600 control steps and 1,800 seconds. Native success stops motion immediately. Unsuccessful episodes continue to a genuine budget limit. The original practice prompt, earlier guide, successful-method hypothesis and read-only library are retained. This is guided practice, not learning from scratch. No repair-specific strategy or evaluator geometry is given to Luna.

Transfer states were predeclared as [41, 29, 45, 0], sampled before any outcomes. Each gets one episode with frozen new memory and one without it; condition order alternates across states. Both receive the same abstract example and tool documentation, with no old practice library or successful-method file. No evaluation-case notes are passed across cases. Optional DINO is unavailable in every condition because its authorized checkpoint is absent on this server.

The states are fresh relative to retained microwave protocols and documented prior practice/pilots. Unavailable historical runs cannot be exhaustively audited. These are four same-asset pairs, not a generalization claim about other objects.

[Registration](../results/tool-repair-learning-20261021/learning-registration.json) · [Study protocol](../results/tool-repair-learning-20261021/study-v1/protocol.json)

## Practice results

Completed 5 episodes; native successes: 4. Attempts started: 5.

| Attempt | Native success | Control steps | Termination |
| --- | --- | ---: | --- |
| 1 | False | 600 | control_budget_exhausted |
| 2 | True | 372 | simulator_success |
| 3 | True | 365 | simulator_success |
| 4 | True | 365 | simulator_success |
| 5 | True | 384 | simulator_success |

Frozen memory contains 4 agent-authored principles. SHA256: `65f9612d3f66380234d78f3e05c6342f784b3f137ac4df8032f7fb04ba0c13c2`.

[Usable frozen memory](../results/tool-repair-learning-20261021/study-v1/inputs/learned-guidelines.json) · [Practice synthesis](../results/tool-repair-learning-20261021/study-v1/practice/agent-workspace/practice-summary.json)

## Paired transfer results

Completed 8 of 8 predeclared episodes.

| State | Without new memory | With new memory |
| --- | --- | --- |
| 41 | Failure; 596; wall_budget_exhausted | Success; 353; simulator_success |
| 29 | Failure; 597; wall_budget_exhausted | Success; 423; simulator_success |
| 45 | Failure; 600; control_budget_exhausted | Failure; 600; control_budget_exhausted |
| 0 | Success; 505; simulator_success | Success; 389; simulator_success |

[Successful state 41 with memory — rollout video](../results/tool-repair-learning-20261021/study-v1/transfer/state041-with_learned_memory/attempt01/initial/rollout.mp4) · [Failed state 45 with memory — rollout video](../results/tool-repair-learning-20261021/study-v1/transfer/state045-with_learned_memory/attempt01/initial/rollout.mp4)


without_learned_memory: 1/4 planned successes (4/4 completed). Unstarted cases are not silently counted as successful or dropped.

with_learned_memory: 3/4 planned successes (4/4 completed). Unstarted cases are not silently counted as successful or dropped.

## What remains unresolved

The memory condition succeeded on states 41 and 29 where the controls failed; both conditions succeeded on state 0 and failed on state 45. On the jointly successful state 0, memory used 389 steps versus 505. This supports usefulness of the evaluated guidance in these cases, while leaving recovery unreliable.

Post-run native diagnostics found no large initial controller-tracking miss: the first executed path ended within 0.37 mm of its requested TCP position in all eight transfer episodes. Reaching that pose does not establish a useful grasp. The failed controls on states 29, 41, and 45 had their last recorded robot–microwave contacts at steps 315, 441, and 429, respectively, before unsuccessful subsequent recovery.

State 45 with memory peaked at 22.89° opening and ended near 21.05°, while its control peaked at 55.19° and also failed. Three memory-case updates predicted over 1° of hand-driven scene motion while the native door changed less than 0.05°. Late robot–microwave contacts did not establish effective opening. These are evaluator diagnostics, not extra actor feedback or new success rules. They do not isolate every mechanical or planning cause of the remaining failure.

The successful state 41 memory episode also used hand-motion prediction when visual fitting was unavailable, with real door progress. The evidence supports keeping predictions explicitly uncertain; it does not justify blanket removal of that fallback. The transferred memory remains exactly the pre-evaluation version.

[Private episode diagnostics](../results/tool-repair-learning-20261021/episode-diagnostics-private.json)

The smallest justified next experiment is recovery-focused development practice on the now-known failed state 45, using normal observations and the same frozen numerical tools. Inspect its retained failure inputs before deciding whether another generic tool repair is warranted. Evaluate any revised agent-authored memory on newly predeclared states, and repeat samples before making reliability or broader generalization claims.

## Audit and limits

Frozen numerical source/configuration verification currently passes: **True**. After practice, the launcher report failed because it used system Python without NumPy. Its reporting interpreter was changed to the validated numerical environment; the report was regenerated from the completed run and transfer resumed. The original failure, original source hashes and exact one-line reporting-only exception remain recorded. No practice episode was repeated or discarded. Per-case audits check isolation, exact issued-reference logging, no post-success control, no execution veto, no forbidden evaluation fields in feedback and unchanged API ledgers. Infrastructure failures remain visible.

[Running audit](../results/tool-repair-learning-20261021/running-audit.json) · [Transfer records](../results/tool-repair-learning-20261021/study-v1/transfer-summary.json)

[Reporting repair and compatibility record](../results/tool-repair-learning-20261021/reporting-fix/compatibility.json)

A solved plan is not task success. Better results after the tool repair alone would not demonstrate learning; the relevant memory comparison is between the two conditions under the same frozen repaired tools. Four pairs and one sample per condition provide development evidence with substantial uncertainty.

The restored server runtime retains the earlier reported Torch/CUDA portability differences. Original raw RGB-D is absent; saved-frame tests use fresh diagnostic captures, while physical prefixes closely match archived control and native trajectories. The baseline and validated source archives, inputs, commands, videos, point clouds, and required sensor observations are retained. Automatic episode and study retention remove verified redundant data.

Paired initial sensor checks: **3/4 available pairs identical** (4 pairs planned). All planned cases audited: **True**. Independent case threads: **True**.

[Paired audit](../results/tool-repair-learning-20261021/paired-audit.json)

State 41, frontview_rgb: 6 pixels/values differ, maximum intensity/value difference 1.0. All depth/world-point arrays identical: True. These observations are not claimed to be bit-for-bit identical.

## Memory handoff

The experimental bundle contains the exact evaluated Luna-authored memory, its hash, tool version, paired results and usage instructions. It is available for explicit use and has not replaced a default policy.

[Memory bundle](../data/agent_memory/luna-scene-motion-v1/README.md) · [Guidelines](../data/agent_memory/luna-scene-motion-v1/guidelines.json) · [Manifest](../data/agent_memory/luna-scene-motion-v1/manifest.json)

## Retention

This repair/study directory retains approximately **6.65 GB**. Recorded episode cleanup and public-sensor pruning removed **18.25 GB**, in addition to lossless compression. Initial/final observations and a bounded set of early planning-failure inputs remain; the evaluator retained 48 extra failure frames. Original handoff evidence and the measured historical source remain untouched.

[Storage record](../results/tool-repair-learning-20261021/storage-summary.json)
