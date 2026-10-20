# Shared tools for scripted orchestration and Luna

The earlier term **numerical pipeline** means a **scripted baseline**: Python
code chooses a task-family adapter and its sequence of grounding, grasp,
planning and execution calls. Previously generated VLM choices are reused only
when their observed inputs match. The 35/35 regression therefore measures this
scripted policy plus its tools, not 35 fresh autonomous Luna episodes.
Its microwave cases still use repeated short planned motions. The prepared
execution change recovered bowl/drawer failures; it did not convert all
thirty-five episodes to single long trajectories.

Luna is a different caller of numerical tools. It chooses regions from current
images, requests plans, executes them, interprets feedback and writes findings.
Its earlier frozen transfer scored 1/4 on four previously evaluated states of
the same microwave task and asset.

## Why the tools diverged

The numerical runner retained a later contact implementation, while the
subscription agent imported an older experiment runner. Separate copies allowed
handle fitting, contact support, IK search breadth, collision geometry and
budgets to diverge. Keeping old result evidence did not require this divergence.
The comparison was confounded and should not have been interpreted as solely
an agent-policy gap.

## Consolidation

`src/vlm_tamp/hinged_contact_tools.py` is now imported by both the active
scripted microwave runner and the agent service. It shares:

- Observed handle fitting/refinement and finite finger-pad support.
- Grasp pose generation, isolated seven-joint cuRobo IK, eight-solution search,
  and approach/contact admission.
- Calibrated convex arm and gripper geometry against observed point clouds.
- The existing bounded local contact permissions and shallow existing-contact
  tolerance from the scripted baseline.
- One MC-VAMP partial-pose arc implementation and common tool search defaults.
- Point, rigid-pose and finger-axis contact constraint options.

The general tool correction introduces no task names, state indices, stored
coordinates, new preferred poses or recovery recipes. Existing upright-panel
and vertical-bar priors remain explicit limitations. The inherited 7 cm
approach standoff and contact thresholds were not newly tuned.

Caller decisions remain separate: selected image regions, requested horizon,
gripper closure duration, trajectory speed/hold, model updates, and recovery.
Both use the same low-level scheduler and execution primitives. A shared tool
contract does not imply identical inputs or identical orchestration.
The scripted regression retains input-validated cached seed-1 grasp proposals;
Luna requests fresh proposals with its selected seed (default 0). Both use the
same hybrid grasp tool, including inherited analytic observed-handle candidates.

## Validation and new comparison

The identical-input audit passes the same recorded RGB-D and region choices
through both callers, comparing clouds, masks, anchors, fitted models and
support diagnostics. It also checks that both callers reference the same
geometry, IK, selector, collision and planner function/class objects.
Those historical inputs are not supplied to the acting agent.

The new experiment is `results/shared-tools-transfer-20261020`:

1. Scripted microwave opening on states 1, 11, 21 and 31 using the consolidated
   tools and input-validated cached predictor outputs.
2. Luna on exactly those states, one physical attempt each, using the exact
   registered task/strategy prompt, guide and successful-method bytes from the
   earlier transfer. Tool documentation and feedback reflect the current tools,
   including the read-only scene diagnostics added after the earlier study.
3. No findings or memory updates pass between cases. No new practice occurs.
4. Each case has 600 control steps, seed 0, native-success termination, no
   extra retreat, and no execution-time collision/constraint/progress veto.
5. Scratch proposal data and redundant observations are cleaned after each run;
   initial/final sensor data, commands, plans, outcomes and videos are retained.

The new four-attempt agent comparison is separate from the completed
twenty-attempt practice study. The original results and source snapshots remain
unchanged. This tests the joint effect of tool consistency; it does not isolate
each geometry or search change, eliminate model nondeterminism, or establish
generalization to new assets.

The human-readable report maps every old folder to its question and result:
`results/shared-tools-transfer-report-20261020/index.html`.

## Measured results

The shared-tool scripted baseline succeeded **4/4**, with exactly the previous
selected grasps, joint solutions, approach/closure references and completion
steps: 507, 521, 497 and 492.

Luna succeeded **1/4**: state 31 at 335 steps. States 1, 11 and 21 ended at
338, 492 and 394 steps, with evaluation-only door openings of approximately
42.7°, 61.6° and 36.6°. The successful state ended at about 74.5°, when the
native predicate became true. No commands followed success.

State 1's requested 85° motion issued every reference, but measured hand
progress was about 53.8° and door progress about 42.7°. State 21 also had
large joint-reference errors during its long motion. On state 11, initial
short arcs moved the door, but a later roughly 10° hand motion moved the door
approximately zero degrees. Scene transport from hand motion then becomes an
incorrect estimate of the door. Follow-up planning and re-grounding were not
reliably able to recover. This establishes planning/execution and observation
gaps, not a proof that every rejected path was physically safe.

The original transfer also scored 1/4, succeeding on state 1 instead.
Tool consistency alone did not improve this measured success rate. Initial
grounding and caller choices differ, so this is not a controlled causal
comparison of individual tool changes or model intelligence. No new practice
or memory update was performed during the rerun.

State 31 exposed a reporting defect: an already-rejected request contained a
dictionary where the operation name should be a string. Postprocessing crashed
while classifying that request; the original runner counted the absent report
as failure. Type-safe log handling and native-outcome collection were fixed
after all physical runs ended. The native action log and evaluation record
agree on success. `verified-summary.json` preserves that result alongside the
unchanged original aggregate and error log; no physical episode was rerun.

## Reproduction

Build the canonical native adapter with
`scripts/build_mcvamp_contact_numerical.py`, then run
`scripts/run_consistent_tool_comparison.py --output results/<new-name>` in
the existing `rekep-libero` environment, with OpenBLAS, OpenMP and MKL each
limited to one thread. This launches four scripted episodes and four
subscription-backed Luna episodes; it requires explicit authorization for a
new physical-attempt budget. The default reference-transfer path must still
contain the preserved memory and profile hashes.

Generate the report with `scripts/report_shared_tool_comparison.py`; add
`--export-videos` after the comparison completes. The experiment protocol pins
source hashes, native build, model request, memory and thread settings. Full
prompts, tool calls, native outcomes and cleanup records remain in each run.
