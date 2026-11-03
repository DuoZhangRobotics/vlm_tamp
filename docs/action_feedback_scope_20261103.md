# Approach and stroke feedback were conflated

The closing failures need task-goal grounding as well as contact localization. The Molmo2
integration did not supply a closed-configuration reference. Reviewing the actual replies also
identified a concrete reporting defect in combined contact actions.

## Confirmed defect

`contact_skill_service.run_skill` copied each `service.issue` visual comparison into the same
top-level `visual_comparison` field. The final stroke overwrote the approach comparison. A caller
receiving a combined push result therefore saw a default image pair that omitted earlier motion.
The per-stage comparison was not retained in the returned stage record.

The recorded native angles below are evaluator-only evidence, never actor/planner inputs:

| Run / event | Approach | Requested stroke | Entire call |
| --- | --- | --- | --- |
| Close 49 / 7 | Controls 60–129: 107.26° → 117.43° open | 129–152: 117.43° → 120.29° | Opened 13.02° |
| Close 17 / 18 | Controls 202–209: 71.82° → 64.91° open | 209–216: 64.91° → 66.41° | Closed 5.41° |

In state 49, the default pair omitted 10.16° of opening during approach. Luna did separately call
`inspect_video(60,152)`, so missing default coverage is not the sole explanation for its mistaken
motion interpretation. Its findings described no clear angle change, and RGB-D feedback had zero
accepted matches on the first stroke. There was no measured task-progress signal to resolve this.

State 17's event 18 was a 5° CW request. Its approach closed the door by 6.91°, while the CW stroke
opened it by 1.50°. Treating the net change as the effect of the named CW stroke would learn the
wrong association. This is a causal-attribution risk supported by the phase measurements, not a
claim to know the model's private reasoning.

## Implemented correction

- Each executed stage retains its own `visual_comparison`.
- The top-level comparison covers the first executed stage's before observation through the last
  stage's after observation and explicitly identifies that scope.
- `phase_visual_comparisons` lists phase names, observation IDs, control intervals and image paths.
  These remain visible in compact replies, including the early approach.
- The feedback explains that the requested direction applies to the manipulation stroke. A
  whole-call change includes approach/repositioning and cannot alone identify the stroke's effect.
- Single-stage actions reuse their existing image pair. Multi-stage actions generate one additional
  comparison from saved RGB files. There is no new capture, simulation, motion gate or semantic label.

`action_comparisons` in `visual_inspection.py` supplies this presentation-only summary. The contact
skill calls it when saving the result, and `compact_robot_interface.py` preserves the scopes.
The controller, IK, planner, allowed contacts and motion budgets are unchanged.

## Validation and limits

85 targeted tests passed with the repository's native VAMP module path enabled; Ruff and diff
checks passed. Tests cover the real combined skill flow, native-success termination, reuse,
release/coasting, whole-call boundaries, preserved phase boundaries, compact replies, unchanged
source observations and single-stage image reuse. The first test invocation omitted the native
VAMP module path; it was corrected rather than changing the planner or skipping those checks.

The saved-recording audit is `results/action-feedback-audit-20261103/index.html`, with source
hashes and exact observations in `audit.json`. It includes the original execution videos and
times: state 49's interval starts at 3.00 seconds; state 17's at 10.10 seconds. Corrected pairs
are generated in a separate diagnostic directory; original actor workspaces remain unchanged.
No new physical episode was run, and this fix is not claimed to solve closing.

The remaining next experiment should ground the goal relationship explicitly: have Luna/Molmo2
select the moving free-edge landmark and the matching fixed reference where it should meet when
closed, then measure their relationship using calibrated observations and tracked correspondences.
Validate the references and closer/farther trend on recorded closing/opening intervals before
adding it to a live episode. Numerical tools cannot supply the semantic closed reference merely
from a face normal or hinge-axis sign. This proposed experiment is not implemented here.
