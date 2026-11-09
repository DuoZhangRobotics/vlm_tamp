# Audit of Luna's action feedback

After a completed close, the useful result is **Fingers settled**, followed by a physical action.
For picking, lift or continue the transfer and compare object motion with hand motion. If the
object stays behind or slips, choose another grasp. No advance holding-certification step is needed.

The previous interface repeated generic uncertainty labels in the gripper measurement, mechanical
completion, grasp summary, object model, tool help and actor prompt. Those messages could make a
normal intermediate state look like a failed grasp. An additional stale prompt still described
closure as a fixed command count even after execution had been changed to measured settling.

## Changes

| Finding | Correction |
| --- | --- |
| `retention: not_tested` / `object_retention: unverified` repeated after closing | Removed from new closing results and policy-facing archived replies. Report measured closure and the next action. |
| Default `grasp_state_unknown` alongside a valid pad-gap measurement | Report `gap_measured` and the measured gap. Actual gap-collapse and handle-width mismatches remain explicit. |
| Constant false attachment, semantic-identity, contact and collision-certificate flags | Omitted from Luna's policy input. The attachment model, coverage settings, measurements and raw operator records remain available. |
| Grasp command-count wording in the actor prompt and compact manual | Replaced with measured finger completion, followed by physical motion and observation. |
| Some top-level stop reasons omitted by compaction | Preserve `stop_reason` and status; show actual approach position/orientation errors and tolerances on failure. |
| Transfer failure lost its budget explanation | Preserve available controls, hold/release cost, query counts, elapsed budget and finite-search outcome. |
| Repeated object-model copies and internal fit-index arrays | Keep IDs, reference coordinates, shape, timestamps, currentness and fit summaries. Full records retain detailed transforms and inlier indices. |
| Initial shape bounds appeared beside a newer pose without their own timestamp | Label shape and pose measurement with their measured control step. |
| Learner notes stamped `learner_note_unverified` in each input | Display `learner_note`, preserving the authored text. Internal historical record format remains compatible. |
| `max_candidates` and `probe_degrees` suggested ineffective exploration | Help now states that selection is limited to the configured pose pool; seed/region change proposals, while the legacy angular field does not rotate GraspGen poses. |
| Nonterminal `finish` looked like termination | Help explicitly describes the existing status-only behavior while `episode_done=false`. The evaluation stopping policy was not changed. |

The presentation is shared by structured-actor feedback, retained memory, optional compact transition
records, visual-tool replies and the standalone Python brief client. The installed brief client
receives the same standard-library helper; it does not require access to the repository package.

## What remains explicit

Real failures are not softened: an unfinished close, missed pose, failed tracking fit, measured zero
object motion, lost engagement, exhausted budget, runtime error and false native task result remain
visible. Actual model assumptions and concrete capability limits remain in the tool contract, for
example robot-only collision modeling for carried-object placement. Measured fit-quality decisions
and provisional joint-estimate admission are unchanged.

Legacy fixed-duration replies are presented as **closing commands sent**, not retroactively relabeled
as mechanically settled. Raw replies are unchanged. No confidence, holding or success claim is
invented to replace a removed warning.

## Validation

Report: `results/feedback-audit-20261109/watch.html`.

Five existing replies were replayed through the previous and revised presentation: measured closure,
a blocked approach, a placement budget rejection, object selection, and grasp-plus-pose measurement.
All raw replies and error strings remained unchanged. Examples:

- A grasp-plus-pose reply decreased from 18,142 to 10,798 serialized bytes and removed ten default
  uncertainty labels, while retaining the current object reference and measured fit summary.
- A placement rejection increased from 267 to 562 bytes because it now exposes the useful facts:
  14 controls available, six hold controls, twenty release controls, and the search work performed.
- A blocked approach keeps its measured goal error and actual stopping reason. It receives no lift
  instruction; the lift/transfer instruction follows measured closure completion.

208 focused tests passed. Two tests were skipped because their saved historical pilot fixtures are
absent. Ruff and `git diff --check` passed. No robot controls or model calls were issued by this audit;
no new policy success rate is claimed. The numerical planners, physical controller, camera set,
collision rules, closure criterion and episode limits are unchanged.
