# Completion evidence and abstract memory

The active profile is `configs/subscription_memory.json`, version two. The original
profile is preserved as `configs/subscription_memory_v1.json`, and the original
episode remains under `data/agent_memory/luna-microwave-v1`. The active library is
`data/agent_memory/luna-microwave-v2/library`.

Memory is retrieved context, not a model-weight update. A small Python client
matches the robot and ranks task descriptions by word overlap. A fresh session
reads the returned procedure and evidence. There is currently one learned episode.
Its JSON includes the task and robot, observed failure, successful correction,
applicability, conditional procedure steps, event references, cautions, limitations
and attribution. The only additional evidence is a categorical event trace.
No historical images, sensor arrays, image regions, world poses, joint trajectories,
tool argument values, numeric schedules or executable analysis code are exposed.
Event identifiers are provenance, not scene coordinates. The library is read-only
and is not updated from evaluation outcomes between runs.

An isolated Luna/high session abstracted its existing prior procedure and common
guide. A second isolated session checked editorial fidelity, including action
ordering and attribution. Neither could read the new evaluation outcomes or
implementation. Their original inputs, outputs, prompts and isolation checks are
preserved under `results/subscription-memory-abstraction-20261019` and
`results/subscription-memory-abstraction-reviewed-20261019`. The common human
example now contains the same rigid-motion/pivot-fitting method as prose, without
historical images or numerical arrays. Both conditions receive identical common
context.

The optional completion protocol is implemented in
`src/vlm_tamp/observation_completion.py` and `src/vlm_tamp/surface_association.py`.
It has no object names, benchmark target
angles, simulator articulation state, motion planner or controller dependency.
Before moving, the agent declares a quantitative observed goal and its rationale.
Supported quantities are full rigid rotation, displacement of a selected material
reference point, and separation between planar surface normals. The agent chooses
material patches in current-session RGB-D.
A robust plane fit reports support, residuals, a visual overlay and resampling
stability. Mutual image-feature matches or agent-supplied current pixel pairs
must support a robust three-dimensional rigid transform. Spatial support,
residuals and consistency with the selected planes are checked. Full rigid
rotation observes motion within a plane, which its normal cannot. Supported
successive associations can link an initial patch to a later observation, with
accumulated fitted rotation margins. Plane-normal comparison is unsigned and
limited to a right angle; rigid rotation magnitude is limited to a half-turn.
An arbitrary difference between selected centroids remains diagnostic only.

A completion check compares fit bounds to the agent's own declared range and
requires current observations. Those bounds are stability estimates, not calibrated
confidence or proof of material identity. The agent must inspect selections and
views. A completed claim needs a fresh supported check; unsupported claims receive
feedback while the episode stays open. The agent may always explicitly stop as
incomplete or uncertain. No check intercepts an issued trajectory. A poor agent
goal or selection can still produce a supported check and fail the benchmark;
native success is independently evaluated and remains the primary metric.

The retest uses the exact twelve-case manifest from the previous comparison:
three previously difficult scenes and three repeated transfer scenes, two memory
conditions each. Luna/high uses ChatGPT subscription, one physical attempt, seed
zero, the same settling and controller settings, and six hundred recorded control
steps. Optional DINO and numerical tools remain unchanged. The source audit checks
the protected grasp, IK, planning, scene-update and execution methods against the
previous source snapshot. All cases remain in the denominator. There are no paid
API fallbacks, automatic retries, model switches or task-specific recovery fixes.

Both groups receive the new shared completion protocol. Only the memory group
receives the abstract episode. Comparing this retest with the older study combines
completion and context changes; it does not isolate their individual effects.
These are reused scenes and a single asset, not new held-out generalization tests.

Run in a new output folder with:

```sh
python3 scripts/run_subscription_completion_study.py --study results/NEW-COMPLETION-RETEST --freeze
```

The initial independent-plane design was stopped after two diagnostic episodes
exposed an invalid cross-surface comparison and an unobservable in-plane rotation.
Their outcomes are preserved under `results/subscription-completion-memory-20261019`;
ten planned episodes there were not launched. They are not the final comparison.
A regression replay rejects the recorded false completion using only retained
RGB-D, without reading native geometry. No diagnostic outcome or coordinates were
added to memory. The library itself is unchanged by the measurement revision.

The current frozen comparison is
`results/subscription-completion-memory-final-20261019`. A second diagnostic batch
under `results/subscription-completion-memory-v2-20261019` exposed a JSON reply
conversion error for matched pixel coordinates. Its first episode completed with
that error; the next launch was stopped after reset, before planned motion. Both
outcomes were retained and no issued trajectory was interrupted. Python-integer
conversion, a successful full-response test, and replay of the exact failing
request passed before the final twelve-case comparison was frozen. Across the
two diagnostic versions, four attempted episodes are retained separately from
the final comparison.

The check validates evidence for the goal selected by the agent. It cannot fix
an insufficient interpretation of the task instruction: an agent may choose an
opening target below the benchmark's private definition. It also may be unable
to certify a physically successful episode when texture or material depth is
ambiguous. The report therefore separates native success, agent claims, declared
targets, supported measurements and uncertainty. It does not equate a valid
measurement, a successful plan or a completed claim with benchmark success.
Its runner compacts and cleans each completed episode, preserving every completion
measurement's input frame, all RGB, commands, traces, videos and outcomes. Final
deduplication preserves byte-identical evidence paths. Active and original memory
stores are protected. The reusable remembered-agent entry point enables the new
profile by default; use the archived profile explicitly to reproduce the older
interface without completion checks.

## Completed retest

The final frozen version completed all twelve episodes. Guide-only reached the
native goal in two of six episodes, but none retained it at the final step.
Abstract memory reached and retained the goal in three of six episodes. By scene
group, the previously difficult cases scored zero of three for guide-only and
two of three for memory; repeated transfer scenes reached two of three and one
of three respectively, retaining zero and one of three. The older comparison
reached and retained eight of twelve goals; this one reached five and retained
three. This candidate did not improve physical execution reliability.

Only one of sixty-eight material comparisons had supported association, and none
of fifteen completion checks supported the declared conditions. No completed
claim was accepted. Thus frequent uncertainty, including on physically successful
episodes, is a major limitation; absence of false completed claims does not make
this a useful completion detector. Some agent-selected target ranges permitted
openings below the benchmark's private definition. Memory/context abstraction
and completion changes were bundled, so the score reduction cannot be attributed
to coordinate removal alone.

Recorded free joint motions after release undid substantial door progress in two
inspected cases. Their traces contain no new grounding, tracking application or
model-setting call between initial grounding and retreat. One retreat omitted the
gripper argument, whose default is to close. These are observed interface/model
freshness concerns, not a controlled proof of a single physical cause. No task
waypoint, contact exception or low-level control fix was added during the retest.

Thirty-three focused tests passed. Independent audits verified all paired initial
sensor arrays (also matching the earlier study), source isolation, unchanged
numerical methods, original and active memory, retained measurement inputs, native
outcomes, all videos, and sixty-eight fully issued trajectories over 4,792 control
steps with zero execution vetoes. The final study reclaimed about 8.07 GiB and
retains about 3.25 GiB of unique evidence; diagnostic cleanup reclaimed another
2.23 GiB. All final-study temporary workspaces and owned workers closed. There
were zero separately billed API calls. The active library itself is 18,480 bytes
in three JSON files. The detailed report includes native motion curves, final
views, agent targets/claims, failure records and the separately preserved aborted
diagnostic versions. Treat this completion protocol as experimental.
