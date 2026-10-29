# Reflexion and the Luna learning pipeline

This is the original pre-implementation design note. The subsequent experiment,
two setup repairs and results are complete; see
[the current study report and handoff](reflexion_checkpoint_study_20261021.md).

This is an evaluator/design note, not learner memory or an implemented protocol.
The paper in the user's photograph is *Reflexion: Language Agents with Verbal
Reinforcement Learning*, arXiv:2303.11366v4, 10 October 2023.

The reviewed paper motivates feedback → reflection → memory → another attempt
with fixed model weights. The authors' ALFWorld implementation stores memory per
environment and generates a plan for retrying that environment. Its results are
not direct evidence of physical-robot recovery or transfer to unseen scenes.
[Paper](https://arxiv.org/pdf/2303.11366) ·
[Reflection generation](https://github.com/noahshinn/reflexion/blob/main/alfworld_runs/generate_reflections.py) ·
[Environment memory](https://github.com/noahshinn/reflexion/blob/main/alfworld_runs/env_history.py).

## What we already have

`prompts/recovery_learning_practice.txt` already requests hypotheses before
attempts and evidence, limitations and conditional lessons afterward. Practice
retains context and notes; `recovery_memory.py` exports an abstract memory.
`episode_persistence.py` supplies continuation and budget rules. The earlier
practice/transfer study therefore already belongs to this general approach.

The latest verification experiment was a different intervention: a fixed
human-authored paragraph added to the actor prompt, with memory held fixed.
It did not test whether newly generated reflections improve the next attempt.
Its limited benefit cannot rule out that learning mechanism.

Our identified gaps remain concrete: practice stops after repeatable completion
on one starting state; the memory exporter checks representation rather than
the truth of causal explanations; and diagnosis/recovery from the target
inconsistency is not necessarily exercised in fresh successful episodes.
[Practice prompt](../prompts/recovery_learning_practice.txt) ·
[Memory exporter](../src/vlm_tamp/recovery_memory.py) ·
[Verification results](verification_prompt_study_20261021.md).

## Proposed adaptation

Use the same Luna model in separate acting and post-attempt review calls.
Keep the native task predicate as the authoritative outcome. A review call
receives only the completed public action/observation record, the allowed
outcome and prior practice notes. It produces a compact candidate lesson:

- An observable discrepancy and a reference to its supporting public evidence.
- A possible causal explanation, with alternatives or an explicit unknown.
- One change to test on the next attempt.
- The observation that would support or contradict that change.
- Scope and remaining uncertainty.

Treat that output as a hypothesis. Check numerical citations and provenance
against public records; do not silently accept a fluent explanation as correct.
The meaningful validation is whether the proposed change improves a subsequent
decision and physical interaction. A reviewer using hidden native poses must
not send those diagnoses, corrected coordinates or contact labels to the actor.

Keep two separate records:

1. Local episode records and a small set of recent candidate lessons for practice.
   These preserve enough detail to revisit the actual failure. The existing
   current-episode observations and per-attempt files already provide much of
   this evidence; a new database is not required for this experiment.
2. Portable memory containing lessons whose applicability and limits have been
   tested on different development conditions. Preserve failures as well as
   successes. Do not export historical poses, pixel selections, trajectories or
   a remembered solution for a specific state as a transferable procedure.

This separation is our proposed adaptation for transfer. It should not be
presented as a capability already demonstrated by the paper or by Luna.

## Small controlled experiment

Start from saved failure prefixes so the target difficulty is actually present.
Use normal observations, the same frozen numerical tools and declared remaining
control budget in every condition. Retrying a practice case must be explicit;
do not reset away from an active episode with remaining budget.

Compare three conditions across a small preregistered practice block:

- Retry with the common starting guidance and no new experience memory.
- Retry with the preceding public episode record or a factual summary.
- Retry with that evidence plus Luna's own post-attempt reflection.

Control attempt/action budgets, bound review/model compute, and report total
cost so an extra model pass is not mistaken for a memory-content advantage.
Track whether reflection identifies a supported discrepancy, whether the next
attempt actually changes the relevant decision, and whether native physical
outcomes improve. Merely producing more inspections or a better explanation
does not establish recovery.

If this works, freeze the resulting agent-authored portable memory and evaluate
new, predeclared states separately. Improvement on the practiced failure and
transfer to a new scene answer different questions. Keep a stronger model as a
private diagnostic reference if needed; do not copy its case solution into
Luna's learning condition.

This proposal does not repair infeasible or unnecessarily restrictive numerical
tools. The separate budget-reservation lead remains an implementation question
to validate on matched inputs, independently of any reflection comparison.

No code, default prompts, frozen learner memory or experiment outcomes were
changed by this literature review.
