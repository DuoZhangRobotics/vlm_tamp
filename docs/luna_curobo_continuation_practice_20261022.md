# Adaptive Luna practice with integrated cuRobo continuation

Status: complete. **All five adaptive attempts reached native success.**
The user authorized adaptive self-practice after the successful
[fixed-decision replay](curobo_v2_continuation_20261022.md). This study lets Luna
choose its own observations, actions, recovery attempts, and notes.

[Live observer report](../results/luna-curobo-continuation-practice-20261022/watch.html).
The report is separate from the learner's isolated workspace and supplies no
feedback to the learner.

## Outcomes and learner decisions

| Attempt | Learner-selected method | Native success | Controls | Attempt wall time |
| --- | --- | --- | ---: | ---: |
| 1 | Six fresh 15-degree point arcs after staged grasp | Yes | 305 | 262.270 s |
| 2 | Fresh-reset confirmation of the 15-degree method | Yes | 305 | 214.519 s |
| 3 | Three fresh 30-degree arcs as an efficiency candidate | Yes | 283 | 170.748 s |
| 4 | Fresh-reset confirmation of the 30-degree candidate | Yes | 283 | 163.981 s |
| 5 | 45-degree planning trial, then recovery with 30-degree arcs | Yes | 283 | 208.069 s |

Luna first established repeatability, then proposed and tested the larger
increment itself. The 30-degree method used 22 fewer controls (7.2%) and five
requested command batches instead of eight, including approach and closure.
Both baseline confirmations and both efficiency confirmations used newly
grounded geometry and newly computed plans. These descriptive comparisons
include inference, warm-up, and timing variation; they are not a controlled
estimate of the effect of changing only the angle.

There was one rejected robot-tool request, event 64 in attempt 5. The 45-degree
request reached the planning deadline while attempting the 42-degree target;
the last connected internal point was approximately **39 degrees**. No partial
motion executed, and the robot remained at 221 controls. Luna then returned to
the confirmed 30-degree method and completed the task. The learner summary's
`at_progress_degrees: 42` denotes the attempted target, not executed progress.

[Backend learning summary](../results/luna-curobo-continuation-practice-20261022/practice-v1/learning-summary.json),
[learner-authored summary](../results/luna-curobo-continuation-practice-20261022/practice-v1/agent-workspace/practice-summary.json),
[exact failure receipt](../results/luna-curobo-continuation-practice-20261022/practice-v1/attempt05/064-arc/result.json).

## Candidate memory and validation

The [candidate memory bundle](../data/agent_memory/luna-curobo-continuation-state45-v1/README.txt)
contains exact copies of Luna's final guidelines, findings, per-attempt
hypotheses, and summary. Its provenance file records source paths and hashes.
The default prior guide is unchanged. Numeric angles are same-scene tested
choices, not universal constants or a reason to replay saved coordinates.

The [independent audit](../results/luna-curobo-continuation-practice-20261022/final-audit.json)
confirms all five native outcomes and learner-reported control counts, all
1,159 issued joint references, correct existing bounded-controller behavior,
and zero controls after first success. All 509 registered implementation and
configuration files remain unchanged. Isolation passes, forbidden evaluation
fields are absent from learner feedback, and API ledgers remain unchanged.
No human task hint or separate reflection reviewer intervened.

Luna used a second turn in the same thread to place final memory files at the
required root paths. The substantive findings are retained under
`agent-workspace/practice/findings.txt`; the administrative follow-up did not
supply a task-solving lesson. One guideline limits sentence still says
"one attempt" although its evidence cites attempts 1 and 2. That conservative
wording inconsistency and the attempted-versus-connected angle distinction are
documented in provenance; the learner's original files are not rewritten.

Whole launcher time was 1,205.118 s (about 20.1 minutes). Actor time was
1,193.703 s: 870.866 s in robot service calls and 322.837 s outside them.
The latter includes inference, shell analysis, transport, and scheduling.
There were 82 measured compact presentations: 165,520 display bytes versus
354,162 full-reply bytes, a 53.3% reduction for those presentations. This does
not measure total model input, image tokens, or active context length. Raw CLI
usage snapshots are retained without treating cumulative input as simultaneous
context size.

## Registered conditions

- One continuing `gpt-6-luna` session, high reasoning, ChatGPT authentication.
- Up to five clean-reset attempts on the same LIBERO-90 microwave development
  scene, state 45, seed 0.
- 600 controls and 1,800 seconds per attempt; 5,400 seconds for the whole
  session. The total cap can prevent all five attempts if earlier attempts run
  long; report the actual count.
- Compact public tools, native success termination, current RGB-D and robot
  observations, and the integrated local-first cuRobo planner with bounded
  global fallback and 3 mm positional tolerance.
- Same prior Luna-authored guide as the previous compact and initial-v2
  studies: `data/agent_memory/luna-scene-motion-v1/guidelines.json`, SHA256
  `65f9612d3f66380234d78f3e05c6342f784b3f137ac4df8032f7fb04ba0c13c2`.
- No separate Reflection reviewer. The recent successful replay, its trajectory,
  and a new success recipe are not supplied to Luna. Source is frozen for the
  session; the existing persistence loop sends budget-only continuation
  reminders when needed.
- Learner-authored notes and guidelines are preserved as candidate memory with
  provenance. They do not automatically replace the default guide.

[Registration](../results/luna-curobo-continuation-practice-20261022/practice-v1-registration.json),
[exact prompt](../results/luna-curobo-continuation-practice-20261022/practice-v1/agent-prompt.txt),
[isolation check](../results/luna-curobo-continuation-practice-20261022/practice-v1/isolation.json),
[study context](../results/luna-curobo-continuation-practice-20261022/study-context.json).

## How to inspect the process

`agent-events.jsonl` records Luna's visible messages and local tool use.
`tool-events.jsonl` records numerical robot requests and complete replies.
Per-arc files retain the exact IK inputs, candidate outputs, constraint and
collision rejections, optimizer lifetime, and native connections. Large JSON
files may be compressed after episode cleanup; the observer reads both JSON
and JSON.gz. Videos appear when attempts close.

Native simulator evaluation stays outside Luna's inputs. Success, control
counts, and timing in the final study summary are checked against backend
receipts, not accepted solely from learner-written findings. Non-service actor
time includes inference, local analysis, transport, and scheduling; it is not
a direct measurement of pure reasoning time.

This is same-scene development practice. Repetition here does not demonstrate
transfer, generalization, or a controlled causal effect of a single numerical
change. Earlier initial-v2 Luna practice was 0/2 before several fixes; that
comparison is historical rather than randomized.
