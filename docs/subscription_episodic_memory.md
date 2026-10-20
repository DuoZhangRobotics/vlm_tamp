# Concrete episodic memory experiment

This experiment separates deterministic physical replay, first-attempt agent reuse,
and transfer to changed observations. It does not remove the inherited geometry,
grasp or contact priors from the numerical tools.

The learning source is the two-attempt public session `nominal-state037-original`
from the 2026-10-19 strategy-transfer study. Attempt 1 failed; attempt 2 ended with
the native goal satisfied. That private selection criterion is recorded by the
experimenter but the memory author receives only public observations, tool events,
and the previous agent's findings and code. A fresh Luna/high session authors the
procedure, and a separate fresh session audits factual attribution. The procedure
is retained verbatim after that audit. No human-written task correction is added.

The persistent library contains the full retained public episode, including its
failure, corrected decisions, RGB-D observations, plans, measured geometry and
analysis code. The generic `memory.recall(instruction, robot)` interface retrieves
entries by lexical task overlap and robot match. Its implementation has no motion
policy. Libraries are copied into each public workspace, checked byte-for-byte, and mounted read-only. Actual retrieval and write denial are tested before simulator reset; project source, private
evaluation and unrelated history remain blocked. The test uses an identical empty
library interface as control. This is an initial retrieval implementation, not a
learned semantic retrieval model.

Five exact-command replays test repeatability of the successful physical solution.
Five fresh sessions per condition then test one-attempt reuse on the same state.
Three reserved states test transfer with nominal, translated and rotated scenes.
Each evaluation permits one reset, 600 control steps including the initial open
hold, and no physical retry. Existing grasp, IK, planning and execution behavior
is unchanged. The agent may replan between fully executed trajectories.

The primary metric is native task success at the end of the single attempt;
native success at any step is reported separately. All planned sessions remain in
the denominator. Capacity/quota errors are recorded and stop the batch; there is
no automatic API, model or physical retry. Code, library, prompts and runtime
artifacts are hashed before evaluation. The library remains frozen during this
comparison so earlier evaluation outcomes cannot contaminate later cases. Each
new experience is retained for a later explicitly versioned learning cycle.

Run from the repository using the existing simulator/back-end environments:

```sh
.runtime/envs/rekep-libero/bin/python scripts/replay_subscription_episode.py --source results/subscription-strategy-transfer-20261019/runs/nominal-state037-original --attempt 2 --output results/NEW-STUDY/exact-replay --repetitions 5
python3 scripts/build_subscription_memory.py --source results/subscription-strategy-transfer-20261019/runs/nominal-state037-original --output results/NEW-STUDY/memory-author
python3 scripts/build_subscription_memory.py --source results/subscription-strategy-transfer-20261019/runs/nominal-state037-original --output results/NEW-STUDY/memory-author-audited --audit-existing results/NEW-STUDY/memory-author/library/episode-0001/procedure.json
python3 scripts/reserve_subscription_states.py --output results/NEW-STUDY/state-reservation.json
python3 scripts/run_subscription_memory_study.py --study results/NEW-STUDY --freeze
```

Create a new results directory first. Reproduction of the exact same transfer
states requires the saved state-reservation file; rerunning reservation selects
unused states from the then-current historical registry. The current study is
`results/subscription-episodic-memory-v2-20261019`. The initial version is retained separately because its shared-library location was inaccessible under the source-isolation profile; its sole session is not part of the corrected matched comparison.

Per-run cleanup retains videos, actions, trajectories, public evidence needed for
analysis, outcomes, authored code and provenance. Remaining private observation
containers are compressed losslessly with per-member SHA-256 verification.

## Reuse the persisted experience

The dedicated entry point enables the registered public memory by default:

```sh
python3 scripts/run_subscription_remembered_agent.py --state 22 --output results/NEW-REMEMBERED-RUN
```

Add `--dry-run` to verify registration and inspect the command without starting
the model or simulator. The entry point checks library, prompt and guide hashes,
permits only a new results directory, keeps Luna/high with ChatGPT authentication,
and performs per-run cleanup. It currently uses the microwave-opening backend.
The original experiment runners retain their defaults; this is an explicit
experimental entry point, not promotion over the accepted multi-task reference.

The library is persistent data, not a model-weight update. New runs retain their
experiences for a later reviewed learning cycle. They do not automatically replace
the saved procedure, and the controlled evaluation never incorporates its own
held-out outcomes into the library. This study reports native goal attainment
alongside the separately predeclared stricter end-state retention measurement.

Active memory, prompt and common guide are registered under
`data/agent_memory/luna-microwave-v1`, outside transient experiment outputs.
Their initial copies share immutable file contents with the study through hard
links. Keep the configuration's `protected_inputs` during cleanup, including the
existing approved example source used by the legacy preparation interface. The
public example is also backed up inside the persistent episode evidence.
