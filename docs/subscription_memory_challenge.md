# Harder episodic-memory transfer test

The preceding same-scene and small-transfer comparison passed in both conditions.
This follow-up tests the same frozen agent and public memory on harder scenes.
It changes evaluation cases, not grasp, IK, collision, planning or control policy.

The known-challenge stratum revisits nominal state 47, negative yaw plus translation
on state 37, and displaced plate clutter on state 47. These cases were selected
because the earlier guide comparison had failures. They are diagnostic challenge
cases, not unseen states or an unbiased population sample. That older experiment
allowed two attempts and used a different prompt; its scores are selection evidence,
not the paired baseline in this study.

The separate reserved-transfer stratum uses states 6, 17 and 25 with negative yaw
plus translation, displaced plate clutter, and a changed frontal camera respectively.
No earlier use was found in retained experiment metadata. Deleted or unrecorded
historical uses cannot be ruled out. Each scenario gets a fresh guide-only agent
and a fresh agent with the registered episodic library. Condition order alternates.

All twelve sessions use Luna/high through ChatGPT, the same initial prompt, prior
fixed guide, approved human example, optional DINO and numerical tools. Each gets
one physical attempt, seed 0, five settling actions, 600 recorded control steps
including the initial open hold, and a 1,800-second agent budget. All issued paths
execute fully. Failures before motion remain in the denominator; quota/capacity
stops the batch without automatic model/API fallback or retries.

Task success is the actual native predicate becoming true. End-state retention is
reported separately, including any reached-but-lost goals. Planning failures are
intermediate diagnostics, not automatic episode failures. The library is frozen
and receives no evaluation updates or human task hints. Simulator identities,
true articulation and evaluation data remain private. Existing tool priors remain.

Per-run cleanup preserves videos, actual commands, tool traces, input evidence,
agent code and outcomes while removing transient proposals and redundant snapshots.
Retained private observations are compressed losslessly. The active memory under
`data/agent_memory/luna-microwave-v1` and registered protected inputs are preserved.

To repeat exactly the same reserved cases, create a new output directory and copy
the reservation. Do not overwrite the existing study:

```sh
mkdir results/subscription-memory-challenge-repeat
cp results/subscription-memory-challenge-20261019/state-reservation.json results/subscription-memory-challenge-repeat/state-reservation.json
python3 scripts/run_subscription_memory_challenge.py --study results/subscription-memory-challenge-repeat --freeze
```

This launches twelve new subscription sessions and physical episodes. Keeping
the old reservation makes this a repeated-case experiment, not a fresh held-out
evaluation. Selecting genuinely new states requires a new audited reservation.

The current study's manifest and protocol are the authoritative settings. Resuming
with the same directory skips completed cases; interrupted cases are preserved and
require diagnosis rather than being overwritten. Source, prompt, memory, guide and
runtime artifact hashes are checked before each launch. The independent audit and
report tools are stored inside the study's reproduction-tools directory.

## Completed result

All twelve sessions completed. On the three previously difficult cases, guide-only
passed 1/3 and guide plus memory passed 3/3. On the three reserved transfer scenes,
both passed 2/3. Goal attainment and end-state retention agreed in every episode.
There is no demonstrated reserved-scene success-rate advantage in this sample.

Two failures stopped prematurely based on visual judgment (including the memory
run ending at 62 degrees); two encountered unsuccessful continuation or recovery
planning. The native microwave threshold is approximately 74.5 degrees. These
post-run measurements were never supplied to the acting agents. The memory agent
also recovered successfully in the shifted-camera scene by rebuilding geometry
and replanning without a new human hint.

Three of six memory sessions chose initial pixel-region fields exactly matching
the prior corrected episode. Current depth fits and plans were fresh, but this
study does not isolate abstract strategy transfer from reuse of example numbers.
The existing microwave tool assumptions remain, and one session per condition
per scene does not establish repeatability or cross-asset generalization.

The independent audit passed all six matched sensor pairs, 72 complete issued
trajectories, 3,907 recorded control steps, zero execution vetoes, frozen inputs,
source isolation, and all videos. Nineteen focused tests passed. No separate API
calls were made. Cleanup reclaimed approximately 9.225 GiB and retained 2.652 GiB
of unique study evidence; all twelve temporary workspaces and owned workers
closed. The persistent library remains unchanged. The report, diagnoses and
reproduction helpers are in `results/subscription-memory-challenge-20261019`.
