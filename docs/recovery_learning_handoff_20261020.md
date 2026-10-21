# Microwave practice and memory transfer conclusion

Luna learned a method that succeeded twice on its practice scene. Its frozen
lessons then succeeded on two of four fresh initial states, compared with one of
four without those lessons. This is evidence of limited, inconsistent transfer,
not a reliable general manipulation policy. Local experiments are concluded for
the server handoff; no further physical runs are needed on this workstation.

## What was tested

The completed study is `results/recovery-learning-transfer-20261020`, after the
older persistent-contact study shown in the browser. Luna used high reasoning
through the ChatGPT subscription. It controlled the same fixed numerical tools
for current RGB-D grounding, optional DINO, grasp/IK queries, contact-aware motion
planning, tracking, and physical execution. Numerical algorithms were not changed
between practice and transfer. No separately billed API requests were made.

Practice used state 1, seed 0, one continuing agent context, five initial attempts
and a maximum of twenty. Each episode had 600 control steps and a 1,800-second
wall budget. The agent could extend practice in batches. It had an abstract
example, earlier guidelines, a prior successful-method description, and an
earlier read-only memory library. This was guided practice, not learning entirely
from scratch. The experimental agent was denied access to the implementation and
private evaluation geometry.

The first five attempts failed. Attempt 5 reached its wall budget at 519 steps;
the other four exhausted 600 steps. Attempts 6 and 7 succeeded at 416 and 414
steps, respectively, in approximately 9.5 and 8.3 minutes. Luna stopped after
these two consecutive successes, as the recovery-study prompt allowed. No
separate post-success efficiency experiment was completed. Both successes used
multiple motion segments; one-trajectory execution was not demonstrated.

The frozen memory is structured JSON containing conditional principles,
supporting findings, unresolved hypotheses, and limits. It contains no historical
world coordinates, pixel boxes, joint trajectories, or replay code. It is prompt
context, not a model-weight update. It recommends fresh grounding, checking actual
motion, changing contact mode when justified, and short point/rigid arcs informed
by tracking. Relative arc settings remain provisional experience from this asset.

## Paired transfer result

Each condition received one episode per state, with no across-case learning.
Both conditions had identical tools, limits, an abstract worked example, and
matching initial observations. All 48 initial sensor arrays matched within each
pair. Only the memory condition received the new frozen practice synthesis;
neither condition received the old practice library or successful-method file.

| Initial state | Without new memory | With new memory |
| --- | --- | --- |
| 2 | Failure, 600 steps | Success, 497 steps |
| 12 | Failure, 600 steps | Success, 495 steps |
| 32 | Failure, 600 steps | Failure, 600 steps |
| 44 | Success, 600 steps | Failure, 600 steps |
| Total | 1/4 | 2/4 |

State 44 without memory succeeded on the final allowed action. All eight planned
episodes remain in the denominator. Success is LIBERO's native success signal;
episodes stop immediately on that signal, with no retreat. All transfer failures
exhausted their control budget. There were no execution-time vetoes, premature
episode endings, or commands after native success. Planning admission checks
remained enabled and could reject candidate plans.

## Failure evidence and interpretation

The most specific contact diagnosis is state 32 with memory: at the first closure,
the finger pads touched each other and only one finger contacted the microwave.
Successful comparison closures had both pads contacting the microwave. This
suggests ineffective acquisition/retention of the intended contact, but a single
snapshot is not a force-closure test. Hooking and pushing are allowed; a pinch is
not a benchmark requirement. This episode peaked at 16.6 degrees and ended at
10.9 degrees. Its later grasp/recovery attempts did not rescue it within budget.

State 44 with memory reached only 19.4 degrees while the control condition
succeeded. We have not isolated one causal explanation for that regression.
State 2 without memory was a near miss: peak 72.1 degrees versus the native
opening threshold of approximately 74.5 degrees. Evaluation angles and native
contacts were inspected after episodes, never supplied as geometry to the agent.

The defensible conclusion is that agent practice can produce useful conditional
lessons, but those lessons can also steer a new scene poorly. Four same-asset
pairs and one sample per condition do not establish statistical reliability,
generalization to new objects, or a speed advantage. The previous 3/4 persistent
study used different states and supplied experience; it is not a matched baseline
for the current 2/4 result.

## Server handoff contents

The compact archive contains the current source tree, including uncommitted work;
the measured source snapshot; frozen prompts and memory; outcome/audit records;
agent/tool transcripts; issued numerical trajectories; all 15 episode videos;
selected initial/final images; dependency revisions and installed-package
inventories. File hashes verify the archive. The standalone handoff HTML links
the videos using relative paths.

It excludes Conda environments, model weights, compiled caches, Git object
history, raw RGB-D arrays, and older experiments except the small prior inputs
needed by the runner. It is a source-and-evidence handoff, not a complete raw-sensor
replay archive. The retained full study remains about 4.6 GB on this machine.
Do not delete that only raw-data copy before deciding whether to transfer it too.

## Restarting on the server

1. Extract into a new directory, or overlay onto a clone at the recorded Git base.
   The archive records working changes and exact upstream revisions. Preserve
   the measured snapshot separately from future modifications.
2. Recreate dependencies on the server. Consult `docs/setup.md`,
   `scripts/setup_rekep_libero.py`, `scripts/setup_curobo_libero.sh`, and
   `docs/mcvamp_microwave.md`, together with the included environment inventories.
   The code expects environments under `.runtime/envs/rekep-libero`,
   `.runtime/envs/curobo-libero`, and `.runtime/envs/dinov3`; the LIBERO simulator
   uses the separate `vlm-tamp` environment. Fetch the pinned upstream repositories
   and authorized model weights. No credentials are included; authenticate on
   the server separately.
3. Review machine-specific paths before running: `/home/duo/miniforge3`,
   `/home/duo/.codex`, `/usr/local/cuda`, executable locations, and GPU architecture
   settings. Rebuild CUDA/MC-VAMP extensions for the server; do not copy the old
   binaries. Verify the Codex CLI supports the recorded isolation options and
   run its isolation probe. The setup has not been validated on the destination.
4. Run the existing targeted tests, headless simulator smoke test, and a small
   controller/planner smoke test before starting any new benchmark. Fifty-two
   targeted tests passed before this study; 56 passed after storage-only fixes.
5. The study entry point is:

   ```bash
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     python scripts/run_recovery_learning_study.py \
       --output results/recovery-learning-transfer-server-01
   ```

   This starts fresh practice and paired transfer; it does not reproduce an
   identical stochastic agent transcript. Use a new output directory. The
   runner's older diagnostic input and small guided-practice inputs are included
   at their original relative paths. Original evidence and frozen memory remain
   unchanged.

After migration, prioritize a controlled repeat of this paired comparison and
contact/recovery diagnosis before adding task-specific rules or broadening tasks.
States 2, 12, 32, and 44 are now observed evaluation states; further debugging on
them must be labeled development work and followed by fresh held-out evaluation.

## Storage policy

The new runner had omitted the final raw-input retention pass. That omission was
fixed after the experiment, along with duplicate workspace copying and a free
space guard. These storage changes did not produce the reported outcomes.
Future runs prune redundant sensor exports, move completed workspaces, and require
20 GiB free on the relevant Linux volume and Windows C: when mounted.

Keep compact outcomes, prompts, memory, source hashes, trajectories, and videos
by default. Transfer full raw sensor evidence deliberately; do not copy the
107 GiB historical results tree or 57 GiB runtime directory merely to resume work.
