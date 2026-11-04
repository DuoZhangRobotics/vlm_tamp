# Direct-tool visual interface and bounded interaction memory

This implements visual-interface elements described by OpenETA and transition retention
inspired by RoboICL. It is a controlled adaptation to the existing numerical backend, not a
reproduction of either complete system.

## Frozen design and information boundary

Luna/high returns one JSON operation and argument object per decision. Argument JSON is data,
never evaluated as Python or shell. The host dispatches through the existing robot client.
Each model call has a fresh restricted workspace containing only the supplied images and schema;
shell and image-retrieval tools are disabled. The robot client and experiment repository remain
outside that workspace. Model-authored policy programs cannot issue robot actions.

Only `agentview` and `wrist` RGB-D enter the visual tools. XY, YZ and XZ views reproject those
observed points. Display sampling and three-pixel splats do not change the planner's cloud.
Each rendered sample retains its exact original camera/pixel. Missing pixels return no sample;
there is no nearest-point substitution, hidden-surface completion or simulator contact overlay.
Free-space marking is an explicit coordinate operation: one projection fixes two world
coordinates, and the second supplies the remaining coordinate.

Whole-hand previews transform known robot geometry into a requested or already planned pose.
The current aperture is used for contact previews; an explicit open preview uses the same
robot-only FK and calibrated opening as the existing grasp planner. Previews report their
aperture and do not issue motion, rerun IK, add collision checks or select a new contact.
Successful numerical plans receive previews in the enhanced condition without adding a gate.

The common backend still uses cuRobo IK, VAMP/MC-VAMP and the existing controller. A wrapper
restriction requiring a hinge before generic GraspGen use was removed: a surface-selected
cloud can use the same direct GraspGen path, with no fabricated joint or angular motion.
Both study conditions share this adapter correction. Joint-linked sequences still require
an explicitly associated model. Official LIBERO task identifiers and suite admission were
extended without adding task-specific actions, poses, directions or object geometry.

## Memory conditions

Both conditions retain at most eight complete positive-length interactions, comprising
before images, the chosen request, its measured receipt and after images. The baseline retains
the first and latest interactions. The enhanced condition reserves fixed temporal anchors
and a latest-interaction slot. Repeated anchors referencing one transition are deduplicated.
Omitted control intervals are explicit `TRAJECTORY_GAP` records. Non-motion results remain
in a bounded immediate tail and do not consume execution anchors.

Native RGB images are supplied separately and byte-for-byte unchanged, with camera identity in
the image manifest. The original per-camera pixel coordinates therefore match the tools directly;
there is no montage offset or header displacement for the agent to subtract.

Memory contains only the current episode. No demonstrations or inherited task lessons are
supplied. A short learner-authored note remains a labelled hypothesis. Input attachments,
hashes, exact prompts, selected memory records and outputs are recorded per decision. The
48-image input ceiling reserves complete selected transitions and current observation before
adding tool illustrations; any omitted illustrations are listed and can be requested explicitly.
Non-motion replies are kept only in the immediate tail, not copied into every executed anchor.

This evaluates within-episode context use. It does not establish cross-episode self-improvement
or transfer of a practiced skill library. No claim of policy-code learning is made.

## Preregistered pilot

The corrected study is `results/luna-visual-memory-native-images-20261104/watch.html`.

`configs/study_visual_memory.json` declares one initial state for each of four tasks:
microwave closure (state 17), drawer closure (state 0), plate sliding (state 0), and bowl placement
(state 0). Each has a fresh baseline and enhanced attempt. Condition order alternates by case.
All eight episodes use Luna/high, 600 controls and 1800 seconds, with native success as the
terminal outcome. The numerical sources are hashed before the batch and checked unchanged.

The comparison tests the combined visual and memory change, not their separate causal effects.
One initial state per task is a pilot, not a statistically reliable generalization result.
Infrastructure failures remain separate and are not replaced silently. Existing memory/I/O/SSH
monitoring, the 32 GiB cgroup and zero swap allowance govern the sequential batch.

`watch.html` shows all cases, videos, timing, exact model images, retained memory, decisions and
tool receipts. It refreshes only on request; small live progress files update per completed
decision. Images are copied once for viewing; depth arrays and videos are not repeatedly
rewritten for the dashboard.

References: [OpenETA interface](https://github.com/OpenMOSS/OpenETA/blob/openeta-for-codex/docs/openeta-light.md),
[RoboICL method](https://arxiv.org/html/2609.34261v1), and
[Codex structured outputs](https://learn.chatgpt.com/docs/non-interactive-mode).

## Interrupted initial pilot

The initial study in `results/luna-visual-memory-20261104` was stopped after one completed
microwave baseline and during its enhanced counterpart. The combined camera sheet added a
36-pixel header while tools expected unshifted per-camera pixels. In the enhanced episode,
Luna described pixel `(276,270)` as a door border, but the original pixel selected the mug
below it. This is consistent with reading coordinates from the sheet. That interpretation is
not a controlled attribution of every failed mark, but the avoidable coordinate transform was
removed before repeating the paired study. The original episode remains recorded and the
interrupted enhanced episode is excluded from completed outcomes.

The initial memory implementation also copied each immediate non-motion tail into the next
executed anchor. This retained old help responses and failed previews indefinitely with those
anchors; the baseline's final model input reached 99,651 tokens. The corrected memory keeps
non-motion replies only in the immediate tail. Complete executed transitions remain unchanged.
These are interface corrections shared by both replacement conditions, not planner changes.
