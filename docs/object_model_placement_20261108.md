# Persistent object placement: success before latency

The user clarified that wall-clock efficiency is the intended benchmark direction, but requested
reliable task completion before choosing a tighter deadline. The current 600-control limit stays
unchanged. No latency cap, model switch or reasoning-budget reduction was introduced.

## Reproduced problems and numerical repair

The object-point tool was missing the object-state connection needed to use it reliably:

- The raw selected rectangle shifted the estimated source reference by about 34 mm compared with
  a SAM object mask. A visible object's lowest surface sample also need not be its resting point.
- Recorded CoTracker/RGB-D correspondences measured motion during acquisition. The previous
  diagnostic had assumed the pre-grasp object reference stayed fixed through closure.
- An endpoint-only transfer tilted the modeled object by about 33 degrees and moved its reference
  14 mm below the selected destination surface. The hand reached its endpoint but the physical
  interaction changed the object/grasp relation.
- A direct low transfer could move the plate. Lift, transfer above the destination, and lowering
  need distinct geometric objectives, even when they are prepared as one complete trajectory.

The integrated workflow uses the caller's selected regions, MobileSAM masks, measured RGB-D,
CoTracker and existing rigid registration. For `supported_body`, it fits the declared resting
surface from valid points outside the object mask within the supplied region and projects the
observed body reference onto that plane. It does not use a native object pose or hidden mesh.
`surface_center` and `surface_point` define destination references explicitly. Mask scores are
not semantic guarantees, and the planar-resting model remains a declared assumption.

Grasp generation keeps its existing caller-region input. An earlier diagnostic changed that to
the SAM mask as well and stopped 8.41 mm short during grasp approach. That failed trial remains in
the report; the integrated repair changes the tracked object model without substituting a second
grasp-generation method or trying an alternate grasp automatically.

## Public workflow

`select_object` retains a separate object ID, reference geometry, mask and observation provenance.
Selecting a destination no longer replaces the source object model.

`grasp_object` uses the existing GraspGen, cuRobo and approach executor, closes physically, and
updates the object's rigid pose from recorded video and measured depth. It does not certify grasp
retention from closure alone. Unresolved tracking remains unresolved; no identity transform is
substituted. Wrist-selected parts use the existing measured cross-view association to supply the
agentview video tracker, without adding a physical camera.

`place_object` accepts source/destination IDs, an explicit rigid-attachment assumption and an
explicit physical lift distance. It prepares lift/transfer/lower paths before motion, holds the
gripper through transport, and releases if the placement reference is reached. Native success
termination remains unchanged. There is no mandatory post-release withdrawal or automatic retry.
`object_status` exposes retained pose timestamps and whether a measurement is current.

The physical lift is a requested movement, not collision padding. Qualification used 12 cm as a
declared relative setting. It supplies no successful world coordinates to Luna. The source and
destination choices in numerical replays are taken from the prior Luna recording; they are not
supplied to the fresh agent evaluation.

## Native carry constraints and search diagnosis

cuRobo proposes object-point goals with a chosen object-axis relation. MC-VAMP maintains that axis
and an observed support half-space throughout each path, with the existing planner projection and
no additional dense validation pass. The constraints add zero geometric clearance. Object-axis
alignment leaves twist free; it is not a prescribed full wrist quaternion.

The first constrained query exposed dynamic-domain pruning: valid endpoints and a blocked direct
connection were followed by 3,000 iterations with only a few collision checks. On the same recorded
query, ordinary C-RRTC exploration found a path in eight iterations, with 39 checks and about 2 ms
native solve time. Pruning is disabled for this new object-carry mode; hinge-arc settings are
unchanged. No alternate planner or CPU endpoint correction was added.

This is a kinematic reference/axis/support constraint, not full carried-object collision modeling,
force prediction or proof of a rigid grasp. Current observed robot/scene collisions remain checked.
The caller-supplied attachment assumption and independently measured outcomes remain separate.

The native extension was built with the existing script, using `cmake --build ... -j32` for both
native targets. Native tests passed. Other machines must rebuild the pinned adapter before using
the new ABI; the existing source/binary identity check reports a stale build explicitly.

## Important scoring correction

Operator qualifications used `stop_on_success=False` so they could finish release and inspect the
final physical state. This also withholds native outcome feedback from the acting interface.
`episode_status` previously returned `task_success=False` in that mode. The operator summaries
incorrectly treated that field as the native score.

The native action logs already recorded success for `physical-supported-transfer` at control 299,
retained through release at control 323. `physical-supported-release` first succeeded at 292 and
remained successful at 396. The extra withdrawal in the latter was therefore not required to
establish success and is not a mandatory part of the integrated tool.

Public status now uses `task_success=null` with `task_success_available=false` when native feedback
is withheld. It does not expose an oracle label to blind conditions. Operator scoring reads the
separate private evaluation and immutable action fields. The correction audit preserves the
original summaries. Earlier fresh Luna failures used native stopping and remain genuine failures.

A private exact-control replay independently reproduced the successful native outcome. Maximum
robot-joint difference from the source was approximately 4e-13 rad. Native object poses were used
only to audit perception/attachment error; none were supplied to models, goal generation or plans.
The replay's first initialization error is retained separately and is not a robot-task outcome.

## Qualification results

All three public-API replays use state 1, seed 0, the same recorded choices, 600 controls and frozen
numerical sources. These establish local execution repeatability, not a general task success rate.

| Replay | First native success | Final controls | Native success after release | Wall s |
| --- | ---: | ---: | --- | ---: |
| Public 01 | 317 | 340 | true | 63.68 |
| Public 02 | 305 | 329 | true | 81.90 |
| Public 03 | 293 | 316 | true | 64.37 |

Every diagnostic, including earlier failures and the scoring correction, appears in
`results/object-state-integration-20261108/watch.html`. Videos stay external and the page refreshes
only on request. The report shows masks, tracked points, exact tool requests and phase timing.

170 focused Python tests pass, including native constraint behavior, old contact/arc behavior,
object identity preservation, failed tracking, schema exposure and withheld-feedback semantics.
Scoped Ruff and `git diff --check` pass. Two older artifact-dependent arc tests were separately
skipped because their fixtures are absent; no missing fixture is treated as a passed experiment.

Each physical job has a 32 GiB cap, zero swap and the existing memory/I/O/SSH watchdog. Replays run
sequentially. There are no changes to another user's jobs, native task predicates or the two-camera
observation contract. A fresh Luna result is reported separately under `luna-fresh`; no success is
inferred for that evaluation from these operator replays.

## Completed fresh Luna evaluation

The clean Luna/high episode at revision `4cabc1c` achieved native success at control 398. It used
its own regions and choices, eight anchored memories, the existing feedback format, two native
cameras, 600 controls and the normal 1800-second wall limit. No operator coordinates, prior task
memory, worked example or private native object pose was supplied.

The success involved recovery, not a perfect high-level-tool sequence. Luna selected source and
destination object models, retried a failed grasp batch with another seed, and later opened and
regrasped after expressing uncertainty about retention. It then called `place_object` with an
8 cm lift. That prepared path plus release required 1,204 controls with only 323 remaining, so it
issued no motion. The separately scheduled phase costs were approximately 214 for lift, 95 for
transfer and 879 for lowering; combined scheduling rounds differently. The current tool takes
the first valid path per phase and checks total cost afterward. Remaining-control-aware goal/path
selection is an unresolved limitation; no geometric impossibility was demonstrated.

Luna recovered with `plan_object_point`, using the measured model references and explicit carry
constraints. After correcting an omitted plan ID, it executed the transfer and requested release.
Native success occurred after three of twelve opening controls; the normal episode terminated
immediately. Thus this run validates successful agent use of object models and numerical tools,
but it does not show that Luna's attempted composite placement path was executed.

Actor session time was 394.40 s, including 305.37 s model/client time. The robot episode from reset
through native termination took 381.06 s. There were 25 model decisions, including nine help calls.
The control-time video is about 19.95 s. These are distinct time scopes, not a tightened deadline
or a matched causal speed comparison against an earlier failed rollout.

A separate private replay used the same commanded prefix, completed the remaining nine opening
controls and continued twenty settling controls. Native success remained true at controls 407
and 427. Maximum measured joint deviation along the replayed prefix was 0.000678 rad; the commands
match, but this is not a claim of bit-identical physical states. No new planner targets, model
calls or native object-pose inputs were used. Normal Luna stopping was not changed.

Fresh-run evidence is in `luna-fresh/watch.html`, with exact requests, tool results, native scoring,
all recovered errors and timing. Continuation evidence is in `retained-luna-success/summary.json`
and its recorded video. These establish one successful fresh episode and local repeatability;
broader task/state reliability remains to be evaluated.

The frozen fresh-evaluation configuration is preserved as
`configs/study_object_models_20261108.json` for the existing supervised visual-memory runner.
It pins numerical revision `4cabc1c`; the usual resource-limited supervisor remains required.

During the fresh episode, peak anonymous/shared job memory was 5.89 GiB, host available RAM stayed
above 70.32 GiB, and maximum sampled host full-I/O pressure was 11.55%. All sampled SSH probes
remained healthy and no OOM kill occurred. The episode, repeatability group and continuation
diagnostic stopped and released their workers.
