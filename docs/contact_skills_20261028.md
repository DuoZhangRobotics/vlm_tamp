# Push, hook-pull and momentum validation

The implementation adds `push` and `hook_pull` to the surface-contact tool
condition. These are complete, bounded skills rather than aliases for
`plan_pose`. Luna chooses current pixel regions, desired motion and contact
mode; numerical code chooses robot postures and execution timing. Existing
grasped manipulation remains available.

Implementation is in `src/vlm_tamp/contact_skills.py` and
`src/vlm_tamp/contact_skill_service.py`. The public contracts and compact display
are in `robot_contracts.py` and `compact_robot_interface.py`. The generic actor
introduction is `prompts/contact_skills_introduction.txt`.

## Boundaries

**User steering:** use **zero added collision padding**. Do not restore the
experimental 3 mm buffer or an adaptive buffer fallback. The user explicitly
asked to test without it. `skill_native_scene` records 0 mm and the regression
test checks that it makes one zero-padding call.

**Later user steering:** stop treating predicted environment contact as a hard
stop for bounded simulation contact experiments. The requested 3 cm diagnostic
executed the previously rejected path with physical contacts enabled. It reached
native success at control 528, five controls (0.25 s) after the stroke began,
and remained successful after withdrawal and one second of settling. Actual
forward hand travel was 19.98 mm; peak landmark speed was 90.58 mm/s for a
120 mm/s reference. The largest recorded individual normal contact force was
76.87 N. Closure occurred during the push.

The contact-skill interface now defaults to `collision_policy="advisory"` for
simulation approach, insertion, contact strokes and withdrawal. Sampled scene
predictions are retained in the response but do not veto robot-valid motion.
They are not an exhaustive collision-free certificate. The planner retains
robot joint and self-collision checks. `collision_policy="veto"` requests the
earlier stricter comparison.
Physical simulator collisions are never disabled. Momentum qualification is
reported separately and does not prevent authorized advisory experiments.

- Inputs are current RGB-D, camera calibration, known robot geometry and
  proprioception. No simulator object geometry, native hinge or private contact
  classification enters the skills.
- `preview=true` issues no physical motion. An execution request performs its
  own search and checks; preview does not grant permission to reuse stale poses.
- Default motion uses complete strokes up to 1 cm at 3 cm/s. Feedback decisions
  occur between complete trajectories, with native-success termination retained.
- Hook-pull does not close the fingers. The existing aperture is held, and an
  initial small motion tests engagement independently of pinch-grasp evidence.
- Contact permissions use distal finger/pad geometry, including overlapping
  collision meshes. Palm contact requires explicit `contact_surface="palm"`.
  Proximal fingers, arms and other observed surfaces remain in the diagnostic
  collision model; advisory experiments report violations without vetoing motion.
- Speed limits apply to trajectory references. Measured hand speed is computed
  from actual joint feedback and reported separately. This is not force control.
- Push-and-release is available for authorized simulation experiments under
  either collision policy. Source-matched validation is a separate field, not
  an execution permission. All three supported speed requests remain listed
  even when strict validation is missing or stale. Diagnostic contact audits
  remain separate from Luna's public feedback.

## Evidence and reproduction

The existing Live Preview page embeds the new report:
`results/luna-live-hinge-comparison-20261027/watch.html`.
The standalone report is `results/contact-skills-20261028/watch.html`.

`scripts/validate_contact_skill_checkpoint.py` restores the original command
prefix physically and checks the measured joint state against control 459.
It uses retained Luna annotations only for numerical diagnostics. The recorded
request is never copied into fresh actor prompts or transferable memory.

`scripts/run_contact_skill_study.py` runs three controlled-push repetitions,
three hook repetitions, and three repetitions at each of 3/6/12 cm/s for
push-and-release. It also supports Luna practice and evaluation phases. Run on
GPU 3 after sourcing `scripts/server_runtime_env.sh`. Use a new `--tag` to
preserve earlier runs. GPU access requires execution outside the current
filesystem sandbox; the sandbox does not expose `/dev/nvidia*`.

```bash
source scripts/server_runtime_env.sh
CUDA_VISIBLE_DEVICES=3 python scripts/run_contact_skill_study.py --phase qualify --tag NEW_TAG
CUDA_VISIBLE_DEVICES=3 python scripts/run_contact_skill_study.py --phase practice --tag NEW_TAG
CUDA_VISIBLE_DEVICES=3 python scripts/run_contact_skill_study.py --phase evaluate --tag NEW_TAG
```

The practice phase keeps Luna/high, three initial attempts, up to six attempts,
600 controls and 1,800 seconds per attempt. It retains learner-authored principles
verbatim in a versioned memory directory and freezes them
before evaluating close states 17 and 49 and open state 13. These cases have
historical use; fresh actors do not make them previously unseen benchmark cases.
There is no teacher, Reflexion model, model substitution or weight update.

## Important diagnostic findings

The first physical diagnostic (`physical-push-01`) reached native success at
control 491 and remained successful after one second of settling. This is not a
qualified fingertip-push success: offline contacts show that the palm hit the
door during the approach. The selected visual tracking region also included
stationary background. Preserve both the native success and the policy failure.

Subsequent changes separated free approach from contact insertion, refitted the
insertion from the measured servo endpoint, filtered target-depth association,
selected informative camera views, and refreshed observed occupancy after
measured object movement. Two controlled strokes in `physical-push-04` measured
4.49 mm and 7.60 mm of object progress before another stroke was rejected.
Later offline auditing also found a force-bearing palm proximity contact;
this does not qualify the skill or establish full-task repeatability.

Bounding the tracker nearest-neighbor query by its existing 4 mm acceptance
radius reduced tracker initialization from about 21.5 seconds to 1.1 seconds in
these diagnostics without changing the accepted neighbor criterion.

The `qualified-v1` cohort tests the subsequent 3 mm clearance version. Early
results reject the initial configuration at finger 2. The hook selection also
includes background through its image rectangle and is rejected as too broad.
These are unresolved implementation/association problems, not evidence that
Luna is incapable of pushing or that the handle cannot physically be hooked.

The buffered `qualified-v2` batch was stopped after its in-flight replay
completed when the user requested zero padding. Its unfinished conditions are
not completed experiments. The subsequent `no-padding-*` runs are the current
comparison; keep buffered results as historical diagnostics only.

With zero padding, `no-padding-push-01` measured 4.49 mm and 7.60 mm in two
strokes, then exhausted the next stroke's search. Total skill time was 39.89 s.
It did not complete closing. The offline log contains one force-bearing palm
contact at a nominal **+0.98 mm separation**, about 12.65 N, and a following
zero-force proximity entry. This is MuJoCo soft-contact behavior, not 1 mm of
geometric penetration. No extra planning buffer is being added in response.

The corrected `no-padding-hook-02` inserted across the handle, retained an open
aperture (approximately ±0.0399 m gripper joint positions), and measured 2.77°
for a 3° request, within its 0.5° progress tolerance. Its 46.43 s included 8.12 s
contact search, 8.77 s stroke planning, and 21.07 s execution plus capture.
Offline contacts were finger-pad only. The zero-padding qualification cohort
reproduces this behavior independently; inspect its completed rows for counts.

`scripts/serve_contact_checkpoint.py` adds a declared fresh-actor checkpoint
initialization. It executes the original complete physical commands, checks
the resulting robot state, and publishes only the current observation. It
supplies no historical regions, successful poses or action recipes to Luna.
The fresh checkpoint actor is separate from both deterministic replays and the
full-initial-state practice/evaluation cohorts.

## Interface repairs and current evidence

Contract v8 fixed a compact-response bug: the generic sequence formatter had
overwritten contact stages, hiding object motion and engagement from Luna.
The first v7 full-start practice finished 0/3; it is preserved under
`trials/advisory-v1-practice-close17`, not promoted as a successful method.

The v8 cohort (`advisory-v2-practice-close17`) was interrupted after one complete
unsuccessful attempt and a second attempt that encountered an implementation
deadlock. Joint 6 measured 3.752909 rad while the planner admitted up to 3.7475.
The gripper command also resent the out-of-range measured configuration as a
reference. Every new plan and the gripper operation then failed. The source
integrity audit confirmed 567 files unchanged during that cohort. Its second
attempt is an implementation interruption, not evidence of model incapacity.

Contract v9 makes `gripper` command one inward 20 Hz reference increment at the
existing 1 rad/s limit when measured joints leave the planner's admitted range.
It reports requested correction and measured recovery. It never assigns robot
state. A physical replay reproduced the stuck configuration to 4.44e-16 rad;
ten controls moved joint 6 from 3.752909 to 3.730327 rad, back inside admission.
See `advisory-v3-joint-recovery/replay.json`.

Advisory contact planning also records Cartesian interpolation deviations rather
than vetoing them at the former 6 mm threshold. Joint validity and self checks
remain. Approach search prefers an observed clear route if it finds one, then
falls back to robot-valid planning when that preference fails. This adds no
collision padding and creates no environmental collision veto.

With these changes, `advisory-v3-service-3cm` completed the full approach,
3 cm pulse, withdrawal and coasting. Native success and retained closure were
both true. Skill wall time was 23.28 s: contact search 3.05 s, stroke planning
0.24 s, execution plus capture 13.83 s; remaining phases are in the JSON/report.
The pulse tracker had only seven correspondences, so its pose was unresolved;
the authorized release procedure still completed, and later RGB-D fit recovered.

An offline tracking probe used all 50 recorded RGB frames from an earlier
failed approach. Only three selected agentview features existed initially;
two survived, still below the eight-pair rigid-fit requirement. More temporal
frames alone did not resolve this case. It used decoded archived video and
original endpoint depth, not simulator object state. Results are in
`advisory-v2-tracking-probe/result.json`.

The v9 advisory hook approach regressed in three matched tests: its first
robot-valid posture led to inconsistent insertion tracks. Contract v10 searches
observed geometry across hook postures first, with an advisory fallback inside
the same contact-search deadline. The corrected `advisory-v4-hook-r1..r3` tests
all achieved -2.787° for -3° requests, with finger-1-pad contacts only. The
minimum gripper-joint span was 79.31 mm throughout; no pinch command was issued.
Skill times were 42.68, 46.13 and 42.41 seconds.

Controlled pushes `advisory-v3-controlled-r1..r3` all reached native closure at
control 542 and remained successful through control 578. The diagnostic held
the final arm pose during settling; this does not establish closure after
withdrawal. Palm contact occurred, so these are closure-without-pinch results,
not pad-only qualifications. The later `no_progress_after_two_strokes` stop
occurred after native success because diagnostic early success stopping was off.

## Learner-authored practice memory

`trials/advisory-v4-practice-close17` completed six full-start Luna/high attempts:
all six reached native success, at controls 152, 136, 187, 156, 336 and 118.
Normal episodes stopped at first native success; post-release retention is a
separate diagnostic. No human task hints or forbidden evaluation fields entered
the actor feedback, and all recorded issued references matched the audit.

Luna identified that a broad window selection could return tabletop depth and
disturb mugs. Its frozen principles select an opaque observed surface, check
the fitted normal against the visible panel, reobserve after movement, and
change direction using the current panel/cavity relationship. They contain no
successful pixel boxes, world poses, joint vectors or state indices. The memory
is verbatim learner output, with explicit one-scene limits:
`results/contact-skills-20261028/frozen-memory/advisory-v4/guidelines.json`.
SHA256: `58f8817207de1e2b1936a9cd02037e857dd03be414aff1fa83166599195dba1b`.

Practice took 1,281.60 s: 527.10 s of robot services and 754.50 s outside the
service (model, transport and local analysis, not pure reasoning time). Contact
skills including previews used 109.71 s perception, 42.24 s contact search,
114.72 s execution/capture and 0.78 s stroke planning. Inclusive and nested
categories must not be summed twice. Detailed timing is retained in the trial.

One startup snapshot predates the final availability-metadata correction in
`contact_skill_service.py`. The service imports this module lazily; the first
actual skill reply confirms the corrected contract. Both source versions and
the event evidence are preserved in `source-override-provenance.json` and
`source-overrides/`; 568 effective sources were unchanged afterwards. This was
an interface correction, not an actor hint or an unrecorded physical change.

## Frozen-memory evaluation and release checks

Fresh actors with the unchanged frozen memory completed closing state 17 at
control 128 and opening state 13 at control 575. Closing state 49 exhausted
600 controls without success. Thus this cohort is 2/3 fresh native successes,
following 6/6 practice successes. These cases have historical use; do not call
them unseen benchmark cases. Every evaluation's source audit was unchanged.

The state-49 failure was not IK refusal: its first push opened the door from
107.26° to 110.52° (offline grading). A selected grasp approach consumed 315
controls; closure consumed 40, and two arc executions consumed 51 and 90.
Those arcs brought the door from 114.61° to 54.19° and then 11.09° open.
The final measured pad gap collapsed to approximately zero versus a modeled
19.45 mm handle. Only seven controls remained. The cheapest candidate recovery
approach found needed 38, with six available after allowing one contact command.
This is a contact and budget limitation, not demonstrated global infeasibility.
Native door angles were never passed to Luna or planning. See the trial's
`failure-analysis.json` for the separate evaluation evidence.

The completed fresh-evaluation wall/service/outside-service times were:

| Case | Actor wall | Robot services | Outside services |
| --- | ---: | ---: | ---: |
| close 17 | 324.60 s | 81.46 s | 243.14 s |
| close 49 | 689.50 s | 209.30 s | 480.20 s |
| open 13 | 611.35 s | 277.78 s | 333.57 s |

Outside-service time includes model calls, transport and local analysis. Time
to the successful tool reply was 286.61 s for close 17 and 531.97 s for open 13;
actor wall includes subsequent reporting. Do not label the remainder pure
reasoning time or add inclusive phase totals twice.

Four separate diagnostics replayed practice successes 1–3 and the fresh close-17
success, then planned a 15 mm withdrawal and advanced physics for one second.
All four remained closed. Three had no force-bearing robot/environment contacts
during coasting. Practice attempt 3 retained palm contact (maximum 6.54 N), so
that row does not establish fully disengaged closure. Prefix robot-state errors
were at most 1.63e-12 rad. Normal Luna stopping was unchanged. The reproducible
diagnostic is `results/contact-skills-20261028/replay_actor_retention.py`; its
results are in `actor-retention-comparison.json`.

Tool availability and selection remain distinct: practice made 35 push calls
including previews and three hook calls (one executed and returned unresolved
motion). Luna did not select push-and-release in practice. The 3/6/12 cm/s
comparison is a scripted physical validation, not a claimed learner discovery.
The frozen guidelines remain learner-authored and were not updated from the
fresh evaluations or these private diagnostics.

The final report is `results/contact-skills-20261028/watch.html`, embedded in the
existing `results/luna-live-hinge-comparison-20261027/watch.html`. It includes
tool inputs/outputs, overlays, videos, phase timing, native outcomes and retention.
The regression run recorded 146 passes; the final availability-metadata change
also passed its 75 relevant interface/contact tests. No added collision padding
or object-state assignment was introduced.

## Reading failures

Every skill result contains the stop reason, relevant measurements, attempted
postures, stage feedback and inclusive timing. Compact replies retain summaries;
the full JSON preserves candidates and collision examples. A finite exhausted
search does not prove global physical infeasibility. Native success, measured
engagement, contact-policy conformance and retained closure are separate claims.

Do not describe exploratory execution as validated contact behavior. Inspect
the individual replays, source hashes and offline contact audits. The later
user-authorized 3 cm comparison achieved retained closure in all nine runs:
3/3 at each of 3, 6 and 12 cm/s reference speed. Time from the prepared pose to
native success was 0.70, 0.45 and 0.25 simulated seconds respectively. These are
matched physical replays, not nine fresh autonomous Luna decisions.

The first fresh Luna checkpoint run also succeeded (control 473, 14 new
controls, 274.08 s actor wall time), but it used the existing 30° arc after a
push preview failed. Do not attribute that success to execution of the new
push skill. Its notes and exact input/output are retained in
`results/contact-skills-20261028/trials/zero-padding-luna-checkpoint459`.
