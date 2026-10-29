# Verified sensor geometry: numerical planning and execution diagnostic

**Complete: seven primary episodes and one separate follow-up.** [Dashboard](../results/verified-grounding-planner-20261026/watch.html)
contains the frozen inputs, stage outcomes, full numerical requests/replies and
timing. This follows the [Luna LIBERO comparison](luna_landmark_libero_20261026.md).

The question is which failures remain when the initial mechanism and part
selections are checked independently and Luna does not revise the model.
There are no model calls, no new learner memory, and no native object geometry
in the planner or geometry-preparation code.

## Findings

Better initial grounding is useful, but it does not fix the current short-arc
policy. The opening reference succeeded; all three closing states failed under
both original and reviewed geometry. A separate fresh full-sequence test then
closed reviewed state 17 in **315 controls and 147.3 seconds**, with no Luna
calls. This establishes feasibility for one existing execution interface on
one development case, not general reliability or a pure arc-length ablation.

| Closing state | Original hinge error | Reviewed hinge error | Final open angle, original | Final open angle, reviewed |
| --- | ---: | ---: | ---: | ---: |
| 9 | 50.1 mm | 4.8 mm | 40.5° | 55.6° |
| 17 | 253.1 mm | 5.2 mm | 88.9° | 48.2° |
| 49 | 4.8 mm | 5.3 mm | 61.4° | 61.2° |

These six closing episodes all failed the native goal. Lower remaining angle
alone is not success. State 9's original model moved farther while losing
contact and accumulating about 19.4° of pose-angle error. Its reviewed version
kept both finger contacts through its two arcs and had at most 0.52° pose-angle
error, yet the next continuation search failed. This is evidence of a remaining
planning/policy problem even when tracking is reasonably accurate.

Reviewed state 17 changed from no grasp admitted to two approximately 20°
physical closing arcs with finger contact retained. Its maximum pose-angle error
was 2.44°. The next 20° planning query internally reached 17.1° before endpoint
goals violated the 43° wrist-rotation limit. Smaller fallback requests also
tighten this limit: a 10° request permits 23°, and a 5° request permits 13°.
No partial search path was executed. The fallback ladder therefore does not
monotonically relax the planning problem.

State 49 already had a good initial hinge estimate. Both versions lost finger
contact after the second arc and ended around 61° open. Original-state-49
recovery admitted and executed a grasp after reserve relaxation, but closure
established no finger contact and subsequent arcs still failed. Geometry,
physical attachment and recovery admission need separate diagnostics.

All three reviewed closing cases solved a complete virtual 90° path with the
extended planning allowance, taking about 27–30 seconds in those queries.
Their 20-second queries had timed out. These paths were not executed in the
primary cohort; 90° also does not finish every initial door angle.

The separate state-17 sequence freshly regenerated its paths and reached a
final open angle of 0.071°. Its planned costs were 148 approach controls,
40 closure controls and 102 manipulation controls. Native success stopped the
manipulation after 67 controls, for 315 including the 60 reset controls.
However, finger contact disappeared at control 296, when the door was still
31.6° open. The sequence finished through hand-body contact and door motion.
Task completion therefore does **not** demonstrate continuous grasp retention
or certify collision-free physical execution. Contact transitions are retained
in the follow-up report. Scene fitting is skipped on terminal success, so the
grader does not label the stale pre-arc estimate as a final pose-estimation error.

The [physical progress plot](../results/verified-grounding-planner-20261026/physical-progress.png)
compares actual door motion and transported scene estimates. Native quantities
are post-run grading only; the model-motion curve is aligned to the native
initial angle only for that comparison.

## Time and implementation bottlenecks

The primary cohort took 82.3 minutes. Five explicitly allowed secondary
reserve-relaxed searches consumed 2,966 seconds, about 60% of that wall time.
This diagnostic overhead must not be presented as the cost of an ordinary
episode that would stop at the initial reserve rejection.

Reviewed state 17's recovery search took 605 seconds. It made 693 approach-path
leg queries, totaling 563.7 seconds: 417.0 seconds of dense resampling/collision
checks, 27.0 seconds of path search, and 119.7 seconds of other query overhead.
Grasp IK took 2.9 seconds. The expensive component was repeated CPU path
validation, not language-model reasoning or GPU IK. The HTML includes these
nested timings and candidate-rejection counts for every grasp search.

The successful separate sequence took:

| Component | Seconds |
| --- | ---: |
| Reset/startup | 21.5 |
| Grounding | 5.6 |
| Grasp generation and selection | 27.4 |
| Two long-path planning queries | 54.4 |
| Sequence execution, sensing and terminal bookkeeping | 37.7 |
| Other trial overhead | 0.7 |

Within the 37.7 seconds, physical trajectory calls/recording took 5.3 seconds,
captures took 20.0 seconds and tracker initialization took 8.1 seconds.
Those values are nested, not additional wall time.

An offline sensor-only mask benchmark also exposed a cheap observation
optimization. The tracker currently performs an unbounded nearest-neighbor
query only to test whether a point is within 4 mm of the moving surface.
Passing that 4 mm bound into the tree query produced exactly the same masks
on ten saved camera views from two snapshots, reducing query time from
9.102 to 0.266 seconds, or 34.3× for that component. Runtime code was not
changed. This is not an end-to-end speedup claim; capture and other tracker
work remain. See [benchmark](../results/verified-grounding-planner-20261026/tracker-mask-benchmark.json).

The next controlled changes should unify selection/execution timing, separate
wrist constraints from arc length, and evaluate longer grasp lookahead with
explicit grasp-versus-push/contact-retention semantics. The observation mask
optimization can be tested independently. These findings do not justify a
stronger language model or injecting native task answers into Luna's memory.

## Design

Seven fresh simulated episodes use the same bounded scripted numerical policy:
original versus reviewed sensor geometry for closing states 9, 17 and 49, plus
opening state 13 as an original-model reference control. These are known
development cases, not an autonomous-agent or held-out benchmark.

The reviewed closing models use operator-selected attachment edges in two
calibrated RGB views to triangulate a vertical hinge line. A third view is
withheld for reprojection checking. Separately annotated handle/plane fits
are compared across views. All three closing cases passed: handle centers
agreed within 1.8 mm and withheld-view hinge residuals were below one pixel.
These checks establish consistency, not exact hidden-joint accuracy.

The reviewed opening-state-13 hinge did not pass the same consistency gates.
Its reviewed model is not used. The original sensor-derived model is retained
only as a positive reference; its successful history does not make it exact.
All annotations, checks and the failed preparation result are preserved.

Within each episode the hinge and axis stay fixed. Standard RGB-D/hand-motion
pose transport remains active, so current object pose can still become wrong
after contact loss. Pose drift is graded separately after the whole cohort.

## Stages and limits

1. Plan a grasp with a 3° contact probe, first 24 and then up to 100 candidates.
2. From the selected virtual grasp, test a planning-only 90° lookahead with
   the existing 20-second limit. A deadline failure permits one 120-second
   planning-only diagnostic. These paths are not executed, and 90° is not
   claimed to complete every closing task.
3. Execute a fresh open-finger approach, close for 40 controls, then request
   complete short arcs: 20/10/5/1° point mode, then 5° jaw-axis/rigid alternatives.
   At most twelve arc iterations are allowed.
4. One recovery is allowed after no executable arc or two consecutive possible
   motion stalls. The trigger uses public point-displacement/depth-support
   evidence, not native contact labels. Recovery releases, tries a bounded
   clearance motion and searches for a new grasp.
5. If the existing manipulation reserve blocks recovery, a labelled secondary
   probe sets that reserve to zero. It may execute only when the exact scheduled
   approach, closure and a minimum continuation fit within the unchanged
   600-control cap. Success after this intervention is reported separately.

All plans are newly generated. No old joint trajectory is replayed. The same
cuRobo v2 backend, 3 mm arc-position tolerance, collision checks and controller
remain in use. Issued trajectories finish or stop on native success; no new
execution-time veto is introduced. The 1,800-second numerical wall budget is
checked between operations, allowing an already-running query or trajectory
to finish. A trial stops at an isolated unrecovered stage rather than consuming
controls merely to reach the cap.

## Verification and interpretation

Eighteen focused tests passed. Integration preflight admitted a fresh grasp
and completed a virtual arc probe without executing a manipulation trajectory.
The separate budget-analysis test also passed. Final integrity checks passed
for all 549 frozen runtime source/config files, 105 primary per-trial checks
and 23 follow-up checks. These checks cover provenance, controller references,
control limits and execution completion; they do not turn native task success
into a physical contact-safety certificate. The follow-up's selection-description
error is separately disclosed below.
Fresh initial metric observations and calibration are compared with the saved
review source. All numerical code, annotations and model inputs were frozen
before scored runs.

Native hinge errors, object motion, finger contacts and model-pose drift are
evaluated only after the full cohort finishes. A failed finite search does not
prove global infeasibility. A geometrically valid path does not certify retained
physical contact or unmodeled moving-object/environment clearance. Any success
with reviewed annotations is an assisted diagnostic result, not a new Luna
capability or an automatically learned memory.

- [Protocol](../results/verified-grounding-planner-20261026/protocol.json)
- [Geometry verification](../results/verified-grounding-planner-20261026/geometry-verification.json)
- [Fixed numerical policy](../results/verified-grounding-planner-20261026/run_diagnostic.py)
- [Sensor geometry check](../src/vlm_tamp/sensor_geometry_check.py)
- [Post-run evaluator](../results/verified-grounding-planner-20261026/evaluate.py)
- [Primary integrity audit](../results/verified-grounding-planner-20261026/audit.json)
- [Worker shutdown audit](../results/verified-grounding-planner-20261026/shutdown-audit.json)

## Separate full-path follow-up

After reviewed closing state 17's 90° virtual lookahead solved,
and before native grading, one fresh execution was registered
to test that result physically. It runs only after the primary cohort, generates
both approach and arc again, and uses the existing `execute_sequence` tool.
It does not replay the previously saved trajectory. The complete scheduled
sequence must fit the same remaining control budget.

This is a prospective diagnostic on one selected case, separate from the seven
primary episodes. It is not a clean ablation of arc length alone: the existing
sequence tool also holds the selected grasp joint target during closure,
whereas the primary policy's separate `gripper` call holds the measured joints.
The existing point-mode wrist allowance also depends on requested arc length:
with the refined flag, a 20° request allows 43° of wrist rotation and a 90°
request allows 183°. This is another difference between the two interfaces.
There is no replanning after closure within the sequence. Its outcome therefore
tests the complete existing sequence interface, including those assumptions.

**Registration correction:** the original record incorrectly called state 17
the first reviewed case with a solved 90° lookahead. Reconciliation of the
completed logs shows reviewed state 9 had already solved one. The original
registration is preserved, with an explicit correction. State 17 was selected
before this cohort's native grading, but it must not be described as selection
by a correctly followed "first solved case" rule. This remains one selected
development example, not an unbiased success-rate estimate.

- [Follow-up registration](../results/verified-grounding-planner-20261026/full-path-followup/protocol.json)
- [Follow-up code](../results/verified-grounding-planner-20261026/full_path_followup.py)
- [Follow-up evaluator and controller audit](../results/verified-grounding-planner-20261026/evaluate_followup.py)

## Reading the evidence

The dashboard links each episode's `tool-events.jsonl`: these are the exact
numerical requests and replies. There are no Luna prompts or model outputs in
this experiment. Within an episode, the public `robot_data/attempt01/obsNNN`
folders hold RGB-D captures, while `attempt01/NNN-arc` and `NNN-grasp` hold
planning receipts. The observation number follows the artifact/event ID,
not the physical control-step count.

Timing is hierarchical. Operation wall times add to service time. Grasp
generation, selection, IK and contact probes are nested inside `plan_grasp`;
approach-path queries and their dense resampling are nested inside selection.
Image capture, tracker initialization and pose fitting are nested inside
execution. Adding all displayed levels would double count work. Trial wall
time additionally includes setup, final cleanup and small gaps between calls.
The two 90° queries add diagnostic overhead and are not the primary policy's
executed arc plans.

cuRobo v2 supplies the current arc/goal backend, but grasp approaches still use
the CPU MC-VAMP path planner. Its `resampling_seconds` includes dense sampled
configuration validation against the observed scene; it is not GPU IK time.
The word "native" in those C++ planner receipts does not mean simulator object
ground truth. Simulator geometry and contact truth appear only in explicitly
labelled post-run evaluation files.

A grasp being "admitted" means an approach and a virtual 3° contact probe
passed numerical checks. It does not establish that the actual closed fingers
formed a stable grasp. Similarly, `start_collision` in a candidate record
usually describes the proposed grasp configuration, not the current robot
configuration. A failed finite search is not a proof of infeasibility.

## Notes for future agents

Preserve this cohort and its source/input hashes. Use a new registered result
directory for any runtime change or new physical trial. Do not use native
evaluation coordinates to repair annotations, prompts or learner memories.
Human-reviewed pixels are assisted diagnostic inputs, not an automatic
perception capability. The original opening reference is deliberately not
labelled as verified geometry.

The recovery budget code and execution interface currently use different
timing assumptions: selection estimates speed 0.30 rad/s and 60 hold controls;
the diagnostic executes at 0.35 rad/s and 6 hold controls. The secondary probe
removes only the 200-control manipulation reserve. A rejected approach that
would fit the actual timing still needs its skipped contact probe and physical
execution checked; the timing discrepancy alone does not certify recovery.

The preserved results support future engineering work without changing Luna's
practice memory. Keep task-specific native grading separate from any generic
procedural lesson later tested on held-out task states.
