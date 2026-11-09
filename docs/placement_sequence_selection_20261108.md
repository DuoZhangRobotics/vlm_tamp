# Whole-sequence placement selection

The saved state-1 `place_object` request had 323 controls remaining. Its first feasible lift,
transfer and lower paths together required 1,204 controls at the requested 0.35 rad/s. The lower
path alone required 879 separately scheduled controls. Feasibility at each stage did not imply
an economical complete sequence: the lift already changed the seventh joint by about 156 degrees.
No motion was issued by that rejected request; Luna later recovered through a separate point plan.

`planner_object_transfer.py` now retains up to five low-cost prefixes through lift, transfer and
lower, using the existing native cuRobo goal batches and constrained MC-VAMP paths. Selection uses
the actual joined executor schedule, including the requested hold and release once. Prefixes
already exceeding the remaining control budget cannot become a fitting complete sequence and
are discarded. An empty finite search reports exhaustion, not proven geometric infeasibility.
Runtime failures propagate. No alternate solver, path shortcut, perturbed target, extra clearance
or changed speed is introduced. This bounded search is not a global shortest-path guarantee.

The source-to-TCP relation stays fixed through the prepared transfer. The existing object-axis and
support-plane constraints, collision world and measured gripper aperture remain unchanged. Payload
geometry and physical attachment are still not certified by these constraints. Additional measured
arrival waiting is not predicted in the nominal schedule; the real execution remains capped at 600.

## Qualification

Artifacts: `results/placement-budget-20261108/watch.html`.

- Three-prefix diagnostic: 333 controls; no motion issued.
- Five-prefix diagnostic: 314 controls; no motion issued.
- Production replay: same saved robot state at control 277, zero endpoint joint difference.
  Found the 314-control sequence in 8.59 s (11 native IK batches, 94 path queries, 73 over-budget
  prefixes discarded). Kept the same 8 cm lift, 0.35 rad/s speed, six holds and twelve release
  controls. No additional arrival controls were needed.
- Physical execution: first native task success at 574, full release completed at 591/600 and
  final native success remained true. Entire qualification, including startup and prefix replay,
  took 81.47 s. This is one local numerical qualification, not a fresh-policy success rate.
- 71 focused Python tests passed. Ruff and `git diff --check` passed. An existing dashboard test
  fixture lacked the action log and request arguments now consumed by its execution report;
  the fixture was corrected. Native code was unchanged; no new native build was necessary.

## Frozen fresh-policy evaluation

Predeclared config: `configs/study_placement_reliability_20261108.json`.
Five additional bowl initialization states: 2, 12, 22, 32, 42. Three fresh Luna/high episodes each,
600 controls and the existing 1,800-second episode budget. Two native camera views; no inherited
experience, demonstrations, coordinates, stronger model or task-specific policy. Physical reset
seed remains zero; model sampling seed is not controlled. All attempted cases remain in the report.

The 15 episodes test policy reliability on additional poses of this task, not transfer to unseen
object categories. Their outcomes must be reported separately from matched numerical replays.
Results and videos will be at `results/placement-budget-20261108/luna-five-state/watch.html`.

## Interrupted first batch and two integration repairs

The first batch completed four native successes: state 2 at 364/210/209 controls and state 12
repeat 3 at 317. State 12 repeat 2 was stopped at control 257 after repeated CUDA errors in grasp
preprocessing. It had 343 controls and about 1,132 wall seconds remaining; its grasp requests each
had a 60-second search budget but failed before pose generation or IK. The interrupted attempt is
preserved separately, with its video at
`luna-five-state/trials/02-place_bowl-s012-repeat2/attempt01/initial/rollout.mp4`. Ten planned
subsequent episodes did not run. This is not a completed 4/15 or 4/5 task-success estimate.

The selected wrist region provided 52,928 points after current-robot filtering. GraspGen's native
outlier filter calls `torch.cdist` on the entire cloud against itself: 2,801,373,184 distances.
The same saved input reproduces `CUDA error: invalid configuration argument` in that call.
`pointcloud_graspgen.py` now evaluates the identical L1, K=20, strict 0.014 m mean-distance rule
in blocks of 1,024 query rows, each compared against every original point. No point subsampling,
altered threshold, alternate device or fallback is used. CPU dense-equivalence tests cover
outliers and duplicate points. Recorded-cloud GPU row comparisons pass; all 52,928 points remain.
The full GraspGen pass then returns eight poses in 5.32 s with about 1.36 GiB peak GPU allocation.
Blocked filtering alone uses about 414 MiB and took 2.75 s in the explicit validation replay.

A separate communication defect appeared before that crash: bounded actor history discarded the
bowl ID `object003`, although the service retained it. Decision input 15 contained no object IDs;
input 16 contained only the newly selected plate. Luna guessed IDs and received repeated KeyErrors.
`object_status` now accepts no ID and lists only the caller's previously selected IDs, labels and
pose currentness. Unknown-ID feedback also names the existing selected references. This supplies
no native object inventory, coordinates, role assignment or automatic target selection.

Finally, `release_complete` now uses issued controls, so native success after one release control
is not mislabeled as a complete 20-control release. Physical execution and stopping are unchanged.

The interrupted batch remains an archived development result. A new frozen 15-episode batch will
repeat all five states with the same model, cameras, memory settings, 600 controls and 1,800-second
budget after these fixes; its outcome must be reported separately.
