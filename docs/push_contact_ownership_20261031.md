# Failed continuation: contact role versus collision padding

Historical diagnosis before the user-authorized physical-push execution policy.
Current behavior is documented in [physical_push_execution_20261031.md](physical_push_execution_20261031.md).

The first fresh Luna trial on closing state 49 ended unsuccessfully at 600
controls. The following opening trial was stopped by Codex for diagnosis; it is
an operator interruption, not a scored task failure. The initial run was stable:
376 resource samples, 5.79 GiB peak job memory, 0.49% maximum full I/O pressure,
zero memory pressure, and at most 2.004 seconds between samples.

## Confirmed defect

Event 28 reused contact and scene correctly, but its 1 cm continuation was blocked.
The reported robot name `gripper0_finger1_collision` does not identify which
clipped portion of the mesh rejected a point. A recorded-state diagnostic showed
that a direct 5 mm hand translation met a stationary point in the **distal**
finger region. That point projects inside Luna's selected contact rectangle and
lies only 3.62 mm from the initial contact cloud. Sparse single-view nearest-point
association had left part of the selected thin door surface in the static role.
The intended fingertip contact exemption therefore did not apply.

The correction associates the entire observed cloud with the caller's projected
contact and target regions, retaining the existing 15 mm local-plane band and
45 cm extent. This applies to direction-based push acquisition; explicitly
joint-associated acquisition retains its existing model association. No obstacle
points are deleted, and no future robot pose determines the assignment. The
moving part and contact surface are retained for subsequent pushes as before.

For the recorded scene, total observed endpoints remain 263,776. Moving endpoints
increase from 8,504 to 10,938; contact references from 1,210 to 1,642. The previously
blocked 1 cm stroke now produces a five-state plan using the same solver and
tolerances. Three matched physical replays all produced 7.147 mm hand travel and
6.624 mm observed object progress, with no scene build/contact search in continuation.
Their matched initial joint error was zero. Tool times were 6.47–6.86 seconds;
peak measured job memory was 4.64 GiB, maximum host full I/O pressure 1.95%, and
memory pressure zero. Regression checks: 126 passed, four historical-fixture skips.
This is a local repair measurement, not proof of complete task closure.

## Separate failures

- Event 17: two 10 mm commands produced 0.968 and 0.958 mm measured object motion.
  The existing 1 mm effective-progress threshold returned control after two
  strokes. The old code also discarded the contact scene. It now retains a
  successfully tracked scene so Luna can explicitly try another direction,
  distance or speed without repeating acquisition. The return condition remains.
- Event 23: six visible correspondences; the corrected target association gives
  seven. Lowering the fit's minimum count did not give reliable motion: diagnostic
  held-out errors reached 21–47 mm. Tracking thresholds are unchanged. A tracking
  failure remains a real limitation, not evidence of physical infeasibility.
- Event 9: Luna supplied an unresolved image-motion hint, and the adjacent-depth
  heuristic could not assign the hinge direction. Direction-only pushes do not
  require that hinge model.
- Contact acquisition failures included arm/palm endpoint collisions. These are
  separate from the confirmed distal-contact labeling error; they are not all
  declared fixed by this change.

The legacy stroke planner still constructs Cartesian waypoints through cuRobo IK
and native collision checks, including its existing 6 mm path-deviation condition.
Correcting surface roles resolves the reproduced blockage without changing those
limits. Do not describe it as a cuRobo MotionGen plan, and do not attribute every
failure to Luna or to a strict IK tolerance.

Diagnostic inputs and code are evaluator-only. They must not be fed to fresh
Luna actors or transferred as successful coordinates. Generic tool documentation
describes role consistency and continuation/recovery; learner notebooks remain
isolated between clean evaluation actors.

The durable explanation, red-point image, videos, measurements and diagnostic code
are in `results/luna-push-continuation-rerun-20261031/contact-diagnosis/index.html`.
The corrected fresh study is `results/luna-contact-ownership-rerun-20261031/watch.html`.
