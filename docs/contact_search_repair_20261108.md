# Contact candidate search repair

The premature-abort software defect is fixed. The recorded drawer request still does not produce
a successful physical push; this work is not a claim that manipulation reliability is restored.

## Retained change

`plan_engagement` iterates over native cuRobo contact postures. Previously, a finite IK rejection
while connecting one posture's contact approach raised `SkillStop` out of the entire search.
Remaining native postures were discarded. In the reproduced drawer case, nine postures were
returned but the search stopped after considering two.

A finite insertion-IK rejection now returns an explicit empty candidate path to that search,
records its failure, and lets the loop consider the remaining existing postures within the same
budget. Budget exhaustion and runtime/numerical exceptions still propagate. The selected pixel,
contact model, directions, solver algorithms, collision permissions and geometry are unchanged.
No previous solution is substituted for a failed candidate.

The selected-candidate index is now accurate. Query counts include contact-approach IK calls.
Public summaries distinguish a solved free-space route from a failed contact approach and retain
the failed step's native feasibility/position evidence.

## Evidence and limits

The reproduction uses the initial drawer-state-1 public request from event 5 of
`luna-compact-transition-20261108/trials/02-close_top_drawer-s001-existing`. No replacement contact
or closing-direction label is supplied.

Before the fix, the local IK placed the point accurately but native feasibility rejected palm
collisions. Native self-collision costs were zero and the returned joints were within bounds.
After the retained loop correction, all nine returned postures are considered. Eight fail their
precontact query; the remaining posture fails during its contact approach. Search takes about
4.13 seconds of the 30-second allowance, and no task motion executes beyond reset control 60.
This exhausts the returned candidates; it does not prove global geometric infeasibility.

A regression fixture supplies two native postures: the first has a failed insertion and the
second has a complete path. The corrected search returns the second posture unchanged, instead
of aborting. Separate tests verify that budgets and numerical errors are not swallowed.

## Experiments that were removed

Two additional changes were tested and reverted:

- Enabling the full native cuRobo optimizer for collision-checked continuation, with one supplied
  seed. This produced some additional solved previews but no reliable physical acquisition.
- Solving the initial stand-off as a full pose preserving the selected contact orientation.
  The fixed request still failed its contact approach.

Neither change remains in the code. The original fast continuation method is retained. In the
full-optimizer experiment, one executed wrist-view request reached its free approach but stopped
34.3 mm short during contact acquisition, at control 255. It did not reach a push stroke.
The free approach occupies video 3.00–10.90 seconds and contact acquisition 10.90–12.75 seconds.
The request used the default 120 skill seconds and retained 112.64 seconds at return;
345 of the 600 control steps remained. This stop was physical arrival failure, not budget expiry.
[Recorded failed execution](/common/users/dz323/vlm_tamp/results/contact-acquisition-repair-20261108/batch/attempt01/initial/rollout.mp4)
is an experimental variant, not the retained implementation. Other failed previews issued no
manipulation motion. All attempts remain under
`results/contact-acquisition-repair-20261108`.

Validation: 108 focused tests passed, including candidate exhaustion, remaining-candidate success,
budget/runtime propagation, contact handoff, existing continuation behavior and compact reporting.
Configured Ruff checks and `git diff --check` passed. No full-task Luna batch was run. No native
build was required. The unresolved repair target remains physically usable contact acquisition.
