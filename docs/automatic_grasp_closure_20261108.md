# Automatic physical closure for grasp execution

Previously, `plan_grasp` reserved closure controls but stored only an open-hand approach.
Luna had to issue a separate `gripper(close=true)` call after `execute`. The earlier successful
operator bowl check explicitly issued that call; the failed baseline Luna bowl run did not.

Now `plan_grasp` stores `close_steps`, and `execute(plan_id)` runs the approach followed by
physical closure after measured arrival. Planning itself makes no motion. Closure holds the
existing arm reference. No lift, transport, pose adjustment, planner fallback or extra collision
validation was added. Ordinary pose/contact/arc plans retain their previous execution behavior.

`execute(..., close_gripper=false)` requests approach-only execution. The existing `gripper`
parameter describes the approach phase; it does not disable the final automatic closure.
Manual gripper commands remain available. Request schema v25, the actual Luna instructions,
compact tool index and detailed help describe this behavior.

The executor retains the existing approach-arrival criterion and episode limits. A blocked
approach does not close at the wrong location. If fewer closure controls remain than requested,
it uses the available controls and reports incomplete closure. Runtime errors propagate.
The response preserves separate approach/closure artifacts, issued controls, observations and
grasp evidence. `grasp_execution` distinguishes closure completion from unverified retention.
Both feedback formats preserve those distinctions; the dashboard counts automatic and explicit
close requests separately.

## Physical qualification

Artifacts: `results/automatic-grasp-closure-20261108/watch.html`, `protocol.json`, `summary.json`,
`execution-evidence.json` and `trial/tool-events.jsonl`. The executed script is preserved there.

One operator component test reused the original Luna bowl surface selection and grasp request
from `luna-compact-transition-20261108/trials/03-place_bowl-s001-compact`, events 5 and 6.
Both initial native camera images matched the source recording exactly. No new model calls or
separate close request were made. The subsequent 12 cm lift was an explicit operator diagnostic,
not an automatic grasp stage. Source hashes remained unchanged throughout execution.

| Stage | Controls | Measured result |
| --- | --- | --- |
| Open-hand approach | 60–171 | Goal reached; 0.174 mm position error |
| Automatic closure | 171–191 | All 20 controls issued; pad gap 78.75 → 3.88 mm |
| Separate upward retention probe | 191–272 | Goal reached; 2.692 mm error; bowl visibly lifted |

The evaluator recorded nine robot/bowl contacts at the end of the lift. These native contacts
were examined only after execution and were not supplied to Luna or planning. This is one local
pickup qualification, not a fresh Luna evaluation, a repeatability result, or successful placement.

Total wall time was 53.21 seconds: reset 18.99 s, selection 0.41 s, grasp planning 15.43 s,
combined approach/closure 7.92 s, lift 6.22 s; remaining time was setup/recording/cleanup.
The video uses 20 Hz physical timing: approach 3.00–8.55 s, closure 8.55–9.55 s,
lift 9.55–13.60 s. It is saved as `trial/attempt01/initial/rollout.mp4`.

The job used a 32 GiB memory cap and no swap. Peak sampled job memory was 4.43 GiB;
host available RAM stayed above 70.73 GiB. Maximum sampled host full-I/O pressure was 0%,
SSH probes succeeded, and no OOM kill occurred. GPU 3 returned to its pre-test allocation;
the test service is inactive with no remaining process.

## Checks and limits

97 focused tests passed, covering inherited timing, automatic/optional closure, blocked arrival,
partial budgets, terminal episodes, unrelated plans, schema validation, both feedback formats,
and executed-stage reporting. Scoped Ruff and `git diff --check` passed.

No new full-task Luna batch was run. Bowl placement and drawer contact acquisition remain
separate unresolved issues; this change removes the omitted-close failure mode only.
