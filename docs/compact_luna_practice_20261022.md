# Compact interface and continuous Luna practice — 22 October 2026

Later backend evaluation: [five adaptive Luna successes with integrated cuRobo continuation](luna_curobo_continuation_practice_20261022.md).
The original compact-interface study below remains a separate historical run.

Status: complete. All five practice attempts succeeded; profiling and cuRobo proposal diagnostics are retained.
The user requested a compact interface, an explanation of latency, and a return to
Luna's own continuing practice without a separate Reflexion reviewer.

## Task and protocol

The active task is LIBERO-90 `KITCHEN_SCENE7_open_the_microwave`, nominal scene,
initial-state index 45, seed 0. State indices are initial configurations of the
same task, not different task families. State 45 is already known development
data. This run begins each attempt from a clean reset; it does not replay the
277-step failure checkpoint used by the preceding Reflexion study.

One Luna/high session has five attempts, each capped at 600 control steps and
1,800 seconds, with a 5,400-second total session limit. The native success flag
stops motion immediately. Failed attempts continue to a genuine budget limit.
The same learner retains its conversation and writes its own plans, findings,
working state and candidate guidelines. There is no separate reviewer call,
reviewer-provided reflection, or model-weight training. Earlier Luna-authored
guidance from `luna-scene-motion-v1` and the existing abstract numerical example
remain explicit inputs. No previous complete episode packet is supplied.

[Registration](../results/compact-luna-practice-20261022/practice-v1-registration.json)
and [exact prompt](../results/compact-luna-practice-20261022/inputs/practice-v1-prompt.txt).

## Completed practice results

| Attempt | Native success | Control steps | Attempt elapsed |
|---|---|---:|---:|
| 1 | True | 352 | 413.0 s |
| 2 | True | 352 | 281.5 s |
| 3 | True | 352 | 304.5 s |
| 4 | True | 324 | 205.7 s |
| 5 | True | 324 | 232.5 s |

Luna preserved its successful baseline, tested an initially rejected 45-degree
proposal, then found and repeated a 35/30/15/5-degree sequence. Attempts 4 and 5
used 324 control steps and six executed trajectories, compared with 352 and
seven for attempts 1–3. These are Luna's same-scene findings, not a universal
recipe. The supplied prior guidance, repeated scene, warm tools and changing
policy prevent attributing all timing differences to compact presentation alone.

The actor session lasted 1,665.092 seconds. Recorded robot calls used 1,205.086
seconds; the remaining 460.006 seconds include model inference, local analysis,
file work and coordination, which cannot be separated retrospectively. The same
Luna thread used a second turn only to place/format its final notes at the required
root paths. No separate reviewer or additional physical attempt was invoked.

Actual compact presentations reduced JSON bytes by 56.6% across 86 displays.
All 85 nonlocal reply presentations matched canonical backend records; local
status feedback retained its full file. All five controller-reference audits
passed, with zero post-success motion and zero execution vetoes. All 500 source
hashes remained unchanged. The final audit passed 207 checks; these establish
integrity and execution properties, not generalization or causal learning benefit.

[Final audit](../results/compact-luna-practice-20261022/final-audit.json),
[learner synthesis](../results/compact-luna-practice-20261022/practice-v1/agent-workspace/practice-summary.json),
and [candidate memory bundle](../data/agent_memory/luna-compact-practice-state45-v1/README.txt).
The bundle copies Luna's guidelines exactly and has not replaced the old default.

## What changed

- `--compact-interface` creates a short `tools.txt`, preserves the entire manual
  as `tools/full.txt`, and provides local `python3 robot.py help OPERATION`.
- CLI replies and Python `brief(...)` return selected fields and a
  `full_response` file containing the exact original reply. Reading that file
  requires no additional robot action. `call(...)` still returns full data for
  numerical analysis. `--full` explicitly requests a full CLI reply.
- Observations preview two RGB paths. Other cameras, joint states, depth arrays
  and calibration remain available. Backend multi-camera processing is unchanged.
- Compact replies retain native success, budgets, errors, planning diagnostics,
  measured-versus-predicted motion provenance, and phase instructions. Detailed
  point clouds and per-camera numerical diagnostics stay in linked artifacts.
- Arc queries report their configured observation-derived hinge/axis, requested
  angle, constraint mode and geometry epoch. These describe existing inputs to
  the solver; they do not add a corrected estimate or native object geometry.
- A generic continuation reminder now uses the registered attempt cap instead
  of always telling the learner it has twenty attempts.

The numerical planning call, collision rules, controller, physics and success
predicate were not changed. An AST comparison confirms `plan_arc` is the only
changed method in `RobotService`, adding description metadata around its
existing numerical call. The previous grasp-budget reservation limitation remains.

## Latency evidence

The successful `study-v2/trial01-reflection` actor ran for 611.7 seconds. Its
post-prefix robot calls consumed 383.7 seconds, including 198.3 seconds for
twelve arc searches. The remaining 228.0 seconds combines model inference,
shell analysis, transport and scheduling. It is not a direct measurement of
reasoning time. Prefix replay cost another 81.3 seconds outside that actor
interval; separate review time is recorded separately.

The paired failed retry used 523.7 actor seconds and 172.1 robot-service seconds.
The first later reviewer-context follow-up used 1,052.3 actor seconds and 476.1
robot-service seconds. The proportions vary: substantial time goes to both the
agent loop and numerical work. This interface change does not make the numerical
searches faster. Eliminating a separate reviewer also removes its additional call.

[All fourteen historical timing records](../results/compact-luna-practice-20261022/historical-timing.json).


Detailed inspection of the first new practice attempt found a wrapper-level
planning bottleneck. Its ten arc queries consumed 175.712 seconds: five successful
queries used 70.685 seconds, and five rejected queries used 105.027 seconds. The
recorded search totals account for 171.020 seconds; surrounding setup, artifacts
and feedback account for the remaining 4.693 seconds.

Within the five successful queries, direct native path-planning calls, including
validation/resampling/process overhead, total only 1.670 seconds (0.228–0.386 each).
The remaining 67.350 seconds inside successful searches cover endpoint generation
and its projection/clearance checks. Their continuation-layer lists are empty.
The five failed searches consume another 102.000 seconds inside the search;
that includes unsuccessful endpoint generation and fallback continuation. There
are no separate timers for every projection, clearance check or fallback edge.

`full_arc_plan` assigns 65% of its 20-second budget to `partial_goals`, which
collects up to eight distinct valid endpoints before trying a global connection.
Four of the five successful requests already accepted seed zero, but continued
sampling. The successful global calls report zero RRT iterations and native solve
times of 0.010–0.016 seconds; the larger wrapper times above include more work.
Thus endpoint collection is a concrete optimization target. Trying a validated
endpoint earlier, with unchanged collision checks and fallback search, requires
a separate matched test; it is not implemented in this frozen practice run.

[Per-call arc breakdown](../results/compact-luna-practice-20261022/attempt01-arc-timing-private.json).

## Validation

The focused suite passed **56 tests**, with one integration test deselected;
the actual sandbox isolation and NumPy/SciPy/Pillow capability probe passed
separately. Tests include exact raw-reply retention, native-success and prediction
provenance, single execution of requests, local help, the unchanged Python call
contract, and agreement between reported arc constraints and solver arguments.

Rendering 930 archived robot replies reduced JSON presentation bytes by **65.1%**.
The largest rendered archived reply was 3,454 bytes. The entry manual decreased
from 16,715 to 2,683 bytes; the new practice prompt is 361 words. These are byte
and word measurements, not model-token or end-to-end speed measurements. Real
usage may differ when Luna requests full replies or prints its own analysis.

[Display audit](../results/compact-luna-practice-20261022/offline-display-audit.json),
[tests](../results/compact-luna-practice-20261022/tests.log), and
[implementation audit](../results/compact-luna-practice-20261022/implementation-audit.json).

## Watching and reproducing

Open [watch.html](../results/compact-luna-practice-20261022/watch.html) through
VS Code Live Preview. It refreshes every fifteen seconds. The final report,
videos, actor-authored notes and timing summary become linked when the session
closes. `robot_data/interface-events.jsonl` records presentation sizes and exact
raw-reply references. Full backend requests/replies remain in `tool-events.jsonl`.
The current CLI log does not enumerate every image presentation or expose exact
active context size; cumulative input usage is not simultaneous context length.

For a new, separately recorded practice session:

```bash
source scripts/server_runtime_env.sh
python scripts/run_compact_luna_practice.py \
  --output results/NEW-STUDY/practice-v1 \
  --state 45 --attempts 5 --agent-seconds 5400 --episode-seconds 1800
```

Use a new output directory; do not overwrite closed evidence. The runner records
its prompt, memory hash, command, source snapshot, outcomes and timing. It does
not launch the old Reflexion runner or automatically promote candidate memory.
Same-scene repetitions do not establish transfer, and this run alone is not a
controlled estimate of the interface's effect on speed or learning.


## Fine-grained profiling and cuRobo diagnostics

The live viewer now contains an observer-only profiling dashboard, with disjoint
service categories and nested original-trace and diagnostic-replay timings.
Headless Chrome rendered the complete data tables and screenshot after the
standard sandbox's Chrome socket restriction was bypassed through approved
execution. JavaScript syntax, timing reconciliation and profile geometry checks
passed. The profile data do not enter the learner's inputs.

Timed saved-query replays retain the original objectives, collision rules, seeds
and time budgets. The saved sensor-array hash and effective geometry counts were
checked. Robot arm volumes were reconstructed from the retained calibrated model
at the recorded joint state, with the recorded gripper volumes retained. There
were no simulator or language-model calls. Timing wrappers can alter the number
of candidates visited before a wall deadline, so replay times are labeled separately.

The detailed successful event-7 replay spent 13.550 seconds generating goals and
0.112 seconds in the global-connection phase. Of that goal-generation time,
10.550 seconds was 956 batches of nearest-neighbor queries for 36 arm spheres
against 253,020 stationary observed points. Palm-neighborhood queries took only
0.217 seconds against that cloud. This identifies a much narrower bottleneck than
"IK is slow": the CPU optimizer repeatedly requests exact obstacle proximity.

The failed event-12 replay spent 13.163 seconds generating goals and 7.113 seconds
in fallback continuation. Stationary-cloud arm proximity queries consumed 10.092
seconds in goal generation and another 4.635 seconds in fallback. Fallback uses
nearby guesses at intermediate <=3-degree arc angles, repeated projection and
collision checking, and native path connectors; it does not move the robot.
The rigid failed query and the successful internal grasp probe were also profiled.

[Full profiling dataset](../results/compact-luna-practice-20261022/profiling.json)
and [rendered preview](../results/compact-luna-practice-20261022/profiling-preview.png).

The user proposed cuRobo IK for goal generation. The existing adapter uses full-pose
IK for grasps, has no scene collision world, and disables posture regularization.
The installed cuRobo 0.7.8 supports partial-pose metrics. Four diagnostic variants
were tested on the ten saved arc queries, followed by the original exact contact
projection, native collision tests, wrist limits and path validation:

| Proposal diagnostic | Originally solved queries receiving an admitted full path |
|---|---:|
| Existing full-pose adapter with transported orientation | 1/5 |
| Position-only GPU proposal + contact polish, eight candidates | 0/5 |
| Above with current-posture regularization | 0/5 |
| Above with all 64 returned seeds screened | 0/5 |

Warm GPU IK calls were typically 10–80 ms; startup and first graph setup were
recorded separately. The single fixed full-pose orientation overconstrains the
nonrigid goal. Position-only proposals often collide or rotate the wrist too far;
the all-seed test also encountered failed connections and a rejected path wrist
excursion. The position-only TCP target approximates the offset contact point;
the existing CPU equality polish then enforces the exact selected contact relation.
These tests do not establish that cuRobo cannot solve the task. They show that
these simple robot-only adapters are not a valid replacement for contact-aware
endpoint generation. No prototype was installed as the production default.

The next planner change should address goal proposal and local/scene feasibility
rather than removing admission checks: use cuRobo with an appropriate contact
frame, posture preference and observed-scene guidance, try admitted goals promptly,
and retain a measured fallback. The exact stationary-cloud proximity bottleneck
and the batching delay are also independent optimization targets.

[cuRobo IKSolver API](https://curobo.org/_api/curobo.wrap.reacher.ik_solver.html)
and [partial-pose cost metrics](https://curobo.org/_api/curobo.rollout.cost.pose_cost.html).
