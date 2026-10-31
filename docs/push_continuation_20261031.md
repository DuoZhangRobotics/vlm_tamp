# Continued finger pushes

Historical implementation notes. The current simulation push policy and direct
hand primitive are documented in [physical_push_execution_20261031.md](physical_push_execution_20261031.md).

`continue_push(direction_world, distance_m)` retains the last reached push contact
and observed scene. It does not need another image region, hinge estimate, preview,
contact search, or full scene build. Its default is one complete requested hand
stroke. Object motion is independently measured, not assumed from hand motion.

An unchanged acquisition preview is also reused by execution. Initial preparation
uses an exact faster voxel grouping and bounded radius queries. Identical contact
targets share IK results across alternative approach distances.

After each stroke, accepted RGB-D rigid motion transports the moving part. Static
points and their native collision index are reused. Native template history stays
at two entries; shared ownership pins live indexes across eviction. No extra
padding or dense post-planner validation was added. Actual current-robot overlap
filtering remains enabled; hypothetical future poses never clear obstacle points.

Another motion operation invalidates the contact; observation alone does not.
Unresolved tracking invalidates continuation. Two ineffective strokes return
control but retain the measured scene for an explicit change of direction,
distance or speed; low progress alone does not prove contact loss. A
bad hinge fit can stop joint-linked motion while preserving independently measured
part pose for direction-only continuation. Grasp/joint-model planning still needs
its own current grounding. These changes assume the retained background is static;
unseen changes elsewhere are not reconstructed by continuation.

## Validation

Three matched closing-state-49 physical replays each acquired fingertip contact
and made two 30 mm continuations. All six continuation calls completed within the
existing 3 mm hand-position tolerance. They used zero scene builds and zero contact
searches. Mean per-call wall time was 5.253 seconds:

| Component | Mean seconds |
| --- | ---: |
| Stroke planning, including IK/collision checks | 1.309 |
| Execution and capture | 3.207 |
| Tracker initialization | 0.333 |
| Current-robot point filtering | 0.150 |
| Motion tracking and moving-part update | 0.057 |
| Other bookkeeping and speed measurement | 0.194 |

Measured hand travel was 27.307 / 27.333 mm; observed object travel was 28.578 /
28.777 mm. Local repeatability does not demonstrate full task success. Recorded
public acquisition requests were used only for the diagnostic, never supplied to
fresh Luna actors. The earlier split-stroke prototype took about 17.6 seconds and
accumulated controller lag; one requested stroke removes those repeated captures
and plans. The former implementation's late-state acquisition could spend over
20 seconds rebuilding the scene alone; that is not a matched full-task speedup.

Regression suite: 156 passed, 7 skipped due to unavailable historical fixtures;
updated dashboard checks: 10 passed. Physical diagnostic reports and video:
`results/push-continuation-physical-r3-20261030/watch.html`.

## Fresh Luna evaluation

`results/luna-push-continuation-rerun-20261031/watch.html` records three repetitions
each of opening states 13/37 and closing states 17/49. Actors start clean, receive
generic tool documentation, and have 600 controls / 1800 seconds. No actor notebook
is fed to the next actor. The previous frozen cohort completed at 7/12 success;
keep it separate from the new implementation.

Run via `scripts/supervise_luna_dashboard_study.py --stage baseline-rerun`, starting
with closing state 49. GPU and EGL are both device 3. The service has a 16 GiB memory
limit, no swap, whole-job OOM termination, and 2-second pressure monitoring. A stable
completed first trial qualifies the remaining batch; task success is not required
for resource qualification. Prepend `/common/home/dz323/.local/bin` to PATH for the
ChatGPT-authenticated Luna CLI. Rebuild the native adapter after merging the C++
change; freeze source/native hashes before starting actors.

The dashboard remains static and loads small per-run data on request. It includes
per-call push timings, reuse counts, task outcomes, videos, and memory/I/O evidence.
Manual display updates remain the default; Live Preview does not reload the page.
