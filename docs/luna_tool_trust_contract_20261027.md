# A dependable toolbox for Luna: boundary and readiness review

Luna should learn to solve tasks using a stable toolbox. Engineering changes
improve that toolbox; learning claims require improvement with the toolbox
held fixed. This review records the boundary for subsequent work. It is
developer documentation, not an instruction or answer guide injected into Luna.

## Responsibility

| Owner | Responsibility |
| --- | --- |
| Tool implementation | Coordinate transforms, units, numerical estimation, reproducible artifacts, explicit assumptions, failure reporting, timing and documented side effects |
| Luna | Select a target/part, request evidence, choose experiments and actions, interpret outcomes, recover, and write evidence-backed procedural memories |
| Evaluation | Keep answer geometry separate, score accuracy and task outcomes, detect false acceptance, and distinguish engineering gains from learner gains |

Generic optimizers and candidate comparison can belong inside a tool. They
need a declared scope and evaluation beyond the case that motivated them.
Manually supplying a difficult case's pixels, axis or recovery sequence is
operator assistance and must not count as autonomous success. Changing a
threshold after seeing that case's answer is development, not validation.

A dependable tool may return insufficient evidence. Its value is that Luna can
rely on the meaning and measured limitations of its output, not that every
request yields a usable answer. Repeated abstention must still be measured:
rejecting everything is not a useful pipeline.

## What is supported today

| Layer | Evidence | Limit |
| --- | --- | --- |
| Software mechanics | Request validation, recorded observation IDs, checksum guards on loading, coordinate tests, retained artifacts, read-only perception effects and regression tests | Tested behavior is not a guarantee for every input or a perception accuracy claim |
| Perception | Luna selected four accurate, motion-supported hinge estimates across the latest development passes on five recordings | One microwave asset; staged changes; no independent success-rate estimate |
| Motion verification | Fixed-axis prediction residuals on a later interval | Offline CoTracker sees the whole supplied clip; this is not independent future evidence |
| Model scope | Explicit upright revolute hypothesis and geometric admission gates | World-Z axis is supplied; neither arbitrary joint recognition nor calibrated correctness probability is established |
| Learning and execution | Luna's choices, outputs and candidate notes are retained | This study demonstrated neither memory transfer nor new closed-loop manipulation success |

The [perception study](part_motion_tools_20261027.md) and its
[verification record](../results/luna-part-motion-tools-20261027/verification-summary.json)
contain the evidence and limitations. All five recordings are development data
for future validation. A different initial state of the same object does not
establish generalization to a new object geometry.

## Requirements for the next acceptance evaluation

1. **Declare scope and freeze before scoring.** Pin tool/model versions,
   thresholds, contracts, prompts and resource budgets. Specify allowable
   geometric error, acceptable false-acceptance rate and required coverage for
   the intended use before looking at results. The current small sample cannot
   justify a reliability percentage.
2. **Use unfamiliar objects and failure conditions.** Separate development and
   acceptance sets by asset/geometry. Include low motion, occlusion, misleading
   masks, calibration error, failed contact and mechanisms outside the declared
   scope. Measure both useful estimates and correct abstentions.
3. **Keep Luna in charge.** Use the same generic tools and instructions for all
   scored cases. No manual masks, case-specific hints or answer-based repairs.
   Native geometry remains grader-only; native task-success feedback can remain
   available during execution under the existing policy. Retain all attempts,
   including abstentions and budget exhaustion.
4. **Test claims with new evidence.** Freeze an estimated model before acquiring
   subsequent validation observations. Track false acceptance, estimate error,
   coverage, reasons for failure, time/calls/controls and actual task outcomes.
   A planner finding a path does not itself verify the object's geometry or
   physical contact. Report component reliability separately from end-to-end
   success.
5. **Keep evaluation honest after debugging.** If a scored failure leads to a
   method change, move that case into development and obtain fresh acceptance
   cases. Preserve prior versions and failures instead of pooling them into an
   apparent improvement rate.
6. **Measure Luna's learning separately.** With the model, tools and prompt
   fixed, compare a fresh start against Luna-authored practice memories on
   separate tasks/assets. Memories should record reusable observations,
   evidence and limitations. Keep coordinates and observation/part IDs scoped
   to their episode. Operator corrections are engineering gains. Candidate
   notes are not automatically promoted to the learner guide.

The output contract should expose provenance, units/frame, assumptions,
measurements, failed checks, side effects and timing in compact form, with
larger artifacts on demand. A SAM score, bootstrap spread or residual is a
specific measurement, not an interchangeable confidence probability. Existing
tools implement parts of this contract; calibrated reliability remains open.

## Change made during this review

In [`PerceptionTools._view`](../src/vlm_tamp/perception_tools.py), the checksum
loop reused the observation/camera cache-key variable. Every tuple lookup
therefore missed and observations were loaded repeatedly. Separate variable
names restore the intended bounded cache. Cache hits reuse the already loaded
snapshot; checksum validation still runs when loading a snapshot.

This was unnecessary I/O, not evidence of wrong-frame reuse or the cause of
the unresolved hinge estimate. No accuracy or end-to-end speedup is claimed.
Two regression tests cover repeated reuse and separation across cameras and
observations. The focused tool, schema and compact-interface suite passed
**51 tests**; see [test output](../results/tool-trust-review-20261027/tests.log)
and [review manifest](../results/tool-trust-review-20261027/review.json).

No new actor experiment, candidate-ranking policy, estimator threshold or
learner-memory promotion was introduced by this review. The next evidence
needed is a frozen evaluation on unfamiliar assets, followed by a separate
memory-transfer comparison with the toolbox held fixed.
