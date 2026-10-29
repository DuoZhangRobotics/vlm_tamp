# Recovery pipeline review, 2026-10-21

This post-study review examines why the repaired pipeline still fails. It adds
no robot trials and changes neither numerical tools nor the evaluated memory.
The 3/4 versus 1/4 transfer result remains a small same-asset pilot.

## What the evidence establishes

**Practice tests repeatability more directly than recovery.** The stopping rule
in `src/vlm_tamp/episode_persistence.py::practice_continuation_decision` accepts
five completed attempts, a verified pair of consecutive successes, and a summary
with a method and stopping reason. The session uses one initial state. It does
not require successful recovery, varied development states, or tests of each
lesson's conditional action. The actual run had one failure followed by four
successful repetitions. Calling this a provisional completion baseline is
accurate; treating it as demonstrated recovery competence would not be.

**The memory schema is useful, but its claims are not behaviorally validated.**
The practice prompt asks for applicability, failure signals, conditional actions,
evidence and limits. Luna's frozen memory includes those fields. Its recovery
guidance remains broad: reobserve, revise the model, and obtain another admitted
arc. It does not establish which observations distinguish a wrong model from
ineffective contact, or demonstrate a successful remedy for either condition.
`recovery_memory.abstract_lessons` checks representation and excludes historical
geometry; it does not verify the causal explanation or whether a lesson improved
a later decision. The four successes support the staged method in the practiced
scene, not every explanatory statement attached to it.

**The remaining technical concern is uncertain object state during recovery.**
`RobotService.issue` explicitly labels hand-motion fallback as a prediction, but
also transports the active planning geometry using that prediction when visual
motion fitting fails. This is an existing assumption, not a newly isolated code
defect. It can work: the state 41 memory run made real progress using fallback.
It can also be wrong: state 45's later hand motions advanced the model while the
door barely moved. Public depth-support feedback already exposed strong
disagreement. The outstanding question is how reliably the actor can reconcile
that disagreement using the existing observation and model tools.

## State 45: separate the events from the explanation

The first approach reached its requested TCP pose within 0.36 mm. This rules out
a large initial tracking miss, not an ineffective grasp. After closure, the first
arc predicted 29.58 degrees of object motion; evaluator diagnostics measured
22.30 degrees. Visual fitting failed. The public response reported 47.5% moving
geometry support and 38% multi-view free-space conflict. Contact-reference
geometry had 27.6% support and 59.3% conflict. These are geometric diagnostics,
not contact certificates.

The next re-grounding (call 11) changed the estimated horizontal hinge-to-contact
radius from about 201 mm to 14 mm and reversed the axis sign. The hinge estimate
itself moved 21.69 mm from its initial value. Call 12 then used the initial
contact anchor's x coordinate as the hinge x coordinate, moving the hinge
198.27 mm. Luna corrected that coordinate at call 18, but the newly grounded
contact anchor remained near the hinge. These observations warrant investigating
contact-region identity and consistency across model replacement. They do not
justify an object-specific radius cutoff or establish the physical truth of
either fit.

Luna's final findings describe the re-grounding itself as moving the hinge by
about 20 cm. That explanation conflates the refit with the subsequent mistaken
model edit. The original notes are preserved. They are evaluation-case notes,
not part of the earlier frozen transfer memory, but demonstrate why future
memory promotion needs evidence checking.

Later, calls 46, 59 and 66 predicted approximately 20, 10 and 10 degrees of
motion. Native door changes were 0, -0.0234 and -0.0060 degrees. Public contact
geometry conflict fractions were 100%, 94.5% and 93.75%. Luna attempted fresh
grounding, surface contact and alternate approaches, but did not restore useful
opening. The three recorded failed grasp searches reported approach failures
or start collisions; their records do not support blaming the fixed manipulation
step reserve for those particular failures. The precise causes of those planning
rejections remain unresolved.

`refined=true` is another interpretation hazard: in point mode it relaxes the
legacy wrist-excursion allowance. It does not estimate a hinge or certify
contact. The tool documentation states this, but the learned phrase "refined
arcs" should not be interpreted as evidence of improved geometry.

## Smallest next sequence

1. Diagnose the retained state 45 observations and planning scenes at the first
   mismatch and failed reacquisition. Test whether ordinary RGB-D can identify
   the moving/contact region consistently and support a current scene. Use
   native geometry only to evaluate the diagnosis. Compare a successful case.
   Repair a tool only after isolating a generic defect on matched inputs.
2. Freeze any resulting tool version. Expand development practice to include
   known failure cases. Record whether Luna recognizes the mismatch and restores
   useful physical progress. Consecutive clean completions remain a separate
   competency. Luna must choose and test its own remedies; do not supply a
   state-specific recovery sequence.
3. Promote conditional lessons with an observable trigger, a tested response,
   an expected observation, supporting and contradicting attempts, and an
   explicit untested status where evidence is missing. Check explanations
   against tool records without turning evaluator geometry into agent advice.
4. Freeze the resulting agent-authored memory and compare it against no new
   memory on newly predeclared states with the same tools and budgets. State 45
   is now development data. Repeat samples before claiming reliability.

The immediate objective is to distinguish failure recognition, scene correction,
contact recovery and task completion. More repetitions of the current successful
starting state alone cannot establish those abilities.

## Evidence and preservation

[Derived evidence and source hashes](../results/tool-repair-learning-20261021/pipeline-review/evidence.json)
records the exact calls above. Original [tool events](../results/tool-repair-learning-20261021/study-v1/transfer/state045-with_learned_memory/tool-events.jsonl),
[Luna findings](../results/tool-repair-learning-20261021/study-v1/transfer/state045-with_learned_memory/agent-workspace/findings.txt),
and [evaluator diagnostics](../results/tool-repair-learning-20261021/episode-diagnostics-private.json)
remain unchanged. Native angles and contact records are evaluator-only.
Retained state 45 failure observations include steps 277 and 495, with calibrated
depth; saved commands permit explicitly labeled replay if additional frames are
needed. They are not new agent trials.

The source-hash check found only the previously documented reporting-interpreter
exception to the original freeze. The evaluated memory hash remains
`65f9612d3f66380234d78f3e05c6342f784b3f137ac4df8032f7fb04ba0c13c2`.
No new physical validation or improvement in recovery is claimed by this review.

## Subsequent private diagnostic

The [completed checkpoint comparison](checkpoint_diagnostic_20261021.md) tested
Luna/high and Sol/high on six saved public prefixes, twice each, without robot
motion or additional task hints. Both understood prediction/measurement limits.
The clearest difference concerned validating a recovered object's identity and
tracing an incorrect model edit. Sol identified both issues consistently at the
two related state 45 checkpoints; Luna missed or partly addressed them. This is
a specific development target, not proof that a replacement model would recover
physically. Answers and evaluator diagnoses remain excluded from practice memory.
