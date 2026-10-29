# Shared robot request contract repair

The unclear grounding contract affected the shared service used by Luna and
Qwen. The generic operation/arguments wrapper was specific to the new local
Qwen adapter. Both are repaired. Luna remains the default actor.

[Inputs, outputs, annotations and timings](../results/robot-io-contract-20261024/watch.html) ·
[Machine-readable findings](../results/robot-io-contract-20261024/summary.json) ·
[Change patch against the previous working files](../results/robot-io-contract-20261024/changes.patch).

## Shared behavior

`src/vlm_tamp/robot_contracts.py` defines the public operation schemas. The
service validates requests before numerical work. Invalid names, missing or
unexpected arguments, wrong types, non-finite numbers, empty/inverted boxes,
out-of-image coordinates, coincident hinge points, and zero direction vectors
produce a structured `request_diagnostic`. This explicitly distinguishes
rejected input from an unsuccessful geometric search. Compact replies preserve
the diagnostic.

The grounding help now defines original image pixels, image axes, box order,
and the intended initial movement of the selected contact point. This is not
camera orientation. A supplied direction must be finite and nonzero; its
magnitude is irrelevant. `null` explicitly declares an unresolved image hint.
The old implicit `[0,1]` default is removed. Pixels are neither silently clipped
nor automatically rescaled.

The current live tool normally determines the final opening direction from
adjacent stationary depth geometry, overriding the image hint. This inherited
rule remains in place and its source is now visible in compact feedback. A
null image hint can succeed when depth resolves the direction. A zero vector
is invalid, and a projection tie no longer silently selects a sign. Without
the required evidence, grounding reports an unresolved direction.

This distinction corrects an earlier incomplete explanation: the zero-vector
fallback was a real defect, but its final effect on Luna's live trials was not
established because the depth rule normally overrides it.

Observations include image sizes and an explicit pixel-coordinate convention.
The compact view includes those sizes for its preview images; full replies
retain all cameras. Each prepared workspace contains `tools/schemas/` and
`tools/schema-index.json`. Luna can inspect a schema without a robot RPC:

```text
python3 robot.py schema ground
```

## Qwen adapter

The local actor now calls named robot functions with the same schemas enforced
by the service. The generic `robot(operation, arguments)` function is removed.
The actor initially receives the small common tool set; `robot_help` or reading
an operation's help enables that named function. The entire schema catalog is
not appended to every model request. Numerical Python access remains available
inside the existing restricted public workspace.

Malformed JSON is also a request rejection; it does not become a robot or
planner failure. Validation does not correct the model's spatial choices or
select an action for it.

## Validation

- 74 focused and integration tests passed; three tests requiring an unavailable
  older calibrated fixture were skipped. A retained current fixture was used
  for the separate geometry regression below.
- A real OS-sandbox preflight verified schema/help access, named-function
  activation, the status wrapper, and rejection before RPC. Private project
  implementation and evaluation data remained inaccessible to the actor.
- All 359 historical Luna requests from the prior comparison were checked.
  All 297 previously accepted tool requests remain valid. Two malformed
  requests that already failed are rejected precisely: a missing `regions`
  wrapper and a missing `degrees` argument. The other 357 requests pass the
  new structural contract; this is not a claim that their plans or hypotheses
  were correct.
- A saved valid Luna grounding request produced identical hinge, axis, anchor,
  contact cloud and mask before and after the repair. The explicit-null case
  selected the same final axis from depth. Extreme finite direction scales
  were also tested for invariant sign selection.
- The five complete but invalid Qwen grounding submissions from the previous
  pilot are rejected by the shared validator before geometry work. Their
  combined warm validation time was under one millisecond in this check.

The calibrated regression fixture initially lacked robot-prior provenance and
the mechanism template. The fixture was corrected using the normal loader,
retained original metadata, exact metric-array/calibration comparisons, and
the normal revolute template. No provenance or geometry checks were weakened.

## Fixed-image Qwen retest

The model, checkpoint, BF16 runtime, two RGB views, temperatures, seeds and
4,096-token output budget were held fixed. The changed package consists of the
clarified contract and exact function schemas. Two approach queries and six
grounding queries were run; an invalid request could receive one logged
schema-correction turn. All attempts are retained.

- Both open-finger approach queries remained correct.
- Three of six grounding queries produced valid requests on the first try,
  versus zero of six meeting the earlier gate. The newly documented null
  option changes the contract, so this is compliance evidence, not a claim of
  improved physical reasoning. In this sample, completed grounding calls used
  nonzero direction vectors.
- The other three queries exhausted the output limit without a complete call;
  their correction turns also exhausted it. These are generation-budget
  failures. No malformed call was sent to numerical geometry.
- The three valid proposals were subsequently checked against the saved RGB-D
  using the existing geometry implementation, without feedback to the model.
  One could not resolve opening direction from observed cabinet bulk. One
  selected a panel region with no supported upright plane. One fit was admitted,
  but its hinge line was **88.0 mm** from the native line in a private audit.
  An accepted fit therefore still does not establish a correct mechanism.

No robot episode, fine-tuning, learner-memory update, stronger model, or
Reflection reviewer was used. The cases are reused development images of one
asset; these results do not establish transfer or improved Luna task success.
The shared fixes are complete. Visual localization and the model output budget
remain separate issues.

## Artifacts and reproduction

The result directory retains the original working-file snapshots, final patch,
test logs, historical-request audits, sandbox preflight, model requests and
outputs, image hashes, numerical fits and timing. The model server is stopped.

Use `scripts/retest_qwen_robot_io.py` with a new output directory for the repaired
interface. The older `run_qwen_capability_probe.py` deliberately reproduces the
old diagnostic inputs and wrapper. `scripts/validate_robot_io_geometry.py`
performs the saved-data regression and observer-only fit checks.

Model-trial source hashes were verified at completion. Subsequent hardening of
extreme-magnitude normalization and malformed-JSON diagnostics changed no
model-facing schema or grounding help; stored complete requests were revalidated
against the final code. The frozen trial source remains available for audit.
Do not copy evaluation coordinates or native hinge positions into learner memory.
