# Frozen full-task Luna validation and clean practice

The user authorized complete evaluation after the hand/IK/contact pipeline
repairs, followed by clean Luna practice and held-out memory testing.

The declared study is `results/luna-pipeline-validation-20261030`. Open its
`watch.html` with VS Code Live Preview. The driver is
`scripts/run_luna_pipeline_validation.py`; it preserves completed attempts and
can continue one stage at a time without rerunning completed entries.

## Declared protocol

- Luna/high throughout, 600 controls and 30 minutes per episode. CUDA and EGL
  rendering both use GPU 3. Runs are sequential on the shared GPU.
- Regression: three fresh no-memory evaluations each of open37, open13,
  close17 and close49. The first open37 run counts as its first repetition.
- One clean persistent learner practices close17, close49 and open13 twice,
  six episodes total. Baseline evaluator notebooks are not provided.
- The learner writes its own concise procedural guidelines. The parent checks
  only transfer scope/leakage and preserves the text verbatim before freezing it.
- Transfer: open36/open15/close38/close23, each with no inherited memory and with
  the same frozen learner notes. Condition order alternates. These states were
  selected deterministically before outcomes and are excluded from this study's
  practice and regression; they are not claimed globally unseen.
- Evaluators may write notebooks. Their notes are archived and never fed to
  subsequent actors. No teacher/reflection model or task-specific solution is
  supplied. Native object geometry remains offline grading only.
- Source and native-binary hashes are fixed across all phases. Failures remain
  in the record. Unexpected infrastructure exits stop the driver for review.

Preflight corrected stale generic tool documentation that still described the
old advisory collision mode and the discarded 24-pose/standoff search. The
current help now states the eight-pose pipeline, open-hand admission, actual
closure feedback, and how changing the seed requests a new learned proposal
batch. This is supplied tool knowledge, not a successful task recipe.

## Operation

After sourcing `scripts/server_runtime_env.sh`, launch with
`CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3`:

```bash
python scripts/run_luna_pipeline_validation.py \
  --study results/luna-pipeline-validation-20261030 --stage baseline --limit 1
python scripts/run_luna_pipeline_validation.py \
  --study results/luna-pipeline-validation-20261030 --stage baseline
python scripts/run_luna_pipeline_validation.py \
  --study results/luna-pipeline-validation-20261030 --stage practice
```

After reviewing the exact learner-authored guidelines, record their SHA256 and
`transfer_scope_passed: true` in the study's `memory-scope-review.json`, then run
the same driver with `--stage transfer`. This is an analyst review step, not a
request for the user to authorize work again. It does not rewrite the learner's
claims or insert a teacher solution.

`study-manifest.json`, `active-process.json` and per-run terminal records are
the continuation sources of truth. The report links every prompt, tool call,
video, native outcome and timing breakdown. `observed-issues.json` lists
reported failure evidence; recovered errors are not automatically treated as
the cause of eventual task failure. Outside-service time includes Luna calls,
transport and local analysis, not isolated reasoning time.

Results and limitations will be appended when the declared phases complete.
