# GPT-6 Luna on frozen DINO RGB-D landmark choices

Executed 2026-10-13 after the user selected Luna and raised this experiment's
API ceiling to **$2**. The run uses the existing isolated `vlm-tamp` environment.
No local feature extraction or segmentation was repeated. The official API
returned `gpt-6-luna` for all twelve real calls.

## Inputs and inference controls

The twelve cases, four saved scenes, DINOv2/v3 candidate bundles, original and
numbered RGB images, prompt, output schema, depth calibration, numerical compiler
and semantic grader are exactly the previous Mini comparison's inputs. Luna sees
RGB and available IDs, without simulator inventory, reference labels, masks,
metric coordinates or depth. Local calibrated depth supplies metric positions
only after the model selects IDs.

Changed controls are the API model, `reasoning.effort=none`, the 1,000-output-token
cap (previously 3,000), and budget/pricing. This is a practical configuration
comparison rather than an isolated model-quality ablation. There are no retries,
substitute predictions or extra paid setup calls.

Standard short-context pricing was checked against the official model page:
input $0.10/M, cached input $0.01/M, cache writes $0.125/M, output $0.50/M.
All input is conservatively accounted at $0.125/M, including admission checks;
cache writes are not added a second time. Provider billing remains authoritative.

## Measured results

| Per six planned cases | DINOv2 candidates | DINOv3 candidates |
|---|---:|---:|
| Actual Luna calls | 6 | 6 |
| Complete valid forms | 6/6 | 6/6 |
| Correct source and target object binding | 4/6 | 4/6 |
| Correct whole-object landmark memberships | 37/42 | 39/42 |
| Depth-accepted landmarks | 42/42 | 42/42 |
| Geometrically screened grasp-frame proposals | 3 | 4 |
| Grasp proposals with all tested anchors on intended objects | 3 | 3 |
| Correct path-type choices | 6/6 | 6/6 |
| Passing equation/Jacobian self-checks | 10/10 | 10/10 |
| Conservative estimated API usage cost | $0.00341850 | $0.00341925 |

The complete run costs **$0.00683775**, using 51,582 input tokens and 780 output
tokens. Reasoning tokens are zero, reported cache-write tokens are 51,546, cached
input tokens are zero, and no unknown-cost reservation remains. All responses
are completed, without output truncations or budget stops. The frozen settings
retain the user-authorized $2 cap; unused budget was not spent.

Luna selected all seven point blanks in every case; the form permits null, but
no null was returned. DINOv2's two goal errors select the wrong plate for the red
mug in ordinary and upright instructions. DINOv3's ordinary red-mug case selects
the wrong plate, and its ordinary yellow-mug case selects the wrong source.
The upright red-mug case succeeds with DINOv3 despite its ordinary variant's
failure, indicating sensitivity to these instruction variants.

The saved Mini run cost $0.12728175 for nine calls and produced five complete
forms; four responses were truncated and three planned calls were not admitted.
Its planned-case source/target count is 3/12 versus Luna's 8/12. Completion and
budget differences must remain explicit when reading that comparison.

## What is still unresolved

Correct source/target binding does not require every grasp/contact/axis landmark
to be correct. Across this run 76/84 landmarks hit the intended whole object, but
that metric does not establish a suitable contact, grasp center or canonical
axis. Seven frames pass the configured geometric screen; six have correct tested
whole-object anchor membership, without physical grasp or collision verification.

All four upright variants choose the requested constraint type. Their estimated
axis error against the held-back canonical body-Z proxy ranges from 11.9 to
108.5 degrees, so none meets the configured five-degree cone under that proxy.
Those reference axes are not human-reviewed affordance labels. Correct constraint
selection and passing analytic formula checks therefore do not imply correct
constraint parameters. All 84 depth samples pass local numerical checks, while
eight are on a wrong object or background.

Robot attachment/TCP transforms, whole-object placement offsets, joint-space
compilation, IK, stable grasping, collision checking and execution remain
unresolved. These twelve correlated cases on four development scenes do not
establish a population model or DINO backbone ranking.

## Artifacts and validation

- `configs/rgbd_dino_luna_comparison.json`: Luna controls and authorized $2 ceiling.
- `configs/frozen_rgbd_dino_luna_comparison.json`: derived parent protocol,
  sixty unchanged candidate/depth hashes, and 279 protected earlier artifacts.
- `results/rgbd-dino-luna-dry-run/`: exact requests audited with zero API calls.
- `results/rgbd-dino-luna-comparison/`: actual requests, raw charged responses,
  selected IDs, numerical compilation and usage ledger.
- `results/rgbd-dino-luna-report/index.html`: images, selected landmarks,
  depth/3D views, exact prompt and matched historical Mini comparison.
- `results/rgbd-dino-luna-report/verification.json`: successful saved-response,
  geometry, grading, cost and historical-file replay.

Validation passed 122 relevant existing/new tests, including 38 new transport
safeguards and ten report tests, plus Ruff. The replay checks twelve raw replies,
eight actual encoder/scene pairs, four matched scenes and depth captures, two
checkpoint hashes, twenty equation/Jacobian checks, 279 protected artifacts and
128 report assets. Browser rendering was not tested. Rebuild and verify without
new API usage:

```bash
cd /home/duo/Rutgers/ARC/random/vlm_tamp
source /home/duo/miniforge3/etc/profile.d/conda.sh
conda activate vlm-tamp
python scripts/report_dino_luna_comparison.py
python scripts/verify_dino_luna_comparison.py
```

Luna is a low-cost working selector for further development. The next geometry
improvement should test object-specific axis and contact estimation independently
of the correct constraint-type choice, while retaining explicit abstention for
unsupported numerical estimates.
