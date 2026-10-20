# DINO landmark selection with RGB-D

Executed on 2026-10-13. DINOv2 was tried first while DINOv3 access was pending.
After the user enabled Hugging Face login and approved model access, both
pretrained backbones were run locally on the same four saved RGB-D scenes.
The subsequent Mini selection comparison is **incomplete**: nine of twelve
planned calls were admitted, four replies were truncated, and the final three
were not attempted under the shared $0.15 conservative admission limit.

## Actual models and environment

- DINOv2 ViT-S/14 uses the existing official local source and cached checkpoint.
  Its patch grid for the padded 512×512 scene is 37×37×384; padding is bottom/right
  to 518×518, followed by dense-feature interpolation and cropping to the original
  pixels. RGB is normalized with ImageNet mean/std.
- DINOv3 ViT-S/16 uses the approved official Hugging Face model
  `facebook/dinov3-vits16-pretrain-lvd1689m`, revision
  `114c1379950215c8b35dfcd4e90a5c251dde0d32`. Its weights SHA256 is
  `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d`.
  The original 512×512 image gives a 32×32×384 patch array; class/register tokens
  are excluded. Remote code is disabled. Repeated real GPU extraction exactly
  reproduces all saved patch arrays.
- The isolated `.runtime/envs/dinov3` environment uses Python3.11.17,
  Torch2.7.1+cu126, torchvision0.22.1+cu126, Transformers4.56.2, NumPy1.26.4,
  scikit-learn1.5.2, SciPy1.14.1 and OpenCV4.10.0.84. Existing simulator/RoboPoint
  environments were not upgraded. DINOv2 extraction used its existing runtime;
  both saved feature grids were subsequently processed in one common proposal
  runtime using identical settings.

RGB enters the feature encoder. Aligned metric depth enters candidate clustering
and supplies numerical positions after selection. Simulator masks, identities
and poses are used only for independent evaluation. Local feature computation
and model downloads have no paid API ledger entries.

## Setup failures corrected before paid selection

The initial DINOv2 bundle is preserved at `data/rgbd-dinov2-candidates/`, with its
zero-call request audit at `results/rgbd-dinov2-choices-dry-run/` and initial frozen
protocol `configs/frozen_rgbd_dinov2_choices_trial.json`. It is not the final paired
comparison.

Two independent issues were identified:

1. The inherited workspace required world Z≥0.65m. The two LIBERO-90 mug/plate
   scenes use a table around 0.48m, so every desired mug and plate pixel was
   discarded even when segmentation covered it. Sensor surface heights agree
   plausibly with held-back body heights; the missing candidates were not evidence
   of a gross depth alignment error.
2. The 8×8 MobileSAM prompt grid missed the bowl-on-plate source: only 10 of 3,066
   visible bowl pixels intersected predicted masks. DINO clustering cannot recover
   an object excluded by its input masks.

The paired development revision applies broad common world bounds
[-1.5,-1.5,0]..[1.5,1.5,2] and a 16×16 RGB mask prompt grid to all four scenes.
It retains up to 32 RGB-predicted regions. These are diagnostic development
changes, not a frozen held-out validation result. Initial data/results are kept;
no paid calls were spent on the known missing-proposal setup.

## Matched candidate and selection contract

Both backbones use the same RGB/depth, MobileSAM masks, ImageNet normalization,
proposal settings, numerical policy, Mini model, prompt and graders. Patch size
and learned descriptors differ by backbone. Eight PCA-appearance/XYZ clusters
per region propose original sensor pixels. Deduplication is within a predicted
region at 3mm; separate regions are not merged. Invalid depth/geometry and
unassigned label zero are excluded. Positive predicted regions can still contain
background, robot links or mixed objects.

Candidate IDs are deterministic, consecutive and zero-based. Mini sees untouched
RGB plus a 1536×1536 numbered view, instruction-derived role descriptions and
available IDs. It fills seven landmark IDs plus grasp/path choices, using null
when no suitable landmark is available. No metric coordinates, hidden scene
inventory, reference masks or simulator identifiers enter inference. The integer
schema uses bounds to avoid exceeding the API's aggregate enum-value limit.

Selected IDs resolve to their exact original pixel centers. The existing depth
compiler is unchanged: depth jumps cause abstention without snapping; support
normals, grasp-frame proposals and typed point/axis relations are instantiated
locally. Whole-object placement offsets, robot TCP/attachment transforms, IK,
collision checks and stable grasping remain unresolved.

## Observations and interpretation

Both final proposal sets cover the expected source and target whole-object
instances in **all four scenes**. This is object-instance coverage only. It does
not establish correct handles, opposing contacts, canonical body axes or usable
support interiors. Candidate sets contain 191–256 points, including many
background and distractor points.

| Per six planned cases | DINOv2 proposals | DINOv3 proposals |
|---|---:|---:|
| Actual Mini attempts | 6 | 3 |
| Complete valid replies | 3 | 2 |
| Output-token truncations | 3 | 1 |
| Not attempted because of admission limit | 0 | 3 |
| Complete source/target matches | 2 | 1 |
| Correct whole-object anchor memberships / 42 planned blanks | 13/42 | 6/42 |
| Depth-accepted landmarks / correctly bound | 17 / 13 | 10 / 6 |
| Geometrically screened grasp-frame proposals | 0 | 0 |
| Point-target equation/Jacobian checks | 3/3 | 2/2 |
| Estimated charged usage | $0.08818650 | $0.03909525 |

Total new usage is **nine calls / $0.12728175**, with no unknown cost reservation.
All four incomplete Responses records explicitly report
`incomplete_details.reason = max_output_tokens`. Three consume all 3,000 output
tokens on reasoning with no answer text; one contains only a partial JSON fragment.
Those are completion failures, not evidence of invalid selected landmarks or
intentional abstention. The last three DINOv3 cases are budget stops, not model
predictions. They remain in planned-case denominators with their causes explicit.

Among completed replies, both versions correctly bind the bowl-on-plate example
and select the wrong bowl in the middle-bowl example. The completed DINOv2 red-mug
reply has correct object/target binding and all seven whole-object landmark
memberships. It still fails the configured grasp primitive screening. No upright
variant completed, so this trial supplies no new upright-constraint generation
or execution evidence.

The historical direct-XY trial remains 2/6 complete goals. Comparing it with this
pipeline changes proposal construction, image presentation and output format at
once. This is not an isolated DINO effect. The paired encoder comparison is
incomplete and cannot rank DINOv3 versus DINOv2 or establish general reliability.

## Artifacts and checks

- `data/rgbd-dinov3-features/`: actual approved DINOv3 patch arrays and provenance.
- `data/rgbd-dino-comparison-candidates/`: shared predicted masks and both candidate
  bundles, produced before paid selection.
- `configs/rgbd_dino_comparison.json` and
  `configs/frozen_rgbd_dino_comparison.json`: twelve planned requests, common
  numerical policy, source hashes and 90 protected prior result files.
- `results/rgbd-dino-comparison-mini/`: exact SDK requests, charged raw replies,
  unchanged selected IDs, numerical estimates and explicit not-run records.
- `results/rgbd-dino-comparison-report/index.html`: visual comparisons, annotated
  points, depth/3D views, raw JSON, completion failures, coverage and downloads.
- `results/dinov3-setup/feature-verification.json`: actual DINOv3 checkpoint and
  repeat-inference evidence; CPU/GPU dense interpolation differs by at most
  4.77e-7.

The full suite passed **259 tests**, including real LIBERO rendering, with 20
existing warnings. Optional clustering packages were read from the existing
isolated environment after the simulator's installed packages on the import
path; its installed environment was not modified. The separate vision environment
passed its targeted tests too. New source passes Ruff.

The final verifier checks nine charged replies (including raw truncations), eight
actual feature-model/scene pairs, two checkpoint hashes, matched images/masks,
all candidate sensor-pixel positions, exact ID resolution, numerical/metric
replay, cost accounting, 90 protected artifacts and 123 linked report assets.
Formula checks are self-consistency tests, not semantic or robot-execution proofs.

Rebuild and verify saved results without new model calls:

```bash
cd /home/duo/Rutgers/ARC/random/vlm_tamp
source /home/duo/miniforge3/etc/profile.d/conda.sh
conda activate vlm-tamp
python scripts/report_dino_landmark_comparison.py
python scripts/verify_dino_landmark_comparison.py
```

The next interface should reduce candidate overload: select an object/region
first, then show only its local landmark candidates. Increasing the output cap
alone would spend more on the same dense seven-landmark reasoning problem.
Grasp feasibility still needs a separate geometric/contact assessment even if
whole-object landmark selection improves.
