# RGB-D numerical grounding pilot

Executed on 2026-10-13. Six real GPT-5.4 Mini calls use four previously queried
LIBERO scenes. This is a development experiment, not a held-out comparison or
an execution benchmark. The existing RGB experiments remain unchanged.

## Input and numerical contract

The VLM sees one untouched 512×512 RGB image, the instruction, source/target
phrases derived from the existing instruction grammar, and a nine-field form:
seven normalized image-point blanks plus grasp-mode and path-constraint choices.
The exact prompt is `configs/prompts/rgbd_numerical_blanks_v1.txt`. No depth map,
inventory, segmentation, crops, DINOv2 proposals, object poses, BDDL answers, or
reference masks enter the API request. The complete SDK arguments and actual
responses are saved per case.

Depth is collected through LIBERO's `camera_depths=True`. Original and recaptured
RGB are pixel-identical. The MuJoCo depth buffer is converted to metric optical-axis
depth, then back-projected using camera calibration. This is an RGB-D system,
with an RGB-only VLM interface. Camera calibration is a system input; simulator
instance identities and object poses are evaluator-only references.

Geometry fills the numerical fields after inference:

- Unchanged selected pixels produce world-coordinate positions in meters.
- A 3×3 depth-neighborhood discontinuity exceeding 3 cm causes abstention;
  selected points are never moved to a nearby object or valid surface.
- A local depth patch estimates the target support normal. Low plane-fit error
  does not establish that it is the correct semantic support.
- Visible grasp-center/closing-edge points can propose a top-down grasp frame,
  opening width and pregrasp position. This is a configurable primitive, not a
  trained grasp detector. The 8 cm opening limit and clearances are configured
  policy values, not VLM measurements.
- A lower-to-upper visible surface chord estimates an object axis. It is not a
  verified canonical object axis or human-reviewed part annotation.
- Typed `point_target`, `axis_alignment`, and `axis_cone` relations have executable
  numerical evaluators and analytic pose Jacobians. No generated Python is executed.

Two instructions explicitly add upright transport. This added language is not
an original LIBERO path requirement. The VLM chooses `keep_upright` versus `none`;
the configured angular tolerance is 5 degrees.

## Measured results

| Measurement | Result |
|---|---:|
| Valid forms | 6/6 |
| Correct source and target under the existing semantic goal grader | **2/6** |
| Correct path-type selection | 6/6 |
| Correct whole-object membership across all landmark blanks | 15/42 |
| Background landmarks | 17/42 |
| Wrong-instance landmarks | 4/42 |
| Landmark abstentions | 6/42 |
| Landmarks passing depth checks | 34/42 |
| Depth-accepted landmarks on the correct object | **14/34** |
| Grasp-frame proposals passing primitive geometric screening | 1/6 |
| Proposals with all source/grasp/closing anchors on the intended object | **0/6** |
| Numerical relation/Jacobian self-checks | 10/10 |
| Real API calls / conservative estimated usage | **6 / $0.039708** |

The ten relation checks cover six point-target instances and two upright cases
with both an equality manifold and a bounded-angle inequality. Maximum analytic
versus finite-difference Jacobian error is approximately 8.23e-11. These checks
verify equations, not semantic correctness, robot kinematics or task completion.

The bowl-on-plate scene has correct source/target selection in both its original
and upright forms. Its closing-edge blanks are null, so no grasp is fabricated.
The middle-bowl scene selects another bowl. Mug cases contain several background
points, including table points selected as plate targets. Those points can have
valid depth and an almost horizontal fitted normal. This explains why sensor and
plane checks alone cannot validate the intended object or support.

Pose generation is therefore not ready for execution. No candidate has fully
correct whole-object anchor bindings, and whole-object labels do not establish
contact quality even when binding is correct.

## Pose constraints and planner boundary

Pose Jacobians use `[world translation, right/body rotation]` perturbations:
`p' = p + dp`, `R' = R exp(skew(dw))`.

`point_target` evaluates `p + R point_local - target`. It concerns an abstract
selected point; object-bottom offsets and the robot TCP relation remain unresolved.
It must not be presented as a verified whole-object On/In goal or a robot goal pose.

`axis_alignment` provides two independent residuals transverse to the desired
world axis, plus an inequality excluding an upside-down solution. This retains
yaw freedom. `axis_cone` separately provides an angular-tolerance inequality;
it is not silently substituted for the exact equality manifold.

Object-axis constraints use an estimated axis frame whose z direction follows
the observed chord. They can be formulated before obtaining a grasp. Binding
them to robot joint coordinates requires an attachment transform and a robot
kinematics adapter. The user's newest ST-pRRTC supports manifold constraints;
its unpublished interface has not been integrated here.

Robot TCP conversion, IK, collision checking, stable contact, whole-object goal
geometry and execution remain unverified. Numerical self-consistency does not
make a VLM prediction or grasp semantically verified. Frames, units, evidence,
configured tolerances and unresolved values are explicit in the results.

## Frozen artifacts and repairs

- `configs/frozen_rgbd_numeric_trial.json` froze the four captures, six requests,
  prompt, geometry policy, implementation and 86 earlier result files before inference.
- `results/rgbd-numerical-mini/` preserves all original real replies and v1 numerical results.
- A reference-decoding bug initially assumed nested rotation matrices, whereas
  stored LIBERO references contain flat nine-element arrays. The original evaluator
  is retained at `results/rgbd-numerical-audit/evaluation_v1_pre_repair.py`.
  `configs/frozen_rgbd_numeric_evaluation_fixed.json` versions only that offline
  decoding repair, without changing inference or numerical policy.
- `results/rgbd-numerical-report-v1/` preserves the initial report, in which upright
  numerical relations were unnecessarily blocked by missing grasp geometry.
- `results/rgbd-numerical-independent-axis/` and
  `configs/frozen_rgbd_numeric_independent_axis.json` add object-axis-frame residuals
  from the same replies. All landmark values, numerical geometry policy and original
  API responses remain unchanged. The derived stage makes **zero additional calls**.

The final report is `results/rgbd-numerical-report/index.html`. It includes RGB
landmarks, aligned depth, interactive 3D observations, numeric estimates, raw
blank values, independent object membership and formula checks.

## Verification and reproduction

The full suite passed **188 tests**, including real LIBERO rendering, with 20
existing dependency warnings. New code passes Ruff. The final verifier replays
all six actual responses, compiles the saved numerical estimates, checks depth
round-trip/reprojection consistency, recomputes object metrics and costs, checks
18 report assets, and verifies 94 preserved result/response files. Inverse
projection consistency is not an independent real-sensor accuracy measurement.
The historical RGB generalization verifier also still passes.

Rebuild and verify the saved final experiment with no model calls:

```bash
cd /home/duo/Rutgers/ARC/random/vlm_tamp
source /home/duo/miniforge3/etc/profile.d/conda.sh
conda activate vlm-tamp
python scripts/report_rgbd_numericals.py
python scripts/verify_rgbd_numericals.py
```

The capture and API entry points are `scripts/prepare_rgbd_numericals.py` and
`scripts/run_rgbd_numericals.py`. Existing captures, frozen protocols and paid
outputs are protected against replacement. Fresh inference needs a fresh output
directory, the explicit `--allow-paid` flag and the authorized external key file.
The repaired evaluation configuration is the runner default. Do not rerun paid
inference to regenerate the report.

The next empirical improvement should address observable part/landmark proposals
and semantic binding. This trial changes the VLM workload relative to earlier
two-point role tests; the 2/6 score is not a controlled estimate of a depth benefit
or disadvantage. No full LIBERO, grasp-success, ReKep-performance or robot-execution
claim follows from this small sample.
