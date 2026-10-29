# Articulation perception tools to replace free-form hinge proposals

Reviewed 25 October 2026. This is a primary-source and code/release review,
not a reproduced benchmark. No weights were downloaded, software installed,
robot action issued, or learner/runtime interface changed.

Follow-up: [the subsequent RoArtNet implementation and offline benchmark](roartnet_perception_20261025.md)
now reports measured accuracy, timing and motion-verification results. The
paragraph above describes the original review's scope, not the later experiment.

**RoArtNet, the perception component of RPMArt, is the strongest immediate
candidate found for an offline microwave test.** It has a released microwave
checkpoint. Combine an initial learned estimate with observed-motion
verification. Luna can select targets, request observations/probes and choose
recovery actions without inventing metric hinge coordinates.

## Separate the capabilities

Object/part segmentation identifies the target, moving link and grasp region.
Articulation estimation supplies joint type, axis direction and a point on the
axis in an explicit frame. Reconstruction supplies surfaces, including possibly
inferred unseen geometry. Motion-based estimation tests or updates a hypothesis
after movement. Our immediate needs are articulation and moving/static geometry;
a complete CAD model or URDF is optional.

## Practical shortlist

| Method | Required input | Useful output | Release and integration assessment |
| --- | --- | --- | --- |
| **RPMArt / RoArtNet, IROS 2024** | Object-centric partial point cloud from one depth view | Joint origin/direction and interaction point | Official code and six checkpoint groups verified, including Microwave. Best first axis-estimation test; configured joint count/type and legacy dependencies need an adapter. |
| **GAPartNet, CVPR 2023** | Point cloud | Actionable part segmentation and canonical poses | Official code, demos and checkpoint links. Useful for door/handle/part selection; not a universal joint-axis estimator by itself. |
| **ArtiPoint, CoRL 2025** | RGB-D interaction sequence and camera poses | Part motion, joint type and axis | Code and tracking/segmentation weight links. Good motion-verification candidate; human-hand preprocessing and dataset defaults need adaptation. |
| **Ditto, CVPR 2022** | Aligned object point clouds before and after interaction | Active-part geometry and articulation model | Official code, posed-depth demo and pretrained-model links. Useful for digital twins after actual movement. Binary downloadability was not verified. |

### RPMArt / RoArtNet

[Paper](https://arxiv.org/html/2403.16023v2) ·
[official code](https://github.com/R-PMArt/rpmart) ·
[checkpoints](https://huggingface.co/dadadadawjb/RoArtNet).

The network estimates axis position/direction and interaction points by voting
from local point-cloud features. The
[standalone inference function](https://github.com/R-PMArt/rpmart/blob/main/inference.py)
accepts XYZ, optional color, network objects and configuration. It requires a
configured joint count. The real-service script supplies joint types by category.

Checkpoint metadata was checked at revision
a915569eed5e3ca212ac479e619740f6ad9bc7c7. Groups cover Drawer, Microwave,
Refrigerator, Safe, StorageFurniture and WashingMachine. The
[microwave configuration](https://huggingface.co/dadadadawjb/RoArtNet/blob/main/formal_microwave_2024_01_16_12_16_57/.hydra/config.yaml)
uses one joint, XYZ-only input, normalization “none”, 1,024 sampled points,
100,000 sampled tuples and resolution 0.01. The two weight files are 1,021,585
and 1,046,929 bytes; that is weight size, not measured runtime memory.

The full manipulation stack uses AnyGrasp, but the standalone perception
function does not import it. We can retain GraspGen and cuRobo. Perception still
depends on CuPy, MinkowskiEngine utilities, SHOT/PCL and older PyTorch/CUDA-era
extensions. Compatibility on the Blackwell server needs an isolated smoke test.
Published discretization and voting scores do not establish millimetre accuracy
or calibrated uncertainty on our data.

### GAPartNet

[Project](https://pku-epic.github.io/GAPartNet/) ·
[code and checkpoint links](https://github.com/PKU-EPIC/GAPartNet).

This can replace fragile door/handle annotations with actionable-part
segmentation and canonical pose estimates. Turning a part pose into a joint
depends on its part-category conventions. The README links two current
checkpoints and says an older dataset-bundled checkpoint is expired.
Its inference stack includes custom point-cloud operators and CUDA 11-era
environment recommendations. Weight links were inspected, not downloaded.

### ArtiPoint

[Paper](https://arxiv.org/html/2509.01708v1) ·
[code](https://github.com/robot-learning-freiburg/artipoint).

ArtiPoint tracks image points, lifts them with depth/camera motion and estimates
articulation using a factor graph. It directly addresses validating a hinge
from actual object movement, including moving-camera observations.

The [configuration](https://github.com/robot-learning-freiburg/artipoint/blob/main/artipoint/configs/artipoint.yaml)
targets human interaction: hand/person segmentation, nearby query points and
interaction intervals. It defaults to ground-truth **camera poses** and contains
cached-track options. Our adapter would use public camera calibration/poses,
target/robot masks and current observations, without native object-joint labels
or stale demo tracks. Dependencies include ripl-articulation; the README links
Mobile-SAM and CoTracker weights. Its estimation core is relevant, while the
whole CLI is not an unchanged robot-video pipeline.

### Ditto

[Project](https://ut-austin-rpl.github.io/Ditto/) ·
[code](https://github.com/UT-Austin-RPL/Ditto).

Ditto predicts geometry, moving/static segmentation and articulation from
before/after point clouds. Its repository includes a posed multiview-depth demo
and Shape2Motion/synthetic pretrained-model links. The Box link could not be
fetched by the web reader, so only the author-provided link was verified.
Two views of the same closed state do not replace changed articulation.
Object cropping, registration, metric conversion and dependencies need adapting.

## Newer and alternative methods

| Method | Relevant capability | Important boundary or release issue |
| --- | --- | --- |
| [GAMMA, ICRA 2024](https://arxiv.org/html/2309.16264v2) | Single partial-cloud parts, joint type/origin/axis; microwave manipulation shown | The public demo expects checkpoint/best.pth. No weights or download link were found in the checked repository/project page. [Checkpoint requests](https://github.com/qiaojunyu/GAMMA-ICRA2024/issues/5) remain unanswered. |
| [CARTO, CVPR 2023](https://github.com/robot-learning-freiburg/CARTO) | Stereo/RGB-D shape, object pose/size and joint type/state | Encoder/decoder links exist. Its representation is not directly our metric axis API; the RGB-D variant and conversion need assessment. |
| [Particulate, CVPR 2026](https://github.com/RuiningLi/particulate) | Parts, kinematic structure and constraints from a static mesh | Complete mesh input, not raw partial depth. Automatic checkpoint download is documented. Connectivity/up-direction assumptions matter. |
| [Artic-O, 2026](https://github.com/Wxyxixixi/Artic-O) | Geometry, active part and joint parameters from sparse RGB views at two states | Released inference reads joint type from the sample and emits dataset-normalized coordinates. World-frame metadata is not shipped. Ground-truth alignment used in evaluation must not enter deployment inference. |
| [FAMOS, September 2026](https://arxiv.org/abs/2609.20817) | Parts/joints from a variable number of partial clouds, including one view | [Checkpoint](https://huggingface.co/kevinqu7/famos-model) exists, but linked kevinqu7/FAMOS repository returned 404 in both checks; project page says code soon. Not yet verified as runnable. |
| [Rest2Art, ECCV 2026](https://da-eun07.github.io/rest2art/) | Rest-state reconstruction with mesh, VLM/segmentation and generated-motion priors | Generated motion provides hypotheses, not measured motion of our object. Not a direct point-cloud joint regressor. |
| [FlowBot++, 2023–2024](https://arxiv.org/abs/2306.12893) | Dense motion and articulation-projection predictions | Relevant representation/policy reference; an official runnable checkpoint/code path was not verified here. |
| [PartField, ICCV 2025](https://github.com/nv-tlabs/PartField) | General 3D part features and segmentation | Useful building block; does not itself supply joint constraints. |

Artic-O provides a 2.7 GB checkpoint and reports subsecond forward inference
on an H100 after warmup. That is not measured end-to-end metric reconstruction
time on our sensor stream. FAMOS metadata lists a 557,324,880-byte checkpoint
at revision 04c31ad178d878735da8fa258d82b1f38028b627; a separate PartField
backbone is required. Neither was downloaded or run.

## Existing local capability

[Our RGB-D tracker](../src/vlm_tamp/rigid_hinge_tracking.py) estimates an upright
hinge numerically from moving point correspondences. It distinguishes measured
pose motion from a stable pivot and reports geometric spread, not calibrated
probability.

At line 60, the fitter rejects a pivot more than 0.10 m from a supplied XY
prior. The tracker wrapper always supplies a prior. This is therefore local
refinement, not independent initialization or unrestricted recovery from a
bad initial hinge. This code fact has not been established as the cause of
historical Luna failures; the static Qwen probes did not use this motion fitter.

[The service](../scripts/serve_subscription_robot.py) already has geometry
export, tracking/model updates and geometry epochs. An estimator backend can
populate these numerical models without asking Luna for hinge pixels. A new
model must still invalidate plans based on an older geometry epoch.

## Recommended next experiment

1. Reproduce **RoArtNet perception alone** on an author example in an isolated
   runtime. Then use saved public RGB-D, with explicit frame/unit checks and
   the published joint-count/preprocessing assumptions.
2. Compare joint-line distance, axis-angle error, coverage, runtime and crop
   sensitivity against existing proposals. Native simulator axes stay private
   to evaluation. Do not substitute native segmentation for sensor-derived
   object crops. Audit training-asset overlap before claiming unseen-object results.
3. Validate estimates against recorded object motion using the existing fitter
   or an adapted ArtiPoint core. Initialization/relocalization and nearby-prior
   refinement are distinct operations. Hand movement does not prove object motion.
4. Integrate only after those checks. Luna selects objects, requests informative
   observations/probes and chooses recovery. Numerical tools supply the axis,
   moving geometry and constraints; GraspGen/cuRobo keep their existing roles.

A proposed tool result should contain an estimate ID, joint type and its source,
axis/origin in a named metric frame, optional moving/static masks, support
diagnostics and whether actual motion has verified the estimate. A raw learned
or voting score must not be called a calibrated probability of correctness.

Hidden mechanisms are not always uniquely identifiable from a closed-state
observation. A learned prior can initialize a hypothesis; actual motion offers
a stronger test. Full reconstruction may supply useful additional geometry,
but it need not be a prerequisite for replacing free-form hinge guesses.
