# RoArtNet offline perception benchmark — 25 October 2026

**The implementation runs on this Blackwell server, but RoArtNet alone is not
accurate enough to become the trusted hinge source.** It is a fast proposal
generator. Observed-motion fitting is the stronger result from this experiment.

- [Browsable report, overlays, raw outputs and timings](../results/roartnet-perception-20261025/watch.html)
- [Original protocol](../results/roartnet-perception-20261025/protocol.json)
- [Summary](../results/roartnet-perception-20261025/summary.json)
- [Per-case timing and accuracy CSV](../results/roartnet-perception-20261025/timing-and-accuracy.csv)
- [Follow-up diagnostics](../results/roartnet-perception-20261025/followup-summary.json)

## Outcome

The 38 predeclared inference cases all completed after dependency repairs.
They comprise six author examples and eight local observations under two
published configuration profiles, three additional crop variants under both
profiles, and four additional seed trials. Two exploratory follow-ups then
tested ten recorded motion pairs and four cleaned clouds. None issued a robot
action, called Luna, changed the live service, or promoted learner memory.

| Primary author-evaluation profile | Cases | Median axis-position error | Median axis-angle error | Median inference |
| --- | ---: | ---: | ---: | ---: |
| Author examples | 6 | 51.0 mm | 7.7° | 2.58 s |
| All local observations | 8 | 63.0 mm | 1.7° | 1.13 s |
| Closed frontal local observations | 5 | 58.6 mm | 1.3° | 1.13 s |

Position is measured where the predicted line intersects the plane through the
reference anchor perpendicular to the reference axis. This avoids dependence on
where the network places a point along its own axis. Point-to-reference-line
and shortest-line distances are also recorded. Angle is modulo sign: opening
direction is **not** established by these accuracy numbers.

The clipped oblique local view has 171 mm position error. Open-door snapshots
have approximately 66–71 mm error. Different seeds on the clean closed view
produce about 56–59 mm error: the position bias is repeatable, not resolved by
retrying random seeds. RoArtNet improves one earlier 267 mm Luna hinge proposal,
but is worse than the other four matched original Luna proposals (24–47 mm).
This does not establish a general model ranking.

Five closed local views are nearly identical. Every local observation uses the
same microwave asset. Training-asset overlap could not be verified from the
released training paths/category configuration. The six author examples are a
runtime reproduction sample, not reproduction of the paper's full metrics.

## What failed, and why

**Runtime failures were compatibility problems and were repaired.** PCL needed
development headers and a current C++ standard. MinkowskiEngine's in-place build
needed its output package directory created first. The original CUDA helper
hard-coded `/usr/local/cuda/include/cuda_runtime.h`, while this server's header
lives under CUDA's targets directory. A single include-line patch uses the
standard header search path. The saved failure and patch are in the report.
There is no substituted SHOT descriptor or reimplemented voting algorithm.

**Clean-cloud position error remains.** Camera reprojection agrees within
0.00008 pixels in the checked case; the optical/SAPIEN/world transforms have
round-trip tests. On state020, the predicted hinge projects close to the left
edge in RGB but is displaced mainly in world Y by about 59 mm. Image overlap
therefore does not establish a correct 3D hinge. The network's learned prior
and domain shift are plausible explanations; this experiment does not isolate
their causal contributions or prove a checkpoint defect.

**Mask contamination has a demonstrated effect.** A two-pixel dilation changes
40,105 points into 41,790 points. Maximum optical depth expands from 1.07 m to
3.48 m because distant background pixels enter the object cloud. The primary
axis error rises from 1.3° to 27.6°. A full bounding-box cloud gives about 201 mm
position error. This is a measured preprocessing sensitivity, not evidence that
Luna needs to reason harder.

**A generic cleanup rule was insufficient.** A separate largest-component test
uses 5 mm voxels and a 15 mm connection radius, without hinge information. It
removes disconnected background, but also removes 7,193 valid points from the
recessed window region. The clean position error worsens from 58.6 to 71.6 mm.
The dilated-cloud axis improves to 6.7°, while its position worsens to 74.6 mm.
The rule remains an experimental utility and is not enabled in the adapter.

## Observed motion is more useful

The follow-up reads consecutive public RGB-D observations from one saved Luna
trajectory. Forward/backward Lucas–Kanade tracks, measured world points and
public robot masks supply correspondences to the existing upright SE2 fitter,
with `prior=None`. A sensor-derived region selects the target neighborhood;
neither RoArtNet's pivot nor native joint information selects the fit.

Four of ten pairs pass the existing geometric tests. Their position errors are
8.8, 9.8, 10.6 and 14.3 mm (median 10.2 mm), and rotation estimates agree with
native motion within 0.32°. Mean wall time per pair is 0.34 s, including saved
sensor loading. The other six pairs reject for inconsistent motion or
insufficient rotation/spatial baseline. Failed pairs stay in the denominator.

The four accepted pairs contain approximately 4–6° of motion. Large jumps can
defeat optical flow; tiny rotations poorly constrain a pivot. The bootstrap
spread remains geometric stability, not calibrated error probability. This
method assumes an upright axis and one dominant moving rigid part; it does not
infer a general 3D axis. It also does not establish that a robot using the
RoArtNet hypothesis can safely create the required probe motion.

These are exploratory results on an existing trajectory, not held-out transfer.
Native references are matched to each observation's exact control step after
the predictions are saved. No initial-state reference was silently reused for
later frames; the recorded base happens to remain fixed in these cases.

## Time and memory

Mean over eight local views, primary profile, instrumented GPU synchronization:

| Stage | Seconds |
| --- | ---: |
| Axis-direction voting | 0.671 |
| SHOT descriptors | 0.179 |
| Tuple preparation, transfers and other inference work | 0.191 |
| Point sampling | 0.042 |
| Normal estimation | 0.041 |
| Joint-origin and affordance voting combined | 0.010 |
| Both neural encoders combined | 0.0049 |
| Voxel quantization | 0.0010 |
| Total | **1.139** |

The saved checkpoint's coarser voting settings take about 0.55 s locally but
give roughly 9.8° axis error on all five closed frontal views. The official
evaluation profile improves direction accuracy; position bias persists.

These are warm inference timings, not full request latency. RoArtNet imports
and device initialization take about 4.93 s; weights load in 0.018 s. MobileSAM
loading took 18.16 s in the separate preparation process, first segmentation
3.62 s, and subsequent segmentations about 0.02 s. Sensor loading/calibration
checks are separately recorded. Keep loaded models alive if integrating this
backend; spawning fresh processes for every observation would lose the warm
latency advantage.

Sampled peak GPU0 allocation was 17,352 MiB (16.95 GiB) across the full mixed
author/local batch, including CUDA contexts, tensor workspaces and caches.
It is not a minimum memory requirement. The two weight files total only about
2 MB; voting intermediates dominate runtime memory. No IK, grasp planning,
trajectory execution or model reasoning is included in these perception times.

## Implementation and reproducibility

Sources: [RPMArt](https://github.com/R-PMArt/rpmart),
[RoArtNet checkpoint](https://huggingface.co/dadadadawjb/RoArtNet),
[RealArt-6 examples](https://huggingface.co/datasets/dadadadawjb/RealArt-6).

- RPMArt revision: `7ba08264d141d2d0f0c772a278e72c1a6525f8c3`.
- Checkpoint revision: `a915569eed5e3ca212ac479e619740f6ad9bc7c7`;
  `formal_microwave_2024_01_16_12_16_57`.
- Dataset revision: `a98345e5fe6096517de959c25688e83f9023a120`.
- MinkowskiEngine revision: `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`.
- Runtime: `.runtime/envs/roartnet`, Python3.10, PyTorch2.7.1+cu128,
  CuPy13.6, PCL1.14.1, Open3D0.18. CPU MinkowskiEngine supplies its original
  sparse-quantization utility; network and voting run on GPU.
- Exact package locks, download hashes, source hashes, logs and the one-line
  compatibility patch are in `results/roartnet-perception-20261025`.

The [adapter](../src/vlm_tamp/roartnet_adapter.py) imports the official network,
SHOT and inference function. It applies the exact rotation from
`real_service.py`: optical `[x,y,z]` becomes `[z,-x,-y]`, then transforms outputs
back with public calibration. It loads weights strictly, without normalization
or rescaling changes. Joint count1 and revolute type come from category config;
they are not predictions. No calibrated uncertainty is claimed.

Preparation uses manually chosen RGB object boxes/points plus existing
MobileSAM. This is assisted object selection, not an autonomous detector.
Prepared NPZs contain only XYZ, RGB and optional public camera calibration.
Author raw fixtures retain the authors' near-zero depth rows to match their
loader; local optical-depth clouds reject nonpositive depth. Missing author
view00 files were replaced with common view02 before any inference or scoring.

Run the existing installed benchmark (completed predictions are reused):

```bash
source scripts/roartnet_runtime_env.sh
python scripts/run_roartnet_benchmark.py
python scripts/evaluate_roartnet.py
python scripts/report_roartnet.py
```

Preparation uses the existing segmentation environment:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .runtime/envs/rekep-libero/bin/python scripts/prepare_roartnet_inputs.py
```

For rebuilding native dependencies, preserve the pinned sources and one-line
patch. Install `eigen=3.4.*`, `libboost-devel=1.86.*` and `openblas` development
packages into the isolated prefix. The conda-forge package is `openblas`, not
`openblas-devel`. Build SHOT using
[the small CMake wrapper](../scripts/roartnet_native/CMakeLists.txt), pointing
`RPMART_ROOT` at the official checkout, `CMAKE_PREFIX_PATH` at the isolated
environment and `CMAKE_LIBRARY_OUTPUT_DIRECTORY` at its `src_shot/build`.
It compiles the unmodified original `shot.cpp` under C++17.

Create `.runtime/MinkowskiEngine/MinkowskiEngineBackend` before running that
repository's `setup.py build_ext --inplace --cpu_only --blas=openblas` with
the isolated environment on PATH and its include/library paths selected.
The setup script cleans its build directory on rerun, so a late packaging
failure can otherwise force recompilation. Its CPU-only warning is expected
because inference uses only CPU coordinate deduplication from this library.

## Decision for future agents

Keep numerical geometry out of Luna's free-form coordinate generation, but
do not treat the first pretrained estimator as ground truth. This result
supports an interface separating a **static hypothesis** from an
**observed-motion estimate**, with explicit support and rejection reasons.

RoArtNet can initialize an estimate; the motion fitter can revise it when
enough object motion is observed. Future integration must handle object
selection, initial probe/contact feasibility, opening sign, generic axes and
geometry-epoch plan invalidation. Current static outputs remain unverified.
The active Luna prompt, memory, GraspGen, cuRobo settings and execution policy
were not changed by this benchmark. Do not promote a per-scene coordinate
correction or the unsuccessful largest-component filter into learner memory.
