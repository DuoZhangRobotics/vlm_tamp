# Collision mapping tools and community evidence — 22 October 2026

The user asked about point-cloud collisions in cuRobo, specifically nvblox,
community experience and established alternatives. This review covers upstream
documentation, maintainer replies and first-person integration reports. Reported
external performance is not treated as a benchmark on this server.

## Recommended boundary

Use an established mapper to turn calibrated depth into a distance-field tensor,
then let the GPU collision/IK layer query that tensor. Preserve the current
native contact and complete-path checks while assessing a new proposal backend.

This recommendation is supported directly by cuRobo maintainer Balakumar
Sundaralingam in issue #572: in October 2025 he proposed moving from direct
`WorldBloxCollision` calls to an ESDF tensor interface; in May 2026 he reported
that cuRobo had its own mapper and that dense-grid lookup was faster than
querying nvblox's mapping structure during optimization. These are maintainer
reports, not our timing measurements.

Source: [maintainer discussion and updates](https://github.com/NVlabs/curobo/issues/572).

## Options

| Tool or stack | Role | Fit for this project |
|---|---|---|
| nvblox | GPU reconstruction and TSDF/ESDF mapping from sensors; Python, C++ and ROS 2 interfaces | Strong independent mapping layer; export a compatible ESDF tensor |
| cuRobo v2 native mapper | Integrated GPU depth fusion and distance fields alongside motion-generation tools | Worth an isolated migration benchmark; public API differs substantially from our v0.7.8 |
| Isaac ROS cuMotion + nvblox | ROS 2 motion/IK services with an ESDF service connection and MoveIt integration | Useful for a ROS-based robot system; our current Python/MuJoCo pipeline would gain a substantial new runtime dependency |
| MoveIt 2 + OctoMap/FCL | Point-cloud/depth occupancy mapping, robot self-filtering and collision checking | Established alternative when the ROS planning ecosystem and body-pair contact permissions are priorities |
| Voxblox | CPU TSDF/ESDF mapping with ROS integration | Relevant CPU alternative; less aligned with our GPU acceleration objective |

Sources:

- [nvblox upstream](https://github.com/nvidia-isaac/nvblox)
- [cuRobo upstream and v1/v2 API notice](https://github.com/NVlabs/curobo)
- [cuMotion ESDF configuration and IK services](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_cumotion/isaac_ros_cumotion/index.html)
- [MoveIt perception/OctoMap integration](https://moveit.picknik.ai/main/doc/concepts/planning_scene_monitor.html)
- [MoveIt collision checking and allowed-collision matrix](https://moveit.picknik.ai/main/doc/concepts/kinematics.html)
- [Voxblox upstream](https://github.com/ethz-asl/voxblox)

## Community findings that affect the choice

An NVIDIA forum thread from February–May 2026 documents an old cuRobo bridge
expecting the nvblox_torch 0.0.5 API while newer bindings use different
constructors and query functions. Matching package names is insufficient;
the replies explicitly distinguish the legacy bridge from cuRobo v2.
[Version-matching thread](https://forums.developer.nvidia.com/t/curobo-nvblox-version-matching-issue/359743).

The curobo_ros author reported an August 2026 migration to cuRobo v2, removing
custom nvblox forks and voxelization machinery. Its migration document shows
the actual API changes and how camera observations enter the new mapper.
[Community announcement](https://discourse.openrobotics.org/t/curobo-ros-v2-gpu-accelerated-motion-planning-wrapper-now-on-curobo-v0-8-0/57147)
and [implementation migration notes](https://github.com/Lab-CORO/curobo_ros/blob/main/docs/MIGRATION_V2.md).

These reports support testing a pinned v2 environment instead of assuming that
the latest nvblox bindings will drop into the installed v1 adapter. They do not
establish an automatic win for our articulated-contact problem.

## Local evidence and status

The server currently uses cuRobo 0.7.8. Neither nvblox nor nvblox_torch is installed
in its environment, and `/opt/ros` is absent. The production cuRobo robot export
has no collision spheres and its IK worker receives no collision world.

A separate geometric prototype under `results/curobo-pointcloud-20261022` prepared
a distance grid from the retained sensor-derived point cloud and a calibrated
336-sphere arm/palm approximation. The CPU field construction took 1.305 seconds;
the float32 grid uses 35,071,284 bytes at 1 cm resolution. This construction is a
reference adapter experiment, not a replacement mapping stack or a real-time
mapping benchmark. The moving cloud is transformed using the existing estimated
hinge for the requested goal; no simulator object geometry is used.

GPU smoke 001 exposed a local compatibility problem: cuRobo's legacy world
adapter references `warp.torch`, removed from installed Warp 1.18. Probe 002 uses
a process-local alias to the modern public interop functions; installed libraries
and the production worker were not modified.

GPU smoke 002 passed: occupied versus free-space queries had the expected sign,
distance gradients were finite and nonzero near the sampled surface, and GPU
robot-sphere positions matched the calibrated reference at three configurations
with maximum error below 0.3 micrometres. Initialization took 7.337 seconds.
This verifies GPU collision plumbing, not full collision-aware IK success,
map accuracy everywhere, or physical task completion.

The proxy covers arm/palm geometry only. Finger collisions and bounded intended
finger/target contact still require the existing native checker. A mapper also
does not decide object ownership or articulation: the moving door must be kept
distinct from stationary obstacles, with scene updates and uncertainty explicit.

[Passing smoke receipt](../results/curobo-pointcloud-20261022/gpu-smoke002/result.json)
and [retained first failure](../results/curobo-pointcloud-20261022/gpu-smoke001/failure.json).

## Next engineering step

Implementation update: the [v2 migration and test report](curobo_v2_migration_20261022.md)
now records the selected backend experiment. Its current scene adapter uses
upstream point-cloud-to-mesh conversion and GPU mesh SDF queries; raw-depth
TSDF/ESDF fusion has not been substituted into the planning scene.

Decision update: the user selected cuRobo v2 and requested a literature review
of uncertain grasps/collisions before migration. That
[review is complete](uncertainty_planning_review_20261022.md); the next backend
benchmark should therefore target an isolated v2 environment.

Benchmark cuRobo v2's integrated mapper and collision-aware motion generation
in an isolated environment. Compare map update time, GPU query/IK time, native
admission rate and complete-path success on identical saved inputs. Verify the
pinned v2 scene/map interface instead of assuming the legacy tensor API carries
over unchanged. Keep robot calibration, intended-contact rules and scene
semantics explicit. The small adapter remains necessary even when mapping and
collision algorithms come from maintained libraries. No production backend has
been switched yet.
