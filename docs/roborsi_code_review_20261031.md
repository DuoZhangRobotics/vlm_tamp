# RoboRSI code comparison

Read-only review of `nssmd/RoboRSI`, commit
`90c8569705bf4004a4f79e34e650c87ab7f075d4`, cloned under
`third_party/RoboRSI`. The checkout includes source, documentation and tests;
media assets, model downloads and the LeRobot submodule were not installed.
No RoboRSI execution or benchmark reproduction was performed. This public code
snapshot is not assumed identical to the paper's evaluated release.

## They have motion planning

The LIBERO backend uses a persistent PyRoKi service with both inverse
kinematics and trajectory optimization. IK starts from the measured current
joint configuration, with pose, joint-limit and seed-regularization terms.
Trajectory optimization starts at the live joints and normally produces ten
waypoints. Its least-squares factors include joint limits, self-collision,
swept capsule/world collision costs, and motion smoothness.

The important limitation is the world model. `Solver.__init__` supplies exactly
one fixed table half-space at z=-0.012 in the robot base frame. The request
protocol does not accept an observed scene point cloud or appliance obstacle
meshes. These are collision-aware optimization costs, not evidence that a
returned trajectory is collision-free in a complete RGB-D scene.

`LiberoControl.servo_to` normally executes one IK goal with a joint-position
servo. It uses trajectory optimization when requested for a longer transfer,
or after detecting a stalled direct drive. It can use Jacobian control when the
service is unavailable, and a final IK correction after a trajectory. These
fallbacks are part of their implementation, not recommendations to add them here.
The server warms and caches JIT graphs; its comments claim short warm solve
times, which this review has not measured.

Sources: [server.py](../third_party/RoboRSI/roborsi/embodied/motion/pyroki/server.py),
[solve_ik.py](../third_party/RoboRSI/roborsi/embodied/motion/pyroki/solve_ik.py),
[solve_trajopt.py](../third_party/RoboRSI/roborsi/embodied/motion/pyroki/solve_trajopt.py),
[_control.py](../third_party/RoboRSI/roborsi/embodied/skills/base/_lib/libero/_control.py).
RoboTwin motion delegates to its own environment's motion interface; its external
planner implementation was not audited in this checkout.

## Toolbox comparison

Static inspection finds 35 LIBERO base skills with callable dispatchers. The
agent additionally receives completion/library tools, evolution tools when
enabled, and task-specific compound skills. RoboTwin has 57 descriptors, 53
wired through plugin or legacy dispatch in this snapshot. These are separate
backend catalogs, not one agent receiving every skill simultaneously.

Our latest completed Luna trial exposed 41 operations; the repository contains
45 schemas including optional practice/DINO interfaces. Counts mix different
levels of abstraction and do not measure capability or reliability.

| Area | RoboRSI LIBERO implementation | Our current interface |
| --- | --- | --- |
| Perception | Semantic pixel selection, detector/SAM routes, RGB-D unprojection | Selected masks, cross-view association, explicit geometry measurements |
| Grasping | GraspGen candidates and several grasp-execution routines | Small GraspGen batch, whole-scene hand filtering, closest valid IK and approach planning |
| Motion world | Robot plus fixed table in PyRoKi trajectory service | Observed moving/static scene geometry and explicit contact permissions |
| Articulation | Door-handle pixel and hinge-side heuristic, followed by an arc | Multiview hinge grounding and motion-based joint estimation with support diagnostics |
| Contact | Planar source/target-pixel pushing and specialized drawer/door skills | Direction/distance strokes, retained pushes, hook-pull and joint-linked manipulation |
| Reuse | Versioned skill code, task wiki and compound policies | Stable numerical tools with Luna practice and learner-authored procedural memory |

The meaningful difference is our explicit articulated-geometry and observed-scene
tooling. RoboRSI has a broader collection of ready-made manipulation routines.
Neither difference establishes superior task performance.

Their `open_hinged_door` samples depth to the selected side of a handle, chooses
a hinge candidate and executes roughly five-degree arc increments. The returned
`achieved_angle_deg` advances when the hand reaches its next target and grip
state remains held; it is not an independently tracked door-angle measurement.
Likewise, `measure_relative_rotation` calculates an angle between supplied
vectors, rather than tracking an object through images. This is a concrete
distinction from our measured part-motion tools.

Sources: [tool catalog builder](../third_party/RoboRSI/roborsi/embodied/agent_loop/prompt_tools.py),
[door skill](../third_party/RoboRSI/roborsi/embodied/skills/base/open_hinged_door/libero/policy.py),
[push skill](../third_party/RoboRSI/roborsi/embodied/skills/base/push_object/libero/policy.py),
[perception implementation](../third_party/RoboRSI/roborsi/embodied/skills/base/_lib/libero/_perception.py),
[our contracts](../src/vlm_tamp/robot_contracts.py).

## Useful lessons without adding another architecture

- Keep live joints, coordinate frames and solver startup consistent. Numerical
  work should remain behind stable tools.
- Reuse procedures that have actually worked, recomputing geometry for each
  episode. A compact procedure can eliminate repeated model decisions.
- Their planar push accepts source and target pixels and computes the world
  direction. This is worth comparing with our world-vector interface, but a
  door needs joint-aware geometry; their planar push is not a drop-in door tool.
- Preserve the distinction between robot motion, object motion and task success.
  Do not import their grasp/door success labels as stronger evidence than the
  underlying measurements provide.
- Their code also contains numerous restrictions, fallback branches and
  object-specific perception heuristics. Copying the entire stack would not
  fulfill the user's request for a simpler pipeline.

No planner replacement, new agent hierarchy or imported recovery policy was
made from this review. The immediate experiment remains the two requested
closing-state regressions using the existing pipeline.
