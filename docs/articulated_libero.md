# Additional official articulated LIBERO tasks

The fixed task set covers opening the top drawer (Kitchen 1), closing the top
drawer (Kitchen 10), opening the microwave (Kitchen 7), closing the microwave
(Kitchen 6), and turning on the stove (Kitchen 3), all from `libero_90`.
Development uses state 0. The first exploratory pilot uses official state 1
for each task, seed 0, five settling actions, and at most 600 control steps per
episode. Its two modes are normal observed-scene admission and a paired
simulation diagnostic that logs the same collision predictions without vetoing
motion. The diagnostic reuses input-verified grounding and GraspGen candidates
from the strict trial. Robot self checks, joint limits, mechanism constraints,
and native contact physics remain active in both modes. The two modes are
reported separately. All planned failures remain in the denominator. Task success is the
actual LIBERO predicate; estimated progress and valid code are intermediate data.

Kitchen scenes have unnamed geometries before the known Panda geometry. LIBERO's
named-geometry list omits those gaps, so its list position is not a native geometry
ID. A versioned robot-only adapter resolves each known robot/gripper name through
its native ID for collision meshes, visual meshes, and rendered self-depth priors.
Scene identities do not enter this name view. The existing simulator environment,
robot, controller, physical geometry and task states are unchanged.

Luna high receives raw and DINOv2-numbered agentview/frontview RGB, the task
instruction, and bounded fill-in choices. It selects the contact part, moving
panel, mechanism family, visual hinge edge or control face, motion direction,
and a bounded travel/angle. No object inventory, scene segmentation IDs, CAD,
joint values, axes, or object poses enter its request. Small parts without a
nearby DINO proposal may use a supplemental VLM RGB pixel; that use is recorded.
MobileSAM refines the predicted contact/panel regions. Depth supplies metric
geometry. Upright drawer/door geometry and control-face normal are explicit
template priors, not measured simulator joint axes.

Prismatic motion follows an observed line. Hinged motion follows a circle with
coupled gripper orientation. A short coupled probe (3 cm or 0.2 rad) and its
sampled joint edges are checked before contact, then motion is replanned in
chunks from measured feedback. The whole goal path is not prevalidated.
GraspGen produces 200 Panda candidates, retaining
100; multiple cuRobo IK postures are tested with joint reserve, VAMP self checks,
full moving robot collision meshes, and predicted moving-panel geometry. The
exploratory search cap is 320 tested IK postures per episode. Search covers
different grasp poses at each IK rank before deeper redundancy ranks; reaching this cap
does not prove that all grasps are infeasible. MotionGen is not used.

Functional fingers may touch the selected handle/control and the nearby panel.
The palm/arm and unrelated surfaces stay checked. The panel association is a
bounded 2 cm measured-face band inside its RGB-predicted region. Hidden mechanism
bodies, continuous collision avoidance, and force control are not certified.
Physical gripper closure and observed RGB-D co-motion establish retention; no
attachment, object state, or joint motion is injected. Geometry that is rotationally
symmetric may make control angle unobservable from depth, which is recorded as
a tracking failure rather than treating commanded rotation as measurement.

Each feedback chunk fits observed handle/control motion along the estimated
mechanism, updates the moving scene from that observation, and checks the actual
robot at action boundaries. Numerical estimates can be wrong even when the
chosen mechanism family is right. Endpoint-only geometric comparisons are
diagnostics of the estimated constraint, not executed physical ablations.

Model requests are cached, have no automatic retries, and use a new ledger
carrying forward the previous cumulative $0.181847 under the same $2 cap.
Historical ledgers and frozen bowl/drawer results remain unchanged. This pilot
uses five prototype requests plus five new official-state requests; the paired
diagnostic makes no new API requests. The durable ledger has a 30-request upper
bound under the same two-dollar ceiling for further explicitly requested work.

Run development with `scripts/run_articulated_libero.py --development --states 0`;
run the strict pilot with its defaults after freezing the prompt and configuration.
Run its paired mode with `--scene-mode diagnostic --reuse-run <strict-output>`.
Use the existing `.runtime/envs/rekep-libero/bin/python`, `PYTHONPATH=src`, and
single-threaded BLAS settings as in the cuRobo experiment. Each run requires a
new `--output` directory. All observations, masks, responses, specs, candidates,
paths, contact checks, feedback, native predicates, and videos are preserved.
