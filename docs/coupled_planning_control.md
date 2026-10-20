# RGB-D articulated planning and physical control

The current runner combines RGB VLM role grounding, measured RGB-D geometry,
GraspGen and calibrated frontal grasp templates, constrained planning, and
physical Panda joint control. Initial development uses official state 1 of
microwave opening and closing; the frozen evaluation uses states 1, 11, 21, 31
and 41. State 1 remains development and is reported separately.

## Grounding and geometry

Luna high reasoning selects the handle, moving panel, mechanism family and
hinge-edge pixels from RGB images with DINOv2 visual proposals. MobileSAM and
metric depth provide the selected surface. No simulator inventory is supplied.
Cached decisions require matching RGB and selected depth; unseen observations
receive one bounded request under the existing cumulative $2 API ledger. There
are no automatic API retries. GraspGen uses 200 diffusion/discriminator samples
and retains 100. Its cache is keyed by the exact predicted input cloud and
checkpoint hashes.

A vertical door plane and vertical handle cylinder are fitted from depth. The
initial hinge uses the RGB-selected edge intersected with that plane and an
explicit upright-axis prior. Observed handle motion later fits a horizontal
circle to refine the hinge. This is not arbitrary-axis hinge identification.

The inferred panel footprint comes from the selected RGB-D plane pixels, rather
than all handle/mount samples. Its finite shape is transported rigidly by
observed panel orientation and handle translation. All measured surfaces remain
included; static observations are retained through occlusion. Transparent or
unseen surfaces and ambiguous moving/static associations remain limitations.

## Contact planning and execution

OMPL Atlas/RRTConnect plans seven arm joints plus door angle. Initial grasp
search prefers a checked full task path. If that is unavailable, it may admit a
checked 20-degree contact segment to gather motion observations. The overall
opening target remains 90 degrees. Closing targets the observed adjacent
cabinet orientation.

After closure and each observation-driven update, the planner tries the full
remaining goal before a 20-degree segment and, if needed, a 10-degree segment.
Every segment is completely planned and checked before motion; the smaller
horizon does not relax clearance, binding, or jaw-tilt limits.
It prefers an aligned-jaw mode with
four equalities. A second mode retains the three grasp-point equalities and
bounds jaw tilt by 60 degrees; every native planner state is checked against
that inequality. The actual controller has a separate two-degree angular
tracking allowance. Switching modes requires a valid measured starting state
and a complete checked path before motion.

Binding requires selected measured handle points inside the configured finger
pad channel, a compatible measured jaw separation and a finite geometry-derived
capture distance. The captured reference preserves the measured grip offset.
The channel accounts for the bar's projected width under wrist tilt. This is a
capture/retention test, not a force-closure certificate. RGB-D co-motion is
checked during execution; physical attachment arises from gripper closure.

The official Robosuite JOINT_POSITION controller follows all seven planned arm
joints at 20Hz, with kp=300 and kd approximately 34.64 plus dynamics/gravity
compensation. The gripper receives a separate close command. There are no
object welds, state teleports or native-object-pose commands.

Each joint step checks measured robot geometry, self collision, clearance and
contact-model residuals. RGB-D updates occur every five control steps. Changes
in hinge or finger geometry, segment endpoints and settled tracking stalls can
request another checked plan. Up to 32 updates and 600 control steps are
allowed. Simulation pauses while planning; this is not a real-time hardware
evaluation.

Numerical constraint validation uses 10 micrometres. Physical tracking allows
12mm of contact-model error and logs the actual error. Prepared paths may be
reprojected by up to 0.2rad after model changes, followed by native edge checks.
The measured start retains its separate 0.04rad correction limit and checked
connector. The search allowance is not a physical joint jump.

## Collision scope

Configured Panda collision volumes use pure forward kinematics. VAMP supplies
robot self-collision filtering. Scene checks use twelve calibrated RGB-D views,
including supplementary arm views. Planning margins are 3mm for arms and 2mm
for tools; tracking has a separate 1mm guard. Finger bodies and pads may touch
the selected observed handle region. Palm and unrelated-object contact remain
blocked by the scene model.

Observed points and the panel prior do not certify all hidden obstacle volumes,
unknown space, continuous collisions, or moving-door versus environment sweeps.
A matched-state native diagnostic found missed plate/table configurations as
well as false-positive panel rejections. Those simulator results are evaluation
only and are not supplied to planning. Native contacts during actual episodes
are recorded separately, with handle-region contacts, other contacts and self
contacts distinguished in the report.

## Reproduction

Use the existing isolated environments; the protected LIBERO environment is
unchanged. First develop and test both tasks with the same sources:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.runtime/envs/rekep-libero/bin/python scripts/run_coupled_door.py \
  --tasks open_microwave --states 1 --output results/new-open-development
```

Run the matching closing development case, then freeze and evaluate the four
unseen states. The parent reuses the explicitly marked state-1 measurements,
checks source hashes and forbids per-state retuning:

```bash
.runtime/envs/rekep-libero/bin/python scripts/evaluate_coupled_states.py \
  --development-runs results/new-open-development results/new-close-development \
  --parallel 2 --output results/new-frozen-evaluation
```

Generate a report with report_coupled_door.py and the completed run paths listed
in the evaluation's report-runs.json. Always use fresh output directories. Each
run preserves sources, configuration, input provenance, role decisions, grasp
samples, selected constraints, checked paths, actions, videos and failures.
After each episode closes, automatic cleanup retains the opened RGB-D snapshot
and latest snapshot, plus lossless collision-scene arrays shared by planner
attempts. Temporary planner pickles are removed as soon as their worker exits.
Repeated older observations and auxiliary robot-camera images are pruned. Large
diagnostic JSON files are archived losslessly; read_json in coupled_storage
loads either form. JPEG video frames are deleted only after the movie's frame
count and endpoints are verified and its control-step index is saved. The report
can render directly from this movie. Each episode records retention.json with
removed files, archive hashes and reclaimed bytes; outcomes remain in the
denominator regardless of cleanup or planning failure.

Success is LIBERO's actual predicate becoming true. The report separates
success at any step, success in the final recorded state and completion of the
requested planned motion. Every planned case remains in the denominator,
including failures before motion. Native object identities, joint axes, poses
and collision geometry are evaluation-only. Hierarchical VLM decisions remain
future work.
