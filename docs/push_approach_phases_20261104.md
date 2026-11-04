# Push approach and contact phases

Targeted `push` now separates free travel from contact acquisition. Luna still selects the
observed point, robot contact face and desired motion. No task identifier, known direction,
successful coordinate, object mesh or scripted task sequence enters the tool.

1. The requested surface normal defines the outward side. Point-only requests use the
   opposite of their explicitly requested push direction for approach, without inventing a normal.
2. One hand extent defines the travel from a precontact point to the selected point. The
   surface-normal mode uses the hand's projected extent; point-only mode uses its bounding-box
   diagonal because wrist orientation remains free. This is approach travel, not inflated
   obstacles or an added clearance margin. Both the extent and the chosen side are returned.
3. cuRobo solves the precontact goal with arm, palm and finger scene collisions enabled.
   VAMP plans the free-space route using the same observed points and sphere model.
4. A straight cuRobo continuation approaches the original selected point at 3 cm/s. Finger
   contact is allowed; arm/palm permissions follow the existing contact policy. The observed
   target remains stationary during this approach. No CPU IK repair or backup branch is used.
5. Existing measured arrival and push execution follow. Physics resolves pushing contact.

The four upstream finger spheres retain their original radii. Their positions use current
robot finger-link transforms, including measured aperture. Link transforms avoid MuJoCo's
internal mesh-centering transformation. Three existing palm spheres supply a rigid coordinate
frame, so native SIMD and GPU environment queries use existing sphere FK. GPU coefficient and
permission tensors update in place for warm CUDA graphs. Original VAMP self-collision remains
unchanged. Contact-phase exemptions do not leak into the next free approach.

`approach_plan` and two-camera `approach_images` expose the precontact point, side, distance
and planned pad trajectory. `free-approach`, `contact-approach` and `push-stroke` remain distinct
executed phases with before/after captures, control timestamps and timing. The dashboard
labels planned overlays separately from execution video and evaluator-only object motion.

The numerical planner still sees only observed surfaces. Missing or misassociated RGB-D
geometry can still produce a physically incorrect plan. A correct approach also does not
establish whether Luna chose the correct face or task direction.

Validation and fresh state-17/state-49 results are recorded in
`results/luna-approach-collision-20261104/watch.html`. The full-episode protocol holds the
600-control limit, Luna/high, two native cameras and fresh memory condition. The existing
32 GiB cgroup, zero swap and RAM/I/O/SSH monitoring remain active. This is a two-state
engineering evaluation, not evidence of transfer to new task families.
