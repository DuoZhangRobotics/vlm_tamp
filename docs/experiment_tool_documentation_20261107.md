# Persistent tool documentation and native collision diagnosis

The documentation pilot did **not** improve task success. Transient documentation succeeded in
one of four cases; retained documentation succeeded in none. Retention reduced repeated help
requests from 47 to one. It remains opt-in, with the default disabled. This eight-episode pilot
is too small to establish a reliable performance difference.

[Interactive results, exact inputs and videos](/common/users/dz323/vlm_tamp/results/luna-tool-documentation-20261107/watch.html)
use manual refresh. All original episodes remain available. Source revision `4161fe9` was
frozen throughout the pilot; subsequent repairs do not change those recorded outcomes.

## Declared comparison

Both conditions used fresh `gpt-6-luna` actors at high reasoning, the same cuRobo/VAMP backend,
two native cameras (`agentview`, `wrist`), observed-cloud views, hand previews, and eight bounded
interaction memories. Each episode allowed 600 controls including reset and 1800 wall seconds.
No inherited practice memory, demonstrations, generated policy code or native contact labels
were supplied. Conditions alternated order across cases.

The sole difference was retaining exact previously consulted function schemas in
`previously_consulted_tool_contracts`. This adds supplied tool documentation, not learned task
experience. Original numerical files matched reference `e2c9860`.

| Case | Documentation | Native success | Controls | Actor wall s | Model/client s | Help / repeats |
|---|---|---|---:|---:|---:|---:|
| Microwave 17 | Transient | No | 401 | 1800.23 | 1471.93 | 25 / 14 |
| Microwave 17 | Retained | No | 421 | 1800.12 | 1673.40 | 10 / 1 |
| Microwave 49 | Retained | No | 600 | 1684.67 | 1496.67 | 6 / 0 |
| Microwave 49 | Transient | No | 600 | 1650.94 | 1510.66 | 26 / 17 |
| Top drawer 0 | Transient | No | 600 | 1169.72 | 989.42 | 13 / 4 |
| Top drawer 0 | Retained | No | 320 | 1800.24 | 1631.94 | 9 / 0 |
| Bowl placement 0 | Retained | No | 600 | 1173.42 | 983.12 | 9 / 0 |
| Bowl placement 0 | Transient | Yes | 481 | 977.71 | 859.94 | 20 / 12 |

Model/client time includes the CLI request round trip; it is not a measured pure inference time.
At 20 Hz, 600 controls represent 30 simulated seconds, distinct from wall time. Reset, gripper
holds and execution waiting consume controls. Planning and model calls consume wall time.

Before the live pilot, four deliberately selected recorded contexts tested repeated-help
behavior. Four fresh baseline decisions repeated help; four with retained schemas chose other
valid operations. No returned request was executed. These probes are not task-success evidence.
An archive audit checked 576 submitted contexts against their recorded schema/image inputs;
three calls had no completed decision before the wall deadline.

## Observed failure mechanisms

The dashboard gives each case its motion stages, exact stop condition, remaining budget,
planning evidence and video timestamps. Native joints/contact identities below were used only
for post-episode diagnosis, never as Luna inputs.

- Microwave 17/transient: the door never moved. Fingers contacted the fixed housing. Several
  hand strokes reached their goals but did not move the door; wall time expired with 199 controls
  unused.
- Microwave 17/retained: the door opened from 88.92 to 119.98 degrees, mainly during a generic
  contact-pose approach. Palm/arm contact occurred before the intended push. The precise
  perception/model contribution was not isolated.
- Microwave 49/retained: the first push opened the door. Over controls 186–250, the RGB-D
  tracker reported 0.018 degrees while the native joint changed about 3.05 degrees. Static
  background dominating the broad region is a hypothesis: individual feature coordinates
  were not archived, so their identities cannot be established retrospectively.
- Microwave 49/transient: early pushes opened the door; a later free approach hit the table
  with the palm. Control exhaustion followed. Available search time was not the immediate
  cause of that physical blockage.
- Drawer/transient: partial closure of 36.43 mm, with 115.23 mm left open at 600 controls.
  Drawer/retained: no drawer movement; repeated palm/finger cabinet contacts stopped approaches
  short, and wall time expired at 320 controls.
- Bowl/retained: the first grasp was empty. The second grasp retained the bowl and lifted it
  about 10 cm; Luna nevertheless described it as remaining on the table and later released it.
  Actual public after-lift images are shown beside this diagnosis. A third attempt exhausted
  the controls. This is not solely a grasp-planning failure.
- Bowl/transient: Luna selected a contact-pose operation after failed learned-grasp searches,
  lifted/carried/released the bowl, then made a corrective approach that reached native success.
  The host did not inject a replacement grasp. Success occurred during that final approach;
  there was no completed second grasp or post-success retention test.

## Confirmed interface repairs

Commit `f47aad0` repaired three mismatches after the pilot:

1. Invalid hinge endpoint depth now names the hinge endpoint, rather than blaming the valid
   contact pixel. The selected pixels and geometric acceptance rules are unchanged.
2. `refresh_contact`, already implemented and documented, is now in the actor's callable tools.
3. Exported contact schemas now express their existing region-or-part selection requirement.
   A bare pixel formerly passed the published schema but failed backend validation. The new
   selection requirement is combined with, rather than replacing, the motion-intent alternatives.

The initial prompt also states the existing 20 Hz control units. Validation: 114 focused tests
passed, two optional integration tests skipped, configured Ruff checks passed, and all 352
previously accepted requests from the pilot remained accepted. No numerical policy changed in
that commit.

## Native collision-tree defect and repair

A separate bowl qualification compared the same Luna-selected 134-point contact crop with its
2849-point selected part mask, using the same seed, eight GraspGen proposals and collision world.
Before repair, the crop produced an empty grasp. The full-mask approach hit the table with finger
2 and stopped 6.72 mm short. A larger input alone did not solve grasping.

Operator diagnosis then found a contradiction: the existing finger sphere intersected a planner
point by 5.22 mm, but the native checker accepted it. Checking a subset containing that same point
correctly rejected it; adding the rest of the unchanged scene incorrectly made it clear.

The CAPT tree's upper-child neighbor propagation scanned the beginning of an ascending lower
partition. Relevant near-split points form its **end**, so a distant first point caused the scan
to stop before copying nearby obstacles. The repair scans backward from the midpoint and copies
that suffix. It changes no point coordinates, sphere radii, margins, robot kinematics or planner.

The dependency is pinned at `CoMMALab/vamp` revision
`24e0f8adef0649a957fec77bf33f8c2f04515a18`. The patch and exact before/after hashes are preserved in
`patches/mcvamp-capt-neighbors.patch` and its JSON manifest. Both the native adapter and Python
binding were rebuilt using CMake `-j32`. Build metadata now records the modified dependency
truthfully, and study source records include the native dependency/binary identities.

- Four-point counterexample: incorrect clear before, collision after.
- 4100 independent distance-oracle queries: 54 scalar and 54 SIMD mismatches before, zero after.
- Original 17,993-point scene and fixed open aperture: finger 2/table collision now detected.
- 60 focused Python/native integration tests passed; Ruff and `git diff --check` passed.
- The existing open-finger collision model is now enabled for GraspGen free approaches. One
  scene replaces the previous two effectively equivalent contexts. Physical closure and
  contact manipulation retain their intended contact behavior; no predicted closed-hand check
  or additional dense execution validation was introduced.

In one post-repair replay per input, both approaches completed without the previous table
contact, but both grasps were empty. Production GraspGen input scope is unchanged. These are
operator qualifications, not Luna success scores or repeatability claims.

[Before/after diagnostic videos and evidence](/common/users/dz323/vlm_tamp/results/luna-native-repair-20261108/diagnostics.html)
are linked from both dashboards. Fresh Luna trials on closing states 17 and 49 use revision
`dab7768`, the same two cameras and budgets, and no task hints. Their results are recorded in
[the follow-up dashboard](/common/users/dz323/vlm_tamp/results/luna-native-repair-20261108/watch.html).

All live runs use the existing 32 GiB whole-job memory limit, zero swap allowance, memory/I/O
telemetry and SSH responsiveness checks. Task failures remain distinct from infrastructure
interruptions. Remaining open questions include contact-side selection, reliable interpretation
of object motion, and grasp quality; the collision repair does not establish those capabilities.

## Fresh post-repair outcomes

Both declared trials completed normally and failed at 600 controls. No extra trials were run
until a success appeared, and no configuration changed during either episode.

| Case | Final opening | Actor wall s | Model/client s | Robot service s | Native success |
|---|---:|---:|---:|---:|---|
| Microwave 17 | 30.16° | 896.64 | 768.47 | 113.98 | No |
| Microwave 49 | 120.20° | 871.19 | 669.74 | 182.68 | No |

State 17 started 88.92° open. A blocked first approach stopped 150.23 mm short; an ineffective
hand stroke followed. A later free approach and contact approach reached within 2.6 mm, but the
8 cm push opened the door to 110.21°. Its automatic RGB-D tracker had one usable correspondence
after contact and zero after the stroke, so it reported unresolved motion. The last approach
closed the door to 30.16° but used the remaining 236 controls (220 nominal references plus 16
arrival controls). It ended 18.32 mm short, with 912.61 wall seconds still available.

State 49 started 107.26° open. Its learned-grasp approach hit the door/table and stopped 88.15 mm
short. Subsequent contact acquisition hit the housing or opened the door farther. Two later
free approaches reached their targets; the corresponding contact approaches stopped 203.49 mm
and 46.79 mm short. The final three-control stroke exhausted the budget with the door 120.20°
open and 938.12 wall seconds still available. The last contact search took 2.57 of its 30 allowed
seconds; this physical failure was not search-budget exhaustion.

A read-only reconstruction of state 49's first grasp scene matched the original point counts.
All 24 recorded path states and 118 measured arm states passed the sphere/point model. At first
palm/door contact (control 150), the physical contact lay 3.21 mm outside the planned hand spheres
and 14.89 mm from the closest observed point; modeled clearance was 7.52 mm. At first finger/table
contact (156), modeled clearance was 0.84 mm. This demonstrates a remaining representation gap,
not a reproduction of the repaired tree defect. Fixed planning aperture was used in the audit;
the original whole-cloud hash was not archived, so reconstruction is not claimed hash-identical.
No sphere inflation or missing-surface completion was added in response.

The final two-case input audit passed for all 91 decisions and verified 117 unique supplied
images. Both used transient documentation and fresh episode memory. The watchdog recorded
123/123 successful SSH probes, zero OOM kills, at least 79.76 GiB host-available RAM, and a
6.64 GiB peak in anonymous/shared job memory. Total cgroup memory reached its 32 GiB cap mainly
through reclaimable file cache. Host I/O pressure peaked briefly at 13%, with responsive SSH.
The job stopped and GPU 3 returned to its previous 59,567 MiB use by other workloads.

After the frozen trials, one source comment was corrected to acknowledge the recorded CAPT
patch. Both binaries were rebuilt and the same 60 focused tests passed again. Numerical behavior
was unchanged by that comment cleanup. The full repository test suite was not run; validation
covered the changed schemas, actor availability, grasp selection and native collision behavior.

The repairs are useful engineering corrections, but **end-to-end success has not improved in
these fresh trials**. The next evidence-driven work is reliable contact acquisition and
selected-part motion feedback. More persistent documentation alone is not a demonstrated remedy,
and these results do not justify switching models, adding task-specific recipes or claiming that
practice will overcome the missing measurements.
