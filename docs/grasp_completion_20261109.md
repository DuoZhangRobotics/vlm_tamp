# Mechanical closure and explicit regrasp

The old grasp executor equated “all requested closing controls were sent” with “closure completed”.
An eight-control close left roughly 20 mm between the pads in the recorded cases, so subsequent
arm motion could begin while the fingers were still moving. A new grasp request then automatically
opened the hand for its approach, even when the previous grasp had never been tested by lifting.

Closing now holds the existing arm reference until both the closing-reference ramp and measured
finger motion settle. The criterion is at most 0.1 mm finger travel over a 20 Hz control interval
and at most 2 mm/s instantaneous speed for each finger, with an unchanged closing reference.
These are explicit mechanical reporting tolerances, not collision padding or object-contact tests.
The recorded cases first satisfied them after 14 and 15 controls. No fixed extra waiting period
was added. The previous 40-control default supplies the ordinary closing ceiling; a larger
caller-requested minimum increases it, and the remaining episode budget always applies.

Automatic `plan_grasp` / `grasp_object` closing defaults to a one-control minimum, followed by
measured completion. Existing explicit `close_steps` values remain minimum intervals. Estimates
are labeled lower bounds because settling can require additional controls. A cutoff reports
unfinished closure and prevents subsequent transport until closing finishes or an explicit open
abandons it. Native task-success stopping remains unchanged. Grasp-and-arc sequence closure uses
the same measured primitive.

A completed closing attempt is preserved across selections and planning requests. Calling a new
grasp does not silently release it: use `regrasp=true` on `plan_grasp` / `grasp_object`, or explicitly
open first. This is an intent transition, not a grasp-success certificate. The compact feedback and
actual callable contracts distinguish mechanical completion, retention not yet tested, and the fact
that a grasp command contains no lift. Luna still chooses transport, a lift test or deliberate retry.

## Targeted validation

Report and videos: `results/grasp-completion-20261109/watch.html`.
No new Luna batch was started. Recorded initial arm states reproduced exactly in all replays.

- State 2: measured closure took 14 controls and a subsequent roughly 97 mm upward motion retained
  the bowl at both pads. A repeat-grasp request preserved the hand and issued no motion.
- The preliminary legacy state-2 IK probe executed only a roughly 1 mm prefix, so it is not scored
  as a failed full lift. Its planning result is preserved separately.
- A separate fixed-arm-trajectory comparison held the exact arm commands constant across legacy
  eight-control and measured closure. Both retained the bowl. This confirms that this particular
  recorded grasp was usable when committed to the lift; it does not prove every short close is safe.
- State 32: the recorded pose lost retention after the roughly 97 mm lift with measured completion.
  Repeating the same arm trajectory with an explicit 40-control close also lost retention. Further
  waiting did not repair that grasp. Contacts at closing involved finger bodies; there were no
  target contacts after lifting. This remains a grasp-quality issue requiring a separate diagnosis.
- Simulator contacts are used only in these private diagnostic scores. Closure control uses only
  finger joint measurements and the robot command reference; no force sensing or object inventory.

133 focused Python tests passed. Tests cover command-ramp completion, moving fingers, early completion without waiting to
the ceiling, budget exhaustion, fixed arm references, explicit regrasp, compact feedback, grasp
execution, and sequence integration. Ruff and diff checks pass. No native planner code changed,
so no native rebuild was needed. The previous 10/15 Luna score applies to the earlier frozen code;
this change has not received a new end-to-end policy success-rate evaluation.

Resource monitoring recorded no OOM kills or failed SSH probes. Peak job memory was 4.73 GiB
including cache; peak active process/shared memory was 2.42 GiB. Shared-host I/O pressure reached
6.41% during the preliminary replay session. All diagnostic workers exited after their fixed runs.
