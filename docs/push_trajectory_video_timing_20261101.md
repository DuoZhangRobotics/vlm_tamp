# Push trajectory and recording timing repair

The previous 4090 study used source `60b21b6`. Closing state 49 initially moved
in the correct direction, from 107.3 degrees open to 29.6 degrees. At control
297 a requested 22.97 mm stroke crossed IK posture families. Joint interpolation
made the configured hand landmark travel 449 mm and deviate 153 mm from the
requested line. The issued prefix consumed the remaining 303 controls. Native
contact ended at 329; door motion stopped at 340. These are evaluator diagnostics,
not information supplied to Luna.

Controlled push now uses a dedicated cuRobo v2 contact-point solver with one
seed: the immediately preceding robot posture. The selected robot contact
offset is part of the calibrated model. Wrist orientation is a soft preference;
the contact point remains the motion objective. The solver is reused for that
contact offset. Grasp and free-approach IK retain their existing implementation.
No collision buffer, branch-jump rejection gate, dense validation pass or
fallback policy was introduced. Existing joint limits, speed limits and episode
budgets remain in effect.

The recorded control-297 numerical regression on the RTX 4090 produced:

- 16 controls, 22.969 mm reference-path length;
- 0.0282 mm maximum sideways deviation and 0.0084 mm endpoint error;
- 2.64 degrees of wrist-orientation adjustment;
- 0.063 seconds for path generation and timing after 1.62 seconds of initial
  solver setup.

This regression issued no simulator motion and supplied no case data to Luna.
Evidence is under `results/push-path-continuation-check-20261101` on the 4090.

Previously, recording saved every fifth initialization tick and every individual
joint tick, then encoded everything at four FPS. State 49 therefore encoded 553
images as 138.25 seconds despite 600 counted ticks at 20 Hz. The shared encoder
now uses control-step timestamps and 20 FPS, holding the preceding image across
uncaptured intervals. The initial frame adds one 50 ms frame to the counted
simulation duration. The index distinguishes recorded source steps from playback
steps. Reading retained evidence skips repeated display frames, so they cannot
be mistaken for new observations. The worker and post-run compaction use the
same encoder. The dashboard distinguishes simulation seconds from run time.

Validation: 121 targeted contact, cuRobo, storage/video, timing and dashboard
tests passed using the existing native dependencies. The first invocation lacked
the VAMP import path; the corrected invocation passed all tests.

Fresh evaluation destination: `results/luna-push-trajectory-video-fix-20261101`.
Run states 17 and 49 sequentially with Luna/high, no inherited memory, 600
controls and 1800 wall seconds per episode. Use the existing 16 GiB/no-swap
watchdog, GPU monitoring and once-per-minute dashboard mirror. Read the study
manifest for the actual result; numerical regression alone is not task success.
