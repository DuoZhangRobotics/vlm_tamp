# Why the scene-cleared grasp closed empty

This is a read-only audit of
`results/small-graspgen-scene-contact-seed1-metric-20261029/open_microwave-state037`.
No runtime behavior, grasp poses, contact rules or motion planning checks were
changed. No new simulator episode or Luna evaluation was run.

The failure is in **grasp proposal selection/acceptance**. The selected pose is
reachable and clear of forbidden scene collisions, but the handle is outside
the volume swept by the actual closing pads.

## Direct evidence

At the intended learned pose, restricting the observed handle cloud to the
pad-height band and the open-jaw span gives 425 points. The nearest surface is
**2.240 mm beyond the pad tips**. There are **zero observed handle points in
the pad closing volume**. Closing moves the pads sideways, not forward toward
this surface.

The earlier 22.0 mm number was pad-midpoint-to-fitted-handle-axis distance. It
was not the minimum surface clearance at the pad tips. The 2.24 mm measurement
describes the miss more directly.

The eight proposals were ranked by discriminator score:

| Rank | Score | Points in calibrated pad closing volume | Selection outcome |
|---|---:|---:|---|
| 1 | 0.986 | 181 | Scene collision |
| 2 | 0.978 | 197 | Scene collision |
| 3 | 0.968 | 226 | Scene collision |
| 4 | 0.967 | 194 | Scene collision |
| 5 | 0.852 | 353 | Scene collision |
| 6 | 0.730 | 405 | Scene collision |
| 7 | 0.551 | 341 | Scene collision |
| 8 | 0.206 | **0** | Accepted, then empty closure |

These scores are not calibrated probabilities. Points inside the closing
volume are useful geometric evidence; their count does not certify friction,
antipodal contact or physical retention. Under occlusion, missing points can
mean unknown geometry rather than an empty grasp. Here, the recorded empty
closure independently corroborates the geometric diagnosis.

The sampler uses `threshold=-1`, retaining all eight candidates. The selector
tries them in score order and stops at the first scene-valid reachable pose.
It checks open/closed-aperture collisions, budget and approach existence, but
does not require positive evidence that closure will capture the target.
Thus a clear empty-hand pose can pass all current acceptance conditions.

## Other explanations checked

- **Frame conversion:** trained and configured finger-body mesh vertices align
  within 2.6e-9 m after the existing attachment/TCP conversion. The 6 mm asset
  correction is not the source of this miss.
- **IK:** the selected IK target differs from the original requested grasp
  position by only 3.87e-8 m. It did not discard the learned translation.
- **Controller:** measured arrival error relative to that IK target is 0.321 mm.
  This leaves a roughly 1.983 mm pad-tip gap against the original cloud.
- **Object motion:** the recorded door moved 0.202 degrees during approach,
  moving input-cloud points by at most 0.898 mm. A separately labelled offline
  calculation using the native joint record gives a 2.765 mm pad-tip gap at
  arrival. The original pose already missed the pad volume before this motion.
- **Closure:** all 40 closure controls were issued. There were no recorded
  robot/environment contacts during those controls, and the pads closed to
  essentially zero gap. The earlier finger-pad contact happened during approach,
  at controls 103–112, not while pinching at controls 126–165.

GraspGen's supplied collision mesh covers the full finger body. Its tip extends
4.849 mm beyond the configured rubber pad's front edge after the correct frame
conversion. This distinction matters for near-tip proposals: agreement of the
full finger frames does not establish a pad-supported pinch. The selected pose
has very little observed support near the body tips and none in the pads.
Do not turn this into a claim that GraspGen always fails or that all finger-body
contacts are physically disabled; the recorded failure is specific to this pose.

## Next change to test

Add positive grasp-capture evidence using the calibrated pads and current
target cloud before accepting a proposal. A batch with no supported pose should
report that outcome and allow another small learned batch. This need not alter
pose translations, add a handcrafted grasp backup, use an arbitrary minimum
score, or restore dense motion-path validation. A supported candidate still
needs measured closure/retention feedback.

## Artifacts and reproduction

- `results/empty-grasp-audit-20261030/watch.html`: explanation and geometry plot.
- `results/empty-grasp-audit-20261030/audit.json`: all measurements/provenance.
- `results/empty-grasp-audit-20261030/empty-grasp.png` and `.pdf`: standalone plot.
- `scripts/audit_empty_grasp.py`: CPU-only reproducible audit.

Run after sourcing `scripts/server_runtime_env.sh`:

```bash
python scripts/audit_empty_grasp.py \
  --run results/small-graspgen-scene-contact-seed1-metric-20261029/open_microwave-state037 \
  --output results/empty-grasp-audit-20261030
```

The native joint/contact record is used only for post-run explanation. It is
not supplied to planners or actors. The pad-capture calculation itself uses
only recorded RGB-D and calibrated robot geometry.
