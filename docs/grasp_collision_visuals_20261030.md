# Visual inspection of the rejected grasp poses

The user requested pictures to judge the reported “scene collisions.” Scene
means the observed environment; it does not mean a sink. The pictures are
planner predictions at hypothetical grasp endpoints, not executed collisions.

Viewer: `results/grasp-collision-visuals-20261030/watch.html`, using the existing
VS Code Live Preview workflow. It provides eight candidate choices, open vs
predicted holding aperture, rotatable 3D geometry, a contact zoom button, and
camera overlays. The Plotly library is included locally; no external server or
CDN is needed. PNG sheets `candidate-1.png` through `candidate-7.png` are also
available. The accepted candidate 8 is included in the 3D viewer.

## Findings

These are the closest saved IK branches of the eight-pose seed-1 batch from
`small-graspgen-scene-contact-seed1-metric-20261029/open_microwave-state037`.

| Candidate | Shown aperture | First rejected volume | Observed points inside | Maximum inward face distance |
|---|---|---|---:|---:|
| 1 | Predicted holding | Finger 2 body, selected handle | **2** | **0.479 mm** |
| 2 | Open | Arm link 6, stationary scene | 317 | 27.425 mm |
| 3 | Open | Arm link 6, stationary scene | 671 | 43.553 mm |
| 4 | Open | Arm link 5, stationary scene | 422 | 50.400 mm |
| 5 | Open | Finger 2 body, stationary scene | 2 | 2.898 mm |
| 6 | Open | Finger 1 body, moving surface outside selected contact patch | 48 | 9.102 mm |
| 7 | Open | Finger 2 body, moving surface outside selected contact patch | 24 | 8.460 mm |

The highest-scored candidate (1) passes at the open-hand endpoint. Its rejection
appears at the predicted holding width, for two points near the handle. This
is a marginal local contact prediction, not evidence of a large arm/palm
obstruction. It deserves specific contact-policy/physical investigation before
being treated as categorically unusable. The visualized arm-link-6 overlap in
candidate 2 lies on the bowl visible in the recorded RGB. This object label is
a human-readable interpretation of the image, not planner-provided object state.

Do not group all seven rejected poses into one severity or failure category.
Also do not conclude that the small finger overlap is harmless without a
targeted physical test. No rejected candidate was physically executed here,
and no collision permission or runtime setting was changed.

## How to interpret the pictures

- Cyan is the proposed robot geometry. In camera overlays, the RGB background
  still contains the original robot in its initial pose.
- Orange is the first reported rejected robot volume (or clipped finger piece).
- Red marks actual observed points recovered inside that volume.
- Green pads in 3D indicate the pad geometry, whose selected-target contact is
  permitted; they do not exempt nearby proximal finger body.
- The “closed” label in source logs means the predicted holding aperture,
  calculated from observed handle radius, not fully shut empty fingers.
- The distance is a convex halfspace margin, including contact-permission cut
  planes. It is not a certificate of real object penetration or contact force.
- Only the closest saved IK branch is visualized for each pose. The original
  search tested additional branches, as listed in the viewer. Highlighting the
  first blocking pair does not enumerate every possible collision.

## Reproduction and verification

`scripts/visualize_grasp_scene_contacts.py` loads the exact exported initial
scene clouds and saved candidate joints. It reconstructs open/holding robot
geometry using calibrated prismatic finger and arm kinematics. It does not
start a simulator or regenerate IK/grasp candidates.

All three native cloud counts match the recorded context (43,607 stationary,
2,692 moving noncontact, 1,067 selected contact points). **All 16 native endpoint
results match** for acceptance, first rejected robot part and surface category.
Gripper hit extraction uses the saved float32 native convex pieces. Arm hit
extraction uses the calibrated convex volumes at the saved joints. No positive
clearance padding is involved in these reported intersections.

The viewer was rendered in headless Chrome with WebGL software rendering. The
PNG overlays were visually inspected. Raw geometry, distances, reference scenes
and the reconstructed native inputs are retained under the viewer directory.

Run after sourcing `scripts/server_runtime_env.sh`:

```bash
python scripts/visualize_grasp_scene_contacts.py \
  --run results/small-graspgen-scene-contact-seed1-metric-20261029/open_microwave-state037 \
  --output results/grasp-collision-visuals-20261030
```
