# Molmo2 pointing diagnostic, 2026-11-03

The user requested Molmo2 pointing and clarified that window contact should not be treated as
intrinsically wrong. This experiment accepts window marks as moving-door selections and records
raw-depth measurements separately. No robot motion, IK, contact force or task completion was tested.

## Inputs and protocol

Model: `allenai/Molmo2-8B`, revision `e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b`.
The official FP32 checkpoint was loaded in BF16 without quantization on arcla GPU 0. Transformers
4.57.1 and Accelerate 1.11.0 use the reviewed, pinned model code. The isolated Molmo2 environment
reuses the existing Qwen environment's PyTorch 2.13.0/CUDA 13 runtime through a documented Python
path entry; the Qwen and robotics environments were not modified. vLLM is not used by this test.

The panel was frozen before inference: five recorded state-17 instants (`obs001`, `obs029`,
`obs041`, `obs051`, `obs068`) with the external context and `arm_lower_right` image, plus the initial
wrist image. Each pair is simultaneous, not a before/after sequence. The opposite-side camera is
an extra recorded diagnostic view, outside the production two-camera benchmark. Every request
starts a fresh conversation while reusing loaded weights. Greedy decoding uses at most 512 new
tokens; none reached the limit. Identical greedy repeats were not used as a consistency claim.

Two predeclared prompts distinguish locating the door from selecting a closing-contact pixel:

- “Point to the microwave door in image 2.”
- “Point to ONE visible contact location in image 2 where a finger could push the door toward its
  closed configuration,” with the same non-pinching, visible-contact and untested-reachability
  qualifications as the prior experiment.

The complete prompts are in `results/molmo2-pointing-20261103/protocol.json`. These are supplied
operator requests, not Luna-authored discoveries. Molmo2 receives only the two RGB images and
text, without earlier answers, successful coordinates, depth, masks, hinge geometry or simulator
state. No tool cascade, point replacement or input-dependent repair is implemented.

The initial side-view input hashes match the previous Luna/Sol comparison. The prompt and native
pointing output differ from their agent/tool/JSON protocol. Results therefore do **not** establish
a matched model ranking. The five instants also come from one episode, not five independent tasks.

## Results

All 12 outputs contain exactly one native point in the requested image. Coordinates use Molmo2's
one-based image index and 0–1000 scale, then map directly to the original 512×512 image. No point
is clipped, replaced or silently discarded. Raw tags and floating-point pixels are retained;
operator-only depth audits explicitly sample the nearest integer pixel without neighborhood search.

| Capture / camera | Locate-door pixel | Image-space label | Closing-contact pixel | Image-space label |
| --- | --- | --- | --- | --- |
| obs001 / side | (310.784,364.032) | Door window | (352.768,333.824) | Door handle |
| obs029 / side | (245.248,353.792) | **Fixed housing** | (279.552,342.528) | Door window |
| obs041 / side | (283.648,357.888) | Door window | (281.088,357.888) | Door window |
| obs051 / side | (281.088,357.888) | Door window | (281.088,352.768) | Door window |
| obs068 / side | (290.816,368.128) | Door window | (282.624,373.248) | Door handle |
| obs001 / wrist | (255.488,473.088) | **Robot foreground** | (200.704,293.888) | Door upper edge |

The closing-contact request placed **6/6 marks on the moving door**: three window, two handle,
one upper-edge. The object-location request placed **4/6 on the door**. Its errors are visible
wrong-object choices, not parser failures: the obs029 point samples unchanged fixed housing
between initial/later captures, and the wrist point lies on the nearby robot foreground.

This is promising evidence for a task-conditioned pointing tool, not proof that the selected
contact produces closure. The requested contact camera is already assigned. This experiment does
not establish that Molmo2 selects the correct door face, infers CW/CCW, or supplies an executable
push direction. The wrist upper-edge point in particular does not establish a useful force direction.

Window marks count as correct image-space door selections. Their raw depth samples still reach
background in this recording; that fact is displayed separately and is not converted into a model
error or execution gate. Handle/edge points supply observed object-surface depth. No plane repair,
surface completion or automatic snapping was added.

## Timing and shared-server health

- One-time 34.65 GB checkpoint download: 897.14 seconds, sequential transfer, bounded write rate.
- Model load: 34.35 seconds; imports plus loading: 44.64 seconds.
- First query: 5.57 seconds, including first-use GPU work.
- Subsequent-query median: **0.786 seconds**; median preprocessing 0.046 seconds and generation
  0.740 seconds. The separately computed component medians need not sum exactly to the total median.
- All model generation: 13.56 seconds. Imports, loading and all 12 requests: 59.08 seconds.
- GPU peak: **20.29 GiB**. Inference anonymous/shared host-memory peak: 1.38 GiB; cgroup memory
  including reclaimable model-file cache reached the 32 GiB limit. This is not a 1.38 GiB total
  host-memory requirement. There were no OOM kills or resource-monitor interruptions.
- Download and inference used the existing 32 GiB/zero-swap watchdog. **63/63 SSH probes passed**.
  Permanent compact telemetry, peak I/O pressure and separate download/inference statistics are in
  `resource-summary.json` and the two resource subdirectories.

The prior Luna/Sol times include agent startup, model calls and tool use, so a direct speed ratio
would compare different interfaces. The warm Molmo2 number measures this two-image pointing call.

## Artifacts and integration boundary

- Results: `results/molmo2-pointing-20261103/watch.html` (about 53 KB; no auto-refresh).
- Each query has exact prompt, raw response, parsed points, rendered chat template, tensor shapes,
  token counts, timing and operator depth audit. The report shows both source images and every mark.
- Code: `src/vlm_tamp/pointing_molmo2.py`, `scripts/probe_molmo2_contact.py`,
  `scripts/report_molmo2_pointing.py`, `templates/molmo2_pointing.html`.
- The reusable pointer loads once per instance, uses one explicitly selected CUDA device and
  reports model-format outcomes without selecting a substitute backend or point.
- Production Luna tools and the robot planner/controller are unchanged. A next integration can
  expose a current-view semantic pointing request to Luna and return unmodified points/overlays.
  That should be evaluated with Luna's own requests before claiming autonomous grounding success.

Validation: 13 relevant parser/adapter tests passed, along with Ruff. Browser checks verified all
12 plotted coordinates, all 36 per-query document links and a 360-pixel layout. No physical trial
was run because this request tested pointing on the recorded inputs.

Primary model/format source: [official Molmo2-8B model card](https://huggingface.co/allenai/Molmo2-8B).

The subsequent integration into Luna's actual callable tools and the three physical evaluations
are documented in [molmo2_pipeline_20261103.md](molmo2_pipeline_20261103.md). That later experiment
uses Luna-authored queries and the two native cameras; its outcomes must not be conflated with
the image-only pointing counts above.
