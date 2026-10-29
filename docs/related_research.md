# Methodological references and comparison boundaries

For a controlled test of VISTA-inspired access to saved visual observations,
see the [Luna visual-inspection pilot](luna_visual_inspection_ab_20261025.md).
It adds an optional retrieval/cropping helper while freezing Luna, its guide,
planning code and budgets, and reports both task outcomes and actual tool use.

The [recorded-evidence loop diagnostic](luna_evidence_loop_20261026.md) then
tests explicit prediction and revision with identical before/after images.
Its [landmark-feedback follow-up](luna_landmark_feedback_20261026.md) tests a
compact summary of measured point displacements that the hinge-fit filter
otherwise omits. These are development diagnostics, not physical transfer tests.

The [full LIBERO comparison](luna_landmark_libero_20261026.md) then tests the
optional summary in complete opening/closing episodes. The
[visual-interface walkthrough](luna_visual_interface_walkthrough.md) explains
what is saved, what Luna actually receives, how `visual.py` works, and the
difference between a verbal expectation and the numerical hand-motion fallback.

For pretrained point-cloud/RGB-D methods that could replace free-form hinge
proposals, see the [articulation perception tools review (25 October 2026)](articulated_perception_tools_20261025.md).
It audits code/checkpoint availability and separates initial estimation,
motion-based verification and full reconstruction.

For planning with uncertain grasps, collisions, articulation and action outcomes,
see the [focused uncertainty-planning review (22 October 2026)](uncertainty_planning_review_20261022.md).
It also covers belief-space TAMP, plan ensembles and autonomous skill practice,
and separates the proposed cuRobo v2 backend from uncertainty estimation.

The initial sources were inspected on 2026-10-11. On 2026-10-12, released RoboPoint weights were evaluated through an RGB-role adapter, and public ReKep/ViLaIn grounding components were tried. See [the measured first runs](external_baselines_first_runs.md). Full published robot systems are not reproduced, and their reported success rates cannot be compared directly with this module-level LIBERO pilot. The other entries below remain representation/evaluation references.

| Work | Verified source / available implementation | Relevant design choice | Compatibility limit |
|---|---|---|---|
| ReKep | [Paper](https://arxiv.org/abs/2409.01652), [official code](https://github.com/huangwl18/ReKep) | Separate subgoal/path constraints, stages, visual keypoints and constraint provenance. | Released pipeline uses code-based 3D keypoint costs and task-space solvers; this Stage 1 stores typed semantic declarations and never executes generated code. |
| CoPa | [Paper](https://arxiv.org/abs/2403.08248), [official code](https://github.com/HaoxuHuang/copa) | Part-level grounding and task-aware grasp/motion constraints. | Code depends on SoM and GraspNet and includes real-world deployment; its execution setting is not an interchangeable LIBERO grounding evaluation. |
| A3VLM | [Paper](https://arxiv.org/abs/2406.07549), [official code](https://github.com/changhaonan/A3VLM) | Object-centric articulated-part affordances and articulation parameters. | Released model/training and manipulation primitives differ from cloud VLM grounding; part identity and numerical frames need compatible annotations before comparison. |
| RoboPoint | [Paper](https://arxiv.org/abs/2406.10721), [official code and evaluation instructions](https://github.com/wentaoyuan/RoboPoint) | Independently score language-conditioned image points and spatial affordances. | Fine-tuned model/datasets and its evaluation are not the LIBERO goal ontology. Adaptation needs target-role mappings and valid visible masks. |
| LIMP | [Paper](https://arxiv.org/abs/2402.11498), [project](https://robotlimp.github.io/) | Explicit symbolic instruction grounding and verification before planning. | The paper is titled “Verifiably Following Complex Robot Instructions with Foundation Models”; its real-world mobile navigation study differs from constrained manipulation. No runnable baseline is claimed here. |
| Lang2LTL-2 | [Official repository](https://github.com/jasonxyliu/Lang2LTL-2) | Separate lifted-command translation and referent grounding; module-level evaluation and synthetic task generation. | Spatiotemporal navigation commands and LTL semantics need a formal adaptation before use as manipulation baselines. |
| ViLaIn-TAMP | [Current paper](https://arxiv.org/abs/2506.03270), [public interpreter branch](https://github.com/omron-sinicx/ViLaIn-TAMP/tree/jeremy_dev) | Domain-grounded object/state/goal specification and planning-feedback repair. | The development-branch interpreter was ported for a small RGB-only component test; private proxy/ROS and fixed-fixture assumptions need explicit adaptation. Complete robot/TAMP execution is not reproduced. |
| GroundedPlanBench | [Official project](https://groundedplanning.github.io/GroundedPlanning/) | Joint action planning and spatial grounding, including long-horizon and implicit instructions. | The official code button says Coming Soon as of 2026-10-12; no runnable benchmark or annotated-data release was found through official channels. |

The current architecture borrows the distinction between semantic declarations, geometric instantiation, and execution from these directions. A3VLM motivates explicit part/joint frames; RoboPoint motivates pixel metrics; formal-specification work motivates preserving logical structure and reporting unsupported constructs. ReKep/CoPa motivate stage-scoped goal/path constraints, but their complete planners are outside Stage 1.

Cloud adapters follow [OpenAI Responses](https://developers.openai.com/api/reference/python/resources/responses), [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses), and [image inputs](https://developers.openai.com/api/docs/guides/images-vision). The pinned pilot model and configurable rates were checked against the [GPT-4.1 Mini model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini). Gemini uses the official SDK and [structured-output guidance](https://ai.google.dev/gemini-api/docs/structured-output); its live API access and accuracy are not yet tested.

For implemented Luna-selected part masks, calibrated associations, CoTracker/LK tracking and independently initialized motion estimates, see [the part/motion tools and replay validation](part_motion_tools_20261027.md). That report distinguishes tool consistency, measured-motion support, native grading and physical task success.
