# Robot-agent literature audit — 2026-11-04

Scope: compare information, action interfaces, learning artifacts and evaluation protocols.
This was a paper/source review, not a benchmark reproduction. No runtime, planner, prompt,
sensor contract or experiment was changed. The user's intended learner makes direct tool
calls and retains advisory experience; it does not generate executable policies.

## Primary-source index

| Work | Evidence inspected | Boundary to retain |
| --- | --- | --- |
| [OpenETA](https://arxiv.org/html/2608.03924v1) | Sections 5.1.2, 5.2.2 and B.8; released interface linked below | Direct tools; matched model comparison; distinct observation and episode budgets. |
| [GPT-Policy](https://arxiv.org/html/2609.19138v1) | Sections 3.2–3.3 and Appendix A | Frozen-model tool requests; reference context and online experience are separate. Cartesian planning does not provide scene collision checking. |
| [RoboICL](https://arxiv.org/html/2609.34261v1) | Sections 3.1–3.4 and 4.1 | Direct action chunks with bounded visual transitions; demonstrations and live history have separate provenance. Main one-shot results include task demonstrations. |
| [RoboAuto](https://arxiv.org/html/2610.36533v1) | Appendices A.2–A.3 and C.1–C.3, Figure 12 | Initial knowledge comes from successful LIBERO-90 videos. It combines object knowledge, program search and independent video review. |
| [RoboRSI](https://arxiv.org/html/2610.12424v1) | Method, physical-development protocol and released door implementation | Skill-code revision, released compound policies and human corrections differ from fixed-tool memory learning. |
| [ASPIRE](https://arxiv.org/html/2607.00272v1) | Sections 2.1–2.3 and Appendix E | Program repair and a reusable library; per-primitive traces support diagnosis. Its protocol forbids hidden simulator-state access. |
| [ENPIRE](https://arxiv.org/html/2606.19980v1) | Environment construction and Appendices A–B; [official results](https://research.nvidia.com/labs/gear/enpire/) | Agents develop policies/training and reset/verifier infrastructure. The site's 99% claim is explicitly recovery-based pass@8. |
| [MOKA](https://arxiv.org/html/2403.03174v1) | Sections IV-A–IV-E and Appendix B-D | Marked choices specify contact and surrounding motion. Its execution has tabletop/4DoF assumptions; it is not a universal hinged-object controller. |
| [FORCA](https://tanoom.github.io/forca/) | Official method and experimental results page | Force-feedback programs and a wrist F/T sensor; those observations are absent from our current actor interface. |

## Source-level checks

- RoboRSI's [released door policy](https://github.com/nssmd/RoboRSI/blob/90c8569705bf4004a4f79e34e650c87ab7f075d4/roborsi/embodied/skills/base/open_hinged_door/libero/policy.py)
  was checked at commit `90c8569705bf4004a4f79e34e650c87ab7f075d4`. The routine explicitly
  assumes a vertical hinge, requests a left/right hinge side and uses five-degree arc increments.
  A released base routine does not establish who originally authored it or its exact evaluation
  lineage. The prior broader local audit is `roborsi_code_review_20261031.md`.
- OpenETA's old `openeta-light` branch link redirects to `openeta-for-codex`.
  Its [interface document](https://github.com/OpenMOSS/OpenETA/blob/openeta-for-codex/docs/openeta-light.md)
  was retrieved with blob SHA `4c97404dbfb421cc5ed69f5f322cdc9fd95e1639`.
  Orthographic views are described as point-cloud projections. RGB-D marking uses aligned
  depth. Move previews expose gripper geometry, and not-reached feedback may show native
  contact locations. Do not describe the projections as extra physical cameras, or assume
  they repair transparent-surface depth. Exact experimental-release parity was not reproduced.

## Interpretation for this project

The literature does not justify another unmeasured microwave-specific correction. Our review
should distinguish four questions: whether the actor identifies the movable part, specifies the
intended object-state change, chooses an appropriate contact/model, and executes that request.
A successful downstream IK result cannot validate the earlier semantic choices.

The useful comparison is a fixed backend and matched information/budgets, with direct tool
calls, explicit contact-pose previews and a bounded record of actual visual transitions. A
minimal interface can be tested against the existing interface without first replacing the
motion planner. Neither fewer tools nor richer images should be assumed to improve performance
without a controlled comparison.

Keep any learned memory non-executable and distinguish supplied tool documentation from
learner-authored discoveries. Geometry must be measured again. Practice and evaluation must
be separated by more than two initial states of the same appliance. A new native-contact
sensor, demonstration source, additional observation or generated-policy layer would change
the experiment and requires an explicit decision; this review does not introduce one.

## Reporting discipline

- Authors' reported success, progress scores, per-episode success and pass-at-k are different
  quantities. Do not compare them as a single leaderboard.
- Model identities and reasoning settings differ. Results for GPT-5.6 Luna do not directly
  measure this project's GPT-6 Luna.
- A simulator-step horizon needs a timebase audit before comparison with our control counts.
- Demonstration-derived knowledge is additional task information even when weights are frozen.
- A promising interface or memory design is a hypothesis for our setup, not a reproduced fix.

Validation: primary papers, official project pages and selected released source were read;
documentation whitespace was checked. No GPU jobs, robot trials or implementation tests were
run for this literature-only change.
