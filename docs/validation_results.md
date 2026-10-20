# Executed validation and initial pilot — 2026-10-11

The first usable Stage 1 implementation was exercised on the new `vlm-tamp` Conda environment. The machine was Ubuntu 22.04 under WSL2 with Python 3.10.22. The final full test run, with real LIBERO rendering enabled, passed **37 tests**. Static checks and `pip check` passed. Upstream deprecation warnings remain in the pinned legacy stack.

The collector rendered and validated 16 observations: four LIBERO-Goal tasks and four LIBERO-Spatial tasks, each with two official initial states. The dataset contains 112 visible whole-object masks. Independent collections of one official initial state produced byte-identical RGB images. An additional headless check passed with `DISPLAY` and `WAYLAND_DISPLAY` unset. One LIBERO-Object and one LIBERO-10 task also rendered and validated (15 masks combined).

Two offline prompt variants each produced and validated 64 predictions across language, RGB, candidate-inventory, and geometric-metadata settings. These are deterministic mock outputs and are not empirical VLM accuracy measurements.

## Real API pilot

The user authorized the supplied external credential file. A public-data provenance audit established that the outgoing instructions, images, and complete candidate inventories came from official LIBERO tasks/assets/initial states. Reference BDDL goals, private code, geometry, and annotation files were excluded. The initial automated review rejection was cleared after these provenance checks. Credentials were not printed or copied into the repository.

The run used `gpt-4.1-mini-2025-04-14`, `structured_v1`, four selected observations, and three conditions. It made **12 API calls**, without retries, using **21,883 input tokens** and **5,021 output tokens**. Applying the recorded configurable prices gives an **estimated cost of $0.0167868**, below the $0.50 admission budget. Ten responses passed Pydantic validation; two failed bounding-box validation. No further paid calls were made.

| Condition | Observations | Schema-valid | Predicate-type exact | Category-goal exact | Complete instance-goal exact |
|---|---:|---:|---:|---:|---:|
| Language only | 4 | 4 | 3/4 | 2/4 | 0/4 |
| Language + RGB | 4 | 2 | 2/4 | 1/4 | 0/4 |
| Language + RGB + candidates (oracle-assisted) | 4 | 4 | 3/4 | 3/4 | 0/4 |

These are measured counts with failed predictions retained in denominators. Predicate-type exact match does not establish object grounding. Complete instance-goal matching preserves benchmark identities and ordered arguments and can use unique independently measured pixel correspondence. No complete instance-level goal matched in this pilot. The four observations are insufficient for a statistically meaningful language/vision comparison.

## Qualitative errors and implementation feedback

- Candidate-assisted outputs often put candidate names in the local semantic `id` field while leaving `reference_id` null. The evaluator does not infer instance identity from this string coincidence. This exposes a field-role ambiguity in the initial schema/prompt as well as unresolved grounding.
- The combined top-drawer insertion task produced an extra final `Open` goal in the language condition. Official BDDL specifies the final containment goal; opening is a manipulation stage. In the candidate condition, the model instead used `On` where `In` was needed.
- RGB localization included inaccurate boxes/points. Two responses supplied invalid corner ordering. The initial prompt did not explicitly specify the box-array order; field descriptions and `structured_v2` now state `[x_min,y_min,x_max,y_max]` and distinguish local IDs from candidate references. **The revised prompt/schema descriptions have not been evaluated through paid calls.**
- Some object categories were broad superclasses (for example receptacle/appliance). No unrestricted semantic aliases were introduced to inflate category accuracy.
- Some plausible contact and articulation declarations appeared, but their correctness cannot be established from final BDDL goals. They remain predicted and unverified.

## Saved outputs and remaining coverage

- `data/pilot/records.jsonl`, `data/pilot/contact-sheet.png`: verified dataset and visual inspection.
- `results/offline/` and `results/offline-direct/`: separately labelled offline reports and galleries.
- `results/openai-pilot/`: actual raw responses, per-example predictions, metrics, task summaries, CSV/JSON aggregates, four-table Markdown report and HTML gallery. The original API-returned output schema and full executed prompt are retained in its manifest.
- `annotations/pilot/`: 16 draft templates, **zero reviewed** manipulation-constraint references. Track B accuracy is N/A.

Gemini's SDK adapter and types were tested without network calls; live Gemini access and accuracy remain untested. At this initial checkpoint, drawer-part masks, expert stage/dependency/path/contact annotations, larger paired experiments, other backbones and geometric compilation remained future research work. ST-pRRTC was not changed or integrated. Subsequent executed work is recorded in `continuation_results.md`, `backbone_results.md`, and `correspondence_results.md`; this document retains the initial pilot's historical results.
