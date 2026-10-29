# Local Qwen3.5-9B capability pilot

The local BF16 model runs on the server, but it did not qualify for a complete
robot episode under this pilot's grounding gate. Luna remains the primary actor.
No simulation episode, weight update, memory promotion, or Reflection reviewer
was used in this test.

[Annotated input/output report](../results/qwen9b-pilot-20261024/watch.html) ·
[Study manifest](../results/qwen9b-pilot-20261024/study-manifest.json) ·
[Machine-readable findings](../results/qwen9b-pilot-20261024/analysis.json).

## Model version

Qwen3.5 is not the newest generation. The official catalog checked on
2026-10-24 lists Qwen3.8, including Qwen3.8-27B. It did not list an official
Qwen3.8-9B. This test deliberately used the previously proposed small model,
`Qwen/Qwen3.5-9B`, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

Sources: [official Qwen catalog](https://huggingface.co/Qwen),
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B),
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

## Results and limits

- Six of six short text/feedback checks passed: open-finger approach arguments,
  control-budget rejection versus geometric infeasibility, and robot execution
  versus established object motion. These were three questions with two samples
  each, not six independent manipulation tasks.
- Zero of six full image-grounding proposals met the registered structure gate.
  Failures included zero direction vectors, incorrect operation/argument nesting,
  out-of-image coordinates, and one response truncated at 4,096 output tokens.
  The plotted regions also often missed the visible handle. No proposed action
  was sent to the robot service.
- A simpler handle-box diagnostic varied one/two images and pixel/normalized
  coordinates, with two samples per condition. Zero of eight met the registered
  approximate-human-box criterion: centre inside the handle and IoU at least
  0.30. This binary count conceals useful variation. One normalized single-image
  box was about 11 pixels from the reference centre and had IoU 0.274, narrowly
  below that threshold. Pixel-coordinate outputs were often much farther away,
  out of bounds, or unfinished. Three responses hit the output-token limit.
- Disabling thinking on the two normalized single-image queries reduced request
  times to about 0.79 seconds each, but the boxes still missed the handle. The
  paired-input audit verifies that only the thinking flag changed.
- A separate deterministic check through Hugging Face Transformers reproduced
  a similar misplaced box. Both runtimes used 706 prompt tokens; the reference
  image grid was 1×32×32 before spatial merging. This makes a vLLM-only serving
  error less likely. It does not rule out every shared preprocessing issue or
  establish an intrinsic model-size limit.

The visual cases come from previously evaluated states of the same microwave
asset with very similar initial views. The coordinate experiment uses one
scene and an approximate human reference. These are development diagnostics,
not held-out generalization, learning-transfer measurements, or a matched
comparison with Luna's Codex harness.

## Concrete interface issue

The current `ground` help lists `initial_image_direction_xy` without defining
its meaning. In its generated explanation, Qwen described this as camera
orientation and supplied `[0, 0]`. The implementation uses the vector to choose
the sign of the contact point's projected motion. In the handle-contact path,
a zero vector produces a tie and silently selects the negative axis.

Follow-up clarification: in the normal live service, the existing adjacent-depth
rule overrides that preliminary image-based axis. A zero-vector fallback was
therefore not established as the cause of Luna's earlier physical failures.
The shared contract and validation have since been repaired; see the
[I/O repair and retest](robot_io_contract_20261024.md). The results below retain
the original pilot conditions.

The contract should explain the intended contact-point motion, coordinate axes,
and unresolved-direction behavior. Exact operation schemas should also remove
ambiguity about nesting `regions`. These are general interface improvements;
they require no task-specific coordinates. Neither documentation nor numerical
behavior was changed during these diagnostics. A stronger model could hide
these ambiguities, so this evidence does not justify blaming model size alone.

## Memory and timing

The official checkpoint contains about 19.3 GB of weight tensors. vLLM reported
17.57 GiB for model loading and reserved about 32.55 GiB across its own GPU
processes with the configured cache budget. Other users' allocations were
excluded. Reserved serving memory is not a measured minimum requirement.

The separate Transformers request peaked at approximately 17.83 GiB of
PyTorch-allocated memory and 17.88 GiB of PyTorch-reserved memory. These allocator
measurements exclude some CUDA/driver overhead and are not directly identical
to the vLLM NVML process measurement.

The twelve initial model requests used 268.02 seconds in total. Text/feedback
requests took 4.38–10.63 seconds; full grounding requests took 10.33–60.89 seconds.
The eight coordinate-format requests used another 289.26 seconds. These times
include request handling and generation, with shared GPU load and cache effects.
They are not direct comparisons with Luna's full-episode reasoning time.
Grasp planning, IK, arc planning and physical execution were not run.

## Reproduction and retained setup

The model is in `.runtime/models/Qwen3.5-9B`; its downloaded files and LFS hashes
were verified. The dedicated serving environment is `.runtime/envs/qwen-vllm`
(vLLM 0.32.0, Torch 2.13.0+cu130, Transformers 5.19.0). Existing robotics
environments were not modified. Exact dependencies are in the runtime freeze.

Start the server with `scripts/serve_local_qwen.py`, using the serving
environment's Python and a new `--output` directory. The tested configuration
uses GPU 2, port 8127, 32,768 context tokens, one active sequence, BF16, and
34% GPU memory utilization. Check GPU availability before another run.
The launcher supplies its environment's `bin` directory and CUDA 13.0 toolkit.
The first startup lacked `ninja` on PATH and failed while compiling FlashInfer's
sampling kernel; that failed startup is preserved separately.

`scripts/run_qwen_capability_probe.py` and
`scripts/probe_qwen_coordinate_format.py` reproduce the two registered offline
studies using new result directories. `scripts/check_qwen_reference.py` performs
the independent-runtime check. Exact requests, image hashes, outputs, token
usage and timings remain in each response directory. Every failed sample is
retained; there were no model retries in either registered suite.

Fifteen focused adapter/interface tests passed. Real OS-sandbox preflights
verified numerical analysis, read-only supplied inputs, blocked private
evaluation reads, and the public status wrapper. The first no-service preflight
fixture omitted the directories normally created by the robot service; the
corrected fixture passed without a production-code change.

`scripts/run_local_agent_pilot.py` prepares a bounded full simulator pilot using
the existing numerical stack. It was not exercised end to end because the
registered grounding gate failed. Treat it as experimental until a complete
integration trial is validated. The existing Luna runner and default model
remain unchanged. Evaluation annotations and these coordinates must not become
learner memory.
