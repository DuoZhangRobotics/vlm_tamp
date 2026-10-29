# Qwen output allowance and Luna budget comparison

Later follow-up: the [sampling and tool-contract diagnostic](qwen_sampling_diagnostic_20261025.md)
resolved these three stalls using the checkpoint's general-thinking sampling
recipe while retaining 32k output. Spatial grounding remained inaccurate.
The results below preserve the earlier output-budget experiment.

New local Qwen runs default to **32,768 generated tokens per request**, including
reasoning and tool output. The local serving window is **65,536 total tokens**,
and the request guard is **900 seconds**. A full actor episode still has its
separate 1,800-second / 600-control budget. The user requested at least 32k;
the earlier 4k pilot allowance must not become the new default again.

[Live results and model outputs](../results/qwen32k-budget-20261025/watch.html) ·
[Registered test](../results/qwen32k-budget-20261025/study/protocol.json) ·
[Luna configuration audit](../results/qwen32k-budget-20261025/luna-budget-audit.json).

## What “same as Luna” establishes

The audited Luna launcher and its recorded initial/resume commands specify
`gpt-6-luna`, high reasoning, ChatGPT authentication and the episode budgets.
They do not explicitly set a response-token cap, context size or compaction
threshold. The effective subscription limits are therefore unknown.

The [official Luna API card](https://developers.openai.com/api/docs/models/gpt-6-luna)
advertises a 128,000-token output maximum and a 1,050,000-token context window.
These API maxima do not establish the settings used by the Codex subscription
route. CLI turn/session usage can aggregate several model/tool interactions;
it is not a per-response limit. The new Qwen setting implements the requested
32k allowance and is not a claim of verified token-budget parity.

## Configuration and reproducibility

`LocalChatClient` and the local full-actor runner now share the larger defaults.
The full-actor runner exposes `--max-output-tokens` and `--model-call-seconds`,
and records the actual context advertised by its server. The server launcher
exposes `--context-tokens` and `--gpu-memory-utilization`; increasing the
response allowance also requires enough room for its input.

The three historical 4k diagnostic drivers explicitly retain `max_tokens=4096`
to preserve their registered reproduction conditions. For new budget tests use
`scripts/retest_qwen_output_budget.py` and a new results directory. Do not reuse
an old result directory or interpret the old drivers as the current defaults.

Streaming calls now retain live elapsed/character/event counts and preserve
partial diagnostic output on exceptions. Character counts are not token usage,
and partial output is not a completed action.

Each new request is measured through the pinned vLLM `/tokenize` endpoint
before generation. The full prompt plus the requested output must fit the
actual serving context. This is necessary because vLLM otherwise silently
reduces output to the remaining context capacity. The client reports a
`ContextBudgetExceeded` error rather than reducing the requested allowance.
A real 40,012-token synthetic prompt was rejected before generation: it would
have left only 25,524 output tokens in the 65,536-token context.

The full actor also rejects every `finish_reason="length"` response before
tool dispatch, even when a truncated reply contains valid-looking arguments.

## Registered rerun

The test replays the three previously truncated primary requests, one fresh
call each, using the same saved text, image bytes, schemas, seeds, model weights,
sampling parameters and thinking mode. Only `max_tokens` changes in the model
request. Context and time headroom are enlarged to make that budget possible.
There are no correction messages, task hints, training or physical robot calls.

GPU 2 was occupied, so this rerun uses GPU 0 with other work on the server.
Latency is descriptive rather than a controlled speed comparison. A fixed seed
does not guarantee identical generated text across these runs; the report
records the common prefix explicitly. These are selected development cases,
not a full agent comparison or transfer evaluation.

## Results

All three requests consumed **32,768 output tokens, all classified as reasoning**,
and ended with `finish_reason="length"`. None produced a complete tool call.
The three prompt counts remained 1,658, so each request used 34,426 total tokens
and had ample context headroom. The new allowance was actually delivered.

| Case | 4k request seconds | 32k request seconds | Complete tool call |
|---|---:|---:|---|
| vision1-r1 | 48.44 | 587.72 | No |
| vision2-r2 | 48.22 | 556.60 | No |
| vision3-r2 | 48.12 | 542.10 | No |

The 32k calls used 98,304 output tokens and 1,686.43 seconds in total.
Generation became strongly repetitive. After lowercasing and collapsing
whitespace, repeated occurrences of paragraphs at least 80 characters long
accounted for approximately **79.7%, 89.8%, and 92.5%** of the corresponding
long-paragraph text. This is a descriptive repetition measure, not a direct
measure of internal reasoning quality.

Increasing the allowance did not resolve the stalls in these selected
one-step grounding probes. This is not a full Qwen-agent learning trial or a
matched Luna comparison. No geometry fit or robot execution was attempted
because no complete request was produced. The 32k default remains in place
as requested; further diagnosis should distinguish decoding behavior, prompting,
visual grounding and access to analysis tools instead of assuming that more
generation alone is progress.

## Validation and retained evidence

Seventy-eight focused tests passed. The final exact-context helper reproduced
the prompt counts of all three actual generations. A separate tiny transport
smoke check succeeded with a 32k allowance and matched tokenizer/generation
prompt counts; it is excluded from the three-case result. The oversized-input
probe performed tokenization only and sent no generation request.

Model-study source hashes and image hashes were verified when all three requests
finished. Subsequent integration of exact context admission and truncated-action
protection changed neither the registered messages nor tool schemas; the frozen
source snapshot is retained. All three request bodies differ from their 4k
predecessors only in `max_tokens`. Their output traces are fresh samples and
are not exact extensions of the earlier reasoning text.

The result directory retains requests, complete or partial outputs, usage,
timing, repetition analysis, input-equivalence audits, source snapshots, code
changes and the Luna configuration audit. The local model server is stopped.
