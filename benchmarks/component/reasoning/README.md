# Dots3 reasoning parser CPU qualification

The native`deepseek_r1` parser incorrectly treats plain nonthinking output as
reasoning in nonstream responses, and treats an earlier turn's`</think>` as
closing a new bare`<|assistant|>` turn. Both behaviors reproduced using the
actual pinned tokenizer; see`deepseek-r1-baseline.jsonl`.

The minimal`dots3` parser scopes prompt-state detection to the current assistant
turn, respects`chat_template_kwargs.enable_thinking=false`, and removes an
opening marker bundled with reasoning text in a speculative-style stream chunk.
It retains the native`dots` tool parser and existing xgrammar integration.

`dots3-cpu.json` records120 stream cases and30 nonstream cases covering thinking
on/off, first/history/tool-return prompts, plain/JSON/auto/named/required tools,
and four chunk sizes. Six cases use the actual CPU xgrammar compiler/matcher.
Three additional repetitions call vLLM0.30's actual speculative validation,
accepted-token update and xgrammar reset: invalid draft tokens immediately after
`</think>` are rejected, valid JSON drafts survive, validation rolls back its
matcher state, and reset clears token count and termination state.

Run`python3 serving/check_reasoning.py --model LOCAL_CACHED_SNAPSHOT` inside the
native image with CUDA hidden. The candidate source and registry port are
installed by both Dockerfiles; enable it for qualification with
`--reasoning-parser dots3`, retaining`--tool-call-parser dots`.

**Live API/engine qualification remains pending.** CPU parser and matcher tests
do not establish speculative decoding correctness on the running model. Check
thinking/nonthinking stream/nonstream requests, JSON and all tool-choice modes,
including a second assistant turn after a real tool response. Preserve reasoning
as`reasoning_content` when returning assistant history to this checkpoint's
template. The release should expose its reasoning parser while each request
chooses thinking explicitly. The live gate keeps`include_reasoning=true`: in
vLLM0.30, false also changes initial structured-output gating and is not the
same setting as the template's`enable_thinking=false`. See
[the live gate procedure](../../../serving/benchmarks/reasoning_api.md).
