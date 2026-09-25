# Live reasoning and speculative grammar gate

Run after installing the candidate image and starting it with
`--reasoning-parser dots3 --tool-call-parser dots`. Keep the model's normal
xgrammar backend. This script does not start or stop a server.

```bash
python3 serving/benchmarks/reasoning_api.py \
  --base-url http://127.0.0.1:8001 \
  --repeats 2 --require-mtp --require-boundary-chunk \
  --output .cache/serving/rtx/reasoning-api.jsonl
```

Run the same command against the Spark head's URL with a distinct output path.
Use an otherwise idle candidate so aggregate speculative counters describe this
gate's traffic. The default1024-token limit allows natural short reasoning;
`length` termination fails the gate.

Each repeat sends40 short requests: first and second arithmetic turns, JSON,
auto/named/required/none tools, and actual tool-return continuations, for both
thinking settings and both response modes. Two repeats therefore send80
requests. All prompts use2+3 (second turns add1); there is no long coding task.
A failed prerequisite is recorded as a failed/skipped continuation, not replaced
with fabricated assistant history.

Every request preserves its actual payload, raw JSON or SSE events, reasoning,
final content, tool calls, usage, finish reason and elapsed time. Returned tool
arguments are validated against the one local`add(2,3)` fixture; arbitrary model
code or tool names are never executed. History returns reasoning in the
checkpoint template's`reasoning_content` field. Thinking-enabled cases require
nonempty reasoning; nonthinking cases require none. Final answers, JSON, call
names/arguments, marker leakage and natural termination have explicit checks.

## MTP and the reasoning-to-grammar boundary

The gate requests actual output token IDs and records streamed token chunks
containing`</think>` (this checkpoint's token151722) followed by content tokens.
For JSON and named/required tools, that crosses from reasoning into a constrained
response. It also records speculative counters before/after the entire gate.
`--require-mtp` requires draft-token activity;`--require-boundary-chunk` requires
at least one such constrained boundary chunk. Missing evidence fails these
explicit gates even when the semantic responses pass. Increase repetitions or
inspect an engine trace if the model's accepted bursts do not expose a crossing.

An SSE burst exposes **accepted output**, not rejected draft candidates. Its
boundary plus global MTP activity is useful live engine evidence, but does not
prove what every rejected draft contained. Retain raw events and server logs;
never infer exact rejected-token behavior from the client response.

Stable vLLM0.30 already contains the relevant mechanisms:

- `StructuredOutputManager._get_constraint_start` locates the thinking end
  inside a speculative sequence and preserves the unconstrained reasoning
  prefix.
- `validate_tokens` grammar-validates only the suffix after that boundary;
  `accept_tokens` latches reasoning completion before advancing the grammar.
- `XgrammarGrammar.validate_tokens` rolls back trial acceptance, and`reset`
  clears processed-token count and termination state as well as the matcher.

`serving/check_reasoning.py` calls those actual methods on CPU with an invalid
draft immediately after`</think>` and a valid JSON draft, three times including
reset. The component receipt records those results. The live gate complements
that evidence with real model decoding; neither alone replaces final-image
qualification on both architectures.

This gate intentionally keeps`include_reasoning=true`, including when
`enable_thinking=false`. In vLLM0.30, setting`include_reasoning=false` also tells
the engine that reasoning has already ended when choosing the structured-output
gate. It is not the same setting as the template's`enable_thinking=false`.
That combination needs its own API contract before documenting it as a supported
way to hide reasoning on a thinking-plus-schema request. No global grammar
settings or speculative-token acceptance code are changed by this recipe parser.
