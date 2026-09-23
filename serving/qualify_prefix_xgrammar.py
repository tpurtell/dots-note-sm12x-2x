#!/usr/bin/env python3
"""Record repeated-prefix cache hits and a constrained JSON API response."""

import argparse
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen


METRIC = re.compile(r"^(vllm:prefix_cache_(?:queries|hits)_total)(?:\{[^}]*\})?\s+(\S+)$")


def metrics(base: str) -> dict[str, float]:
    with urlopen(base + "/metrics", timeout=30) as response:
        lines = response.read().decode().splitlines()
    result = {}
    for line in lines:
        match = METRIC.match(line)
        if match:
            result[match[1]] = result.get(match[1], 0) + float(match[2])
    if len(result) != 2:
        raise RuntimeError(f"missing prefix-cache counters: {result}")
    return result


def chat(base: str, model: str, prompt: str, *, schema=None) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 48,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "dots3_probe",
                "strict": True,
                "schema": schema,
            },
        }
    started = time.perf_counter()
    request = Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    first_token = None
    text = ""
    finish = None
    usage = None
    with urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                fragment = delta.get("content") or ""
                if fragment and first_token is None:
                    first_token = time.perf_counter() - started
                text += fragment
                finish = choice.get("finish_reason") or finish
    if first_token is None or usage is None or finish is None:
        raise RuntimeError("incomplete streamed chat response")
    return {
        "ttft_seconds": first_token,
        "elapsed_seconds": time.perf_counter() - started,
        "content": text,
        "finish_reason": finish,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="dots3-note-exl3-k4")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base = args.base_url.rstrip("/").removesuffix("/v1")
    if args.output.exists():
        raise FileExistsError(args.output)
    prompt = "Use the context, then answer in one short sentence.\n" + (
        "This paragraph is repeated to test prefix caching in Dots3 Note. " * 256
    )
    before = metrics(base)
    cold = chat(base, args.model, prompt)
    after_cold = metrics(base)
    warm = chat(base, args.model, prompt)
    after_warm = metrics(base)
    hits_key = "vllm:prefix_cache_hits_total"
    queries_key = "vllm:prefix_cache_queries_total"
    warm_hits = after_warm[hits_key] - after_cold[hits_key]
    warm_queries = after_warm[queries_key] - after_cold[queries_key]
    if warm_hits <= 0 or warm_queries <= 0:
        raise AssertionError("repeated request produced no prefix-cache hits")

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    constrained = chat(
        base, args.model,
        "Return a JSON object whose answer is the integer 42.", schema=schema,
    )
    value = json.loads(constrained["content"])
    if set(value) != {"answer"} or type(value["answer"]) is not int:
        raise AssertionError(f"xgrammar response broke JSON schema: {value}")
    result = {
        "schema": "dots3-prefix-xgrammar-v1",
        "model": args.model,
        "base_url": base,
        "cold": cold,
        "warm": warm,
        "metrics_before": before,
        "metrics_after_cold": after_cold,
        "metrics_after_warm": after_warm,
        "warm_prefix_hits": warm_hits,
        "warm_prefix_queries": warm_queries,
        "xgrammar": constrained,
        "xgrammar_json": value,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "event": "prefix-xgrammar-qualified",
        "warm_prefix_hits": warm_hits,
        "warm_prefix_queries": warm_queries,
        "cold_ttft_seconds": cold["ttft_seconds"],
        "warm_ttft_seconds": warm["ttft_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
