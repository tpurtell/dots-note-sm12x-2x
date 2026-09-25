#!/usr/bin/env python3
"""Probe the source model's public image and audio examples through vLLM."""

import argparse
import json
import re
from pathlib import Path
import time
from urllib.request import Request, urlopen


CASES = {
    "image": {
        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
        "question": "How many cats are in this image? Answer in one short sentence.",
    },
    "audio": {
        "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mary_had_lamb.mp3",
        "question": "Transcribe this nursery rhyme.",
    },
}


def request(base: str, model: str, kind: str, url: str, question: str) -> dict:
    content = [
        {"type": f"{kind}_url", f"{kind}_url": {"url": url}},
        {"type": "text", "text": question},
    ]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    req = Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=900) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - started
    message = body["choices"][0]["message"]
    answer = message.get("content") or ""
    if not answer.strip() or not body.get("usage"):
        raise AssertionError(f"empty {kind} multimodal result")
    normalized = re.sub(r"[^a-z0-9 ]", "", answer.lower())
    contract = (
        bool(re.search(r"\b(?:2|two) cats\b", normalized))
        if kind == "image" else
        all(fragment in normalized for fragment in (
            "mary had a little lamb", "white as snow", "lamb was sure to go",
        ))
    )
    return {
        "kind": kind,
        "asset_url": url,
        "question": question,
        "elapsed_seconds": elapsed,
        "answer": answer,
        "usage": body["usage"],
        "finish_reason": body["choices"][0].get("finish_reason"),
        "contract_passed": contract and body["choices"][0].get("finish_reason") == "stop",
        "raw_response": body,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="dots3-note-exl3-k4")
    parser.add_argument("--image-url", default=CASES["image"]["url"])
    parser.add_argument("--audio-url", default=CASES["audio"]["url"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = [
        request(args.base_url, args.model, kind, getattr(args, f"{kind}_url"), case["question"])
        for kind, case in CASES.items()
    ]
    report = {
        "schema": "dots3-multimodal-api-v1",
        "model": args.model,
        "base_url": args.base_url,
        "source_examples": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if not all(result["contract_passed"] for result in results):
        raise AssertionError("multimodal content or completion contract failed; see report")
    print(json.dumps({"event": "multimodal-api-complete", "cases": [x["kind"] for x in results]}), flush=True)


if __name__ == "__main__":
    main()
