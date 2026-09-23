#!/usr/bin/env python3
"""Authenticated Spark endpoint for GPTQModel's remote EXL3 projection API."""

import argparse
import hashlib
import hmac
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from gptqmodel.utils.exl3_projection_checkpoint import EXL3ProjectionCheckpointStore
from gptqmodel.utils.exl3_remote import (
    DEFAULT_MAX_BODY_BYTES,
    REMOTE_CONTRACT,
    REMOTE_REQUEST_SCHEMA,
    REMOTE_RESULT_SCHEMA,
    decode_tensor_envelope,
    encode_tensor_envelope,
    execute_remote_projection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--name", required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    token = os.environ.get("DOTS3_EXL3_REMOTE_TOKEN", "").encode()
    if len(token) < 32:
        raise ValueError("DOTS3_EXL3_REMOTE_TOKEN must be at least 32 bytes")
    identity = {
        "contract": REMOTE_CONTRACT,
        "name": args.name,
        "preflight_sha256": args.preflight_sha256,
        "image_digest": args.image_digest,
    }
    if len(args.preflight_sha256) != 64 or not args.image_digest.startswith("sha256:"):
        raise ValueError("invalid worker identity")
    store = EXL3ProjectionCheckpointStore(args.checkpoint_dir)
    quantize_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-DS4RT-Signature", hmac.new(token, payload, hashlib.sha256).hexdigest())
            self.end_headers()
            self.wfile.write(payload)

        def _authorized(self, payload: bytes) -> bool:
            claimed = self.headers.get("X-DS4RT-Signature", "")
            expected = hmac.new(token, payload, hashlib.sha256).hexdigest()
            return hmac.compare_digest(claimed, expected)

        def _error(self, status: int, message: str) -> None:
            payload = json.dumps({"status": "error", "message": message}).encode()
            self._reply(status, payload, "application/json")

        def do_GET(self) -> None:
            if self.path != "/v1/identity":
                self._error(404, "unknown endpoint")
            elif not self._authorized(b"GET /v1/identity"):
                self._error(403, "invalid signature")
            else:
                self._reply(200, json.dumps(identity, sort_keys=True).encode(), "application/json")

        def do_POST(self) -> None:
            if self.path != "/v1/exl3/quantize":
                self._error(404, "unknown endpoint")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._error(400, "invalid content length")
                return
            if length <= 0 or length > DEFAULT_MAX_BODY_BYTES:
                self._error(413, "request size out of bounds")
                return
            payload = self.rfile.read(length)
            if len(payload) != length or not self._authorized(payload):
                self._error(403, "truncated request or invalid signature")
                return
            try:
                envelope, tensors = decode_tensor_envelope(payload)
                if envelope.get("schema") != REMOTE_REQUEST_SCHEMA or envelope.get("contract") != REMOTE_CONTRACT:
                    raise ValueError("invalid request envelope")
                request = envelope.get("request")
                queued_at = time.perf_counter()
                with quantize_lock:
                    queue_wait = time.perf_counter() - queued_at
                    packed, result, checkpoint_hit = execute_remote_projection(
                        request=request,
                        tensors=tensors,
                        device="cuda:0",
                        worker_identity=identity,
                        checkpoint_store=store,
                    )
                result = {**result, "worker_queue_wait_seconds": queue_wait}
                response = encode_tensor_envelope(
                    {
                        "schema": REMOTE_RESULT_SCHEMA,
                        "contract": REMOTE_CONTRACT,
                        "request_sha256": request["request_sha256"],
                        "checkpoint_hit": checkpoint_hit,
                        "result": result,
                    },
                    packed,
                )
                self._reply(200, response, "application/octet-stream")
            except Exception as error:
                self._error(400, f"{type(error).__name__}: {error}")

    print(json.dumps({"event": "remote-worker-ready", **identity, "port": args.port}), flush=True)
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
