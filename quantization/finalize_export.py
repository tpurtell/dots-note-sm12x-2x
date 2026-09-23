#!/usr/bin/env python3
"""Supply source tokenizer and multimodal assets missing from GPTQModel export."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ASSETS = (
    "LICENSE",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def finalize(source: Path, output: Path) -> dict:
    if not (output / "config.json").is_file() or not (output / "model.safetensors.index.json").is_file():
        raise ValueError("quantized export is missing its config or tensor index")
    files = {}
    for name in ASSETS:
        original = source / name
        if not original.is_file():
            raise FileNotFoundError(original)
        target = output / name
        if target.is_symlink():
            raise ValueError(f"export asset must be a regular file: {target}")
        copied = False
        if not target.exists():
            shutil.copy2(original, target)
            copied = True
        elif not target.is_file():
            raise ValueError(f"export asset is not a regular file: {target}")
        files[name] = {
            "copied": copied,
            "source_sha256": digest(original),
            "export_sha256": digest(target),
        }
    return {"schema": "dots3-export-assets-v1", "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp8-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = finalize(args.fp8_source, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
