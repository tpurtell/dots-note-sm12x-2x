#!/usr/bin/env python3
"""Hash every published checkpoint file for transfer and Hub receipts."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.artifact.is_symlink() or not args.artifact.is_dir():
        raise ValueError("artifact root must be a real directory")
    if not (args.artifact / "model.safetensors.index.json").is_file():
        raise ValueError("artifact has no safetensors index")
    files = {}
    for path in sorted(args.artifact.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"published artifact contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"published artifact contains an unsupported entry: {path}")
        relative = path.relative_to(args.artifact).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        files[relative] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    report = {
        "schema": "dots3-artifact-file-hashes-v1",
        "artifact": str(args.artifact.resolve()),
        "total_bytes": sum(row["bytes"] for row in files.values()),
        "files": files,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "artifact-files-hashed", "files": len(files), "total_bytes": report["total_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
