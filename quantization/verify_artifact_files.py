#!/usr/bin/env python3
"""Verify a copied export or Hugging Face snapshot against acceptance hashes."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(root: Path, manifest: dict) -> dict:
    if not root.is_dir():
        raise ValueError(f"artifact directory is missing: {root}")
    if manifest.get("schema") != "dots3-artifact-file-hashes-v1":
        raise ValueError("unsupported artifact hash manifest")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("artifact hash manifest is empty")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    if actual != set(expected):
        raise ValueError(
            f"artifact file set differs: missing={sorted(set(expected) - actual)[:8]}, "
            f"extra={sorted(actual - set(expected))[:8]}"
        )
    total = 0
    for name, row in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError(f"invalid relative file name: {name}")
        path = root / relative
        size = path.stat().st_size
        if size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"artifact file differs: {name}")
        total += size
    if total != manifest.get("total_bytes"):
        raise ValueError("artifact total byte count differs")
    return {"schema": "dots3-artifact-verification-v1", "root": str(root.resolve()),
            "files": len(expected), "total_bytes": total, "verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.artifact, json.loads(args.manifest.read_text()))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
