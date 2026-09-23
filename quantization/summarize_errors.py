#!/usr/bin/env python3
"""Summarize and validate the Dots3 uniform-K4 projection journal."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


MODULE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)$"
)
PROJECTION = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}


def summarize(path: Path, *, complete: bool) -> dict:
    seen = set()
    by_layer = defaultdict(list)
    totals = Counter()
    numerators = defaultdict(float)
    denominators = defaultdict(float)
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        match = MODULE.fullmatch(row.get("module", ""))
        if match is None:
            raise ValueError(f"journal line {line_number} has an unexpected module")
        layer, expert = map(int, match.group(1, 2))
        projection = match.group(3)
        if not 1 <= layer <= 45 or not 0 <= expert < 256:
            raise ValueError(f"journal line {line_number} is outside Dots3 routed geometry")
        key = (layer, expert, projection)
        if key in seen:
            raise ValueError(f"duplicate routed projection: {key}")
        seen.add(key)
        if (row.get("bits"), row.get("codebook")) != (4, "mcg"):
            raise ValueError(f"nonuniform projection format at line {line_number}")
        if row.get("logical_layer") != layer or row.get("expert") != expert:
            raise ValueError(f"journal identity mismatch at line {line_number}")
        if row.get("projection") != PROJECTION[projection]:
            raise ValueError(f"journal projection mismatch at line {line_number}")
        metrics = row.get("quantizer_metrics", {})
        if metrics.get("hessian_metric_status") != "ok":
            raise ValueError(f"incomplete Hessian metric at line {line_number}")
        numerator = metrics.get("hessian_weighted_error_numerator")
        denominator = metrics.get("hessian_weighted_reference_denominator")
        if not isinstance(numerator, (float, int)) or not isinstance(denominator, (float, int)) or denominator <= 0:
            raise ValueError(f"invalid Hessian metric at line {line_number}")
        owner = tuple(row.get("devices", ()))
        if owner not in (("cuda:0",), ("remote:moa/cuda:0",)):
            raise ValueError(f"unexpected quantization device at line {line_number}: {owner}")
        by_layer[layer].append(row)
        totals["rhea" if owner == ("cuda:0",) else "moa"] += 1
        numerators[projection] += numerator
        denominators[projection] += denominator

    layer_reports = {}
    for layer, rows in sorted(by_layer.items()):
        projections = Counter(row["projection"] for row in rows)
        owners = Counter("rhea" if row["devices"] == ["cuda:0"] else "moa" for row in rows)
        routes = [
            row["route_evidence"]["expert_route_count"]
            for row in rows if row["projection"] == "w1"
        ]
        layer_reports[str(layer)] = {
            "records": len(rows),
            "projections": dict(sorted(projections.items())),
            "owners": dict(sorted(owners.items())),
            "zero_route_experts": sum(count == 0 for count in routes),
            "experts_below_1024_natural_routes": sum(count < 1024 for count in routes),
            "complete": projections == {"w1": 256, "w2": 256, "w3": 256},
        }
    if complete:
        if len(seen) != 45 * 256 * 3 or set(by_layer) != set(range(1, 46)):
            raise ValueError("complete journal lacks one or more routed projections")
        if not all(item["complete"] for item in layer_reports.values()):
            raise ValueError("complete journal has an incomplete routed layer")
    return {
        "schema": "dots3-uniform-k4-journal-summary-v1",
        "complete_required": complete,
        "records": len(seen),
        "owners": dict(sorted(totals.items())),
        "hessian_weighted_relative_error_by_projection": {
            PROJECTION[name]: numerators[name] / denominators[name]
            for name in PROJECTION
        },
        "layers": layer_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    report = summarize(args.journal, complete=args.complete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "layers"}), flush=True)


if __name__ == "__main__":
    main()
