"""Harvest MMV assets into seed inputs for SFT annotation (SPEC_v0_2.md §12.1).

Reads (read-only) from the sibling MOBIUS_MMV repo:
  - operate-fr-bench/data/core500.jsonl + labels/core500_route_labels.jsonl
    -> queries with ANCHORED search/route expectations (the verified half)
  - config/pattern_library/*.jsonl -> intent-labelled example queries

Writes data/seed_inputs.jsonl: one record per input query with anchor fields.
The anchors are ground truth for search_decision / non-escalation; the L3-L5
annotation layer is generated on top of these later (by Claude + Evaluator audit).

Usage: ../venv313/bin/python scripts/make_seed_inputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MMV = PROJECT_ROOT.parent / "MOBIUS_MMV"
OUT = PROJECT_ROOT / "data" / "seed_inputs.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def harvest_core500() -> list[dict]:
    data = load_jsonl(MMV / "operate-fr-bench" / "data" / "core500.jsonl")
    labels = {
        r["task_id"]: r
        for r in load_jsonl(MMV / "operate-fr-bench" / "data" / "labels" / "core500_route_labels.jsonl")
    }
    seeds = []
    for row in data:
        label = labels.get(row["id"], {})
        # anchored ground truth: a verify/freshness route implies search_needed=true
        preferred = label.get("preferred_route", "")
        seeds.append(
            {
                "seed_id": f"core500:{row['id']}",
                "input_text": row["user_prompt"],
                "language": row.get("language", "en"),
                "anchors": {
                    "search_needed": preferred in ("verify", "date_bound_answer")
                    or row.get("requires_current_verification", False),
                    "preferred_route": preferred,
                    "disallowed_routes": label.get("disallowed_routes", []),
                    "temporal_volatility": row.get("temporal_volatility"),
                    "failure_modes": label.get("failure_modes_to_check", []),
                    "route_notes": label.get("route_notes", ""),
                },
                "category": row.get("family", "unknown"),
                "deepening_expected": False,  # core500 items are routing probes:
                # most are §11.1 rule-6 cases (answer/verify, do not force depth)
            }
        )
    return seeds


def harvest_pattern_library() -> list[dict]:
    seeds = []
    lib_dir = MMV / "config" / "pattern_library"
    for path in sorted(lib_dir.glob("*.jsonl")):
        if path.stat().st_size == 0:
            continue
        for row in load_jsonl(path):
            intent = row.get("intent", "")
            for example in row.get("examples", []):
                seeds.append(
                    {
                        "seed_id": f"pattern:{row.get('id', path.stem)}:{len(seeds)}",
                        "input_text": example,
                        "language": row.get("lang", "en"),
                        "anchors": {
                            "search_needed": False,
                            "intent": intent,
                            "negative_examples": row.get("negative_examples", [])[:3],
                        },
                        "category": row.get("topic", path.stem),
                        "deepening_expected": intent
                        in ("ask_definition", "concept_explanation", "ask_reasoning"),
                    }
                )
    return seeds


def main() -> int:
    if not MMV.is_dir():
        print(f"MMV repo not found at {MMV}", file=sys.stderr)
        return 1
    seeds = harvest_core500() + harvest_pattern_library()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for s in seeds:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    by_cat: dict[str, int] = {}
    for s in seeds:
        by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
    print(f"wrote {len(seeds)} seed inputs -> {OUT}")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:25s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
