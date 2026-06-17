"""Export training data from accumulated run records (SPEC_v0_2.md §12).

  ../venv313/bin/python scripts/export_sft.py sft   # state/runs/ -> data/sft_from_runs.jsonl
  ../venv313/bin/python scripts/export_sft.py dpo   # state/runs/ -> data/dpo_from_runs.jsonl
  ../venv313/bin/python scripts/export_sft.py lint <file.jsonl>   # lint any SFT corpus
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rqa.sft import build_sft_example, corpus_stats, extract_dpo_pairs, lint_sft_example  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "state" / "runs"
DATA_DIR = PROJECT_ROOT / "data"


def load_records() -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS_DIR.glob("*.json"))
    ]


def cmd_sft() -> int:
    records = load_records()
    examples, rejected = [], 0
    for rec in records:
        ex = build_sft_example(rec)
        if ex is None:
            rejected += 1
            continue
        problems = [p for p in lint_sft_example(ex) if not p.startswith("note:")]
        if problems:
            rejected += 1
            print(f"  reject {rec['session']}: {problems[0]}")
            continue
        examples.append(ex)
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "sft_from_runs.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"runs: {len(records)}, exported: {len(examples)}, rejected: {rejected} -> {out}")
    print("corpus:", json.dumps(corpus_stats(examples), ensure_ascii=False))
    return 0


def cmd_dpo() -> int:
    records = load_records()
    pairs = [p for rec in records for p in extract_dpo_pairs(rec)]
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "dpo_from_runs.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p.as_dict(), ensure_ascii=False) + "\n")
    print(f"runs: {len(records)}, selector-log pairs (gap>=2): {len(pairs)} -> {out}")
    print("reminder: selector-log pairs <= 50% of final DPO corpus (§12.2)")
    return 0


def cmd_lint(path_str: str) -> int:
    path = Path(path_str)
    examples = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    bad = 0
    for i, ex in enumerate(examples):
        problems = lint_sft_example(ex)
        fatal = [p for p in problems if not p.startswith("note:")]
        if fatal:
            bad += 1
            print(f"[{i}] FAIL: {fatal}")
        elif problems:
            print(f"[{i}] pass (notes: {problems})")
    print(f"{len(examples) - bad}/{len(examples)} pass")
    print("corpus:", json.dumps(corpus_stats(examples), ensure_ascii=False))
    return 1 if bad else 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "sft":
        return cmd_sft()
    if cmd == "dpo":
        return cmd_dpo()
    if cmd == "lint" and len(sys.argv) == 3:
        return cmd_lint(sys.argv[2])
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
