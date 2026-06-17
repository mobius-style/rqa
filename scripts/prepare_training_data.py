"""Materialize training data for QLoRA SFT (SPEC §13 Phase 2).

- substitutes the {{SYSTEM_RQA}} placeholder with the real system prompt
  (single source of truth: rqa/prompts.py)
- strips meta fields; emits plain {"messages": [...]} chat JSONL
- splits a small validation set off the training corpus (NOT the human
  holdout — that stays evaluation-only and untouched)

Usage: .venv-train/bin/python scripts/prepare_training_data.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rqa.prompts import SYSTEM_RQA  # noqa: E402

SRC = PROJECT_ROOT / "data" / "sft_phase1_train.jsonl"
OUT_TRAIN = PROJECT_ROOT / "data" / "train_materialized.jsonl"
OUT_VAL = PROJECT_ROOT / "data" / "val_materialized.jsonl"
VAL_N = 16


def materialize(example: dict) -> dict:
    messages = []
    for m in example["messages"]:
        content = m["content"]
        if m["role"] == "system" and content == "{{SYSTEM_RQA}}":
            content = SYSTEM_RQA
        messages.append({"role": m["role"], "content": content})
    return {"messages": messages}


def main() -> int:
    examples = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(20260612)
    rng.shuffle(examples)
    val, train = examples[:VAL_N], examples[VAL_N:]
    for path, subset in ((OUT_TRAIN, train), (OUT_VAL, val)):
        with path.open("w", encoding="utf-8") as fh:
            for ex in subset:
                fh.write(json.dumps(materialize(ex), ensure_ascii=False) + "\n")
    print(f"train {len(train)} -> {OUT_TRAIN}")
    print(f"val   {len(val)} -> {OUT_VAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
