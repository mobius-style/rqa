"""Assemble a DPO preference corpus (SPEC §12.2) from existing data — no new LLM calls.

Sources (all provenance-tagged; §12.2 caps selector-log <= 50% of corpus):
  A. selector-log pairs (state/runs/ via export_sft): within the adapter's own
     K candidates, evaluator winner vs loser. The highest-value signal — it
     pushes the adapter to prefer its OWN better question.
  B. synthesized memory-fabrication negatives: take a real well-formed output
     (chosen) and corrupt it by injecting a fabricated memory_ref (rejected).
     Targets the one defect currently only code-suppressed (sanitize_memory_refs).
  C. blind adapter-vs-raw wins (chosen=adapter turn, rejected=raw turn, same
     input): reinforces structured/diverse output over the base's flat answer.

DPO format: {"prompt": <chat msgs as text>, "chosen": <text>, "rejected": <text>,
             "source": ..., "provenance": ...}

Usage: ../venv313/bin/python scripts/build_dpo_corpus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rqa.prompts import SYSTEM_RQA, build_user_prompt  # noqa: E402
from rqa.schema import SchemaError, parse_analysis  # noqa: E402

RUNS = PROJECT_ROOT / "state" / "runs"
EVAL = PROJECT_ROOT / "data" / "adapter_eval_results.json"
BLIND = PROJECT_ROOT / "data" / "raw_vs_adapter_final.json"
TRAIN = PROJECT_ROOT / "data" / "sft_phase1_train.jsonl"
OUT = PROJECT_ROOT / "data" / "dpo_corpus.jsonl"

SELECTOR_CAP = 0.50  # §12.2


def prompt_text(messages_or_user: str) -> str:
    """DPO prompt = system + user, rendered as a single chat string marker."""
    return messages_or_user


def from_selector_log() -> list[dict]:
    """A: winner vs loser within one run's shortlist (needs the full candidate
    objects; run records store kept_candidates + selection scores)."""
    pairs = []
    for p in sorted(RUNS.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        sel = rec.get("selection")
        shortlist = rec.get("shortlist") or []
        if not sel or len(shortlist) < 2:
            continue
        scores = {s["index"]: s["total"] for s in sel.get("scores", []) if "index" in s}
        best = sel.get("best_index")
        if best not in scores:
            continue
        user = build_user_prompt(rec["input_text"], rec.get("fragments") or [], rec.get("k", 6))
        for idx, total in scores.items():
            if idx == best or idx >= len(shortlist):
                continue
            if scores[best] - total >= 2:  # gap threshold
                pairs.append({
                    "prompt": user,
                    "chosen": shortlist[best],
                    "rejected": shortlist[idx],
                    "source": "selector_log",
                    "provenance": f"run {rec['session']} gap {scores[best]-total}",
                })
    return pairs


def from_memory_fabrication() -> list[dict]:
    """B: synthesize fabrication negatives from real SFT training outputs.
    chosen = the real output; rejected = same output with a fabricated
    memory-cross tension citing a node that was never injected."""
    pairs = []
    fabricated_ref = {
        "tension": "(過去の議論と整合しない可能性がある)",
        "memory_ref": "2025-11-02 09:14:00",  # a node id format that cannot exist
        "recorded_at": "2025-11-02",
        "confidence": "high",
    }
    for line in TRAIN.read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        msgs = ex["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        # only inputs WITHOUT memory_context: fabricating a ref there is the bug
        if "[memory_context]" in user:
            continue
        assistant = msgs[-1]["content"]
        try:
            obj = json.loads(assistant)
        except (json.JSONDecodeError, TypeError):
            continue
        fm = obj.get("feature_map", {})
        if fm.get("tensions_memory_cross"):
            continue  # already has one; skip
        corrupted = json.loads(assistant)
        corrupted["feature_map"]["tensions_memory_cross"] = [fabricated_ref]
        pairs.append({
            "prompt": user,
            "chosen": assistant,
            "rejected": json.dumps(corrupted, ensure_ascii=False),
            "source": "fabrication_negative",
            "provenance": "synthesized: no memory injected -> any memory_ref is fabricated",
        })
    return pairs


def from_blind_wins() -> list[dict]:
    if not BLIND.exists() or not EVAL.exists():
        return []
    blind = json.loads(BLIND.read_text())["blind"]
    ev = json.loads(EVAL.read_text())["results"]
    base = {r["seed_id"]: r for r in ev["base"]}
    adpt = {r["seed_id"]: r for r in ev["adapter"]}
    pairs = []
    for b in blind:
        if b["winner"] != "adapter":
            continue
        sid = b["seed_id"]
        if sid not in base or sid not in adpt:
            continue
        pairs.append({
            "prompt": f"[input] {sid}",
            "chosen": adpt[sid]["output"],
            "rejected": base[sid]["output"],
            "source": "blind_adapter_win",
            "provenance": f"MMV-L blind verdict, posture {b['posture']}",
        })
    return pairs


def main() -> int:
    a = from_selector_log()
    b = from_memory_fabrication()
    c = from_blind_wins()
    print(f"A selector_log:        {len(a)}")
    print(f"B fabrication_negative:{len(b)}")
    print(f"C blind_adapter_win:   {len(c)}")

    corpus = b + c + a  # selector-log last so we can cap it
    # enforce §12.2: selector-log pairs <= 50% of final corpus
    non_sel = [p for p in corpus if p["source"] != "selector_log"]
    sel = [p for p in corpus if p["source"] == "selector_log"]
    max_sel = len(non_sel)  # so sel <= 50% means sel <= non_sel
    if len(sel) > max_sel:
        sel = sel[:max_sel]
    final = non_sel + sel

    with OUT.open("w", encoding="utf-8") as fh:
        for p in final:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter

    dist = Counter(p["source"] for p in final)
    sel_ratio = dist.get("selector_log", 0) / len(final) if final else 0
    print(f"\nfinal corpus: {len(final)}  {dict(dist)}")
    print(f"selector-log ratio: {sel_ratio:.0%} (cap 50%)  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
