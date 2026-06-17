"""Raw-vs-Adapter final comparison (promotion-gate measurements A/C + quality).

Inputs: data/adapter_eval_results.json (full holdout, both arms).

1. Blind quality scoring: for each case, the pinned Evaluator (MMV-L) sees the
   input and two anonymized candidate sets (X/Y, order randomized per case),
   scores each 0-10 on depth/sharpness/novelty/actionability, picks a winner.
2. Diversity Index (gate C): entropy of target_layer and stance distributions
   over VALID candidates, per arm.
3. Non-Escalation (gate B analog): on rule6/anchor_search-posture cases, rate of
   forced deepening (share of L5 candidates) and search calibration vs anchors.

Usage: .venv-train/bin/python scripts/compare_raw_vs_adapter.py
Output: data/raw_vs_adapter_final.json
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rqa.config import EvaluatorBinding  # noqa: E402
from rqa.schema import SchemaError, parse_analysis, validate_diversity  # noqa: E402

EVAL_RESULTS = PROJECT_ROOT / "data" / "adapter_eval_results.json"
BATCH = PROJECT_ROOT / "data" / "phase1_batch.jsonl"
HOLDOUT = PROJECT_ROOT / "data" / "sft_phase1_holdout.jsonl"
OUT = PROJECT_ROOT / "data" / "raw_vs_adapter_final.json"

BLIND_RUBRIC = """You are a question-quality judge. Given a USER INPUT and two anonymized \
candidate-question sets (X and Y) produced by two different systems, score each SET as a whole, \
integers 0-10 per axis:
- depth: how far the questions lift the input toward hidden assumptions and frames
- sharpness: would answering them actually change the asker's understanding or decisions?
- novelty: specific to THIS input (10) vs generic templates (0)
- restraint: for simple factual/casual inputs, practical restrained clarification is CORRECT \
and scores HIGH; forced philosophical depth scores LOW. For genuinely deep inputs, depth is correct.
Pick the overall winner ("X", "Y", or "tie").
Respond with ONE JSON object only:
{"X": {"depth": int, "sharpness": int, "novelty": int, "restraint": int},
 "Y": {"depth": int, "sharpness": int, "novelty": int, "restraint": int},
 "winner": "X"|"Y"|"tie", "reason": "<=25 words"}"""


def candidates_of(output_text: str):
    try:
        a = parse_analysis(output_text)
    except SchemaError:
        return None, None
    report = validate_diversity(a.candidates, max(len(a.candidates), 1))
    return a, report.kept


def fmt_set(cands) -> str:
    return "\n".join(f"- ({c.target_layer}/{c.stance}) {c.question}" for c in cands)


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counter.values() if n)


def main() -> int:
    data = json.loads(EVAL_RESULTS.read_text(encoding="utf-8"))
    base_rows = data["results"]["base"]
    adpt_rows = data["results"]["adapter"]

    seeds = {r["seed_id"]: r for r in (json.loads(l) for l in BATCH.read_text().splitlines())}
    holdout = [json.loads(l) for l in HOLDOUT.read_text().splitlines()]
    user_by_seed = {
        ex.get("meta", {}).get("seed_id"): next(m["content"] for m in ex["messages"] if m["role"] == "user")
        for ex in holdout
    }

    binding = EvaluatorBinding.load()
    key = binding.api_key()
    rng = random.Random(20260613)

    blind, layer_counts, stance_counts = [], {"base": Counter(), "adapter": Counter()}, {"base": Counter(), "adapter": Counter()}
    nonesc = {"base": {"rule_cases": 0, "deep_candidates": 0, "search_miscal": 0},
              "adapter": {"rule_cases": 0, "deep_candidates": 0, "search_miscal": 0}}

    for b_row, a_row in zip(base_rows, adpt_rows):
        seed_id = b_row["seed_id"]
        seed = seeds.get(seed_id, {})
        posture = seed.get("directives", {}).get("posture", "?")
        anchor_search = bool(seed.get("anchors", {}).get("search_needed", False))

        b_a, b_kept = candidates_of(b_row["output"])
        a_a, a_kept = candidates_of(a_row["output"])

        for arm, analysis, kept in (("base", b_a, b_kept), ("adapter", a_a, a_kept)):
            if not kept:
                continue
            for c in kept:
                layer_counts[arm][c.target_layer] += 1
                stance_counts[arm][c.stance] += 1
            if posture in ("rule6", "anchor_search"):
                nonesc[arm]["rule_cases"] += 1
                nonesc[arm]["deep_candidates"] += sum(1 for c in kept if c.target_layer == "L5")
                got_search = bool(analysis.search_decision.get("search_needed"))
                if got_search != anchor_search:
                    nonesc[arm]["search_miscal"] += 1

        # blind quality judgment
        if not (b_kept and a_kept and key):
            continue
        flip = rng.random() < 0.5
        x_cands, y_cands = (a_kept, b_kept) if flip else (b_kept, a_kept)
        user_input = (user_by_seed.get(seed_id) or "")[:2500]
        payload = {
            "model": binding.model,
            "messages": [
                {"role": "system", "content": BLIND_RUBRIC},
                {"role": "user", "content": f"USER INPUT:\n{user_input}\n\nSET X:\n{fmt_set(x_cands)}\n\nSET Y:\n{fmt_set(y_cands)}"},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        verdict = None
        for attempt in range(3):
            try:
                resp = requests.post(f"{binding.endpoint}/chat/completions",
                                     headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=60)
                if resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                verdict = json.loads(resp.json()["choices"][0]["message"]["content"])
                break
            except (requests.RequestException, KeyError, json.JSONDecodeError):
                time.sleep(3)
        if verdict is None:
            continue
        winner_label = verdict.get("winner", "tie")
        winner = "tie"
        if winner_label in ("X", "Y"):
            winner = ("adapter" if flip else "base") if winner_label == "X" else ("base" if flip else "adapter")
        adapter_key, base_key = ("X", "Y") if flip else ("Y", "X")
        blind.append({
            "seed_id": seed_id,
            "posture": posture,
            "winner": winner,
            "adapter_scores": verdict.get(adapter_key),
            "base_scores": verdict.get(base_key),
            "reason": str(verdict.get("reason", ""))[:200],
        })

    def mean_axis(rows, who, axis):
        vals = [r[f"{who}_scores"][axis] for r in rows if isinstance(r.get(f"{who}_scores"), dict) and axis in r[f"{who}_scores"]]
        return round(sum(vals) / len(vals), 2) if vals else None

    summary = {
        "blind_n": len(blind),
        "wins": Counter(r["winner"] for r in blind),
        "mean_scores": {
            who: {ax: mean_axis(blind, who, ax) for ax in ("depth", "sharpness", "novelty", "restraint")}
            for who in ("base", "adapter")
        },
        "diversity_index": {
            arm: {"layer_entropy_bits": round(entropy(layer_counts[arm]), 3),
                  "stance_entropy_bits": round(entropy(stance_counts[arm]), 3),
                  "layer_dist": dict(layer_counts[arm]), "stance_dist": dict(stance_counts[arm])}
            for arm in ("base", "adapter")
        },
        "non_escalation": nonesc,
    }
    summary["wins"] = dict(summary["wins"])
    OUT.write_text(json.dumps({"summary": summary, "blind": blind}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
