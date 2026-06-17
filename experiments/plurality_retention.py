"""Empirical companion — live-alternative retention (paper v1.0 §6.3, §8.5).

Tests the paper's central claim that bounded reflective questioning (Condition E)
PRESERVES answer-space plurality that a plain personalized answer narrows.

Plurality is measured the way v1.0 §6.2-6.3 demands: NOT as embedding/semantic
diversity, but as retention of PRE-REGISTERED live alternatives
(experiments/prereg_live_alternatives.json, frozen before any generation).

Arms per frame:
  plain : a normal concise assistant answer (SYSTEM_VOICE) — the frame-mirroring,
          micro-echo-chamber-prone baseline (~Condition B/C surface).
  rqa   : RQA's reflective output — surfaced tensions / assumptions / frames +
          deepening question candidates (the Condition E intervention).

Judge: MMV-L, BLIND to arm identity, counts how many pre-registered live
alternatives each response RETAINS (surfaced intelligibly, appropriately
qualified, not straw-manned, connected to the user's thinking — §6.1/§6.3).

Usage: ../venv313/bin/python experiments/plurality_retention.py [--n 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plurality_rubric import JUDGE_RUBRIC  # noqa: E402,F401

from rqa.chat import SYSTEM_VOICE  # noqa: E402
from rqa.config import Config  # noqa: E402
from rqa.controller import Controller  # noqa: E402
from rqa.llm import AdapterError, OllamaAdapter  # noqa: E402

PREREG = PROJECT_ROOT / "experiments" / "prereg_live_alternatives.json"
OUT = PROJECT_ROOT / "experiments" / "plurality_retention_results.json"


def plain_answer(adapter: OllamaAdapter, frame: str) -> str:
    try:
        return adapter.chat(SYSTEM_VOICE, [{"role": "user", "content": frame}], json_mode=False)
    except AdapterError as exc:
        return f"(answer failed: {exc})"


def rqa_response_text(cfg: Config, frame: str) -> str:
    """Assemble what RQA surfaces to the user: tensions, assumptions, frames, questions."""
    result = Controller(cfg).run(frame)
    final = result.final_round
    if final is None:
        return "(no rqa output)"
    fm = final.analysis.feature_map
    parts = []
    for label, key in (("tensions", "tensions_input_internal"), ("assumptions", "assumptions"),
                       ("frames", "frames")):
        items = [x for x in (fm.get(key) or []) if isinstance(x, str)]
        if items:
            parts.append(label + ": " + " | ".join(items))
    qs = [c.question for c in final.kept_candidates] or ([final.chosen.question] if final.chosen else [])
    if qs:
        parts.append("questions: " + " | ".join(qs))
    return "\n".join(parts) if parts else "(no rqa output)"


def judge(binding, key, frame, alternatives, response, rng_flip) -> dict:
    listing = "\n".join(f"[{i}] {a}" for i, a in enumerate(alternatives))
    user = (f"USER FRAME:\n{frame}\n\nLIVE ALTERNATIVES:\n{listing}\n\n"
            f"AI RESPONSE:\n{response[:3500]}")
    payload = {
        "model": binding.model,
        "messages": [{"role": "system", "content": JUDGE_RUBRIC}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"}, "temperature": 0.1,
    }
    for attempt in range(3):
        try:
            r = requests.post(f"{binding.endpoint}/chat/completions",
                              headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            d = json.loads(r.json()["choices"][0]["message"]["content"])
            idx = [i for i in d.get("retained_indices", []) if isinstance(i, int) and 0 <= i < len(alternatives)]
            return {"retained": sorted(set(idx)), "reason": str(d.get("reason", ""))[:200]}
        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError):
            time.sleep(3)
    return {"retained": [], "reason": "(judge error)"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--prereg", help="path to a pre-registered file (default batch 1)")
    parser.add_argument("--out", help="output filename under experiments/")
    args = parser.parse_args()

    prereg_path = Path(args.prereg) if args.prereg else PREREG
    out_path = (PROJECT_ROOT / "experiments" / args.out) if args.out else OUT
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))["prompts"][: args.n]
    cfg = Config()
    adapter = OllamaAdapter(cfg.adapter_model, cfg.ollama_url, cfg.num_ctx, cfg.temperature)
    binding = cfg.evaluator_binding
    key = binding.api_key()
    if not key:
        print("evaluator key missing", file=sys.stderr)
        return 1

    rows = []
    for p in prereg:
        frame, alts = p["frame"], p["live_alternatives"]
        print(f"[{p['id']}] generating...", flush=True)
        plain = plain_answer(adapter, frame)
        rqa = rqa_response_text(cfg, frame)
        j_plain = judge(binding, key, frame, alts, plain, False)
        j_rqa = judge(binding, key, frame, alts, rqa, False)
        rows.append({
            "id": p["id"], "frame": frame, "n_alternatives": len(alts),
            "plain_retained": len(j_plain["retained"]), "plain_which": j_plain["retained"],
            "rqa_retained": len(j_rqa["retained"]), "rqa_which": j_rqa["retained"],
            "plain_text": plain, "rqa_text": rqa,
            "plain_reason": j_plain["reason"], "rqa_reason": j_rqa["reason"],
        })
        print(f"  plain {len(j_plain['retained'])}/{len(alts)}  vs  rqa {len(j_rqa['retained'])}/{len(alts)}", flush=True)

    tot_alt = sum(r["n_alternatives"] for r in rows)
    plain_ret = sum(r["plain_retained"] for r in rows)
    rqa_ret = sum(r["rqa_retained"] for r in rows)
    # union: does answer+reflection (Condition E) cover more than plain alone?
    union_ret = sum(len(set(r["plain_which"]) | set(r["rqa_which"])) for r in rows)
    summary = {
        "frames": len(rows), "total_live_alternatives": tot_alt,
        "plain_retention_rate": round(plain_ret / tot_alt, 3) if tot_alt else 0.0,
        "rqa_retention_rate": round(rqa_ret / tot_alt, 3) if tot_alt else 0.0,
        "conditionE_union_retention_rate": round(union_ret / tot_alt, 3) if tot_alt else 0.0,
        "plain_retained": plain_ret, "rqa_retained": rqa_ret, "union_retained": union_ret,
        "model": cfg.adapter_model, "judge": binding.model,
    }
    print("\n=== Live-alternative retention: plain answer vs RQA reflective questioning ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    out_path.write_text(json.dumps({"summary": summary, "per_frame": rows}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
