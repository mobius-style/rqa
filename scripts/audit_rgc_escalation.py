"""RGC escalation audit (Gate-D evidence, 2026-06-13).

Question: did v0.4's RGC front-door strangle v0.1's deepening capability?

The adapter won the blind eval 43-2 on DEEP QUESTIONS, but those were produced
by the FULL reflection loop. v0.4's `ask` now routes via rgc_route:
  L0 canned        -> no machinery
  L1 direct        -> NO reflection loop at all (pure conversational answer)
  L2 guided        -> answer first + 6s sidecar (may time out -> no question)
  L3 instrument    -> full Controller.run

Risk = under-escalation: if real inputs that DESERVE deepening route to L1,
the 43-2 advantage is stranded behind a door that won't open. This is the
mirror image of the original "問いに問いで返す" complaint and the Condition-I
restraint-collapse shape, relocated to the RGC front-door.

This audit is pure analysis: rgc_route is deterministic (no LLM), and we
cross-reference the existing full-machinery eval (adapter_eval_results.json)
to find inputs whose deep question would now be diverted away from the loop.

Usage: ../venv313/bin/python scripts/audit_rgc_escalation.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rqa.chat import canned_light_reply, canned_missing_target_reply, rgc_route  # noqa: E402
from rqa.schema import SchemaError, parse_analysis, validate_diversity  # noqa: E402

HOLDOUT = PROJECT_ROOT / "data" / "sft_phase1_holdout.jsonl"
BATCH = PROJECT_ROOT / "data" / "phase1_batch.jsonl"
EVAL = PROJECT_ROOT / "data" / "adapter_eval_results.json"
DEPTH_THRESHOLD = 30


def route_label(text: str) -> str:
    if canned_light_reply(text) is not None:
        return "L0_canned"
    if canned_missing_target_reply(text) is not None:
        return "L0_clarify"
    return {1: "L1_direct", 2: "L2_guided", 3: "L3_instrument"}[rgc_route(text).level]


def user_input_of(ex: dict) -> str:
    """Recover the raw user text (strip the memory_context block and K line)."""
    raw = next(m["content"] for m in ex["messages"] if m["role"] == "user")
    if "[/memory_context]" in raw:
        raw = raw.split("[/memory_context]", 1)[1]
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("K =")]
    return "\n".join(lines).strip()


def main() -> int:
    holdout = [json.loads(l) for l in HOLDOUT.read_text(encoding="utf-8").splitlines()]
    seeds = {r["seed_id"]: r for r in (json.loads(l) for l in BATCH.read_text().splitlines())}
    eval_data = json.loads(EVAL.read_text(encoding="utf-8"))
    adapter_rows = {r["seed_id"]: r for r in eval_data["results"]["adapter"]}

    # 1. routing distribution over the 47 real holdout inputs
    dist = Counter()
    diverted = []  # inputs whose full-machinery run produced a threshold question
    #                but RGC would now route to L1 (no loop) or L0
    per_posture = {}
    for ex in holdout:
        seed_id = ex.get("meta", {}).get("seed_id")
        text = user_input_of(ex)
        label = route_label(text)
        dist[label] += 1
        posture = seeds.get(seed_id, {}).get("directives", {}).get("posture", "?")
        per_posture.setdefault(posture, Counter())[label] += 1

        # would the full loop have surfaced a strong question for this input?
        row = adapter_rows.get(seed_id)
        produced_strong = False
        if row and row.get("parsed"):
            try:
                a = parse_analysis(row["output"])
                rep = validate_diversity(a.candidates, max(len(a.candidates), 1))
                produced_strong = rep.ok and len(rep.kept) >= 3
            except SchemaError:
                pass
        reaches_loop = label in ("L2_guided", "L3_instrument")
        if produced_strong and not reaches_loop:
            diverted.append({"seed_id": seed_id, "route": label, "posture": posture,
                             "input": text[:60]})

    print("=== RGC routing distribution over 47 real holdout inputs ===")
    for label in ("L0_canned", "L0_clarify", "L1_direct", "L2_guided", "L3_instrument"):
        n = dist.get(label, 0)
        print(f"  {label:15s} {n:3d}  ({n / len(holdout):.0%})")

    reaches = dist.get("L2_guided", 0) + dist.get("L3_instrument", 0)
    print(f"\nreaches reflection loop (L2+L3): {reaches}/{len(holdout)} ({reaches / len(holdout):.0%})")
    print(f"L3 (full machinery)            : {dist.get('L3_instrument', 0)}/{len(holdout)}")

    print("\n=== by generation posture (was this input MEANT to be deepened?) ===")
    for posture in sorted(per_posture):
        c = per_posture[posture]
        reach = c.get("L2_guided", 0) + c.get("L3_instrument", 0)
        tot = sum(c.values())
        print(f"  {posture:16s} reaches loop {reach}/{tot}  {dict(c)}")

    print(f"\n=== STRANDED CAPABILITY: inputs that produced a strong question under full")
    print(f"    machinery but RGC now diverts away from the loop: {len(diverted)} ===")
    for d in diverted[:15]:
        print(f"  [{d['route']}] ({d['posture']}) {d['input']}")

    # quick sanity: light fast-path catches greetings
    print("\n=== light fast-path spot check ===")
    for probe in ("こんにちは", "ありがとう", "OK", "あなたは何者ですか",
                  "RAGはハルシネーションを完全に無くすのか", "この仕様の前提を見て"):
        print(f"  {route_label(probe):15s} <- {probe!r}")

    out = PROJECT_ROOT / "data" / "rgc_escalation_audit.json"
    out.write_text(json.dumps({
        "distribution": dict(dist),
        "reaches_loop_rate": reaches / len(holdout),
        "l3_rate": dist.get("L3_instrument", 0) / len(holdout),
        "stranded": diverted,
        "by_posture": {k: dict(v) for k, v in per_posture.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
