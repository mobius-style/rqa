"""Empirical companion — independent second judge on the M4 plurality outputs.

Addresses the #1 persuasiveness threat (paper §14): the primary judge (MMV-L =
gpt-oss-120B) shares lineage with nothing in the generation loop, but it is a
single judge. Here a DIFFERENT model family (qwen3.6 via Ollama, unrelated to
the gemma-4 adapter and to gpt-oss) re-scores the SAME stored plain/RQA outputs
against the SAME pre-registered live alternatives — no regeneration.

Reports: independent-judge retention rates, whether the direction (RQA >= plain)
holds across judges, and inter-judge agreement (Cohen's kappa over per-alternative
retained/not-retained decisions).

Usage: ../venv313/bin/python experiments/independent_judge.py [--judge qwen3.6:27b]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plurality_rubric import JUDGE_RUBRIC  # noqa: E402

from rqa.config import Config  # noqa: E402
from rqa.llm import AdapterError, OllamaAdapter  # noqa: E402

COMBINED = PROJECT_ROOT / "experiments" / "plurality_retention_combined.json"
PREREG1 = PROJECT_ROOT / "experiments" / "prereg_live_alternatives.json"
PREREG2 = PROJECT_ROOT / "experiments" / "prereg_live_alternatives_batch2.json"
OUT = PROJECT_ROOT / "experiments" / "independent_judge_results.json"


def load_alternatives() -> dict:
    alts = {}
    for p in (PREREG1, PREREG2):
        for row in json.loads(p.read_text(encoding="utf-8"))["prompts"]:
            alts[row["id"]] = row["live_alternatives"]
    return alts


def judge(adapter: OllamaAdapter, frame: str, alternatives: list[str], response: str) -> list[int]:
    listing = "\n".join(f"[{i}] {a}" for i, a in enumerate(alternatives))
    user = (f"USER FRAME:\n{frame}\n\nLIVE ALTERNATIVES:\n{listing}\n\n"
            f"AI RESPONSE:\n{response[:3500]}\n\n"
            'Respond with ONE JSON object only: {"retained_indices": [int,...]}')
    try:
        out = adapter.chat(JUDGE_RUBRIC, [{"role": "user", "content": user}])
        data = json.loads(out)
    except (AdapterError, json.JSONDecodeError):
        # tolerate fenced / noisy JSON
        try:
            s = out[out.find("{"): out.rfind("}") + 1]
            data = json.loads(s)
        except Exception:  # noqa: BLE001
            return []
    return sorted({i for i in data.get("retained_indices", [])
                   if isinstance(i, int) and 0 <= i < len(alternatives)})


def kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa over binary labels a,b (1=retained)."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default="qwen3.6:27b")
    args = parser.parse_args()

    cfg = Config()
    adapter = OllamaAdapter(args.judge, cfg.ollama_url, cfg.num_ctx, cfg.temperature)
    if not adapter.health():
        print(f"judge model {args.judge} not reachable", file=sys.stderr)
        return 1
    alts_by_id = load_alternatives()
    rows = json.loads(COMBINED.read_text(encoding="utf-8"))["per_frame"]

    # binary vectors over every (frame, alternative) for kappa, per arm
    prim_plain, indep_plain, prim_rqa, indep_rqa = [], [], [], []
    out_rows = []
    ip_ret = ir_ret = tot = 0
    for r in rows:
        alts = alts_by_id[r["id"]]
        m = len(alts)
        ind_plain = judge(adapter, r["frame"], alts, r["plain_text"])
        ind_rqa = judge(adapter, r["frame"], alts, r["rqa_text"])
        ip_ret += len(ind_plain); ir_ret += len(ind_rqa); tot += m
        prim_p = set(r["plain_which"]); prim_r = set(r["rqa_which"])
        for i in range(m):
            prim_plain.append(1 if i in prim_p else 0); indep_plain.append(1 if i in ind_plain else 0)
            prim_rqa.append(1 if i in prim_r else 0); indep_rqa.append(1 if i in ind_rqa else 0)
        out_rows.append({"id": r["id"],
                         "primary_plain": len(prim_p), "indep_plain": len(ind_plain),
                         "primary_rqa": len(prim_r), "indep_rqa": len(ind_rqa)})
        print(f"[{r['id']}] indep plain {len(ind_plain)}/{m} rqa {len(ind_rqa)}/{m}", flush=True)

    all_prim = prim_plain + prim_rqa
    all_indep = indep_plain + indep_rqa
    summary = {
        "judge_independent": args.judge,
        "judge_primary": cfg.evaluator_binding.model,
        "frames": len(rows), "total_live_alternatives": tot,
        "independent_plain_retention_rate": round(ip_ret / tot, 3) if tot else 0,
        "independent_rqa_retention_rate": round(ir_ret / tot, 3) if tot else 0,
        "direction_rqa_ge_plain_independent": ir_ret >= ip_ret,
        "cohens_kappa_overall": round(kappa(all_prim, all_indep), 3),
        "cohens_kappa_plain": round(kappa(prim_plain, indep_plain), 3),
        "cohens_kappa_rqa": round(kappa(prim_rqa, indep_rqa), 3),
        "agreement_raw": round(sum(1 for x, y in zip(all_prim, all_indep) if x == y) / len(all_prim), 3),
    }
    print("\n=== Independent second judge (cross-family) on M4 outputs ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    OUT.write_text(json.dumps({"summary": summary, "per_frame": out_rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
