"""Empirical companion — retention criterion two-variant operationalization (paper §6.3).

Measurement 5 found the per-alternative metric had low inter-judge reliability
(kappa ~= 0), with disagreement concentrated on RQA/question outputs: judges
disagreed on whether a frame-reopening QUESTION "retains" an alternative.

This experiment makes that axis EXPLICIT with two retention criteria:
  STRICT  ("stated"):             the alternative's CONTENT must be surfaced.
  LENIENT ("reopened-or-stated"): a question that specifically reopens the frame
                                  toward the alternative also counts.

Both judges (gpt-oss-120B primary, qwen3.6 independent) re-score the SAME stored
M4 outputs under BOTH criteria (no regeneration). We report retention rate per
(variant x judge x arm), the direction, and Cohen's kappa per variant — testing
whether making the criterion explicit raises inter-judge agreement to an
acceptable level, and whether the RQA>plain direction holds under both.

Usage: ../venv313/bin/python experiments/retention_variants.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plurality_rubric import RUBRIC_LENIENT, RUBRIC_STRICT  # noqa: E402

from rqa.config import Config  # noqa: E402
from rqa.llm import AdapterError, OllamaAdapter  # noqa: E402

COMBINED = PROJECT_ROOT / "experiments" / "plurality_retention_combined.json"
PREREG = [PROJECT_ROOT / "experiments" / "prereg_live_alternatives.json",
          PROJECT_ROOT / "experiments" / "prereg_live_alternatives_batch2.json"]
OUT = PROJECT_ROOT / "experiments" / "retention_variants_results.json"

VARIANTS = {"strict": RUBRIC_STRICT, "lenient": RUBRIC_LENIENT}


def build_user(frame, alts, response):
    listing = "\n".join(f"[{i}] {a}" for i, a in enumerate(alts))
    return f"USER FRAME:\n{frame}\n\nLIVE ALTERNATIVES:\n{listing}\n\nAI RESPONSE:\n{response[:3500]}"


def parse_idx(text, m):
    try:
        s = text[text.find("{"): text.rfind("}") + 1]
        d = json.loads(s)
    except Exception:  # noqa: BLE001
        return []
    return sorted({i for i in d.get("retained_indices", []) if isinstance(i, int) and 0 <= i < m})


def groq_judge(binding, key, rubric, frame, alts, response):
    payload = {"model": binding.model,
               "messages": [{"role": "system", "content": rubric},
                            {"role": "user", "content": build_user(frame, alts, response)}],
               "response_format": {"type": "json_object"}, "temperature": 0.1}
    for attempt in range(3):
        try:
            r = requests.post(f"{binding.endpoint}/chat/completions",
                              headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            return parse_idx(r.json()["choices"][0]["message"]["content"], len(alts))
        except (requests.RequestException, KeyError, ValueError):
            time.sleep(3)
    return []


def ollama_judge(adapter, rubric, frame, alts, response):
    try:
        out = adapter.chat(rubric, [{"role": "user", "content": build_user(frame, alts, response)}])
    except AdapterError:
        return []
    return parse_idx(out, len(alts))


def kappa(a, b):
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    p1, q1 = sum(a) / n, sum(b) / n
    pe = p1 * q1 + (1 - p1) * (1 - q1)
    return round((po - pe) / (1 - pe), 3) if pe != 1 else 1.0


def main() -> int:
    cfg = Config()
    binding = cfg.evaluator_binding
    key = binding.api_key()
    qwen = OllamaAdapter("qwen3.6:27b", cfg.ollama_url, cfg.num_ctx, cfg.temperature)
    if not key or not qwen.health():
        print("need both Groq key and qwen3.6:27b", file=sys.stderr)
        return 1

    alts_by_id = {}
    for p in PREREG:
        for row in json.loads(p.read_text(encoding="utf-8"))["prompts"]:
            alts_by_id[row["id"]] = row["live_alternatives"]
    rows = json.loads(COMBINED.read_text(encoding="utf-8"))["per_frame"]

    # accumulate binary vectors: [variant][judge][arm] -> list of 0/1 per alternative
    vec = {v: {"primary": {"plain": [], "rqa": []}, "indep": {"plain": [], "rqa": []}} for v in VARIANTS}
    for r in rows:
        alts = alts_by_id[r["id"]]
        m = len(alts)
        for vname, rubric in VARIANTS.items():
            for arm, text in (("plain", r["plain_text"]), ("rqa", r["rqa_text"])):
                p_idx = set(groq_judge(binding, key, rubric, r["frame"], alts, text))
                i_idx = set(ollama_judge(qwen, rubric, r["frame"], alts, text))
                vec[vname]["primary"][arm] += [1 if i in p_idx else 0 for i in range(m)]
                vec[vname]["indep"][arm] += [1 if i in i_idx else 0 for i in range(m)]
        print(f"[{r['id']}] judged (strict+lenient, both judges)", flush=True)

    tot = sum(len(alts_by_id[r["id"]]) for r in rows)
    summary = {"frames": len(rows), "total_live_alternatives": tot,
               "judge_primary": binding.model, "judge_independent": "qwen3.6:27b", "variants": {}}
    for v in VARIANTS:
        pp, pr = vec[v]["primary"]["plain"], vec[v]["primary"]["rqa"]
        ip, ir = vec[v]["indep"]["plain"], vec[v]["indep"]["rqa"]
        summary["variants"][v] = {
            "primary_plain_rate": round(sum(pp) / tot, 3), "primary_rqa_rate": round(sum(pr) / tot, 3),
            "indep_plain_rate": round(sum(ip) / tot, 3), "indep_rqa_rate": round(sum(ir) / tot, 3),
            "direction_rqa_gt_plain_primary": sum(pr) > sum(pp),
            "direction_rqa_gt_plain_indep": sum(ir) > sum(ip),
            "kappa_overall": kappa(pp + pr, ip + ir),
            "kappa_plain": kappa(pp, ip), "kappa_rqa": kappa(pr, ir),
        }
    print("\n=== Retention criterion: STRICT vs LENIENT, two judges ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
