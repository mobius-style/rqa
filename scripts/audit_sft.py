"""External Evaluator audit of the SFT corpus (SPEC §12.1 審査 — Evaluator role B).

Sends each example to the pinned MMV-L Evaluator for rubric scoring.
The rubric explicitly rewards NOT forcing depth on simple inputs (anti-over-ask).

  ../venv313/bin/python scripts/audit_sft.py data/sft_phase1_raw.jsonl

Outputs:
  data/audit_results.jsonl      per-example scores + verdict
  data/sft_phase1_v1.jsonl      accepted examples only
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rqa.config import EvaluatorBinding  # noqa: E402

# Frozen audit rubric (evaluator_criteria — changes require human approval).
AUDIT_RUBRIC = """You are the External Evaluator (role B: training-data audit) for MOBIUS-RQA, \
a reflective question-generation system. You are given a USER INPUT (possibly with a \
[memory_context] data block) and a candidate TRAINING OUTPUT (feature_map, search_decision, \
question_candidates, self_update_proposal).

Score each axis 0-10:
- fidelity: feature_map faithfully reflects the input — claims actually present, tensions \
grounded in the text/memory, nothing hallucinated. Memory-cross tensions must reference the \
provided memory records. NOTE: the line "K = n" in the user input is a control parameter \
(the number of candidates to generate), NOT content — it must not appear in feature_map and \
its absence there must never be penalized.
- depth: question candidates are specific to THIS input (not generic templates) and the \
deepening level fits the input. IMPORTANT: for simple factual / casual / greeting inputs, \
restrained practical clarifications are CORRECT and score HIGH; forced philosophical depth \
on such inputs scores LOW.
- calibration: search_decision is sensible (volatile/current facts -> search; stable or \
conceptual -> no search); false embedded premises are challenged, not accepted.
- format_language: output language matches the input language; structure is coherent.

verdict: "accept" if this example should enter the training corpus, else "reject".
Respond with ONE JSON object only:
{"fidelity": int, "depth": int, "calibration": int, "format_language": int,
 "verdict": "accept"|"reject", "reason": "<=30 words"}"""

MIN_AXIS = 5
WORKERS = 4


def audit_one(binding: EvaluatorBinding, key: str, idx: int, example: dict) -> dict:
    user_content = next(m["content"] for m in example["messages"] if m["role"] == "user")
    assistant = example["messages"][-1]["content"]
    payload = {
        "model": binding.model,
        "messages": [
            {"role": "system", "content": AUDIT_RUBRIC},
            {
                "role": "user",
                "content": f"USER INPUT:\n{user_content[:3500]}\n\nTRAINING OUTPUT:\n{assistant[:5000]}",
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }
    for attempt in range(4):
        try:
            resp = requests.post(
                f"{binding.endpoint}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = json.loads(resp.json()["choices"][0]["message"]["content"])
            axes = {a: int(data.get(a, 0)) for a in ("fidelity", "depth", "calibration", "format_language")}
            accepted = data.get("verdict") == "accept" and all(v >= MIN_AXIS for v in axes.values())
            return {
                "index": idx,
                "seed_id": example.get("meta", {}).get("seed_id"),
                **axes,
                "verdict": data.get("verdict"),
                "accepted": accepted,
                "reason": str(data.get("reason", ""))[:200],
            }
        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as exc:
            if attempt == 3:
                return {"index": idx, "seed_id": example.get("meta", {}).get("seed_id"),
                        "accepted": False, "verdict": "error", "reason": str(exc)[:200]}
            time.sleep(3 * (attempt + 1))
    return {"index": idx, "accepted": False, "verdict": "error", "reason": "exhausted retries"}


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "sft_phase1_raw.jsonl"
    examples = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    binding = EvaluatorBinding.load()
    key = binding.api_key()
    if not key:
        print("evaluator key not found", file=sys.stderr)
        return 1

    # --reaudit <pass1 results>: re-audit only pass-1 rejects (after a documented
    # rubric clarification); pass-1 accepts are kept as-is.
    prior_accepted: set[int] = set()
    target_indices = list(range(len(examples)))
    suffix = ""
    if len(sys.argv) > 3 and sys.argv[2] == "--reaudit":
        prior = [json.loads(l) for l in Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()]
        prior_accepted = {r["index"] for r in prior if r["accepted"]}
        target_indices = [r["index"] for r in prior if not r["accepted"]]
        suffix = "_pass2"
        print(f"re-auditing {len(target_indices)} pass-1 rejects (keeping {len(prior_accepted)} accepts)")

    print(f"auditing {len(target_indices)} examples with {binding.release} ({binding.model}) ...")
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(audit_one, binding, key, i, examples[i]) for i in target_indices]
        for n, fut in enumerate(futures):
            results.append(fut.result())
            if (n + 1) % 50 == 0:
                acc = sum(1 for r in results if r["accepted"])
                print(f"  {n + 1}/{len(target_indices)} audited, {acc} accepted so far")

    results.sort(key=lambda r: r["index"])
    out_audit = PROJECT_ROOT / "data" / f"audit_results{suffix}.jsonl"
    with out_audit.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    accepted_idx = prior_accepted | {r["index"] for r in results if r["accepted"]}
    out_corpus = PROJECT_ROOT / "data" / "sft_phase1_v1.jsonl"
    with out_corpus.open("w", encoding="utf-8") as fh:
        for i, ex in enumerate(examples):
            if i in accepted_idx:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    n_acc = len(accepted_idx)
    errors = sum(1 for r in results if r["verdict"] == "error")
    print(f"accepted total {n_acc}/{len(examples)} ({n_acc / len(examples):.1%}), errors {errors}")
    print(f"-> {out_corpus}\n-> {out_audit}")
    rejected = [r for r in results if not r["accepted"] and r["verdict"] != "error"]
    if rejected:
        print("\nsample rejections:")
        for r in rejected[:8]:
            print(f"  [{r['index']}] {r.get('seed_id')}: {r['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
