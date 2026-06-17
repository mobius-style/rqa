"""Verify RQA Adapter v0.1 against the frozen base on holdout inputs (SPEC §13 Phase 8 precursor).

For N holdout examples, generate with (a) base model and (b) base+adapter,
using the SAME system prompt and inputs. Reports schema-compliance and
diversity-validity rates for both arms — the mechanical-reliability metric
that Phase −1 identified as the SFT target — and saves all outputs for
qualitative comparison.

Usage: .venv-train/bin/python scripts/eval_adapter.py [--n 12]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rqa.schema import SchemaError, parse_analysis, validate_diversity  # noqa: E402

MODEL_ID = "google/gemma-4-12B-it"
ADAPTER_DIR = PROJECT_ROOT / "models" / "rqa_adapter_v0_1"
HOLDOUT = PROJECT_ROOT / "data" / "sft_phase1_holdout.jsonl"
OUT = PROJECT_ROOT / "data" / "adapter_eval_results.json"


def generate(model, tokenizer, messages: list[dict]) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def score(text: str, k: int) -> dict:
    try:
        analysis = parse_analysis(text)
    except SchemaError:
        return {"parsed": False, "valid_candidates": 0, "diversity_ok": False}
    report = validate_diversity(analysis.candidates, k)
    return {
        "parsed": True,
        "n_candidates": len(analysis.candidates),
        "valid_candidates": len(report.kept),
        "diversity_ok": report.ok,
        "has_memory_cross": bool((analysis.feature_map.get("tensions_memory_cross") or [])),
        "search_needed": bool(analysis.search_decision.get("search_needed")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    args = parser.parse_args()

    holdout = [json.loads(l) for l in HOLDOUT.read_text(encoding="utf-8").splitlines()][: args.n]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager",
    )
    model.eval()

    # materialize prompts: same system as training
    from rqa.prompts import SYSTEM_RQA

    cases = []
    for ex in holdout:
        msgs = [dict(m) for m in ex["messages"][:-1]]
        for m in msgs:
            if m["role"] == "system":
                m["content"] = SYSTEM_RQA
        user = next(m["content"] for m in msgs if m["role"] == "user")
        k = 6
        for line in user.splitlines():
            if line.strip().startswith("K ="):
                k = int(line.split("=")[1].strip())
        cases.append({"messages": msgs, "k": k, "seed_id": ex.get("meta", {}).get("seed_id")})

    results = {"base": [], "adapter": []}

    print(f"=== arm 1: frozen base, {len(cases)} holdout cases")
    for i, case in enumerate(cases):
        text = generate(model, tokenizer, case["messages"])
        results["base"].append({"seed_id": case["seed_id"], "output": text, **score(text, case["k"])})
        print(f"  [{i}] parsed={results['base'][-1]['parsed']} valid={results['base'][-1].get('valid_candidates', 0)}")

    print("=== arm 2: base + RQA adapter v0.1")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()
    for i, case in enumerate(cases):
        text = generate(model, tokenizer, case["messages"])
        results["adapter"].append({"seed_id": case["seed_id"], "output": text, **score(text, case["k"])})
        print(f"  [{i}] parsed={results['adapter'][-1]['parsed']} valid={results['adapter'][-1].get('valid_candidates', 0)}")

    def agg(rows):
        n = len(rows)
        return {
            "n": n,
            "parse_rate": sum(r["parsed"] for r in rows) / n,
            "diversity_ok_rate": sum(bool(r.get("diversity_ok")) for r in rows) / n,
            "mean_valid_candidates": sum(r.get("valid_candidates", 0) for r in rows) / n,
        }

    summary = {"base": agg(results["base"]), "adapter": agg(results["adapter"])}
    OUT.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
