"""Patent-reinforcing test A — model-independence of the plurality keystone.

The keystone (ungoverned memory C collapses live-alternative retention; governed
memory D recovers it) was measured on the project's gemma-4-based adapter. To
support the patent's enablement/generality (the claims are method-general, not
model-specific) and to rebut a "single-model artifact" attack, this re-runs the
keystone conditions A / C / D with a DIFFERENT-FAMILY base model as the assistant
(default llama3.1:8b — unrelated to the gemma-4 adapter and to both judges).

If C collapses and D recovers on a second, unrelated model too, the effect is a
property of the GOVERNANCE METHOD, not of the original model.

Reuses the validated condition construction and judges from condition_ladder.py.

Usage: ../venv313/bin/python experiments/model_independence.py [--assistant llama3.1:8b]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from condition_ladder import (  # noqa: E402
    PREREG, answer, governed_memory, groq_judge, kappa, parse_idx, raw_memory,
)
from plurality_rubric import RUBRIC_STRICT  # noqa: E402

from rqa.config import Config  # noqa: E402
from rqa.llm import AdapterError, OllamaAdapter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assistant", default="llama3.1:8b")
    ap.add_argument("--no-indep", action="store_true",
                    help="skip the local qwen judge (use when the assistant IS qwen, "
                         "to avoid circularity and VRAM swap; gpt-oss remains the judge)")
    args = ap.parse_args()

    cfg = Config()
    binding = cfg.evaluator_binding
    key = binding.api_key()
    assistant = OllamaAdapter(args.assistant, cfg.ollama_url, cfg.num_ctx, cfg.temperature)
    qwen = None if args.no_indep else OllamaAdapter("qwen3.6:27b", cfg.ollama_url, cfg.num_ctx, cfg.temperature)
    if not key or not assistant.health():
        print(f"need Groq key and assistant model {args.assistant}", file=sys.stderr)
        return 1

    frames = []
    for p in PREREG:
        frames += json.loads(p.read_text(encoding="utf-8"))["prompts"]

    conds = ["A", "C", "D"]  # baseline / raw memory / governed memory (the keystone)
    prim = {c: [] for c in conds}
    indep = {c: [] for c in conds}
    for fr in frames:
        frame, alts, m = fr["frame"], fr["live_alternatives"], len(fr["live_alternatives"])
        texts = {
            "A": answer(assistant, "", frame),
            "C": answer(assistant, raw_memory(frame), frame),
            "D": answer(assistant, governed_memory(frame), frame),
        }
        for c in conds:
            p_idx = set(groq_judge(binding, key, frame, alts, texts[c]))
            i_idx = set()
            if qwen is not None:
                try:
                    i_idx = set(parse_idx(qwen.chat(RUBRIC_STRICT, [{"role": "user", "content":
                        f"USER FRAME:\n{frame}\n\nLIVE ALTERNATIVES:\n" +
                        "\n".join(f"[{i}] {a}" for i, a in enumerate(alts)) +
                        f"\n\nAI RESPONSE:\n{texts[c][:3500]}"}]), m))
                except AdapterError:
                    i_idx = set()
            prim[c] += [1 if i in p_idx else 0 for i in range(m)]
            indep[c] += [1 if i in i_idx else 0 for i in range(m)]
        print(f"[{fr['id']}] judged A/C/D", flush=True)

    tot = sum(len(fr["live_alternatives"]) for fr in frames)
    summary = {"assistant_model": args.assistant, "frames": len(frames),
               "total_live_alternatives": tot, "criterion": "STRICT",
               "judge_primary": binding.model, "judge_independent": "qwen3.6:27b",
               "conditions": {}}
    for c in conds:
        summary["conditions"][c] = {
            "primary_retention": round(sum(prim[c]) / tot, 3),
            "independent_retention": round(sum(indep[c]) / tot, 3),
            "kappa": kappa(prim[c], indep[c]),
        }
    a_p, c_p, d_p = (summary["conditions"][k]["primary_retention"] for k in conds)
    summary["keystone_reproduced_primary"] = c_p < a_p and d_p >= a_p * 0.8
    print("\n=== Model-independence of the keystone (assistant = %s) ===" % args.assistant)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    out = PROJECT_ROOT / "experiments" / f"model_independence_{args.assistant.replace(':', '_').replace('.', '')}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
