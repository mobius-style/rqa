"""QLoRA DPO — RQA Adapter v0.2 (SPEC §13 Phase 5).

Continues from the v0.1 SFT adapter. In PEFT mode the reference policy is the
same model with the adapter disabled (no extra weights), so QLoRA-DPO fits the
same 16GB budget as SFT — but it computes logprobs for chosen AND rejected, so
sequences are kept short to avoid the 256k-vocab logit OOM.

Usage:
  .venv-train/bin/python scripts/train_dpo.py --smoke   # 4 steps, VRAM check
  .venv-train/bin/python scripts/train_dpo.py           # full DPO -> v0.2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

# trl 1.6 logs a per-token ENTROPY metric by reshaping the full [seq, 256k] logit
# tensor — a vanity metric that OOMs on Gemma-4's 256k vocab in 16GB. It does not
# touch the DPO loss or gradients, so we replace it with a zero tensor of the
# right shape: the learned model is identical; only the logged entropy reads 0.
import trl.trainer.dpo_trainer as _dpot  # noqa: E402


def _cheap_entropy(logits):
    return torch.zeros(logits.shape[:-1], device=logits.device, dtype=logits.dtype)


_dpot.entropy_from_logits = _cheap_entropy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rqa.prompts import SYSTEM_RQA  # noqa: E402

MODEL_ID = "google/gemma-4-12B-it"
SFT_ADAPTER = PROJECT_ROOT / "models" / "rqa_adapter_v0_1"
OUT_DIR = PROJECT_ROOT / "models" / "rqa_adapter_v0_2"
CORPUS = PROJECT_ROOT / "data" / "dpo_corpus.jsonl"

# Full-fidelity sequence length (no global content trim — OOM is addressed by
# trimming the SPECIFIC offending objects, per owner OOM policy, not by
# shrinking max_length, which would delete completions and void the signal).
MAX_LEN = 1536


def to_chat(example: dict, tokenizer) -> dict:
    """DPO row: prompt (system+user rendered), chosen, rejected as assistant turns.
    prompt fields in the corpus are already the user content; wrap with template."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_RQA}, {"role": "user", "content": example["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"prompt": prompt, "chosen": example["chosen"], "rejected": example["rejected"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="eager",
    )
    # start the DPO policy from the v0.1 SFT adapter (trainable);
    # reference = same model with adapter disabled (PEFT implicit ref)
    model = PeftModel.from_pretrained(base, str(SFT_ADAPTER), is_trainable=True)
    model.config.use_cache = False
    # TARGETED OOM trim (T policy: trim the specific offending object, not content):
    # the offending object at backward is unfreed activations. PEFT + gradient
    # checkpointing silently no-ops unless inputs require grad and reentrant is
    # off — enabling it properly frees activation memory with ZERO effect on the
    # learned weights (correctness-preserving). max_length stays full (no content trim).
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    ds = load_dataset("json", data_files={"train": str(CORPUS)})["train"]
    ds = ds.map(lambda e: to_chat(e, tokenizer))
    if args.smoke:
        ds = ds.select(range(8))

    cfg = DPOConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=args.epochs,
        max_steps=4 if args.smoke else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="no" if args.smoke else "epoch",
        bf16=True,
        optim="paged_adamw_8bit",
        max_length=args.max_len,
        beta=args.beta,
        # precompute the frozen-reference logprobs in a no-grad pass first, so the
        # training step only materializes POLICY logits (not policy+ref together).
        # Halves concurrent 256k-vocab logit allocations -> fits the long
        # (853-token system prompt + ~500-token completion) DPO sequences in 16GB.
        precompute_ref_log_probs=True,
        report_to=[],
        seed=20260613,
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tokenizer)

    print(f"DPO: {len(ds)} pairs, beta={args.beta}, smoke={args.smoke}")
    result = trainer.train()
    print(f"train loss: {result.training_loss:.4f}")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if not args.smoke:
        trainer.save_model(str(OUT_DIR))
        tokenizer.save_pretrained(str(OUT_DIR))
        (OUT_DIR / "training_meta.json").write_text(json.dumps({
            "base_model": MODEL_ID, "from_adapter": "rqa_adapter_v0_1",
            "adapter": "rqa_adapter_v0_2", "method": "QLoRA-DPO",
            "pairs": len(ds), "beta": args.beta, "train_loss": result.training_loss,
            "corpus": "dpo_corpus.jsonl (fabrication-neg + blind-win + selector-log)",
        }, indent=1), encoding="utf-8")
        print(f"adapter saved -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
