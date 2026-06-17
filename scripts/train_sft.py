"""QLoRA SFT — RQA Adapter v0.1 (SPEC §13.2).

Gemma 4 12B unified (multimodal) base, text-only training:
- 4-bit NF4 quantization, gradient checkpointing
- LoRA rank 16 on the language tower's attention + MLP projections only
  (vision/audio towers untouched — update_scope discipline)
- completion-only loss (train on assistant JSON, not on the prompt)

Usage:
  .venv-train/bin/python scripts/train_sft.py [--smoke]
    --smoke: 8 examples, 4 steps — VRAM/stack verification only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "google/gemma-4-12B-it"
OUT_DIR = PROJECT_ROOT / "models" / "rqa_adapter_v0_1"

MAX_SEQ_LEN = 3072


def find_lora_targets(model) -> list[str]:
    """Target attention+MLP linear projections in the LANGUAGE tower only."""
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    names = set()
    for name, module in model.named_modules():
        if not name.endswith(suffixes):
            continue
        low = name.lower()
        if any(t in low for t in ("vision", "audio", "image", "tower")) and "language" not in low:
            continue
        names.add(name)
    if not names:
        raise RuntimeError("no LoRA target modules found")
    return sorted(names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=float, default=2.0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",  # safest across new architectures
    )
    model.config.use_cache = False

    targets = find_lora_targets(model)
    print(f"LoRA targets: {len(targets)} modules (language tower)")

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )

    data_files = {
        "train": str(PROJECT_ROOT / "data" / "train_materialized.jsonl"),
        "validation": str(PROJECT_ROOT / "data" / "val_materialized.jsonl"),
    }
    ds = load_dataset("json", data_files=data_files)
    if args.smoke:
        ds["train"] = ds["train"].select(range(8))
        ds["validation"] = ds["validation"].select(range(4))

    cfg = SFTConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=args.epochs,
        max_steps=4 if args.smoke else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="no" if args.smoke else "epoch",
        save_strategy="no" if args.smoke else "epoch",
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        max_length=MAX_SEQ_LEN,
        packing=False,
        completion_only_loss=True,
        # 256k-vocab logits at seq 3072 are the peak allocation (~1.6GB);
        # chunked NLL computes the loss without materializing them at once
        loss_type="chunked_nll",
        report_to=[],
        seed=20260612,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        peft_config=lora,
    )

    print(f"training: {len(ds['train'])} examples, smoke={args.smoke}")
    result = trainer.train()
    print(f"train loss: {result.training_loss:.4f}")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if not args.smoke:
        trainer.save_model(str(OUT_DIR))
        tokenizer.save_pretrained(str(OUT_DIR))
        (OUT_DIR / "training_meta.json").write_text(
            json.dumps(
                {
                    "base_model": MODEL_ID,
                    "adapter": "rqa_adapter_v0_1",
                    "rank": 16,
                    "epochs": args.epochs,
                    "train_examples": len(ds["train"]),
                    "train_loss": result.training_loss,
                    "data": "sft_phase1_train.jsonl (426) minus 16 val",
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"adapter saved -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
