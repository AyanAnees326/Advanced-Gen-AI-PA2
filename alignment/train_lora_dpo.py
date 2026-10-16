"""Task 5: TRL DPOTrainer + PEFT LoRA persona tuning of SmolLM2-360M-Instruct.

    python -m alignment.train_lora_dpo --out outputs/lora_dpo

The 220 persona pairs are split 200 train / 20 eval (seed 42) and converted to
TRL's conversational format (role-message lists). With a PEFT model TRL uses the
base weights with the adapter disabled as the frozen reference.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from alignment.data import ROOT
from alignment.train import keep_awake

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def conversational(rows):
    return Dataset.from_list([{"prompt": [{"role": "user", "content": r["prompt"]}],
                               "chosen": [{"role": "assistant", "content": r["chosen"]}],
                               "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in rows])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "lora_dpo"))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=-1, help="debug")
    args = ap.parse_args(argv)
    keep_awake()

    rows = json.loads((ROOT / "data" / "persona_lora" / "ramsay_style_preferences.json").read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(rows)
    train_rows, eval_rows = rows[args.n_eval :], rows[: args.n_eval]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    peft_config = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
                             task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    config = DPOConfig(
        output_dir=args.out, beta=args.beta, learning_rate=args.lr, num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine", warmup_ratio=0.1, weight_decay=0.0, max_grad_norm=1.0, max_length=512,
        fp16=torch.cuda.is_available(), bf16=False, logging_steps=1, eval_strategy="steps", eval_steps=5,
        save_strategy="no", report_to="none", seed=args.seed, dataset_num_proc=None,
    )
    trainer = DPOTrainer(model=model, args=config, train_dataset=conversational(train_rows),
                         eval_dataset=conversational(eval_rows), processing_class=tokenizer, peft_config=peft_config)
    trainer.model.print_trainable_parameters()
    initial = {**trainer.evaluate(), "step": 0}  # train() resets log_history, so keep the step-0 eval
    trainer.train()

    out = Path(args.out)
    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    with open(out / "metrics.jsonl", "w", encoding="utf-8") as f:
        for record in [initial, *trainer.state.log_history]:
            f.write(json.dumps(record) + "\n")
    trainable, total = trainer.model.get_nb_trainable_parameters()
    (out / "config.json").write_text(json.dumps({**vars(args), "model": MODEL, "train_pairs": len(train_rows),
                                                 "eval_pairs": len(eval_rows), "trainable_params": trainable,
                                                 "total_params": total, "target_modules": sorted(peft_config.target_modules),
                                                 "effective_batch": args.batch_size * args.grad_accum,
                                                 "precision": "fp16 AMP" if config.fp16 else "fp32",
                                                 "optimizer": "AdamW (TRL default)", "scheduler": "cosine, 10% warmup",
                                                 "steps": trainer.state.global_step}, indent=1))


if __name__ == "__main__":
    main()
