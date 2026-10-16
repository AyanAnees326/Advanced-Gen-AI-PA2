#!/usr/bin/env bash
# End-to-end local pipeline (everything except Kaggle judging). Every stage is skipped when its output
# already exists, and training stages resume from <out>/last/state.pt, so the script can simply be re-run
# after a crash or reboot.
#
#   PY=python bash scripts/run_local_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
GEN=outputs/generations
# Same decoding for every model in a comparison (manual 8 / 12): greedy, chosen after the sanity check.
DECODE=(--max_new_tokens 512 --repetition_penalty 1.0 --no_repeat_ngram_size 0)

train() { local out=$1; shift; [ -d "$out/final" ] || "$PY" -m alignment.train --out "$out" --resume "$@"; }
gen() { "$PY" -m alignment.generate "${DECODE[@]}" "$@"; }

[ -f data/processed/sft_train.jsonl ] || "$PY" -m alignment.data

# Task 3: SFT
train outputs/sft --mode sft

# Task 6 sanity check (2-5 prompts, inspected by hand before the full runs)
gen --model Qwen/Qwen2.5-0.5B --generator qwen2.5-0.5b-base --eval_set eval_sets/alpaca_eval_300_seed42.json --out $GEN/sanity/base_alpaca.json --limit 5
gen --model outputs/sft/best --generator qwen2.5-0.5b-sft --eval_set eval_sets/alpaca_eval_300_seed42.json --out $GEN/sanity/sft_alpaca.json --limit 5
gen --model outputs/sft/best --generator qwen2.5-0.5b-sft --eval_set eval_sets/hh_rlhf_eval_300_seed42.json --out $GEN/sanity/sft_hh.json --limit 5

# Task 2 baseline + SFT generations
gen --model Qwen/Qwen2.5-0.5B --generator qwen2.5-0.5b-base --eval_set eval_sets/alpaca_eval_300_seed42.json --out $GEN/base_alpaca.json
gen --model outputs/sft/best --generator qwen2.5-0.5b-sft --eval_set eval_sets/alpaca_eval_300_seed42.json --out $GEN/sft_alpaca.json
gen --model outputs/sft/best --generator qwen2.5-0.5b-sft --eval_set eval_sets/hh_rlhf_eval_300_seed42.json --out $GEN/sft_hh.json

# Task 4: two from-scratch DPO runs from the same SFT best checkpoint (also the frozen reference)
train outputs/dpo_uf --mode dpo --data uf --init outputs/sft/best
gen --model outputs/dpo_uf/best --generator qwen2.5-0.5b-dpo-uf --eval_set eval_sets/alpaca_eval_300_seed42.json --out $GEN/dpo_uf_alpaca.json
train outputs/dpo_hh --mode dpo --data hh --init outputs/sft/best
gen --model outputs/dpo_hh/best --generator qwen2.5-0.5b-dpo-hh --eval_set eval_sets/hh_rlhf_eval_300_seed42.json --out $GEN/dpo_hh_hh.json

# Task 5: TRL DPOTrainer + LoRA persona tuning
[ -d outputs/lora_dpo/adapter ] || "$PY" -m alignment.train_lora_dpo --out outputs/lora_dpo
gen --model HuggingFaceTB/SmolLM2-360M-Instruct --chat_template --generator smollm2-360m-instruct --eval_set eval_sets/persona_style_eval.json --out $GEN/smollm_persona.json
gen --model HuggingFaceTB/SmolLM2-360M-Instruct --adapter outputs/lora_dpo/adapter --chat_template --generator smollm2-360m-lora-dpo --eval_set eval_sets/persona_style_eval.json --out $GEN/lora_persona.json

# Task 6 pair files (blinded, written once) + Task 8 plots
"$PY" -m alignment.pairs
"$PY" scripts/plot_metrics.py
echo "done: judge outputs/eval/*_pairs.jsonl on Kaggle with notebooks/02_kaggle_judge.ipynb"
