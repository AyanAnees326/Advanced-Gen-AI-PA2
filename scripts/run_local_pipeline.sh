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
# Same decoding for every model in a comparison (manual 12): greedy + repetition penalty 1.1, chosen by the
# pilot below (SFT at penalty 1.0 looped to max_new_tokens on 7/40 prompts, 0/40 at 1.1).
DECODE=(--max_new_tokens 512 --repetition_penalty 1.1 --no_repeat_ngram_size 0)

train() { local out=$1; shift; [ -d "$out/final" ] || "$PY" -m alignment.train --out "$out" --resume "$@"; }
gen() { "$PY" -m alignment.generate "${DECODE[@]}" "$@"; }
plain() { "$PY" -m alignment.generate --max_new_tokens 512 --no_repeat_ngram_size 0 "$@"; }

[ -f data/processed/sft_train.jsonl ] || "$PY" -m alignment.data

# Task 3: SFT
train outputs/sft --mode sft

# Task 6 sanity check (5 prompts, inspected by hand before the full runs; plain greedy)
plain --repetition_penalty 1.0 --model Qwen/Qwen2.5-0.5B --generator qwen2.5-0.5b-base --eval_set eval_sets/alpaca_eval_300_seed42.json --out $GEN/sanity/base_alpaca.json --limit 5
plain --repetition_penalty 1.0 --model outputs/sft/best --generator qwen2.5-0.5b-sft --eval_set eval_sets/alpaca_eval_300_seed42.json --out $GEN/sanity/sft_alpaca.json --limit 5
plain --repetition_penalty 1.0 --model outputs/sft/best --generator qwen2.5-0.5b-sft --eval_set eval_sets/hh_rlhf_eval_300_seed42.json --out $GEN/sanity/sft_hh.json --limit 5

# Decoding pilot: SFT on the first 20 prompts of each set, repetition penalty 1.0 vs 1.1
for rp in 1.0 1.1; do
  plain --repetition_penalty $rp --model outputs/sft/best --generator qwen2.5-0.5b-sft --eval_set eval_sets/alpaca_eval_300_seed42.json --out $GEN/pilot/sft_alpaca_rp$rp.json --limit 20
  plain --repetition_penalty $rp --model outputs/sft/best --generator qwen2.5-0.5b-sft --eval_set eval_sets/hh_rlhf_eval_300_seed42.json --out $GEN/pilot/sft_hh_rp$rp.json --limit 20
done

# Task 2 baseline + SFT generations
gen --model Qwen/Qwen2.5-0.5B --generator qwen2.5-0.5b-base --eval_set eval_sets/alpaca_eval_300_seed42.json --out $GEN/base_alpaca.json
gen --model outputs/sft/best --generator qwen2.5-0.5b-sft --eval_set eval_sets/alpaca_eval_300_seed42.json --out $GEN/sft_alpaca.json
gen --model outputs/sft/best --generator qwen2.5-0.5b-sft --eval_set eval_sets/hh_rlhf_eval_300_seed42.json --out $GEN/sft_hh.json

# Task 5: TRL DPOTrainer + LoRA persona tuning (short, so it runs before the two long DPO runs)
[ -d outputs/lora_dpo/adapter ] || "$PY" -m alignment.train_lora_dpo --out outputs/lora_dpo
gen --model HuggingFaceTB/SmolLM2-360M-Instruct --chat_template --generator smollm2-360m-instruct --eval_set eval_sets/persona_style_eval.json --out $GEN/smollm_persona.json
gen --model HuggingFaceTB/SmolLM2-360M-Instruct --adapter outputs/lora_dpo/adapter --chat_template --generator smollm2-360m-lora-dpo --eval_set eval_sets/persona_style_eval.json --out $GEN/lora_persona.json

# Task 4: two from-scratch DPO runs from the same SFT best checkpoint (also the frozen reference)
train outputs/dpo_uf --mode dpo --data uf --init outputs/sft/best
plain --repetition_penalty 1.0 --model outputs/dpo_uf/best --generator qwen2.5-0.5b-dpo-uf --eval_set eval_sets/alpaca_eval_300_seed42.json --out $GEN/sanity/dpo_uf_alpaca.json --limit 5
gen --model outputs/dpo_uf/best --generator qwen2.5-0.5b-dpo-uf --eval_set eval_sets/alpaca_eval_300_seed42.json --out $GEN/dpo_uf_alpaca.json
train outputs/dpo_hh --mode dpo --data hh --init outputs/sft/best
plain --repetition_penalty 1.0 --model outputs/dpo_hh/best --generator qwen2.5-0.5b-dpo-hh --eval_set eval_sets/hh_rlhf_eval_300_seed42.json --out $GEN/sanity/dpo_hh_hh.json --limit 5
gen --model outputs/dpo_hh/best --generator qwen2.5-0.5b-dpo-hh --eval_set eval_sets/hh_rlhf_eval_300_seed42.json --out $GEN/dpo_hh_hh.json

# Task 6 pair files (blinded, written once) + Task 8 plots
"$PY" -m alignment.pairs
"$PY" scripts/plot_metrics.py
echo "done: judge outputs/eval/*_pairs.jsonl on Kaggle with notebooks/02_kaggle_judge.ipynb"
