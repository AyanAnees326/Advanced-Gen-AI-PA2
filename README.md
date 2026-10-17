# Small-Model Alignment: SFT + DPO

SFT, two from-scratch DPO runs and a TRL/LoRA persona DPO run on small open models, with blinded
pairwise evaluation by a Kaggle model-proxy judge. The report is `report/report.pdf`; the
assignment is `docs/assignment_manual.pdf`.

## Layout

```text
alignment/core.py            batching, response-only masking, log-probs, SFT and DPO losses (graded math)
alignment/data.py            builds data/processed/*.jsonl, eval_sets/alpaca_eval_300_seed42.json, outputs/data_manifest.json
alignment/train.py           custom SFT and DPO training loop (fp32 weights + fp16 autocast + GradScaler, resumable)
alignment/train_lora_dpo.py  Task 5: TRL DPOTrainer + PEFT LoRA on SmolLM2-360M-Instruct
alignment/generate.py        greedy generation (repetition penalty 1.1 in the pipeline) on a fixed eval set -> outputs/generations/<name>.json (+ .meta.json)
alignment/pairs.py           blinded DPO-vs-SFT pair files -> outputs/eval/<comparison>_pairs.jsonl
scripts/judge_pairwise_kaggle.py   resumable batched judge (kaggle_benchmarks), failures + summary files
scripts/plot_metrics.py      outputs/plots/{sft,dpo,lora_dpo}.png
scripts/run_local_pipeline.sh      every stage except judging, re-runnable after a crash
notebooks/01_kaggle_setup.ipynb    Task 1: clone, install without reinstalling torch, 2x T4 check, versions, tests
notebooks/02_kaggle_judge.ipynb    Task 7: judge saved pair files with google/gemini-3.6-flash
submission_adapter.py        grading interface, thin wrappers over alignment/core.py
eval_sets/                   hh_rlhf_eval_300_seed42.json, alpaca_eval_300_seed42.json, persona_style_eval.json
```

Each run directory `outputs/<run>/` holds `config.json` (the values actually used), `metrics.jsonl`,
`summary.json` and, locally only, the `best/` and `final/` checkpoints (not committed: size).

## Reproduce

```bash
python -m pip install -r requirements.txt          # on Kaggle use the install cell in notebooks/01_kaggle_setup.ipynb
PYTHONPATH=. python -m pytest -q tests/
bash scripts/run_local_pipeline.sh                  # data -> SFT -> sanity + decoding pilot -> generations -> LoRA-DPO -> DPO x2 -> pairs -> plots
```

Then judge `outputs/eval/*_pairs.jsonl` in a Kaggle Benchmarks utility notebook
(`notebooks/02_kaggle_judge.ipynb`) and copy its `/kaggle/working/eval/*` back into `outputs/eval/`.
A single file can also be judged directly:

```bash
python scripts/judge_pairwise_kaggle.py \
  --pairs_path outputs/eval/uf_dpo_vs_sft_alpaca_pairs.jsonl \
  --output_path outputs/eval/uf_dpo_vs_sft_alpaca_judged.jsonl \
  --judge_model google/gemini-3.6-flash --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments
```

## Submission adapter

`submission_adapter.py` keeps the six required signatures and calls `alignment/core.py`, the same code
the training loops use. SFT loss reduction: mean over all unmasked response tokens in the batch.

```bash
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```
