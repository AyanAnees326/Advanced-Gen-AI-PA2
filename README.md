# Small-Model Alignment: SFT + DPO

SFT, two from-scratch DPO runs and a TRL/LoRA persona DPO run on small open models, with blinded
pairwise evaluation by a Kaggle model-proxy judge. The report is `report/report.pdf`; the
assignment is `docs/assignment_manual.pdf`.

Repository: https://github.com/AyanAnees326/Advanced-Gen-AI-PA2 (the executed Kaggle notebooks cloned the same
history from its earlier copy, `AyanAnees326/small-model-alignment`).

## Submission checklist (manual Section 16)

| Item | Where |
|---|---|
| 1. GitHub repository URL | https://github.com/AyanAnees326/Advanced-Gen-AI-PA2 (branch `main`) |
| 2. Short final report (PDF) | `report/report.pdf` (8 pages incl. figures, tables, references; source `report/report.tex`) |
| 3. Training plots for SFT and DPO | `outputs/plots/sft.png`, `outputs/plots/dpo.png` (both DPO runs), `outputs/plots/lora_dpo.png` |
| 4. Evaluation summary, DPO vs SFT | table below; report Sections 5-6; `outputs/eval/*_judged_summary.json`, `outputs/eval/order_debiased_summary.json` |
| 5. Root-level `submission_adapter.py`, public tests passing | `submission_adapter.py`; `PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py` (7 passed) |

## Results

Judge `google/gemini-3.6-flash` (blinded, randomised A/B order); 0 retries, 0 failures, 0 unparseable verdicts.
Every pair was judged a second time with A and B swapped; the last column averages the two orders per prompt
(95% CI from a bootstrap over distinct prompts). Per-pair verdicts and summaries: `outputs/eval/*_judged.jsonl`,
`*_judged_summary.json`, `*_swapped_judged*`, `order_debiased_summary.json`. Plots: `outputs/plots/`.

| Comparison (candidate vs baseline) | n | W / T / L | Win rate, ties = 1/2 | Win rate, no ties | Both orders |
|---|---|---|---|---|---|
| HH-RLHF DPO vs SFT (HH eval-300, safety rubric) | 300 | 167 / 22 / 111 | 59.3% (95% CI 54.0-64.8) | 60.1% | 58.0% (50.7-65.2) |
| UltraFeedback DPO vs SFT (AlpacaEval-300) | 300 | 112 / 56 / 132 | 46.7% (95% CI 41.7-51.8) | 45.9% | 46.5% (42.2-50.7) |
| LoRA-DPO persona vs SmolLM2-360M-Instruct, style | 32 | 17 / 9 / 6 | 67.2% | 73.9% | 60.9% (49.2-72.7) |
| LoRA-DPO persona vs SmolLM2-360M-Instruct, usefulness/safety | 32 | 12 / 1 / 19 | 39.1% | 38.7% | 37.5% (22.7-53.1) |
| UltraFeedback DPO, LR 5e-6 pilot (150 steps) vs SFT (AlpacaEval-300) | 300 | 119 / 34 / 147 | 45.3% | 44.7% | 45.3% (40.9-49.9) |

The LoRA persona rows are reported separately and not combined. The last row is an ablation, not a submitted model:
UltraFeedback DPO at 5x the guide learning rate for 150 steps (`outputs/dpo_uf_lr5e-6/`) gives longer answers and no
better win rate, so the guide run is not mainly learning-rate-limited. Analysis is in Sections 5-6 of the report.

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
`outputs/logs/pipeline_run{1,2,3}.log` are the console logs of the three pipeline invocations: run 1 was cut by a
power failure during generation, run 2 during HH-RLHF DPO at step 355, and run 3 resumed from the step-350 state.

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
