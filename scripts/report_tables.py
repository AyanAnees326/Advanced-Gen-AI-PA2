"""LaTeX tables for the report, generated from the saved run/eval files so the PDF matches the repository.

    python scripts/report_tables.py   # -> report/generated/{judge,genstats,dpo}.tex
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL, GENDIR = ROOT / "outputs" / "eval", ROOT / "report" / "generated"
LABEL = {"uf_dpo_vs_sft_alpaca": "UF-DPO vs SFT (AlpacaEval-300)", "hh_dpo_vs_sft_hh": "HH-DPO vs SFT (HH eval-300)",
         "lora_vs_base_persona_style": "LoRA-DPO vs SmolLM2 (persona: style)",
         "lora_vs_base_persona_quality": "LoRA-DPO vs SmolLM2 (persona: usefulness/safety)"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(x):
    return "--" if x is None else f"{100 * x:.1f}"


def judge_table():
    rows = []
    for key, label in LABEL.items():
        s = load(EVAL / f"{key}_judged_summary.json")
        if s is None:
            rows.append(f"{label} & \\multicolumn{{8}}{{c}}{{pending Kaggle judging}} \\\\")
            continue
        rows.append(f"{label} & {s['n_pairs']} & {s['win']} / {s['tie']} / {s['loss']} & {s['unknown']} & {s['failure']} & "
                    f"{pct(s['win_rate_ties_half'])} & {pct(s['win_rate_excluding_ties'])} & "
                    f"{s['retried_pairs']} & {s['judge_picked_A']}/{s['judge_picked_B']} \\\\")
    judge = next((s["judge_model"] for k in LABEL if (s := load(EVAL / f"{k}_judged_summary.json"))), "google/gemini-3.6-flash")
    return ("\\begin{tabular}{lrcrrrrrr}\n\\toprule\n"
            "Comparison (candidate vs baseline) & $n$ & W / T / L & Unk. & Fail & WR$_{\\text{tie}=\\frac12}$ & WR$_{\\text{no ties}}$ & Retried & A/B \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
            f"% judge: {judge}\n")


def genstats_table():
    stats = load(EVAL / "generation_stats.json") or {}
    rows = ["\\texttt{%s} & %d & %s & %s & %s & %d & %s \\\\" % (name.replace("_", "\\_"), s["n"], s["mean_words"],
            s["median_words"], pct(s["unfinished_rate"]), s["empty"], pct(s["refusal_rate"])) for name, s in stats.items()]
    return ("\\begin{tabular}{lrrrrrr}\n\\toprule\nGenerations & $n$ & Mean words & Median & Hit max tokens (\\%) & "
            "Empty & Refusal kw. (\\%) \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def dpo_table():
    rows = []
    for run, label in (("dpo_hh", "HH-RLHF DPO"), ("dpo_uf", "UltraFeedback DPO")):
        metrics = ROOT / "outputs" / run / "metrics.jsonl"
        if not metrics.exists():
            continue
        val = [json.loads(line) for line in metrics.read_text().splitlines() if '"val"' in line]
        best = max(val, key=lambda r: (r["val_reward_acc"], -r["step"]))  # first step reaching the max, as in training
        for tag, r in (("step 0", val[0]), ("best", best), ("final", val[-1])):
            rows.append(f"{label} & {tag} & {r['step']} & {r['val_loss']:.4f} & {r['val_reward_acc']:.3f} & "
                        f"{r['val_reward_margin']:.3f} & {r['val_chosen_reward']:.3f} & {r['val_rejected_reward']:.3f} \\\\")
    return ("\\begin{tabular}{llrrrrrr}\n\\toprule\nRun & Checkpoint & Step & Val loss & Reward acc. & Margin & "
            "$\\bar r_w$ & $\\bar r_l$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    GENDIR.mkdir(parents=True, exist_ok=True)
    for name, fn in (("judge", judge_table), ("genstats", genstats_table), ("dpo", dpo_table)):
        (GENDIR / f"{name}.tex").write_text(fn(), encoding="utf-8")
    print(sorted(p.name for p in GENDIR.iterdir()))
