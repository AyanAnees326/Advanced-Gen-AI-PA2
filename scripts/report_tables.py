"""LaTeX tables for the report, generated from the saved run/eval files so the PDF matches the repository.

    python scripts/report_tables.py   # -> report/generated/{judge,genstats,dpo,categories,order}.tex
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL, GENDIR = ROOT / "outputs" / "eval", ROOT / "report" / "generated"
LABEL = {"uf_dpo_vs_sft_alpaca": "UF-DPO vs SFT (AlpacaEval-300)", "hh_dpo_vs_sft_hh": "HH-DPO vs SFT (HH eval-300)",
         "lora_vs_base_persona_style": "LoRA-DPO vs SmolLM2 (persona: style)",
         "lora_vs_base_persona_quality": "LoRA-DPO vs SmolLM2 (persona: usefulness/safety)",
         "uf_dpo_lr5e-6_vs_sft_alpaca": "UF-DPO, LR $5{\\times}10^{-6}$, vs SFT (AlpacaEval-300)"}
SCORE = {"win": 1.0, "tie": 0.5, "loss": 0.0}  # unknown verdicts count as ties


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
    for run, label in (("dpo_hh", "HH-RLHF DPO"), ("dpo_uf", "UltraFeedback DPO"),
                       ("dpo_uf_lr5e-6", "UF DPO, LR $5{\\times}10^{-6}$")):
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


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def wtl(rows):
    w, t, l = (sum(r["outcome"] == o for r in rows) for o in ("win", "tie", "loss"))
    return f"{w}/{t}/{l} ({pct((w + t / 2) / len(rows)) if rows else '--'})"


def category_table():
    """Per-category W/T/L (win rate, ties as half) and the split by which answer is longer (judge length bias)."""
    out = []
    for key, label in LABEL.items():
        judged = jsonl(EVAL / f"{key}_judged.jsonl")
        if not judged:
            continue
        pairs = {p["pair_id"]: p for p in jsonl(EVAL / f"{key}_pairs.jsonl")}

        def cand_longer(r):
            p = pairs[r["pair_id"]]
            cand, base = (p["response_a"], p["response_b"]) if p["hidden"]["candidate_slot"] == "A" else \
                         (p["response_b"], p["response_a"])
            return len(cand.split()) > len(base.split())
        cats = sorted({r["category"] for r in judged})
        cells = [f"{c.replace('_', ' ')}: {wtl([r for r in judged if r['category'] == c])}" for c in cats]
        longer = [r for r in judged if cand_longer(r)]
        shorter = [r for r in judged if not cand_longer(r)]
        cells += [f"\\emph{{cand.\\ longer}} (n={len(longer)}): {wtl(longer)}",
                  f"\\emph{{cand.\\ not longer}} (n={len(shorter)}): {wtl(shorter)}"]
        out.append(f"{label} & " + "; ".join(cells) + " \\\\")
    return ("\\begin{tabular}{lp{0.68\\linewidth}}\n\\toprule\nComparison & W/T/L (WR$_{\\text{tie}=\\frac12}$, \\%) "
            "per prompt category and by answer length \\\\\n\\midrule\n" + "\n".join(out) + "\n\\bottomrule\n\\end{tabular}\n")


def sign_test(k, n):
    """Two-sided exact binomial test of k successes in n trials at p = 1/2."""
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n) if n else 1.0


def order_summary(key):
    """Combine the original and the A/B-swapped judging pass: every prompt is judged once with each model in slot A.

    Per-prompt score = mean of the two outcomes (win 1, tie 1/2, loss 0); the CI is a bootstrap over distinct
    instructions (HH repeats prompts), and the sign test counts prompts the candidate wins / loses on balance.
    """
    orig = {r["prompt_id"]: r for r in jsonl(EVAL / f"{key}_judged.jsonl")}
    swap = {r["prompt_id"]: r for r in jsonl(EVAL / f"{key}_swapped_judged.jsonl")}
    ids = [k for k in orig if k in swap]
    if not ids:
        return None
    instruction = {p["prompt_id"]: p["instruction"] for p in jsonl(EVAL / f"{key}_pairs.jsonl")}
    score = {k: (SCORE.get(orig[k]["outcome"], 0.5) + SCORE.get(swap[k]["outcome"], 0.5)) / 2 for k in ids}
    groups = {}
    for k in ids:
        groups.setdefault(instruction[k], []).append(score[k])
    groups, rng, boots = list(groups.values()), random.Random(42), []
    for _ in range(10000):
        draw = [groups[rng.randrange(len(groups))] for _ in groups]
        boots.append(sum(map(sum, draw)) / sum(map(len, draw)))
    boots.sort()
    picks = [r["verdict"] for k in ids for r in (orig[k], swap[k]) if r["verdict"] in ("A", "B")]
    better, worse = sum(s > 0.5 for s in score.values()), sum(s < 0.5 for s in score.values())
    return {"n_prompts": len(ids),
            "wr_original": sum(SCORE.get(orig[k]["outcome"], 0.5) for k in ids) / len(ids),
            "wr_swapped": sum(SCORE.get(swap[k]["outcome"], 0.5) for k in ids) / len(ids),
            "wr_debiased": sum(score.values()) / len(ids), "ci95": [boots[249], boots[9749]],
            "same_outcome": sum(orig[k]["outcome"] == swap[k]["outcome"] for k in ids),
            "reversed": sum({orig[k]["outcome"], swap[k]["outcome"]} == {"win", "loss"} for k in ids),
            "slot_a_share": picks.count("A") / len(picks) if picks else None,
            "prompts_better_worse": [better, worse], "sign_test_p": sign_test(better, better + worse)}


def order_table():
    rows, summary = [], {}
    for key, label in LABEL.items():
        s = summary[key] = order_summary(key)
        if s is None:
            continue
        rows.append(f"{label} & {s['n_prompts']} & {pct(s['wr_original'])} & {pct(s['wr_swapped'])} & "
                    f"\\textbf{{{pct(s['wr_debiased'])}}} ({pct(s['ci95'][0])}--{pct(s['ci95'][1])}) & "
                    f"{pct(s['same_outcome'] / s['n_prompts'])} & {s['reversed']} & {pct(s['slot_a_share'])} & "
                    f"{s['prompts_better_worse'][0]}/{s['prompts_better_worse'][1]}, {s['sign_test_p']:.2g} \\\\")
    (EVAL / "order_debiased_summary.json").write_text(json.dumps({k: v for k, v in summary.items() if v}, indent=1),
                                                     encoding="utf-8")
    return ("\\begin{tabular}{lrrrlrrrl}\n\\toprule\n"
            "Comparison & $n$ & WR orig. & WR swap. & WR both orders (95\\% CI) & Same (\\%) & Rev. & A (\\%) & "
            "better/worse, $p$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    GENDIR.mkdir(parents=True, exist_ok=True)
    for name, fn in (("judge", judge_table), ("genstats", genstats_table), ("dpo", dpo_table),
                     ("categories", category_table), ("order", order_table)):
        (GENDIR / f"{name}.tex").write_text(fn(), encoding="utf-8")
    print(sorted(p.name for p in GENDIR.iterdir()))
