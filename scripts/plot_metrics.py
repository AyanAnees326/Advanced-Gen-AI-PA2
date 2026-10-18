"""Training plots for the report (manual 14.4).

    python scripts/plot_metrics.py            # -> outputs/plots/{sft,dpo,lora_dpo}.png

Reads outputs/<run>/metrics.jsonl. Raw per-step training values are drawn faintly, with a bias-corrected EMA
(weight 0.9 on the previous value) on top; validation points are unsmoothed. The dashed vertical
line marks the step of the saved best checkpoint (min val loss for SFT, max val reward accuracy for DPO).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
EMA = 0.9


def read(run):
    path = OUT / run / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def ema(values):  # bias-corrected, so the first points are not pinned to the first noisy value
    out, s = [], 0.0
    for t, v in enumerate(values, 1):
        s = EMA * s + (1 - EMA) * v
        out.append(s / (1 - EMA**t))
    return out


def best_step(run):
    path = OUT / run / "summary.json"
    return json.loads(path.read_text())["best"]["step"] if path.exists() else None


def series(ax, rows, key, label, color, smooth):
    rows = [r for r in rows if key in r]
    if not rows:
        return
    x, y = [r["step"] for r in rows], [r[key] for r in rows]
    if smooth:
        ax.plot(x, y, color=color, alpha=0.2, lw=0.8)
        ax.plot(x, ema(y), color=color, lw=1.6, label=f"{label} (EMA {EMA})")
    else:
        ax.plot(x, y, "o-", color=color, ms=3.5, lw=1.2, label=label)


def mark_best(ax, step):
    if step is not None:
        ax.axvline(step, color="k", ls="--", lw=0.9, label=f"best ckpt (step {step})")


def plot_sft():
    rows = read("sft")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    series(ax, [r for r in rows if r["split"] == "train"], "loss", "train loss", "tab:blue", True)
    series(ax, [r for r in rows if r["split"] == "val"], "val_loss", "val loss (1,000 ex.)", "tab:red", False)
    mark_best(ax, best_step("sft"))
    ax.set(xlabel="optimizer step (effective batch 32)", ylabel="response-token NLL",
           title="SFT: Qwen2.5-0.5B on 30k instruction examples")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "sft.png", dpi=160)


def plot_dpo():
    runs = [(run, title) for run, title in (("dpo_hh", "HH-RLHF DPO"), ("dpo_uf", "UltraFeedback DPO")) if read(run)]
    if not runs:
        return
    panels = (("loss", "val_loss", "DPO loss"), ("reward_acc", "val_reward_acc", "reward accuracy"),
              ("reward_margin", "val_reward_margin", "reward margin"))
    fig, axes = plt.subplots(len(runs), 3, figsize=(12, 3.3 * len(runs)), squeeze=False)
    for (run, title), row in zip(runs, axes):
        rows, best = read(run), best_step(run)
        for ax, (train_key, val_key, name) in zip(row, panels):
            series(ax, [r for r in rows if r["split"] == "train"], train_key, "train (per step)", "tab:blue", True)
            series(ax, [r for r in rows if r["split"] == "val"], val_key, "val (1,000 pairs)", "tab:red", False)
            mark_best(ax, best)
            ax.set(xlabel="optimizer step (effective batch 32)", ylabel=name, title=f"{title}: {name}")
            ax.grid(alpha=0.3)
        row[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "dpo.png", dpi=160)


def plot_lora():
    rows = read("lora_dpo")
    if not rows:
        return
    panels = (("loss", "eval_loss", "DPO loss"), ("rewards/accuracies", "eval_rewards/accuracies", "reward accuracy"),
              ("rewards/margins", "eval_rewards/margins", "reward margin"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6))  # printed at half width: small figure = readable labels
    for ax, (train_key, val_key, name) in zip(axes, panels):
        series(ax, rows, train_key, "train (per step)", "tab:blue", True)
        series(ax, rows, val_key, "eval (20 held-out pairs)", "tab:red", False)
        ax.set(xlabel="step (batch 8)", title=name)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("LoRA-DPO: SmolLM2-360M-Instruct, persona pairs")
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "lora_dpo.png", dpi=160)


if __name__ == "__main__":
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    plot_sft()
    plot_dpo()
    plot_lora()
    print(sorted(p.name for p in (OUT / "plots").glob("*.png")))
