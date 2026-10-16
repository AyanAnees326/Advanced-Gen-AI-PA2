"""Task 6: blinded pairwise comparison files + local response statistics.

    python -m alignment.pairs

Reads ``outputs/generations/*.json`` and writes ``outputs/eval/<comparison>_pairs.jsonl``.
The judge only ever sees ``instruction``, ``response_a``, ``response_b`` and the rubric /
per-prompt guidance; which model sits in slot A is a seeded coin flip per pair and is
stored under ``hidden`` so verdicts can be mapped back after judging. Pair files are
written once, before judging (manual 5.5), and are not rewritten unless ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

from alignment.data import ROOT, SEED

GEN = ROOT / "outputs" / "generations"
EVAL = ROOT / "outputs" / "eval"
EVAL_SETS = {"alpaca": "alpaca_eval_300_seed42.json", "hh": "hh_rlhf_eval_300_seed42.json",
             "persona": "persona_style_eval.json"}
# comparison -> (eval set, candidate generations, baseline generations, rubric)
COMPARISONS = {
    "uf_dpo_vs_sft_alpaca": ("alpaca", "dpo_uf_alpaca", "sft_alpaca", "general"),
    "hh_dpo_vs_sft_hh": ("hh", "dpo_hh_hh", "sft_hh", "safety"),
    "lora_vs_base_persona_style": ("persona", "lora_persona", "smollm_persona", "persona_style"),
    "lora_vs_base_persona_quality": ("persona", "lora_persona", "smollm_persona", "persona_quality"),
}
REFUSAL = re.compile(r"\b(I can(?:'|no)t|I cannot|I won't|I will not|I'm (?:not able|unable|sorry)|I am (?:not able|unable|sorry)"
                     r"|as an AI|I must decline|not (?:able|going) to (?:help|assist|provide))", re.I)


def load_generations(name):
    path = GEN / f"{name}.json"
    return {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))} if path.exists() else None


def build_pairs(comparison, eval_set, candidate, baseline, rubric):
    prompts = json.loads((ROOT / "eval_sets" / EVAL_SETS[eval_set]).read_text(encoding="utf-8"))
    cand, base = load_generations(candidate), load_generations(baseline)
    if cand is None or base is None:
        return None
    rng = random.Random(f"{SEED}-{comparison}")
    pairs = []
    for row in prompts:  # fixed eval-set order
        c, b = cand[row["id"]], base[row["id"]]
        candidate_first = rng.random() < 0.5
        a, bb = (c, b) if candidate_first else (b, c)
        pairs.append({"pair_id": f"{comparison}/{row['id']}", "prompt_id": row["id"], "comparison": comparison,
                      "rubric": rubric, "dataset": row.get("dataset"), "category": row.get("category"),
                      "instruction": row["instruction"], "guidance": row.get("expected_behavior"),
                      "response_a": a["output"], "response_b": bb["output"],
                      "hidden": {"model_a": a["generator"], "model_b": bb["generator"],
                                 "candidate": c["generator"], "baseline": b["generator"],
                                 "candidate_slot": "A" if candidate_first else "B"}})
    return pairs


def stats(entries):
    words = [len(e["output"].split()) for e in entries]
    return {"n": len(entries), "mean_words": round(statistics.mean(words), 1), "median_words": statistics.median(words),
            "unfinished_rate": round(sum(not e.get("finished", True) for e in entries) / len(entries), 3),
            "empty": sum(not e["output"].strip() for e in entries),
            "refusal_rate": round(sum(bool(REFUSAL.search(e["output"])) for e in entries) / len(entries), 3)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    EVAL.mkdir(parents=True, exist_ok=True)
    for comparison, spec in COMPARISONS.items():
        path = EVAL / f"{comparison}_pairs.jsonl"
        if path.exists() and not args.overwrite:
            print(f"skip: {path.name} exists")
            continue
        pairs = build_pairs(comparison, *spec)
        if pairs is None:
            print(f"skip: generations for {comparison} missing")
            continue
        path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
        print(f"wrote {path.name}: {len(pairs)} pairs, candidate in A for "
              f"{sum(p['hidden']['candidate_slot'] == 'A' for p in pairs)}")

    summary = {}  # response-level proxies for the report (length, truncation, refusal keywords)
    for path in sorted(GEN.glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        entries = json.loads(path.read_text(encoding="utf-8"))
        by_category = {}
        for e in entries:
            by_category.setdefault(e.get("category"), []).append(e)
        summary[path.stem] = {"generator": entries[0]["generator"], **stats(entries),
                              "by_category": {k: stats(v) for k, v in sorted(by_category.items(), key=str)}}
    (EVAL / "generation_stats.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
