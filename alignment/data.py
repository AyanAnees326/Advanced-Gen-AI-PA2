"""Build every fixed training / validation / evaluation subset (seed 42).

    python -m alignment.data            # writes data/processed/*.jsonl + manifest

Outputs (one JSON object per line):
    sft_{train,val}.jsonl   {"id", "instruction", "response"}          UltraChat 200k, first turn
    hh_{train,val}.jsonl    {"id", "instruction", "chosen", "rejected"} HH-RLHF base subsets, single turn
    uf_{train,val}.jsonl    {"id", "instruction", "chosen", "rejected"} UltraFeedback binarized
    eval_sets/alpaca_eval_300_seed42.json                              AlpacaEval, 300 prompts
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from alignment.core import format_prompt

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {  # pinned dataset commits (what was downloaded for the submitted runs)
    "HuggingFaceH4/ultrachat_200k": "8049631c405ae6576f93f445c6b8166f76f5505a",
    "Anthropic/hh-rlhf": "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa",
    "HuggingFaceH4/ultrafeedback_binarized": "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85",
    "tatsu-lab/alpaca_eval": "2edc6fad8be6b14ea7230aabfd08188da6b8b814",
}


def fetch(repo, filename):
    return hf_hub_download(repo, filename, repo_type="dataset", revision=REVISIONS[repo])


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def take(rows, n, rng, keep):
    """Shuffle, then keep the first n rows passing ``keep`` (lazy, so we only tokenize what we need)."""
    rows = list(rows)
    rng.shuffle(rows)
    out = []
    for row in rows:
        if keep(row):
            out.append(row)
            if len(out) == n:
                break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} rows passed the filter, wanted {n}")
    return out


def ultrachat(split, n, tok, max_length, rng):
    files = {"train": "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
             "val": "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"}
    df = pd.read_parquet(fetch("HuggingFaceH4/ultrachat_200k", files[split]))
    rows = []
    for prompt_id, messages in zip(df.prompt_id, df.messages):
        if len(messages) >= 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
            instruction, response = messages[0]["content"].strip(), messages[1]["content"].strip()
            if instruction and response:
                rows.append({"id": f"ultrachat-{prompt_id}", "instruction": instruction, "response": response})

    def fits(row):  # whole example + EOS fits, so the model always sees where answers end
        return len(tok.encode(format_prompt(row["instruction"]) + row["response"])) + 1 <= max_length

    return take(rows, n, rng, fits)


def split_hh(text):
    """'\\n\\nHuman: q\\n\\nAssistant: a' -> (q, a); None if not exactly one turn each."""
    if text.count("\n\nHuman:") != 1 or text.count("\n\nAssistant:") != 1:
        return None
    human, _, assistant = text.partition("\n\nAssistant:")
    prompt = human.replace("\n\nHuman:", "", 1).strip()
    return (prompt, assistant.strip()) if prompt and assistant.strip() else None


def hh_rlhf(split, n, tok, max_prompt, rng):
    rows = []
    for subset in ("harmless-base", "helpful-base"):
        path = fetch("Anthropic/hh-rlhf", f"{subset}/{'train' if split == 'train' else 'test'}.jsonl.gz")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                pair = json.loads(line)
                chosen, rejected = split_hh(pair["chosen"]), split_hh(pair["rejected"])
                if chosen and rejected and chosen[0] == rejected[0] and chosen[1] != rejected[1]:
                    rows.append({"id": f"hh-{subset}-{split}-{i}", "subset": subset, "instruction": chosen[0],
                                 "chosen": chosen[1], "rejected": rejected[1]})
    return take(rows, n, rng, lambda r: len(tok.encode(format_prompt(r["instruction"]))) <= max_prompt)


def ultrafeedback(split, n, tok, max_prompt, rng):
    name = "train_prefs" if split == "train" else "test_prefs"
    df = pd.read_parquet(fetch("HuggingFaceH4/ultrafeedback_binarized", f"data/{name}-00000-of-00001.parquet"))
    rows = []
    for r in df.itertuples():
        chosen, rejected = r.chosen[-1]["content"].strip(), r.rejected[-1]["content"].strip()
        if r.score_chosen > r.score_rejected and r.prompt.strip() and chosen and rejected and chosen != rejected:
            rows.append({"id": f"uf-{r.prompt_id}", "instruction": r.prompt.strip(), "chosen": chosen,
                         "rejected": rejected, "score_chosen": float(r.score_chosen),
                         "score_rejected": float(r.score_rejected)})
    return take(rows, n, rng, lambda r: len(tok.encode(format_prompt(r["instruction"]))) <= max_prompt)


def alpaca_eval(n=300):
    with open(fetch("tatsu-lab/alpaca_eval", "alpaca_eval.json"), encoding="utf-8") as f:
        rows = json.load(f)
    picked = random.Random(SEED).sample(range(len(rows)), n)
    return [{"id": f"alpaca-eval-42-{k:04d}", "source_index": i, "instruction": rows[i]["instruction"],
             "dataset": "alpaca_eval", "category": rows[i]["dataset"]} for k, i in enumerate(picked)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--sft_train", type=int, default=30_000)
    ap.add_argument("--pref_train", type=int, default=15_000)
    ap.add_argument("--val", type=int, default=1_000)
    args = ap.parse_args()
    out = Path(args.out_dir)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    counts = {}
    builders = {
        "sft": lambda split, n, rng: ultrachat(split, n, tok, 512, rng),
        "hh": lambda split, n, rng: hh_rlhf(split, n, tok, 256, rng),
        "uf": lambda split, n, rng: ultrafeedback(split, n, tok, 256, rng),
    }
    for name, build in builders.items():
        for split, n in (("train", args.sft_train if name == "sft" else args.pref_train), ("val", args.val)):
            rows = build(split, n, random.Random(f"{SEED}-{name}-{split}"))
            write_jsonl(out / f"{name}_{split}.jsonl", rows)
            counts[f"{name}_{split}"] = len(rows)
            print(f"{name}_{split}: {len(rows)}", flush=True)

    alpaca = alpaca_eval()
    with open(ROOT / "eval_sets" / "alpaca_eval_300_seed42.json", "w", encoding="utf-8") as f:
        json.dump(alpaca, f, ensure_ascii=False, indent=1)
    counts["alpaca_eval"] = len(alpaca)

    manifest = {
        "seed": SEED, "revisions": REVISIONS, "counts": counts, "prompt_template": format_prompt("{instruction}"),
        "filters": {
            "sft": "UltraChat train_sft shard 0 / test_sft; first user->assistant turn; "
                   "formatted prompt + response + EOS <= 512 Qwen tokens",
            "hh": "harmless-base + helpful-base; exactly one Human and one Assistant turn; same prompt; "
                  "chosen != rejected; formatted prompt <= 256 tokens",
            "uf": "train_prefs / test_prefs; score_chosen > score_rejected (ties dropped); formatted prompt <= 256 tokens",
            "alpaca_eval": "random.Random(42).sample(range(805), 300), in sampled order",
        },
    }
    (ROOT / "outputs").mkdir(exist_ok=True)
    with open(ROOT / "outputs" / "data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
