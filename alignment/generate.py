"""Greedy generation on a fixed evaluation set.

    python -m alignment.generate --model outputs/sft/best --generator qwen2.5-0.5b-sft \
        --eval_set eval_sets/alpaca_eval_300_seed42.json --out outputs/generations/sft_alpaca.json

Qwen checkpoints use the same ChatML prompt as training (``core.format_prompt``);
``--chat_template`` uses the tokenizer's own template instead (SmolLM2). Decoding
stops at EOS or ``<|im_end|>``. The decoding settings are written next to the
outputs in ``<out>.meta.json`` so every comparison can be audited.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment import core
from alignment.train import keep_awake


def load(model_path, adapter=None, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(adapter or model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16 if device == "cuda" else torch.float32)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.to(device).eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def stop_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
        ids.add(im_end)
    return sorted(i for i in ids if i is not None)


def build_prompt(tokenizer, instruction, chat_template):
    if chat_template:
        return tokenizer.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False,
                                             add_generation_prompt=True)
    return core.format_prompt(instruction)


@torch.no_grad()
def generate(model, tokenizer, instructions, args):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [build_prompt(tokenizer, x, args.chat_template) for x in instructions]
    order = sorted(range(len(prompts)), key=lambda i: -len(tokenizer.encode(prompts[i])))
    eos = stop_ids(tokenizer)
    results = [None] * len(prompts)
    for start in range(0, len(order), args.batch_size):
        idx = order[start : start + args.batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        out = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new_tokens, eos_token_id=eos,
                             pad_token_id=tokenizer.pad_token_id, repetition_penalty=args.repetition_penalty,
                             no_repeat_ngram_size=args.no_repeat_ngram_size, temperature=None, top_p=None,
                             top_k=None)
        for row, i in enumerate(idx):
            new = out[row, enc["input_ids"].shape[1] :].tolist()
            finished = any(t in eos for t in new)
            if finished:
                new = new[: min(new.index(t) for t in eos if t in new)]
            results[i] = {"output": tokenizer.decode(new, skip_special_tokens=True).strip(),
                          "finished": finished, "new_tokens": len(new)}
        print(f"{min(start + args.batch_size, len(order))}/{len(order)}", flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir (merged before generation)")
    ap.add_argument("--generator", required=True, help="model label stored in every entry")
    ap.add_argument("--eval_set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat_template", action="store_true", help="use the tokenizer chat template (SmolLM2)")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="sanity check on the first N prompts")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    rows = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))[: args.limit]
    if out.exists() and not args.overwrite and len(json.loads(out.read_text(encoding="utf-8"))) == len(rows):
        print(f"skip: {out} is complete")
        return
    keep_awake()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load(args.model, args.adapter, device)
    t0 = time.time()
    results = generate(model, tokenizer, [r["instruction"] for r in rows], args)
    entries = [{"id": r["id"], "instruction": r["instruction"], "output": res["output"], "generator": args.generator,
                "dataset": r.get("dataset", Path(args.eval_set).stem), "category": r.get("category"),
                "finished": res["finished"], "new_tokens": res["new_tokens"]} for r, res in zip(rows, results)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {k: v for k, v in vars(args).items() if k not in ("overwrite",)}
    meta.update(decoding="greedy (do_sample=False, temperature/top_p unused)", stop_token_ids=stop_ids(tokenizer),
                prompt_format="tokenizer chat template" if args.chat_template else core.PROMPT_TEMPLATE,
                minutes=round((time.time() - t0) / 60, 2),
                unfinished=sum(not e["finished"] for e in entries))
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {out} ({meta['minutes']} min, {meta['unfinished']} hit max_new_tokens)")


if __name__ == "__main__":
    main()
