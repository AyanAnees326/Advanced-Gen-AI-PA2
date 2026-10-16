"""Custom SFT and DPO training loops (no HF Trainer / TRL).

    python -m alignment.train --mode sft --out outputs/sft
    python -m alignment.train --mode dpo --data hh --init outputs/sft/best --out outputs/dpo_hh

Both modes share: fp32 master weights + fp16 autocast + GradScaler, gradient
accumulation, unscale-then-clip, validation every ``--eval_every`` optimizer steps,
best + final checkpoints, and an append-only ``metrics.jsonl`` written as we go.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment import core
from alignment.data import ROOT, read_jsonl

DEFAULTS = {  # guide configuration (manual sections 9.2 and 10.4)
    "sft": dict(lr=2e-5, batch_size=2, grad_accum=16, warmup_ratio=0.03, schedule="cosine", weight_decay=0.1,
                max_length=512, eval_every=100, optimizer="adamw", init_scale=2.0**16),
    "dpo": dict(lr=1e-6, batch_size=1, grad_accum=32, warmup_ratio=0.0, schedule="constant", weight_decay=0.0,
                max_length=512, eval_every=50, optimizer="adamw", init_scale=2.0**10),
}


def keep_awake():
    if sys.platform == "win32":  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no sleep while training
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def load_model(path, device, dtype=torch.float32):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    return model


def make_optimizer(model, args):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if args.optimizer == "rmsprop":
        return torch.optim.RMSprop(groups, lr=args.lr)
    return torch.optim.AdamW(groups, lr=args.lr)


def lr_lambda(total_steps, warmup_ratio, schedule):
    warmup = int(round(total_steps * warmup_ratio))

    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return fn


def micro_batches(rows, batch_size):
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def sft_batch(tokenizer, rows, max_length):
    return core.build_lm_batch(tokenizer, [core.format_prompt(r["instruction"]) for r in rows],
                               [r["response"] for r in rows], max_length)


def pref_batch(tokenizer, rows, max_length):
    return core.build_preference_batch(tokenizer, [core.format_prompt(r["instruction"]) for r in rows],
                                       [r["chosen"] for r in rows], [r["rejected"] for r in rows], max_length)


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def eval_sft(model, tokenizer, val_rows, args, device):
    model.eval()
    nll, tokens = 0.0, 0
    for rows in micro_batches(val_rows, args.eval_batch_size):
        with autocast(device):
            logps, counts = core.sequence_logprobs(model, sft_batch(tokenizer, rows, args.max_length))
        nll -= logps.sum().item()
        tokens += counts.sum().item()
    model.train()
    return {"val_loss": nll / max(tokens, 1), "val_tokens": tokens}


@torch.no_grad()
def response_logps(model, tokenizer, val_rows, args, device):
    """(chosen, rejected) summed response log-probs for every validation pair."""
    chosen, rejected = [], []
    for rows in micro_batches(val_rows, args.eval_batch_size):
        batch = pref_batch(tokenizer, rows, args.max_length)
        with autocast(device):
            chosen.append(core.sequence_logprobs(model, batch["chosen"])[0].float().cpu())
            rejected.append(core.sequence_logprobs(model, batch["rejected"])[0].float().cpu())
    return torch.cat(chosen), torch.cat(rejected)


def eval_dpo(policy, tokenizer, val_rows, reference_logps, args, device):
    policy.eval()
    policy_chosen, policy_rejected = response_logps(policy, tokenizer, val_rows, args, device)
    policy.train()
    loss, r_w, r_l = core.dpo_loss(policy_chosen, policy_rejected, *reference_logps, args.beta)
    return {"val_loss": loss.item(), "val_reward_acc": (r_w > r_l).float().mean().item(),
            "val_reward_margin": (r_w - r_l).mean().item(), "val_chosen_reward": r_w.mean().item(),
            "val_rejected_reward": r_l.mean().item(),
            "val_policy_chosen_logp": policy_chosen.mean().item(),
            "val_policy_rejected_logp": policy_rejected.mean().item()}


def save(model, tokenizer, path, info):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    (path / "checkpoint_info.json").write_text(json.dumps(info, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sft", "dpo"], required=True)
    ap.add_argument("--data", default=None, help="sft | hh | uf (prefix of data/processed/<data>_{train,val}.jsonl)")
    ap.add_argument("--init", default="Qwen/Qwen2.5-0.5B", help="policy init (SFT best for DPO)")
    ap.add_argument("--reference", default=None, help="frozen DPO reference (defaults to --init)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--grad_accum", type=int)
    ap.add_argument("--warmup_ratio", type=float)
    ap.add_argument("--schedule", choices=["cosine", "constant"])
    ap.add_argument("--weight_decay", type=float)
    ap.add_argument("--optimizer", choices=["adamw", "rmsprop"])
    ap.add_argument("--max_length", type=int)
    ap.add_argument("--init_scale", type=float, help="GradScaler initial scale (DPO grads are summed log-probs, so start low)")
    ap.add_argument("--eval_every", type=int)
    ap.add_argument("--eval_batch_size", type=int, default=2,
                    help="no_grad eval still materialises fp32 logits over the 151k vocab; >2 overflows 12 GB next to Adam state")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train", type=int, default=None, help="debug: cap train rows")
    ap.add_argument("--max_val", type=int, default=None, help="debug: cap val rows")
    ap.add_argument("--max_steps", type=int, default=None, help="debug: stop after N optimizer steps")
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--no_save", action="store_true", help="debug: skip checkpoints")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last/state.pt if it exists")
    args = ap.parse_args(argv)
    for key, value in DEFAULTS[args.mode].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.data = args.data or ("sft" if args.mode == "sft" else None)
    if args.data is None:
        ap.error("--data hh|uf is required for dpo")
    args.reference = args.reference or args.init

    keep_awake()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data_dir = ROOT / "data" / "processed"
    train_rows = read_jsonl(data_dir / f"{args.data}_train.jsonl")[: args.max_train]
    val_rows = read_jsonl(data_dir / f"{args.data}_val.jsonl")[: args.max_val]
    tokenizer = AutoTokenizer.from_pretrained(args.init)
    # Sort validation rows by length so eval batches carry little padding (order does not change the metric).
    length = (lambda r: len(r["instruction"]) + len(r["response"])) if args.mode == "sft" else \
        (lambda r: len(r["instruction"]) + max(len(r["chosen"]), len(r["rejected"])))
    val_rows = sorted(val_rows, key=length)

    policy = load_model(args.init, device)
    if not args.no_grad_ckpt:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.train()
    reference = reference_logps = None
    if args.mode == "dpo":
        reference = load_model(args.reference, ref_device, torch.float16 if ref_device.type == "cuda" else torch.float32)
        reference.eval().requires_grad_(False)
        # Validation reference = the SFT weights scored in the policy's own precision before any update, so the
        # step-0 rewards are exactly 0 (manual 10.5). Training pairs use the frozen fp16 copy above. Cached on
        # disk because after a resume the policy no longer equals the reference.
        cache = out / "reference_val_logps.pt"
        if args.resume and cache.exists():
            reference_logps = torch.load(cache)
        else:
            t0 = time.time()
            policy.eval()
            reference_logps = response_logps(policy, tokenizer, val_rows, args, device)
            policy.train()
            torch.save(reference_logps, cache)
            print(f"cached reference val log-probs in {time.time() - t0:.0f}s", flush=True)

    steps_per_epoch = len(train_rows) // (args.batch_size * args.grad_accum)
    total_steps = min(steps_per_epoch * args.epochs, args.max_steps or 10**12)
    optimizer = make_optimizer(policy, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(total_steps, args.warmup_ratio, args.schedule))
    scaler = torch.amp.GradScaler(device.type, init_scale=args.init_scale, enabled=device.type == "cuda")

    config = {**vars(args), "total_steps": total_steps, "train_rows": len(train_rows), "val_rows": len(val_rows),
              "effective_batch": args.batch_size * args.grad_accum, "device": str(device),
              "reference_device": str(ref_device) if reference is not None else None,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "torch": torch.__version__, "precision": "fp32 weights + fp16 autocast + GradScaler",
              "reference_dtype": "fp16" if reference is not None and ref_device.type == "cuda" else "fp32",
              "val_reference_logps": "SFT init weights in policy precision, scored before step 1" if reference is not None else None,
              "prompt_template": core.PROMPT_TEMPLATE, "sft_loss_reduction": "token mean over the effective batch"}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    metrics_path = out / "metrics.jsonl"
    state_path = out / "last" / "state.pt"
    step, best, elapsed = 0, None, 0.0
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        policy.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, best, elapsed = state["step"], state["best"], state["elapsed"]
        del state
        print(f"resumed at step {step}", flush=True)
    if step:  # drop records logged after the saved state
        kept = [line for line in metrics_path.read_text(encoding="utf-8").splitlines() if json.loads(line)["step"] <= step]
        metrics_path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    else:
        metrics_path.unlink(missing_ok=True)

    def save_state():
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".tmp")
        torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step, "best": best,
                    "elapsed": time.time() - start}, tmp)
        tmp.replace(state_path)  # atomic, so a crash mid-save keeps the previous state

    def log(record):
        record["time"] = round(time.time() - start, 1)
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def evaluate(step):
        if args.mode == "sft":
            return {"step": step, **eval_sft(policy, tokenizer, val_rows, args, device)}
        return {"step": step, **eval_dpo(policy, tokenizer, val_rows, reference_logps, args, device)}

    def is_better(record, best):
        if best is None:
            return True
        if args.mode == "sft":
            return record["val_loss"] < best["val_loss"]
        return record["val_reward_acc"] > best["val_reward_acc"]

    start = time.time() - elapsed
    if args.mode == "dpo" and step == 0:  # val reference logps were scored by this policy, so r_w = r_l = 0 and accuracy is exactly 0
        log({"split": "val", **evaluate(0)})

    done, seen, val = step, 0, {}
    for epoch in range(args.epochs):
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        batches = micro_batches([train_rows[i] for i in order], args.batch_size)
        for start_idx in range(0, len(batches) - args.grad_accum + 1, args.grad_accum):
            seen += 1
            if seen <= done:  # resumed: the data order is seeded, so skipping reproduces the original run
                continue
            window = batches[start_idx : start_idx + args.grad_accum]
            stats = {"loss": 0.0, "tokens": 0, "acc": 0.0, "margin": 0.0, "pairs": 0}
            if args.mode == "sft":
                lm_batches = [sft_batch(tokenizer, rows, args.max_length) for rows in window]
                total_tokens = sum(int(b["labels"][:, 1:].ne(core.IGNORE_INDEX).sum()) for b in lm_batches)
                for batch in lm_batches:  # sum of NLL / tokens in the whole effective batch = exact token mean
                    with autocast(device):
                        logps, counts = core.sequence_logprobs(policy, batch)
                        loss = -logps.sum() / max(total_tokens, 1)
                    scaler.scale(loss).backward()
                    stats["loss"] += loss.item()
                stats["tokens"] = total_tokens
            else:
                for rows in window:
                    batch = pref_batch(tokenizer, rows, args.max_length)
                    with autocast(device):
                        logps = core.preference_logprobs(policy, reference, batch)
                        loss, r_w, r_l = core.dpo_loss(*logps, args.beta)
                    scaler.scale(loss / args.grad_accum).backward()
                    stats["loss"] += loss.item() / args.grad_accum
                    stats["acc"] += (r_w > r_l).float().sum().item()
                    stats["margin"] += (r_w - r_l).sum().item()
                    stats["pairs"] += len(rows)

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip).item()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr = scheduler.get_last_lr()[0]
            scheduler.step()
            step += 1

            record = {"split": "train", "step": step, "epoch": epoch, "loss": stats["loss"], "lr": lr,
                      "grad_norm": grad_norm, "scale": scaler.get_scale()}
            if args.mode == "sft":
                record["tokens"] = stats["tokens"]
            else:
                record["reward_acc"] = stats["acc"] / stats["pairs"]
                record["reward_margin"] = stats["margin"] / stats["pairs"]
            log(record)

            if step % args.eval_every == 0 or step == total_steps:
                val = evaluate(step)
                log({"split": "val", **val})
                if is_better(val, best):
                    best = val
                    if not args.no_save:
                        save(policy, tokenizer, out / "best", {"selected_by": "val_loss" if args.mode == "sft"
                                                               else "val_reward_acc", **val})
                if not args.no_save and step < total_steps:
                    save_state()
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    if not args.no_save:
        save(policy, tokenizer, out / "final", {"step": step, **val})
    shutil.rmtree(state_path.parent, ignore_errors=True)
    (out / "summary.json").write_text(json.dumps({"best": best, "final_step": step,
                                                  "minutes": round((time.time() - start) / 60, 1)}, indent=1))
    print("done", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
