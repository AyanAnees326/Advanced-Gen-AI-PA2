"""Batch building, response log-probabilities, and the SFT / DPO objectives.

Everything here works with any tokenizer exposing ``encode`` / ``eos_token_id`` /
``pad_token_id`` and any causal LM returning ``.logits`` of shape (B, T, V).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

# Single-turn ChatML prompt (Qwen's native control tokens, no system message).
# Used verbatim for SFT, DPO and every Qwen generation run.
PROMPT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"


def format_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def _pad_id(tokenizer) -> int:
    for token_id in (getattr(tokenizer, "pad_token_id", None), getattr(tokenizer, "eos_token_id", None)):
        if token_id is not None:
            return int(token_id)
    return 0


def build_lm_batch(tokenizer, prompts, responses, max_length):
    """Right-padded batch where only response tokens (+ EOS) carry labels.

    Prompt and response are tokenized separately, EOS is appended to the response
    (when the tokenizer has one), and the concatenation is truncated to
    ``max_length`` -- so a long response can lose its EOS. Padding is to the
    longest row in the batch. Labels are built from positions, never from token
    ids, so pad_token_id == eos_token_id (Qwen) is handled correctly.
    """
    prompts, responses = list(prompts), list(responses)
    if len(prompts) != len(responses):
        raise ValueError(f"{len(prompts)} prompts vs {len(responses)} responses")
    eos = getattr(tokenizer, "eos_token_id", None)
    rows = []
    for prompt, response in zip(prompts, responses):
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        response_ids = list(tokenizer.encode(response, add_special_tokens=False))
        if eos is not None:
            response_ids.append(int(eos))
        ids = (prompt_ids + response_ids)[:max_length]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
        rows.append((ids, labels))

    width = max([len(ids) for ids, _ in rows] + [1])
    input_ids = torch.full((len(rows), width), _pad_id(tokenizer), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, row_labels) in enumerate(rows):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
        labels[row, : len(row_labels)] = torch.tensor(row_labels, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_preference_batch(tokenizer, prompts, chosen, rejected, max_length):
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen, max_length),
        "rejected": build_lm_batch(tokenizer, prompts, rejected, max_length),
    }


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def sequence_logprobs(model, batch):
    """(summed response log-prob, response token count) per row.

    Logits at position t score token t+1; positions whose shifted label is
    IGNORE_INDEX (prompt, padding) contribute exactly zero. The log-softmax runs
    in fp32 even under fp16 autocast.
    """
    device = model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)[:, 1:]
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    token_logps = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none"
    ).view(labels.shape)
    return token_logps.sum(-1), labels.ne(IGNORE_INDEX).sum(-1)


def sft_loss(model, batch):
    """Mean NLL over every unmasked response token in the batch (token-level mean)."""
    logps, counts = sequence_logprobs(model, batch)
    return -logps.sum() / counts.sum().clamp_min(1)


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta):
    """Batch-mean DPO loss plus detached implicit rewards (r_w, r_l) per row.

    Reference log-probs are treated as constants (detached).
    """
    policy_chosen, policy_rejected = torch.as_tensor(policy_chosen), torch.as_tensor(policy_rejected)
    if not policy_chosen.is_floating_point():
        policy_chosen, policy_rejected = policy_chosen.float(), policy_rejected.float()
    reference_chosen = torch.as_tensor(reference_chosen).detach().to(policy_chosen)
    reference_rejected = torch.as_tensor(reference_rejected).detach().to(policy_chosen)

    chosen_logratio = policy_chosen - reference_chosen
    rejected_logratio = policy_rejected - reference_rejected
    loss = -F.logsigmoid(beta * (chosen_logratio - rejected_logratio)).mean()
    return loss, (beta * chosen_logratio).detach(), (beta * rejected_logratio).detach()


def preference_logprobs(policy, reference, batch):
    """Policy log-probs with grad, reference log-probs under no_grad (on the policy's device)."""
    policy_chosen, _ = sequence_logprobs(policy, batch["chosen"])
    policy_rejected, _ = sequence_logprobs(policy, batch["rejected"])
    with torch.no_grad():
        reference_chosen, _ = sequence_logprobs(reference, batch["chosen"])
        reference_rejected, _ = sequence_logprobs(reference, batch["rejected"])
    device = policy_chosen.device
    return policy_chosen, policy_rejected, reference_chosen.to(device), reference_rejected.to(device)
