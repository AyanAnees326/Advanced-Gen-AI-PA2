"""Stable grading interface for the SFT and DPO implementation.

Each function is a thin wrapper around ``alignment/core.py``, which is the exact
code used by the SFT and DPO training loops (``alignment/train.py``).

SFT loss reduction: mean over all unmasked response tokens in the batch.
Batches are right-padded to exactly ``max_length``.
"""

from __future__ import annotations

from alignment import core


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return core.build_lm_batch(tokenizer, prompts, responses, max_length, pad_to_max_length=True)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    batch = core.build_lm_batch(tokenizer, prompts, responses, max_length, pad_to_max_length=True)
    return core.sequence_logprobs(model, batch)[0]


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    return core.sft_loss(model, build_sft_batch(tokenizer, prompts, responses, max_length))


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return core.build_preference_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length,
                                       pad_to_max_length=True)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return core.dpo_loss(
        policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta
    )[0]


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    return core.dpo_loss(*core.preference_logprobs(policy_model, reference_model, batch), beta)[0]
