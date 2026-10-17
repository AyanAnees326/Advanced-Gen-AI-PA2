"""Edge cases beyond the public smoke tests: truncation, tokenizer quirks, row order, exact math."""

from __future__ import annotations

import math

import torch

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class PadIsEosTokenizer(ToyTokenizer):  # like Qwen: pad_token_id == eos_token_id
    pad_token_id = 1


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


class PositionalLM(torch.nn.Module):
    """Logits depend on the current token, so prompt context actually matters."""

    def __init__(self, vocab_size):
        super().__init__()
        torch.manual_seed(0)
        self.table = torch.nn.Parameter(torch.randn(vocab_size, vocab_size))

    def forward(self, input_ids, attention_mask=None):
        return type("Out", (), {"logits": self.table[input_ids]})()


def manual_logp(model, tok, prompt, response, max_length):
    ids = (tok.encode(prompt) + tok.encode(response) + ([tok.eos_token_id] if tok.eos_token_id is not None else []))
    ids = ids[:max_length]
    n_prompt = len(tok.encode(prompt))
    logits = model(torch.tensor([ids])).logits[0].log_softmax(-1)
    return sum(logits[t - 1, ids[t]].item() for t in range(max(n_prompt, 1), len(ids)))


def test_truncation_drops_eos_only_when_response_is_cut():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab", "ab"], ["cd", "cdefgh"], max_length=6)
    assert batch["input_ids"].shape == (2, 6)
    assert batch["labels"][0].tolist() == [-100, -100, *tok.encode("cd"), tok.eos_token_id, -100]
    assert batch["labels"][1].tolist() == [-100, -100, *tok.encode("cdef")]  # EOS truncated away
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 0], [1] * 6]


def test_prompt_longer_than_max_length_has_no_labels_and_zero_logp():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=1.0)
    batch = adapter.build_sft_batch(tok, ["abcdefgh", "a"], ["xyz", "b"], max_length=5)
    assert batch["labels"][0].eq(-100).all()
    logps = adapter.compute_response_logprobs(model, tok, ["abcdefgh", "a"], ["xyz", "b"], max_length=5)
    assert logps[0].item() == 0.0 and logps[1].item() < 0
    loss = adapter.compute_sft_loss(model, tok, ["abcdefgh"], ["xyz"], max_length=5)
    assert torch.isfinite(loss) and loss.item() == 0.0


def test_pad_equal_to_eos_keeps_eos_label_and_attention():
    tok = PadIsEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: ", "Q: "], ["a", "abc"], max_length=16)
    assert batch["input_ids"][0].tolist() == [*tok.encode("Q: a"), 1] + [1] * 11
    assert batch["labels"][0].tolist() == [-100] * 3 + [tok.encode("a")[0], 1] + [-100] * 11
    assert batch["attention_mask"][0].tolist() == [1] * 5 + [0] * 11


def test_missing_pad_or_eos_tokens():
    batch = adapter.build_sft_batch(NoPadTokenizer(), ["Q: "], ["a"], max_length=8)
    assert batch["labels"][0, 4].item() == 1 and batch["input_ids"][0, 5:].eq(1).all()  # EOS appended, pads with EOS id
    batch = adapter.build_sft_batch(NoEosTokenizer(), ["Q: ", "Q: "], ["a", "abc"], max_length=8)
    assert batch["labels"][0].tolist() == [-100] * 3 + NoEosTokenizer().encode("a") + [-100] * 4
    assert batch["labels"][1].ne(-100).sum().item() == 3


def test_adapter_batches_are_padded_and_truncated_to_max_length():
    tok = ToyTokenizer()
    for max_length in (4, 9, 20):
        sft = adapter.build_sft_batch(tok, ["Q: ", "long prompt: "], ["a", "longer answer"], max_length)
        dpo = adapter.build_dpo_batch(tok, ["Q: ", "P: "], ["a", "bb"], ["ccc", "d"], max_length)
        for batch in (sft, dpo["chosen"], dpo["rejected"]):
            assert all(v.shape == (2, max_length) for v in batch.values())


def test_float64_model_keeps_float64_logprobs():
    tok = ToyTokenizer()
    model = PositionalLM(tok.vocab_size).double()
    logps = adapter.compute_response_logprobs(model, tok, ["Q: "], ["abc"], 16)
    assert logps.dtype == torch.float64
    torch.testing.assert_close(logps, torch.tensor([manual_logp(model, tok, "Q: ", "abc", 16)], dtype=torch.float64))


def test_logprobs_match_manual_sum_and_are_order_and_padding_invariant():
    tok = ToyTokenizer()
    model = PositionalLM(tok.vocab_size)
    prompts = ["Write: ", "Q? ", "long prompt here: "]
    responses = ["hello there", "x", "mid"]
    for max_length in (64, 12):
        logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length)
        expected = [manual_logp(model, tok, p, r, max_length) for p, r in zip(prompts, responses)]
        torch.testing.assert_close(logps, torch.tensor(expected), rtol=1e-5, atol=1e-5)
        reversed_logps = adapter.compute_response_logprobs(model, tok, prompts[::-1], responses[::-1], max_length)
        torch.testing.assert_close(reversed_logps, logps.flip(0))


def test_sft_loss_is_token_mean():
    tok = ToyTokenizer()
    model = PositionalLM(tok.vocab_size)
    prompts, responses = ["P: ", "Prompt: "], ["a", "abcdef"]
    logps = [manual_logp(model, tok, p, r, 64) for p, r in zip(prompts, responses)]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, 64)
    assert math.isclose(loss.item(), -sum(logps) / (2 + 7), rel_tol=1e-5)


def test_dpo_loss_matches_formula_gradients_and_extremes():
    pc = torch.tensor([-1.0, -3.0, -2.0], requires_grad=True)
    pr = torch.tensor([-2.0, -1.0, -2.5], requires_grad=True)
    rc = torch.tensor([-1.5, -2.0, -2.0], requires_grad=True)
    rr = torch.tensor([-1.5, -2.0, -3.0], requires_grad=True)
    beta = 0.1
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    z = beta * ((pc - pr) - (rc - rr))
    torch.testing.assert_close(loss, -torch.log(torch.sigmoid(z)).mean())
    loss.backward()
    torch.testing.assert_close(pc.grad, -beta * torch.sigmoid(-z.detach()) / 3)
    torch.testing.assert_close(pr.grad, beta * torch.sigmoid(-z.detach()) / 3)
    torch.testing.assert_close(rc.grad, beta * torch.sigmoid(-z.detach()) / 3)  # pure function of all four inputs
    torch.testing.assert_close(rr.grad, -beta * torch.sigmoid(-z.detach()) / 3)
    double = [torch.randn(3, dtype=torch.float64, requires_grad=True) for _ in range(4)]
    assert torch.autograd.gradcheck(lambda *x: adapter.compute_dpo_loss_from_logps(*x, 0.3), double)

    big = torch.tensor([1e4, -1e4])
    zeros = torch.zeros(2)
    extreme = adapter.compute_dpo_loss_from_logps(big, -big, zeros, zeros, 1.0)
    torch.testing.assert_close(extreme, torch.tensor(1e4))  # mean of (0, 2e4)


def test_compute_dpo_loss_equals_tensor_loss_on_separate_logps():
    tok = ToyTokenizer()
    policy, reference = PositionalLM(tok.vocab_size), PositionalLM(tok.vocab_size)
    with torch.no_grad():
        reference.table.add_(0.5 * torch.randn_like(reference.table))
    prompts, chosen, rejected = ["Pick: ", "Q: "], ["good answer", "y"], ["no", "bad bad"]
    loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, beta=0.3, max_length=14)
    lp = lambda m, r: adapter.compute_response_logprobs(m, tok, prompts, r, 14)
    expected = adapter.compute_dpo_loss_from_logps(lp(policy, chosen), lp(policy, rejected),
                                                   lp(reference, chosen), lp(reference, rejected), 0.3)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert reference.table.grad is None and policy.table.grad.abs().sum() > 0


def test_small_optimization_steps_reduce_losses():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    losses = []
    for _ in range(3):
        loss = adapter.compute_sft_loss(model, tok, ["Q: "], ["abc"], 16)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    assert losses[0] > losses[1] > losses[2]

    policy, reference = ToyCausalLM(tok.vocab_size), ToyCausalLM(tok.vocab_size)
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    first = adapter.compute_dpo_loss(policy, reference, tok, ["Q: "], ["a"], ["b"], beta=0.5, max_length=8)
    assert math.isclose(first.item(), math.log(2), rel_tol=1e-6)
    opt.zero_grad(); first.backward(); opt.step()
    second = adapter.compute_dpo_loss(policy, reference, tok, ["Q: "], ["a"], ["b"], beta=0.5, max_length=8)
    assert second.item() < first.item()
