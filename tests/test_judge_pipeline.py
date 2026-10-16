"""Offline checks for the Task 6/7 evaluation plumbing: blinding, A/B mapping, retries, resume."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import judge_pairwise_kaggle as J  # noqa: E402


def make_pairs(path, n=12):
    rows = []
    for i in range(n):
        slot = "A" if i % 3 else "B"
        cand, base = "a long detailed answer " * 3, "short"
        a, b = (cand, base) if slot == "A" else (base, cand)
        rows.append({"pair_id": f"secret_cmp/{i}", "prompt_id": i, "comparison": "secret_cmp", "rubric": "safety",
                     "category": "c", "instruction": f"question {i}", "guidance": "refuse if harmful",
                     "response_a": a, "response_b": b,
                     "hidden": {"model_a": "dpo" if slot == "A" else "sft", "model_b": "sft" if slot == "A" else "dpo",
                                "candidate": "dpo", "baseline": "sft", "candidate_slot": slot}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


def test_prompt_is_blinded(tmp_path):
    prompt = J.build_prompt(make_pairs(tmp_path / "p.jsonl")[:5])
    for leak in ("secret_cmp", "dpo", "sft", "candidate", "baseline"):
        assert leak not in prompt
    assert "Response A" in prompt and "Response B" in prompt and "safe redirection" in prompt


def test_judge_maps_verdicts_retries_and_resumes(tmp_path):
    pairs = make_pairs(tmp_path / "p.jsonl")
    call, name = J.fake_judge()  # picks the longer response = the candidate; batch 1 fails once
    out = tmp_path / "judged.jsonl"
    s = J.judge_file(tmp_path / "p.jsonl", out, call, name, batch_size=5, sleep_seconds=0, reuse=True)
    assert (s["win"], s["loss"], s["failure"]) == (12, 0, 0)
    assert s["total_attempts"] == 5 + 5 * 2 + 2  # batch 1 needed a retry
    judged = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(r["winner"] == "dpo" and r["verdict"] == p["hidden"]["candidate_slot"] for r, p in zip(judged, pairs))

    def never(prompt, chat_name):
        raise AssertionError("completed pairs must not be re-judged")

    assert J.judge_file(tmp_path / "p.jsonl", out, never, name, sleep_seconds=0, reuse=True)["win"] == 12


def test_unresolved_batches_logged_and_parse_fallbacks(tmp_path):
    make_pairs(tmp_path / "p.jsonl", n=5)
    s = J.judge_file(tmp_path / "p.jsonl", tmp_path / "j.jsonl", lambda p, c: "no json", "bad", sleep_seconds=0)
    assert s["failure"] == 5 and s["total_attempts"] == 3
    assert json.loads((tmp_path / "j_failures.jsonl").read_text())["attempts"] == 3
    v = J.parse_verdicts('```json\n{"judgments":[{"pair":1,"verdict":"Response B"},{"pair":2,"verdict":"?"}]}\n```', 2)
    assert v == {1: ("B", ""), 2: ("unknown", "")}
