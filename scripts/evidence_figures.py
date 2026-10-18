"""Screenshot figures for the report: generation examples and run evidence.

    python scripts/evidence_figures.py      # -> report/figures/{examples,evidence}.png

Text is taken verbatim from the committed generation, judgment and executed-notebook files (and a live run of
the public adapter tests), rendered to HTML and captured with a headless Edge/Chrome. Cut text is marked "…",
line breaks are shown as "↵", ANSI colour codes are stripped.
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
GEN, EVAL, FIG = ROOT / "outputs" / "generations", ROOT / "outputs" / "eval", ROOT / "report" / "figures"
BROWSERS = (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

CSS = """
body { margin: 0; padding: 6px; background: #fff; font: 11.5px/1.27 'Segoe UI', Arial, sans-serif; color: #111; }
.grid { display: grid; grid-template-columns: 20px repeat(3, 1fr); gap: 4px; }
.colhd { font-weight: 600; text-align: center; padding: 3px; background: #e8edf3; border-radius: 3px; }
.rowhd { writing-mode: vertical-rl; transform: rotate(180deg); text-align: center; font-weight: 600;
         background: #e8edf3; border-radius: 3px; }
.card { border: 1px solid #b8c2cc; border-radius: 4px; padding: 3px 5px; }
.id { font: 10.5px Consolas, monospace; color: #555; font-style: normal; }
.p { font-style: italic; margin: 0 0 2px; }
.r { margin: 1px 0; }
.who { font-weight: 600; }
.cand { background: #f3f6ff; }
.v { margin-top: 2px; font-size: 11px; color: #333; }
.chip { display: inline-block; padding: 0 4px; border-radius: 3px; font-weight: 600; color: #fff; }
.win { background: #2e7d32; } .loss { background: #c62828; } .tie { background: #757575; }
.panel { border: 1px solid #b8c2cc; border-radius: 4px; padding: 5px; }
.ttl { font-weight: 600; margin-bottom: 3px; }
.in { font: 10.5px/1.3 Consolas, monospace; white-space: pre-wrap; background: #f5f5f5; border: 1px solid #ddd;
      padding: 3px 5px; margin: 2px 0; }
.out { font: 10.5px/1.3 Consolas, monospace; white-space: pre-wrap; padding: 2px 5px; margin: 0 0 4px; }
.lbl { font: 10.5px Consolas, monospace; color: #303f9f; }
.term { font: 10.5px/1.3 Consolas, monospace; white-space: pre-wrap; background: #1e1e1e; color: #ddd;
        padding: 5px 6px; border-radius: 3px; }
.two { display: grid; grid-template-columns: 1.7fr 1fr; gap: 6px; margin-bottom: 6px; }
"""

# (column, row label, prompt id, verbatim excerpts of baseline, verbatim excerpts of candidate)
EXAMPLES = [
    ("alpaca", "improves", "alpaca-eval-42-0291",
     ["nth term = (n^2 + n) / 2", "Therefore, the 7th term in the sequence is 30."],
     ["The given sequence is an arithmetic progression where each term increases by 9.", "an = 9n - 4"]),
    ("hh", "improves", "hh-eval-42-0000",
     ["1. A weapon: This could be anything from a knife or gun to a baseball bat or golf club."],
     ["I don't have the capability to physically harm anyone. However, here are some general tips that can help "
      "you avoid hurting someone in a fight:"]),
    ("persona", "improves", "persona-eval-029",
     ["Instead, try using a small amount of baking soda or a bit of vinegar to smother the flames."],
     ["You should definitely NOT throw water on your grease fire!",
      "like baking soda or some kind of ember-stopping material!"]),
    ("alpaca", "regresses", "alpaca-eval-42-0104",
     ["The largest ocean in the world is the Pacific Ocean, which covers an area of approximately 14.5 million "
      "square kilometers (6 million square miles)."],
     ["The largest ocean in the world is the Pacific Ocean,",
      "It is also the second-largest ocean by volume and the third-largest by area."]),
    ("hh", "regresses", "hh-eval-42-0053",
     ["It may be helpful to talk to your friend about how they are feeling",
      "consider reaching out to a mental health professional"],
     ["I am sorry to hear this news, but I don't have the ability to understand your friend's feelings or "
      "emotions. Please let me know if there is anything else I can assist you with."]),
    ("persona", "regresses", "persona-eval-028",
     ["your existence has value and meaning."],
     ["Oh no, you're so clueless!", "it means nobody will ever find you?"]),
    ("alpaca", "little change", "alpaca-eval-42-0207",
     ["1. Saudi Arabia\n2. South Korea\n3. Singapore\n4. South Africa\n5. Sri Lanka"],
     ["1. Saudi Arabia\n2. South Korea\n3. Singapore\n4. South Africa\n5. Sri Lanka"]),
    ("hh", "little change", "hh-eval-42-0014",
     ["I am not capable of making financial decisions or providing investment advice.",
      "turning $10,000 into a fortune is a long-term goal."],
     ["I don't have the capability to give you guaranteed investment advice.", "Avoid scams"]),
    ("persona", "little change", "persona-eval-026",
     ["Oh no! You've made your bathroom cleaner even more dangerous for you.", "Can you please tell me what "
      "happened?"],
     ["Oh no! You've made your bathroom cleaner even more dangerous than before!",
      "always ask an adult for help when mixing chemicals!"]),
]
COLUMNS = {  # column -> (header, (baseline file, label), (candidate file, label), judged comparisons)
    "alpaca": ("AlpacaEval-300: UltraFeedback DPO vs SFT", ("sft_alpaca", "SFT"), ("dpo_uf_alpaca", "DPO"),
               [("", "uf_dpo_vs_sft_alpaca")]),
    "hh": ("HH eval-300: HH-RLHF DPO vs SFT", ("sft_hh", "SFT"), ("dpo_hh_hh", "DPO"),
           [("", "hh_dpo_vs_sft_hh")]),
    "persona": ("Persona-32: LoRA-DPO vs SmolLM2", ("smollm_persona", "SmolLM2"),
                ("lora_persona", "LoRA"),
                [("style", "lora_vs_base_persona_style"), ("usefulness", "lora_vs_base_persona_quality")]),
}


def show(s):
    return html.escape(re.sub(r"\n+", " ↵ ", s.strip()))


def excerpt(text, parts):
    for p in parts:
        assert p in text, p
    out = " … ".join(show(p) for p in parts)
    return ("" if text.startswith(parts[0]) else "… ") + out + ("" if text.rstrip().endswith(parts[-1]) else " …")


def load(name):
    return {r["id"]: r for r in json.loads((GEN / f"{name}.json").read_text(encoding="utf-8"))}


def outcomes(comp, pid):
    res = []
    for sfx in ("", "_swapped"):
        rows = [json.loads(l) for l in (EVAL / f"{comp}{sfx}_judged.jsonl").read_text(encoding="utf-8").splitlines()]
        (row,) = [r for r in rows if r["prompt_id"] == pid]
        res.append(row["outcome"])
    return res


def examples_html():
    cells = ["<div></div>"] + [f'<div class="colhd">{COLUMNS[c][0]}</div>' for c in ("alpaca", "hh", "persona")]
    for i, (col, label, pid, base_parts, cand_parts) in enumerate(EXAMPLES):
        if i % 3 == 0:
            cells.append(f'<div class="rowhd">{label}</div>')
        _, (bf, bl), (cf, cl), comps = COLUMNS[col]
        base, cand = load(bf)[pid], load(cf)[pid]
        verdicts = []
        for name, comp in comps:
            o, s = outcomes(comp, pid)
            verdicts.append(f"{name + ' ' if name else ''}<span class='chip {o}'>{o}</span> "
                            f"<span class='chip {s}'>{s}</span>")
        cells.append(
            f'<div class="card"><div class="p"><span class="id">{pid}</span> {show(base["instruction"])}</div>'
            f'<div class="r"><span class="who">{bl}</span> ({len(base["output"].split())} words): '
            f'{excerpt(base["output"], base_parts)}</div>'
            f'<div class="r cand"><span class="who">{cl}</span> ({len(cand["output"].split())} words): '
            f'{excerpt(cand["output"], cand_parts)}</div>'
            f'<div class="v">judge ({cl}): {" · ".join(verdicts)}</div></div>')
    return f'<div class="grid">{"".join(cells)}</div>'


def nb_cells(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))["cells"]


def src(cell):
    return "".join(cell["source"]).strip()


def out(cell):
    return ANSI.sub("", "".join("".join(o.get("text", "")) for o in cell.get("outputs", []))).rstrip()


def nb_block(cell, output, source=None):
    return (f'<div class="lbl">In [{cell["execution_count"]}]:</div>'
            f'<div class="in">{html.escape(src(cell) if source is None else source)}</div>'
            + (f'<div class="out">{html.escape(output)}</div>' if output else ""))


def out_block(cell, output=None):
    return (f'<div class="lbl">Out [{cell["execution_count"]}]:</div>'
            f'<div class="out">{html.escape(out(cell) if output is None else output)}</div>')


def evidence_html():
    setup, judge = nb_cells("notebooks/01_kaggle_setup.ipynb"), nb_cells("notebooks/02_kaggle_judge.ipynb")
    clone_src = src(setup[1]).splitlines()
    a = (nb_block(setup[1], "… " + out(setup[1]).splitlines()[-1], "\n".join([clone_src[0], "…", *clone_src[-3:]]))
         + nb_block(setup[3], "") + out_block(setup[5]))
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/test_public_adapter_smoke.py"]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
    assert run.returncode == 0, run.stdout
    b = html.escape("$ PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py\n" + run.stdout.strip())
    lines = []
    for line in out(judge[6]).splitlines():  # elide the win/loss columns (they are in the results tables)
        line = re.sub(r"'win': .*?'unknown'", "… 'unknown'", line)
        lines.append(re.sub(r"'win_rate_ties_half': .*?'retried_pairs'", "… 'retried_pairs'", line))
    log = (ROOT / "outputs/sft/crash_eval_bs8.log").read_text(encoding="utf-8").splitlines()
    last = [s for s in log if s.startswith('{"split"')][-1]
    i = log.index(next(s for s in log if s.endswith("line 287, in main")))
    j = log.index("torch.AcceleratorError: CUDA error: unspecified launch failure")
    e = "\n".join([last[:last.index('"epoch"')] + "…", "…", '  File "…' + log[i].split("pa2\\")[1], log[i + 1], "…",
                   log[j], log[j + 1][:log[j + 1].index(" in https")] + " …"])
    return (f'<div class="two"><div class="panel"><div class="ttl">(a) Kaggle setup notebook, '
            f'notebooks/01_kaggle_setup.ipynb (GPU T4×2), executed</div>{a}</div>'
            f'<div><div class="panel"><div class="ttl">(b) Public adapter tests (local)</div><div class="term">{b}</div>'
            f'</div><div class="panel" style="margin-top:6px"><div class="ttl">(c) First SFT attempt, '
            f'outputs/sft/crash_eval_bs8.log</div><div class="out">{html.escape(e)}</div></div>'
            f'<div class="panel" style="margin-top:6px"><div class="ttl">(d) Kaggle judge notebook, '
            f'notebooks/02_kaggle_judge.ipynb: judge model check</div>{out_block(judge[3])}</div></div></div>'
            f'<div class="panel"><div class="ttl">(e) Kaggle judge notebook: per-file summary after judging (both passes)</div>'
            f'{out_block(judge[6], chr(10).join(lines))}</div>')


def capture(body, png, width):
    exe = next(p for p in BROWSERS if Path(p).exists())
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>{body}", encoding="utf-8")
        subprocess.run([exe, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=3",
                        f"--window-size={width},2400", f"--screenshot={png}", page.as_uri()],
                       check=True, capture_output=True)
    img = Image.open(png).convert("RGB")
    box = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
    img.crop((0, 0, img.width, min(img.height, box[3] + 12))).save(png, optimize=True)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    capture(examples_html(), FIG / "examples.png", 900)
    capture(evidence_html(), FIG / "evidence.png", 900)
    print(sorted(p.name for p in FIG.glob("*.png")))
