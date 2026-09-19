"""
Why exactly does the base model score 0/45? Reasoning failure, or format?

This distinction matters for what the headline result means. "Base 0/45 ->
SFT 13/45" can be read two ways:

  (a) fine-tuning taught the model to solve equations it could not solve, or
  (b) fine-tuning taught the model to *emit answers in the shape the
      evaluator can read*, on a task it partly understood already.

The evaluation only ever counted correct answers, so it could not tell these
apart, and neither could this repo's write-ups. That was a real gap: the
claim being made is about capability, and the measurement is capability
*plus* format compliance. This script exists to separate them.

It prints, for each problem, the base model's raw decoded output alongside
what the regex extracted, what the verifier said, and the true answer -- and
does the same for the SFT model for contrast. Then it summarises how the
failures break down by verifier reason code.

How to read the result:

  - If base failures are mostly `unparsable`, the 0/45 is substantially a
    formatting story. The honest headline is then that fine-tuning taught
    the output format on a task the model had some grip on, which is a
    weaker and more interesting claim than "taught it to solve equations".
  - If base failures are mostly `wrong` / `incomplete`, the model is
    producing readable answers that are simply incorrect, and the 0/45 is a
    capability story.
  - Most likely it is a mixture, and the point is to report the mixture
    rather than assert either reading.

Whatever it shows gets written up. This is a diagnostic, not a search for a
flattering interpretation.

    python inspect_base_outputs.py            # 12 problems, both models
    python inspect_base_outputs.py 45         # all of them

Needs a GPU; roughly one generation per problem per model.
"""
import re
import sys
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tasks import generate_problems
from verifier import _true_solution_set, verify

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_MERGED_DIR = "../results/sft-merged-model"
HELD_OUT_SEED = 99
N_PER_FAMILY = 15
SYSTEM = "You are a careful algebra solver. Show brief work, then give the final answer line."
OUT_TXT = "../results/base_output_inspection.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_raw(model, tok, equation_str):
    """Identical prompt and decoding to evaluate_redesign_v2.py, but returns
    the raw text as well as the extracted answer, which the evaluator throws
    away."""
    prompt = (
        f"Solve for x: {equation_str}\n"
        "Give the final answer on its own line as 'x = v1, v2' or 'x = no solution'."
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.findall(r"x\s*=\s*([^\n]+)", decoded)
    extracted = f"x = {m[-1]}" if m else (
        decoded.strip().splitlines()[-1] if decoded.strip() else "no solution")
    return decoded, extracted


def main():
    n_show = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    problems = generate_problems(n_per_family=N_PER_FAMILY, seed=HELD_OUT_SEED)[:n_show]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    lines = []

    def say(t=""):
        print(t)
        lines.append(t)

    say(f"Running on: {DEVICE} | inspecting {len(problems)} held-out problems")

    reasons = {}
    for label, path in (("BASE", BASE_MODEL_NAME), ("SFT", SFT_MERGED_DIR)):
        say("\n" + "=" * 74)
        say(f"{label} MODEL RAW OUTPUTS")
        say("=" * 74)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(DEVICE)
        model.eval()
        counter = Counter()
        for p in problems:
            raw, extracted = generate_raw(model, tok, p["equation_str"])
            res = verify(p["equation_str"], extracted)
            counter[res.reason] += 1
            true_sol, _ = _true_solution_set(p["equation_str"])
            say(f"\n--- {p['id']}  {p['equation_str']}   (true: {true_sol})")
            say(f"  raw output   : {raw.strip()[:300]!r}")
            say(f"  extracted    : {extracted!r}")
            say(f"  verdict      : {'CORRECT' if res.correct else res.reason}")
        reasons[label] = counter
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    say("\n" + "=" * 74)
    say("FAILURE BREAKDOWN BY VERIFIER REASON")
    say("=" * 74)
    all_reasons = sorted(set().union(*[set(c) for c in reasons.values()]))
    say(f"{'reason':>18} | " + " | ".join(f"{k:>10}" for k in reasons))
    say("-" * (20 + 13 * len(reasons)))
    for r in all_reasons:
        say(f"{r:>18} | " + " | ".join(f"{reasons[k][r]:>10}" for k in reasons))

    n = len(problems)
    base_unp = reasons["BASE"]["unparsable"]
    say(f"\nOf {n} base-model attempts, {base_unp} were unparsable "
        f"({base_unp/n*100:.0f}%).")
    if base_unp / n >= 0.5:
        say("  => Base failures are dominated by FORMAT. The 0/45 headline is")
        say("     substantially about output shape, not only about capability,")
        say("     and the write-ups should say so.")
    elif base_unp / n <= 0.2:
        say("  => Base failures are dominated by WRONG ANSWERS, not format. The")
        say("     model produces readable answers that are simply incorrect, so")
        say("     0/45 is a capability result.")
    else:
        say("  => Mixed. Report the mixture; neither pure reading is supported.")

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    say(f"\nWritten to {OUT_TXT}")


if __name__ == "__main__":
    main()
