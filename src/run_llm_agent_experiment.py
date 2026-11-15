"""
The experiment that actually tests the project's central claim, with a
real LLM and a proper control.

WHY THIS EXISTS (read this before quoting any turn-budget number):
`run_experiment.py` runs the agent loop against `flawed_solver`, a synthetic
stand-in whose error rate is *stipulated* to fall across turns
(`severity = max(1, 4 - turn)` in agent_loop.py) and which ignores the
verifier's feedback text entirely. That experiment therefore measures how
much of a given error distribution a turn budget recovers. It CANNOT show
that verifier feedback causes the improvement, because the improvement is a
modelling assumption, not a finding. Treating its 0.295 / 0.540 / 0.797
curve as evidence that "verifier feedback works" is circular, and earlier
drafts of this repo's report overstated it in exactly that way. See
PROVENANCE.md, "An overclaim corrected".

This script fixes that by running a real LLM in the loop against a control
that isolates the one thing in question: does the *content* of the
verifier's feedback help, or would any second and third attempt help just
as much?

  condition "feedback": turn k>1 sees the equation, its own previous wrong
                        answer, and the verifier's structured feedback.
  condition "control":  turn k>1 sees the identical original prompt again
                        and simply re-samples.

The control is the important half. A model given three attempts will solve
more problems than a model given one, simply because sampling three times
is three chances (the pass@k effect). Without the control, that alone would
look like "feedback works". The comparison that carries information is
feedback-vs-control at the same turn budget, not turn 1 vs turn 3.

PAIRING: for a given (problem, seed), turn 1 is generated with the same RNG
seed in both conditions, so the two conditions produce *identical* turn-1
answers and only diverge once feedback enters at turn 2. Turns 2 and 3 are
likewise seeded identically across conditions, so the sampling randomness is
held fixed and the prompt content is the only difference. Turn-1 solve rates
being exactly equal between the conditions is a built-in correctness check.

A CAVEAT REGISTERED IN ADVANCE, not after seeing the result: the SFT and
SFT+DPO models were fine-tuned hard on one prompt shape (see
build_sft_data.py). A revision-style prompt is out of distribution for
them, so they may ignore the feedback and simply re-emit the template. If
the feedback condition shows no gain over the control, that is a genuine
possible outcome and it is at least as likely to be a fact about a
narrowly-fine-tuned 0.5B model as a fact about verifier feedback in
general. Report it either way; do not re-run with a different model until
it comes out favourable.

Requires a GPU. Cost is roughly 45 problems x up to 3 turns x 2 conditions
x N_SEEDS generations; at T4 speeds one seed is a few minutes.

    python run_llm_agent_experiment.py
"""
import csv
import json
import os
import random
import re
from datetime import datetime

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import splits
from tasks import generate_problems
from verifier import verify

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_MERGED_DIR = "../results/sft-merged-model"
DPO_ADAPTER_DIR = "../results/lora-dpo-zeroshot-adapter"

# Which model plays the agent. "sft_dpo" is the strongest available and
# leaves the most headroom to recover; "base" cannot do the task at all
# (0/45 zero-shot) so its loop has nothing to refine and is only useful as a
# sanity check.
MODEL_CHOICE = "sft_dpo"  # one of: "sft_dpo", "sft", "base"

HELD_OUT_SEED = 99
N_PER_FAMILY = 15  # 45 problems, the same held-out set as evaluate_redesign_v2.py
MAX_TURNS = 3
N_SEEDS = 3
TEMPERATURE = 0.7

SYSTEM = "You are a careful algebra solver. Show brief work, then give the final answer line."

# "verbose" reproduces the first recorded run (and its format-breakage
# artifact); "minimal" is the repaired prompt. See revision_prompt() below.
REVISION_PROMPT_STYLE = "minimal"  # one of: "minimal", "verbose"

# The style is part of the filename on purpose. The first run's trajectory
# log is evidence for a documented finding, and a re-run under a different
# prompt must not silently overwrite it. "verbose" keeps the original,
# unsuffixed name so the existing file stays the file it always was.
OUT_CSV = (
    "../results/llm_agent_experiment.csv"
    if REVISION_PROMPT_STYLE == "verbose"
    else f"../results/llm_agent_experiment_{REVISION_PROMPT_STYLE}.csv"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    if MODEL_CHOICE == "base":
        return AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, dtype=torch.float32).to(DEVICE)
    model = AutoModelForCausalLM.from_pretrained(SFT_MERGED_DIR, dtype=torch.float32).to(DEVICE)
    if MODEL_CHOICE == "sft_dpo":
        model = PeftModel.from_pretrained(model, DPO_ADAPTER_DIR).to(DEVICE)
    return model


def base_prompt(equation_str):
    return (
        f"Solve for x: {equation_str}\n"
        "Give the final answer on its own line as 'x = v1, v2' or 'x = no solution'."
    )


def revision_prompt(equation_str, prev_candidate, feedback):
    """The turn-2/3 prompt for the feedback condition.

    Two variants, because the first one turned out to be a broken measuring
    instrument. See results/colab_run/agent_experiment.md.

    "verbose" is what produced the first recorded run. It adds two lines AND
    rewrites the final instruction line ("a corrected final answer" instead
    of "the final answer"). For a model SFT'd 300 times on one exact prompt,
    that was enough to wreck its output format: parse failures went from
    6.2% in the control to 38.5% in the feedback condition at turn 2. Those
    all score as wrong, so the comparison was measuring format breakage, not
    whether feedback helps.

    "minimal" keeps the first and last lines byte-identical to the training
    prompt and inserts exactly one line of feedback between them, which is
    the smallest perturbation that still delivers the verifier's message.

    Changing this and re-running is legitimate here for a specific reason,
    not because the first answer was unwelcome: the hypothesis that the
    revision prompt breaks the output format was registered in this file's
    docstring BEFORE the run, the analysis confirmed it, and an instrument
    that corrupts 1 in 3 of its own measurements cannot answer the question
    it was built for. The original run stays on the record either way, and
    whatever the re-run shows gets reported.
    """
    if REVISION_PROMPT_STYLE == "verbose":
        return (
            f"Solve for x: {equation_str}\n"
            f"Your previous answer was: {prev_candidate}\n"
            f"That answer is wrong. Verifier feedback: {feedback}\n"
            "Give a corrected final answer on its own line as 'x = v1, v2' or 'x = no solution'."
        )
    # "minimal": line 1 and line 3 are exactly the trained prompt.
    return (
        f"Solve for x: {equation_str}\n"
        f"A previous attempt answered {prev_candidate}, which is wrong. {feedback}\n"
        "Give the final answer on its own line as 'x = v1, v2' or 'x = no solution'."
    )


def generate(model, tok, user_content, gen_seed):
    """One sampled attempt. gen_seed fixes the sampling randomness so the two
    conditions differ only in user_content."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    torch.manual_seed(gen_seed)
    out = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=0.9,
    )
    decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.findall(r"x\s*=\s*([^\n]+)", decoded)
    if m:
        return f"x = {m[-1]}"
    # An empty generation is NOT a claim of "no solution". Returning that
    # string handed the model a free correct answer on the 8 of 45 held-out
    # problems that have none; "" scores as unparsable, which is what an
    # unreadable answer is. See PROVENANCE.md, bug #6.
    return decoded.strip().splitlines()[-1] if decoded.strip() else ""


def run_trajectory(model, tok, problem, seed, use_feedback):
    """Returns (solved, turns_used, rows). Identical control flow in both
    conditions; only the turn>=2 prompt differs."""
    eq = problem["equation_str"]
    rng = random.Random(f"{problem['id']}|{seed}")
    gen_seeds = [rng.randrange(2**31) for _ in range(MAX_TURNS)]

    rows = []
    prev_candidate = None
    prev_feedback = None
    for turn in range(1, MAX_TURNS + 1):
        if turn == 1 or not use_feedback:
            user_content = base_prompt(eq)
        else:
            user_content = revision_prompt(eq, prev_candidate, prev_feedback)

        candidate = generate(model, tok, user_content, gen_seeds[turn - 1])
        result = verify(eq, candidate)
        rows.append(
            {
                "condition": "feedback" if use_feedback else "control",
                "seed": seed,
                "problem_id": problem["id"],
                "family": problem["family"],
                "turn": turn,
                "candidate": candidate,
                "correct": result.correct,
                "reason": result.reason,
            }
        )
        if result.correct:
            return True, turn, rows
        prev_candidate = candidate
        prev_feedback = result.feedback
    return False, MAX_TURNS, rows


def describe_model_under_test():
    """Identify WHICH trained model this run used.

    This matters: train_lora_sft.py and train_lora_dpo_zeroshot.py overwrite
    their output directories, and GPU training is not bit-reproducible, so
    re-running the pipeline replaces the model here with a different one
    (four observed runs spanned 0.444 to 0.311 held-out). The
    feedback-vs-control comparison below is paired *within* whatever model is
    loaded, so it stays valid either way -- but the absolute rates are tied
    to one specific checkpoint, and without this stamp a results file cannot
    be matched back to the model that produced it.
    """
    info = {"model_choice": MODEL_CHOICE}
    for label, path in (("sft_merged", SFT_MERGED_DIR), ("dpo_adapter", DPO_ADAPTER_DIR)):
        if MODEL_CHOICE == "base" and label != "base":
            continue
        if os.path.isdir(path):
            newest = max(
                (os.path.getmtime(os.path.join(r, f))
                 for r, _, fs in os.walk(path) for f in fs),
                default=0.0,
            )
            info[label] = {
                "path": path,
                "modified": datetime.fromtimestamp(newest).isoformat(timespec="seconds"),
            }
        else:
            info[label] = {"path": path, "modified": "MISSING"}
    return info


def main():
    # The canonical benchmark, from splits.held_out(). Constructing it here
    # with a bare generate_problems() is how six held-out equations ended up
    # in the SFT training file; splits.py is the single definition and the
    # builders exclude it by construction. See PROVENANCE.md, bug 7.
    problems = splits.held_out()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = load_model()
    model.eval()
    print(f"Running on: {DEVICE} | model: {MODEL_CHOICE} | {len(problems)} problems "
          f"x {N_SEEDS} seeds x 2 conditions", flush=True)
    model_info = describe_model_under_test()
    print("Model under test (re-training the pipeline replaces this):", flush=True)
    for k, v in model_info.items():
        if isinstance(v, dict):
            print(f"  {k}: {v['path']}  (last modified {v['modified']})", flush=True)
    with open(OUT_CSV.replace(".csv", "_model.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    all_rows = []
    # solved_at[condition][seed][turn_budget] = count
    solved_at = {"feedback": {}, "control": {}}

    for seed in range(N_SEEDS):
        for use_feedback in (True, False):
            cond = "feedback" if use_feedback else "control"
            counts = {1: 0, 2: 0, 3: 0}
            for i, p in enumerate(problems, 1):
                solved, turns_used, rows = run_trajectory(model, tok, p, seed, use_feedback)
                all_rows.extend(rows)
                if solved:
                    for budget in range(turns_used, MAX_TURNS + 1):
                        counts[budget] += 1
                if i % 15 == 0:
                    print(f"  [seed {seed}/{cond}] {i}/{len(problems)} problems", flush=True)
            solved_at[cond][seed] = counts
            print(f"  seed {seed} {cond}: "
                  + ", ".join(f"turn<={b}: {counts[b]/len(problems):.3f}" for b in (1, 2, 3)),
                  flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    n = len(problems)
    print(f"\n=== Real-LLM agent loop ({MODEL_CHOICE}), {n} held-out problems, "
          f"{N_SEEDS} seeds ===")
    print(f"{'turn budget':>12} | {'feedback':>9} | {'control':>9} | {'difference':>10}")
    print("-" * 50)
    means = {}
    for budget in (1, 2, 3):
        fb = sum(solved_at["feedback"][s][budget] for s in range(N_SEEDS)) / (N_SEEDS * n)
        ct = sum(solved_at["control"][s][budget] for s in range(N_SEEDS)) / (N_SEEDS * n)
        means[budget] = (fb, ct)
        print(f"{budget:>12} | {fb:>9.3f} | {ct:>9.3f} | {fb - ct:>+10.3f}")

    print("\nHow to read this:")
    print("  Turn budget 1 must be identical in both columns (same prompt, same")
    print("  sampling seed). If it is not, the pairing is broken -- treat every")
    print("  other number here as invalid until that is fixed.")
    print("  The 'difference' column at budgets 2 and 3 is the actual result:")
    print("  it is what the verifier's feedback CONTENT buys, over and above")
    print("  simply being allowed to try again.")
    if abs(means[1][0] - means[1][1]) > 1e-9:
        print("\n  WARNING: turn-1 rates differ between conditions. Pairing is broken.")

    try:
        from scipy import stats
        for budget in (2, 3):
            fb = [solved_at["feedback"][s][budget] / n for s in range(N_SEEDS)]
            ct = [solved_at["control"][s][budget] / n for s in range(N_SEEDS)]
            t, p = stats.ttest_rel(fb, ct)
            print(f"\n  paired t-test at turn budget {budget}: t={t:.3f}, p={p:.4f} "
                  f"(n={N_SEEDS} seeds -- very small, treat as indicative only)")
    except Exception as e:
        print(f"\n  (significance test skipped: {e})")

    print(f"\nPer-turn rows written to {OUT_CSV}")


if __name__ == "__main__":
    main()
