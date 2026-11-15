"""
Feedback-protocol ablation: is verifier feedback unhelpful, or is it being
delivered badly?

WHY THIS EXPERIMENT EXISTS

Two controlled runs (results/colab_run/agent_experiment.md) found that
verifier feedback never beat a re-sampling control, and that the feedback
prompt simultaneously wrecked the model's output format (26-38% unparsable
against the control's 5-9%). Those two facts are confounded: a prompt that
destroys a quarter of its own measurements cannot be used to judge whether
the information it carries is useful.

This experiment separates them by varying *only* the feedback protocol while
holding the model, problems, turn budget and sampling seeds fixed. The
conditions form a ladder of increasing information, with the last one adding
format protection rather than information:

  retry        original prompt again. Pure pass@k control; no signal at all.
  generic      told the answer is wrong, told nothing about why.
  diagnostic   the verifier's structured reason (what previous runs used).
  actionable   the reason plus a concrete procedure to apply it.
  constrained  actionable, plus an explicit restatement of the answer format.

The comparisons that carry information:

  generic vs retry          does merely being told "wrong" change anything?
  diagnostic vs generic     does the CONTENT of the verifier's message help?
  actionable vs diagnostic  does making the message procedural help?
  constrained vs actionable does protecting the output format recover the
                            loss that the format degradation causes?

If `constrained` closes the gap to `retry`, the earlier negative result was
about delivery, and the fix is a protocol. If every feedback condition
trails `retry` even with the format protected, the negative result is about
the feedback content on this model, which is a cleaner finding than anything
the confounded runs could support.

WHAT IT MEASURES BEYOND ACCURACY

Solve rate alone cannot distinguish "feedback fixed a wrong answer" from
"feedback broke a right one". This logs every attempt and reports:

  repair rate        wrong at turn 1 -> correct later
  regression rate    correct at turn 1 -> wrong later (see MEASURE_REGRESSION)
  format validity    fraction of attempts the extractor could read
  answer-change rate did the model change its answer at all
  feedback use       when the verifier named a specific bad value, did the
                     model actually drop that value from its next answer

The last one is the most direct test of whether the model uses the
verifier's information rather than merely re-rolling.

NO PREDICTION IS MADE ABOUT THE OUTCOME. Every condition's result is
reported, including the possibility that nothing beats `retry`.

Cost: CONDITIONS x problems x turns x seeds generations. The default is
about 2000 generations, roughly 45 minutes on a T4. The script prints an
estimate and the exact budget before starting.

    python feedback_protocol_ablation.py
    python feedback_protocol_ablation.py retry diagnostic constrained   # subset
"""
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as avs_config
from feedback_prompts import (ALL_CONDITIONS, SYSTEM, base_prompt,
                               build_prompt, named_values)
import splits
from tasks import generate_problems
from verifier import verify

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_MERGED_DIR = "../results/sft-merged-model"
DPO_ADAPTER_DIR = "../results/lora-dpo-zeroshot-adapter"
# Every setting below resolves through avs_config: env var > the YAML named by
# AVS_CONFIG (configs/agent_ablation.yaml) > the literal default here. That is
# what makes `AVS_N_SEEDS=10 bash scripts/run_ablation.sh` and the Slurm array
# job sweep without editing this file, and config.dump() records what a run
# actually used beside its results.
MODEL_CHOICE = avs_config.get("model_choice", "sft_dpo")  # "sft_dpo" | "sft" | "base"

HELD_OUT_SEED = avs_config.get("held_out_seed", 99)
N_PER_FAMILY = avs_config.get("n_per_family", 15)   # x3 families = 45 problems
MAX_TURNS = avs_config.get("max_turns", 3)
N_SEEDS = avs_config.get("n_seeds", 3)
TEMPERATURE = avs_config.get("temperature", 0.7)

# Keep revising even after a correct answer, to observe regression
# (correct -> wrong). OFF by default, for a reason worth stating.
#
# With an exact verifier the deployed agent STOPS as soon as an answer is
# correct, so regression cannot occur in deployment at all. Measuring it
# therefore means prompting the model after a correct answer, and the
# feedback conditions have nothing truthful to say at that point -- an
# earlier version of this file told the model "your previous answer was
# incorrect. Correct." That false premise would have produced a regression
# number that measured the lie rather than the model.
#
# When enabled, revisions after a correct answer now use a truthful
# confirmation prompt (see build_prompt). Read any resulting regression rate
# as a robustness probe -- what would happen if the verifier were fallible --
# and not as a deployed failure mode.
MEASURE_REGRESSION = avs_config.get("measure_regression", False)

OUT_CSV = "../results/feedback_protocol_ablation.csv"
OUT_TXT = "../results/feedback_protocol_ablation.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Prompt construction lives in feedback_prompts.py: it is pure, it is where
# the experiment's validity actually rests (no answer leakage; `retry` must be
# a true control; never tell the model it was wrong when it was right), and
# keeping it out of this file means those properties can be tested on CPU
# instead of only observed on a GPU. See tests/test_experiment_validity.py.


# ------------------------------------------------------------- generation --

def load_model():
    if MODEL_CHOICE == "base":
        return AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, dtype=torch.float32).to(DEVICE)
    model = AutoModelForCausalLM.from_pretrained(SFT_MERGED_DIR, dtype=torch.float32).to(DEVICE)
    if MODEL_CHOICE == "sft_dpo":
        model = PeftModel.from_pretrained(model, DPO_ADAPTER_DIR).to(DEVICE)
    return model


def generate(model, tok, user_content, gen_seed):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_content}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    torch.manual_seed(gen_seed)
    out = model.generate(**inputs, max_new_tokens=150, do_sample=True,
                         temperature=TEMPERATURE, top_p=0.9)
    decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.findall(r"x\s*=\s*([^\n]+)", decoded)
    if m:
        return f"x = {m[-1]}"
    # An empty generation is NOT a claim of "no solution". Returning that
    # string handed the model a free correct answer on the 8 of 45 held-out
    # problems that have none; "" scores as unparsable, which is what an
    # unreadable answer is. See PROVENANCE.md, bug #6.
    return decoded.strip().splitlines()[-1] if decoded.strip() else ""


def run_trajectory(model, tok, problem, seed, condition):
    """One problem under one condition. Turn 1 uses the same seed in every
    condition, so all conditions share an identical turn-1 answer -- the
    built-in check that the pairing holds."""
    eq = problem["equation_str"]
    rng = random.Random(f"{problem['id']}|{seed}")
    gen_seeds = [rng.randrange(2**31) for _ in range(MAX_TURNS)]

    rows, prev_candidate, prev_result = [], None, None
    solved_at = None
    for turn in range(1, MAX_TURNS + 1):
        if turn == 1:
            user = base_prompt(eq)
        else:
            user = build_prompt(condition, eq, prev_candidate, prev_result)

        cand = generate(model, tok, user, gen_seeds[turn - 1])
        res = verify(eq, cand)

        dropped = None
        if prev_result is not None:
            named = named_values(prev_result.feedback)
            dropped = bool(named) and not (named & named_values(cand))

        rows.append({
            "condition": condition, "seed": seed, "problem_id": problem["id"],
            "family": problem["family"], "turn": turn, "candidate": cand,
            "correct": res.correct, "reason": res.reason,
            "parsable": res.reason != "unparsable",
            "changed": (prev_candidate is not None and cand.strip() != prev_candidate.strip()),
            "acted_on_named_value": dropped,
        })
        if res.correct and solved_at is None:
            solved_at = turn
            if not MEASURE_REGRESSION:
                break
        prev_candidate, prev_result = cand, res
    return solved_at, rows


# ---------------------------------------------------------------- metrics --

def summarise(rows, conditions, n_problems, say):
    by = defaultdict(list)
    for r in rows:
        by[(r["condition"], r["seed"], r["problem_id"])].append(r)
    for v in by.values():
        v.sort(key=lambda r: r["turn"])

    say("\n" + "=" * 78)
    say("SOLVE RATE BY TURN BUDGET (first correct turn, as a stopping loop would)")
    say("=" * 78)
    say(f"{'condition':>12} | {'turn<=1':>8} | {'turn<=2':>8} | {'turn<=3':>8} | {'vs retry@3':>10}")
    say("-" * 60)
    solve = {}
    for c in conditions:
        per_budget = []
        for b in (1, 2, 3):
            hits = 0
            total = 0
            for (cond, seed, pid), turns in by.items():
                if cond != c:
                    continue
                total += 1
                first = next((t["turn"] for t in turns if t["correct"]), None)
                if first is not None and first <= b:
                    hits += 1
            per_budget.append(hits / total if total else 0.0)
        solve[c] = per_budget
    for c in conditions:
        delta = solve[c][2] - solve["retry"][2] if "retry" in solve else float("nan")
        d = f"{delta:+.3f}" if "retry" in solve else "n/a"
        say(f"{c:>12} | {solve[c][0]:>8.3f} | {solve[c][1]:>8.3f} | {solve[c][2]:>8.3f} | {d:>10}")

    t1 = {c: solve[c][0] for c in conditions}
    if len(set(round(v, 6) for v in t1.values())) != 1:
        say("\n  WARNING: turn-1 rates differ between conditions. The pairing is")
        say("  broken and every other number here is invalid until that is fixed.")
    else:
        say(f"\n  Turn-1 identical across all conditions ({solve[conditions[0]][0]:.3f}) -- pairing holds.")

    say("\n" + "=" * 78)
    say("MECHANISM METRICS")
    say("=" * 78)
    say(f"{'condition':>12} | {'repair':>8} | {'regress':>8} | {'format ok':>9} | "
        f"{'changed':>8} | {'used fb':>8}")
    say("-" * 70)
    for c in conditions:
        wrong1 = repaired = corr1 = regressed = 0
        parsable = attempts = changed = changeable = 0
        acted = acted_n = 0
        for (cond, seed, pid), turns in by.items():
            if cond != c:
                continue
            first = turns[0]
            later = turns[1:]
            if not first["correct"]:
                wrong1 += 1
                if any(t["correct"] for t in later):
                    repaired += 1
            else:
                corr1 += 1
                if later and not later[-1]["correct"]:
                    regressed += 1
            for t in turns:
                attempts += 1
                parsable += int(t["parsable"])
            for t in later:
                changeable += 1
                changed += int(bool(t["changed"]))
                if t["acted_on_named_value"] is not None:
                    acted_n += 1
                    acted += int(bool(t["acted_on_named_value"]))
        rep = f"{repaired/wrong1:.3f}" if wrong1 else "n/a"
        reg = f"{regressed/corr1:.3f}" if corr1 else "n/a"
        fmt = f"{parsable/attempts:.3f}" if attempts else "n/a"
        chg = f"{changed/changeable:.3f}" if changeable else "n/a"
        use = f"{acted/acted_n:.3f}" if acted_n else "n/a"
        say(f"{c:>12} | {rep:>8} | {reg:>8} | {fmt:>9} | {chg:>8} | {use:>8}")

    say("\n  repair    = of answers wrong at turn 1, fraction corrected later")
    say("  regress   = of answers correct at turn 1, fraction later broken")
    say("  format ok = fraction of all attempts the extractor could read")
    say("  changed   = fraction of revisions that differ from the previous answer")
    say("  used fb   = when the verifier named a bad value, fraction of revisions")
    say("              that dropped it (the direct test of using the message)")

    say("\n" + "=" * 78)
    say("HOW TO READ THIS")
    say("=" * 78)
    say("  'retry' carries no information, so any condition that fails to beat it")
    say("  is not being helped by the verifier's message.")
    say("  generic vs retry        -> does being told 'wrong' matter at all?")
    say("  diagnostic vs generic   -> does the verifier's CONTENT matter?")
    say("  actionable vs diagnostic-> does phrasing it as a procedure matter?")
    say("  constrained vs actionable-> does protecting the answer format recover")
    say("                             the loss caused by format degradation?")
    say("  If 'format ok' is much lower for the feedback conditions, the accuracy")
    say("  comparison is still partly measuring formatting, not reasoning.")


# ------------------------------------------------------------------- main --

def main():
    conditions = [c for c in sys.argv[1:] if c in ALL_CONDITIONS] or ALL_CONDITIONS
    if "retry" not in conditions:
        conditions = ["retry"] + conditions  # the control is not optional

    # The canonical benchmark, from splits.held_out(). Constructing it here
    # with a bare generate_problems() is how six held-out equations ended up
    # in the SFT training file; splits.py is the single definition and the
    # builders exclude it by construction. See PROVENANCE.md, bug 7.
    problems = splits.held_out()
    budget = len(conditions) * len(problems) * MAX_TURNS * N_SEEDS
    print(f"Device: {DEVICE} | model: {MODEL_CHOICE}")
    print(f"Conditions: {conditions}")
    print(f"Budget: {len(conditions)} x {len(problems)} problems x {MAX_TURNS} turns "
          f"x {N_SEEDS} seeds = {budget} generations")
    print(f"  ~{budget * 1.3 / 60:.0f} min on a T4 at ~1.3 s/generation\n", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = load_model()
    model.eval()

    stamp = {"model_choice": MODEL_CHOICE, "conditions": conditions,
             "n_problems": len(problems), "n_seeds": N_SEEDS,
             "measure_regression": MEASURE_REGRESSION}
    for label, path in (("sft_merged", SFT_MERGED_DIR), ("dpo_adapter", DPO_ADAPTER_DIR)):
        if os.path.isdir(path):
            newest = max((os.path.getmtime(os.path.join(r, f))
                          for r, _, fs in os.walk(path) for f in fs), default=0.0)
            stamp[label] = datetime.fromtimestamp(newest).isoformat(timespec="seconds")
    with open(OUT_CSV.replace(".csv", "_model.json"), "w") as f:
        json.dump(stamp, f, indent=2)
    # What the run actually resolved, and from where (env / yaml / default).
    avs_config.dump(OUT_CSV.replace(".csv", "_config.json"))

    rows = []
    for c in conditions:
        for seed in range(N_SEEDS):
            for i, p in enumerate(problems, 1):
                _, r = run_trajectory(model, tok, p, seed, c)
                rows.extend(r)
                if i % 15 == 0:
                    print(f"  [{c} seed {seed}] {i}/{len(problems)}", flush=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out = []
    def say(t=""):
        print(t)
        out.append(t)
    summarise(rows, conditions, len(problems), say)
    with open(OUT_TXT, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nPer-attempt rows: {OUT_CSV}\nSummary: {OUT_TXT}")


if __name__ == "__main__":
    main()
