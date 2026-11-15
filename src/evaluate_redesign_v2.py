"""
Colab-side evaluation for the "redesign v2" pipeline (train_lora_sft.py
then train_lora_dpo_zeroshot.py): does prompt-matched SFT + DPO raise
zero-shot solve rate over the base model, and does DPO add anything on top
of SFT alone?

Reports THREE numbers, not two, on purpose:
  1. Base model (no fine-tuning) -- reproduces results/colab_run's 0.022
     baseline; deterministic (greedy decoding, fixed held-out set), so it
     should match exactly.
  2. SFT-only (../results/sft-merged-model) -- isolates what the SFT
     warm-start alone contributes.
  3. SFT+DPO (SFT-merged model + the zero-shot-format DPO adapter) -- the
     actual redesign-v2 result.
This ablation is here so that if the final number is still disappointing,
it's possible to say *which* stage did or didn't help, rather than only
having a single before/after pair the way the first LoRA-DPO run did.

Same held-out set as the first run: 45 problems, seed=99, n_per_family=15,
drawn with a different seed from every training draw in this repo (SFT and
DPO-zeroshot data both come from seed=0; see PROVENANCE.md for the note
that SFT's 100-per-family and DPO-zeroshot's 60-per-family draws overlap
each other -- both are training-side data, neither touches seed=99, so
that overlap has no effect on this evaluation's validity).

Prompt and decoding exactly match evaluate_finetuned.py (greedy,
do_sample=False, same system/user template, same regex answer extraction)
so all three numbers here are comparable to the first run's 0.022/0.000.

    pip install -q peft trl transformers accelerate bitsandbytes datasets sympy
    python evaluate_redesign_v2.py
"""
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import splits
from tasks import generate_problems
from verifier import verify

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_MERGED_DIR = "../results/sft-merged-model"  # train_lora_sft.py's output
DPO_ADAPTER_DIR = "../results/lora-dpo-zeroshot-adapter"  # train_lora_dpo_zeroshot.py's output
HELD_OUT_SEED = 99  # same held-out set as evaluate_finetuned.py. A different seed from
                    # training, but 6 of these 45 are in sft_pairs.jsonl: PROVENANCE.md bug #7

SYSTEM = "You are a careful algebra solver. Show brief work, then give the final answer line."

# Put the model on the GPU if there is one. Without this, from_pretrained()
# leaves the model on CPU even inside a GPU runtime, and every one of the
# 135 generations here (45 problems x 3 models x up to 150 new tokens) runs
# on Colab's 2 vCPUs -- tens of minutes, with no output while it grinds.
# dtype stays float32 deliberately rather than dropping to float16: float32
# on GPU is the same arithmetic as float32 on CPU, so this is purely a speed
# fix and the numbers stay comparable to the first run's 0.022 baseline,
# which was produced on CPU (see results/colab_run/training_log.md).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def hf_attempt(model, tok, equation_str):
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
    if m:
        return f"x = {m[-1]}"
    # An empty generation is NOT a claim of "no solution". Returning that
    # string handed the model a free correct answer on the 8 of 45 held-out
    # problems that have none; "" scores as unparsable, which is what an
    # unreadable answer is. See PROVENANCE.md, bug #6.
    return decoded.strip().splitlines()[-1] if decoded.strip() else ""


# Where the per-attempt rows go. Overridable so that the score of a checkpoint
# chosen by select_checkpoint.py can be written somewhere other than the
# directory holding the unselected training passes: a selected model is not
# another draw from the same distribution, and summarising the two together
# would average a chosen number into a sample of unchosen ones.
RAW_DIR = os.environ.get("AVS_EVAL_DIR", "../results/raw/eval")


def eval_rate(model, tok, problems, label, reason_counts=None, rows=None):
    """Solve rate over `problems`, printing progress.

    `reason_counts` collects the verifier's reason code for every attempt.
    Counting only correct answers, as this script originally did, cannot
    distinguish "the model reasoned wrongly" from "the model answered in a
    shape the extractor could not read" -- and that distinction decides what
    a score of 0/45 actually means. See inspect_base_outputs.py.

    The progress line exists because the first attempt at running this
    script looked like a hang: it printed "Evaluating BASE model..." and
    then nothing at all for 15+ minutes while it generated on CPU. A silent
    long-running loop is indistinguishable from a crashed one, so it now
    reports as it goes.
    """
    model.eval()
    correct = 0
    for i, p in enumerate(problems, 1):
        cand = hf_attempt(model, tok, p["equation_str"])
        res = verify(p["equation_str"], cand)
        if reason_counts is not None:
            reason_counts[res.reason] += 1
        if rows is not None:
            # Per-problem rows, because an aggregate reason table shows THAT
            # something changed and never WHICH problems changed. The first
            # regeneration found the DPO stage turning 1 `incomplete` into 14,
            # and that could not be diagnosed at all from counts alone.
            rows.append({
                "model": label,
                "problem_id": p["id"],
                "family": p["family"],
                "equation_str": p["equation_str"],
                "candidate": cand,
                "correct": res.correct,
                "reason": res.reason,
                "true_solutions": " | ".join(str(v) for v in res.true_solutions),
                "n_true": len(res.true_solutions),
            })
        if res.correct:
            correct += 1
        if i % 5 == 0 or i == len(problems):
            print(f"  [{label}] {i}/{len(problems)} done, {correct} correct so far", flush=True)
    return correct / len(problems)


def main():
    # The canonical benchmark, from splits.held_out(). Constructing it here
    # with a bare generate_problems() is how six held-out equations ended up
    # in the SFT training file; splits.py is the single definition and the
    # builders exclude it by construction. See PROVENANCE.md, bug 7.
    held_out_problems = splits.held_out()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    reasons = {k: Counter() for k in ("base", "sft", "sft+dpo")}
    rows = []
    print(f"Running on: {DEVICE}", flush=True)

    print("Evaluating BASE model zero-shot...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, dtype=torch.float32).to(DEVICE)
    base_rate = eval_rate(base_model, tok, held_out_problems, "base", reasons["base"], rows)
    print(f"Base model zero-shot solve rate: {base_rate:.3f}", flush=True)
    del base_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()  # free the base model's weights before loading the next one

    print("Evaluating SFT-only model zero-shot...", flush=True)
    sft_model = AutoModelForCausalLM.from_pretrained(SFT_MERGED_DIR, dtype=torch.float32).to(DEVICE)
    sft_rate = eval_rate(sft_model, tok, held_out_problems, "sft", reasons["sft"], rows)
    print(f"SFT-only zero-shot solve rate: {sft_rate:.3f}", flush=True)

    print("Evaluating SFT+DPO model zero-shot...", flush=True)
    # Wraps sft_model in place with the DPO LoRA layers; safe here only
    # because the SFT-only evaluation above has already finished.
    sft_dpo_model = PeftModel.from_pretrained(sft_model, DPO_ADAPTER_DIR).to(DEVICE)
    sft_dpo_rate = eval_rate(sft_dpo_model, tok, held_out_problems, "sft+dpo", reasons["sft+dpo"], rows)
    print(f"SFT+DPO zero-shot solve rate: {sft_dpo_rate:.3f}", flush=True)

    print(f"\nHeld-out set: {len(held_out_problems)} problems (seed={HELD_OUT_SEED})")
    print(f"Base:     {base_rate:.3f}")
    print(f"SFT-only: {sft_rate:.3f}  (delta vs base: {sft_rate - base_rate:+.3f})")
    print(f"SFT+DPO:  {sft_dpo_rate:.3f}  (delta vs SFT-only: {sft_dpo_rate - sft_rate:+.3f}, "
          f"delta vs base: {sft_dpo_rate - base_rate:+.3f})")

    # What the failures actually were. A score alone cannot distinguish a
    # model that reasons badly from one that answers in an unreadable shape.
    all_reasons = sorted(set().union(*[set(c) for c in reasons.values()]))
    print("\nFailure breakdown by verifier reason:")
    print(f"  {'reason':>18} | " + " | ".join(f"{k:>8}" for k in reasons))
    print("  " + "-" * (20 + 11 * len(reasons)))
    for r in all_reasons:
        print(f"  {r:>18} | " + " | ".join(f"{reasons[k][r]:>8}" for k in reasons))

    # Which problems each stage gained and lost. A stage that trades four
    # correct answers for four different ones is a different event from one
    # that simply loses four, and the rates are identical in both cases.
    solved = {m: {r["problem_id"] for r in rows if r["model"] == m and r["correct"]}
              for m in ("base", "sft", "sft+dpo")}
    print("\nWhat the DPO stage changed, problem by problem:")
    gained = sorted(solved["sft+dpo"] - solved["sft"])
    lost = sorted(solved["sft"] - solved["sft+dpo"])
    print(f"  gained ({len(gained)}): {gained}")
    print(f"  lost   ({len(lost)}): {lost}")
    by_id = {(r["model"], r["problem_id"]): r for r in rows}
    for pid in lost:
        before, after = by_id[("sft", pid)], by_id[("sft+dpo", pid)]
        print(f"    {pid:16} {before['equation_str']:28} "
              f"{before['candidate']!r} -> {after['candidate']!r} ({after['reason']})")

    os.makedirs(RAW_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = f"{RAW_DIR}/attempts_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"{RAW_DIR}/summary_{stamp}.json", "w") as f:
        json.dump({"utc": stamp, "n_problems": len(held_out_problems),
                   "held_out_seed": HELD_OUT_SEED, "decoding": "greedy",
                   "base": base_rate, "sft": sft_rate, "sft_dpo": sft_dpo_rate,
                   "reasons": {k: dict(v) for k, v in reasons.items()},
                   "dpo_gained": gained, "dpo_lost": lost}, f, indent=2)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {RAW_DIR}/summary_{stamp}.json")


if __name__ == "__main__":
    main()
