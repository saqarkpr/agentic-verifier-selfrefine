"""
Build a supervised fine-tuning (SFT) dataset in the EXACT prompt/response
format used by the zero-shot evaluation (evaluate_finetuned.py), so there is
no prompt-format mismatch between what the model is trained on and what it
is tested on (see PROVENANCE.md: "redesign v2" for why this exists).

Rationale for adding an SFT stage at all (not just fixing the DPO pairs):
the base model's zero-shot solve rate was measured at 2.2% (see
results/colab_run/training_log.md). DPO only teaches a *preference* between
two candidate responses; it has very little signal to work with if the
model essentially cannot produce a correct response in the first place. SFT
on correct worked solutions teaches the capability directly, the same way
the SFT stage in a standard SFT-then-DPO pipeline does. See
train_lora_sft.py for how this is consumed.

Output: data/sft_pairs.jsonl, one {"problem_id", "family", "equation_str",
"messages"} row per problem, where "messages" is the same
[system, user, assistant] structure evaluate_finetuned.py's hf_attempt()
builds for [system, user] at eval time, so the assistant turn is a direct,
matched-format training target for that exact prompt.
"""
import json

import splits
from reference_solution import worked_solution
from tasks import contamination

SYSTEM = "You are a careful algebra solver. Show brief work, then give the final answer line."
N_PER_FAMILY = 100
SEED = splits.TRAIN_SEED
# The draw comes from splits.training(), which EXCLUDES every held-out equation
# and asserts the result. Before that, this file used a bare
# generate_problems(seed=0) and 6 of the 45 held-out equations landed in it with
# their correct answers, while the write-up claimed no leakage. A different seed
# is not a disjoint sample. See PROVENANCE.md, bug 7.


def user_prompt(equation_str):
    return (
        f"Solve for x: {equation_str}\n"
        "Give the final answer on its own line as 'x = v1, v2' or 'x = no solution'."
    )


def build_rows():
    problems = splits.training(n_per_family=N_PER_FAMILY, seed=SEED)
    rows = []
    for p in problems:
        assistant = worked_solution(p["family"], p["equation_str"])
        rows.append(
            {
                "problem_id": p["id"],
                "family": p["family"],
                "equation_str": p["equation_str"],
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_prompt(p["equation_str"])},
                    {"role": "assistant", "content": assistant},
                ],
            }
        )
    return rows


def main():
    rows = build_rows()
    with open("../data/sft_pairs.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(splits.report(N_PER_FAMILY, SEED))
    print(f"{len(rows)} SFT examples written to data/sft_pairs.jsonl")
    # Stated in the build output, not just asserted, so a training log records
    # the split it actually used.
    check = contamination(splits.training(N_PER_FAMILY, SEED), splits.held_out())
    print(f"held-out contamination: {check['n_overlapping']}/{check['n_held_out']}")
    print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()
