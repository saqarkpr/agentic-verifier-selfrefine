"""
Build DPO preference pairs in the SAME zero-shot prompt format as the
evaluation (unlike the original data/dpo_pairs.jsonl, whose prompts used
the multi-turn "revision" template -- equation + prior wrong attempt +
verifier feedback -- which never matched the zero-shot eval prompt at all;
see results/colab_run/training_log.md for the mismatch this caused).

chosen = the correct worked solution (reference_solution.worked_solution).
rejected = the SAME step-by-step template text, but with a wrong final
answer line swapped in (from flawed_solver, severity=3: a first-attempt-
style error). Using matching steps text for chosen/rejected is deliberate:
it keeps the two responses close in length and style, so DPO has a better
chance of learning "this final answer is right/wrong" rather than
picking up a spurious "prefer the longer/more-detailed response" signal.

Output: data/dpo_pairs_zeroshot.jsonl, one {"problem_id", "family",
"equation_str", "system", "user", "chosen", "rejected"} row per problem.
"prompt" is intentionally NOT pre-assembled here: the Colab training script
builds it via tok.apply_chat_template([system, user], ...,
add_generation_prompt=True) -- the exact same call evaluate_finetuned.py
uses -- so train- and eval-time prompt formatting are guaranteed identical
byte-for-byte, not just "the same words".
"""
import json
import random

from flawed_solver import attempt as flawed_attempt
from reference_solution import worked_solution, worked_solution_with_answer
from tasks import generate_problems

SYSTEM = "You are a careful algebra solver. Show brief work, then give the final answer line."
N_PER_FAMILY = 60
SEED = 0  # same problems as data/dpo_pairs.jsonl's build, disjoint from held-out seed=99


def user_prompt(equation_str):
    return (
        f"Solve for x: {equation_str}\n"
        "Give the final answer on its own line as 'x = v1, v2' or 'x = no solution'."
    )


def build_rows():
    problems = generate_problems(n_per_family=N_PER_FAMILY, seed=SEED)
    rng = random.Random(SEED)
    rows = []
    for p in problems:
        chosen = worked_solution(p["family"], p["equation_str"])
        wrong_final_line = flawed_attempt(p["equation_str"], rng, severity=3)
        rejected = worked_solution_with_answer(p["family"], p["equation_str"], wrong_final_line)
        if chosen == rejected:
            continue  # flawed_solver got lucky and produced the correct answer; not a valid pair
        rows.append(
            {
                "problem_id": p["id"],
                "family": p["family"],
                "equation_str": p["equation_str"],
                "system": SYSTEM,
                "user": user_prompt(p["equation_str"]),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return rows


def main():
    rows = build_rows()
    with open("../data/dpo_pairs_zeroshot.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} zero-shot-format DPO pairs written to data/dpo_pairs_zeroshot.jsonl")
    print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()
