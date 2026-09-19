"""
Turn agentic trajectories into DPO preference pairs.

For every trajectory that was wrong on turn 1 but solved by turn 2 or 3:
  prompt   = the problem statement (+ any prior wrong attempt + verifier
             feedback, for turn>=2 pairs)
  chosen   = the correct revision
  rejected = the wrong attempt that preceded it at the same turn

This is the same idea as the RLVF project's verifier-derived DPO pairs, but
the pairs come from *agentic self-revision* trajectories rather than
single-shot sampling: the model is being taught to do, directly, what the
multi-turn loop currently has to do at inference time across 2-3 calls.
Output format matches what trl's DPOTrainer expects (prompt/chosen/rejected).
"""
import json

from agent_loop import run_all
from flawed_solver import attempt as flawed_attempt
from tasks import generate_problems

PROBLEM_TEMPLATE = "Solve for x: {equation}\nGive the answer as 'x = v1, v2' or 'no solution'."
REVISION_TEMPLATE = (
    "Solve for x: {equation}\n"
    "Previous attempt: {prev_candidate}\n"
    "Verifier feedback: {feedback}\n"
    "Give a corrected answer as 'x = v1, v2' or 'no solution'."
)


def build_pairs(problems, solve_fn, seed):
    trajs = run_all(problems, solve_fn, seed=seed)
    pairs = []
    for t in trajs:
        if not t.solved or t.turns_used == 1:
            continue  # either never solved, or correct on the first try (no revision to teach)
        for i in range(1, len(t.turns)):
            prev_turn = t.turns[i - 1]
            this_turn = t.turns[i]
            if prev_turn["correct"] or not this_turn["correct"]:
                continue  # only take the exact wrong->correct transition
            prompt = REVISION_TEMPLATE.format(
                equation=t.equation_str,
                prev_candidate=prev_turn["candidate"],
                feedback=prev_turn["feedback"],
            )
            pairs.append(
                {
                    "problem_id": t.problem_id,
                    "prompt": prompt,
                    "chosen": this_turn["candidate"],
                    "rejected": prev_turn["candidate"],
                }
            )
    return pairs


def main():
    problems = generate_problems(n_per_family=40, seed=0)
    all_pairs = []
    for seed in range(8):
        all_pairs.extend(build_pairs(problems, flawed_attempt, seed))

    with open("../data/dpo_pairs.jsonl", "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"{len(all_pairs)} preference pairs written to data/dpo_pairs.jsonl")


if __name__ == "__main__":
    main()
