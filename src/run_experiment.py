"""
Main experiment: does multi-turn agentic revision (with verifier feedback)
improve solve rate over a single unrevised attempt, and how much of the
gain comes from turn 2 vs turn 3? Run across multiple seeds (matching the
noise-floor methodology used in transformer-from-scratch /
tokenization-ablation-study) so the effect isn't a single-seed artifact.

Outputs:
  results/trajectories.csv   -- every (problem, seed, turn) row
  results/summary.csv        -- solve rate by turn-budget x seed
"""
import csv

from agent_loop import run_all
from flawed_solver import attempt as flawed_attempt
from tasks import generate_problems

N_PER_FAMILY = 40
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 seeds, same order of magnitude as her other ablations


def main():
    problems = generate_problems(n_per_family=N_PER_FAMILY, seed=0)
    traj_rows = []
    summary_rows = []

    for seed in SEEDS:
        trajs = run_all(problems, flawed_attempt, seed=seed)
        for t in trajs:
            for row in t.turns:
                traj_rows.append(
                    {
                        "seed": seed,
                        "problem_id": t.problem_id,
                        "family": t.family,
                        "turn": row["turn"],
                        "candidate": row["candidate"],
                        "correct": row["correct"],
                        "reason": row["reason"],
                    }
                )
        for turn_budget in [1, 2, 3]:
            solved = sum(
                1 for t in trajs if t.solved and t.turns_used <= turn_budget
            )
            summary_rows.append(
                {
                    "seed": seed,
                    "turn_budget": turn_budget,
                    "n_problems": len(problems),
                    "n_solved": solved,
                    "solve_rate": solved / len(problems),
                }
            )

    with open("../results/trajectories.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()))
        w.writeheader()
        w.writerows(traj_rows)

    with open("../results/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"{len(problems)} problems x {len(SEEDS)} seeds written.")
    for turn_budget in [1, 2, 3]:
        rates = [r["solve_rate"] for r in summary_rows if r["turn_budget"] == turn_budget]
        mean = sum(rates) / len(rates)
        print(f"turn_budget={turn_budget}: mean solve rate = {mean:.3f} over {len(rates)} seeds")


if __name__ == "__main__":
    main()
