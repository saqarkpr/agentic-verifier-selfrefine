"""
Statistical analysis of run_experiment.py's output, in the same style as
transformer-from-scratch/tokenization-ablation-study: paired per-seed
comparison + significance test, not just a headline mean.
"""
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def load_summary(path="../results/summary.csv"):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["seed"] = int(r["seed"])
            r["turn_budget"] = int(r["turn_budget"])
            r["solve_rate"] = float(r["solve_rate"])
            rows.append(r)
    return rows


def load_trajectories(path="../results/trajectories.csv"):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["seed"] = int(r["seed"])
            r["turn"] = int(r["turn"])
            r["correct"] = r["correct"] == "True"
            rows.append(r)
    return rows


def main():
    summary = load_summary()
    by_budget = defaultdict(dict)
    for r in summary:
        by_budget[r["turn_budget"]][r["seed"]] = r["solve_rate"]

    seeds = sorted(by_budget[1].keys())
    r1 = [by_budget[1][s] for s in seeds]
    r2 = [by_budget[2][s] for s in seeds]
    r3 = [by_budget[3][s] for s in seeds]

    t_12, p_12 = stats.ttest_rel(r2, r1)
    t_13, p_13 = stats.ttest_rel(r3, r1)
    t_23, p_23 = stats.ttest_rel(r3, r2)

    lines = []
    lines.append(f"n_seeds = {len(seeds)}")
    lines.append(f"turn_budget=1: mean={sum(r1)/len(r1):.3f}")
    lines.append(f"turn_budget=2: mean={sum(r2)/len(r2):.3f}")
    lines.append(f"turn_budget=3: mean={sum(r3)/len(r3):.3f}")
    lines.append(f"paired t-test 1->2: t={t_12:.3f}, p={p_12:.5f}")
    lines.append(f"paired t-test 1->3: t={t_13:.3f}, p={p_13:.5f}")
    lines.append(f"paired t-test 2->3: t={t_23:.3f}, p={p_23:.5f} (marginal-gain check: is turn 3 still worth it?)")

    # error-reason breakdown at turn 1 (what kinds of mistakes does the un-revised attempt make?)
    traj = load_trajectories()
    reason_counts = defaultdict(int)
    turn1_wrong = [r for r in traj if r["turn"] == 1 and not r["correct"]]
    for r in turn1_wrong:
        reason_counts[r["reason"]] += 1
    lines.append("\nTurn-1 error breakdown (n=%d wrong):" % len(turn1_wrong))
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {reason}: {count} ({count/len(turn1_wrong)*100:.1f}%)")

    with open("../results/analysis.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # figure: solve rate by turn budget across seeds (box + individual seed lines)
    fig, ax = plt.subplots(figsize=(5, 4))
    budgets = [1, 2, 3]
    data = [r1, r2, r3]
    ax.boxplot(data, positions=budgets, widths=0.5)
    for s_idx in range(len(seeds)):
        ax.plot(budgets, [r1[s_idx], r2[s_idx], r3[s_idx]], color="gray", alpha=0.4, linewidth=0.8)
    ax.set_xticks(budgets)
    ax.set_xlabel("Agent turn budget")
    ax.set_ylabel("Solve rate")
    # Read the problem count off the data rather than hardcoding "120", so the
    # figure can't silently disagree with the run it was generated from.
    n_problems = int(summary[0]["n_problems"])
    ax.set_title("Solve rate vs. agentic revision budget\n(%d problems x %d seeds)" % (n_problems, len(seeds)))
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig("../results/solve_rate_by_turn_budget.png", dpi=150)

    # figure: error-reason breakdown
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    reasons = sorted(reason_counts.items(), key=lambda kv: -kv[1])
    ax2.barh([r[0] for r in reasons][::-1], [r[1] for r in reasons][::-1], color="#4C72B0")
    ax2.set_xlabel("Count (turn-1 wrong attempts)")
    ax2.set_title("Error-mode breakdown, unrevised attempt")
    fig2.tight_layout()
    fig2.savefig("../results/error_breakdown.png", dpi=150)


if __name__ == "__main__":
    main()
