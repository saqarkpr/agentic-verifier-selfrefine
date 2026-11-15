"""Paper-grade re-analysis of every recorded GPU run. CPU only, ~20 s.

WHY THIS SCRIPT EXISTS

The per-run scripts (summarize_eval.py, feedback_protocol_ablation.py,
run_llm_agent_experiment.py) each print the numbers of ONE run. A paper needs
three things none of them provide:

1. The right unit of analysis. The September ladder was reported with a sign
   test over 135 "problem-seed units". The three seeds of one problem share
   the problem's difficulty, so they are not independent; treating them as
   such is pseudo-replication and understates p. Here the unit is the
   PROBLEM: each problem's three seeds are averaged first, and inference is a
   sign-flip permutation test and a cluster bootstrap over the 45 problems.

2. Multiplicity. Four feedback conditions share one control. Holm-Bonferroni
   is applied within each run, and to the pooled analysis.

3. Replication. The ladder and the agent loop were run on three different
   checkpoints (Sep pass 4, Nov pass 4, Nov selected). The question a reader
   will ask -- does the September positive result hold up? -- is only
   answerable by putting them side by side, which this does.

Inputs: results/recorded/2026-09-22/ and results/recorded/2026-11-14/.
Outputs: results/paper/analysis.json, results/paper/tables.tex,
         paper/figures/*.pdf (+ .png).

    python paper_analysis.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REC = os.path.join(ROOT, "results", "recorded")
OUT = os.path.join(ROOT, "results", "paper")
FIG = os.path.join(ROOT, "paper", "figures")

CONDITIONS = ["retry", "generic", "diagnostic", "actionable", "constrained"]
FEEDBACK_CONDITIONS = CONDITIONS[1:]
N_PERM = 20000
N_BOOT = 10000
RNG_SEED = 12345

# Every checkpoint on which an inference-time experiment was run. `greedy` is
# that checkpoint's own held-out greedy solve rate, measured on the same 45
# problems -- the x-axis of the "does feedback only help broken models?" plot.
CHECKPOINTS = {
    "sep_pass4": {
        "label": "Sep, pass 4",
        "greedy": 0.133,
        "ladder_csv": "2026-09-22/feedback_protocol_ablation.csv",
        "agent_csv": "2026-09-22/llm_agent_experiment_minimal.csv",
    },
    "nov_pass4": {
        "label": "Nov, pass 4",
        "greedy": 0.244,
        "ladder_csv": "2026-11-14/pass4_checkpoint/feedback_protocol_ablation.csv",
        "agent_csv": "2026-11-14/pass4_checkpoint/llm_agent_experiment_minimal.csv",
    },
    "nov_selected": {
        "label": "Nov, selected",
        "greedy": 0.444,
        # Only the printed summary survives: the per-attempt CSV is inside a
        # 1.8 GB backup zip that also holds the model weights. Aggregates
        # below are transcribed from the executed notebook, and every
        # problem-level statistic for this checkpoint is reported as missing
        # rather than approximated.
        "ladder_summary": {
            "solve3": {"retry": 0.437, "generic": 0.437, "diagnostic": 0.415,
                       "actionable": 0.444, "constrained": 0.393},
            "turn1": 0.363,
            "format_ok": {"retry": 0.956, "generic": 0.880, "diagnostic": 0.799,
                          "actionable": 0.725, "constrained": 0.635},
        },
    },
}


# ----------------------------------------------------------------------------
# generic statistics
# ----------------------------------------------------------------------------

def sign_flip_p(diffs, n_perm=N_PERM, rng=None):
    """Two-sided paired permutation test on per-unit differences.

    Exact when there are few non-zero units, Monte-Carlo otherwise.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    obs = abs(d.sum())
    if len(d) <= 16:
        signs = np.array(np.meshgrid(*[[-1, 1]] * len(d))).T.reshape(-1, len(d))
        null = np.abs((signs * d).sum(axis=1))
        return float((null >= obs - 1e-12).mean())
    rng = rng or np.random.default_rng(RNG_SEED)
    signs = rng.choice([-1, 1], size=(n_perm, len(d)))
    null = np.abs((signs * d).sum(axis=1))
    return float(((null >= obs - 1e-12).sum() + 1) / (n_perm + 1))


def bootstrap_ci(diffs, n_boot=N_BOOT, rng=None, alpha=0.05):
    d = np.asarray(diffs, dtype=float)
    rng = rng or np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, same order as the input."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def binom_sign_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    return float(stats.binomtest(wins, n, 0.5).pvalue)


# ----------------------------------------------------------------------------
# multi-turn runs (ladder and agent loop share a format)
# ----------------------------------------------------------------------------

def solved_by_budget(df, budget):
    """(condition, seed, problem_id) -> solved within `budget` turns."""
    d = df[df["turn"] <= budget]
    g = d.groupby(["condition", "seed", "problem_id"])["correct"].apply(
        lambda s: bool(pd.Series(s).astype(str).str.lower().eq("true").any()))
    return g


def load_multiturn(path):
    df = pd.read_csv(os.path.join(REC, path))
    df["correct"] = df["correct"].astype(str).str.lower().eq("true")
    if "parsable" in df:
        df["parsable"] = df["parsable"].astype(str).str.lower().eq("true")
    return df


def paired_contrast(df, cond, control, budget=3, rng=None):
    """Feedback condition vs control at a turn budget, problem as the unit."""
    s = solved_by_budget(df, budget).unstack(level=0)  # index (seed, problem)
    s = s[[cond, control]].astype(float)
    unit = s[cond] - s[control]                          # per (seed, problem)
    per_problem = unit.groupby(level="problem_id").mean()  # average the seeds
    wins = int((unit > 0).sum())
    losses = int((unit < 0).sum())
    return {
        "rate_condition": float(s[cond].mean()),
        "rate_control": float(s[control].mean()),
        "diff": float(unit.mean()),
        "ci95_problem_bootstrap": bootstrap_ci(per_problem.values, rng=rng),
        "p_problem_permutation": sign_flip_p(per_problem.values, rng=rng),
        "problems_better": int((per_problem > 0).sum()),
        "problems_worse": int((per_problem < 0).sum()),
        "naive_unit_sign_test": {"wins": wins, "losses": losses,
                                 "p": binom_sign_p(wins, losses),
                                 "note": "135 problem-seed units treated as independent "
                                         "(pseudo-replication); kept for comparison with "
                                         "the September write-up only"},
        "per_problem_diffs": per_problem.round(4).to_dict(),
    }


def ladder_summary(df):
    out = {"solve_by_budget": {}, "format_ok_by_turn": {}, "format_ok_overall": {}}
    for b in (1, 2, 3):
        s = solved_by_budget(df, b)
        out["solve_by_budget"][b] = {c: float(s[c].mean()) for c in CONDITIONS}
    for c in CONDITIONS:
        d = df[df.condition == c]
        out["format_ok_by_turn"][c] = [float(d[d.turn == t]["parsable"].mean()) for t in (1, 2, 3)]
        out["format_ok_overall"][c] = float(d["parsable"].mean())
    return out


def accuracy_given_parsable(df):
    """Turn 2-3 accuracy among parsable answers -- separates format from reasoning."""
    d = df[(df.turn >= 2) & df.parsable]
    return {c: {"acc": float(d[d.condition == c]["correct"].mean()),
                "n": int((d.condition == c).sum())} for c in CONDITIONS}


def analyse_ladders(rng):
    runs = {}
    pooled_rows = []
    for key, ck in CHECKPOINTS.items():
        if "ladder_csv" not in ck:
            runs[key] = {"checkpoint": ck["label"], "greedy": ck["greedy"],
                         "raw_available": False, **ck["ladder_summary"]}
            continue
        df = load_multiturn(ck["ladder_csv"])
        r = {"checkpoint": ck["label"], "greedy": ck["greedy"], "raw_available": True}
        r.update(ladder_summary(df))
        r["acc_given_parsable_turn2plus"] = accuracy_given_parsable(df)
        contrasts = {c: paired_contrast(df, c, "retry", rng=rng) for c in FEEDBACK_CONDITIONS}
        adj = holm([contrasts[c]["p_problem_permutation"] for c in FEEDBACK_CONDITIONS])
        for c, a in zip(FEEDBACK_CONDITIONS, adj):
            contrasts[c]["p_holm"] = a
        r["vs_retry_budget3"] = contrasts
        # turn-1 identity check (pairing)
        t1 = df[df.turn == 1].pivot_table(index=["seed", "problem_id"], columns="condition",
                                          values="candidate", aggfunc="first")
        r["turn1_identical_across_conditions"] = bool((t1.nunique(axis=1) == 1).all())
        runs[key] = r
        for c in FEEDBACK_CONDITIONS:
            for pid, v in contrasts[c]["per_problem_diffs"].items():
                pooled_rows.append({"run": key, "condition": c, "problem_id": pid, "diff": v})

    # Pooled over the runs with raw data: unit = (run, problem). The same 45
    # problems appear in both runs, but on different checkpoints, so the
    # stratum is the run; we average within problem across runs first (the
    # conservative choice: 45 units, not 90).
    pooled = {}
    P = pd.DataFrame(pooled_rows)
    for c in FEEDBACK_CONDITIONS:
        sub = P[P.condition == c]
        per_problem = sub.groupby("problem_id")["diff"].mean()
        pooled[c] = {"diff": float(sub["diff"].mean()),
                     "ci95_problem_bootstrap": bootstrap_ci(per_problem.values, rng=rng),
                     "p_problem_permutation": sign_flip_p(per_problem.values, rng=rng),
                     "runs": sorted(sub.run.unique().tolist())}
    adj = holm([pooled[c]["p_problem_permutation"] for c in FEEDBACK_CONDITIONS])
    for c, a in zip(FEEDBACK_CONDITIONS, adj):
        pooled[c]["p_holm"] = a
    return runs, pooled


def analyse_agents(rng):
    out = {}
    for key, ck in CHECKPOINTS.items():
        if "agent_csv" not in ck:
            out[key] = {"checkpoint": ck["label"], "raw_available": False,
                        "note": "The agent run on the selected checkpoint (notebook section 17) "
                                "left no output: the backup taken after it contains no CSV."}
            continue
        df = load_multiturn(ck["agent_csv"])
        r = {"checkpoint": ck["label"], "greedy": ck["greedy"], "raw_available": True,
             "by_budget": {}}
        for b in (1, 2, 3):
            s = solved_by_budget(df, b)
            r["by_budget"][b] = {"feedback": float(s["feedback"].mean()),
                                 "control": float(s["control"].mean())}
        for b in (2, 3):
            c = paired_contrast(df, "feedback", "control", budget=b, rng=rng)
            c.pop("per_problem_diffs")
            r[f"contrast_budget{b}"] = c
        out[key] = r
    return out


# ----------------------------------------------------------------------------
# training: SFT and DPO across every clean pass
# ----------------------------------------------------------------------------

def _read_summaries(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(REC, pattern)))]


def _read_attempts(pattern):
    frames = []
    for i, f in enumerate(sorted(glob.glob(os.path.join(REC, pattern)))):
        d = pd.read_csv(f)
        d["pass_file"] = os.path.basename(f)
        frames.append(d)
    return frames


def analyse_training(rng):
    batches = {
        "sep": ("2026-09-22/eval/summary_*.json", "2026-09-22/eval/attempts_*.csv"),
        "nov": ("2026-11-14/eval/summary_*.json", "2026-11-14/eval/attempts_*.csv"),
    }
    passes = []
    family_rows = []
    for batch, (spat, apat) in batches.items():
        for s, a in zip(_read_summaries(spat), _read_attempts(apat)):
            a["correct"] = a["correct"].astype(str).str.lower().eq("true")
            passes.append({"batch": batch, "utc": s["utc"], "base": s["base"],
                           "sft": s["sft"], "sft_dpo": s["sft_dpo"],
                           "delta": s["sft_dpo"] - s["sft"]})
            piv = a.pivot_table(index=["problem_id", "family"], columns="model",
                                values="correct", aggfunc="first").reset_index()
            for fam, g in piv.groupby("family"):
                family_rows.append({"batch": batch, "utc": s["utc"], "family": fam,
                                    "sft": float(g["sft"].mean()),
                                    "sft_dpo": float(g["sft+dpo"].mean())})
    P = pd.DataFrame(passes)
    deltas = P["delta"].values
    t = stats.ttest_1samp(deltas, 0.0)
    w = stats.wilcoxon(deltas) if np.any(deltas != 0) else None
    F = pd.DataFrame(family_rows)
    fam = {}
    for f, g in F.groupby("family"):
        d = (g["sft_dpo"] - g["sft"]).values
        fam[f] = {"sft_mean": float(g["sft"].mean()), "sft_dpo_mean": float(g["sft_dpo"].mean()),
                  "delta_mean": float(d.mean()), "delta_by_pass": d.round(3).tolist(),
                  "passes_negative": int((d < 0).sum()), "passes_positive": int((d > 0).sum()),
                  "p_sign_flip": sign_flip_p(d, rng=rng)}

    sel = _read_summaries("2026-11-14/selected_eval/summary_*.json")[0]
    # Same breakdown for the selected checkpoint, kept apart from the eight passes.
    sa = _read_attempts("2026-11-14/selected_eval/attempts_*.csv")[0]
    sa["correct"] = sa["correct"].astype(str).str.lower().eq("true")
    spiv = sa.pivot_table(index=["problem_id", "family"], columns="model",
                          values="correct", aggfunc="first").reset_index()
    sel_family = {f: {"sft": float(g["sft"].mean()), "sft_dpo": float(g["sft+dpo"].mean())}
                  for f, g in spiv.groupby("family")}
    return {
        "passes": passes,
        "n_passes": len(passes),
        "sft": {"mean": float(P.sft.mean()), "sd": float(P.sft.std(ddof=1)),
                "min": float(P.sft.min()), "max": float(P.sft.max())},
        "sft_dpo": {"mean": float(P.sft_dpo.mean()), "sd": float(P.sft_dpo.std(ddof=1)),
                    "min": float(P.sft_dpo.min()), "max": float(P.sft_dpo.max())},
        "dpo_delta": {"mean": float(deltas.mean()), "sd": float(deltas.std(ddof=1)),
                      "t": float(t.statistic), "p_t": float(t.pvalue),
                      "p_wilcoxon": float(w.pvalue) if w else 1.0,
                      "positive": int((deltas > 0).sum()), "negative": int((deltas < 0).sum()),
                      "zero": int((deltas == 0).sum()),
                      "ci95_bootstrap": bootstrap_ci(deltas, rng=rng)},
        "dpo_delta_by_family": fam,
        "selected_checkpoint": {"sft": sel["sft"], "sft_dpo": sel["sft_dpo"],
                                "delta": sel["sft_dpo"] - sel["sft"],
                                "by_family": sel_family,
                                "note": "chosen on a validation set, scored once on the "
                                        "benchmark; NOT a ninth draw and never averaged in"},
        "base_reasons": sel["reasons"]["base"],
    }


# ----------------------------------------------------------------------------
# where DPO's probability mass went: the `no solution` displacement
# ----------------------------------------------------------------------------

def _says_no_solution(candidate):
    return "no solution" in str(candidate).lower()


def analyse_displacement():
    """Does DPO move answers onto `no solution`, an answer absent from its data?

    Every attempt is classified by whether the problem has a real solution
    (n_true > 0) and whether the model answered `no solution`. If DPO were
    simply sharpening correct answers, `no solution` on SOLVABLE problems
    would not rise; if it is displacing probability away from the numeric
    answers it was trained against, it will.
    """
    frames = []
    for batch, pat in [("sep", "2026-09-22/eval/attempts_*.csv"),
                       ("nov", "2026-11-14/eval/attempts_*.csv"),
                       ("selected", "2026-11-14/selected_eval/attempts_*.csv")]:
        for d in _read_attempts(pat):
            d["batch"] = batch
            frames.append(d)
    A = pd.concat(frames, ignore_index=True)
    A["correct"] = A["correct"].astype(str).str.lower().eq("true")
    A["says_nosol"] = A["candidate"].map(_says_no_solution)
    A["solvable"] = A["n_true"] > 0
    A["eight_passes"] = A["batch"].isin(["sep", "nov"])

    out = {"by_model_family_solvable": [], "n_problems": {}}
    E = A[A.eight_passes]
    for (model, fam, solv), g in E.groupby(["model", "family", "solvable"]):
        out["by_model_family_solvable"].append({
            "model": model, "family": fam, "solvable": bool(solv),
            "n_attempts": int(len(g)), "accuracy": float(g["correct"].mean()),
            "says_no_solution": float(g["says_nosol"].mean())})
    held = A[(A.batch == "nov")].drop_duplicates("problem_id")
    out["n_problems"] = {f"{f}_{'solvable' if s else 'no_real_root'}": int(len(g))
                         for (f, s), g in held.groupby(["family", "solvable"])}

    def rate(df, model, solvable=True):
        g = df[(df.model == model) & (df.solvable == solvable)]
        return float(g["says_nosol"].mean())

    out["no_solution_on_solvable"] = {m: rate(E, m) for m in ["base", "sft", "sft+dpo"]}
    out["no_solution_on_solvable_selected"] = {m: rate(A[A.batch == "selected"], m)
                                               for m in ["sft", "sft+dpo"]}
    # How much of DPO's quadratic "gain" is the no-real-root problems?
    q = E[E.family == "quadratic"]
    out["quadratic_accuracy"] = {
        m: {"solvable": float(q[(q.model == m) & q.solvable]["correct"].mean()),
            "no_real_root": float(q[(q.model == m) & ~q.solvable]["correct"].mean())}
        for m in ["sft", "sft+dpo"]}
    # Per pass: does the size of the displacement track the size of the damage?
    per_pass = []
    for (batch, f), g in E.groupby(["batch", "pass_file"]):
        s = g[g.model == "sft"]; d = g[g.model == "sft+dpo"]
        per_pass.append({"batch": batch, "file": f,
                         "delta": float(d["correct"].mean() - s["correct"].mean()),
                         "linear_nosol": float(d[(d.family == "linear")]["says_nosol"].mean()),
                         "solvable_nosol": float(d[d.solvable]["says_nosol"].mean())})
    out["per_pass"] = per_pass
    r = stats.spearmanr([p["solvable_nosol"] for p in per_pass], [p["delta"] for p in per_pass])
    out["per_pass_spearman_nosol_vs_delta"] = {"rho": float(r.statistic), "p": float(r.pvalue)}

    tm = json.load(open(os.path.join(REC, "2026-11-14", "dpo_trainer_metrics.json")))["runs"]
    out["trainer_metrics"] = {k: {"rewards_chosen": v["eval_rewards/chosen"],
                                  "rewards_rejected": v["eval_rewards/rejected"],
                                  "reward_accuracy": v["eval_rewards/accuracies"],
                                  "margin": v["eval_rewards/margins"]}
                              for k, v in tm.items() if k != "smoke_test"}
    return out


# ----------------------------------------------------------------------------
# figures and tables
# ----------------------------------------------------------------------------

def make_figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"generic": "#7f7f7f", "diagnostic": "#1f77b4", "actionable": "#2ca02c",
              "constrained": "#d62728", "retry": "#000000"}

    # Fig 1: feedback benefit vs checkpoint quality, all three checkpoints.
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    L = res["ladder"]
    for c in FEEDBACK_CONDITIONS:
        xs, ys, lo, hi = [], [], [], []
        for key in CHECKPOINTS:
            r = L[key]
            xs.append(r["greedy"])
            if r["raw_available"]:
                v = r["vs_retry_budget3"][c]
                ys.append(v["diff"]); lo.append(v["ci95_problem_bootstrap"][0]); hi.append(v["ci95_problem_bootstrap"][1])
            else:
                ys.append(r["solve3"][c] - r["solve3"]["retry"]); lo.append(np.nan); hi.append(np.nan)
        off = {"generic": -0.012, "diagnostic": -0.004, "actionable": 0.004, "constrained": 0.012}[c]
        xs = np.array(xs) + off
        ax.plot(xs, ys, "-o", color=colors[c], label=c, ms=3.5, lw=1)
        ys, lo, hi = np.array(ys), np.array(lo), np.array(hi)
        m = ~np.isnan(lo)
        ax.errorbar(xs[m], ys[m], yerr=[ys[m] - lo[m], hi[m] - ys[m]], fmt="none",
                    ecolor=colors[c], elinewidth=0.7, capsize=1.5, alpha=0.7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("checkpoint greedy solve rate (held-out)")
    ax.set_ylabel("solve@3 minus retry@3")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_benefit_vs_quality.{ext}"), dpi=200)
    plt.close(fig)

    # Fig 2: format validity at turns 2-3 by condition, three checkpoints.
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    width = 0.26
    keys = list(CHECKPOINTS)
    for i, key in enumerate(keys):
        r = L[key]
        if r["raw_available"]:
            vals = [np.mean(r["format_ok_by_turn"][c][1:]) for c in CONDITIONS]
        else:
            vals = [r["format_ok"][c] for c in CONDITIONS]  # all turns: not comparable
        ax.bar(np.arange(5) + (i - 1) * width, vals, width,
               label=CHECKPOINTS[key]["label"] + ("" if r["raw_available"] else " (all turns)"),
               color=["#9ecae1", "#4292c6", "#08519c"][i])
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(CONDITIONS, rotation=20)
    ax.set_ylabel("parsable fraction")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_format_validity.{ext}"), dpi=200)
    plt.close(fig)

    # Fig 3: DPO delta across 8 passes + selected, overall and by family.
    T = res["training"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    d = [p["delta"] for p in T["passes"]]
    cols = ["#9ecae1" if p["batch"] == "sep" else "#08519c" for p in T["passes"]]
    ax.bar(range(len(d)), d, color=cols)
    ax.bar(len(d) + 0.6, T["selected_checkpoint"]["delta"], color="#fdae6b", hatch="//")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(T["dpo_delta"]["mean"], color="#d62728", lw=0.8, ls="--")
    ax.set_xticks(list(range(len(d))) + [len(d) + 0.6])
    ax.set_xticklabels([f"S{i+1}" for i in range(4)] + [f"N{i+1}" for i in range(4)] + ["sel."], fontsize=7)
    ax.set_ylabel("SFT+DPO minus SFT")
    ax.set_title("DPO effect per training pass", fontsize=8)
    ax = axes[1]
    fams = ["linear", "quadratic", "rational"]
    for i, f in enumerate(fams):
        vals = T["dpo_delta_by_family"][f]["delta_by_pass"]
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.15, 0.15, len(vals)), vals, s=10, color="#4292c6")
        ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color="#d62728")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(fams)
    ax.set_title("by equation family (15 problems each)", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_dpo_passes.{ext}"), dpi=200)
    plt.close(fig)

    # Fig 5: the displacement. Outcome composition, SFT vs SFT+DPO, eight passes.
    D = {(r["model"], r["family"], r["solvable"]): r for r in res["displacement"]["by_model_family_solvable"]}
    groups = [("linear", True, "linear"), ("quadratic", True, "quadratic\n(real roots)"),
              ("quadratic", False, "quadratic\n(no real root)"), ("rational", True, "rational")]
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    xs = []
    for gi, (fam, solv, lab) in enumerate(groups):
        for mi, model in enumerate(["sft", "sft+dpo"]):
            r = D[(model, fam, solv)]
            x = gi * 2.6 + mi
            xs.append((x, f"{'SFT' if model == 'sft' else '+DPO'}"))
            acc = r["accuracy"]
            nos = r["says_no_solution"] - (acc if not solv else 0.0)  # wrong 'no solution'
            nos = max(nos, 0.0)
            other = 1 - acc - nos
            ax.bar(x, acc, color="#2ca02c", width=0.85, label="correct" if gi == mi == 0 else None)
            ax.bar(x, nos, bottom=acc, color="#d62728", width=0.85,
                   label="wrong: `no solution'" if gi == mi == 0 else None)
            ax.bar(x, other, bottom=acc + nos, color="#c7c7c7", width=0.85,
                   label="other wrong / unparsable" if gi == mi == 0 else None)
        ax.text(gi * 2.6 + 0.5, -0.30, lab, ha="center", va="top", fontsize=7)
    ax.set_xticks([x for x, _ in xs]); ax.set_xticklabels([l for _, l in xs], fontsize=6.5)
    ax.set_ylabel("fraction of attempts"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3)
    fig.subplots_adjust(bottom=0.3, top=0.85, left=0.12, right=0.98)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_displacement.{ext}"), dpi=200)
    plt.close(fig)

    # Fig 4: agent loop feedback vs control by budget, both runs.
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2), sharey=True)
    for ax, key in zip(axes, ["sep_pass4", "nov_pass4"]):
        r = res["agent"][key]
        b = [1, 2, 3]
        ax.plot(b, [r["by_budget"][k]["feedback"] for k in b], "-o", label="verifier feedback", ms=3.5)
        ax.plot(b, [r["by_budget"][k]["control"] for k in b], "-s", label="re-sample (control)", ms=3.5, color="k")
        ax.set_title(CHECKPOINTS[key]["label"] + f" (greedy {CHECKPOINTS[key]['greedy']:.3f})", fontsize=8)
        ax.set_xticks(b); ax.set_xlabel("turn budget")
    axes[0].set_ylabel("solve rate"); axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_agent_loop.{ext}"), dpi=200)
    plt.close(fig)


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def make_tables(res):
    lines = ["% Auto-generated by src/paper_analysis.py -- do not edit by hand.", ""]
    # Table: ladder
    lines += [r"\begin{table}[t]\centering\small",
              r"\caption{Feedback ladder: solve rate within three turns, minus the re-sampling control "
              r"(\texttt{retry}), on three checkpoints. Unit of inference is the problem ($n{=}45$; "
              r"three seeds averaged within problem). 95\% CI: problem-level bootstrap. $p$: sign-flip "
              r"permutation test, Holm-adjusted over the four conditions. The selected checkpoint's "
              r"per-attempt rows were not retained, so only point estimates are available.}",
              r"\label{tab:ladder}",
              r"\resizebox{\linewidth}{!}{%",
              r"\begin{tabular}{lccc}\toprule",
              r" & Sep, pass 4 & Nov, pass 4 & Nov, selected \\",
              r"checkpoint greedy solve rate & 0.133 & 0.244 & 0.444 \\",
              r"\texttt{retry} solve@3 & %.3f & %.3f & %.3f \\ \midrule" % (
                  res["ladder"]["sep_pass4"]["solve_by_budget"][3]["retry"],
                  res["ladder"]["nov_pass4"]["solve_by_budget"][3]["retry"],
                  res["ladder"]["nov_selected"]["solve3"]["retry"])]
    for c in FEEDBACK_CONDITIONS:
        cells = []
        for key in ["sep_pass4", "nov_pass4"]:
            v = res["ladder"][key]["vs_retry_budget3"][c]
            cells.append(r"$%+.3f$ {\scriptsize[%+.3f, %+.3f]} $p_H$=%s" % (
                v["diff"], *v["ci95_problem_bootstrap"], fmt_p(v["p_holm"])))
        s = res["ladder"]["nov_selected"]["solve3"]
        cells.append(r"$%+.3f$" % (s[c] - s["retry"]))
        lines.append(r"\texttt{%s} & %s \\" % (c, " & ".join(cells)))
    lines += [r"\bottomrule\end{tabular}}\end{table}", ""]

    # Table: training
    T = res["training"]
    lines += [r"\begin{table}[t]\centering\small",
              r"\caption{Held-out greedy solve rate over eight independent training passes on the "
              r"disjoint split (four in September, four in November; identical data and configuration). "
              r"The DPO row is SFT+DPO minus SFT within the same pass.}",
              r"\label{tab:training}",
              r"\begin{tabular}{lcccc}\toprule",
              r" & mean & sd & range & passes $>0$ / $<0$ \\ \midrule",
              r"Base & 0.000 & 0.000 & 0.000--0.000 & -- \\",
              r"+ LoRA-SFT & %.3f & %.3f & %.3f--%.3f & -- \\" % (T["sft"]["mean"], T["sft"]["sd"], T["sft"]["min"], T["sft"]["max"]),
              r"+ LoRA-SFT + LoRA-DPO & %.3f & %.3f & %.3f--%.3f & -- \\" % (T["sft_dpo"]["mean"], T["sft_dpo"]["sd"], T["sft_dpo"]["min"], T["sft_dpo"]["max"]),
              r"DPO effect & $%+.3f$ & %.3f & $%+.3f$--$%+.3f$ & %d / %d \\" % (
                  T["dpo_delta"]["mean"], T["dpo_delta"]["sd"], min(p["delta"] for p in T["passes"]),
                  max(p["delta"] for p in T["passes"]), T["dpo_delta"]["positive"], T["dpo_delta"]["negative"]),
              r"\bottomrule\end{tabular}\end{table}", ""]
    # Table: displacement
    D = {(r["model"], r["family"], r["solvable"]): r for r in res["displacement"]["by_model_family_solvable"]}
    npb = res["displacement"]["n_problems"]
    lines += [r"\begin{table}[t]\centering\small",
              r"\caption{Where DPO's probability went. Accuracy, and the fraction of answers that are "
              r"\emph{no solution}, by family and by whether the problem has a real root; eight training "
              r"passes pooled ($n$ = held-out problems, each scored once per pass). No preference pair contains \emph{no "
              r"solution} on either side.}",
              r"\label{tab:displacement}",
              r"\begin{tabular}{lrcccc}\toprule",
              r" & & \multicolumn{2}{c}{accuracy} & \multicolumn{2}{c}{answers \emph{no solution}} \\",
              r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
              r"problems & $n$ & SFT & +DPO & SFT & +DPO \\ \midrule"]
    for fam, solv, lab, key in [("linear", True, "linear", "linear_solvable"),
                                ("quadratic", True, "quadratic, real roots", "quadratic_solvable"),
                                ("quadratic", False, "quadratic, no real root", "quadratic_no_real_root"),
                                ("rational", True, "rational, real root", "rational_solvable"),
                                ("rational", False, "rational, no real root", "rational_no_real_root")]:
        s, d = D[("sft", fam, solv)], D[("sft+dpo", fam, solv)]
        lines.append(r"%s & %d & %.3f & %.3f & %.3f & %.3f \\" % (
            lab, npb[key], s["accuracy"], d["accuracy"], s["says_no_solution"], d["says_no_solution"]))
    ns = res["displacement"]["no_solution_on_solvable"]
    lines += [r"\midrule all with a real root & %d & & & %.3f & %.3f \\" % (
                  sum(v for k, v in npb.items() if k.endswith("_solvable")), ns["sft"], ns["sft+dpo"]),
              r"\bottomrule\end{tabular}\end{table}", ""]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "tables.tex"), "w").write("\n".join(lines))


def main():
    rng = np.random.default_rng(RNG_SEED)
    ladder, pooled = analyse_ladders(rng)
    res = {
        "_README": "Generated by src/paper_analysis.py from results/recorded/. Do not edit.",
        "training": analyse_training(rng),
        "displacement": analyse_displacement(),
        "ladder": ladder,
        "ladder_pooled_raw_runs": pooled,
        "agent": analyse_agents(rng),
        "settings": {"n_perm": N_PERM, "n_boot": N_BOOT, "rng_seed": RNG_SEED},
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "analysis.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    make_tables(res)
    make_figures(res)

    T = res["training"]
    print(f"Training, {T['n_passes']} passes: SFT {T['sft']['mean']:.3f} (sd {T['sft']['sd']:.3f}), "
          f"SFT+DPO {T['sft_dpo']['mean']:.3f} (sd {T['sft_dpo']['sd']:.3f})")
    print(f"  DPO delta mean {T['dpo_delta']['mean']:+.3f}, sd {T['dpo_delta']['sd']:.3f}, "
          f"t={T['dpo_delta']['t']:.2f} p={T['dpo_delta']['p_t']:.3f}, "
          f"{T['dpo_delta']['positive']}+/{T['dpo_delta']['negative']}-/{T['dpo_delta']['zero']}0")
    for f, v in T["dpo_delta_by_family"].items():
        print(f"  {f:9s} delta {v['delta_mean']:+.3f}  ({v['passes_positive']}+/{v['passes_negative']}-)  p={v['p_sign_flip']:.3f}")
    print(f"  selected checkpoint: SFT {T['selected_checkpoint']['sft']:.3f} -> {T['selected_checkpoint']['sft_dpo']:.3f}")
    Dp = res["displacement"]
    print(f"\nDisplacement: `no solution` on SOLVABLE problems: "
          + ", ".join(f"{m} {v:.3f}" for m, v in Dp["no_solution_on_solvable"].items())
          + f"  | selected ckpt: " + ", ".join(f"{m} {v:.3f}" for m, v in Dp["no_solution_on_solvable_selected"].items()))
    print(f"  quadratic accuracy: {Dp['quadratic_accuracy']}")
    print(f"  per-pass spearman(no-sol rate, DPO delta) = {Dp['per_pass_spearman_nosol_vs_delta']}")
    for r in Dp["by_model_family_solvable"]:
        if r["model"] != "base":
            print(f"    {r['model']:8s} {r['family']:9s} solvable={r['solvable']!s:5s} acc={r['accuracy']:.3f} "
                  f"says_nosol={r['says_no_solution']:.3f} (n={r['n_attempts']})")
    print("  trainer rewards/chosen: " + ", ".join(f"{v['rewards_chosen']:+.2f}" for v in Dp["trainer_metrics"].values()))
    print("\nLadder, solve@3 minus retry (problem-level):")
    for key, r in ladder.items():
        if not r["raw_available"]:
            s = r["solve3"]
            print(f"  {r['checkpoint']:14s} " + "  ".join(f"{c}={s[c]-s['retry']:+.3f}" for c in FEEDBACK_CONDITIONS) + "  (summary only)")
            continue
        print(f"  {r['checkpoint']:14s} turn-1 paired: {r['turn1_identical_across_conditions']}")
        for c in FEEDBACK_CONDITIONS:
            v = r["vs_retry_budget3"][c]
            print(f"     {c:11s} {v['diff']:+.3f} CI[{v['ci95_problem_bootstrap'][0]:+.3f},{v['ci95_problem_bootstrap'][1]:+.3f}] "
                  f"p_perm={v['p_problem_permutation']:.3f} p_holm={v['p_holm']:.3f} | naive 135-unit sign p={v['naive_unit_sign_test']['p']:.3f}")
    print("  pooled (raw runs):")
    for c, v in pooled.items():
        print(f"     {c:11s} {v['diff']:+.3f} CI[{v['ci95_problem_bootstrap'][0]:+.3f},{v['ci95_problem_bootstrap'][1]:+.3f}] p_holm={v['p_holm']:.3f}")
    print("\nAgent loop, feedback minus control:")
    for key, r in res["agent"].items():
        if r["raw_available"]:
            for b in (2, 3):
                v = r[f"contrast_budget{b}"]
                print(f"  {r['checkpoint']:14s} budget {b}: {v['diff']:+.3f} CI[{v['ci95_problem_bootstrap'][0]:+.3f},{v['ci95_problem_bootstrap'][1]:+.3f}] p={v['p_problem_permutation']:.3f}")
        else:
            print(f"  {r['checkpoint']:14s} (no data)")
    print(f"\nWrote {os.path.relpath(OUT, ROOT)}/analysis.json, tables.tex and {os.path.relpath(FIG, ROOT)}/")


if __name__ == "__main__":
    main()
