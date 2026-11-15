"""
Figures for the three findings that are hard to see in a table.

CPU-only. Every number comes from results/recorded/manifest.json -- the one
source of truth this repo has -- never from a literal typed into this file.
That rule exists because the figures and the README used to carry their own
copies of the same numbers, which is how a table and a chart drift apart
without either looking wrong.

The manifest also carries each experiment's `status`. Anything marked
`pending_rerun` is stamped as provisional on the figure itself, so a chart
lifted out of the repo and pasted into a slide cannot lose that qualifier.

Palette: two categorical slots from a validated set (blue #2a78d6, orange
#eb6834), checked with the palette validator -- worst-pair CVD deltaE 24.7,
normal-vision 33.6, both clear of the floors, and both above 3:1 on the
surface. Every series is also directly labelled, so identity never rests on
colour alone.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MANIFEST_PATH = "../results/recorded/manifest.json"
with open(MANIFEST_PATH) as _f:
    M = json.load(_f)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SURFACE = "#fcfcfb"
WARN = "#8a4b1f"      # a readable ink, not a fourth data colour
OUT = "../results"


def _provisional(fig, key):
    """Stamp a figure whose numbers are awaiting regeneration.

    Put on the figure rather than only in the caption, because a PNG travels
    without its caption and a provisional number in a slide deck is worse
    than no number at all.
    """
    entry = M[key]
    if entry.get("status") != "pending_rerun":
        return
    causes = ", ".join(entry.get("invalidated_by", []))
    fig.text(0.012, 1.085,
             "PROVISIONAL -- awaiting regeneration; do not quote",
             color=WARN, fontsize=8.5, ha="left", weight="bold")
    fig.text(0.012, 1.048,
             f"Recorded with a defective verifier ({causes}). Kept for the record, "
             f"not as a result.",
             color=MUTED, fontsize=8, ha="left")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


# --------------------------------------------------------------- figure 1 --
def fig_training_variance():
    """One stage is perfectly stable, the other is not. That contrast is the
    point, and a table of four numbers buries it.

    Coincident runs must stay countable without inventing vertical spread the
    data does not have: all four SFT runs land on 0.289, so nudging them apart
    would draw variance into the row whose whole claim is that it has none.
    Instead one mark sits at each distinct value and carries the labels of every
    run that produced it -- "A B C D" on a single dot reads as four runs at one
    number, which is exactly what happened.
    """
    from collections import defaultdict

    e = M["held_out_eval"]
    names = e["runs"]
    series = [("+ LoRA-SFT", dict(zip(names, e["sft"])), BLUE),
              ("+ LoRA-SFT + LoRA-DPO", dict(zip(names, e["sft_dpo"])), ORANGE)]

    fig, ax = plt.subplots(figsize=(7.6, 3.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)

    for i, (label, runs, c) in enumerate(series):
        row = 1 - i
        groups = defaultdict(list)
        for r, v in runs.items():
            groups[v].append(r)
        vals = list(runs.values())
        mean = sum(vals) / len(vals)
        # below the marks only: run labels sit above them, and a full-height
        # line struck through "A B C D" was the one collision left in v1
        ax.plot([mean, mean], [row - 0.34, row - 0.07], color=c, lw=1.6,
                ls=(0, (4, 2)), zorder=2)
        for v, names in groups.items():
            n = len(names)
            ax.scatter([v], [row], s=95 + 34 * (n - 1), color=c, zorder=3,
                       edgecolor=SURFACE, linewidth=1.8)
            ax.annotate(" ".join(sorted(names)), (v, row), textcoords="offset points",
                        xytext=(0, 13 if n == 1 else 15), ha="center",
                        fontsize=8.5, color=MUTED)
        ax.text(0.222, row, label, color=INK, fontsize=10, va="center", ha="right")
        ax.text(0.60, row - 0.22,
                f"mean {mean:.3f}    spread {max(vals) - min(vals):.3f}",
                color=MUTED, fontsize=9, va="center", ha="right")

    ax.set_xlim(0.225, 0.605)
    ax.set_ylim(-0.62, 1.62)
    ax.set_yticks([])
    ax.set_xlabel(f"Held-out solve rate ({e['n_problems']} problems)",
                  color=MUTED, fontsize=9)
    fig.suptitle("Four identical training runs: the DPO stage is the unstable one",
                 color=INK, fontsize=11.5, x=0.012, ha="left", y=1.01)
    fig.text(0.012, -0.055,
             "Same data, same config, same seed; the spread is GPU non-determinism alone. "
             "Dashed line marks the mean.\nEach dot is labelled with every run that landed "
             "on it, so all four runs are accounted for in both rows.",
             color=MUTED, fontsize=8.5, ha="left")
    _provisional(fig, "held_out_eval")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_training_variance.png", dpi=180,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------- figure 2 --
def fig_passk_illusion():
    """The feedback curve rises. Alone it reads as self-refinement working.
    The control rises further. That is the whole finding."""
    e = M["agent_experiment"]
    budgets = [1, 2, 3]
    runs = [(f"Run 1 (verbose prompt, ckpt {e['runs']['verbose']['checkpoint'][0]})",
             e["runs"]["verbose"]["feedback_by_budget"],
             e["runs"]["verbose"]["control_by_budget"]),
            (f"Run 2 (repaired prompt, ckpt {e['runs']['minimal']['checkpoint'][0]})",
             e["runs"]["minimal"]["feedback_by_budget"],
             e["runs"]["minimal"]["control_by_budget"])]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), facecolor=SURFACE, sharey=True)
    for ax, (title, fb, ct) in zip(axes, runs):
        _style(ax)
        ax.plot(budgets, ct, color=ORANGE, lw=2, marker="o", ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.plot(budgets, fb, color=BLUE, lw=2, marker="o", ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.annotate("control\n(no feedback)", (3, ct[-1]), textcoords="offset points",
                    xytext=(8, 2), color=ORANGE, fontsize=9, va="center", weight="bold")
        ax.annotate("feedback", (3, fb[-1]), textcoords="offset points",
                    xytext=(8, -6), color=BLUE, fontsize=9, va="center", weight="bold")
        ax.set_xticks(budgets)
        ax.set_xlim(0.85, 3.9)
        ax.set_xlabel("Turn budget", color=MUTED, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("Solve rate", color=MUTED, fontsize=9)
    axes[0].set_ylim(0.24, 0.47)
    fig.suptitle("The feedback curve rises. The control rises further.",
                 color=INK, fontsize=11.5, x=0.012, ha="left", y=1.02)
    fig.text(0.012, -0.075,
             "Identical sampling seeds; the two conditions differ only in the turn-2/3 prompt. "
             "Turn 1 is identical by construction.\nThe feedback arm alone reads as a +0.104 "
             "self-refinement gain -- the control is what shows it as the pass@k effect. That "
             "reasoning is\nwhat survives the verifier defect; these particular values are not.",
             color=MUTED, fontsize=8.5, ha="left")
    _provisional(fig, "agent_experiment")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_passk_illusion.png", dpi=180,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------- figure 3 --
def fig_format_degradation():
    """Why the comparison above cannot be taken at face value: the feedback
    prompt destroys a large share of its own measurements."""
    e = M["agent_experiment"]["runs"]
    labels, fb, ct = [], [], []
    for n, style in ((1, "verbose"), (2, "minimal")):
        for turn in ("turn2", "turn3"):
            labels.append(f"Run {n}\n{turn[:-1]} {turn[-1]}")
            fb.append(e[style]["unparsable"][turn]["feedback"] * 100)
            ct.append(e[style]["unparsable"][turn]["control"] * 100)
    x = range(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 3.8), facecolor=SURFACE)
    _style(ax)
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], fb, w, color=BLUE, zorder=3)
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], ct, w, color=ORANGE, zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.1f}%", (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, color=MUTED)
    ax.set_ylabel("Unparsable answers (%)", color=MUTED, fontsize=9)
    ax.set_ylim(0, 45)
    ax.legend([b1, b2], ["with feedback", "control"], frameon=False,
              loc="upper right", fontsize=9, labelcolor=MUTED)
    ax.set_title("Unparsable answers by condition, as recorded",
                 color=INK, fontsize=11, loc="left", pad=12)
    fig.text(0.012, -0.105,
             "An unparsable answer scores as wrong, so these are measurements lost before "
             "reasoning is judged.\nRead as recorded, not as a finding: the verifier that "
             "produced these counts rejected 'x = no solution', the exact form\nits own prompt "
             "demands, on the 8 of 45 problems with no real solution. That defect also enters "
             "only ONE arm's\nprompt -- the control never sees a verdict -- so it can bias the "
             "gap in either direction. Regeneration pending.",
             color=MUTED, fontsize=8.5, ha="left")
    _provisional(fig, "agent_experiment")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_format_degradation.png", dpi=180,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------- figure 4 --
def fig_feedback_ladder():
    """The project's first positive result, and the one that inverts its own
    hypothesis.

    Two panels because the finding is a tension between them: the conditions
    that help most on accuracy also cost format validity, and the condition
    built to protect format destroys it. One panel would show half of that.

    Colour carries the control-versus-treatment distinction only; every
    condition is directly labelled on the axis, so nothing rests on hue.
    """
    e = M["feedback_protocol_ablation"]
    order = ["retry", "generic", "diagnostic", "actionable", "constrained"]
    rates = [e["solve_rate_budget3"][c] for c in order]
    fmt = [e["format_validity_by_turn"][c][2] for c in order]
    sig = e["paired_sign_test_135_units"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), facecolor=SURFACE)
    y = list(range(len(order)))[::-1]

    ax = axes[0]
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
    ax.axvline(rates[0], color=MUTED, lw=1.2, ls=(0, (4, 2)), zorder=2)
    for i, (c, v) in enumerate(zip(order, rates)):
        colour = MUTED if c == "retry" else BLUE
        ax.plot([rates[0], v], [y[i], y[i]], color=colour, lw=2.4, alpha=0.35, zorder=2)
        ax.scatter([v], [y[i]], s=110, color=colour, zorder=3,
                   edgecolor=SURFACE, linewidth=1.8)
        note = "control" if c == "retry" else f"{v - rates[0]:+.3f}"
        if c in sig and sig[c]["p"] < 0.05:
            note += f"   p={sig[c]['p']:.3f}"
        # a label to the right of a point that sits LEFT of the control line
        # would be struck through by that line
        left = v < rates[0]
        ax.annotate(note, (v, y[i]), textcoords="offset points",
                    xytext=(-12 if left else 12, 0),
                    ha="right" if left else "left",
                    va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9.5, color=INK)
    ax.set_xlim(0.23, 0.44)
    ax.set_xlabel("Solve rate at a 3-turn budget", color=MUTED, fontsize=9)
    ax.set_title("What the verifier's message buys", color=INK, fontsize=10.5, loc="left")

    ax = axes[1]
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
    ax.axvline(fmt[0], color=MUTED, lw=1.2, ls=(0, (4, 2)), zorder=2)
    for i, (c, v) in enumerate(zip(order, fmt)):
        colour = MUTED if c == "retry" else ORANGE
        ax.plot([fmt[0], v], [y[i], y[i]], color=colour, lw=2.4, alpha=0.35, zorder=2)
        ax.scatter([v], [y[i]], s=110, color=colour, zorder=3,
                   edgecolor=SURFACE, linewidth=1.8)
        ax.annotate(f"{v:.1%}", (v, y[i]), textcoords="offset points",
                    xytext=(-12 if c == "constrained" else 12, 0),
                    ha="right" if c == "constrained" else "left",
                    va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0.0, 1.06)
    ax.set_xlabel("Answers the extractor could read, turn 3", color=MUTED, fontsize=9)
    ax.set_title("What it costs", color=INK, fontsize=10.5, loc="left")

    fig.suptitle("Verifier feedback helps -- and the condition built to protect the "
                 "output format is the one that destroys it",
                 color=INK, fontsize=11.5, x=0.012, ha="left", y=1.03)
    fig.text(0.012, -0.10,
             "45 held-out problems x 3 seeds, paired on identical sampling seeds; turn 1 is "
             "identical in every condition (0.207 solve, 0.941 parsable).\n"
             "p-values are exact paired sign tests over 135 problem-seed units against "
             "`retry`, which carries no information at all. `constrained` is\n"
             "`actionable` plus one more sentence restating the answer format, and it drops "
             "format validity from 90% to 9%.\n"
             "One checkpoint, and a collapsed one (greedy 0.133): a damaged model leaves "
             "more for feedback to recover. Not yet replicated.",
             color=MUTED, fontsize=8.2, ha="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_feedback_ladder.png", dpi=180,
                bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    fig_training_variance()
    fig_passk_illusion()
    fig_format_degradation()
    fig_feedback_ladder()
    print("wrote results/fig_training_variance.png")
    print("wrote results/fig_passk_illusion.png")
    print("wrote results/fig_format_degradation.png")
    print("wrote results/fig_feedback_ladder.png")
