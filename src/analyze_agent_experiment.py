"""
Post-hoc analysis of run_llm_agent_experiment.py's trajectory log.

The headline of that experiment is that verifier feedback did not beat a
re-sampling control (see results/colab_run/agent_experiment.md). This script
interrogates the logged trajectories for *why*, and in particular tests the
one hypothesis that was registered in advance: that the revision prompt is
out of distribution for a model fine-tuned on a single rigid prompt shape,
and degrades its output format.

That hypothesis makes a sharp, falsifiable prediction: at turns 2-3 the
feedback condition should produce more `unparsable` verdicts than the
control. If it does, the feedback prompt is breaking the answer format
rather than the feedback content being unhelpful -- a fixable engineering
problem, not a finding about verifier feedback. If it does not, that
explanation is dead and hypotheses 2 and 3 (anchoring, feedback too generic)
are what remain.

CPU-only, reads results/llm_agent_experiment.csv, writes a summary next to it.

    python analyze_agent_experiment.py
"""
import csv
import sys
from collections import Counter, defaultdict

# Defaults to the first (verbose-prompt) run. Pass a path to analyse another,
# e.g. `python analyze_agent_experiment.py ../results/llm_agent_experiment_minimal.csv`
IN_CSV = sys.argv[1] if len(sys.argv) > 1 else "../results/llm_agent_experiment.csv"
OUT_TXT = IN_CSV.replace(".csv", "_analysis.txt").replace(
    "llm_agent_experiment", "agent_experiment")


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["turn"] = int(r["turn"])
        r["seed"] = int(r["seed"])
        r["correct"] = r["correct"] in ("True", "true", "1")
    return rows


def main():
    rows = load(IN_CSV)
    out = []

    def say(line=""):
        print(line)
        out.append(line)

    say(f"Trajectory rows: {len(rows)}")
    say(f"Conditions: {sorted(set(r['condition'] for r in rows))}")
    say(f"Seeds: {sorted(set(r['seed'] for r in rows))}")
    say()

    # --- The pre-registered test -------------------------------------------
    say("=" * 66)
    say("PRE-REGISTERED TEST: does the revision prompt break the answer format?")
    say("=" * 66)
    say("Prediction if the OOD-prompt hypothesis is right: the feedback")
    say("condition shows MORE 'unparsable' verdicts at turns 2-3.")
    say()
    say(f"{'turn':>5} | {'condition':>9} | {'n':>5} | {'unparsable':>11} | {'rate':>7}")
    say("-" * 52)
    unparsable = {}
    for turn in (2, 3):
        for cond in ("feedback", "control"):
            sel = [r for r in rows if r["turn"] == turn and r["condition"] == cond]
            if not sel:
                continue
            n_unp = sum(1 for r in sel if r["reason"] == "unparsable")
            rate = n_unp / len(sel)
            unparsable[(turn, cond)] = rate
            say(f"{turn:>5} | {cond:>9} | {len(sel):>5} | {n_unp:>11} | {rate:>7.3f}")
    say()
    verdicts = []
    for turn in (2, 3):
        fb = unparsable.get((turn, "feedback"))
        ct = unparsable.get((turn, "control"))
        if fb is None or ct is None:
            continue
        diff = fb - ct
        verdicts.append(diff)
        say(f"  turn {turn}: feedback - control = {diff:+.3f}")
    if verdicts:
        if all(d > 0.02 for d in verdicts):
            say("\n  => SUPPORTED. The feedback prompt degrades the output format.")
            say("     That makes this an engineering problem (prompt/format), not")
            say("     evidence that verifier feedback is unhelpful in principle.")
        elif all(abs(d) <= 0.02 for d in verdicts):
            say("\n  => NOT SUPPORTED. Parse failure rates are essentially equal, so")
            say("     the feedback prompt is not breaking the format. The null result")
            say("     is about the feedback CONTENT, not about output formatting.")
        else:
            say("\n  => MIXED. No clean verdict; inspect per-turn numbers above.")

    # --- What the artifact does to the headline comparison -----------------
    say()
    say("=" * 66)
    say("CORRECTNESS AMONG ANSWERS THAT WERE PARSABLE AT ALL")
    say("=" * 66)
    say("An unparsable answer scores as wrong. If one condition produces far")
    say("more of them, the headline comparison is partly measuring format")
    say("breakage rather than reasoning. This re-computes accuracy over only")
    say("the answers the extractor could actually read.")
    say()
    say(f"{'turn':>5} | {'condition':>9} | {'n':>4} | {'parsable':>8} | "
        f"{'correct':>7} | {'raw acc':>7} | {'acc|parsable':>12}")
    say("-" * 74)
    cond_rate = {}
    raw_rate = {}
    for turn in (2, 3):
        for cond in ("feedback", "control"):
            sel = [r for r in rows if r["turn"] == turn and r["condition"] == cond]
            if not sel:
                continue
            n = len(sel)
            n_unp = sum(1 for r in sel if r["reason"] == "unparsable")
            n_par = n - n_unp
            n_cor = sum(1 for r in sel if r["correct"])
            raw = n_cor / n if n else 0.0
            cond_acc = n_cor / n_par if n_par else float("nan")
            raw_rate[(turn, cond)] = raw
            cond_rate[(turn, cond)] = cond_acc
            say(f"{turn:>5} | {cond:>9} | {n:>4} | {n_par:>8} | {n_cor:>7} | "
                f"{raw:>7.3f} | {cond_acc:>12.3f}")
    say()
    flipped = []
    for turn in (2, 3):
        if (turn, "feedback") not in raw_rate:
            continue
        rd = raw_rate[(turn, "feedback")] - raw_rate[(turn, "control")]
        cd = cond_rate[(turn, "feedback")] - cond_rate[(turn, "control")]
        flipped.append((rd, cd))
        say(f"  turn {turn}: raw difference {rd:+.3f}   ->   parsable-only difference {cd:+.3f}")
    if flipped and all(rd < 0 < cd for rd, cd in flipped):
        say()
        say("  => THE SIGN FLIPS. Feedback looks worse overall but is equal or")
        say("     better among answers that could be read. The headline gap is")
        say("     therefore attributable to the format artifact, and this")
        say("     experiment CANNOT settle whether the feedback content helps.")
        say("     Treat the primary result as inconclusive, not as a null.")
        say()
        say("  CAVEAT: this is a post-hoc subgroup analysis. Parsability is not")
        say("  randomly assigned -- a model may emit unreadable text precisely")
        say("  when it is confused -- so this is not a clean causal estimate.")
        say("  It is strong enough to invalidate the raw comparison, not strong")
        say("  enough to claim feedback helps.")

    # --- Full error-mode breakdown -----------------------------------------
    say()
    say("=" * 66)
    say("ERROR MODES BY CONDITION AND TURN")
    say("=" * 66)
    by = defaultdict(Counter)
    for r in rows:
        by[(r["turn"], r["condition"])][r["reason"]] += 1
    reasons = sorted({r["reason"] for r in rows})
    header = f"{'turn':>5} | {'condition':>9} | " + " | ".join(f"{x:>15}" for x in reasons)
    say(header)
    say("-" * len(header))
    for turn in sorted({r["turn"] for r in rows}):
        for cond in ("feedback", "control"):
            c = by.get((turn, cond))
            if not c:
                continue
            total = sum(c.values())
            cells = " | ".join(f"{c[x]:>5} ({c[x]/total*100:>4.1f}%)" for x in reasons)
            say(f"{turn:>5} | {cond:>9} | {cells}")

    # --- Did feedback ever fix the specific error it named? ----------------
    say()
    say("=" * 66)
    say("DID A WRONG ANSWER CHANGE AT ALL AFTER FEEDBACK?")
    say("=" * 66)
    say("If the model largely repeats its previous answer, that points at")
    say("anchoring rather than at the feedback being uninformative.")
    say()
    traj = defaultdict(dict)
    for r in rows:
        traj[(r["condition"], r["seed"], r["problem_id"])][r["turn"]] = r
    for cond in ("feedback", "control"):
        same = changed = 0
        for key, turns in traj.items():
            if key[0] != cond:
                continue
            for t in (2, 3):
                if t in turns and (t - 1) in turns and not turns[t - 1]["correct"]:
                    if turns[t]["candidate"].strip() == turns[t - 1]["candidate"].strip():
                        same += 1
                    else:
                        changed += 1
        tot = same + changed
        if tot:
            say(f"  {cond:>9}: repeated previous answer {same}/{tot} times ({same/tot*100:.1f}%)")

    # --- Recovery: how many wrong-at-turn-1 problems were fixed? -----------
    say()
    say("=" * 66)
    say("RECOVERY FROM A WRONG FIRST ATTEMPT")
    say("=" * 66)
    for cond in ("feedback", "control"):
        wrong_first = 0
        recovered = 0
        for key, turns in traj.items():
            if key[0] != cond or 1 not in turns:
                continue
            if turns[1]["correct"]:
                continue
            wrong_first += 1
            if any(turns[t]["correct"] for t in turns if t > 1):
                recovered += 1
        if wrong_first:
            say(f"  {cond:>9}: {recovered}/{wrong_first} recovered "
                f"({recovered/wrong_first*100:.1f}% of wrong first attempts)")

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(out) + "\n")
    say()
    say(f"Written to {OUT_TXT}")


if __name__ == "__main__":
    main()
