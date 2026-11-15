"""
Combine the per-pass evaluation outputs into one table.

`rerun_all.py sft dpo eval --repeat=N` trains N fresh models and evaluates
each, leaving N files in results/raw/eval/. Reading them one at a time is how
a person ends up quoting whichever pass they happened to look at -- which is
exactly what an earlier version of this project did when it reported run A's
0.444 as "the" result, a figure 1.22 standard deviations above the mean of the
four passes that existed.

So this reads all of them and reports the spread, never a single pass. It
prints:

  - every pass, in the order it ran
  - mean, sd and range per stage
  - the DPO delta per pass, which is the quantity actually in question
  - the verifier's reason breakdown, averaged, because a solve rate alone
    cannot distinguish "reasoned wrongly" from "answered in an unreadable
    shape", and the two call for different fixes
  - which held-out problems every pass solved, which no pass solved, and
    which moved between passes

    python summarize_eval.py                 # every pass of the current run
    python summarize_eval.py --last 4        # the four most recent
    python summarize_eval.py --dir ../results/recorded/2026-09-22/eval
                                             # the committed passes of record
"""
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

RAW_DIR = "../results/raw/eval"
STAGES = ("base", "sft", "sft_dpo")
LABELS = {"base": "Base", "sft": "+ LoRA-SFT", "sft_dpo": "+ LoRA-SFT + LoRA-DPO"}
SHORT = {"base": "base", "sft": "SFT", "sft_dpo": "SFT+DPO"}


def load(last=None, directory=None):
    paths = sorted(glob.glob(os.path.join(directory or RAW_DIR, "summary_*.json")))
    if last:
        paths = paths[-last:]
    return paths, [json.load(open(p)) for p in paths]


def _spread(values):
    if len(values) == 1:
        return f"{values[0]:.3f}  (one pass -- a draw, not a measurement)"
    sd = statistics.stdev(values)
    return (f"mean {statistics.mean(values):.3f}   sd {sd:.3f}   "
            f"range {min(values):.3f}-{max(values):.3f}")


def main():
    last = None
    if "--last" in sys.argv:
        last = int(sys.argv[sys.argv.index("--last") + 1])

    directory = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else None
    paths, runs = load(last, directory)
    if not runs:
        raise SystemExit(
            f"no evaluation summaries in {directory or RAW_DIR}.\n"
            f"Run:  python rerun_all.py sft dpo eval --repeat=4")

    n = len(runs)
    n_problems = runs[0]["n_problems"]
    print("=" * 74)
    print(f"HELD-OUT EVALUATION -- {n} pass{'es' if n > 1 else ''}, "
          f"{n_problems} problems, {runs[0]['decoding']} decoding")
    print("=" * 74)

    header = "".join(f"{'pass ' + str(i + 1):>10}" for i in range(n))
    print(f"{'':>24}{header}{'':>4}summary")
    print("-" * 74)
    for stage in STAGES:
        values = [r[stage] for r in runs]
        cells = "".join(f"{v:>10.3f}" for v in values)
        print(f"{LABELS[stage]:>24}{cells}    {_spread(values)}")

    deltas = [r["sft_dpo"] - r["sft"] for r in runs]
    cells = "".join(f"{d:>+10.3f}" for d in deltas)
    print(f"{'DPO delta':>24}{cells}    {_spread(deltas)}")
    if n > 1:
        positive = sum(1 for d in deltas if d > 0)
        print(f"\n  DPO helped in {positive} of {n} passes, hurt in {n - positive}.")
        if 0 < positive < n:
            print("  The sign is not stable across passes, so the stage has no "
                  "established direction on this setup.")

    # Failure modes. This is where the DPO stage's damage showed its shape:
    # `incomplete` rising while `wrong` fell means the model started dropping
    # roots, which a solve rate alone would have hidden entirely.
    print("\n" + "=" * 74)
    print("VERIFIER REASON, mean count per pass")
    print("=" * 74)
    reasons = sorted({k for r in runs for stage in r["reasons"] for k in r["reasons"][stage]})
    print(f"{'reason':>18} | " + " | ".join(f"{SHORT[s]:>10}" for s in STAGES))
    print("-" * (20 + 13 * len(STAGES)))
    for reason in reasons:
        cells = []
        for stage in STAGES:
            key = "sft+dpo" if stage == "sft_dpo" else stage
            counts = [r["reasons"].get(key, {}).get(reason, 0) for r in runs]
            cells.append(f"{statistics.mean(counts):>10.1f}")
        print(f"{reason:>18} | " + " | ".join(cells))

    # Stability across passes, per problem. A benchmark where the same
    # problems move every time is telling you about training noise; one where
    # a fixed set is never solved is telling you about the task.
    if n > 1:
        print("\n" + "=" * 74)
        print("PER-PROBLEM STABILITY (SFT+DPO stage)")
        print("=" * 74)
        solved_counts = Counter()
        for path in paths:
            attempts = path.replace("summary_", "attempts_").replace(".json", ".csv")
            if not os.path.exists(attempts):
                print("  per-problem CSVs not found; skipping")
                break
            import csv
            with open(attempts) as f:
                for row in csv.DictReader(f):
                    if row["model"] == "sft+dpo" and row["correct"] == "True":
                        solved_counts[row["problem_id"]] += 1
        else:
            always = [p for p, c in solved_counts.items() if c == n]
            sometimes = sorted(p for p, c in solved_counts.items() if 0 < c < n)
            print(f"  solved by every pass : {len(always)}")
            print(f"  solved by some passes: {len(sometimes)}  {sometimes[:8]}"
                  f"{' ...' if len(sometimes) > 8 else ''}")
            print(f"  solved by no pass    : {n_problems - len(solved_counts)}")
            if sometimes:
                print(f"\n  {len(sometimes)} problems flip between passes on identical "
                      f"data and config.\n  That is the GPU non-determinism, and it is why "
                      f"one pass is not a measurement.")

    print("\n" + "=" * 74)
    print("Files read:")
    for p in paths:
        print(f"  {os.path.relpath(p, '..')}")
    if directory and "recorded" in os.path.normpath(directory).split(os.sep):
        return   # committed evidence: already in the manifest, nothing to promote
    print("\nThese numbers are NOT automatically promoted into")
    print("results/recorded/manifest.json. Compare them there, then update it "
          "deliberately.")


if __name__ == "__main__":
    main()
