"""
Train until the DPO stage produces a checkpoint that did not collapse.

WHY THIS EXISTS

The DPO stage fails bimodally. Across four identical passes it landed on 0.289
twice and 0.133 twice, with nothing in between -- a cliff, not a spread. The
two experiments that produced this project's first positive result (the agent
loop and the five-condition ablation) both ran on the last checkpoint trained,
which happened to be one of the collapsed ones. A damaged model has more to
recover, so feedback has an easier time helping it. That confound is the
largest one attached to those results, and removing it means running them
again on a checkpoint that works.

SELECTION MUST NOT TOUCH THE BENCHMARK

Retraining until a checkpoint looks good, judged by its score on the held-out
set, is selection on the test set. It would inflate the reported number by
exactly the amount the selection was worth, and the inflation would be
invisible. So:

    splits.validation()   45 problems, decides whether a checkpoint is kept
    splits.held_out()     45 problems, scored once afterwards, never consulted

The two are disjoint from each other. The validation set is NOT excluded from
training, deliberately: excluding it would change the training draw and
invalidate the four-pass table already measured on the current one. Selection
only has to RANK candidates that were all trained on identical data, so any
overlap is constant across them and cancels. The consequence is that a
validation score is not a measurement and is never quoted as one -- see the
docstring in splits.py.

WHAT IT RECORDS, AND WHY THAT IS ALSO A RESULT

Every attempt is logged, not just the surviving one. The number of attempts
needed is a direct estimate of the collapse rate, which is a finding in its
own right: "DPO on 118 near-identical preference pairs collapses on roughly
half of runs" is worth more than a single working checkpoint. Reporting only
the survivor would hide it.

THE RULE IS RELATIVE, NOT A FIXED THRESHOLD

A checkpoint is kept when its SFT+DPO validation rate is within `--max-drop`
(default 0.075, three problems of 45) of the SAME attempt's SFT-only rate. An
earlier draft used a fixed floor of 0.25, calibrated on the two modes seen on
the benchmark (0.133 and 0.289). That calibration does not transfer: 9 of the
45 validation equations also sit in the SFT training file, so validation
scores run higher than benchmark scores by an amount nobody has measured, and
a fixed floor could have kept a collapsed model or rejected every healthy one.

Collapse is a property of what the DPO stage did to the model it was given,
so it is measured that way: across the four recorded passes the DPO delta was
+0.000 and +0.022 on the healthy ones and -0.178 and -0.156 on the collapsed
ones. The default sits midway. Both rates come from the same problems on the
same attempt, so the overlap with training cancels out of the comparison.

    python select_checkpoint.py                    # up to 6 attempts
    python select_checkpoint.py --max-attempts 10 --max-drop 0.05
    python select_checkpoint.py --min-validation 0.2   # optional absolute floor too

On success the surviving SFT and DPO artefacts are left in place, so
`rerun_all.py agent` and `rerun_all.py ablation` run against them next.
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import splits

# torch, peft, transformers and evaluate_redesign_v2 are imported inside
# validation_score(), not here, so that is_collapsed() -- the one decision this
# script makes -- can be imported and tested without a GPU stack.

HERE = os.path.dirname(os.path.abspath(__file__))
SFT_MERGED_DIR = "../results/sft-merged-model"
DPO_ADAPTER_DIR = "../results/lora-dpo-zeroshot-adapter"
KEEP_DIR = "../results/selected-checkpoint"
LOG_PATH = "../results/raw/checkpoint_selection.json"

DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_MAX_DROP = 0.075        # midway between the healthy and collapsed DPO deltas


def is_collapsed(sft_rate, sft_dpo_rate, max_drop=DEFAULT_MAX_DROP, floor=None):
    """Did the DPO stage wreck the model it was handed?

    Relative to the same attempt's SFT-only rate, on the same problems. See
    the module docstring for why this is not a fixed threshold. `floor` is an
    optional absolute minimum on top, off by default.
    """
    # the 1e-9 keeps "exactly max_drop below" on the kept side despite float noise
    if sft_dpo_rate < sft_rate - max_drop - 1e-9:
        return True
    return floor is not None and sft_dpo_rate < floor


def _arg(flag, default, cast):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def train_once():
    for script in ("train_lora_sft.py", "train_lora_dpo_zeroshot.py"):
        print(f"\n--- {script}", flush=True)
        proc = subprocess.run([sys.executable, script], cwd=HERE)
        if proc.returncode != 0:
            raise SystemExit(f"{script} exited {proc.returncode}")


def validation_score():
    """Solve rate on the VALIDATION set. Never the benchmark."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from evaluate_redesign_v2 import BASE_MODEL_NAME, eval_rate

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(SFT_MERGED_DIR, dtype=torch.float32).to(DEVICE)
    sft = eval_rate(model, tok, splits.validation(), "sft/val")
    model = PeftModel.from_pretrained(model, DPO_ADAPTER_DIR).to(DEVICE)
    dpo = eval_rate(model, tok, splits.validation(), "sft+dpo/val")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return sft, dpo


def main():
    max_attempts = _arg("--max-attempts", DEFAULT_MAX_ATTEMPTS, int)
    max_drop = _arg("--max-drop", DEFAULT_MAX_DROP, float)
    floor = _arg("--min-validation", None, float)

    # Start from a clean slate, so a log left by an earlier selection can never
    # be mistaken for this one's verdict if this run dies part-way.
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    print(f"Selecting on {len(splits.validation())} VALIDATION problems "
          f"(seed {splits.VALIDATION_SEED}); the {len(splits.held_out())}-problem "
          f"benchmark is not consulted.")
    print(f"Keeping the first checkpoint whose SFT+DPO validation rate is within "
          f"{max_drop} of its own SFT-only rate"
          + (f" and at least {floor}." if floor is not None else "."))
    print(f"At most {max_attempts} attempts.\n", flush=True)

    attempts = []
    kept = None
    for i in range(1, max_attempts + 1):
        print(f"{'=' * 70}\nattempt {i} of {max_attempts}\n{'=' * 70}", flush=True)
        train_once()
        sft, dpo = validation_score()
        collapsed = is_collapsed(sft, dpo, max_drop, floor)
        attempts.append({"attempt": i, "sft_validation": sft, "sft_dpo_validation": dpo,
                         "dpo_delta": dpo - sft, "collapsed": collapsed,
                         "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        print(f"\nattempt {i}: SFT {sft:.3f} | SFT+DPO {dpo:.3f} | delta {dpo - sft:+.3f} "
              f"-> {'COLLAPSED' if collapsed else 'kept'}", flush=True)
        if not collapsed:
            shutil.rmtree(KEEP_DIR, ignore_errors=True)
            os.makedirs(KEEP_DIR, exist_ok=True)
            shutil.copytree(SFT_MERGED_DIR, f"{KEEP_DIR}/sft-merged-model")
            shutil.copytree(DPO_ADAPTER_DIR, f"{KEEP_DIR}/lora-dpo-zeroshot-adapter")
            kept = attempts[-1]
            break

    collapses = sum(1 for a in attempts if a["collapsed"])
    summary = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": f"kept if sft_dpo >= sft - {max_drop} on validation"
                + (f" and sft_dpo >= {floor}" if floor is not None else ""),
        "max_drop": max_drop,
        "absolute_floor": floor,
        "validation_scores_are_not_measurements": (
            "9 of the 45 validation equations are also in the SFT training file. These "
            "rates rank checkpoints; they are never quoted as results."),
        "selected_on": f"validation (seed {splits.VALIDATION_SEED}, "
                       f"{len(splits.validation())} problems)",
        "benchmark_consulted": False,
        "attempts": attempts,
        "n_attempts": len(attempts),
        "n_collapsed": collapses,
        "collapse_rate": collapses / len(attempts),
        "kept": kept,
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"{collapses} of {len(attempts)} attempts collapsed "
          f"({collapses / len(attempts):.0%}). That rate is itself a result; "
          f"report it alongside\nwhatever the surviving checkpoint goes on to show.")
    if kept is None:
        raise SystemExit(
            f"\nNo checkpoint survived in {max_attempts} attempts. The DPO stage is worse "
            f"than bimodal on this configuration.\nDO NOT run eval / agent / ablation now: "
            f"the model left on disk is a collapsed one.\nLowering beta or the learning "
            f"rate is the cheaper next test; see configs/dpo.yaml.")
    print(f"kept attempt {kept['attempt']}: validation {kept['sft_dpo_validation']:.3f}")
    print(f"copied to {KEEP_DIR}/")
    print(f"log: {LOG_PATH}")
    print("\nThe working checkpoint is also still in place at "
          f"{SFT_MERGED_DIR} and {DPO_ADAPTER_DIR},")
    print("so `python rerun_all.py eval`, then `agent`, then `ablation`, now run against it.")


if __name__ == "__main__":
    main()
