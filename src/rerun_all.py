"""
Regenerate every recorded result from the corrected code, in one place.

WHY THIS EXISTS

Three defects found after the GPU runs (PROVENANCE.md bugs 5, 6 and 7) mean
every number in results/recorded/manifest.json marked `pending_rerun` was
produced by an instrument that was wrong. They are not "probably still about
right": bug 5 made the verifier reject `x = no solution`, the exact form its
own prompt demands, on the 8 of 45 held-out problems that have no real
solution. Every solve rate ever recorded here is therefore a LOWER bound, and
in the agent experiment the defect entered one arm's prompt and not the
other's, so even the gap between conditions is unusable.

The fix is not to re-run the pieces by hand. Doing that is how the original
runs ended up spread across four Colab sessions, three prompt variants and two
checkpoints that can no longer be told apart. This script runs the stages in
order, in one session, and writes:

  results/raw/<stage>/*.csv        every per-attempt row, committed
  results/raw/<stage>/config.json  the settings that stage resolved
  results/raw/run_manifest_<utc>_<stages>.json
                                   code hashes, versions, timestamps, device;
                                   one per invocation, never overwritten

so that the next person to ask "which model produced this number" has an
answer, which is exactly what the recorded runs cannot supply.

STAGES, and what each needs

  data      CPU, seconds.    Rebuild sft_pairs.jsonl and dpo_pairs_zeroshot.jsonl.
  cpu       CPU, ~2 min.     The synthetic turn-budget experiment and its figures.
  sft       GPU, ~3 min.     LoRA-SFT, merged into base weights.
  dpo       GPU, ~4 min.     LoRA-DPO on top of the merged SFT model.
  select    GPU, ~25 min.    Retrain until DPO produces a checkpoint that did not
                             collapse, judged on the VALIDATION set and never on
                             the benchmark. Logs every attempt, so the collapse
                             rate is measured rather than worked around.
  eval      GPU, ~6 min.     Base / SFT / SFT+DPO on the held-out set.
  agent     GPU, ~50 min.    Feedback vs re-sampling control, both prompt styles.
  ablation  GPU, ~45 min.    The five-condition feedback ladder.

  python rerun_all.py data cpu              # no GPU needed
  python rerun_all.py sft dpo eval          # the headline table, ~15 min
  python rerun_all.py agent                 # the central experiment
  python rerun_all.py ablation
  python rerun_all.py all

Stages are ordered and dependency-checked: `eval` refuses to start if the SFT
model it would evaluate does not exist, rather than evaluating a stale one.

ON REPEATING THE TRAINING RUNS

`--repeat N` runs the chosen stages N times end to end, and is only meaningful
when `sft` or `dpo` is among them. Evaluation is greedy, so it is deterministic
on a fixed checkpoint: `eval --repeat=4` yields four byte-identical tables and
no variance estimate at all. The script refuses that combination outright,
because a table printed four times looks like a replication and is not one.

The useful invocation is `sft dpo eval --repeat=4`. Each pass trains a fresh
model, and the spread between passes is the GPU non-determinism -- which is the
most informative thing in the whole table, since a single pass is one draw
rather than a measurement. `run_dpo_seed_sweep.py` is cheaper if only the DPO
stage's variance is wanted.

WHAT THIS SCRIPT DOES NOT DO

It does not touch results/recorded/manifest.json. Promoting a regenerated
number to `valid` is a decision a person makes after looking at the output,
not something a script does to its own report card.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "results", "raw")

STAGES = {
    "data": {
        "needs_gpu": False,
        "scripts": ["build_sft_data.py", "build_dpo_pairs_zeroshot.py"],
        "requires": [],
        "produces": ["data/sft_pairs.jsonl", "data/dpo_pairs_zeroshot.jsonl"],
        "minutes": 1,
    },
    "cpu": {
        "needs_gpu": False,
        "scripts": ["run_experiment.py", "analysis.py", "make_figures.py"],
        "requires": [],
        "produces": ["results/trajectories.csv", "results/summary.csv"],
        "minutes": 2,
    },
    "sft": {
        "needs_gpu": True,
        "scripts": ["train_lora_sft.py"],
        "requires": ["data/sft_pairs.jsonl"],
        "produces": ["results/sft-merged-model"],
        "minutes": 3,
    },
    "dpo": {
        "needs_gpu": True,
        "scripts": ["train_lora_dpo_zeroshot.py"],
        "requires": ["results/sft-merged-model", "data/dpo_pairs_zeroshot.jsonl"],
        "produces": ["results/lora-dpo-zeroshot-adapter"],
        "minutes": 4,
    },
    "select": {
        "needs_gpu": True,
        "scripts": ["select_checkpoint.py"],
        "requires": ["data/sft_pairs.jsonl", "data/dpo_pairs_zeroshot.jsonl"],
        "produces": ["results/selected-checkpoint"],
        "minutes": 25,
    },
    "eval": {
        "needs_gpu": True,
        "scripts": ["evaluate_redesign_v2.py"],
        "requires": ["results/sft-merged-model", "results/lora-dpo-zeroshot-adapter"],
        "produces": [],
        "minutes": 6,
    },
    "agent": {
        "needs_gpu": True,
        "scripts": ["run_llm_agent_experiment.py"],
        "requires": ["results/sft-merged-model", "results/lora-dpo-zeroshot-adapter"],
        "produces": [],
        "minutes": 50,
    },
    "ablation": {
        "needs_gpu": True,
        "scripts": ["feedback_protocol_ablation.py"],
        "requires": ["results/sft-merged-model", "results/lora-dpo-zeroshot-adapter"],
        "produces": [],
        "minutes": 45,
    },
}
ORDER = ["data", "cpu", "sft", "dpo", "select", "eval", "agent", "ablation"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def code_fingerprint():
    """Hash every source file, so a results file can be tied to the code that
    made it. The recorded runs cannot be, which is the reason this exists."""
    return {name: _sha256(os.path.join(HERE, name))
            for name in sorted(os.listdir(HERE)) if name.endswith(".py")}


def environment():
    env = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": "cpu",
    }
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["device"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
    except ImportError:
        env["torch"] = None
    for module in ("transformers", "trl", "peft", "accelerate", "datasets", "sympy"):
        try:
            env[module] = __import__(module).__version__
        except Exception:
            env[module] = None
    return env


def have_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _missing(paths):
    return [p for p in paths if not os.path.exists(os.path.join(ROOT, p))]


def run_stage(name, log):
    stage = STAGES[name]
    missing = _missing(stage["requires"])
    if missing:
        raise SystemExit(
            f"stage '{name}' needs {missing}, which do not exist.\n"
            f"Run the earlier stages first: {' '.join(ORDER[:ORDER.index(name)])}\n"
            f"Refusing to continue rather than evaluating a stale or absent model."
        )
    if stage["needs_gpu"] and not have_gpu():
        raise SystemExit(
            f"stage '{name}' needs a GPU and torch.cuda.is_available() is False.\n"
            f"CPU-only stages: {[s for s in ORDER if not STAGES[s]['needs_gpu']]}"
        )

    out_dir = os.path.join(RAW, name)
    os.makedirs(out_dir, exist_ok=True)
    for script in stage["scripts"]:
        print(f"\n{'=' * 70}\n[{name}] {script}\n{'=' * 70}", flush=True)
        started = time.time()
        proc = subprocess.run([sys.executable, script], cwd=HERE)
        elapsed = round(time.time() - started, 1)
        log.append({"stage": name, "script": script, "seconds": elapsed,
                    "returncode": proc.returncode})
        if proc.returncode != 0:
            raise SystemExit(f"[{name}] {script} exited {proc.returncode}; stopping.")
        print(f"[{name}] {script} finished in {elapsed}s", flush=True)

    produced = _missing(stage["produces"])
    if produced:
        raise SystemExit(f"stage '{name}' claimed to produce {produced}, which are absent.")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    repeat = 1
    for flag in sys.argv[1:]:
        if flag.startswith("--repeat="):
            repeat = int(flag.split("=", 1)[1])

    if not args or args == ["all"]:
        stages = list(ORDER)
    else:
        unknown = [a for a in args if a not in STAGES]
        if unknown:
            raise SystemExit(f"unknown stage(s) {unknown}; choose from {ORDER}")
        stages = [s for s in ORDER if s in args]   # always in dependency order

    os.makedirs(RAW, exist_ok=True)
    total = sum(STAGES[s]["minutes"] for s in stages) * repeat
    print(f"Stages: {stages}" + (f" x{repeat}" if repeat > 1 else ""))
    print(f"Rough budget: ~{total} min")
    print(f"GPU available: {have_gpu()}\n", flush=True)

    # --repeat only means something if the pass actually retrains. Greedy
    # decoding on a fixed checkpoint is deterministic, so `eval --repeat=4`
    # produces four byte-identical tables and measures nothing -- which is
    # exactly what happened on the first regeneration attempt, at a cost of
    # 24 GPU-minutes. Refuse it rather than let it look like a variance
    # estimate.
    TRAINING = {"sft", "dpo"}
    if repeat > 1 and not (TRAINING & set(stages)):
        raise SystemExit(
            f"--repeat={repeat} was given, but none of {stages} retrains anything.\n"
            f"Evaluation is greedy and therefore deterministic: every pass would\n"
            f"produce the identical table, which is not a variance estimate.\n"
            f"What you want is:  python rerun_all.py sft dpo eval --repeat={repeat}\n"
            f"so each pass trains a fresh model and the spread between passes is\n"
            f"the GPU non-determinism you are trying to measure."
        )
    if repeat > 1 and "eval" not in stages:
        print(f"note: --repeat={repeat} will retrain {sorted(TRAINING & set(stages))} "
              f"but never evaluate, so the passes leave no comparable numbers.\n",
              flush=True)

    log = []
    for pass_index in range(repeat):
        if repeat > 1:
            print(f"\n########## pass {pass_index + 1} of {repeat} ##########", flush=True)
        for name in stages:
            run_stage(name, log)

    manifest = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stages": stages,
        "repeat": repeat,
        "environment": environment(),
        "code_sha256": code_fingerprint(),
        "log": log,
        "note": ("Produced by rerun_all.py. results/recorded/manifest.json is NOT "
                 "updated automatically -- promoting a regenerated number to "
                 "`valid` is a judgement, not a side effect."),
    }
    # One manifest per invocation, never overwritten. A single fixed filename
    # kept only the last stage's record: after the 2026-09-22 session the file
    # on disk said `stages: ["ablation"]` and the provenance of the four
    # training passes it was supposed to vouch for was gone.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(RAW, f"run_manifest_{stamp}_{'-'.join(stages)}.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(RAW, "run_manifest.json"), "w") as f:   # the latest, for convenience
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {os.path.relpath(path, ROOT)}")
    print("Next: compare the new numbers against results/recorded/manifest.json, "
          "then update it and the docs that read from it.")


if __name__ == "__main__":
    main()
