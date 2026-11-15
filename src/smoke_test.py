"""
Four minutes that decide whether the next four hours are worth starting.

WHY THIS EXISTS

The full run is four to five hours of GPU time, and every stage of it sits
directly on library APIs that move between releases: trl's SFTConfig and
DPOConfig arguments, `assistant_only_loss`, peft's `merge_and_unload`,
transformers' `dtype=` keyword, the chat template. If one of them has changed,
the run does not fail at the start. It fails wherever that call first happens,
which for the DPO stage is after a full SFT training, and for the evaluation
is after both.

So this exercises the whole chain once, as small as it will go, before
anything expensive starts:

  1. the installed trl / peft / transformers match requirements-colab.txt
  2. a GPU is attached and torch can see it
  3. LoRA-SFT trains for one epoch, merges, and saves
  4. LoRA-DPO trains for one epoch on top of that and saves its adapter
  5. both models load back, generate, and the verifier reads what they wrote

It runs the REAL training scripts, not copies of them, with one setting
overridden through the same config layer the sweeps use
(AVS_NUM_TRAIN_EPOCHS=1). A separate miniature would test the miniature.

WHAT IT DOES NOT TELL YOU

Nothing about quality. One epoch is not a trained model and its answers are
not expected to be right; the only question asked of them is whether they
came out and could be graded. It also removes the checkpoints it made, so
nothing half-trained is left where a later stage would find it.

If real checkpoints are already on disk it refuses to touch them and exits
cleanly: a stack that produced them has already passed this test.

    python smoke_test.py
"""
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PINS_FILE = os.path.join(ROOT, "requirements-colab.txt")
LOG_PATH = os.path.join(ROOT, "results", "raw", "smoke_test.json")

SFT_ADAPTER_DIR = os.path.join(ROOT, "results", "lora-sft-adapter")
SFT_MERGED_DIR = os.path.join(ROOT, "results", "sft-merged-model")
DPO_ADAPTER_DIR = os.path.join(ROOT, "results", "lora-dpo-zeroshot-adapter")
CHECKPOINTS = (SFT_ADAPTER_DIR, SFT_MERGED_DIR, DPO_ADAPTER_DIR)

REPORTED = ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "sympy")
N_PROBLEMS = 3


def read_pins(path=PINS_FILE):
    """{package: version} for every `name==version` line in the pins file."""
    pins = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "==" in line:
                name, version = line.split("==", 1)
                pins[name.strip().lower()] = version.strip()
    return pins


def mismatches(pins, installed):
    """Pinned packages whose installed version is missing or different."""
    return {name: (want, installed.get(name))
            for name, want in pins.items() if installed.get(name) != want}


def installed_versions(names):
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _fail(phase, detail):
    raise SystemExit(
        f"\nSMOKE TEST FAILED at: {phase}\n{detail}\n\n"
        f"Do not start the full run. Send the output above; the fix is almost "
        f"certainly a library version, and it is a four-minute problem here "
        f"instead of a two-hour one later.")


def _train(script, phase):
    env = dict(os.environ, AVS_NUM_TRAIN_EPOCHS="1")
    print(f"\n--- {phase}: {script} (1 epoch)", flush=True)
    proc = subprocess.run([sys.executable, script], cwd=HERE, env=env)
    if proc.returncode != 0:
        _fail(phase, f"{script} exited {proc.returncode}; its traceback is above.")


def _has_files(path):
    return os.path.isdir(path) and any(os.scandir(path))


def main():
    started = time.time()
    record = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "python": sys.version.split()[0], "passed": False}

    # 1. versions ----------------------------------------------------------
    pins = read_pins()
    versions = installed_versions(sorted(set(REPORTED) | set(pins)))
    record["versions"], record["pins"] = versions, pins
    print("Installed versions")
    for name in sorted(versions):
        note = f"   (pinned {pins[name]})" if name in pins else ""
        print(f"  {name:14} {versions[name]}{note}")
    wrong = mismatches(pins, versions)
    if wrong:
        _fail("version check",
              "\n".join(f"  {n}: pinned {want}, installed {got}"
                        for n, (want, got) in sorted(wrong.items()))
              + "\nThe install cell did not take effect. Re-run it, then this one.")

    # 2. GPU ---------------------------------------------------------------
    try:
        import torch
    except Exception as exc:                      # an ABI break shows up here
        _fail("import torch", repr(exc))
    if not torch.cuda.is_available():
        _fail("GPU check", "torch.cuda.is_available() is False.\n"
                           "Runtime -> Change runtime type -> T4 GPU, then start again.")
    record["device"] = torch.cuda.get_device_name(0)
    record["cuda"] = torch.version.cuda
    print(f"\nGPU: {record['device']} | CUDA {record['cuda']} | torch {torch.__version__}")

    # never destroy a real checkpoint to test whether training works
    existing = [os.path.relpath(p, ROOT) for p in CHECKPOINTS if os.path.exists(p)]
    if existing:
        print(f"\nCheckpoints already on disk: {existing}")
        print("Leaving them alone and skipping the training check -- a stack that "
              "produced them has already passed it.")
        print("SMOKE TEST SKIPPED (nothing was changed)")
        return

    try:
        # 3-4. the real training scripts, one epoch each ---------------------
        _train("train_lora_sft.py", "SFT train + merge + save")
        if not _has_files(SFT_MERGED_DIR):
            _fail("SFT train + merge + save", f"{SFT_MERGED_DIR} is missing or empty.")
        _train("train_lora_dpo_zeroshot.py", "DPO train + save")
        if not _has_files(DPO_ADAPTER_DIR):
            _fail("DPO train + save", f"{DPO_ADAPTER_DIR} is missing or empty.")

        # 5. load, generate, grade -- the evaluation's own code path ----------
        print("\n--- load both models, generate, grade", flush=True)
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            import splits
            from evaluate_redesign_v2 import BASE_MODEL_NAME, hf_attempt
            from verifier import verify

            tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                SFT_MERGED_DIR, dtype=torch.float32).to("cuda")
            problems = splits.held_out()[:N_PROBLEMS]
            graded = []
            for label in ("sft", "sft+dpo"):
                if label == "sft+dpo":
                    model = PeftModel.from_pretrained(model, DPO_ADAPTER_DIR).to("cuda")
                model.eval()
                for p in problems:
                    candidate = hf_attempt(model, tok, p["equation_str"])
                    result = verify(p["equation_str"], candidate)
                    graded.append({"model": label, "equation": p["equation_str"],
                                   "candidate": candidate, "reason": result.reason})
                    print(f"  [{label:7}] {p['equation_str']:22} -> "
                          f"{candidate!r:24} ({result.reason})")
            record["generations"] = graded
        except SystemExit:
            raise
        except Exception as exc:
            import traceback
            traceback.print_exc()
            _fail("load + generate + grade", repr(exc))
    finally:
        # Whatever happened, leave nothing half-trained where a later stage,
        # or the next attempt at this test, would find it.
        for path in CHECKPOINTS:
            shutil.rmtree(path, ignore_errors=True)

    record["passed"] = True
    record["seconds"] = round(time.time() - started, 1)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSMOKE TEST PASSED in {record['seconds']:.0f}s -- train, merge, save, load, "
          f"generate and grade all work on this stack.")
    print("The one-epoch checkpoints it made have been removed. The answers above are "
          "from an under-trained model and say nothing about quality.")


if __name__ == "__main__":
    main()
