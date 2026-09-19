"""
How big is the DPO gain, really? Train the DPO stage under several seeds and
report the spread instead of a single number.

WHY THIS EXISTS: two runs of the identical pipeline, on identical 118-pair
data, produced held-out solve rates of 0.444 (20/45) and 0.378 (17/45) for
the SFT+DPO model. The SFT stage was stable across those runs (0.289 both
times); the variance is in the DPO stage. Reporting either number alone as
"the result" would be picking a draw from a distribution and calling it a
measurement, and picking the higher one would be worse than that.

The cause is ordinary GPU training non-determinism: CUDA matmul and
reduction kernels are not deterministic by default, so even with `seed` and
`data_seed` fixed (they now are, in train_lora_dpo_zeroshot.py) two runs
diverge numerically and end in different minima. Note what the two runs
looked like from inside training:

    run A: eval_loss 0.2401, reward acc 0.7917, margin 1.816 -> 0.444
    run B: eval_loss 0.2194, reward acc 0.8333, margin 1.872 -> 0.378

Run B was better on *every* DPO training metric and worse on the thing we
actually care about. That is worth keeping in mind: the DPO objective is a
proxy, and on a 45-problem held-out set the proxy does not reliably rank
checkpoints by solve rate.

WHAT THIS SCRIPT DOES: keeps the SFT stage fixed (it reuses the existing
../results/sft-merged-model, so run train_lora_sft.py first), then for each
seed trains a fresh DPO adapter and evaluates it on the same held-out 45
problems. It reports every seed's rate plus mean, spread, and the honest
comparison against the SFT-only baseline.

Deliberately NOT done here: sweeping the SFT stage too. That would be more
complete, but each extra stage multiplies GPU time, and the observed
variance is in DPO. If the sweep shows the SFT-only baseline itself moving,
that assumption needs revisiting.

    python run_dpo_seed_sweep.py    # needs a GPU; ~5 min per seed on a T4
"""
import json
import statistics

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from evaluate_redesign_v2 import eval_rate  # same eval code, not a copy of it
from tasks import generate_problems

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_MERGED_DIR = "../results/sft-merged-model"
DATA_PATH = "../data/dpo_pairs_zeroshot.jsonl"
HELD_OUT_SEED = 99
N_PER_FAMILY = 15
SEEDS = [0, 1, 2]
OUT_JSON = "../results/dpo_seed_sweep.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_prompt(tok, system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_pairs(tok):
    rows = [json.loads(l) for l in open(DATA_PATH)]
    return Dataset.from_list(
        [
            {
                "prompt": build_prompt(tok, r["system"], r["user"]),
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }
            for r in rows
        ]
    )


def train_dpo_for_seed(tok, seed):
    """Fresh DPO adapter on top of the fixed SFT model, trained with `seed`."""
    model = AutoModelForCausalLM.from_pretrained(SFT_MERGED_DIR)
    ds = load_pairs(tok).train_test_split(test_size=0.15, seed=0)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    out_dir = f"../results/lora-dpo-zeroshot-adapter-seed{seed}"
    config = DPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        beta=0.1,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=seed,
        data_seed=seed,
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tok,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(out_dir)
    metrics = trainer.evaluate()
    del trainer, model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out_dir, metrics


def main():
    problems = generate_problems(n_per_family=N_PER_FAMILY, seed=HELD_OUT_SEED)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    print(f"Running on: {DEVICE} | {len(problems)} held-out problems | seeds {SEEDS}",
          flush=True)

    print("\n=== SFT-only baseline (trained once, evaluated once) ===", flush=True)
    sft_model = AutoModelForCausalLM.from_pretrained(SFT_MERGED_DIR, dtype=torch.float32).to(DEVICE)
    sft_rate = eval_rate(sft_model, tok, problems, "sft")
    print(f"SFT-only zero-shot solve rate: {sft_rate:.3f}", flush=True)
    del sft_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    results = []
    for seed in SEEDS:
        print(f"\n=== DPO seed {seed}: training ===", flush=True)
        adapter_dir, metrics = train_dpo_for_seed(tok, seed)

        print(f"=== DPO seed {seed}: evaluating ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(SFT_MERGED_DIR, dtype=torch.float32).to(DEVICE)
        model = PeftModel.from_pretrained(model, adapter_dir).to(DEVICE)
        rate = eval_rate(model, tok, problems, f"dpo-s{seed}")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        results.append(
            {
                "seed": seed,
                "solve_rate": rate,
                "n_correct": round(rate * len(problems)),
                "eval_loss": metrics.get("eval_loss"),
                "reward_accuracy": metrics.get("eval_rewards/accuracies"),
                "reward_margin": metrics.get("eval_rewards/margins"),
            }
        )
        print(f"seed {seed}: SFT+DPO solve rate {rate:.3f}", flush=True)

    rates = [r["solve_rate"] for r in results]
    summary = {
        "held_out_n": len(problems),
        "sft_only_rate": sft_rate,
        "per_seed": results,
        "mean": statistics.mean(rates),
        "stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "min": min(rates),
        "max": max(rates),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    n = len(problems)
    print(f"\n{'=' * 62}")
    print(f"DPO seed sweep, {n} held-out problems")
    print("=" * 62)
    print(f"SFT-only baseline: {sft_rate:.3f} ({round(sft_rate * n)}/{n})")
    print()
    print(f"{'seed':>6} | {'solve rate':>11} | {'eval_loss':>10} | {'reward acc':>10}")
    print("-" * 48)
    for r in results:
        el = f"{r['eval_loss']:.4f}" if r["eval_loss"] is not None else "n/a"
        ra = f"{r['reward_accuracy']:.4f}" if r["reward_accuracy"] is not None else "n/a"
        print(f"{r['seed']:>6} | {r['solve_rate']:>11.3f} | {el:>10} | {ra:>10}")
    print("-" * 48)
    print(f"  mean {statistics.mean(rates):.3f}   "
          f"sd {summary['stdev']:.3f}   "
          f"range {min(rates):.3f}-{max(rates):.3f}")
    print(f"\n  DPO gain over SFT-only: mean {statistics.mean(rates) - sft_rate:+.3f}, "
          f"range {min(rates) - sft_rate:+.3f} to {max(rates) - sft_rate:+.3f}")
    print("\nQuote the mean and the range, not the best seed. With n=45 one problem")
    print("is 2.2 points, so differences of a few points are within noise.")
    print(f"\nWritten to {OUT_JSON}")


if __name__ == "__main__":
    main()
