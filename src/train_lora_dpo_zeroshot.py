"""
Stage 2 of the "redesign v2" pipeline: LoRA-DPO on top of the SFT
warm-start (train_lora_sft.py), trained on data/dpo_pairs_zeroshot.jsonl --
preference pairs built in the SAME zero-shot prompt format the eval uses,
unlike the original data/dpo_pairs.jsonl (revision-template prompts; see
results/colab_run/training_log.md and PROVENANCE.md for the mismatch that
run's null result is attributed to).

Starts from MERGED_DIR (train_lora_sft.py's output: base weights + the
merged SFT-LoRA adapter), not from the raw base model, and attaches a
fresh LoRA adapter for this stage -- standard SFT-then-DPO chaining.

data/dpo_pairs_zeroshot.jsonl rows have "system"/"user"/"chosen"/"rejected"
but deliberately no pre-assembled "prompt" (see that file's build script
docstring): built here via tok.apply_chat_template([system, user], ...,
add_generation_prompt=True), the exact same call evaluate_redesign_v2.py
uses at eval time, so train- and eval-time prompt formatting are
guaranteed identical byte-for-byte rather than merely "the same words".

Run on Colab GPU, after train_lora_sft.py has produced ../results/sft-merged-model:

    pip install -q peft trl transformers accelerate bitsandbytes datasets sympy
    python train_lora_dpo_zeroshot.py
"""
import json

import torch
from datasets import Dataset
import config as avs_config
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # for the tokenizer only
SFT_MERGED_DIR = "../results/sft-merged-model"  # train_lora_sft.py's output
# v1 (the default) is the file every recorded run trained on. It silently
# contains no no-solution problems (PROVENANCE.md, bug 8); the fixed v2 file is
# selected with AVS_DATA_PATH=../data/dpo_pairs_zeroshot_v2.jsonl.
DATA_PATH = avs_config.get("data_path", "../data/dpo_pairs_zeroshot.jsonl")
OUTPUT_DIR = "../results/lora-dpo-zeroshot-adapter"

# Training seed, passed explicitly rather than left to the library default.
# Be aware of what this does and does not buy: it fixes data order and
# dropout draws, but it does NOT make GPU training bit-reproducible, because
# CUDA matmul and reduction kernels are not deterministic by default. Two
# runs of this script on identical data measurably differ -- observed
# directly: four runs produced held-out solve rates of 0.444, 0.378, 0.378
# and 0.311 (see results/colab_run/redesign_v2.md). A single run of this script is
# therefore one draw from a distribution, not "the" result; use
# run_dpo_seed_sweep.py to characterise the spread before quoting a number.
SEED = avs_config.get("seed", 0)


def build_prompt(tok, system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_dataset(path, tok):
    rows = [json.loads(l) for l in open(path)]
    out = [
        {
            "prompt": build_prompt(tok, r["system"], r["user"]),
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }
        for r in rows
    ]
    return Dataset.from_list(out)



def require_gpu(script_name):
    """Fail early and legibly when there is no GPU.

    Without this the failure surfaces from deep inside transformers as
    "Your setup doesn't support bf16/gpu", which points at a dtype setting
    rather than at the actual cause: the Colab runtime is not attached to a
    GPU. This task needs one -- a 0.5B model trains in minutes on a T4 and
    is impractically slow on Colab's two CPU cores.
    """
    import sys
    if not torch.cuda.is_available():
        print(f"\n{script_name}: no GPU detected.\n", file=sys.stderr)
        print("  In Colab: Runtime -> Change runtime type -> T4 GPU, then re-run.", file=sys.stderr)
        print("  Verify with `!nvidia-smi` before continuing.", file=sys.stderr)
        print("  Changing the runtime restarts it, so the upload/install cells", file=sys.stderr)
        print("  need re-running too.\n", file=sys.stderr)
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)


def main():
    require_gpu("train_lora_dpo_zeroshot.py")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(SFT_MERGED_DIR)

    ds = load_dataset(DATA_PATH, tokenizer).train_test_split(test_size=0.15, seed=0)

    # Resolved through avs_config: env var > AVS_CONFIG yaml > the literal here,
    # which is what the four recorded runs used. See train_lora_sft.py.
    lora_config = LoraConfig(
        r=avs_config.get("lora_r", 16),
        lora_alpha=avs_config.get("lora_alpha", 32),
        lora_dropout=avs_config.get("lora_dropout", 0.05),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=avs_config.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=avs_config.get("gradient_accumulation_steps", 4),
        num_train_epochs=avs_config.get("num_train_epochs", 3),
        learning_rate=avs_config.get("learning_rate", 5e-5),
        beta=avs_config.get("beta", 0.1),
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
        seed=SEED,
        data_seed=SEED,
    )
    avs_config.dump(f"{OUTPUT_DIR}/resolved_config.json")

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"LoRA-DPO (zero-shot-format) adapter saved to {OUTPUT_DIR}")
    print(trainer.evaluate())


if __name__ == "__main__":
    main()
