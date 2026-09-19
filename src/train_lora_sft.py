"""
Stage 1 of the "redesign v2" fine-tuning pipeline: LoRA-SFT warm-start on
data/sft_pairs.jsonl, using the EXACT prompt format (system + user message,
apply_chat_template with add_generation_prompt=True) that
evaluate_redesign_v2.py tests zero-shot at the end. See PROVENANCE.md
("A third bug found and fixed") and README.md for why this second attempt
at the distillation experiment exists: the first LoRA-DPO run (see
results/colab_run/training_log.md) trained only on the *revision* prompt
template and was evaluated on the *zero-shot* prompt, a mismatch that is
the leading explanation for that run's null result. This script does not
fix that by itself; it is one half of a two-stage fix (SFT here, then
train_lora_dpo_zeroshot.py on top of this stage's output), both of which
train and are evaluated on the same zero-shot prompt shape.

Why add an SFT stage at all, not just re-run DPO with fixed data: the base
model's own zero-shot solve rate was measured at 2.2% (1/45, see
results/colab_run/training_log.md). DPO only teaches a *preference*
between two already-plausible responses; with a base rate this low, the
model rarely produces anything DPO can push towards "more right". SFT on
correct worked solutions teaches the underlying capability directly, the
standard first stage of an SFT-then-DPO pipeline.

Run on Colab GPU (T4 is enough for a 0.5B model), same pattern as
train_lora_dpo.py:

    pip install -q peft trl transformers accelerate bitsandbytes datasets sympy
    python train_lora_sft.py

Output: this script does two things, not just one --
  1. trains a LoRA adapter and saves it to OUTPUT_DIR (for inspection /
     the record), and
  2. merges that adapter into the base model's weights with
     `merge_and_unload()` and saves the result as a full model to
     MERGED_DIR, because train_lora_dpo_zeroshot.py's DPO stage needs a
     plain AutoModelForCausalLM to attach its own, separate LoRA adapter
     to -- DPOTrainer's peft_config path expects to add LoRA to a base
     model, not to stack a second adapter on top of a PeftModel. Merging
     is the standard way to chain LoRA stages (SFT-LoRA -> merge -> fresh
     DPO-LoRA), used so each stage's adapter stays small and inspectable
     on its own, while the next stage still starts from a normal model.
"""
import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "../data/sft_pairs.jsonl"
OUTPUT_DIR = "../results/lora-sft-adapter"
MERGED_DIR = "../results/sft-merged-model"

# See the note in train_lora_dpo_zeroshot.py: this fixes data order and
# dropout, but GPU kernels are not deterministic, so runs still differ
# slightly. In practice the SFT stage has been the stable one (0.289
# held-out on repeated runs); the DPO stage is where the variance showed up.
SEED = 0


def load_dataset(path):
    rows = [json.loads(l) for l in open(path)]
    # SFTTrainer auto-detects the conversational "messages" format and
    # applies the tokenizer's own chat template internally -- the same
    # template evaluate_redesign_v2.py calls at eval time via
    # tok.apply_chat_template(..., add_generation_prompt=True) -- so the
    # training target and the eval-time prompt stay byte-for-byte
    # consistent without this script re-implementing the template.
    return Dataset.from_list([{"messages": r["messages"]} for r in rows])



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
    require_gpu("train_lora_sft.py")
    ds = load_dataset(DATA_PATH).train_test_split(test_size=0.15, seed=0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
        max_length=512,
        packing=False,
        assistant_only_loss=True,
        seed=SEED,
        data_seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"LoRA-SFT adapter saved to {OUTPUT_DIR}")
    print(trainer.evaluate())

    print("Merging SFT adapter into base weights...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Merged SFT model saved to {MERGED_DIR} (input for train_lora_dpo_zeroshot.py)")


if __name__ == "__main__":
    main()
