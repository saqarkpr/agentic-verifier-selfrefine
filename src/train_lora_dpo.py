"""
LoRA-DPO fine-tuning on the agent-derived preference pairs (data/dpo_pairs.jsonl).

Run on Colab GPU (T4 is enough for a 0.5B model), same workflow as the
RLVF project -- not executable inside this repo's CPU sandbox. This has
actually been run: see results/colab_run/training_log.md for the real
loss curve and PROVENANCE.md for the full account (including a data
caveat: that run used 196 pairs from a since-fixed non-deterministic
version of tasks.py, not the 180 pairs currently in data/dpo_pairs.jsonl).
notebooks/colab_train.ipynb is a ready-to-run notebook version of this
same script for reproducing it.

    pip install -q peft trl transformers accelerate bitsandbytes datasets
    python train_lora_dpo.py

Net-new skill vs. RLVF (which did full-parameter DPO): this uses LoRA
adapters (peft), so it's a much smaller artifact to ship/evaluate and is
the standard way small models get post-trained in production LLM/MLOps
work.
"""
import json

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "../data/dpo_pairs.jsonl"
OUTPUT_DIR = "../results/lora-dpo-adapter"


def load_dataset(path):
    rows = [json.loads(l) for l in open(path)]
    return Dataset.from_list(rows)


def main():
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

    config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        beta=0.1,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
    )

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
    print(f"LoRA adapter saved to {OUTPUT_DIR}")
    print(trainer.evaluate())


if __name__ == "__main__":
    main()
