"""
Colab-side evaluation: does LoRA-DPO fine-tuning (on agent-derived
revision pairs) let the model solve MORE problems zero-shot (single
attempt, no agent loop, no verifier feedback at inference time) than the
base model?

This is a faithful, standalone reproduction of the exact script actually
run on a Colab T4 GPU (pasted directly into the notebook during a live
browser session, not run via this file -- see PROVENANCE.md). It
deliberately does NOT reuse hf_solver.attempt(): that function samples
(do_sample=True, temperature=0.7) because it is meant for the agent loop
and for generating diverse real-LLM error trajectories, whereas a
solve-rate *evaluation* should be deterministic. Using greedy decoding
here, on the exact same bare "Solve for x: ..." prompt as the training
data's zero-shot case, is what makes the result in
results/colab_run/training_log.md reproducible rather than a moving
target.

n_per_family=15 (45 total) and seed=99 match the actual run: 45 held-out
problems, drawn independently of the seed=0 problem set used for the training
pairs. Result (from the real run): base=0.022 (1/45), fine-tuned=0.000
(0/45) -- a null result, most likely explained by a prompt-format
mismatch between the revision-conditioned training pairs and this
unconditioned zero-shot prompt (see training_log.md for the full
argument, not attempted here as a post-hoc rescue of the number).
"""
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import splits
from tasks import generate_problems
from verifier import verify

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "../results/lora-dpo-adapter"
HELD_OUT_SEED = 99  # a different seed from training, but NOT a disjoint set: see tasks.py

# The original run left the model on CPU (from_pretrained does not move it
# to the GPU on its own), which is why it took so long. Added afterwards as
# a pure speed fix: float32 on GPU is the same arithmetic as float32 on CPU
# under greedy decoding, so the reported 0.022 / 0.000 numbers stand.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def hf_attempt(model, tok, equation_str):
    prompt = (
        f"Solve for x: {equation_str}\n"
        "Give the final answer on its own line as 'x = v1, v2' or 'x = no solution'."
    )
    messages = [
        {"role": "system", "content": "You are a careful algebra solver. Show brief work, then give the final answer line."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.findall(r"x\s*=\s*([^\n]+)", decoded)
    if m:
        return f"x = {m[-1]}"
    # An empty generation is NOT a claim of "no solution". Returning that
    # string handed the model a free correct answer on the 8 of 45 held-out
    # problems that have none; "" scores as unparsable, which is what an
    # unreadable answer is. See PROVENANCE.md, bug #6.
    return decoded.strip().splitlines()[-1] if decoded.strip() else ""


def eval_rate(model, tok, problems, label):
    model.eval()
    correct = 0
    for i, p in enumerate(problems, 1):
        cand = hf_attempt(model, tok, p["equation_str"])
        if verify(p["equation_str"], cand).correct:
            correct += 1
        if i % 5 == 0 or i == len(problems):
            print(f"  [{label}] {i}/{len(problems)} done, {correct} correct so far", flush=True)
    return correct / len(problems)


def main():
    # The canonical benchmark, from splits.held_out(). Constructing it here
    # with a bare generate_problems() is how six held-out equations ended up
    # in the SFT training file; splits.py is the single definition and the
    # builders exclude it by construction. See PROVENANCE.md, bug 7.
    held_out_problems = splits.held_out()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(DEVICE)
    print(f"Running on: {DEVICE}", flush=True)

    print("Evaluating BASE model zero-shot...", flush=True)
    base_rate = eval_rate(base_model, tok, held_out_problems, "base")
    print(f"Base model zero-shot solve rate: {base_rate:.3f}", flush=True)

    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(DEVICE)
    print("Evaluating FINE-TUNED model zero-shot...", flush=True)
    ft_rate = eval_rate(ft_model, tok, held_out_problems, "fine-tuned")
    print(f"LoRA-DPO fine-tuned zero-shot solve rate: {ft_rate:.3f}", flush=True)

    print(f"\nHeld-out set: {len(held_out_problems)} problems (seed={HELD_OUT_SEED}; NOT disjoint from training -- 6 of 45 also appear in sft_pairs.jsonl, see PROVENANCE.md bug #7)")
    print(f"Base:       {base_rate:.3f}")
    print(f"Fine-tuned: {ft_rate:.3f}")
    print(f"Delta:      {ft_rate - base_rate:+.3f}")


if __name__ == "__main__":
    main()
