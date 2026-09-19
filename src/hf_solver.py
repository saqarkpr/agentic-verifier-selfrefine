"""
Real-LLM solver, same
solve_fn(equation_str, rng, severity, prior_feedback=None) interface as
flawed_solver.attempt(), so it drops into agent_loop.run_all /
build_preference_data.build_pairs unchanged. This is for the (optional)
"regenerate DPO pairs from a real LLM's own trajectories" path
(notebooks/colab_train.ipynb cell 3) -- it samples (do_sample=True,
temperature=0.7) on purpose, to get a varied, realistic spread of error
trajectories to learn from, the same way the actually-run flawed_solver
uses a random perturbation rather than one fixed wrong answer.

Unlike flawed_solver, this one genuinely uses `prior_feedback`: it builds a
revision-style prompt so turn 2/3 is an actual revision of the previous
wrong attempt rather than an independent re-guess. (agent_loop.py used to
collect that feedback and never pass it on -- see PROVENANCE.md, bug 2.)

Not run inside the CPU sandbox this repo was built in (no GPU there);
intended for Colab, exactly like the RLVF and biomedical-rag-eval
generators. `severity` is accepted for interface compatibility but ignored
here -- a real model's error rate isn't something we dial, it's what we're
measuring. `rng` IS used, to seed sampling per call: without that, this
path samples from torch's global RNG and the "same seed" passed to
agent_loop.run_all would not actually reproduce the same trajectories --
the same reproducibility trap as the seeding bug in PROVENANCE.md, bug 1.

For the base-vs-fine-tuned *evaluation* (where a deterministic, single
number is the point), see evaluate_finetuned.py, which uses its own
greedy-decoding (do_sample=False) generation function rather than this
one -- that is the function that actually produced the numbers in
results/colab_run/training_log.md.
"""
import re

_MODEL = None
_TOKENIZER = None
_LOADED_NAME = None


def _load(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    global _MODEL, _TOKENIZER, _LOADED_NAME
    # Keyed on model_name, not just "is it loaded": caching on `_MODEL is
    # None` alone would silently keep serving the first model if a caller
    # later asked for a different one.
    if _MODEL is None or _LOADED_NAME != model_name:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            # `dtype`, not `torch_dtype`: the latter is deprecated in current
            # transformers and warns on every load (it did in the real Colab
            # run -- see results/colab_run/training_log.md).
            model_name, dtype=torch.bfloat16, device_map="auto"
        )
        _LOADED_NAME = model_name
    return _MODEL, _TOKENIZER


def attempt(equation_str, rng, severity, model_name="Qwen/Qwen2.5-0.5B-Instruct", prior_feedback=None):
    import torch

    model, tok = _load(model_name)
    # Derive this call's sampling seed from the agent loop's own rng, so a
    # trajectory run with seed=k is actually reproducible. torch.manual_seed
    # is global, which is fine here: generation is sequential, one call at a
    # time, and re-seeding per call is what makes the sequence deterministic.
    torch.manual_seed(rng.randrange(2**31))

    prompt = f"Solve for x: {equation_str}\n"
    if prior_feedback:
        prompt += f"Previous attempt was wrong. Verifier feedback: {prior_feedback}\n"
    prompt += "Give the final answer on its own line as 'x = v1, v2' or 'x = no solution'."

    messages = [
        {"role": "system", "content": "You are a careful algebra solver. Show brief work, then give the final answer line."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
    decoded = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    m = re.findall(r"x\s*=\s*([^\n]+)", decoded)
    return f"x = {m[-1]}" if m else decoded.strip().splitlines()[-1] if decoded.strip() else "no solution"
