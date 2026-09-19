"""
A synthetic 'solver' that stands in for a small LLM's error distribution.

WHY A SYNTHETIC MODEL, DOCUMENTED HONESTLY (see README/PROVENANCE):
This sandbox has no GPU (2 vCPU, no CUDA). A real small causal LM
(Qwen2.5-0.5B-Instruct, matching the RLVF project) can run here for
inference, but a full LoRA-DPO training loop over enough trajectories to
mean anything is a multi-hour CPU job. Rather than claim a "real" training
run that is actually a 3-example toy, this repo separates two honestly-
labeled things:
  1. A synthetic flawed solver (this file) with realistic, seeded error
     modes -- used to validate the agent-loop + verifier infrastructure
     end-to-end, produce real trajectories, and run the "does agentic
     revision improve accuracy" experiment for real, on CPU, in this repo.
  2. A LoRA-DPO training script (train_lora_dpo.py) written against a real
     HF causal LM + trl, run on a Colab GPU exactly like the RLVF
     project's fine-tuning was (not runnable inside this CPU sandbox).
     Real results: results/colab_run/training_log.md and PROVENANCE.md.

The flawed solver's error modes are exactly the ones small LLMs actually
make on this task (per the verifier's own feedback categories):
  - forgetting a domain restriction (extraneous root)
  - dropping one root of a quadratic
  - arithmetic slip (off-by-something wrong value)
  - occasionally just correct
Error probabilities *decrease* with each verifier-feedback turn it receives,
modeling partial in-context correction -- the same behavior a real LLM shows
when given specific structured feedback rather than a bare "wrong".
"""
import sympy as sp

from verifier import _true_solution_set


def _perturb(true_set, domain, rng, severity):
    """Return a plausible wrong answer, or the true answer if 'lucky'."""
    # sorted(..., key=str), NOT list(): iterating a Python set of sympy
    # objects returns them in hash order, and a sympy expression's hash
    # ultimately hashes strings, which Python salts per process
    # (PYTHONHASHSEED). For the 28 quadratics here with irrational roots,
    # that made list(true_set)[0] a different root in different processes,
    # so "drop a root" and "perturb one value" produced different candidate
    # strings on every run of identical, fully-seeded code. Solve *rates*
    # were unaffected (the verifier compares sets, so dropping either root
    # is equally wrong), which is exactly why this hid for so long -- but
    # trajectories.csv and the DPO pair counts were not reproducible. Same
    # root cause as the seeding bug in PROVENANCE.md, different place.
    # key=str matches how verifier.py and reference_solution.py already
    # order solution sets.
    true_set = sorted(true_set, key=str)
    roll = rng.random()
    if roll < max(0.05, 0.55 - 0.18 * (severity - 1)):
        # get it right
        return true_set
    if domain and rng.random() < 0.4:
        # forget the domain restriction: include the excluded point
        bad = sorted(domain, key=str)[0]
        return true_set + [bad]
    if len(true_set) >= 2 and rng.random() < 0.5:
        # drop a root
        return true_set[:1]
    if true_set:
        # arithmetic slip: perturb one value
        v = true_set[0]
        return [v + rng.choice([1, -1, sp.Rational(1, 2), -sp.Rational(1, 2)])]
    return []


def attempt(equation_str, rng, severity, prior_feedback=None):
    """severity: 3 = first attempt (most errors), 1 = after 2 rounds of feedback.

    prior_feedback is accepted for interface parity with hf_solver.attempt()
    (agent_loop.run_trajectory passes it to every solve_fn) but intentionally
    ignored here: this synthetic solver models "feedback helps" through the
    severity schedule alone, not by reading the feedback text.
    """
    true_sol, domain = _true_solution_set(equation_str)
    true_set = set(true_sol) if isinstance(true_sol, sp.FiniteSet) else set()
    guess = _perturb(true_set, domain, rng, severity)
    if not guess:
        return "no solution"
    return ", ".join(f"x = {g}" for g in guess)
