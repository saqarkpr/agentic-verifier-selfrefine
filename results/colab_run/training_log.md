# Colab LoRA-DPO training run (real, executed on a T4 GPU)

Run date: 2026-09-14/15, Google Colab, T4 GPU, `Qwen/Qwen2.5-0.5B-Instruct`.

## Data used for this run

This run trained on **196 preference pairs** generated inside the Colab
kernel by the (then-unfixed) `generate_problems()`, which seeded its
per-family RNG with `hash(family) % 997` -- a value salted per Python
process. The sandbox run of the identical code produced 197 pairs. Both
counts are real outputs of the same source code running in two different
processes; the difference is the nondeterminism bug, not a data error. The
bug is now fixed in `src/tasks.py` (fixed per-family offsets instead of
`hash()`), and re-running the pipeline after the fix produces 180 pairs
deterministically, every time. See PROVENANCE.md for the full account.
**The adapter and eval numbers below are from the original 196-pair Colab
run**; they were not regenerated after the fix, since that would mean
re-running the GPU training, which honesty about the fix does not require.

## Training

`peft` LoRA (r=16, alpha=32, dropout=0.05, target_modules=[q_proj, k_proj,
v_proj, o_proj]) + `trl` `DPOTrainer`, 3 epochs, batch size 4 x grad accum
4, lr 5e-5, beta 0.1. Wall-clock: 205.4s on a T4.

| Epoch | Train loss | Eval loss |
|-------|-----------:|----------:|
| 1     | 0.501367   | 0.386461  |
| 2     | 0.287273   | 0.217145  |
| 3     | 0.175505   | 0.160955  |

Final `trainer.evaluate()`: `eval_loss = 0.1609552949666977` (matches the
epoch-3 val loss above). The loss curve is monotonic and healthy -- the
model is fitting the preference pairs as intended.

Adapter saved to `lora-dpo-adapter/` (LoRA weights only, not committed to
this repo -- see PROVENANCE.md for why).

## Held-out zero-shot evaluation (the actual research question)

Question: does LoRA-DPO fine-tuning on agent-loop-derived preference pairs
let the base model solve more problems **zero-shot, in a single call**,
without the multi-turn agent loop?

45 held-out problems (`generate_problems(n_per_family=15, seed=99)`,
drawn with a different seed from the seed-0 training problems -- though see
PROVENANCE.md bug #7: a different seed is not a disjoint set), greedy decoding,
plain "Solve for x: {equation}" prompt (no agent loop, no verifier
feedback at inference time):

| Model              | Zero-shot solve rate |
|--------------------|----------------------:|
| Base (no fine-tune)| 0.022 (1/45)          |
| LoRA-DPO fine-tuned| 0.000 (0/45)          |
| Delta              | -0.022                |

## Honest reading of this result

This is a **null result, not a positive one**: fine-tuning did not improve,
and if anything slightly lowered, zero-shot solve rate. With n=45 and a
1-vs-0 count, this delta is not distinguishable from noise on its own --
but the more important, structural explanation is a **prompt-distribution
mismatch between training and evaluation**:

- Every DPO training pair uses the *revision* prompt template: equation +
  "Previous attempt: ..." + "Verifier feedback: ..." + "Give a corrected
  answer...". The model was never trained on a bare "Solve for x: ..."
  prompt with no prior attempt to react to.
- The zero-shot eval prompt is exactly that bare, unconditioned prompt --
  the one case the fine-tuning data never covered.

So this experiment does not show "agentic self-refinement cannot be
distilled into a single call" -- it shows that DPO-tuning on
revision-conditioned pairs does not, by itself, transfer to an
unconditioned zero-shot prompt, which is close to expected given the data.
A fairer test of the original question would fine-tune on (or evaluate
with) matched prompt formats -- e.g. DPO pairs built from turn-1 wrong
attempts to a zero-shot chosen answer, not from turn-k to turn-(k+1). That
is a concrete, correctly-scoped follow-up this run motivates, not a
contradiction of the turn-budget result in Section 4 of the report (which
is about a synthetic solver's error-recovery dynamics, not this small
real-LLM's zero-shot competence).
