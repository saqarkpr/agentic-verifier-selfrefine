# Redesign v2: prompt-matched SFT + DPO

Status: **run end to end on a Colab T4 GPU, four times.** All runs are
reported below, including the fact that they did not agree. The headline:
verifier-only training takes Qwen2.5-0.5B-Instruct from 0/45 on held-out
problems to 14-20/45 (mean 17/45). Four runs give a variance estimate, so
the DPO increment can finally be stated with an error bar rather than as a
single number.

## Why this exists

The first LoRA-DPO run (`training_log.md` in this same directory) gave a
null result: base model zero-shot solve rate 0.022 (1/45), fine-tuned
0.000 (0/45). The leading explanation there is a prompt-format mismatch:
every DPO training pair used the *revision* prompt template (equation +
prior wrong attempt + verifier feedback), but the held-out evaluation used
a bare, unconditioned zero-shot prompt. The model was never trained on the
prompt shape it was tested on, so a null transfer result is close to what
you'd expect from a mismatched objective, not necessarily evidence against
"does agentic-revision-derived preference data improve a model."

That run's own writeup proposed the fair follow-up rather than attempting
a post-hoc rescue: build DPO pairs from turn-1-wrong to a zero-shot-shaped
chosen answer, and evaluate on that same shape. This is that follow-up,
done properly -- not just fixed DPO data, but an SFT warm-start first,
because the base model's zero-shot solve rate (2.2%) is so low that DPO
alone (a *preference* signal between two already-plausible responses) has
very little to work with.

## Pipeline

1. **`src/reference_solution.py`** -- deterministic, template-based worked
   solutions, honestly labeled as such (not LLM-generated): reasoning text
   plus a final answer line sourced from `verifier._true_solution_set`, so
   correctness is guaranteed by construction. This is what makes an SFT
   target and a DPO "chosen" response available at all without a stronger
   teacher model.
2. **`src/build_sft_data.py`** -> `data/sft_pairs.jsonl` (300 examples,
   100/family, seed=0): `[system, user, assistant]` conversational rows in
   the exact prompt shape the eval uses.
3. **`src/build_dpo_pairs_zeroshot.py`** -> `data/dpo_pairs_zeroshot.jsonl`
   (118 pairs; see `PROVENANCE.md` for the severity-schedule fix and the
   sympy set-ordering fix that between them determine this count -- the
   DPO stage already run on Colab used the pre-fix 123-pair version of
   this file, and is reported as such below):
   chosen = correct worked solution, rejected = same steps text with a
   wrong final answer line (from `flawed_solver`, severity=3), so DPO
   learns "is the final answer right", not "is the response longer."
   `system`/`user` kept separate rather than a pre-assembled `prompt`, so
   the training script builds the prompt via
   `tok.apply_chat_template(..., add_generation_prompt=True)` -- the same
   call the eval script uses -- rather than risking two hand-written
   copies of the same template drifting apart.
4. **`src/train_lora_sft.py`** -- LoRA-SFT on `sft_pairs.jsonl`
   (`trl.SFTTrainer`, conversational format, `assistant_only_loss=True` so
   the loss is computed only on the assistant's response, not the
   system/user turns), then merges the adapter into the base weights
   (`merge_and_unload()`) and saves a full model to
   `results/sft-merged-model/`.
5. **`src/train_lora_dpo_zeroshot.py`** -- loads `sft-merged-model/`,
   attaches a *fresh* LoRA adapter, and trains DPO on
   `dpo_pairs_zeroshot.jsonl`, building each prompt at load time via the
   chat template (never a hand-typed string). Saved to
   `results/lora-dpo-zeroshot-adapter/`.
6. **`src/evaluate_redesign_v2.py`** -- same held-out set as the first run
   (45 problems, seed=99, greedy decoding, identical regex answer
   extraction), but reports **three** numbers instead of two: Base,
   SFT-only, and SFT+DPO. The SFT-only number is deliberately isolated so
   that if the final result disappoints, it's possible to say which stage
   did or didn't help, instead of only having one before/after pair.

## What is and isn't controlled for, honestly

- Both training stages and the evaluation now use byte-for-byte the same
  chat-template call (`add_generation_prompt=True`), closing the specific
  gap the first run's null result was attributed to.
- `sft_pairs.jsonl` (100/family) and `dpo_pairs_zeroshot.jsonl`
  (60/family, before the "chosen==rejected" dedup) are both drawn from
  `generate_problems(seed=0)`, and since that generator is a deterministic
  prefix, the first 60 problems per family are literally the same
  equations in both files. This means the DPO stage's prompts were also
  seen (with the correct answer) during SFT -- both training-side, no
  leakage into the seed=99 held-out set, but worth stating plainly rather
  than leaving it implicit.
- The reasoning text in every "chosen" (and matching "rejected") response
  is templated, not LLM-generated -- see `src/reference_solution.py`'s
  docstring. A 0.5B model trained on this will likely pick up the
  template's exact phrasing rather than learning to reason flexibly; this
  is a legitimate way to seed a narrow, exactly-checkable task, not a
  claim that the model has learned open-ended algebra explanation.
- This is still a small model (0.5B) on a narrow task family, evaluated on
  a small held-out set (45 problems); a single-digit change in raw solve
  count moves the reported rate by more than two points.

## Training runs (real, executed on a Colab T4 GPU)

Both training stages ran on a T4, four times. The per-epoch tables below
are from run A; the later runs were close but never identical (SFT eval loss
0.1105, 0.1071, 0.1071, ...; DPO eval loss 0.2401, 0.2194, 0.2194, ...),
which is the non-determinism discussed under the evaluation result.

### Stage 1: LoRA-SFT (`train_lora_sft.py`)

300 examples, split 255 train / 45 eval, 3 epochs, LoRA r=16 alpha=32
dropout=0.05 on [q_proj, k_proj, v_proj, o_proj], lr 1e-4, batch 4 x grad
accum 4, `assistant_only_loss=True`. Wall-clock 144s.

| Epoch | Train loss | Eval loss | Eval token accuracy |
|-------|-----------:|----------:|--------------------:|
| 1     | 0.903      | 0.6434    | 0.810               |
| 2     | 0.254      | 0.1649    | 0.955               |
| 3     | 0.111      | 0.1105    | 0.966               |

Monotonic and healthy. Adapter merged into the base weights and saved to
`results/sft-merged-model/`.

### Stage 2: LoRA-DPO on the SFT model (`train_lora_dpo_zeroshot.py`)

123 pairs (the pre-set-ordering-fix count -- see the note above), split
104 train / 19 eval, 3 epochs, same LoRA config, lr 5e-5, beta 0.1.
Wall-clock 188s.

| Epoch | Train loss | Eval loss | Eval reward accuracy | Eval reward margin |
|-------|-----------:|----------:|---------------------:|-------------------:|
| 1     | 0.583      | 0.3367    | 0.958                | 1.082              |
| 2     | 0.372      | 0.2290    | 0.958                | 1.814              |
| 3     | 0.201      | 0.2135    | 0.958                | 2.069              |

Reward accuracy (how often the model prefers `chosen` over `rejected`) is
high and the margin grows steadily, so the DPO objective is being
optimized as intended. Note this says the model learned the *preference*;
whether that raises zero-shot solve rate is the separate question the
evaluation below answers.

## Held-out evaluation result (real, executed on a Colab T4 GPU)

45 held-out problems, seed=99, disjoint from all training data. Greedy
decoding, identical prompt format at train and eval time.

The pipeline has now been run **four** times, end to end, on identical data.
All four are reported.

| Model | Run A | Run B | Run C | Run D |
|-------|------:|------:|------:|------:|
| Base (Qwen2.5-0.5B-Instruct) | 0.000 (0/45) | 0.000 (0/45) | 0.000 (0/45) | 0.000 (0/45) |
| + LoRA-SFT | 0.289 (13/45) | 0.289 (13/45) | 0.289 (13/45) | 0.289 (13/45) |
| + LoRA-SFT + LoRA-DPO | 0.444 (20/45) | 0.378 (17/45) | 0.378 (17/45) | 0.311 (14/45) |

**Four runs is a variance estimate, so the numbers can finally be stated
properly:**

| quantity | mean | sd | range |
|---|---:|---:|---|
| SFT-only | 0.289 | 0.000 | 13/45 every single time |
| SFT+DPO | 0.378 | 0.054 | 0.311 - 0.444 |
| DPO gain over SFT | +0.089 | 0.054 | +1 to +7 problems |

A one-sample t-test on the four gains gives t=3.27, df=3, p=0.047. That is
positive, small, and *barely* significant: one more run landing near run D
would push it past 0.05. The defensible claim is that DPO adds roughly four
problems out of 45 on average, with a spread nearly as large as the effect.

Two things this settles that earlier runs could not:

1. **Run A's 0.444 was an outlier**, 1.22 sd above the mean of four runs. An
   earlier version of this repo reported it as *the* result. Reporting the
   best of several runs as the headline would have overstated the effect by
   roughly a factor of two.
2. **The SFT stage is extraordinarily stable** -- 13/45, the same 13
   problems, in all four runs -- while the DPO stage's output ranges over
   6 percentage points. Whatever DPO is doing here, it is not a robust
   transformation of the model.

Note that this was not obtained from `run_dpo_seed_sweep.py`. It came from
four independent end-to-end reruns that happened for unrelated reasons
(dead sessions, re-uploads). It is a *weaker* design than the deliberate
sweep, because the seeds were incidental rather than controlled, but four
independent draws is four independent draws, and it is what the numbers
above rest on.

### Why the runs disagree

Ordinary GPU training non-determinism. CUDA matmul and reduction kernels
are not deterministic by default, so repeated runs of identical code on
identical data follow slightly different numerical paths and land in
different minima. (`seed` and `data_seed` are now passed explicitly in both
training scripts, which fixes data order and dropout but does not make CUDA
kernels deterministic, so it narrows this without eliminating it.)

The SFT stage was stable: 0.289 on all four runs, the same 13 problems. The
variance is in the DPO stage, which trains on only 100 pairs for 21
optimizer steps, so it is the more fragile of the two.

One detail worth recording, because it is the opposite of what one would
assume. Here are the DPO stage's own training metrics against the held-out
solve rate they produced:

| run | DPO eval loss | reward accuracy | reward margin | **held-out solve** |
|---|---:|---:|---:|---:|
| A | 0.2401 | 0.7917 | 1.816 | **0.444** (20/45) |
| B | 0.2194 | 0.8333 | 1.872 | **0.378** (17/45) |
| D | 0.2325 | 0.8333 | 1.870 | **0.311** (14/45) |

Look at B against D. Identical reward accuracy (0.8333 both), reward margins
differing by 0.002, eval losses differing by 0.013 -- and a **three-problem
gap** in the only quantity anyone cares about. Run A, meanwhile, was the
*worst* of the three on every DPO training metric and the best on solve rate.
Across these three runs the correlation between DPO eval loss and solve rate
is +0.36, i.e. lower loss went with *worse* held-out performance.

With n=3 that correlation is not itself evidence of anything. The B-versus-D
pair is the part that carries weight: two checkpoints indistinguishable on
every training metric available, three problems apart on the test set.

The practical consequence is concrete. **Selecting a checkpoint by
validation loss or reward accuracy would have been no better than choosing
at random here, and in run A's case actively worse.** The DPO objective
measures whether the model prefers the chosen response to the rejected one;
it does not measure whether the model can solve an equation. On a task where
those two can be checked against each other, they come apart.

`src/run_dpo_seed_sweep.py` trains the DPO stage under several seeds against
the fixed SFT model and reports mean, standard deviation and range, which is
what should be quoted once it has been run.

**This is a positive result, and it answers the question the first run's
null result left open.** The redesign changed one thing of substance --
training on the same prompt format the evaluation uses -- and the
distillation that previously did nothing now moves the model from solving
none of the held-out problems to solving roughly two fifths of them. Two effects of very
different evidential strength are bundled in that sentence, and they should
not be quoted with the same confidence:

- **Base to SFT is solid.** 0/45 to 13/45, identical on all four runs. A
  13-problem gap on a model that otherwise solves nothing is not a
  candidate for training noise.
- **SFT to DPO is positive but small and fragile.** All four runs put
  SFT+DPO above SFT-only, by 7, 4, 4 and 1 problems. Mean +0.089 with an sd
  of 0.054 and p=0.047 on four runs: real enough to report, small enough
  that its spread is nearly its own size, and fragile enough that one more
  low run would take it past p=0.05. It stays
  an observation until `run_dpo_seed_sweep.py` has been run.

Two things are worth separating carefully:

1. **SFT did the heavy lifting** (+0.289). That is expected: the base model
   at 0/45 cannot do the task at all, and SFT teaches the capability
   directly from correct worked solutions.
2. **DPO added a further +0.09 to +0.16 on top of a model that had already
   been SFT'd on the same problems.** This is the more interesting effect,
   because it is not explained by "the model saw correct answers": the SFT
   stage had already seen correct worked solutions for these same 60
   problems per family. What DPO adds is the contrast against a *wrong*
   final answer of the same shape, which is exactly the signal the
   verifier-derived preference pairs carry. The three-way ablation is what
   makes this separable; a base-vs-final comparison alone could not
   distinguish the two stages.

The base model measuring 0.000 here versus 0.022 (1/45) in the first run is
the same model and the same problems. The single problem it got right
before was a coin flip on a task it cannot do, and it is not a meaningful
difference.

### An open question about the base=0 baseline

Raised in review and not yet settled: **what does 0/45 actually measure?**

The evaluation prompt and system message are specific to this task and this
output format. The base model was never trained on either. So "base 0/45 ->
SFT 13/45" admits two readings:

- fine-tuning taught the model to *solve* equations it could not solve, or
- fine-tuning taught it to *emit answers in the shape the extractor reads*,
  on a task it had some grip on already.

The evaluation counted correct answers and nothing else, so it could not
distinguish these, and earlier write-ups in this repo implied the first
reading without having tested it. That was an overreach in the same family
as the turn-budget overclaim: measuring capability-plus-format and
describing it as capability.

Two things are now in place. `evaluate_redesign_v2.py` records the verifier's
reason code for every attempt, so any future run reports whether failures
were `unparsable` (format) or `wrong`/`incomplete` (capability).
`src/inspect_base_outputs.py` dumps the base model's raw generations beside
the extracted answer and the verdict, which settles it directly.
**Neither has been run yet, so the question stays open and is flagged here
rather than resolved by assertion.**

One piece of evidence does bear on it. Across five evaluations the base model
scored 1/45 once (the first run, on CPU) and 0/45 four times (all on GPU).
Greedy decoding is deterministic, so that difference is float
non-determinism between devices flipping a single token. Whatever the
mechanism, the base model sits at the floor.

### A note on why the DPO gain is small

Also raised in review, and it fits the data. In `dpo_pairs_zeroshot.jsonl`
the chosen and rejected responses share *identical* reasoning text and
differ only in the final answer line. That design deliberately isolates
answer correctness from length and style, and it worked for that purpose.
But it also makes the training signal narrow: the model is being taught
which final answer to prefer, not a different way to reach one. A gain of
about four problems is consistent with the signal on offer, and the
reward-accuracy-versus-solve-rate gap above is the same point seen from the
other side -- the model learned the preference well (0.83 reward accuracy)
without that transferring into solving much more.

### What this result does and does not show

It shows that preference pairs derived from a programmatic verifier, in a
prompt format matched to evaluation, improve a small model's zero-shot
solve rate on this task, with the preference stage contributing measurably
beyond supervised fine-tuning alone.

It does not show that *agentic multi-turn refinement* was distilled into
the model, because these DPO pairs were built from template-generated
correct-vs-wrong answers, not from a real model's own multi-turn revision
trajectories. The turn-budget experiment that motivates the whole project
is still a synthetic-solver result (see `PROVENANCE.md` and the honest
framing note below). Connecting the two properly requires the real-LLM
agent-loop experiment in `src/run_llm_agent_experiment.py`, which has not
been run yet.

### Caveats on this number

45 problems is a small held-out set: one problem is 2.2 points, so the DPO
gain is 4 to 7 problems and must not be quoted to three decimals. Two
training runs, greedy decoding at eval, one hyperparameter setting, one
0.5B model, one narrow task family. The base-to-SFT jump is solid; the DPO
increment appeared in all four runs but is small relative to its spread, and a controlled seed
sweep is what would settle both its existence and its size.
