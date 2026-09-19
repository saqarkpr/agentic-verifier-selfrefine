# agentic-verifier-selfrefine

Does giving a solver multiple turns to revise its answer, using structured
feedback from a programmatic verifier, recover accuracy that a single
attempt misses -- and can the *pattern of successful revisions* be
distilled back into the model itself via LoRA-DPO, so it needs fewer turns
at inference time?

This extends two earlier pieces of this portfolio:
- the M.Sc. thesis (`Symbolic-RL-Fredholm-Solver`), which used a
  programmatic (non-LLM) verifier as ground truth for RL;
- `verifier-based-preference-optimization` (RLVF), which DPO-tuned an LLM
  against a programmatic verifier's pass/fail signal, single-shot.

Here the verifier feedback is not just a training signal after the fact.
It's used **inside a multi-turn agent loop at inference time**, and the
correct-versus-wrong contrasts it produces become DPO preference pairs.

Short version of what came out, both halves:

- **The training half worked.** Verifier-only training (no human labels, no
  LLM-as-judge) took Qwen2.5-0.5B-Instruct from **0/45 to 14-20/45** on
  held-out problems across four runs (mean 17/45). The supervised stage
  gives 13/45 with zero run-to-run variance; the preference stage adds about
  four more, but with a spread nearly as large as the effect.
- **The agentic half went through three readings and ended short of a
  claim.** (1) Against a re-sampling control, the apparent self-refinement
  gain (0.289 to 0.393) was fully accounted for by simply being allowed more
  attempts (control 0.422). (2) A pre-registered diagnostic then showed the
  feedback prompt was wrecking the model's output format (38.5% unparsable
  vs 6.2%), and correcting for that flipped the sign, so even the negative
  reading was unsafe. (3) A repaired prompt was written and run. **The
  format problem survived the repair**, and this time the correction did not
  flip the sign. Across both runs feedback never beat the control and was
  worse on five of six paired comparisons (mean -0.041, p=0.131).

Consistent in direction, short of significance, and not a general claim
about verifier feedback -- one narrow task, one 0.5B model. The sequence is
reported as prominently as the positive result, because each step is a claim
this project made and then had to withdraw on its own evidence.

## Task
Symbolic equation solving (linear / quadratic / rational, the last with a
domain restriction that is the most common source of "extraneous root"
errors), verified exactly with SymPy -- no human labels, no LLM-as-judge.

## Pipeline
```
tasks.py            generate verifiable problems
verifier.py          ground truth + structured feedback (independently checked)
flawed_solver.py      synthetic solver w/ realistic error modes (CPU-only stand-in, see PROVENANCE.md)
hf_solver.py           real Qwen2.5-0.5B-Instruct solver (Colab)
agent_loop.py         multi-turn: attempt -> verify -> feedback -> revise (max 3 turns)
run_experiment.py     120 problems x 8 seeds, solve rate by turn budget
analysis.py           paired-seed significance test + figures
build_preference_data.py  trajectories -> DPO pairs
train_lora_dpo.py      LoRA-DPO fine-tune (Colab; peft + trl)
evaluate_finetuned.py  base vs. fine-tuned zero-shot solve rate (Colab)

reference_solution.py  verifier-checked worked solutions (templated, not LLM-written)
build_sft_data.py      -> data/sft_pairs.jsonl        (prompt-matched SFT targets)
build_dpo_pairs_zeroshot.py -> data/dpo_pairs_zeroshot.jsonl (prompt-matched pairs)
train_lora_sft.py      stage 1: LoRA-SFT, then merge into base weights (Colab)
train_lora_dpo_zeroshot.py  stage 2: LoRA-DPO on the SFT model (Colab)
evaluate_redesign_v2.py     base vs. SFT vs. SFT+DPO, held out (Colab)
run_llm_agent_experiment.py real LLM in the loop, feedback vs. re-sampling
                            control (the project's central test)
analyze_agent_experiment.py diagnoses that result: parse failures, anchoring
run_dpo_seed_sweep.py       controlled DPO seed sweep (four incidental
                            reruns already gave a variance estimate)
```

## Headline result: verifier-only training takes a 0.5B model from 0% to ~38%
Executed for real on a Colab T4 GPU. 45 held-out problems, seed 99,
disjoint from all training data, greedy decoding, no human labels and no
LLM-as-judge anywhere in the pipeline.

The pipeline was run **four** times end to end on identical data, which is
enough to put an error bar on it.

| Model | Run A | Run B | Run C | Run D | mean (sd) |
|---|---|---|---|---|---|
| Base | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| + LoRA-SFT | 0.289 | 0.289 | 0.289 | 0.289 | **0.289 (0.000)** |
| + LoRA-SFT + LoRA-DPO | 0.444 | 0.378 | 0.378 | 0.311 | **0.378 (0.054)** |

DPO gain over SFT: +7, +4, +4, +1 problems out of 45. Mean **+0.089**, sd
0.054, one-sample t-test t=3.27, p=0.047.

**One caveat on the 0/45 baseline, flagged rather than resolved.** The eval
prompt and output format are task-specific and the base model saw neither in
training, so 0/45 may measure format compliance as much as capability. The
evaluation only counted correct answers, so it could not tell those apart.
`evaluate_redesign_v2.py` now logs the verifier's reason code for every
attempt and `src/inspect_base_outputs.py` dumps raw base generations to
settle it. Neither has been run yet.

The three-way split is the interesting part. SFT supplies nearly all of the
capability and does so with remarkable stability: **13/45, the same 13
problems, in all four runs**. DPO then adds a further +0.089 on average on a
model already supervised on these same problems, which is not explained by
exposure to correct answers since SFT already supplied those.

But state that gain honestly: it is about four problems, its sd (0.054) is
nearly as large as the effect itself, and p=0.047 on four runs is fragile
enough that one more low run would cross 0.05. Positive, small, not robust.

**Why four runs, and why all are shown.** GPU training is not
bit-reproducible (CUDA kernels are non-deterministic by default), so a
single run is one draw rather than a measurement. The SFT stage was stable
at exactly 0.289 all four times; the variance is entirely in the DPO stage,
which trains on 100 pairs for 21 steps.

An earlier version of this README reported run A's 0.444 as *the* result.
With four runs it is visibly an outlier, 1.22 sd above the mean. Reporting
the best run would have overstated the effect by roughly a factor of two.
Also worth noting: run B beat run A on *every* DPO training metric (eval
loss, reward accuracy, reward margin) and was worse on held-out solve rate,
so selecting a checkpoint by validation loss would have picked the weaker
model.

This result is a correction of an earlier null result (documented below
rather than deleted). The single change that mattered was training and
evaluating on the same prompt format. Full account:
`results/colab_run/redesign_v2.md`.

Two claims of unequal strength, now separable: **base to SFT (0/45 to
13/45, identical in all four runs) is solid**; the **DPO increment is
positive in all four runs but small relative to its own spread** (+0.089
mean, sd 0.054, p=0.047).

## Secondary result: turn budget (CPU-only, **synthetic** solver, read the caveat)
| Turn budget | Mean solve rate (8 seeds) |
|---|---|
| 1 (no revision) | 0.295 |
| 2 | 0.540 |
| 3 | 0.797 |

**What this does and does not show.** This runs against a *synthetic*
solver whose error rate is defined to decrease each turn
(`severity = max(1, 4 - turn)`) and which ignores the verifier's feedback
text entirely. So it measures **how much of a given error distribution a
revision budget recovers**, and it validates the loop, verifier, trajectory
logging and preference-pair extraction end-to-end on real data. It does
**not** show that verifier feedback causes the improvement, because the
improvement is a modelling assumption, not a finding. Earlier versions of
this README and the report claimed the causal version. That was an
overclaim, it was caught by external code review, and it is corrected here
and in `PROVENANCE.md`. The experiment that actually tests it has since been
run, and what it did and did not settle is below.

Paired t-test, 1 vs. 2 turns: t=63.79, p<0.00001. 2 vs. 3 turns: t=31.77,
p<0.00001. The gains do not meaningfully diminish: the second revision
(turn 2 -> 3, +0.257) recovers essentially as much as the first (turn 1 ->
2, +0.245) does. (An earlier version of this table read
0.614/0.761/0.801 with visibly diminishing returns -- that was a bug in the
synthetic solver's error schedule, not a real finding; see bug 3 below.)
Turn-1 errors are dominated by plain wrong values (72.5%), with
extraneous-root errors (15.5%, from ignoring the domain restriction) and
incomplete solution sets (12.0%) as the two systematic failure modes. See
`results/analysis.txt` and the two figures in `results/`.

## The central question: still open, and here is exactly why
Run on a Colab T4 against the SFT+DPO checkpoint. 45 held-out problems, 3
seeds, sampling at temperature 0.7. Each problem is run twice under
**identical sampling seeds**: once where turns 2-3 see the previous wrong
answer plus the verifier's feedback, once where they just see the original
prompt again.

| Turn budget | feedback | control | difference |
|---|---|---|---|
| 1 | 0.289 | 0.289 | +0.000 |
| 2 | 0.356 | 0.385 | **-0.030** |
| 3 | 0.393 | 0.422 | **-0.030** |

Paired t-test: p=0.456 at budget 2, p=0.270 at budget 3. Turn budget 1 is
identical in both columns, which is the built-in check that the pairing
held.

At face value that says feedback did not help. **A pre-registered diagnostic
shows that reading is wrong**, and the experiment is inconclusive rather
than null.

The hypothesis registered *before* the run was that a revision prompt is out
of distribution for a model fine-tuned on one rigid prompt shape and would
break its output format. It predicted more `unparsable` verdicts in the
feedback condition. Confirmed, emphatically:

| turn | feedback unparsable | control unparsable |
|---|---|---|
| 2 | **38.5%** (37/96) | 6.2% (6/96) |
| 3 | **19.5%** (17/87) | 1.2% (1/83) |

Unparsable scores as wrong, so over a third of the feedback condition's
turn-2 measurements were discarded before the reasoning was ever judged.
Restricting to answers the extractor could read, **the sign flips**:
accuracy given parsable is 0.153 vs 0.144 at turn 2 and 0.071 vs 0.061 at
turn 3. The raw -0.042 and -0.003 become +0.008 and +0.010.

So the honest verdict is: **this experiment cannot settle whether feedback
helps, because the feedback prompt destroyed roughly a third of its own
measurements.** (Conditioning on parsability is post-hoc and parsability is
not randomly assigned, so this invalidates the raw comparison without
replacing it with a positive one.)

Two things do stand independently:

- **Anchoring is real.** With feedback the model repeated its previous
  answer verbatim 32.8% of the time vs 21.8% in the control.
- **Re-sampling saturates.** The control hits exactly 0.422 at budget 3 on
  all three seeds, the same 19 of 45 problems, so failures are systematic
  rather than stochastic.

The experiment script now has a repaired prompt variant that keeps the first
and last lines byte-identical to the training prompt. The original run stays
on the record and the repaired run writes to a separate file.

Why this matters more than a positive result would have: the feedback
condition improves from 0.289 to 0.393 across turns. Reported on its own
that is a +0.104 gain from "multi-turn verifier-guided self-refinement",
which is exactly the number this project was built to produce. The control
shows the same model going from 0.289 to 0.422 with no feedback at all. The
whole apparent gain was the pass@k effect of being allowed to try again.

This is a negative result about *this setup*, not about the idea: one narrow
task, one 0.5B model fine-tuned into a single rigid prompt format, feedback
strings originally written for a synthetic solver, three seeds. Full
analysis, per-seed numbers and the candidate explanations are in
`results/colab_run/agent_experiment.md`.

### The repair was attempted, and failed

`REVISION_PROMPT_STYLE = "minimal"` keeps the revision prompt's first and
last lines byte-identical to the training prompt. Re-run on a fresh
checkpoint:

| Turn budget | feedback | control | difference |
|---|---|---|---|
| 1 | 0.274 | 0.274 | +0.000 |
| 2 | 0.326 | 0.356 | -0.030 |
| 3 | 0.363 | 0.415 | -0.052 |

The format gap narrowed at turn 2 (38.5% → 26.5%) and **widened** at turn 3
(19.5% → 27.5%), staying above 17 points at both. Two materially different
prompt designs both wrecked the output format, which points at a property of
a 0.5B model fine-tuned on 300 examples of one shape rather than a bug in
one prompt.

And the escape hatch closed: in the first run, conditioning on parsability
turned −0.042/−0.003 into +0.008/+0.010; in the repaired run the same
correction leaves −0.026/−0.021. The deficit survives.

Two further notes. A new mechanism appears that parsing does not explain:
feedback nearly doubles *incomplete* answers at turn 2 (42.9% vs 24.5%),
i.e. shown its own wrong answer the model shifts toward dropping roots. And
one earlier claim is **withdrawn** -- the first run's "anchoring" finding
(32.8% vs 21.8% repeated answers) reversed in the repaired run (24.3% vs
27.6%). An effect that flips sign between runs is not an effect.

Full account: `results/colab_run/agent_experiment.md`.

## Earlier null result, kept on the record: LoRA-DPO on mismatched prompts
482 DPO preference pairs, built from the wrong-to-correct transitions in
the agent trajectories, were used to LoRA-DPO fine-tune
Qwen2.5-0.5B-Instruct (peft + trl, 3 epochs). Training loss dropped
cleanly from 0.50 to 0.16 over 3 epochs -- the model is fitting the
preference data. The actual research question, though, was whether that
translates into a higher **zero-shot** solve rate (single call, no agent
loop, no verifier feedback at inference time), on 45 held-out problems:

| Model | Zero-shot solve rate |
|---|---|
| Base | 0.022 (1/45) |
| LoRA-DPO fine-tuned | 0.000 (0/45) |
| Delta | -0.022 |

**This is a null result**, and it's reported as one rather than
reframed. The most likely explanation is a prompt-format mismatch: every
training pair uses the *revision* prompt (equation + prior wrong attempt +
verifier feedback), while the zero-shot eval prompt has none of that --
the fine-tuning data never covered the exact prompt shape it was tested
on. Full numbers, the training curve, and the reasoning are in
`results/colab_run/training_log.md`. The fair follow-up this motivates is
DPO pairs matched to the zero-shot prompt format, not a re-run of this
same setup expecting a different answer.

## How the redesign was built
The fair follow-up promised above: an SFT warm-start
(`src/train_lora_sft.py`) on `data/sft_pairs.jsonl`, then LoRA-DPO on top
of it (`src/train_lora_dpo_zeroshot.py`) on `data/dpo_pairs_zeroshot.jsonl`
-- both trained and evaluated on the *same* zero-shot prompt shape this
time, built once via `tok.apply_chat_template(..., add_generation_prompt=True)`
and reused for training and eval so the two can't drift apart again.
`src/evaluate_redesign_v2.py` reports Base / SFT-only / SFT+DPO, not just
a before/after pair, so it's possible to see which stage helps. Full
design rationale and caveats: `results/colab_run/redesign_v2.md`. This was
written and committed before the run, with no promise about the outcome --
the result it produced is the headline table at the top.

## What's real vs. what's ready-to-run
See `PROVENANCE.md`. Every number above came from a run that actually
happened: the agent-loop/verifier experiment in a CPU sandbox, and all
three fine-tuning/evaluation runs on a Colab T4 GPU. The one thing in this
repo that has **not** been run is `src/run_dpo_seed_sweep.py`, which would
pin down the DPO increment's variance across seeds.
`PROVENANCE.md` also documents every bug found and fixed during later
review passes, including what each one did and did not change:
1. Non-determinism in the problem generator (Python's per-process
   `hash()` salting made "same seed" not actually reproducible across
   runs) -- `PROVENANCE.md` is explicit about which numbers predate the fix.
2. `agent_loop.py` collected each turn's verifier feedback but never
   actually passed it to the solver for the next attempt -- harmless for
   the synthetic solver (which never reads it either way), but it meant a
   real-LLM agent loop would not actually see feedback on revision turns.
   Fixed; re-running the CPU pipeline after the fix reproduced identical
   numbers, confirming this particular fix did not itself change the
   headline result.
3. `flawed_solver.py`'s per-turn error-rate schedule had its sign
   inverted from its own documented intent, so turn 3 ("after feedback")
   was secretly the *hardest* setting instead of the easiest. This one
   **did** change headline result 1 -- from an artificially diminishing
   0.614/0.761/0.801 to the corrected, non-diminishing 0.295/0.540/0.797
   above, and changed the DPO pair count from 180 to 482. Full before/after
   numbers and how it was caught are in `PROVENANCE.md`.

4. `flawed_solver.py` iterated a Python `set` of sympy objects, whose hash
   order is salted per process, so *which* root it dropped or perturbed
   varied between runs of identical seeded code (28 of the 180 problems,
   the quadratics with irrational roots). Solve rates were unaffected --
   the verifier compares sets, so dropping either root is equally wrong,
   which is exactly why it hid -- but `results/trajectories.csv` and the
   DPO pair files were not byte-reproducible. Fixed; both are now
   checksum-identical across runs.

Plus three robustness bugs in the evaluation path, none of which changed a
reported number: the verifier could hang forever on adversarial model
output (`9**9**9` makes sympy try to build a 369-million-digit integer);
the evaluation scripts never moved the model onto the GPU, so they ran
generation on CPU inside a GPU runtime; and the verifier crashed outright
when the model answered in prose, because sympy's parser turns an English
word into a product of one-letter symbols and then refuses to decide
whether that symbolic expression is an excluded root. All three surfaced
while actually running the evaluation, not in review.
`python src/verifier.py` now runs a standing adversarial-input check.

As with bug 1, the already-completed Colab runs were **not** re-done after
later fixes. Headline result 2 used the pre-fix 196-pair dataset, and the
redesign-v2 DPO stage used a 123-pair build of
`data/dpo_pairs_zeroshot.jsonl` that now reproduces as 118. Both are
reported as such rather than silently backdated.

## Reproduce
```
cd src
python run_experiment.py
python analysis.py
python build_preference_data.py
# LoRA-DPO training + held-out eval require a GPU; run in Colab:
#   notebooks/colab_train.ipynb (or paste src/train_lora_dpo.py /
#   src/evaluate_finetuned.py into a GPU runtime)

# Redesign v2 (prompt-matched SFT + DPO -- the headline result) also needs
# a GPU; run notebooks/colab_train.ipynb's "Redesign v2" section, or:
python build_sft_data.py
python build_dpo_pairs_zeroshot.py
# then on a GPU runtime:
#   python train_lora_sft.py
#   python train_lora_dpo_zeroshot.py
#   python evaluate_redesign_v2.py
#
# The controlled real-LLM agent-loop test (GPU), then its analysis (CPU):
#   python run_llm_agent_experiment.py
python analyze_agent_experiment.py
# Still to run (GPU): python run_dpo_seed_sweep.py
```

## Reproducibility layer

Configs, launch scripts, a Slurm array job and a pinned container, so a run's
settings are recoverable rather than living as literals inside a script.

```
configs/        sft.yaml, dpo.yaml, agent_ablation.yaml
src/config.py   env var > YAML > call-site default; dumps what was used
scripts/        single-GPU entrypoints
slurm/          array job, one task per ablation condition
Dockerfile      pinned env; build fails if the verifier self-test fails
```

`AVS_N_SEEDS=10 bash scripts/run_ablation.sh` sweeps without editing code.
The Slurm file and Dockerfile are written against standard conventions and
exercised locally. **Neither has been run on an HPC cluster**, and nothing
here should be read as claiming otherwise.
