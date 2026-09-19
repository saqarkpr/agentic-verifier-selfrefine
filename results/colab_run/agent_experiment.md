# Does verifier feedback actually help? A controlled test, and a diagnosed confound

Status: **two controlled runs on a Colab T4 GPU, plus a pre-registered
diagnostic and a failed repair attempt.** This is the experiment the whole
project is named after, and it went through three readings before settling.

1. Raw result: feedback loses to a re-sampling control. Looks like a clean no.
2. Diagnostic: the feedback prompt was breaking the model's output format,
   and correcting for that flipped the sign. Verdict downgraded to
   *inconclusive*.
3. Repaired prompt, second run: the format problem **survived the repair**,
   and this time the correction did not flip the sign.

Where it lands: feedback never beat the control in either run, was worse on
five of six paired comparisons (mean -0.041, p=0.131), and the format
degradation now looks like a property of a narrowly fine-tuned 0.5B model
rather than a prompt bug. Consistent in direction, not statistically
significant, and not a general claim about verifier feedback. Read the
sections in order -- each one takes something back from the one before.

## Provenance of this result

Checked rather than assumed, because the code moved while this was running.
The experiment was executed from an earlier delivered copy of the repo than
the one that now records it, so the two were diffed file by file:

- `verifier.py`, `tasks.py`, `agent_loop.py`, `evaluate_redesign_v2.py` and
  every data-building script are **byte-identical** between the version that
  ran and the current one. Correctness judging and the held-out problem set
  did not move.
- `run_llm_agent_experiment.py` differs only by added metadata logging
  (a `describe_model_under_test()` helper and a JSON stamp of which
  checkpoint was loaded). Every line of experimental logic -- prompts,
  per-turn seeding, the two conditions, decoding parameters, answer
  extraction, counting -- is unchanged. **This result reproduces against
  current code given the same checkpoint.**
- `train_lora_sft.py` and `train_lora_dpo_zeroshot.py` differ only by the
  addition of `SEED = 0` and `seed=SEED, data_seed=SEED`. These affect
  training only. Re-running training under current code therefore produces
  a *new* checkpoint rather than this one, which is precisely why the model
  stamp was added.

The checkpoint used here scored 0.000 / 0.289 / 0.378 (base / SFT / SFT+DPO)
on the greedy held-out evaluation in the same session, matching run B of the
three recorded training runs.

Worth being explicit about why the checkpoint's identity does not undermine
the finding: the feedback-versus-control comparison is paired *within*
whichever model is loaded, under identical sampling seeds. A different
checkpoint would shift both columns together; it would not manufacture or
erase a gap between them.

## Setup

Model under test: the SFT+DPO checkpoint from the run below (referred to as
run C; greedy held-out solve rate 0.378). 45 held-out problems (seed 99),
3 seeds, 2 conditions, up to 3 turns, sampling at temperature 0.7.

The two conditions differ in exactly one thing, the turn-2/3 prompt:

- **feedback**: the equation, the model's own previous wrong answer, and the
  verifier's structured feedback on it.
- **control**: the original prompt again, re-sampled.

Every generation in the two conditions uses the *same* RNG seed, so sampling
randomness is held fixed and prompt content is the only difference.

## Result

| Turn budget | feedback | control | difference |
|---|---:|---:|---:|
| 1 | 0.289 | 0.289 | +0.000 |
| 2 | 0.356 | 0.385 | **-0.030** |
| 3 | 0.393 | 0.422 | **-0.030** |

Paired t-test across the 3 seeds: turn budget 2, t=-0.918, p=0.456; turn
budget 3, t=-1.512, p=0.270.

**Turn budget 1 is identical in both columns**, which is the built-in check
that the pairing held: with the same prompt and the same seed the two
conditions must produce the same answers, and they did.

## Reading it honestly, after the diagnostic

The first reading of this table was "verifier feedback provided no
measurable benefit over re-sampling". The pre-registered diagnostic
(`src/analyze_agent_experiment.py`, run afterwards on the logged
trajectories) shows that reading was wrong, and that the experiment cannot
support it.

### The instrument was broken, and the diagnostic caught it

The hypothesis registered in `run_llm_agent_experiment.py`'s docstring
*before* the run was that the revision prompt is out of distribution for a
model fine-tuned on one rigid prompt shape, and would degrade its output
format. Its falsifiable prediction was more `unparsable` verdicts in the
feedback condition. That prediction is confirmed, emphatically:

| turn | feedback unparsable | control unparsable | difference |
|---|---:|---:|---:|
| 2 | 38.5% (37/96) | 6.2% (6/96) | **+32.3 pts** |
| 3 | 19.5% (17/87) | 1.2% (1/83) | **+18.3 pts** |

At turn 2, more than one in three feedback-condition answers could not be
read by the extractor at all. Unparsable scores as wrong, so those are
counted as failures regardless of what the model reasoned.

### Correcting for it flips the sign

Restricting to answers the extractor could actually read:

| turn | condition | parsable | correct | accuracy given parsable |
|---|---|---:|---:|---:|
| 2 | feedback | 59 | 9 | 0.153 |
| 2 | control  | 90 | 13 | 0.144 |
| 3 | feedback | 70 | 5 | 0.071 |
| 3 | control  | 82 | 5 | 0.061 |

| turn | raw difference | parsable-only difference |
|---|---:|---:|
| 2 | −0.042 | **+0.008** |
| 3 | −0.003 | **+0.010** |

**The sign flips.** Feedback looks worse overall but is marginally better
among answers that could be read. The entire headline deficit is
attributable to the format artifact.

So the honest conclusion is not "feedback did not help". It is:

> **This experiment cannot settle whether the feedback content helps,
> because the feedback prompt destroyed roughly a third of its own
> measurements. The primary comparison is inconclusive, not null.**

That is a weaker claim about verifier feedback and a stronger finding about
why: a diagnosable failure mode rather than an unexplained negative. Whether
it was a *fixable* one is tested in the next section, where a deliberate
repair was attempted. It failed.

**Caveat, stated rather than buried.** Conditioning on parsability is a
post-hoc subgroup analysis, and parsability is not randomly assigned. A
model may well emit unreadable text precisely when it is confused, which
would make the surviving feedback answers a favourably selected subset. The
+0.008 and +0.010 are far too small and too selected to claim feedback
helps. The analysis is strong enough to invalidate the raw comparison; it is
not strong enough to replace it with a positive one.

### Two further observations from this run

**Anchoring.** In the feedback condition the model repeated its previous
answer verbatim 32.8% of the time (60/183), against 21.8% (39/179) in the
control. An earlier version of this file called this "a genuine behavioural
effect". **It is not** -- the repaired run reversed it (24.3% against
27.6%), and the retraction is recorded below. Kept here only so the sequence
of claims is visible.

**Recovery, as raw counts.** 14.6% (14/96) of wrong first attempts were
recovered with feedback against 18.8% (18/96) without. This inherits the
same confound as the headline number.

## The repaired prompt was run. It did not fix the problem.

`REVISION_PROMPT_STYLE = "minimal"`, on a freshly trained checkpoint (run D,
greedy held-out 0.311). Same 45 problems, 3 seeds, same pairing.

| Turn budget | feedback | control | difference |
|---|---:|---:|---:|
| 1 | 0.274 | 0.274 | +0.000 |
| 2 | 0.326 | 0.356 | -0.030 |
| 3 | 0.363 | 0.415 | -0.052 |

Paired t-tests: p=0.529 at budget 2, p=0.369 at budget 3. Turn 1 identical,
so the pairing held again.

### The format problem survived the repair

| | verbose prompt | minimal prompt |
|---|---:|---:|
| unparsable, feedback, turn 2 | 38.5% | 26.5% |
| unparsable, control, turn 2 | 6.2% | 9.2% |
| **gap** | **+32.3 pts** | **+17.3 pts** |
| unparsable, feedback, turn 3 | 19.5% | 27.5% |
| unparsable, control, turn 3 | 1.2% | 4.6% |
| **gap** | **+18.3 pts** | **+22.9 pts** |

Better at turn 2, *worse* at turn 3, and still enormous in both. Keeping the
first and last lines byte-identical to the training prompt was not enough.

**This failure is more informative than a success would have been.** Two
materially different revision-prompt designs both wrecked this model's output
format. That makes the degradation look like a property of a 0.5B model
fine-tuned on 300 examples of a single prompt shape, rather than a bug in one
prompt that a better prompt would fix.

### And the correction no longer rescues feedback

| | verbose: raw -> parsable-only | minimal: raw -> parsable-only |
|---|---|---|
| turn 2 | -0.042 -> **+0.008** (flips) | -0.041 -> **-0.026** (stays negative) |
| turn 3 | -0.003 -> **+0.010** (flips) | -0.037 -> **-0.021** (stays negative) |

In the first run, conditioning on parsability turned the deficit into a tiny
surplus, which is what made that run inconclusive. In the repaired run the
deficit survives the correction. The earlier "the sign flips" escape hatch
does not apply here.

### Pooling the two runs

Six paired seed-level comparisons at a three-turn budget, across both runs:

| run / checkpoint | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| verbose / ckpt C | -0.022 | +0.000 | -0.066 |
| minimal / ckpt D | +0.022 | -0.133 | -0.044 |

Mean -0.041, sd 0.055, **five of six at or below zero**, one-sample
t = -1.805, p = 0.131.

Consistently negative in direction, still not statistically significant. The
pooling is also not clean: the two runs used **different checkpoints** (C at
0.378 greedy, D at 0.311), so the comparison across runs mixes the prompt
change with a model change. Within each run the pairing is exact; across
runs it is not.

### A second mechanism, not explained by parsing

Error modes at turn 2 in the repaired run:

| | feedback | control |
|---|---:|---:|
| incomplete (missing a root) | **42.9%** | 24.5% |
| wrong value | 22.4% | 50.0% |
| unparsable | 26.5% | 9.2% |

Feedback nearly doubles the rate of *incomplete* answers. Shown its own
wrong answer plus a verifier comment, the model shifts from producing wrong
values toward producing answers that are missing roots. That is a
behavioural change, not a formatting artifact, and it is a plausible reason
feedback could actively hurt on a task where dropping a root is a scored
error.

### A finding retracted: anchoring

The first run's analysis reported that feedback made the model repeat its
previous answer more often (32.8% against 21.8%), and this file previously
called that "a genuine behavioural effect". The repaired run reverses it:
24.3% with feedback against 27.6% in the control. An effect that flips sign
between two runs is not an effect. That claim is withdrawn.

### Where this leaves the central question

Stated as precisely as the evidence allows:

> Across two controlled runs on two checkpoints, verifier feedback never
> outperformed simply re-sampling, and was worse on five of six paired
> comparisons (mean -0.041, p=0.131). The direction is consistent and the
> effect is not significant. In both runs the feedback prompt substantially
> degraded the model's output format, and a deliberate repair aimed at that
> failed, which suggests the degradation is a property of this narrowly
> fine-tuned model rather than a fixable prompt defect.

What that does **not** license: "verifier feedback does not work." One task
family, one 0.5B model, two checkpoints, three seeds each, p=0.131. A model
that could actually follow a revision instruction might behave completely
differently, and testing that requires a larger or less narrowly-tuned model,
which is the obvious next step and is not attempted here.

What it does license, and what is worth more than the original hypothesis
would have been: a controlled design plus a pre-registered diagnostic turned
an apparent +0.104 "self-refinement gain" into a sampling artifact, then
turned the apparent refutation into a measurement artifact, then showed the
measurement artifact was itself resistant to repair. Each step is a claim
this project made and then had to take back on its own evidence.

### What happens next, and why it is not result-shopping

`run_llm_agent_experiment.py` now has a `REVISION_PROMPT_STYLE` switch. The
original prompt (`"verbose"`) added two lines *and* rewrote the final
instruction line. The repaired one (`"minimal"`) keeps the first and last
lines byte-identical to the training prompt and inserts a single feedback
line between them.

Re-running with a changed prompt was legitimate for a specific reason: the
hypothesis was registered before the run, the diagnostic confirmed it, and
an instrument that corrupts a third of its measurements cannot answer the
question it was built for. Two commitments kept that honest: the original
run stays on the record permanently, and the repaired run wrote to a
separate file (`llm_agent_experiment_minimal.csv`).

The repaired run has now been executed. It did not vindicate the repair,
and its results are reported above exactly as they came out.

## Per-seed detail, and one striking pattern

| seed | condition | turn<=1 | turn<=2 | turn<=3 |
|---|---|---:|---:|---:|
| 0 | feedback | 0.244 | 0.356 | 0.400 |
| 0 | control  | 0.244 | 0.333 | 0.422 |
| 1 | feedback | 0.311 | 0.400 | 0.422 |
| 1 | control  | 0.311 | 0.422 | 0.422 |
| 2 | feedback | 0.311 | 0.311 | 0.356 |
| 2 | control  | 0.311 | 0.400 | 0.422 |

The control column at turn budget 3 is **0.422 on all three seeds** -- the
same 19 of 45 problems every time. The model's failures are systematic
rather than stochastic: re-sampling recovers a fixed recoverable set and
then saturates. That also caps how much *any* turn-based method can gain
here, feedback included.

Seed 2's feedback row is the clearest single case: turn 2 recovered
*nothing* (0.311 to 0.311) while the control recovered four problems over
the same budget.

## Why might feedback hurt rather than help?

Stated as hypotheses, not findings. Only the first was registered in advance
(it is in `run_llm_agent_experiment.py`'s docstring, written before the run):

1. **The revision prompt is out of distribution.** The model was SFT'd on
   300 examples of exactly one prompt shape and then DPO'd on the same
   shape. A prompt containing a previous attempt and a feedback sentence is
   unlike anything it was trained on, and a heavily fine-tuned 0.5B model
   may simply handle it worse -- including by breaking the answer format the
   evaluation regex depends on.
2. **Anchoring.** Showing the model its own wrong answer may make it more
   likely to repeat or lightly edit that answer than to re-derive.
3. **The feedback may be too generic to act on.** "At least one proposed
   value does not satisfy the equation" tells the model that it is wrong,
   which it already learns from being asked again, without telling it how to
   fix the error.

Hypothesis 1 makes a testable prediction that the logged trajectories can
settle: if the revision prompt is breaking the output format, the feedback
condition should show more `unparsable` verdicts at turns 2-3 than the
control. `src/analyze_agent_experiment.py` checks exactly that against
`results/llm_agent_experiment.csv`.

## What this does and does not say about the project

It does **not** say verifier-guided self-refinement cannot work. It says it
did not work *here*: on one narrow task, with one 0.5B model that had been
fine-tuned into a single rigid prompt format, with feedback strings written
for a synthetic solver rather than for a real model, at n=3 seeds. Every one
of those is a plausible reason for a null result, and a larger or less
narrowly-tuned model is the obvious next thing to try.

It does say that the project's headline claim cannot be "verifier feedback
improves LLM solve rate", because the one experiment built to test that
claim, with the control that makes it testable, does not support it.

What survives, and is worth more than a claim that failed its own test:

- Verifier-only training (no human labels, no LLM-as-judge) takes the model
  from 0/45 to about 17-20/45 held out. That reproduced across three runs.
- A controlled design that can tell feedback apart from repetition, which
  the synthetic turn-budget experiment could not, and which caught an
  apparent +0.104 "self-refinement gain" that was really a sampling artifact.
