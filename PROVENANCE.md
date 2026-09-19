# Provenance

## Compute environment this repo was built in
CPU-only sandbox, 2 vCPU, no GPU, ~8GB RAM. This shaped two honest scope
decisions, documented here rather than glossed over:

1. **The main experiment (`results/`) uses a synthetic "flawed solver"
   (`src/flawed_solver.py`), not a real LLM.** It stands in for a small
   LLM's error distribution on this task, with seeded error modes
   (dropped root, extraneous root, arithmetic slip) matched to the
   verifier's own feedback categories, and an error rate that decreases
   across revision turns to model partial in-context correction. This is
   **not** a claim about a real model's behavior; it validates the
   agent-loop + verifier + preference-pair infrastructure end-to-end and
   answers a real, well-posed question on its own terms: *given an agent
   that makes these kinds of mistakes, how much does multi-turn verifier
   feedback recover, and where do the returns taper off?* Full numbers,
   code, and the paired-seed significance test are in `results/`.

2. **The LoRA-DPO fine-tuning step (`src/train_lora_dpo.py`,
   `src/hf_solver.py`, `src/evaluate_finetuned.py`,
   `notebooks/colab_train.ipynb`) is real, runnable code against a real
   model (Qwen2.5-0.5B-Instruct) and a real training library (trl +
   peft). It was not executable inside this CPU-only sandbox, so it was
   run for real on a Google Colab T4 GPU** -- the same pattern already
   used in `RLVF` and `biomedical-rag-eval`/`sci-lit-rag-eval` (LLM
   components run on Colab). **This has now been done**: training
   completed (real loss curve, adapter saved) and the held-out zero-shot
   evaluation ran against it. Full numbers and an honest reading of the
   (null) result are in `results/colab_run/training_log.md`.

## A bug found and fixed along the way: non-deterministic problem seeding
`src/tasks.py`'s `generate_problems()` originally seeded each family's RNG
with `seed * 1000 + hash(family) % 997`. Python salts `str.__hash__` per
process by default (`PYTHONHASHSEED`), so the *same* nominal `seed=0` could
silently produce different problem sets -- and therefore a different number
of DPO pairs -- in two separate runs of identical code. This surfaced when
the sandbox run (197 pairs) and the Colab run (196 pairs) of the same
`build_preference_data.py` disagreed. Root-caused and fixed: `tasks.py` now
uses a fixed `_FAMILY_OFFSET` dict instead of `hash()`, so
`generate_problems(seed=0)` is deterministic across processes and machines.
Re-running the full pipeline after the fix (the numbers now reported
throughout this repo) gives 180 DPO pairs and a very slightly different
turn-budget curve (0.614 / 0.761 / 0.801 instead of the original 0.617 /
0.767 / 0.809 -- both real, both from 120 problems x 8 seeds, differing only
because the fix changed which 120 problems were generated). The Colab
training run itself was **not** re-done after this fix -- its adapter and
eval numbers are from the original, pre-fix 196-pair data, and are reported
as such rather than silently backdated to the fixed pipeline.

## A second bug found and fixed: verifier feedback wasn't actually reaching the solver
`agent_loop.py`'s `run_trajectory()` collected each turn's verifier
feedback into a list (`feedback_so_far`) but never passed it into the next
call to `solve_fn` -- only an abstract `severity` scalar (3/2/1 by turn)
was passed. For the synthetic `flawed_solver`, this happened to be
harmless: its error model already encodes "feedback helps" through the
severity schedule, not by reading feedback text, and it never had a
parameter to receive feedback in the first place. But for a **real LLM**
plugged in via `hf_solver.py` -- which already had a `prior_feedback`
parameter specifically for this -- the bug meant every turn would see an
identical prompt with no memory of the previous wrong attempt or the
verifier's comment on it, making turn 2/3 an independent re-guess rather
than a genuine revision. This directly undermines the project's central
mechanism ("verifier feedback consumed inside a multi-turn loop") for
anyone who used `notebooks/colab_train.ipynb`'s optional cell to
regenerate DPO pairs from a real model's own agentic trajectories, rather
than the synthetic solver.

Fixed: `run_trajectory` now threads the previous turn's feedback string
through as `prior_feedback` on every call to `solve_fn`; `flawed_solver.attempt()`
gained a `prior_feedback=None` parameter it accepts and intentionally
ignores (documented in its docstring), for interface parity.
`hf_solver.attempt()` needed no change -- it already used the parameter
correctly whenever it was given one. Re-running the full CPU pipeline
after this fix reproduced **identical** numbers (0.614/0.761/0.801, 180
pairs, same error breakdown), as expected, since the synthetic solver
never read the parameter either way; the fix only changes behavior for
the (optional, not used to produce this repo's headline numbers) real-LLM
trajectory path.

## A third bug found and fixed: the per-turn error schedule was inverted
`src/flawed_solver.py`'s `_perturb()` decides how often the synthetic
solver "gets it right" using a threshold formula meant to implement its
own documented intent: error probability *decreases* across revision
turns (severity 3 = first attempt = most errors, severity 1 = after two
rounds of feedback = fewest errors). The original formula was
`max(0.05, 0.55 - 0.18 * (3 - severity))`, which is backwards: plugging in
severity=3 (turn 1) gives the *largest* subtracted term and therefore the
*lowest* error rate, and severity=1 (turn 3) gives the *smallest*
subtracted term and the *highest* error rate -- the exact opposite of the
docstring. This was caught while investigating why
`build_dpo_pairs_zeroshot.py` was dropping roughly half its severity=3
pairs as "chosen == rejected" (the flawed solver getting lucky far more
often than a first attempt should). Empirical check of correct-rate by
severity confirmed it: **before** the fix, severity 3/2/1 gave
59.2%/47.5%/31.7% correct (turn 3, "after feedback," had the *most*
errors); **after** the fix (`max(0.05, 0.55 - 0.18 * (severity - 1))`),
severity 3/2/1 gives 31.7%/47.5%/59.2% correct (turn 3 now has the
*fewest* errors, as intended).

This bug directly inflated the apparent "diminishing returns" story in
every version of this report up to and including two prior PDF deliveries
to the user: with the schedule backwards, the turn-1-to-turn-2 gain looked
large and the turn-2-to-turn-3 gain looked small, because turn 3 was
secretly the *hardest* setting, not the easiest. Re-running the full
pipeline after the fix changes the headline turn-budget numbers from
0.614/0.761/0.801 (both already reflecting the seeding fix above) to
**0.295/0.540/0.797** -- solve rate at turn 1 drops sharply (fewer lucky
correct first attempts now that severity=3 is genuinely the hardest
setting), and the story changes from clearly-diminishing returns to
**roughly constant, non-diminishing returns**: the turn-2-to-turn-3 gain
(0.257) is essentially the same size as the turn-1-to-turn-2 gain (0.245),
not a fraction of it. The DPO pair count in `data/dpo_pairs.jsonl` also
changed, from 180 to **482** pairs -- with severity=3 genuinely producing
more wrong first attempts, fewer trajectories get discarded as
"already correct, no revision to learn from." As with the first bug, the
already-completed Colab LoRA-DPO run (`results/colab_run/training_log.md`)
was **not** re-done after this fix -- it used the pre-fix 196-pair
dataset, and is reported as such rather than silently attributed to the
corrected pipeline. `report/report.tex`, `README.md`, and this file have
all been updated to the corrected numbers; the pre-fix numbers are kept
here, explicitly labeled, rather than deleted, since they were already
reported to the user before the bug was found.

## A fourth bug found and fixed: sympy set iteration order was process-dependent
Same class as the first bug, in a different place, and found the same way:
a count that would not sit still. Running `build_dpo_pairs_zeroshot.py`
three times on identical, fully-seeded code gave 123, then 121, then 122
pairs.

Root cause: `flawed_solver._perturb()` did `true_set = list(true_set)` on a
Python `set` of sympy objects. Iterating a set yields elements in hash
order, and a sympy expression's hash ultimately hashes strings, which
Python salts per process (`PYTHONHASHSEED`) -- the same salting behind bug
1. Checked directly: of the 180 problems in the seed=0 set, **28** (the
quadratics with irrational roots, e.g. `sqrt(17)/2 + 1/2`) iterate in a
different order in different processes. Integer and rational roots hash
stably, which is why this was invisible on most problems.

The consequence was that `_perturb`'s "drop a root" branch dropped a
*different* root, and its "arithmetic slip" branch perturbed a *different*
value, from one process to the next. Fixed by ordering deterministically
with `sorted(true_set, key=str)` (and likewise for the domain set), which
is the same ordering `verifier.py` and `reference_solution.py` already use.

**This changed no reported solve rate, and that is not luck -- it is why
the bug hid.** The verifier compares solution *sets*, so dropping either
root of a two-root quadratic is equally "incomplete", and perturbing
either value is equally wrong. Re-running the full pipeline after the fix
reproduces every headline number exactly (0.295 / 0.540 / 0.797, the same
t-statistics, the same 72.5/15.5/12.0 error breakdown, the same 482 DPO
pairs). What the bug *did* break was byte-level reproducibility: the
candidate strings in `results/trajectories.csv` and the contents of the
DPO pair files differed between runs. After the fix,
`results/trajectories.csv` and `data/dpo_pairs.jsonl` are byte-identical
across repeated runs (verified by checksum across separate processes), and
`data/dpo_pairs_zeroshot.jsonl` is a stable **118** pairs instead of
drifting between 121 and 123.

As with the earlier fixes, the GPU runs already completed are **not**
backdated: the redesign-v2 DPO stage documented in `redesign_v2.md` trained
on the 123-pair version of that file, and is reported as such.

## Three robustness bugs found in the evaluation path
None changed a reported number, but each could have produced a wrong or
missing one. Two of the three were found the hard way, by an evaluation run
on a GPU failing:

1. **The verifier could hang forever on adversarial model output.**
   `verify()` feeds a model's raw answer line to sympy's `parse_expr`,
   which then *evaluates* what it parses. A generated answer of
   `9**9**9` makes sympy try to materialize a 369-million-digit integer
   and the process never returns (reproduced directly: it hangs past any
   timeout, while `2**100000` returns instantly). An evaluation loop that
   hangs on one bad generation out of 45 is indistinguishable from a
   crashed script. Fixed in `verifier.py` with explicit guards before
   parsing (length caps, a cap on how many comma-separated values an
   answer may contain, and rejection of power towers or exponents above
   1000); anything rejected scores as `unparsable`, i.e. wrong, which is
   the correct outcome for a malformed answer. Verified that every
   legitimate answer form still parses to exactly the same value as
   before, and that the full pipeline reproduces identical numbers.
2. **The evaluation scripts never moved the model to the GPU.**
   `AutoModelForCausalLM.from_pretrained(...)` leaves the model on CPU;
   neither `evaluate_finetuned.py` nor `evaluate_redesign_v2.py` called
   `.to("cuda")`, so inside a Colab GPU runtime every generation ran on
   two vCPUs. The redesign-v2 script (45 problems x 3 models x up to 150
   new tokens) showed no output at all for over 15 minutes before this was
   caught. Fixed by placing the models on the GPU when one is available,
   and by printing progress every 5 problems so a slow run can no longer
   be mistaken for a hung one. `dtype` stays `float32` rather than
   dropping to `float16`: float32 on GPU is the same arithmetic as float32
   on CPU under greedy decoding, so this is purely a speed fix and the
   first run's 0.022 baseline remains comparable.
3. **The verifier crashed on a model that answered in prose.** With the
   GPU fix in place, the redesign-v2 evaluation got 40 problems into the
   base model before dying with
   `TypeError: did not evaluate to a bool: Eq(-1, 24*b*d*e*h*i*o**3*s**2*t**3)`.
   The cause is subtle and worth recording. The parser is configured with
   sympy's `implicit_multiplication_application`, which includes
   `split_symbols`: it breaks a multi-letter name into a product of
   one-letter symbols. So an ordinary English word on the answer line
   parses successfully -- `24 doubts hot` becomes
   `24*b*d*h*o**2*s*t**2*u` -- and the exponents are *manufactured by the
   parser*, not present in the text, which is why the string-level guards
   added for the hang above could not catch it. That symbolic expression
   then reached `c in domain`, and sympy raises rather than returning False
   when it cannot decide whether a symbolic expression equals an excluded
   point.

   Fixed at the right layer: a solution to "solve for x" is a *number*, so
   `_parse_candidate` now rejects any parse result that still carries free
   symbols (or is not a numeric value at all), scoring it `unparsable`,
   i.e. wrong -- which is the correct verdict for prose. A second guard
   wraps the membership test itself, so an undecidable comparison returns
   False instead of raising: a verifier that can raise is a verifier that
   can lose an entire evaluation run 40 problems in. `python src/verifier.py`
   now runs a standing adversarial-input check (power tower, prose, empty
   string, over-long input, too many values) that asserts each is scored
   wrong without hanging or raising. Re-running the full pipeline after
   this fix reproduces every reported number exactly.

## An overclaim corrected (not a code bug, and the most important entry here)
Every bug above was found by code misbehaving. This one was not: the code
did exactly what it was written to do, and the *claim* made about it was
wrong.

The README, the report abstract and the report conclusion all said some
version of "multi-turn verifier feedback substantially and significantly
improves solve rate". The turn-budget experiment cannot support that
sentence. It runs against `flawed_solver`, which (a) receives a `severity`
parameter from `agent_loop.py` that is *defined* to decrease with each turn
(`severity = max(1, 4 - turn)`), and (b) states in its own docstring that
it ignores the feedback text entirely and models "feedback helps" through
that schedule alone. An experiment whose solver is instructed to improve on
retry, and which then measures improvement on retry, has confirmed its own
premise. The rising curve is an artifact of the design, not evidence about
verifier feedback.

This was never concealed -- the synthetic solver is disclosed prominently
here, in the README and in the report's Section 3.1 and Limitations -- but
disclosure in the methods section does not license a causal claim in the
abstract, and the causal phrasing was the one a reader would carry away.
The framing has been corrected throughout: the turn-budget experiment is
now described as measuring *how much of a given error distribution a
revision budget recovers*, plus end-to-end validation that the loop,
verifier, trajectory logging and preference-pair extraction work on real
data. Those claims the experiment does support.

Credit where due: this was caught by an external review of the code, not by
the author, after the numbers had already been reported several times.

What would actually test the causal claim is now written and included as
`src/run_llm_agent_experiment.py`, and the design point is the control, not
the LLM. A model given three attempts solves more than a model given one
purely because three samples are three chances, so turn 1 vs. turn 3 cannot
separate feedback from repetition. The script runs each problem twice under
*identical* sampling seeds: in one condition turns 2 and 3 see the previous
wrong answer plus the verifier's feedback, in the other they see the
original prompt again and simply re-sample. Turn 1 is generated from the
same seed in both and must come out identical, which is a built-in check
that the pairing holds. The gap between conditions at a fixed turn budget
is what the feedback content buys.

That experiment has since been run, and then **the result it appeared to
give was itself retracted by a pre-registered diagnostic.** Both stages are
recorded because the sequence is the point.

**What the run showed.** At a matched three-turn budget the feedback
condition reached 0.393 and the re-sampling control 0.422; at two turns,
0.356 against 0.385. Turn budget 1 was identical in both conditions,
confirming the pairing held. Read at face value: feedback did not help, and
was nominally slightly worse.

**What the diagnostic then showed.** The hypothesis registered in the
script's docstring before the run -- that a revision prompt is out of
distribution for a model fine-tuned on one rigid prompt shape and would
degrade its output format -- predicted more `unparsable` verdicts in the
feedback condition. It was confirmed emphatically: 38.5% (37/96) against
6.2% (6/96) at turn 2, and 19.5% against 1.2% at turn 3. An unparsable
answer scores as wrong, so more than a third of the feedback condition's
turn-2 measurements were discarded before the model's reasoning was ever
assessed.

Restricting to answers the extractor could read reverses the comparison:
accuracy given parsable is 0.153 vs 0.144 at turn 2 and 0.071 vs 0.061 at
turn 3, turning raw differences of -0.042 and -0.003 into +0.008 and +0.010.
The headline deficit is an artifact of the prompt breaking the output
format, not evidence about the feedback content.

**So the defensible claim is neither positive nor null but inconclusive:**
this experiment cannot settle whether verifier feedback helps, because the
feedback prompt destroyed roughly a third of its own measurements. That
conditioning is a post-hoc subgroup analysis on a non-randomly-assigned
variable, so it invalidates the raw comparison without replacing it with a
positive result.

Two effects survive the artifact independently. Anchoring: the model
repeated its previous answer verbatim 32.8% of the time with feedback
against 21.8% without. Saturation: the control reaches exactly 0.422 at
budget 3 on all three seeds, the same 19 of 45 problems, so failures are
systematic rather than stochastic.

The instrument has been repaired rather than the finding reworded.
`run_llm_agent_experiment.py` now carries a `REVISION_PROMPT_STYLE` switch;
the repaired variant keeps the first and last lines of the revision prompt
byte-identical to the training prompt. Re-running under a changed prompt is
defensible here only because the hypothesis was registered in advance and
then confirmed -- repairing an instrument that corrupts a third of its
readings is not the same as retrying until the answer is agreeable. Two
commitments keep that distinction concrete: the original run stays on the
record permanently, and the repaired run writes to a separate file so it
cannot overwrite it. The repaired run has **not** been executed, and no
claim is made about its outcome.

Full account, per-seed numbers and the full error-mode breakdown:
`results/colab_run/agent_experiment.md`.

## What IS a completed, real result in this repo
- The problem generator (`src/tasks.py`) and SymPy verifier
  (`src/verifier.py`) are fully real and independently checked against
  ground truth (`python src/verifier.py`).
- The full multi-turn agent-loop experiment: 120 problems x 8 seeds x up
  to 3 turns, all actually executed, with a paired-seed t-test (not just
  a headline mean) -- `results/trajectories.csv`, `results/summary.csv`,
  `results/analysis.txt`.
- The 482 DPO preference pairs in `data/dpo_pairs.jsonl` are real, derived
  directly from the executed trajectories (not fabricated examples), from
  the fixed, deterministic generator and the corrected severity schedule
  (see "A third bug found and fixed" above).
- The LoRA-DPO training run and the base-vs-fine-tuned held-out zero-shot
  evaluation were both actually executed on a Colab T4 GPU. The result is
  a **null result** (fine-tuning did not improve zero-shot solve rate),
  most likely explained by a prompt-format mismatch between the
  revision-conditioned training pairs and the unconditioned zero-shot
  eval prompt -- see `results/colab_run/training_log.md` for the full
  account and the follow-up this motivates.
