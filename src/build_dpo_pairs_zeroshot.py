"""
Build DPO preference pairs in the SAME zero-shot prompt format as the
evaluation (unlike the original data/dpo_pairs.jsonl, whose prompts used
the multi-turn "revision" template -- equation + prior wrong attempt +
verifier feedback -- which never matched the zero-shot eval prompt at all;
see results/colab_run/training_log.md for the mismatch this caused).

chosen = the correct worked solution (reference_solution.worked_solution).
rejected = the SAME step-by-step template text, but with a wrong final
answer line swapped in (from flawed_solver, severity=3: a first-attempt-
style error). Using matching steps text for chosen/rejected is deliberate:
it keeps the two responses close in length and style, so DPO has a better
chance of learning "this final answer is right/wrong" rather than
picking up a spurious "prefer the longer/more-detailed response" signal.

KNOWN DEFECT OF THE v1 FILE, KEPT BECAUSE EVERY RECORDED GPU RUN USED IT

`flawed_solver` has no error mode for an equation with no real solution: its
"wrong" answer there is the empty set, i.e. `no solution`, i.e. the correct
answer. So chosen == rejected and the pair is dropped -- for EVERY such
problem. data/dpo_pairs_zeroshot.jsonl therefore contains 118 pairs and not one
of them has `no solution` on either side (all 25 no-real-root training problems
vanished), and every quadratic pair is "two roots beats one root". After DPO
on it, the model answers `no solution` to 86% of the solvable held-out
quadratics and 25% of the linears. See PROVENANCE.md, bug 8, and the paper.

`--variant v2` (or AVS_DPO_VARIANT=v2) fixes the coverage, not the method:
for a no-real-root problem the rejected answer is a spurious root a small
model plausibly produces (the vertex -b/2a of a quadratic, i.e. dropping the
imaginary part; the excluded point of a rational equation). It writes to a
SEPARATE file, data/dpo_pairs_zeroshot_v2.jsonl,
and it adds the same number of pairs the other way round -- a solvable problem
whose rejected answer is `no solution` -- so that `no solution` sits on the
chosen side and the rejected side equally often and DPO cannot learn the answer
string itself as a shortcut. That second half matters: the observed failure is
`no solution` on SOLVABLE problems, which v1 never penalises and a v2 that only
added chosen-side `no solution` would reward. It goes to a so the file the recorded runs
trained on is never overwritten. Train on it with
AVS_DATA_PATH=../data/dpo_pairs_zeroshot_v2.jsonl.

Output: data/dpo_pairs_zeroshot.jsonl, one {"problem_id", "family",
"equation_str", "system", "user", "chosen", "rejected"} row per problem.
"prompt" is intentionally NOT pre-assembled here: the Colab training script
builds it via tok.apply_chat_template([system, user], ...,
add_generation_prompt=True) -- the exact same call evaluate_finetuned.py
uses -- so train- and eval-time prompt formatting are guaranteed identical
byte-for-byte, not just "the same words".
"""
import json
import os
import random
import sys

import sympy as sp

import splits
from flawed_solver import attempt as flawed_attempt
from reference_solution import _true_answer_line, worked_solution, worked_solution_with_answer
from verifier import _true_solution_set
from tasks import contamination

SYSTEM = "You are a careful algebra solver. Show brief work, then give the final answer line."
N_PER_FAMILY = 60
SEED = splits.TRAIN_SEED
VARIANTS = {"v1": "../data/dpo_pairs_zeroshot.jsonl",
            "v2": "../data/dpo_pairs_zeroshot_v2.jsonl"}
# Drawn via splits.training(), which excludes every held-out equation and
# asserts it. The unfiltered draw this replaced overlapped the held-out set on
# 3 equations. See PROVENANCE.md, bug 7.


def user_prompt(equation_str):
    return (
        f"Solve for x: {equation_str}\n"
        "Give the final answer on its own line as 'x = v1, v2' or 'x = no solution'."
    )


def spurious_root_line(family, equation_str):
    """A plausible WRONG answer for an equation that has no real solution.

    quadratic: the vertex -b/(2a) -- what you get by dropping the imaginary
               part of the complex roots, the commonest small-model slip.
    rational:  the excluded point x = -c, i.e. ignoring the domain.
    Returns None if the equation does have a real solution.
    """
    if _true_answer_line(equation_str) != "no solution":
        return None
    eq = sp.sympify(equation_str)
    x = sp.Symbol("x")
    expr = sp.together(eq.lhs - eq.rhs)
    if family == "quadratic":
        poly = sp.Poly(sp.numer(expr), x)
        a, b = poly.coeff_monomial(x**2), poly.coeff_monomial(x)
        v = sp.nsimplify(-b / (2 * a))
    else:
        _sol, domain = _true_solution_set(equation_str)
        v = sorted(domain, key=str)[0] if domain else sp.Integer(0)
    return f"x = {v}"


def build_rows(variant="v1"):
    problems = splits.training(n_per_family=N_PER_FAMILY, seed=SEED)
    rng = random.Random(SEED)
    rows = []
    for p in problems:
        chosen = worked_solution(p["family"], p["equation_str"])
        wrong_final_line = flawed_attempt(p["equation_str"], rng, severity=3)
        if variant == "v2" and wrong_final_line == "no solution":
            spurious = spurious_root_line(p["family"], p["equation_str"])
            if spurious is not None:
                wrong_final_line = spurious
        rejected = worked_solution_with_answer(p["family"], p["equation_str"], wrong_final_line)
        if chosen == rejected:
            continue  # flawed_solver got lucky and produced the correct answer; not a valid pair
        rows.append(_row(p, chosen, rejected))
    if variant == "v2":
        rows += _no_solution_rejected_rows(problems, sum(
            r["chosen"].splitlines()[-1] == "no solution" for r in rows))
    return rows


def _row(p, chosen, rejected):
    return {
        "problem_id": p["id"],
        "family": p["family"],
        "equation_str": p["equation_str"],
        "system": SYSTEM,
        "user": user_prompt(p["equation_str"]),
        "chosen": chosen,
        "rejected": rejected,
    }


def _no_solution_rejected_rows(problems, n):
    """`n` solvable problems, each paired with a `no solution` rejected answer.

    Stratified evenly over the three families (the failure was concentrated in
    quadratics, which a plain random sample under-covers). A separate RNG, so
    adding these never shifts the draws used above.
    """
    rng = random.Random(SEED + 1)
    fams = ["linear", "quadratic", "rational"]
    quota = {f: n // 3 + (1 if i < n % 3 else 0) for i, f in enumerate(fams)}
    picked = []
    for f in fams:
        solvable = [p for p in problems if p["family"] == f
                    and _true_answer_line(p["equation_str"]) != "no solution"]
        picked += rng.sample(solvable, quota[f])
    return [_row(p, worked_solution(p["family"], p["equation_str"]),
                 worked_solution_with_answer(p["family"], p["equation_str"], "no solution"))
            for p in sorted(picked, key=lambda q: q["id"])]


def main():
    variant = os.environ.get("AVS_DPO_VARIANT", "v1")
    if "--variant" in sys.argv:
        variant = sys.argv[sys.argv.index("--variant") + 1]
    assert variant in VARIANTS, f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}"
    rows = build_rows(variant)
    path = VARIANTS[variant]
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(splits.report(N_PER_FAMILY, SEED))
    n_c = sum(r["chosen"].splitlines()[-1] == "no solution" for r in rows)
    n_r = sum(r["rejected"].splitlines()[-1] == "no solution" for r in rows)
    print(f"{len(rows)} zero-shot-format DPO pairs ({variant}) written to {path}; "
          f"`no solution` is chosen in {n_c} and rejected in {n_r}")
    check = contamination(splits.training(N_PER_FAMILY, SEED), splits.held_out())
    print(f"held-out contamination: {check['n_overlapping']}/{check['n_held_out']}")
    print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()
