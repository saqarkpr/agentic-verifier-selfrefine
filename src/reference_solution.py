"""
Template-based worked-solution generator, used to build SFT and (redesigned,
zero-shot-format) DPO training targets for the "make the distillation result
actually positive" follow-up (see PROVENANCE.md and
results/colab_run/redesign_v2.md).

IMPORTANT, stated honestly: these are NOT LLM-written solutions. They are
deterministic text templates whose final numeric answer always comes from
verifier._true_solution_set (the same ground truth the verifier itself
checks against), so the "chosen" targets are guaranteed correct by
construction -- but the prose is templated, not generated, and reads that
way. This is a legitimate and common way to build a seed SFT/DPO dataset
for a narrow, exactly-checkable task; it is not being passed off as
anything else.

solution_steps(family, equation_str) -> reasoning text, NOT including the
final "x = ..." line (so the same steps text can be paired with either the
true final line, for a chosen response, or a wrong one, for a rejected
response with matching structure/length -- avoiding a length/style
confound between chosen and rejected in the DPO pairs).
"""
import sympy as sp

from verifier import _true_solution_set, x


def _true_answer_line(equation_str):
    true_sol, _domain = _true_solution_set(equation_str)
    true_set = sorted(set(true_sol), key=str) if isinstance(true_sol, sp.FiniteSet) else []
    if not true_set:
        return "no solution"
    return ", ".join(f"x = {v}" for v in true_set)


def solution_steps(family, equation_str):
    eq = sp.sympify(equation_str)
    lhs_minus_rhs = sp.together(eq.lhs - eq.rhs)

    try:
        if family == "linear":
            poly = sp.Poly(sp.numer(lhs_minus_rhs), x)
            a, b = poly.all_coeffs()
            return (
                f"This is linear. Moving the constant term gives {a} times the "
                f"unknown equals {-b}. Dividing both sides by {a} isolates the unknown."
            )

        if family == "quadratic":
            poly = sp.Poly(sp.numer(lhs_minus_rhs), x)
            a, b, c = poly.all_coeffs()
            disc = sp.nsimplify(b**2 - 4 * a * c)
            if disc < 0:
                return (
                    f"This is quadratic with a={a}, b={b}, c={c}. "
                    f"The discriminant (b squared minus 4ac) is {disc}, which is "
                    "negative, so there is no real solution."
                )
            return (
                f"This is quadratic with a={a}, b={b}, c={c}. "
                f"The discriminant (b squared minus 4ac) is {disc}. "
                "Applying the quadratic formula (negative b, plus or minus the "
                "square root of the discriminant, all over 2a) gives the roots below."
            )

        if family == "rational":
            return (
                "This is rational and has a domain restriction. Multiply both "
                "sides by the denominator to clear the fraction, solve the "
                "resulting linear equation for x, then check the result doesn't "
                "make the original denominator zero."
            )
    except Exception:
        pass  # fall through to the generic template below

    return "Solve the equation for x directly."


def worked_solution(family, equation_str):
    """Full chosen/reference response: steps + the TRUE final answer line."""
    return solution_steps(family, equation_str) + "\n" + _true_answer_line(equation_str)


def worked_solution_with_answer(family, equation_str, answer_line):
    """Same steps text, but with a given (possibly wrong) final answer line --
    used to build a rejected DPO response with matching structure/length to
    the chosen one, isolating "is the answer right" as the only real
    difference DPO can learn from, rather than "is it longer / does it show
    work at all"."""
    return solution_steps(family, equation_str) + "\n" + answer_line


if __name__ == "__main__":
    from tasks import generate_problems

    for p in generate_problems(n_per_family=2, seed=0):
        print("---", p["id"], p["equation_str"])
        print(worked_solution(p["family"], p["equation_str"]))
