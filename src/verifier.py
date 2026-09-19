"""
Programmatic SymPy verifier. This is the sole source of ground truth and
of the structured feedback the agent loop feeds back to the solver -- no
human labels and no LLM-as-judge, matching the verified-feedback approach
already used in the RLVF project.

verify(equation_str, candidate_str) -> VerifierResult
"""
from dataclasses import dataclass, field

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

x = sp.symbols("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


@dataclass
class VerifierResult:
    correct: bool
    reason: str  # short machine-readable code
    feedback: str  # human/agent-readable natural-language feedback
    true_solutions: list = field(default_factory=list)


def _true_solution_set(equation_str):
    eq = sp.sympify(equation_str)
    lhs_minus_rhs = eq.lhs - eq.rhs
    domain = sp.solveset(sp.Eq(sp.denom(sp.together(lhs_minus_rhs)), 0), x, domain=sp.S.Reals)
    sol = sp.solveset(eq, x, domain=sp.S.Reals)
    if isinstance(sol, sp.FiniteSet):
        sol = sp.FiniteSet(*[s for s in sol if s not in domain])
    return sol, domain


# Guards against adversarial/garbage model output. A real LLM's answer line
# is free-form text, and sympy will happily try to *evaluate* whatever it
# parses: parse_expr("9**9**9") attempts to materialize a 369-million-digit
# integer and hangs the process indefinitely. That is not a hypothetical --
# it is a reproducible hang (verified), and an evaluation loop that hangs on
# one bad generation out of 45 looks exactly like a crashed script. A
# malformed answer must score as wrong, never as an infinite loop, so
# anything outside the shape of a real answer is rejected before parsing.
_MAX_CANDIDATE_CHARS = 200  # a genuine answer line ("x = -1/2, 3") is far shorter
_MAX_PARTS = 12  # these equations have at most 2 roots; 12 is already absurd
_MAX_PART_CHARS = 40
_POWER_TOKENS = ("**", "^")


def _is_number(expr):
    """True only for a concrete numeric value (2, -1/3, sqrt(17)/2 + 1/2).

    False for anything symbolic, and for non-expressions such as a parsed
    Eq(...) or a boolean, which have no business being a solution value.
    """
    try:
        if getattr(expr, "free_symbols", None):
            return False
        return bool(expr.is_number)
    except (AttributeError, TypeError):
        return False


def _is_safe_to_parse(part):
    """Reject anything that could make sympy evaluate a huge number."""
    if len(part) > _MAX_PART_CHARS:
        return False
    # A power tower (two or more exponentiations) is the dangerous case.
    if sum(part.count(tok) for tok in _POWER_TOKENS) > 1:
        return False
    for tok in _POWER_TOKENS:
        idx = part.find(tok)
        if idx == -1:
            continue
        exponent = part[idx + len(tok):].strip().lstrip("+-")
        digits = ""
        for ch in exponent:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits and int(digits) > 1000:
            return False
    return True


def _parse_candidate(candidate_str):
    """Parse a model's answer string into a set of sympy numbers.
    Accepts forms like 'x = 2', 'x = 2, -3', '{2, -3}', 'no solution', '2'.
    Returns None for anything unparsable or rejected by the safety guards
    above; callers treat None as an incorrect (unparsable) answer.
    """
    s = candidate_str.strip().lower()
    if s in ("no solution", "no real solution", "none", "{}", "empty set", "no solutions", "emptyset"):
        return set()
    if len(candidate_str) > _MAX_CANDIDATE_CHARS:
        return None
    s = candidate_str.replace("x", "").replace("=", "")
    parts = [p.strip().strip("{}") for p in s.replace(";", ",").split(",") if p.strip()]
    if len(parts) > _MAX_PARTS:
        return None
    out = set()
    for p in parts:
        if not _is_safe_to_parse(p):
            return None
        try:
            expr = parse_expr(p, transformations=_TRANSFORMS)
        except Exception:
            return None  # unparsable
        # A solution to "solve for x" is a NUMBER. Anything still carrying
        # free symbols is prose the model wrote, not an answer. This matters
        # more than it looks: the parser's split_symbols transformation turns
        # an ordinary English word into a product of one-letter symbols, so
        # "24 doubts hot" parses happily as 24*b*d*h*o**2*s*t**2*u. Those
        # exponents are created by the parser, not present in the text, so no
        # amount of string inspection beforehand can catch them. Letting such
        # an expression through crashed a real evaluation run: sympy cannot
        # decide whether a symbolic expression lies in the excluded-value set,
        # so `c in domain` raised TypeError and killed the whole loop 40
        # problems in. A model writing prose is simply wrong, not fatal.
        if not _is_number(expr):
            return None
        try:
            out.add(sp.nsimplify(expr))
        except Exception:
            return None
    return out


def _in_domain(value, domain):
    """Is `value` one of the equation's excluded points?

    Belt-and-braces around sympy's set membership, which raises TypeError
    rather than returning False when it cannot decide (see _parse_candidate).
    _parse_candidate should already have rejected everything undecidable, so
    this should never trigger -- but a verifier that can raise is a verifier
    that can lose a whole evaluation run, and "I could not decide" is not the
    same as "this is an extraneous root".
    """
    try:
        return bool(value in domain)
    except TypeError:
        return False


def verify(equation_str, candidate_str):
    true_sol, domain = _true_solution_set(equation_str)
    true_set = set(true_sol) if isinstance(true_sol, sp.FiniteSet) else set()

    cand = _parse_candidate(candidate_str)
    if cand is None:
        return VerifierResult(
            correct=False,
            reason="unparsable",
            feedback="Your answer could not be parsed. Give solutions as 'x = v1, v2' or 'no solution'.",
            true_solutions=sorted(true_set, key=str),
        )

    if cand == true_set:
        return VerifierResult(True, "correct", "Correct.", sorted(true_set, key=str))

    extraneous = {c for c in cand if _in_domain(c, domain)}
    if extraneous:
        return VerifierResult(
            False,
            "extraneous_root",
            f"x = {extraneous} makes a denominator zero and is not a valid solution. Check the domain restriction.",
            sorted(true_set, key=str),
        )

    missing = true_set - cand
    wrong = cand - true_set
    if missing and not wrong:
        return VerifierResult(
            False, "incomplete", f"Missing solution(s): your answer has {len(cand)} root(s), "
            f"but the equation has {len(true_set)}.", sorted(true_set, key=str),
        )
    if wrong and not missing:
        return VerifierResult(
            False, "does_not_satisfy",
            "At least one proposed value does not satisfy the equation when substituted back in.",
            sorted(true_set, key=str),
        )
    return VerifierResult(
        False, "wrong",
        "Proposed solution set does not match: wrong count and/or wrong value(s).",
        sorted(true_set, key=str),
    )


if __name__ == "__main__":
    from tasks import generate_problems

    for p in generate_problems(n_per_family=2, seed=0):
        true_sol, _ = _true_solution_set(p["equation_str"])
        r = verify(p["equation_str"], str(true_sol))
        print(p["id"], p["equation_str"], "->", true_sol, r.correct, r.reason)

    # Adversarial inputs. A real LLM's answer line is free-form text, and both
    # of the failures below actually happened during evaluation runs on a GPU:
    # a hang (sympy evaluating a power tower) and a crash (sympy refusing to
    # decide set membership for a symbolic expression). The verifier must
    # score a malformed answer as wrong and keep going -- never hang, never
    # raise -- so these stay as a standing check.
    print("\nAdversarial input check (must all return quickly, no exception):")
    rational_eq = "Eq((2*x + 3)/(x + 1), 5)"  # excluded point: x = -1
    adversarial = [
        "9**9**9",                 # power tower: used to hang forever
        "x = 24 doubts hot",       # prose -> symbolic product: used to crash
        "x = I am not sure",
        "x = Eq(-1, 24*b)",
        "x = " + "1" * 300,
        "x = " + ", ".join(str(i) for i in range(200)),
        "",
        "x = ",
    ]
    for bad in adversarial:
        res = verify(rational_eq, bad)
        assert res.correct is False, f"adversarial input scored correct: {bad!r}"
        print(f"  ok  {bad[:34]!r:38} -> {res.reason}")
    print("All adversarial inputs handled.")
