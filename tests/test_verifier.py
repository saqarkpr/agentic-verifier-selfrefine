"""The verifier is the only source of ground truth in this project.

Every number in results/ is whatever this file said, so a silent change here
rewrites history without touching a single result. These tests therefore pin
three separate things:

  1. the verdicts (does it grade algebra correctly),
  2. the accepted answer *spellings* -- in BOTH directions, because a format
     the verifier rejects is a measurement thrown away, and a format it
     wrongly accepts is a free point,
  3. the two crashes that actually killed GPU evaluation runs, which are
     regression tests in the literal sense: they reproduce a hang and a
     TypeError that both happened for real.
"""
import time

import pytest
import sympy as sp

from verifier import _parse_candidate, _true_solution_set, verify

LINEAR = "Eq(3*x + 6, 0)"              # x = -2
QUADRATIC_2 = "Eq(x**2 - 1, 0)"        # x = -1, 1
QUADRATIC_0 = "Eq(x**2 + 1, 0)"        # no real solution
RATIONAL = "Eq((2*x + 3)/(x + 1), 5)"  # x = -2/3, excluded point x = -1


# ------------------------------------------------------------ the verdicts --

@pytest.mark.parametrize("equation, answer", [
    (LINEAR, "x = -2"),
    (QUADRATIC_2, "x = 1, -1"),
    (QUADRATIC_2, "x = -1, 1"),      # order must not matter: it is a set
    (QUADRATIC_0, "x = no solution"),
    (RATIONAL, "x = -2/3"),
])
def test_correct_answers_are_accepted(equation, answer):
    assert verify(equation, answer).correct


def test_extraneous_root_is_named_as_such():
    """The excluded point of a rational equation is the error mode the whole
    task family exists to produce, and the feedback string for it is the most
    informative one the verifier emits."""
    res = verify(RATIONAL, "x = -2/3, -1")
    assert not res.correct
    assert res.reason == "extraneous_root"
    assert "-1" in res.feedback  # the feedback must name the offending value


def test_missing_a_root_is_incomplete_not_generic_wrong():
    res = verify(QUADRATIC_2, "x = 1")
    assert res.reason == "incomplete"


def test_extra_value_is_does_not_satisfy():
    res = verify(QUADRATIC_2, "x = 1, -1, 5")
    assert res.reason == "does_not_satisfy"


def test_wrong_value_is_wrong():
    assert verify(LINEAR, "x = 2").reason == "wrong"


def test_claiming_no_solution_when_there_is_one_is_not_correct():
    res = verify(LINEAR, "x = no solution")
    assert not res.correct


def test_true_solutions_are_reported_on_every_failure():
    """Downstream code (DPO pair construction, the analysis scripts) reads
    `true_solutions` off failed results; an empty list there would silently
    produce garbage preference pairs."""
    for answer in ("x = 5", "x = banana", "x = 1"):
        res = verify(QUADRATIC_2, answer)
        assert not res.correct
        assert set(res.true_solutions) == {-1, 1}


# ------------------------------------------------- the accepted spellings --
# The prompt used for training and evaluation says, verbatim:
#   "Give the final answer on its own line as 'x = v1, v2' or 'x = no solution'."
# Anything that spelling covers MUST parse.

@pytest.mark.parametrize("answer, expected", [
    ("x = 2", {2}),
    ("x= 2", {2}),
    ("X = 2", {2}),                  # a capitalised line start is the same answer
    ("x = +2", {2}),
    ("x = -2/3", {sp.Rational(-2, 3)}),
    ("x = 2.5", {sp.Rational(5, 2)}),
    ("x = sqrt(2)", {sp.sqrt(2)}),
    ("x = 1, -1", {1, -1}),
    ("x = 1; -1", {1, -1}),
    ("{1, -1}", {1, -1}),
    ("1, -1", {1, -1}),
    ("x = 1/2, x = -3", {sp.Rational(1, 2), -3}),
])
def test_specified_value_forms_parse(answer, expected):
    assert _parse_candidate(answer) == expected


@pytest.mark.parametrize("answer", [
    "no solution",
    "x = no solution",
    "X = no solution",
    "x = No Solution.",
    "x = no real solution",
    "x = no real roots",
    "x = none",
    "x = {}",
    "empty set",
])
def test_no_solution_phrasings_parse_to_the_empty_set(answer):
    """Regression test for bug #5. The prompt asks for 'x = no solution' and
    the verifier used to accept only the bare 'no solution', scoring the
    requested format as unparsable on the 8 of 45 held-out problems that have
    no real solution."""
    assert _parse_candidate(answer) == set()


@pytest.mark.parametrize("answer", ["", "   ", "x = ", "x =", "X="])
def test_an_empty_answer_is_unparsable_not_no_solution(answer):
    """An empty answer is a measurement the extractor could not read. It used
    to fall through to the empty set, which is the CORRECT answer on the
    no-solution problems -- points for saying nothing."""
    assert _parse_candidate(answer) is None
    assert verify(QUADRATIC_0, answer).reason == "unparsable"


@pytest.mark.parametrize("answer", [
    "x = 3 and x = -1",      # prose separator: not in the specified format
    "x = 2 (approximately)",
    r"x = \frac{1}{2}",      # LaTeX
    "x = plus or minus 2",
    "the answer is 2",
])
def test_unspecified_forms_stay_rejected(answer):
    """The other direction, pinned deliberately. Widening the parser to accept
    these would change what every recorded number means, so the boundary is a
    decision, not an accident -- if a future change accepts one of them, that
    change has to come with a re-run, and this test is what forces the
    conversation."""
    assert _parse_candidate(answer) is None


# ------------------------------------------------------ the real crashes --

def test_power_tower_does_not_hang():
    """`parse_expr("9**9**9")` makes sympy materialise a 369-million-digit
    integer. This hung a GPU evaluation run indefinitely -- indistinguishable
    from a crashed script. The wall-clock bound is the assertion; a verdict
    alone would pass even if it took an hour."""
    start = time.monotonic()
    res = verify(RATIONAL, "9**9**9")
    assert time.monotonic() - start < 5.0
    assert not res.correct


def test_prose_answer_does_not_raise():
    """sympy's implicit-multiplication parser turns an English word into a
    product of one-letter symbols -- "24 doubts hot" becomes
    24*b*d*h*o**2*s*t**2*u -- and sympy then raises TypeError rather than
    returning False when asked whether that lies in the excluded-value set.
    This killed a real evaluation run 40 problems in."""
    for prose in ("x = 24 doubts hot", "x = I am not sure", "x = Eq(-1, 24*b)"):
        res = verify(RATIONAL, prose)
        assert res.reason == "unparsable"


@pytest.mark.parametrize("answer", [
    "x = " + "1" * 300,                                 # over the length cap
    "x = " + ", ".join(str(i) for i in range(200)),     # over the part cap
    "x = 2**999999",                                    # huge exponent
])
def test_oversized_answers_score_wrong_without_hanging(answer):
    start = time.monotonic()
    assert not verify(RATIONAL, answer).correct
    assert time.monotonic() - start < 5.0


# --------------------------------------------------------- ground truth --

def test_extraneous_roots_are_excluded_from_the_true_solution_set():
    """`_true_solution_set` must remove domain-excluded points itself, or the
    verifier would mark a correct answer incomplete for omitting one."""
    sol, domain = _true_solution_set("Eq((x**2 - 1)/(x - 1), 0)")
    assert sp.Integer(1) in domain
    assert sp.Integer(1) not in set(sol)
