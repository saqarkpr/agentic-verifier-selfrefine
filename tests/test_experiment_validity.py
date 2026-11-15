"""Tests that guard the experiment, not the code.

Every test here fails on something that would still run perfectly, produce a
full results table, and be wrong. A GPU cannot tell you that your control
condition was not a control, or that your feedback string quietly contained
the answer -- you would simply get a beautiful number and believe it.

These are the assertions that would have caught the two real design bugs in
this project: a revision prompt that told the model its answer was "incorrect"
when the verifier had accepted it, and (in the earlier turn-budget experiment)
a solver told to improve on retry, which made the result circular by
construction.
"""
import re

import pytest

from feedback_prompts import (ALL_CONDITIONS, ANSWER_FORMAT, _procedure_for,
                              base_prompt, build_prompt, named_values)
from verifier import verify

RATIONAL = "Eq((2*x + 3)/(x + 1), 5)"   # x = -2/3, excluded point x = -1
QUADRATIC = "Eq(x**2 - 1, 0)"           # x = -1, 1
FEEDBACK_CONDITIONS = [c for c in ALL_CONDITIONS if c != "retry"]


def _wrong_result(equation, answer):
    res = verify(equation, answer)
    assert not res.correct, "fixture must be a wrong answer"
    return res


# ------------------------------------------- the control must be a control --

def test_retry_prompt_is_byte_identical_to_turn_one():
    """`retry` is the pass@k baseline every other condition is measured
    against. If its revision prompt differs from the turn-1 prompt by even a
    word, it is a sixth treatment, and every 'vs retry' delta in the results
    table is comparing against the wrong thing."""
    res = _wrong_result(RATIONAL, "x = 5")
    assert build_prompt("retry", RATIONAL, "x = 5", res) == base_prompt(RATIONAL)


def test_retry_prompt_ignores_the_verifier_entirely():
    """Stronger form: `retry` must not vary with the verifier's output at all,
    whatever it says or however wrong the previous answer was."""
    prompts = {
        build_prompt("retry", RATIONAL, cand, verify(RATIONAL, cand))
        for cand in ("x = 5", "x = -1", "x = banana", "x = no solution")
    }
    assert len(prompts) == 1


# -------------------------------------------- feedback must not leak truth --

# Lines that every prompt is allowed to contain: the problem itself and the
# answer-format instruction. Both are fixed text the model already had at turn
# 1, so neither can leak anything -- but both contain digits (the equation's
# coefficients, and the "v1, v2" of the format line), so the leak check has to
# subtract them or it flags itself.
_SHAPE_LINE = "Reply with brief work, then the final line and nothing after it."


def _inserted_text(condition, equation, prev_candidate, result):
    """Exactly the part of the prompt the condition adds: not the restated
    equation, not the format line, and not the model's own previous answer
    echoed back to it."""
    prompt = build_prompt(condition, equation, prev_candidate, result)
    safe = {f"Solve for x: {equation}", ANSWER_FORMAT, _SHAPE_LINE}
    middle = "\n".join(line for line in prompt.split("\n") if line not in safe)
    return middle.replace(prev_candidate, "")


@pytest.mark.parametrize("condition", FEEDBACK_CONDITIONS)
@pytest.mark.parametrize("equation, wrong_answer, true_values", [
    (RATIONAL, "x = 5", ["-2/3"]),
    (QUADRATIC, "x = 1", ["-1"]),        # a partial answer: the echo contains a true root
    (QUADRATIC, "x = 7", ["-1", "1"]),
])
def test_no_condition_leaks_the_true_answer(condition, equation, wrong_answer, true_values):
    """The single failure that would invalidate the whole ablation. If the
    revision prompt contains the answer, a 'gain from feedback' measures
    nothing but the model's ability to copy."""
    inserted = _inserted_text(condition, equation, wrong_answer, _wrong_result(equation, wrong_answer))
    numbers = set(re.findall(r"-?\d+(?:/\d+)?", inserted))
    for value in true_values:
        assert value not in numbers, (
            f"condition {condition!r} leaked the true value {value}:\n{inserted}"
        )


def test_the_leak_check_would_actually_catch_a_leak():
    """A test that only ever passes proves nothing. This one confirms the
    extraction above is looking at the right text, by feeding it a feedback
    string that does leak."""
    class Leaky:
        correct = False
        reason = "incomplete"
        feedback = "The solutions are -1 and 1."

    inserted = _inserted_text("diagnostic", QUADRATIC, "x = 7", Leaky())
    assert "-1" in set(re.findall(r"-?\d+(?:/\d+)?", inserted))


def test_incomplete_feedback_reveals_the_count_but_not_the_values():
    """A deliberate design line, worth pinning: telling the model 'you have 1
    root, the equation has 2' is a hint about structure. Telling it *which*
    root is missing would be the answer."""
    res = _wrong_result(QUADRATIC, "x = 1")
    assert res.reason == "incomplete"
    assert "2" in res.feedback                      # the count is given
    assert "-1" not in named_values(res.feedback)   # the missing root is not


# ----------------------------------------- feedback must not state untruths --

def test_a_correct_answer_is_never_called_incorrect():
    """The bug this test exists for: an earlier version told the model
    'Your previous answer x = -3 was incorrect. Correct.' after the verifier
    had ACCEPTED x = -3, in order to measure a regression rate. The resulting
    number would have measured the false premise, not the model."""
    res = verify(RATIONAL, "x = -2/3")
    assert res.correct
    for condition in ALL_CONDITIONS:
        prompt = build_prompt(condition, RATIONAL, "x = -2/3", res)
        assert "incorrect" not in prompt.lower(), condition


def test_the_confirmation_prompt_says_the_checker_accepted_it():
    res = verify(RATIONAL, "x = -2/3")
    prompt = build_prompt("diagnostic", RATIONAL, "x = -2/3", res)
    assert "accepted" in prompt


# --------------------------------------- the ladder must be a real ladder --

def test_each_rung_adds_information_to_the_one_below():
    """The conditions are meant to form a monotone ladder: generic < diagnostic
    < actionable in information content. If two rungs are the same text, the
    comparison between them is empty."""
    res = _wrong_result(RATIONAL, "x = -1")
    prompts = {c: build_prompt(c, RATIONAL, "x = -1", res) for c in ALL_CONDITIONS}
    assert len(set(prompts.values())) == len(ALL_CONDITIONS), "two conditions are identical"
    assert len(prompts["retry"]) < len(prompts["generic"]) < len(prompts["diagnostic"]) \
        < len(prompts["actionable"]) < len(prompts["constrained"])


def test_only_constrained_changes_the_prompt_shape():
    """Every rung but the last must keep the same three-part shape -- restate
    the problem, say something, restate the format -- so that a difference
    between them is a difference in information, not in layout."""
    res = _wrong_result(RATIONAL, "x = -1")
    for condition in ("generic", "diagnostic", "actionable"):
        lines = build_prompt(condition, RATIONAL, "x = -1", res).split("\n")
        assert len(lines) == 3, condition
        assert lines[0] == f"Solve for x: {RATIONAL}"
        assert lines[-1] == ANSWER_FORMAT
    assert len(build_prompt("constrained", RATIONAL, "x = -1", res).split("\n")) == 4


def test_every_condition_restates_the_answer_format():
    """A revision prompt that drops the format line invites exactly the
    unparsable-answer degradation the ablation is trying to measure, and would
    confound the information ladder with a formatting artefact."""
    res = _wrong_result(RATIONAL, "x = -1")
    for condition in ALL_CONDITIONS:
        assert ANSWER_FORMAT in build_prompt(condition, RATIONAL, "x = -1", res), condition


def test_procedures_are_actions_not_answers():
    """The `actionable` rung must tell the model what to DO. A procedure that
    named values would collapse into a hint."""
    for reason in ("extraneous_root", "incomplete", "does_not_satisfy", "wrong", "unparsable"):
        procedure = _procedure_for(reason)
        assert not re.search(r"\d", procedure), (reason, procedure)
        assert procedure.split()[0] in ("Check", "Substitute", "Write"), procedure


def test_an_unknown_condition_is_an_error_not_a_silent_fallback():
    res = _wrong_result(RATIONAL, "x = -1")
    with pytest.raises(ValueError):
        build_prompt("diagnostc", RATIONAL, "x = -1", res)  # typo'd condition


# --------------------------------------- the utilisation metric must work --

def test_named_values_extracts_the_value_the_verifier_called_out():
    """The feedback-utilisation metric asks: the verifier named a bad value --
    did the model's next answer drop it? That only measures anything if the
    extractor actually finds the named value in the feedback string."""
    res = _wrong_result(RATIONAL, "x = -2/3, -1")
    assert res.reason == "extraneous_root"
    assert "-1" in named_values(res.feedback)


def test_named_values_handles_rationals_and_signs():
    assert named_values("x = -2/3 makes a denominator zero") == {"-2/3"}
    assert named_values("values 1, -1 and 3/4") == {"1", "-1", "3/4"}
    assert named_values(None) == set()
    assert named_values("no numbers here") == set()


def test_utilisation_is_undefined_when_the_verifier_named_nothing():
    """`does_not_satisfy` feedback names no value, so 'did the model drop the
    named value' has no answer. The metric must be skipped, not scored zero --
    scoring it would dilute the rate with rows that could never count."""
    res = _wrong_result(QUADRATIC, "x = 1, -1, 5")
    assert res.reason == "does_not_satisfy"
    assert named_values(res.feedback) == set()
