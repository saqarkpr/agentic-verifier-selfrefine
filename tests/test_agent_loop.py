"""The multi-turn loop, and one executable warning about what it can prove.

The loop itself is small, but two of its properties decide whether a
trajectory means anything: it must stop at the first correct answer (a loop
that kept going would let a correct answer be overwritten by a later wrong
one), and it must actually pass the verifier's feedback forward (a loop that
did not would make every "revision" an independent re-sample).

The last test in this file is different in kind. It does not check that the
code works; it checks that a documented limitation is still true, so that the
circular claim this project once made cannot be quietly re-derived.
"""
import random

import pytest

from agent_loop import MAX_TURNS, run_all, run_trajectory
from flawed_solver import attempt as flawed_attempt
from tasks import generate_problems
from verifier import verify

PROBLEM = {"id": "t_0000", "family": "quadratic", "equation_str": "Eq(x**2 - 1, 0)"}
RATIONAL = {"id": "t_0001", "family": "rational", "equation_str": "Eq((2*x + 3)/(x + 1), 5)"}


class Scripted:
    """A solver that returns a fixed sequence and records what it was given."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, equation_str, rng, severity, prior_feedback=None):
        self.calls.append({"severity": severity, "prior_feedback": prior_feedback})
        return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]


def test_loop_stops_at_the_first_correct_answer():
    solver = Scripted(["x = 1, -1", "x = 99"])
    traj = run_trajectory(PROBLEM, solver, random.Random(0))
    assert traj.solved and traj.turns_used == 1
    assert len(solver.calls) == 1, "kept generating after a correct answer"


def test_loop_uses_its_full_budget_when_never_correct():
    solver = Scripted(["x = 99"])
    traj = run_trajectory(PROBLEM, solver, random.Random(0))
    assert not traj.solved
    assert traj.turns_used == MAX_TURNS == len(traj.turns) == len(solver.calls)


def test_turn_one_gets_no_feedback_and_later_turns_get_the_previous_turns():
    """The property that makes turn 2 a *revision* rather than a re-roll. If
    prior_feedback were always None, the loop would silently be running the
    pass@k control while reporting itself as self-refinement."""
    solver = Scripted(["x = -2/3, -1", "x = 99", "x = 1"])
    traj = run_trajectory(RATIONAL, solver, random.Random(0))

    assert solver.calls[0]["prior_feedback"] is None
    for turn_index in range(1, len(solver.calls)):
        passed = solver.calls[turn_index]["prior_feedback"]
        assert passed == traj.turns[turn_index - 1]["feedback"]
        assert passed, "empty feedback passed forward"

    # and the feedback carried real content, not a generic string
    assert traj.turns[0]["reason"] == "extraneous_root"
    assert "-1" in solver.calls[1]["prior_feedback"]


def test_every_attempt_is_recorded_with_its_verdict():
    """DPO pair construction reads these rows; a missing or mislabelled turn
    becomes a mislabelled preference pair."""
    solver = Scripted(["x = 99", "x = 1", "x = 1, -1"])
    traj = run_trajectory(PROBLEM, solver, random.Random(0))
    assert [t["turn"] for t in traj.turns] == [1, 2, 3]
    assert [t["correct"] for t in traj.turns] == [False, False, True]
    for row in traj.turns:
        assert verify(PROBLEM["equation_str"], row["candidate"]).correct == row["correct"]


def test_run_all_is_reproducible_for_a_given_seed():
    problems = generate_problems(n_per_family=4, seed=0)
    first = run_all(problems, flawed_attempt, seed=7)
    second = run_all(problems, flawed_attempt, seed=7)
    assert [t.turns for t in first] == [t.turns for t in second]


# ------------------------------------------------ the limitation, pinned --

def test_the_synthetic_solver_improves_without_reading_feedback():
    """THIS TEST DOCUMENTS A LIMITATION, AND IS MEANT TO PASS.

    `flawed_solver.attempt` ignores `prior_feedback` entirely; its error rate
    falls because `agent_loop` hands it a lower `severity` on each later turn.
    So in the CPU experiment, "accuracy rises with turn budget" is true *by
    construction* -- the solver was told to improve, and improved.

    An earlier version of this project reported that rise as evidence that
    agentic self-refinement works. It was circular, and the GPU experiment
    (results/colab_run/agent_experiment.md) that used a real LLM and a proper
    re-sampling control found no such benefit.

    The assertions below make that circularity executable, so the claim cannot
    be reconstructed by someone reading only the CPU numbers. If the synthetic
    solver ever does start reading feedback, this test fails and the person
    changing it has to decide, deliberately, what the CPU numbers now mean.
    """
    equation = PROBLEM["equation_str"]

    # 1. the feedback argument makes no difference whatsoever
    with_feedback = flawed_attempt(equation, random.Random(5), 3,
                                   prior_feedback="Missing solution(s): you have 1 root, "
                                                  "the equation has 2.")
    without = flawed_attempt(equation, random.Random(5), 3, prior_feedback=None)
    assert with_feedback == without

    # 2. the severity argument alone is what drives the improvement
    def accuracy(severity):
        rng = random.Random(0)
        hits = sum(verify(equation, flawed_attempt(equation, rng, severity)).correct
                   for _ in range(400))
        return hits / 400

    assert accuracy(1) > accuracy(3) + 0.15, (
        "the severity schedule no longer drives the CPU experiment's turn-budget "
        "effect -- re-read what the CPU numbers are claiming before changing this"
    )


def test_severity_schedule_is_the_one_the_loop_actually_applies():
    """Pins the schedule itself: turn 1 -> severity 3, turn 3 -> severity 1.
    An inverted schedule (bug #3) made later turns *more* error-prone while
    the write-up claimed the opposite."""
    solver = Scripted(["x = 99"])
    run_trajectory(PROBLEM, solver, random.Random(0))
    assert [c["severity"] for c in solver.calls] == [3, 2, 1]


@pytest.mark.parametrize("severity", [1, 2, 3])
def test_the_synthetic_solver_always_emits_something_the_verifier_can_read(severity):
    """The synthetic solver stands in for a model's error distribution, so its
    errors must be algebra errors -- not formatting errors that would show up
    as `unparsable` and confound the error-mode breakdown."""
    rng = random.Random(3)
    for problem in generate_problems(n_per_family=6, seed=0):
        candidate = flawed_attempt(problem["equation_str"], rng, severity)
        assert verify(problem["equation_str"], candidate).reason != "unparsable", candidate
