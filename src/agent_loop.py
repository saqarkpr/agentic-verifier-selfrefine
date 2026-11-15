"""
Multi-turn agentic revision loop.

For each problem: solver proposes an answer -> verifier checks it -> if
wrong, the verifier's structured feedback is appended to context and the
solver gets another attempt, up to MAX_TURNS. Every attempt (correct or
not) is logged, so a full trajectory can be turned into a DPO preference
pair later: the final correct attempt in a trajectory that started wrong is
the "chosen" continuation, the first (unrevised) attempt is "rejected".

solve_fn(equation_str, rng, severity, prior_feedback=None) -> candidate_str
is pluggable: this file is run against flawed_solver.attempt() for the CPU
experiment, and the same interface (see hf_solver.py) plugs in a real HF
causal LM for the Colab run. prior_feedback is the verifier's feedback
string from the immediately preceding turn (None on turn 1) -- this is
what actually makes turn 2/3 a *revision* rather than an independent
re-attempt. flawed_solver.attempt() accepts and ignores it (its synthetic
error model already encodes "feedback helps" through the severity
schedule instead, see flawed_solver.py); hf_solver.attempt() uses it to
build a revision-style prompt for a real LLM.
"""
import random
from dataclasses import dataclass, field

from verifier import verify

MAX_TURNS = 3


@dataclass
class Trajectory:
    problem_id: str
    family: str
    equation_str: str
    turns: list = field(default_factory=list)  # list of dicts: {turn, candidate, correct, reason, feedback}
    solved: bool = False
    turns_used: int = 0


def run_trajectory(problem, solve_fn, rng):
    traj = Trajectory(problem["id"], problem["family"], problem["equation_str"])
    last_feedback = None
    for turn in range(1, MAX_TURNS + 1):
        severity = max(1, 4 - turn)  # fewer errors as feedback accumulates
        candidate = solve_fn(problem["equation_str"], rng, severity, prior_feedback=last_feedback)
        result = verify(problem["equation_str"], candidate)
        traj.turns.append(
            {
                "turn": turn,
                "candidate": candidate,
                "correct": result.correct,
                "reason": result.reason,
                "feedback": result.feedback,
            }
        )
        traj.turns_used = turn
        if result.correct:
            traj.solved = True
            break
        last_feedback = result.feedback
    return traj


def run_all(problems, solve_fn, seed=0):
    rng = random.Random(seed)
    return [run_trajectory(p, solve_fn, rng) for p in problems]


if __name__ == "__main__":
    from tasks import generate_problems
    from flawed_solver import attempt as flawed_attempt

    probs = generate_problems(n_per_family=3, seed=0)
    trajs = run_all(probs, flawed_attempt, seed=1)
    for t in trajs:
        print(t.problem_id, "solved" if t.solved else "FAILED", f"in {t.turns_used} turn(s)")
