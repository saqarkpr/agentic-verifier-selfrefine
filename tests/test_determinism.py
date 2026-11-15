"""Cross-process determinism: the two bugs that only exist between processes.

Two separate bugs in this repo produced different output from identical,
fully-seeded code, and neither could be caught by a test running in a single
interpreter -- which is exactly why both survived so long:

  bug #1  `hash(family)` seeded the per-family RNG. Python salts `str.__hash__`
          per process (PYTHONHASHSEED), so `generate_problems(seed=0)` built a
          different problem set in each run. Caught only because a sandbox run
          produced 197 DPO pairs and a Colab run of the same code produced 196.

  bug #4  `list(true_set)` iterated a Python set of sympy objects, whose hashes
          ultimately hash strings and so are salted the same way. 28 of 180
          problems iterated in a different order per process, so "drop a root"
          dropped a different root and the recorded candidate strings differed.
          Solve *rates* were unaffected -- the verifier compares sets -- which
          is precisely why it hid: the headline numbers were right and the
          artefacts were not reproducible.

So these tests re-run the generators in genuinely separate interpreters under
different hash seeds and require byte-identical output. Running the same code
twice in one process cannot fail this way and would be a test that only looks
like one.
"""
import os
import subprocess
import sys

import pytest

from conftest import SRC

HASH_SEEDS = ["1", "12345", "random"]


def _run(snippet, hashseed):
    env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONPATH=SRC)
    out = subprocess.run([sys.executable, "-c", snippet], env=env, cwd=SRC,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


PROBLEMS = """
from tasks import generate_problems
for p in generate_problems(n_per_family=20, seed=0):
    print(p["id"], p["family"], p["equation_str"], sep="|")
"""

CANDIDATES = """
import random
from tasks import generate_problems
from flawed_solver import attempt
rng = random.Random(0)
for p in generate_problems(n_per_family=20, seed=0):
    for severity in (3, 2, 1):
        print(p["id"], severity, attempt(p["equation_str"], rng, severity), sep="|")
"""

TRAJECTORIES = """
from tasks import generate_problems
from flawed_solver import attempt
from agent_loop import run_all
for t in run_all(generate_problems(n_per_family=10, seed=0), attempt, seed=1):
    for turn in t.turns:
        print(t.problem_id, turn["turn"], turn["candidate"], turn["reason"], sep="|")
"""


@pytest.mark.parametrize("name, snippet", [
    ("problem set", PROBLEMS),
    ("solver candidates", CANDIDATES),
    ("full trajectories", TRAJECTORIES),
])
def test_identical_across_hash_seeds(name, snippet):
    outputs = [_run(snippet, hs) for hs in HASH_SEEDS]
    first = outputs[0]
    assert first.strip(), f"{name}: produced no output"
    for hs, out in zip(HASH_SEEDS[1:], outputs[1:]):
        assert out == first, (
            f"{name} differs under PYTHONHASHSEED={hs}. Something is seeded by "
            f"a salted hash again -- see bug #1 and bug #4 in PROVENANCE.md."
        )


def test_problem_set_is_stable_across_seeds_but_not_identical():
    """A guard against the opposite failure: a 'deterministic' generator that
    ignores its seed would pass every test above while making the train/held-out
    split meaningless, because both splits would be the same problems."""
    from tasks import generate_problems

    train = {p["equation_str"] for p in generate_problems(n_per_family=15, seed=0)}
    held_out = {p["equation_str"] for p in generate_problems(n_per_family=15, seed=99)}
    assert train != held_out
    overlap = len(train & held_out) / len(held_out)
    # This compares two 15-per-family draws, which is NOT the train/eval
    # split the project actually uses -- that is 100 or 60 per family against
    # 15, and it is 13.3% / 4.4% contaminated. See tests/test_splits_and_data.py
    # for the real figures, measured against the shipped data files. The
    # assertion here only guards the weaker property that two seeds do not
    # collapse onto the same problem set.
    assert overlap < 0.10, f"held-out set is {overlap:.0%} contaminated by train"
