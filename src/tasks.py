"""
Task generator: symbolic equation-solving problems, verifiable with SymPy.

Kept in the same spirit as the thesis (Fredholm integral equations solved via
symbolic RL) and RLVF (DPO with a programmatic SymPy verifier): every problem
has an exact, checkable closed-form answer, so correctness never depends on
human judgment or an LLM-as-judge.

Problem families (increasing difficulty):
  - linear:      a*x + b = 0
  - quadratic:   a*x**2 + b*x + c = 0   (may have 0, 1, or 2 real roots)
  - rational:    (a*x + b) / (x + c) = d   (has a domain restriction x != -c,
                 the most common source of "extraneous root" agent errors)

Each problem is a dict: {id, family, equation_str, symbol, seed}.
Ground truth is never given to the solver; only recovered by the verifier
via sympy.solveset, which is why the verifier -- not a stored answer key --
is the source of truth (mirrors the thesis's use of a numerical baseline
rather than a lookup table).
"""
import random
import sympy as sp

x = sp.symbols("x")


def _linear(rng):
    a = rng.choice([i for i in range(-6, 7) if i != 0])
    b = rng.randint(-10, 10)
    return sp.Eq(a * x + b, 0)


def _quadratic(rng):
    a = rng.choice([i for i in range(-4, 5) if i != 0])
    b = rng.randint(-8, 8)
    c = rng.randint(-8, 8)
    return sp.Eq(a * x**2 + b * x + c, 0)


def _rational(rng):
    a = rng.choice([i for i in range(-5, 6) if i != 0])
    b = rng.randint(-6, 6)
    c = rng.choice([i for i in range(-6, 7) if i != 0])
    d = rng.randint(-5, 5)
    if a == d and b == d * c:
        # identity equation (true for all x != -c): not a valid problem, force a retry
        return sp.true
    return sp.Eq((a * x + b) / (x + c), d)


_FAMILIES = {"linear": _linear, "quadratic": _quadratic, "rational": _rational}

# Fixed per-family offsets, NOT Python's built-in hash(). str/hash() is salted
# per-process (PYTHONHASHSEED) unless explicitly disabled, so two separate
# interpreter runs of generate_problems(seed=0) could silently produce
# different problem sets and therefore a different number of DPO pairs, even
# though every call claims the same "seed=0". Discovered when the sandbox run
# (197 pairs) and a fresh Colab run of the identical code (196 pairs)
# disagreed; see PROVENANCE.md.
_FAMILY_OFFSET = {"linear": 0, "quadratic": 1, "rational": 2}


def generate_problems(n_per_family=20, seed=0, exclude_equations=None):
    """Deterministic problem set. Returns a list of problem dicts.

    `exclude_equations` skips any equation in the given set. It exists because
    a different `seed` does NOT guarantee a disjoint split: the `linear` family
    has only 12 x 21 = 252 distinct equations, so drawing 100 for training and
    15 for evaluation collides by arithmetic. Measured on the shipped data,
    6 of the 45 held-out equations (13.3%) also appear in sft_pairs.jsonl with
    their correct answers -- see PROVENANCE.md, bug #7, and the contamination
    note under the headline result in README.md.

    The default is None on purpose. Every recorded result in results/ was
    produced without this filter, and turning it on by default would silently
    change the held-out set out from under numbers already written down. New
    runs should pass it; see tests/test_splits_and_data.py for the intended use.
    """
    exclude = set(exclude_equations or ())
    problems = []
    pid = 0
    for family, fn in _FAMILIES.items():
        rng = random.Random(seed * 1000 + _FAMILY_OFFSET[family])
        seen = set()
        tries = 0
        while len([p for p in problems if p["family"] == family]) < n_per_family:
            tries += 1
            if tries > n_per_family * 50:
                break
            eq = fn(rng)
            if eq is sp.true or eq is sp.false or not isinstance(eq, sp.Eq):
                continue  # degenerate/identity equation, retry
            key = str(eq)
            if key in seen or key in exclude:
                continue
            seen.add(key)
            problems.append(
                {
                    "id": f"{family}_{pid:04d}",
                    "family": family,
                    "equation_str": key,
                    "symbol": "x",
                }
            )
            pid += 1
    return problems


def contamination(train_problems, held_out_problems):
    """How much of `held_out_problems` also appears in `train_problems`.

    Exists so that nobody computes this by hand again. The obvious way --
    intersecting `problem_id` -- is wrong, and wrong in the direction that
    makes the project look worse than it is. Ids are assigned by position
    within a draw, so the first linear problem is `linear_0000` whatever
    equation it happens to be. The 15-per-family held-out draw and the
    100-per-family training draw therefore share the ids
    `linear_0000`..`linear_0014` while not one of them is the same equation,
    and an id-based comparison reports 15 of 45 (33%) where the truth is
    6 of 45 (13.3%).

    Contamination is a property of the equations. This compares those.
    """
    from collections import Counter

    train_equations = {p["equation_str"] for p in train_problems}
    overlapping = [p for p in held_out_problems
                   if p["equation_str"] in train_equations]
    return {
        "n_held_out": len(held_out_problems),
        "n_overlapping": len(overlapping),
        "fraction": len(overlapping) / len(held_out_problems) if held_out_problems else 0.0,
        "equations": sorted(p["equation_str"] for p in overlapping),
        "by_family": dict(Counter(p["family"] for p in overlapping)),
    }


if __name__ == "__main__":
    probs = generate_problems(n_per_family=5, seed=0)
    for p in probs:
        print(p["id"], p["equation_str"])
