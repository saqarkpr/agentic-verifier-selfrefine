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


def generate_problems(n_per_family=20, seed=0):
    """Deterministic problem set. Returns a list of problem dicts."""
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
            if key in seen:
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


if __name__ == "__main__":
    probs = generate_problems(n_per_family=5, seed=0)
    for p in probs:
        print(p["id"], p["equation_str"])
