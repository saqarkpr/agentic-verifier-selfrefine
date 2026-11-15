"""Bug 8: the DPO preference data silently contained no no-solution problems.

`flawed_solver` has no way to be wrong about an equation with no real root --
its "wrong" answer there is the empty set, which IS the right answer -- so the
builder dropped every such pair as chosen == rejected. Nothing crashed, the
pair count looked plausible (118), and every individual pair was valid. The
defect is in what is ABSENT, which no per-row test can see.

What DPO then did with that data is the main finding of the paper: the
probability it removed from wrong numeric answers went to `no solution`, an
answer that appears on neither side of any pair (86% of solvable held-out
quadratics answered `no solution` after DPO).

These tests pin the defect in the v1 file (which every recorded run trained on,
so it must not change) and the coverage of the v2 fix.
"""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "data", "dpo_pairs_zeroshot.jsonl")
V2 = os.path.join(ROOT, "data", "dpo_pairs_zeroshot_v2.jsonl")


def _rows(path):
    return [json.loads(line) for line in open(path)]


def _final(text):
    return text.strip().splitlines()[-1].strip()


def test_v1_contains_no_no_solution_pair_on_either_side():
    rows = _rows(V1)
    assert len(rows) == 118
    assert not any(_final(r["chosen"]) == "no solution" for r in rows)
    assert not any(_final(r["rejected"]) == "no solution" for r in rows)


def test_v1_dropped_every_no_solution_training_problem():
    import build_dpo_pairs_zeroshot as b
    import splits
    from reference_solution import _true_answer_line

    problems = splits.training(n_per_family=b.N_PER_FAMILY, seed=b.SEED)
    no_sol = {p["id"] for p in problems if _true_answer_line(p["equation_str"]) == "no solution"}
    assert len(no_sol) == 25, "the training draw changed; every number in the paper depends on it"
    kept = {r["problem_id"] for r in _rows(V1)}
    assert not (no_sol & kept), "v1 is supposed to be the defective file the GPU runs used"


def test_flawed_solver_cannot_be_wrong_without_a_real_root():
    """The root cause, stated as behaviour."""
    import random
    from flawed_solver import attempt

    rng = random.Random(0)
    eq = "Eq(x**2 + 1, 0)"
    assert {attempt(eq, rng, severity=3) for _ in range(50)} == {"no solution"}


def test_v1_builder_output_is_unchanged():
    import build_dpo_pairs_zeroshot as b
    assert b.build_rows("v1") == _rows(V1)


@pytest.fixture(scope="module")
def v2_rows():
    import build_dpo_pairs_zeroshot as b
    rows = b.build_rows("v2")
    assert rows == _rows(V2), "data/dpo_pairs_zeroshot_v2.jsonl is stale -- rebuild it"
    return rows


def test_v2_balances_no_solution_on_both_sides(v2_rows):
    chosen = sum(_final(r["chosen"]) == "no solution" for r in v2_rows)
    rejected = sum(_final(r["rejected"]) == "no solution" for r in v2_rows)
    assert chosen == rejected == 25


def test_v2_covers_no_solution_rejections_in_every_family(v2_rows):
    fams = {r["family"] for r in v2_rows if _final(r["rejected"]) == "no solution"}
    assert fams == {"linear", "quadratic", "rational"}


def test_every_v2_pair_prefers_a_correct_response_over_a_wrong_one(v2_rows):
    from verifier import verify
    for r in v2_rows:
        assert verify(r["equation_str"], _final(r["chosen"])).correct, r["problem_id"]
        assert not verify(r["equation_str"], _final(r["rejected"])).correct, r["problem_id"]


def test_v2_is_a_superset_of_v1(v2_rows):
    v1 = {(r["problem_id"], r["chosen"], r["rejected"]) for r in _rows(V1)}
    v2 = {(r["problem_id"], r["chosen"], r["rejected"]) for r in v2_rows}
    assert v1 <= v2
