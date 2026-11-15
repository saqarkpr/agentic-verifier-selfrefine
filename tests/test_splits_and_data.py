"""Train/held-out separation, and the integrity of the shipped datasets.

The most damaging error in this project was not a crash. It was a sentence in
results/colab_run/redesign_v2.md asserting "no leakage into the seed=99
held-out set", which was wrong: 6 of the 45 held-out equations appear in
data/sft_pairs.jsonl with their correct answers. Nothing failed, nothing
looked odd, and the number it protected -- base 0/45 -> SFT 13/45 -- was
quoted everywhere in the repo.

A different `seed` does not produce a disjoint split. The `linear` family has
only 12 x 21 = 252 distinct equations, so drawing 100 per family for training
and 15 for evaluation collides by arithmetic. These tests measure the
contamination against the files actually shipped, pin it so it cannot grow
unnoticed, and check that the fix works for future runs.
"""
import json
import os

import pytest

from tasks import generate_problems

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELD_OUT = generate_problems(n_per_family=15, seed=99)
HELD_OUT_EQUATIONS = {p["equation_str"] for p in HELD_OUT}


def _equations(filename):
    path = os.path.join(ROOT, "data", filename)
    rows = [json.loads(line) for line in open(path)]
    if "equation_str" in rows[0]:
        return {r["equation_str"] for r in rows}, rows
    return None, rows  # the legacy revision-format file carries no equation field


# ------------------------------------------------------- the contamination --

@pytest.mark.parametrize("filename", ["sft_pairs.jsonl", "dpo_pairs_zeroshot.jsonl"])
def test_the_shipped_training_data_is_disjoint_from_the_held_out_set(filename):
    """Asserted to be exactly zero, on the files actually shipped.

    It was not always zero. The builders used a bare `generate_problems(seed=0)`
    and trusted that a different seed gave a disjoint sample; 6 held-out
    equations landed in sft_pairs.jsonl and 3 in dpo_pairs_zeroshot.jsonl, with
    their correct answers, while the write-up said there was no leakage. Both
    builders now draw through `splits.training()`, which excludes the held-out
    equations by construction and asserts the result inside the build itself.
    """
    equations, _ = _equations(filename)
    overlap = HELD_OUT_EQUATIONS & equations
    assert overlap == set(), (
        f"{filename} shares {len(overlap)} equations with the held-out set: "
        f"{sorted(overlap)}. The split must be built with splits.training()."
    )


def test_the_contamination_the_old_split_would_have_had_is_still_pinned():
    """The historical fact, kept measurable rather than only described.

    If someone reverts to a bare `generate_problems(seed=0)` this is the
    damage they reintroduce, and it is the number the superseded results in
    results/recorded/manifest.json were produced under."""
    import splits
    from tasks import contamination

    for n_per_family, expected, families in ((100, 6, {"linear": 5, "rational": 1}),
                                             (60, 3, {"linear": 3})):
        unfiltered = splits.training(n_per_family, strict=False)
        report = contamination(unfiltered, HELD_OUT)
        assert report["n_overlapping"] == expected
        assert report["by_family"] == families


@pytest.mark.parametrize("n_per_family", [100, 60, 15])
def test_the_clean_split_is_disjoint_and_complete(n_per_family):
    """`splits.training()` must yield a full-sized draw, not quietly return
    fewer problems because the exclusion exhausted a family."""
    import splits
    from tasks import contamination

    train = splits.training(n_per_family)
    assert len(train) == n_per_family * 3
    assert contamination(train, splits.held_out())["n_overlapping"] == 0


def test_the_held_out_set_did_not_move_when_the_split_was_cleaned():
    """The benchmark is canonical. Fixing the training draw must not change
    which 45 problems the models are scored on, or the new numbers would not
    be comparable to the superseded ones at all."""
    import splits

    assert [p["equation_str"] for p in splits.held_out()] == \
           [p["equation_str"] for p in HELD_OUT]


@pytest.mark.parametrize("n_per_family, seed, digest, clean", [
    (15, 99, "375f98ae198067a5493231b1616a874b", False),  # held-out benchmark, never moves
    (100, 0, "6aa8396c7d5e2c13c277ced70d45dce1", True),   # SFT training draw, held-out excluded
    (60, 0, "64df62db8431ca9dd52f365a0af92257", True),    # DPO training draw, held-out excluded
    (100, 0, "f0a9029c9cdbad25e5f0a387d2925a75", False),  # the contaminated draw, superseded
    (60, 0, "2a51ff1971791cc5517540337df39fd3", False),   # ditto
])
def test_the_canonical_splits_are_frozen(n_per_family, seed, digest, clean):
    """Every number in results/ refers to these exact problem sets. Any change
    to the generator -- including the `exclude_equations` parameter added for
    future runs -- must leave them byte-identical, or the recorded results
    silently start describing a different benchmark.

    Pinned as a digest rather than a spot check: a spot check on one equation
    passes while the other 44 move."""
    import hashlib

    import splits

    problems = (splits.training(n_per_family, seed) if clean
                else generate_problems(n_per_family=n_per_family, seed=seed))
    actual = hashlib.sha256(
        "|".join(f"{p['id']}:{p['equation_str']}" for p in problems).encode()
    ).hexdigest()[:32]
    assert actual == digest


# --------------------------------------------------------- dataset integrity --

def test_every_sft_target_is_actually_correct():
    """The SFT targets are templated, with the final answer taken from the
    verifier's own ground truth. If any target were wrong, the model would be
    trained on it -- and the verifier would then mark the model wrong for
    reproducing exactly what it was taught."""
    from verifier import verify

    _, rows = _equations("sft_pairs.jsonl")
    for row in rows:
        answer = row["messages"][-1]["content"].strip().splitlines()[-1]
        assert verify(row["equation_str"], answer).correct, (row["problem_id"], answer)


def test_every_dpo_pair_prefers_a_correct_response_over_a_wrong_one():
    """A preference pair whose `rejected` is also correct teaches the model to
    dislike a right answer."""
    from verifier import verify

    _, rows = _equations("dpo_pairs_zeroshot.jsonl")
    for row in rows:
        chosen = row["chosen"].strip().splitlines()[-1]
        rejected = row["rejected"].strip().splitlines()[-1]
        assert verify(row["equation_str"], chosen).correct, row["problem_id"]
        assert not verify(row["equation_str"], rejected).correct, row["problem_id"]


def test_dpo_pairs_differ_only_in_their_final_line():
    """The deliberate design that isolates 'is this answer right' from 'is this
    response longer'. If the reasoning text ever differs, DPO can learn a
    length or style preference instead, and the stage stops measuring what the
    write-up says it measures."""
    _, rows = _equations("dpo_pairs_zeroshot.jsonl")
    for row in rows:
        assert row["chosen"].rsplit("\n", 1)[0] == row["rejected"].rsplit("\n", 1)[0], \
            row["problem_id"]


def test_the_training_prompt_matches_the_evaluation_prompt_exactly():
    """The first null result in this project was traced to a train/eval prompt
    mismatch. Both files now build the user turn from the same template; this
    pins that they still agree, byte for byte."""
    import build_dpo_pairs_zeroshot as dpo
    import build_sft_data as sft

    equation = "Eq(3*x + 6, 0)"
    assert sft.user_prompt(equation) == dpo.user_prompt(equation)
    assert sft.SYSTEM == dpo.SYSTEM

    _, sft_rows = _equations("sft_pairs.jsonl")
    _, dpo_rows = _equations("dpo_pairs_zeroshot.jsonl")
    assert sft_rows[0]["messages"][0]["content"] == dpo_rows[0]["system"]

    by_equation = {r["equation_str"]: r for r in dpo_rows}
    for row in sft_rows:
        other = by_equation.get(row["equation_str"])
        if other is not None:
            assert row["messages"][1]["content"] == other["user"], row["problem_id"]


def test_shipped_data_files_are_reproducible_from_source():
    """`python build_sft_data.py` must still produce the file in data/. A
    dataset nobody can regenerate is a dataset nobody can check."""
    import build_dpo_pairs_zeroshot as dpo
    import build_sft_data as sft

    _, sft_rows = _equations("sft_pairs.jsonl")
    assert sft.build_rows() == sft_rows

    _, dpo_rows = _equations("dpo_pairs_zeroshot.jsonl")
    assert dpo.build_rows() == dpo_rows


# ------------------------------------------- the identifier trap, pinned --

def test_problem_id_is_positional_and_means_nothing_across_draws():
    """The trap this exists for, found the expensive way.

    `problem_id` is assigned by position within a draw: the first linear
    problem is always `linear_0000`, whatever equation it happens to be. So
    comparing training and held-out sets BY ID measures how the numbering
    lines up, not how the equations overlap. Done that way, the contamination
    reads as 15 of 45 (33%) instead of the true 6 of 45 (13.3%), because the
    held-out draw and the 100-per-family training draw share the ids
    `linear_0000`..`linear_0014` and not one of them is the same equation.

    Contamination is a property of the equations. Always compare
    `equation_str`.
    """
    train = generate_problems(n_per_family=100, seed=0)
    by_id_train = {p["id"]: p["equation_str"] for p in train}
    by_id_held = {p["id"]: p["equation_str"] for p in HELD_OUT}

    shared_ids = set(by_id_train) & set(by_id_held)
    assert len(shared_ids) == 15, "the id collision this test warns about has changed"
    assert not any(by_id_train[i] == by_id_held[i] for i in shared_ids), (
        "an id now denotes the same equation in both draws; if problem_id has "
        "been made stable, update this test and the comparison helper"
    )

    by_equation = {p["equation_str"] for p in train} & HELD_OUT_EQUATIONS
    assert len(by_equation) == 6   # the unfiltered draw; splits.training() gives 0
    assert len(shared_ids) != len(by_equation), (
        "id-based and equation-based comparison now agree, which would make "
        "this test vacuous"
    )


def test_contamination_helper_compares_equations_not_ids():
    """One helper, so nobody hand-rolls the comparison and gets it wrong."""
    from tasks import contamination

    train = generate_problems(n_per_family=100, seed=0)   # unfiltered, on purpose
    report = contamination(train, HELD_OUT)
    assert report["n_overlapping"] == 6
    assert report["fraction"] == pytest.approx(6 / 45)
    assert all(e.startswith("Eq(") for e in report["equations"])
    assert report["by_family"] == {"linear": 5, "rational": 1}
