"""The canonical train / held-out split, defined in one place and enforced.

WHY THIS MODULE EXISTS

Until now the split was a convention: draw training data with `seed=0`, draw
the held-out set with `seed=99`, and assume two different seeds give two
disjoint sets. They do not. Independent draws from a finite family collide by
the birthday problem, and the `linear` family has only 12 x 21 = 252 distinct
equations:

    family      distinct equations    drawn for training
    linear                     252                   100
    quadratic                2,312                   100
    rational               ~17,160                   100

So 6 of the 45 held-out equations (5 linear, 1 rational) ended up in
`sft_pairs.jsonl` with their correct answers, and the write-up claimed there
was no leakage. See PROVENANCE.md, bug 7.

THE RULE THIS ENCODES

The held-out set is canonical and never moves: it is the benchmark, and a
benchmark that shifts when the training data changes measures nothing. The
training draw is what yields. `training()` therefore excludes every held-out
equation by construction and asserts the result, rather than leaving the
disjointness to be checked afterwards by whoever remembers to.

MEASURE CONTAMINATION BY EQUATION, NEVER BY ID

`problem_id` is positional within a draw: the first linear problem is
`linear_0000` whatever equation it happens to be. The held-out draw and the
100-per-family training draw therefore share the ids `linear_0000` through
`linear_0014` while not one of them is the same equation. Comparing ids
reports 15 of 45 (33%) where the truth is 6 of 45 (13.3%) -- an error that
makes the project look three times worse than it is, and that a person
checking for leakage is very likely to make. `tasks.contamination()` is the
one correct comparison; `tests/test_splits_and_data.py` pins the trap.
"""
from tasks import contamination, generate_problems

HELD_OUT_SEED = 99
HELD_OUT_N_PER_FAMILY = 15      # x3 families = 45 problems
VALIDATION_SEED = 7
VALIDATION_N_PER_FAMILY = 15
TRAIN_SEED = 0


def held_out():
    """The benchmark. Fixed forever; nothing may change this set."""
    return generate_problems(n_per_family=HELD_OUT_N_PER_FAMILY, seed=HELD_OUT_SEED)


def validation():
    """A RANKING set, used only to decide which checkpoint to keep.

    The DPO stage collapses on roughly half of its training runs, from 0.289
    to 0.133. Keeping a checkpoint that did not collapse is legitimate; picking
    it by its score on the benchmark is not, because that is selection on the
    test set and inflates the reported number invisibly. Hence a second set,
    disjoint from `held_out()`, that does the choosing.

    IT IS NOT EXCLUDED FROM TRAINING, AND THAT IS DELIBERATE.

    Excluding it would change the training draw, which would invalidate the
    four-pass table already measured on the current draw and force every
    result to be regenerated -- a real cost, for no benefit here. Selection
    only needs to RANK candidate checkpoints, and every candidate is trained
    on identical data, so whatever overlap exists is constant across them and
    cancels out of the comparison. The collapse it has to detect is a factor
    of two, not a subtle margin.

    The consequence, stated so it cannot be forgotten: a validation score is
    not a measurement of anything and must never be quoted as one. Only
    `held_out()` is, and it is scored once, afterwards, having played no part
    in the choice.
    """
    return generate_problems(n_per_family=VALIDATION_N_PER_FAMILY,
                             seed=VALIDATION_SEED,
                             exclude_equations={p["equation_str"] for p in held_out()})


def training(n_per_family, seed=TRAIN_SEED, strict=True):
    """A training draw guaranteed disjoint from `held_out()`.

    `strict` exists only so that the contaminated draw can still be
    reconstructed for comparison against the superseded results. Leave it
    True for anything that will be trained on.
    """
    if not strict:
        return generate_problems(n_per_family=n_per_family, seed=seed)

    excluded = {p["equation_str"] for p in held_out()}
    problems = generate_problems(n_per_family=n_per_family, seed=seed,
                                 exclude_equations=excluded)
    report = contamination(problems, held_out())
    # Asserted, not merely intended. The whole class of error this module
    # addresses is a disjointness that was believed rather than checked.
    assert report["n_overlapping"] == 0, (
        f"training draw still shares {report['n_overlapping']} equations with the "
        f"held-out set: {report['equations']}"
    )
    assert len(problems) == n_per_family * 3, (
        f"asked for {n_per_family * 3} problems, got {len(problems)}; the exclusion "
        f"has exhausted a family's equation space -- reduce n_per_family"
    )
    return problems


def report(n_per_family, seed=TRAIN_SEED):
    """One line a build script can print before it writes a dataset."""
    clean = training(n_per_family, seed)
    dirty = training(n_per_family, seed, strict=False)
    before = contamination(dirty, held_out())
    return (f"split: {len(clean)} training / {len(held_out())} held-out, "
            f"0 overlapping equations "
            f"(the unfiltered draw would have overlapped {before['n_overlapping']}: "
            f"{before['by_family'] or 'none'})")


if __name__ == "__main__":
    print(f"held-out   {len(held_out())} problems (seed {HELD_OUT_SEED})")
    print(f"validation {len(validation())} problems (seed {VALIDATION_SEED})")
    v = contamination(validation(), held_out())
    print(f"validation vs held-out overlap: {v['n_overlapping']}")
    print(report(100))
    print(report(60))
    for n in (100, 60):
        c = contamination(training(n), held_out())
        print(f"n_per_family={n}: overlap {c['n_overlapping']}/{c['n_held_out']}")
