"""Choosing a checkpoint without choosing the result.

The DPO stage collapses on about half of its training runs, so the experiments
that need a working model have to pick one. That choice is the kind of step
that quietly decides a headline number, so the rule and the data it looks at
are pinned here.
"""
import pytest

import splits
from select_checkpoint import DEFAULT_MAX_DROP, is_collapsed
from tasks import contamination


# The four passes on record (results/recorded/2026-09-22/eval/summary_*.json):
# two healthy, two collapsed, nothing in between.
@pytest.mark.parametrize("sft, sft_dpo, collapsed", [
    (0.289, 0.289, False),
    (0.267, 0.289, False),
    (0.311, 0.133, True),
    (0.289, 0.133, True),
])
def test_the_rule_separates_the_recorded_passes(sft, sft_dpo, collapsed):
    assert is_collapsed(sft, sft_dpo) is collapsed


def test_the_rule_is_relative_to_the_same_attempts_sft_rate():
    """A fixed floor calibrated on the benchmark does not transfer to the
    validation set, whose scores run higher because part of it overlaps the
    training data. The same absolute DPO rate must be judged differently
    depending on what the SFT model it started from could do."""
    assert not is_collapsed(0.30, 0.28)      # lost one problem: fine
    assert is_collapsed(0.50, 0.28)          # same DPO rate, but it lost ten
    assert not is_collapsed(0.50, 0.50 - DEFAULT_MAX_DROP)   # the boundary is kept


def test_an_absolute_floor_is_available_but_off_by_default():
    assert not is_collapsed(0.10, 0.10)
    assert is_collapsed(0.10, 0.10, floor=0.2)


def test_selection_never_looks_at_the_benchmark():
    """Retraining until the benchmark score looks good is selection on the
    test set. The validation set must share nothing with it."""
    validation, held_out = splits.validation(), splits.held_out()
    assert len(validation) == 45
    assert contamination(validation, held_out)["n_overlapping"] == 0


def test_validation_overlaps_training_and_that_is_on_the_record():
    """Deliberately not excluded from training (see splits.validation). The
    consequence is that validation rates rank checkpoints and are never
    quoted, and the size of the overlap is pinned so that statement stays
    true to the data."""
    overlap = contamination(splits.training(100), splits.validation())
    assert overlap["n_overlapping"] == 9
    assert overlap["by_family"] == {"linear": 8, "quadratic": 1}
