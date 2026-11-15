"""The GPU stack is pinned to what actually ran, and the smoke test checks it.

An unpinned `pip install -U` installs whatever is newest on the day of the run.
The training scripts sit directly on APIs that move between releases, so the
honest statement of "this code works" names the versions it worked with.
"""
import json
import os

import smoke_test

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = json.load(open(os.path.join(ROOT, "results", "recorded", "manifest.json")))


def test_the_pins_are_the_versions_the_recorded_run_used():
    """A pin is a claim that the code was run against that version. This ties
    the claim to the environment written down by the run itself, so the two
    cannot drift apart."""
    recorded = MANIFEST["clean_split_rerun"]["environment"]
    pins = smoke_test.read_pins()
    assert set(pins) == {"trl", "peft", "transformers"}
    for name, version in pins.items():
        assert recorded[name] == version, (name, version, recorded[name])


def test_torch_is_deliberately_not_pinned():
    """Colab ships a torch build matched to its own CUDA runtime. Replacing it
    is the install most likely to break the GPU, so it is left alone and the
    version in use is recorded per run instead."""
    assert "torch" not in smoke_test.read_pins()


def test_a_version_mismatch_is_reported_not_ignored():
    pins = {"trl": "1.13.0", "peft": "0.21.0"}
    assert smoke_test.mismatches(pins, {"trl": "1.13.0", "peft": "0.21.0"}) == {}
    assert smoke_test.mismatches(pins, {"trl": "1.14.0", "peft": "0.21.0"}) == \
        {"trl": ("1.13.0", "1.14.0")}
    assert smoke_test.mismatches(pins, {"trl": "1.13.0"}) == {"peft": ("0.21.0", None)}


def test_the_smoke_test_overrides_epochs_through_the_real_config_layer():
    """It must run the real training scripts with one setting changed, not a
    miniature of them. The override name has to be one the scripts read."""
    src = open(os.path.join(ROOT, "src", "smoke_test.py")).read()
    assert 'AVS_NUM_TRAIN_EPOCHS="1"' in src
    for script in ("train_lora_sft.py", "train_lora_dpo_zeroshot.py"):
        assert script in src
        body = open(os.path.join(ROOT, "src", script)).read()
        assert 'avs_config.get("num_train_epochs"' in body, script
