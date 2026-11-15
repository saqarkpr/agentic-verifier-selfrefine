"""The documentation may not quote a number the manifest does not hold.

Bug 7 was a sentence, not a defect in code: `redesign_v2.md` asserted the
held-out set was clean, nothing checked it, and the claim guarded the most
quoted figure in the repo. The general fix is not "read more carefully". It
is to give every reported number exactly one home -- results/recorded/manifest.json
-- have the figures read from it, and make the tables that restate it fail a
test when they disagree.

These tests parse the actual markdown tables in README.md. That is a little
brittle by design: a table that moves has to be re-pointed here, which is a
cheaper failure than a table that quietly says something else.
"""
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = json.load(open(os.path.join(ROOT, "results", "recorded", "manifest.json")))
README = open(os.path.join(ROOT, "README.md")).read()


def _table_after(heading_fragment, header_fragment):
    """The first markdown table following a heading, as a list of cell-lists."""
    start = README.index(heading_fragment)
    body = README[start:]
    head = body.index(header_fragment)
    rows = []
    for line in body[head:].splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    return rows


def _floats(cells):
    out = []
    for c in cells:
        m = re.search(r"-?\d+\.\d+", c.replace("**", ""))
        if m:
            out.append(float(m.group()))
    return out


# ------------------------------------------------------- the headline table --

def test_readme_headline_table_matches_the_manifest():
    rows = _table_after("## Superseded: the four-pass table", "| Model | Run A")
    by_label = {r[0].strip("* "): _floats(r[1:5]) for r in rows[1:]}
    e = MANIFEST["held_out_eval"]

    assert by_label["Base"] == e["base"]
    assert by_label["+ LoRA-SFT"] == e["sft"]
    assert by_label["+ LoRA-SFT + LoRA-DPO"] == e["sft_dpo"]


def test_readme_agent_tables_match_the_manifest():
    for heading, key in (("## The central question", "verbose"),
                         ("### The repair was attempted", "minimal")):
        rows = _table_after(heading, "| Turn budget |")
        run = MANIFEST["agent_experiment"]["runs"][key]
        feedback = [_floats(r[1:2])[0] for r in rows[1:]]
        control = [_floats(r[2:3])[0] for r in rows[1:]]
        assert feedback == run["feedback_by_budget"], (heading, feedback)
        assert control == run["control_by_budget"], (heading, control)


def test_readme_unparsable_table_matches_the_manifest():
    rows = _table_after("| turn | feedback unparsable |", "| turn | feedback unparsable |")
    run = MANIFEST["agent_experiment"]["runs"]["verbose"]["unparsable"]
    quoted = {r[0]: (_floats(r[1:2])[0] / 100, _floats(r[2:3])[0] / 100) for r in rows[1:]}
    assert quoted["2"] == (run["turn2"]["feedback"], run["turn2"]["control"])
    assert quoted["3"] == (run["turn3"]["feedback"], run["turn3"]["control"])


def test_the_table_parser_actually_found_tables():
    """Guard against the failure where every assertion above passes because
    the parser silently returned nothing."""
    assert len(_table_after("## Superseded: the four-pass table", "| Model | Run A")) == 4


# ---------------------------------------- provisional numbers must be flagged --

PENDING = [k for k, v in MANIFEST.items()
           if isinstance(v, dict) and v.get("status") == "pending_rerun"]


VALID_STATUSES = {"valid", "valid_one_checkpoint", "pending_rerun",
                  "superseded", "not_run"}
ENTRIES = {k: v for k, v in MANIFEST.items()
           if isinstance(v, dict) and "status" in v}


def test_every_entry_carries_a_recognised_status():
    """The status field is what decides how a number may be presented, so an
    unrecognised one is a number with no rules attached to it."""
    assert ENTRIES
    for key, entry in ENTRIES.items():
        assert entry["status"] in VALID_STATUSES, (key, entry["status"])


@pytest.mark.parametrize("key", sorted(
    k for k, v in MANIFEST.items()
    if isinstance(v, dict) and v.get("status") == "valid_one_checkpoint"))
def test_single_checkpoint_results_name_their_checkpoint_and_caveats(key):
    """`valid_one_checkpoint` is the most dangerous status in the file: the
    numbers are real and reproducible, and they still may not generalise one
    step. The clean-split agent experiment and ablation both ran on the pass-4
    model, which had collapsed to 0.133 greedy -- a damaged model leaves more
    room for feedback to recover, so the positive result they produced cannot
    be quoted without that fact attached."""
    entry = MANIFEST[key]
    assert entry.get("checkpoint"), f"{key} does not say which checkpoint"
    caveats = entry.get("caveat") or entry.get("caveats")
    assert caveats, f"{key} is a single-checkpoint result with no caveat"
    text = caveats if isinstance(caveats, str) else " ".join(caveats)
    assert "checkpoint" in text.lower()


def test_superseded_entries_name_their_successor():
    for key, entry in ENTRIES.items():
        if entry["status"] == "superseded":
            successor = entry.get("superseded_by")
            assert successor, f"{key} is superseded by nothing"
            assert successor in MANIFEST, f"{key} cites unknown successor {successor}"


@pytest.mark.parametrize("key", PENDING)
def test_every_pending_experiment_is_flagged_in_the_readme(key):
    """A number awaiting regeneration must not appear in the README without
    the reader being told. The whole point of the manifest is that a status
    lives beside the value rather than in someone's memory."""
    assert "## Status of the recorded numbers" in README, \
        "README has no status section; pending numbers would read as results"
    section = README[README.index("## Status of the recorded numbers"):]
    section = section[:section.index("\n## ", 3)]
    assert key in section, f"{key} is pending_rerun but the status section does not mention it"


@pytest.mark.parametrize("key", PENDING)
def test_pending_entries_say_how_to_regenerate_themselves(key):
    entry = MANIFEST[key]
    assert entry.get("invalidated_by"), f"{key} is pending but names no cause"
    assert entry.get("how_to_regenerate"), f"{key} is pending but names no remedy"
    for cause in entry["invalidated_by"]:
        assert cause in MANIFEST["defects"], f"{key} cites unknown defect {cause}"


def test_every_defect_records_its_direction_on_the_numbers():
    """'This might be wrong' is not a caveat. Each defect has to say which way
    it pushes a number, or say explicitly that the direction is undeterminable
    -- as bug 5 does for the agent experiment."""
    for name, defect in MANIFEST["defects"].items():
        directions = [v for k, v in defect.items() if k.startswith("direction_on")]
        assert directions, f"{name} states no direction"
        for text in directions:
            assert any(w in text.lower() for w in
                       ("understates", "overstates", "not determinable", "does not affect")), \
                f"{name}: direction is vague -- {text[:60]}"


# ------------------------------------------------ the CPU result is not pending --

def test_the_cpu_result_is_marked_valid_and_says_why():
    cpu = MANIFEST["cpu_turn_budget"]
    assert cpu["status"] == "valid"
    assert "flawed_solver" in cpu["why_valid"]
    assert cpu["standing_caveat"]


def test_the_cpu_numbers_match_the_committed_summary_csv():
    """`valid` has to mean reproducible, not merely unflagged."""
    import csv
    from collections import defaultdict

    path = os.path.join(ROOT, "results", "summary.csv")
    by_budget = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_budget[row["turn_budget"]].append(float(row["solve_rate"]))

    for budget, expected in MANIFEST["cpu_turn_budget"]["solve_rate_by_budget"].items():
        actual = sum(by_budget[budget]) / len(by_budget[budget])
        assert abs(actual - expected) < 0.0005, (budget, actual, expected)


# ----------------------------------- the manifest must match the raw evidence --

def test_the_clean_split_numbers_are_reproduced_by_the_committed_raw_files():
    """The manifest is the source the README and figures read from; this is
    what the manifest itself is checked against. Four per-pass summaries,
    written by the evaluation on the GPU, committed under results/recorded/."""
    import glob

    entry = MANIFEST["clean_split_rerun"]
    directory = os.path.join(ROOT, entry["raw_files"])
    summaries = [json.load(open(p))
                 for p in sorted(glob.glob(os.path.join(directory, "summary_*.json")))]
    assert len(summaries) == entry["passes"] == 4
    for stage in ("base", "sft", "sft_dpo"):
        assert [round(s[stage], 3) for s in summaries] == entry[stage], stage


def test_the_readme_clean_split_table_matches_the_manifest():
    rows = _table_after("### 1. The DPO stage does not work", "| | pass 1")
    by_label = {r[0].strip("* "): _floats(r[1:5]) for r in rows[1:]}
    entry = MANIFEST["clean_split_rerun"]
    assert by_label["Base"] == entry["base"]
    assert by_label["+ LoRA-SFT"] == entry["sft"]
    assert by_label["+ LoRA-SFT + LoRA-DPO"] == entry["sft_dpo"]
    assert by_label["DPO delta"] == entry["dpo_delta"]
