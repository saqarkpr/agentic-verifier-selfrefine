"""The Colab notebook is the one artefact that cannot be run here, so it is
linted instead.

A version of it once shipped with its last three sections duplicated, which
would have run checkpoint selection twice and silently overwritten the first
selection's log. Nothing caught it, because nothing looked.
"""
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = json.load(open(os.path.join(ROOT, "notebooks", "colab_clean_run.ipynb")))
CELLS = NB["cells"]
CODE = ["".join(c["source"]) for c in CELLS if c["cell_type"] == "code"]
MARKDOWN = ["".join(c["source"]) for c in CELLS if c["cell_type"] == "markdown"]
HEADINGS = [m.splitlines()[0] for m in MARKDOWN if m.startswith("## ")]


def _cells_running(fragment):
    return [c for c in CODE if any(fragment in line and line.lstrip().startswith("!")
                                   for line in c.splitlines())]


def test_sections_are_numbered_once_and_in_order():
    numbers = [int(re.match(r"## (\d+)\.", h).group(1)) for h in HEADINGS]
    assert numbers == list(range(1, len(numbers) + 1)), HEADINGS


def test_no_code_cell_is_repeated_by_accident():
    """The agent loop and the ladder legitimately run twice, on two different
    models, and are told apart by their backup tags. Nothing else may repeat."""
    assert len(CODE) == len(set(CODE)), [c[:60] for c in CODE if CODE.count(c) > 1]


@pytest.mark.parametrize("fragment, times", [
    ("rerun_all.py sft dpo eval --repeat=4", 1),
    ("rerun_all.py select", 1),
    ("smoke_test.py", 1),
    ("rerun_all.py agent", 2),
    ("rerun_all.py ablation", 2),
])
def test_each_stage_runs_the_intended_number_of_times(fragment, times):
    assert len(_cells_running(fragment)) == times


def test_every_gpu_stage_is_followed_by_a_failure_check_or_a_gate():
    """A failed `!command` is only red text; the next cell starts anyway. Each
    stage must stop the run itself -- or, for selection, be followed by the
    gate that reads its log."""
    for cell in _cells_running("rerun_all.py") + _cells_running("smoke_test.py"):
        if "rerun_all.py select" in cell:
            continue
        assert "check(" in cell, cell[:80]
    position = next(i for i, c in enumerate(CODE) if "rerun_all.py select" in c)
    assert "log['kept'] is not None" in CODE[position + 1], "selection is not followed by its gate"


def test_every_expensive_stage_is_backed_up_before_its_check():
    """Backup first, so a stage that failed half-way still leaves what it
    produced; and after every stage, so a dropped session costs one stage."""
    for fragment in ("--repeat=4", "rerun_all.py agent", "rerun_all.py ablation",
                     "rerun_all.py select", "rerun_all.py eval"):
        for cell in _cells_running(fragment):
            assert "backup(" in cell, cell[:80]
            if "check(" in cell:
                assert cell.index("backup(") < cell.index("check("), cell[:80]
    tags = re.findall(r"backup\('([^']+)'\)", "\n".join(CODE))
    assert len(tags) == len(set(tags)), tags
    assert tags[-1] == "final"


def test_the_smoke_test_and_the_gates_come_before_any_training():
    order = {name: next(i for i, c in enumerate(CODE) if needle in c) for name, needle in {
        "install": "requirements-colab.txt -r",
        "tests": "python -m pytest",
        "smoke": "python smoke_test.py",     # the cell that runs it, not the unpack check
        "split gate": "split is contaminated",
        "clean slate": "nothing stale on disk",
        "training": "--repeat=4",
    }.items()}
    assert list(order.values()) == sorted(order.values()), order


def test_the_install_is_pinned_and_leaves_torch_alone():
    install = next(c for c in CODE if "requirements-colab.txt -r" in c)
    assert "install -U" not in "\n".join(CODE), "an unpinned upgrade is back in the notebook"
    assert not re.search(r"pip[^\n]*install[^\n]*\btorch\b(?!ao)", install)
    assert "pip freeze" in install


def test_the_selected_checkpoint_is_scored_apart_from_the_passes():
    """A chosen model is not a fifth draw from the same distribution."""
    cell = next(c for c in CODE if "AVS_EVAL_DIR" in c)
    assert "AVS_EVAL_DIR=../results/raw/selected_eval python rerun_all.py eval" in cell
    src = open(os.path.join(ROOT, "src", "evaluate_redesign_v2.py")).read()
    assert 'os.environ.get("AVS_EVAL_DIR"' in src


def test_shell_commands_use_absolute_paths():
    """After a runtime restart the working directory is /content again, and a
    relative `cd src` would run nothing."""
    for cell in CODE:
        for line in cell.splitlines():
            if line.startswith("!cd "):
                assert line.startswith("!cd /content/agentic-verifier-selfrefine"), line


def test_pure_python_cells_compile():
    for cell in CODE:
        python = "\n".join(l for l in cell.splitlines() if not l.lstrip().startswith("!"))
        compile(python, "<notebook cell>", "exec")


def test_the_zip_ships_no_outputs_where_a_new_run_writes():
    """The clean-slate gate asserts this at run time; this asserts it for the
    repository, so a fresh run can never be mistaken for an old one."""
    import glob

    for pattern in ("results/raw/eval/*", "results/llm_agent_experiment*",
                    "results/feedback_protocol_ablation*",
                    "results/raw/checkpoint_selection.json", "results/raw/run_manifest*"):
        assert not glob.glob(os.path.join(ROOT, pattern)), pattern
