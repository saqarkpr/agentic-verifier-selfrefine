"""Config resolution: env > YAML > call-site default.

This matters for one reason. A Slurm array job sweeps a hyperparameter by
setting an environment variable per task, and the run dumps `resolved()`
beside its outputs. If the precedence were wrong, every task in the array
would silently train the same model and the dump would say so convincingly.
"""
import importlib
import json
import os

import pytest


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A freshly imported config module with no cached YAML and no AVS_* env."""
    for key in [k for k in os.environ if k.startswith("AVS_")]:
        monkeypatch.delenv(key, raising=False)
    import config
    module = importlib.reload(config)
    module._cache = None
    module._resolved = {}
    return module


def _write_yaml(tmp_path, body):
    path = tmp_path / "cfg.yaml"
    path.write_text(body)
    return str(path)


def test_default_is_used_when_nothing_else_is_set(cfg):
    assert cfg.get("learning_rate", 2e-4) == 2e-4
    assert cfg.resolved()["learning_rate"]["source"] == "default"


def test_yaml_overrides_the_default(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("AVS_CONFIG", _write_yaml(tmp_path, "learning_rate: 5e-5\n"))
    assert cfg.get("learning_rate", 2e-4) == 5e-5
    assert cfg.resolved()["learning_rate"]["source"] == "yaml"


def test_env_overrides_yaml(cfg, tmp_path, monkeypatch):
    """The precedence the Slurm sweep depends on."""
    monkeypatch.setenv("AVS_CONFIG", _write_yaml(tmp_path, "learning_rate: 5e-5\n"))
    monkeypatch.setenv("AVS_LEARNING_RATE", "1e-3")
    assert cfg.get("learning_rate", 2e-4) == 1e-3
    assert cfg.resolved()["learning_rate"]["source"] == "env"


def test_value_is_cast_to_the_type_of_the_default(cfg, monkeypatch):
    """Env vars are strings. An epoch count that stays "3" would be passed to
    the trainer as a string and fail deep inside it, or worse, not fail."""
    monkeypatch.setenv("AVS_NUM_EPOCHS", "3")
    value = cfg.get("num_epochs", 1)
    assert value == 3 and isinstance(value, int)


@pytest.mark.parametrize("raw, expected", [
    ("1", True), ("true", True), ("True", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("no", False), ("", False),
])
def test_bools_are_parsed_as_words_not_by_truthiness(cfg, monkeypatch, raw, expected):
    """`bool("false")` is True. A naive cast would turn AVS_BF16=false into
    bf16 enabled -- silently training in a different precision than the config
    file records."""
    monkeypatch.setenv("AVS_BF16", raw)
    assert cfg.get("bf16", False) is expected


def test_comments_and_quotes_in_yaml_are_stripped(cfg, tmp_path, monkeypatch):
    """The fallback parser (used when PyYAML is absent, as on a bare cluster
    node) has to handle the files actually in configs/."""
    monkeypatch.setenv("AVS_CONFIG", _write_yaml(
        tmp_path, 'model_name: "Qwen/Qwen2.5-0.5B-Instruct"  # base\n\n# a comment\n'))
    assert cfg.get("model_name", "other") == "Qwen/Qwen2.5-0.5B-Instruct"


def test_a_missing_config_file_is_not_fatal(cfg, monkeypatch):
    """A typo'd AVS_CONFIG on a cluster must not take the job down; the run
    falls back to defaults and `resolved()` records that it did."""
    monkeypatch.setenv("AVS_CONFIG", "/nonexistent/path.yaml")
    assert cfg.get("learning_rate", 2e-4) == 2e-4
    assert cfg.resolved()["learning_rate"]["source"] == "default"


def test_dump_records_every_setting_with_its_source(cfg, tmp_path, monkeypatch):
    """A run whose settings you cannot recover is a run you cannot reproduce."""
    monkeypatch.setenv("AVS_LEARNING_RATE", "1e-3")
    cfg.get("learning_rate", 2e-4)
    cfg.get("num_epochs", 2)

    path = cfg.dump(str(tmp_path / "out" / "resolved.json"))
    written = json.loads(open(path).read())
    assert written["learning_rate"] == {"value": 1e-3, "source": "env"}
    assert written["num_epochs"] == {"value": 2, "source": "default"}


def test_every_shipped_config_file_parses(cfg, monkeypatch):
    """configs/*.yaml are the reproducibility layer; one that does not parse
    is worse than none, because a run would start anyway on the defaults."""
    for path in _config_files():
        monkeypatch.setenv("AVS_CONFIG", path)
        cfg._cache = None
        loaded = cfg._load_yaml()
        assert loaded, f"{os.path.basename(path)} parsed to nothing"


def test_the_fallback_parser_agrees_with_pyyaml(cfg, monkeypatch):
    """A cluster node without PyYAML must resolve the same settings, not
    silently fall back to the defaults and record a config that was never
    used. Values come back as strings from the fallback and are cast at
    `get()`, so the comparison is on the string form."""
    import yaml

    for path in _config_files():
        with open(path) as f:
            expected = yaml.safe_load(f)
        monkeypatch.setenv("AVS_CONFIG", path)
        cfg._cache = None
        monkeypatch.setitem(__import__("sys").modules, "yaml", None)  # force ImportError
        try:
            fallback = cfg._load_yaml()
        finally:
            cfg._cache = None
        assert set(fallback) == set(expected), os.path.basename(path)
        for key, value in expected.items():
            assert _same_value(fallback[key], value), (os.path.basename(path), key,
                                                       fallback[key], value)


# --------------------------------------------- the layer must be connected --

def _config_files():
    import glob

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    files = sorted(glob.glob(os.path.join(root, "*.yaml")))
    assert files, "no config files found"
    return files


def _keys_read_by_scripts():
    """Every `avs_config.get("key", default)` appearing anywhere in src/."""
    import glob
    import re

    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    found = {}
    for path in glob.glob(os.path.join(src, "*.py")):
        for key, default in re.findall(r'avs_config\.get\(\s*"([^"]+)"\s*,\s*([^)]+)\)',
                                       open(path).read()):
            found.setdefault(key, []).append((os.path.basename(path), default.strip()))
    return found


def test_every_config_key_is_actually_read_by_a_script():
    """The failure this catches is a reproducibility layer that only looks
    like one. `configs/*.yaml` and `AVS_*` env vars did nothing at all for a
    while -- `config.py` existed, the shell scripts exported `AVS_CONFIG`, the
    README advertised `AVS_N_SEEDS=10 bash scripts/run_ablation.sh`, and no
    script in `src/` ever called `config.get`. Every run used the literals and
    every sweep would have produced identical results under different labels.
    """
    import yaml

    read = _keys_read_by_scripts()
    assert read, "no script reads any config key -- the config layer is decorative"

    unread = {}
    for path in _config_files():
        with open(path) as f:
            for key in (yaml.safe_load(f) or {}):
                if key not in read:
                    unread.setdefault(os.path.basename(path), []).append(key)
    assert not unread, f"config keys no script reads: {unread}"


# Which script each config file configures. A key is shared across files
# (`learning_rate` appears in both sft.yaml and dpo.yaml with different
# values), so the comparison has to be per-file, not global.
CONFIG_OWNER = {
    "sft.yaml": "train_lora_sft.py",
    "dpo.yaml": "train_lora_dpo_zeroshot.py",
    "agent_ablation.yaml": "feedback_protocol_ablation.py",
}


@pytest.mark.parametrize("filename", sorted(CONFIG_OWNER))
def test_script_defaults_match_the_yaml_so_defaults_reproduce_the_recorded_runs(filename):
    """`configs/sft.yaml` says its defaults 'reproduce the four recorded runs'.
    That is only true if the literals in the scripts agree with the YAML --
    otherwise running with the config and running without it silently train
    two different models, and only one of them matches what is written down."""
    import yaml

    owner = CONFIG_OWNER[filename]
    read = _keys_read_by_scripts()
    path = next(p for p in _config_files() if os.path.basename(p) == filename)
    with open(path) as f:
        settings = yaml.safe_load(f) or {}
    assert settings

    for key, value in settings.items():
        defaults = [d for script, d in read.get(key, []) if script == owner]
        assert defaults, f"{owner} does not read {key!r}, which {filename} sets"
        for default in defaults:
            assert _same_value(default, value), (
                f"{filename} sets {key}={value!r} but {owner} defaults to {default!r}"
            )


def _same_value(a, b):
    """`1e-4` in source, `1.0e-4` in YAML and `"0.0001"` from the fallback
    parser are all the same setting."""
    def parse(v):
        if isinstance(v, bool):
            return v
        text = str(v).strip().strip('"\'')
        if text.lower() in ("true", "false"):
            return text.lower() == "true"
        try:
            return float(text)
        except ValueError:
            return text

    x, y = parse(a), parse(b)
    if isinstance(x, float) and isinstance(y, float):
        return abs(x - y) <= 1e-12 * max(1.0, abs(x), abs(y))
    return x == y
