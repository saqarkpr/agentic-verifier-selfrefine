"""
Config resolution for the training and evaluation scripts.

Why this exists rather than hard-coded constants: the hyperparameters were
literals inside each script, which meant an experiment could only be varied
by editing source. That is fine for one run and useless for a sweep, and it
makes a run's settings unrecoverable after the fact unless you happen to
still have the exact file.

Resolution order, highest priority first:
  1. environment variable  (AVS_LEARNING_RATE=2e-4 python train_lora_sft.py)
  2. the YAML file named by AVS_CONFIG
  3. the default passed at the call site (the original literal)

Env-var override is what makes a Slurm array job able to sweep a parameter
without touching code or writing a config per point. YAML is what makes a
run's settings reproducible and diffable. The defaults mean every script
still runs with no configuration at all, exactly as before.

`resolved()` returns everything that was actually used, so scripts can dump
it alongside their results -- a run whose settings you cannot recover is a
run you cannot reproduce.
"""
import json
import os

_PREFIX = "AVS_"
_cache = None
_resolved = {}


def _load_yaml():
    global _cache
    if _cache is not None:
        return _cache
    path = os.environ.get(f"{_PREFIX}CONFIG")
    if not path or not os.path.exists(path):
        _cache = {}
        return _cache
    try:
        import yaml
        with open(path) as f:
            _cache = yaml.safe_load(f) or {}
    except ImportError:
        # yaml is not a hard dependency: fall back to a minimal parser that
        # handles the flat "key: value" files in configs/.
        _cache = {}
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                _cache[k.strip()] = v.strip().strip('"\'')
    return _cache


def get(key, default, cast=None):
    """Resolve one setting. `cast` defaults to the type of `default`."""
    cast = cast or (type(default) if default is not None else str)
    raw = os.environ.get(_PREFIX + key.upper())
    source = "env"
    if raw is None:
        cfg = _load_yaml()
        if key in cfg:
            raw, source = cfg[key], "yaml"
    if raw is None:
        _resolved[key] = (default, "default")
        return default
    if cast is bool:
        value = str(raw).strip().lower() in ("1", "true", "yes", "on")
    else:
        value = cast(raw)
    _resolved[key] = (value, source)
    return value


def resolved():
    return {k: {"value": v, "source": s} for k, (v, s) in _resolved.items()}


def dump(path):
    """Write the settings actually used next to the run's outputs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(resolved(), f, indent=2, default=str)
    return path
