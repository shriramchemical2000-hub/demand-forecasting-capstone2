"""Configuration loading with inheritance and dotted-key overrides."""
import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def _deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path=None, overrides=None) -> dict:
    """Load YAML config. A file may contain ``extends: default.yaml`` to inherit and override.

    ``overrides`` is a dict like {"data.source": "m5"}.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        cfg = _deep_merge(base, cfg)
    for dotted, value in (overrides or {}).items():
        node = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return cfg


def periods_per_year(cfg: dict) -> int:
    return 52 if cfg["data"].get("frequency", "D") == "W" else 365


# Dataset names used by the notebooks and the CLI (``--config m5``): (config file, overrides)
NAMED = {
    "synthetic": ("base.yaml", {"data.processed_path": "data/processed/synthetic_model_table.csv.gz", "output_dir": "outputs/synthetic"}),
    "synthetic_calm": ("base.yaml", {"data.synthetic.shocks": False, "data.processed_path": "data/processed/synthetic_calm_model_table.csv.gz",
                                     "output_dir": "outputs/synthetic_calm"}),
    "m5": ("m5.yaml", {}),
    "m5_christmas": ("m5_christmas.yaml", {}),
    "walmart": ("walmart.yaml", {}),
    "walmart_synthetic": ("walmart.yaml", {"data.source": "walmart_synthetic", "output_dir": "outputs/walmart_synthetic",
                                           "data.processed_path": "data/processed/walmart_synthetic_model_table.csv.gz"}),
}


def load_named(name: str, overrides=None) -> dict:
    """Load a configuration by dataset name, e.g. ``load_named("m5")``."""
    if name not in NAMED:
        raise KeyError(f"unknown dataset '{name}'. Choose one of: {', '.join(NAMED)}")
    file, base = NAMED[name]
    merged = dict(base)
    merged.update(overrides or {})
    return load_config(PROJECT_ROOT / "configs" / file, merged)
