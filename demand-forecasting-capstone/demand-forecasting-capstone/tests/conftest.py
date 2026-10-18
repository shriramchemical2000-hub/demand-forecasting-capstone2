import pytest

from demand_agents.config import load_config


@pytest.fixture
def small_cfg(tmp_path):
    cfg = load_config()
    cfg["data"]["synthetic"].update(n_stores=2, n_items=5, n_days=500)
    cfg["split"].update(val_days=70, test_days=42)
    cfg["uncertainty"]["check_days"] = 28
    cfg["forecast"]["baselines"] = ["seasonal_naive"]
    cfg["forecast"]["selection"] = {"blend_with_ets": False}   # keep tests fast
    cfg["forecast"]["model"]["params"]["max_iter"] = 60
    cfg["output_dir"] = str(tmp_path / "out")
    return cfg
