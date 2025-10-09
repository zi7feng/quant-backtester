# simulator/config/sim_config.py
import yaml
import copy
from pathlib import Path

DEFAULT_CONFIG = {
    "account": {
        "initial_cash": 100000,
        "currency": "USD",
        "allow_short": False,
    },
    "broker": {
        "fee_type": "rate",
        "fee_rate": 0.001,
        "fixed_fee": 1.0,
        "min_fee": 1.0,
        "slippage": 0.0005,
    },
    "env": {
        "feed_delay_minutes": 2,
        "latency_ms": 0,
        "volatility_factor": 0.1,
        "random_seed": 37,
    },
    "symbols": ["SPY.US"],           # default single symbol
    "strategy": {"name": "DemoStrategy"},  # default strategy name
}


def load_sim_config(path: str = None):
    """
    Load simulator configuration from YAML file.
    If file not found, fallback to default config.
    """
    config_path = Path(path or Path(__file__).parent / "sim_config.yaml")

    if not config_path.exists():
        print(f"Config file not found at {config_path}, using defaults.")
        return DEFAULT_CONFIG

    with open(config_path, "r") as f:
        user_cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # Merge user config into default
    for section, params in user_cfg.items():
        if section in cfg and isinstance(cfg[section], dict) and isinstance(params, dict):
            cfg[section].update(params)
        else:
            cfg[section] = params

    return cfg
