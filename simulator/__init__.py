# simulator/__init__.py
from simulator.account import Account
from simulator.broker import Broker
from simulator.config.sim_config import load_sim_config

def init_simulator_environment(config_path=None):
    """
    Initialize Account and Broker from configuration file.
    Returns: (account, broker, cfg)
    """
    cfg = load_sim_config(config_path)

    # --- init account ---
    acc_cfg = cfg["account"]
    account = Account(
        initial_cash=acc_cfg["initial_cash"]
    )
    account.allow_short = acc_cfg.get("allow_short", False)
    account.currency = acc_cfg.get("currency", "USD")

    # --- init broker ---
    br_cfg = cfg["broker"]
    broker = Broker(
        account=account,
        fee_type=br_cfg.get("fee_type", "rate"),
        fee_rate=br_cfg.get("fee_rate", 0.001),
        fixed_fee=br_cfg.get("fixed_fee", 1.0),
        min_fee=br_cfg.get("min_fee", 1.0),
        slippage=br_cfg.get("slippage", 0.0005),
    )

    print(f"Simulator initialized:")
    print(f"   • Cash: {account.cash:,.2f} {account.currency}")
    print(f"   • Fee model: {br_cfg['fee_type']} ({br_cfg})")

    return account, broker, cfg
