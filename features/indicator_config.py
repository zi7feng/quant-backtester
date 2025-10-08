class IndicatorConfig:
    """
    IndicatorConfig defines all adjustable parameters for feature generation.
    Each trading strategy can create its own instance of this class to
    customize indicator settings for trend, momentum, volatility, and mean reversion groups.

    Usage Example:
        # 1. Default configuration (auto-loads standard values)
        cfg = IndicatorConfig()

        # 2. Custom configuration for a specific strategy
        cfg = IndicatorConfig(
            trend={
                "SMA": [50, 200],
                "EMA": [12, 26],
                "MACD": {"short": 12, "long": 26, "signal": 9}
            },
            momentum={
                "RSI": [14],
                "Momentum": [10]
            },
            volatility={
                "ATR": [14],
                "Bollinger": {"window": 20, "num_std": 2.0}
            },
            mean_reversion={
                "ZScore": [30]
            },
            version="trend_v1"
        )

        # 3. Convert to dictionary before passing to feature_loader
        config_dict = cfg.to_dict()
    """

    def __init__(
        self,
        trend=None,
        momentum=None,
        volatility=None,
        mean_reversion=None,
        version="v1.0"
    ):
        # --- Trend-based indicators ---
        # Default includes moving averages and MACD family.
        self.trend = trend or {
            "SMA": [20, 50],
            "EMA": [12, 26],
            "MACD": {"short": 12, "long": 26, "signal": 9}
        }

        # --- Momentum-based indicators ---
        # Default includes RSI, ROC, and simple momentum.
        self.momentum = momentum or {
            "RSI": [14],
            "ROC": [12],
            "Momentum": [10],
        }

        # --- Volatility-based indicators ---
        # Default includes ATR and Bollinger Bands.
        self.volatility = volatility or {
            "ATR": [14],
            "Bollinger": {"window": 20, "num_std": 2.0}
        }

        # --- Mean Reversion indicators ---
        # Default includes Z-Score over a rolling window.
        self.mean_reversion = mean_reversion or {
            "ZScore": [30]
        }

        # Version tag for tracking configuration changes.
        self.version = version

    def to_dict(self):
        """
        Convert the configuration to a unified dictionary format.
        This dictionary can be directly passed to the feature_loader module
        to compute features based on the current indicator configuration.
        """
        return {
            "trend": self.trend,
            "momentum": self.momentum,
            "volatility": self.volatility,
            "mean_reversion": self.mean_reversion,
            "version": self.version,
        }
