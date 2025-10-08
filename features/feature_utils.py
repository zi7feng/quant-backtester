"""
Feature Utilities
-----------------
Helper functions for feature generation and indicator configuration.
Includes:
- get_max_window(): automatically detect max lookback window needed for indicators
"""

def get_max_window(config: dict) -> int:
    """
    Inspect an indicator configuration (dict or IndicatorConfig.to_dict())
    and return the largest lookback window required across all indicators.

    Parameters
    ----------
    config : dict
        Dictionary containing indicator parameters by category.

    Returns
    -------
    int
        Maximum window/period/span value detected across all indicators.
    """
    max_w = 0

    for section_name, section in config.items():
        if not isinstance(section, dict):
            continue

        for name, params in section.items():
            # Case 1: list of window sizes (e.g. {"SMA": [10, 50, 200]})
            if isinstance(params, list):
                max_w = max(max_w, max(params))

            # Case 2: dict of parameter pairs (e.g. {"MACD": {"short": 12, "long": 26}})
            elif isinstance(params, dict):
                for key, val in params.items():
                    if isinstance(val, (int, float)) and key.lower() in ["window", "span", "short", "long"]:
                        max_w = max(max_w, int(val))

    # Default fallback
    return max_w or 50
