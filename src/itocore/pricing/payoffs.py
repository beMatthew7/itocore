from __future__ import annotations

import numpy as np

from itocore.stochastics.base import FloatArray


def vanilla_payoff(terminal_values: FloatArray, strike: float, option_type: str = "call") -> FloatArray:
    """Return European vanilla option payoffs for terminal underlying values."""
    if strike <= 0.0:
        raise ValueError("strike must be positive")

    option = option_type.lower()
    if option == "call":
        return np.maximum(terminal_values - strike, 0.0)
    if option == "put":
        return np.maximum(strike - terminal_values, 0.0)

    raise ValueError("option_type must be either 'call' or 'put'")


def asian_payoff(path_averages: FloatArray, strike: float, option_type: str = "call") -> FloatArray:
    """Return arithmetic Asian option payoffs from per-path average underlying values."""
    return vanilla_payoff(path_averages, strike, option_type)
