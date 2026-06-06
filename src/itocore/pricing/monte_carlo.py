from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from itocore.pricing.payoffs import vanilla_payoff
from itocore.stochastics import GeometricBrownianMotion, SimulationSpec
from itocore.stochastics.base import FloatArray


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Monte Carlo estimate with sampling uncertainty."""

    price: float
    standard_error: float
    paths: int


def price_vanilla_from_paths(
    paths: FloatArray,
    *,
    strike: float,
    rate: float,
    maturity: float,
    option_type: str = "call",
) -> MonteCarloResult:
    """Price a European vanilla option from precomputed simulated paths."""
    if paths.ndim != 2:
        raise ValueError("paths must be a 2D array with shape (paths, steps + 1)")
    if paths.shape[0] <= 0:
        raise ValueError("paths must contain at least one path")
    if paths.shape[1] < 2:
        raise ValueError("paths must contain at least one time step")
    if maturity <= 0.0:
        raise ValueError("maturity must be positive")

    try:
        from itocore import _core

        price, standard_error = _core.price_vanilla_terminal(
            np.ascontiguousarray(paths, dtype=np.float64),
            strike,
            rate,
            maturity,
            option_type.lower(),
        )
        return MonteCarloResult(float(price), float(standard_error), int(paths.shape[0]))
    except ImportError:
        payoffs = vanilla_payoff(paths[:, -1], strike, option_type)
        discount = np.exp(-rate * maturity)
        price = discount * np.mean(payoffs)
        standard_error = 0.0

        if paths.shape[0] > 1:
            standard_error = discount * np.std(payoffs, ddof=1) / np.sqrt(paths.shape[0])

        return MonteCarloResult(float(price), float(standard_error), int(paths.shape[0]))


def price_vanilla_monte_carlo(
    *,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    steps: int,
    paths: int,
    option_type: str = "call",
    seed: int | None = None,
) -> MonteCarloResult:
    """Simulate GBM paths and price a European vanilla option by Monte Carlo."""
    model = GeometricBrownianMotion(rate=rate, vol=vol)
    spec = SimulationSpec(
        spot=spot,
        maturity=maturity,
        steps=steps,
        paths=paths,
        seed=seed,
    )
    simulated_paths = model.simulate(spec)
    return price_vanilla_from_paths(
        simulated_paths,
        strike=strike,
        rate=rate,
        maturity=maturity,
        option_type=option_type,
    )
