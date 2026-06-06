from __future__ import annotations

from typing import Any

import numpy as np

from .base import FloatArray, SimulationSpec, StochasticModel


class OrnsteinUhlenbeckProcess(StochasticModel):
    """Ornstein-Uhlenbeck mean-reverting process for spreads and stat-arb signals."""

    name: str = "ornstein_uhlenbeck"

    def validate_parameters(self) -> None:
        """Verify that OU parameters are inside the model domain."""
        speed = self.parameters.get("speed", -1.0)
        mean = self.parameters.get("mean", np.nan)
        vol = self.parameters.get("vol", -1.0)

        if speed <= 0.0:
            raise ValueError("OU requires a positive 'speed' (mean reversion speed) parameter.")
        if not np.isfinite(mean):
            raise ValueError("OU requires a finite 'mean' (long-term mean) parameter.")
        if vol <= 0.0:
            raise ValueError("OU requires a positive 'vol' (volatility) parameter.")

    def simulate(self, spec: SimulationSpec) -> FloatArray:
        """Simulate OU paths using the exact Gaussian transition."""
        speed = self.parameters.get("speed", 0.1)
        mean = self.parameters.get("mean", 0.0)
        vol = self.parameters.get("vol", 1.0)
        seed = spec.seed if spec.seed is not None else 42

        try:
            from itocore import _core

            paths = np.empty((spec.paths, spec.steps + 1), dtype=np.float64)
            _core.simulate_ou(paths, spec.spot, speed, mean, vol, spec.dt, seed)
            return paths
        except ImportError:
            rng = self.rng(spec.seed)
            paths = np.empty((spec.paths, spec.steps + 1), dtype=np.float64)
            paths[:, 0] = spec.spot

            exp_neg_speed_dt = np.exp(-speed * spec.dt)
            conditional_std = vol * np.sqrt((1.0 - np.exp(-2.0 * speed * spec.dt)) / (2.0 * speed))

            for i in range(1, spec.steps + 1):
                z = rng.standard_normal(spec.paths)
                prev = paths[:, i - 1]
                paths[:, i] = mean + (prev - mean) * exp_neg_speed_dt + conditional_std * z

            return paths

    def drift(self, state: FloatArray, time: float) -> FloatArray:
        speed = self.parameters.get("speed", 0.1)
        mean = self.parameters.get("mean", 0.0)
        return speed * (mean - state)

    def diffusion(self, state: FloatArray, time: float) -> FloatArray:
        vol = self.parameters.get("vol", 1.0)
        return np.full_like(state, vol, dtype=np.float64)

    def half_life(self) -> float:
        """Return the expected time for a shock to decay by half."""
        speed = self.parameters.get("speed", 0.1)
        return float(np.log(2.0) / speed)

    def stationary_variance(self) -> float:
        """Return the long-run variance of the stationary OU distribution."""
        speed = self.parameters.get("speed", 0.1)
        vol = self.parameters.get("vol", 1.0)
        return float(vol**2 / (2.0 * speed))

    def price_vanilla(
        self,
        *,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        option_type: str = "call",
        **kwargs: Any,
    ) -> float:
        """Price a European option on the terminal OU value under the exact Gaussian law."""
        if maturity <= 0.0:
            raise ValueError("maturity must be positive")

        speed = self.parameters.get("speed", 0.1)
        mean = self.parameters.get("mean", 0.0)
        vol = self.parameters.get("vol", 1.0)

        exp_neg_speed_t = np.exp(-speed * maturity)
        forward_mean = mean + (spot - mean) * exp_neg_speed_t
        variance = vol**2 * (1.0 - np.exp(-2.0 * speed * maturity)) / (2.0 * speed)
        std = np.sqrt(variance)
        discount = np.exp(-rate * maturity)

        try:
            from scipy.stats import norm
        except ImportError as exc:
            raise ImportError("OU option pricing requires scipy.") from exc

        d = (forward_mean - strike) / std
        if option_type.lower() == "call":
            return float(discount * ((forward_mean - strike) * norm.cdf(d) + std * norm.pdf(d)))
        if option_type.lower() == "put":
            return float(discount * ((strike - forward_mean) * norm.cdf(-d) + std * norm.pdf(d)))

        raise ValueError("option_type must be either 'call' or 'put'")
