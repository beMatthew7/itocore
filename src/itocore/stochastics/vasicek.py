from __future__ import annotations

from typing import Any

import numpy as np

from .base import FloatArray, SimulationSpec, StochasticModel


class VasicekModel(StochasticModel):
    """Vasicek mean-reverting short-rate model."""

    name: str = "vasicek"

    def validate_parameters(self) -> None:
        """Verify that Vasicek parameters are inside the model domain."""
        speed = self.parameters.get("speed", -1.0)
        mean = self.parameters.get("mean", np.nan)
        vol = self.parameters.get("vol", -1.0)

        if speed <= 0.0:
            raise ValueError("Vasicek requires a positive 'speed' (mean reversion speed) parameter.")
        if not np.isfinite(mean):
            raise ValueError("Vasicek requires a finite 'mean' (long-term mean) parameter.")
        if vol <= 0.0:
            raise ValueError("Vasicek requires a positive 'vol' (volatility) parameter.")

    def simulate(self, spec: SimulationSpec) -> FloatArray:
        """Simulate short-rate paths using the exact Gaussian transition."""
        speed = self.parameters.get("speed", 0.1)
        mean = self.parameters.get("mean", 0.05)
        vol = self.parameters.get("vol", 0.02)
        seed = spec.seed if spec.seed is not None else 42

        try:
            from itocore import _core

            paths = np.empty((spec.paths, spec.steps + 1), dtype=np.float64)
            _core.simulate_vasicek(paths, spec.spot, speed, mean, vol, spec.dt, seed)
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
        mean = self.parameters.get("mean", 0.05)
        return speed * (mean - state)

    def diffusion(self, state: FloatArray, time: float) -> FloatArray:
        vol = self.parameters.get("vol", 0.02)
        return np.full_like(state, vol, dtype=np.float64)

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
        """Price a simple option on a Vasicek zero-coupon bond by discounted intrinsic value."""
        bond_price = self.zero_coupon_bond_price(spot, maturity)
        strike_bond_price = self.zero_coupon_bond_price(strike, maturity)
        discount = np.exp(-rate * maturity)

        if option_type.lower() == "call":
            return float(discount * max(bond_price - strike_bond_price, 0.0))
        if option_type.lower() == "put":
            return float(discount * max(strike_bond_price - bond_price, 0.0))

        raise ValueError("option_type must be either 'call' or 'put'")

    def zero_coupon_bond_price(self, short_rate: float, maturity: float) -> float:
        """Analytical zero-coupon bond price under the Vasicek model."""
        if maturity <= 0.0:
            raise ValueError("maturity must be positive")

        speed = self.parameters.get("speed", 0.1)
        mean = self.parameters.get("mean", 0.05)
        vol = self.parameters.get("vol", 0.02)

        b_factor = (1.0 - np.exp(-speed * maturity)) / speed
        a_factor = (
            (mean - (vol**2) / (2.0 * speed**2)) * (b_factor - maturity)
            - (vol**2) * (b_factor**2) / (4.0 * speed)
        )
        return float(np.exp(a_factor - b_factor * short_rate))
