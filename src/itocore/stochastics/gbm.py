from __future__ import annotations
import numpy as np
from scipy.stats import norm
from .base import StochasticModel, SimulationSpec, FloatArray

class GeometricBrownianMotion(StochasticModel):
    """Geometric Brownian Motion (GBM) implementation."""

    name: str = "gbm"

    def validate_parameters(self) -> None:
        """Verify that volatility is positive."""
        if self.parameters.get("vol", -1) <= 0:
            raise ValueError("GBM requires a positive 'vol' (volatility) parameter.")

    def drift(self, state: FloatArray, time: float) -> FloatArray:
        # dS = r * S * dt
        r = self.parameters.get("rate", 0.0)
        return r * state

    def diffusion(self, state: FloatArray, time: float) -> FloatArray:
        # dS = sigma * S * dW
        sigma = self.parameters.get("vol", 0.2)
        return sigma * state

    def simulate(self, spec: SimulationSpec) -> FloatArray:
        """Simulate asset paths using either the fast C++ core or Python fallback."""
        try:
            # Try to import the native C++ compiled engine
            from itocore import _core

            # Pre-allocate a contiguous NumPy array for C++ zero-copy mapping
            paths = np.empty((spec.paths, spec.steps + 1), dtype=np.float64)

            # Extract parameters
            mu = self.parameters.get("rate", 0.0)
            sigma = self.parameters.get("vol", 0.2)
            seed_val = spec.seed if spec.seed is not None else 42

            # Execute calculation natively in C++
            _core.simulate_gbm(paths, spec.spot, mu, sigma, spec.dt, seed_val)
            return paths

        except ImportError:
            # Fallback to standard Python/NumPy implementation if binary is not compiled
            rng = self.rng(spec.seed)
            paths = np.zeros((spec.paths, spec.steps + 1))

            r = self.parameters.get("rate", 0.0)
            sigma = self.parameters.get("vol", 0.2)
            dt = spec.dt

            for i in range(1, spec.steps + 1):
                z = rng.standard_normal(spec.paths)
                paths[:, i] = paths[:, i-1] * np.exp(
                    (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
                )

            return paths

    def price_vanilla(
            self,
            *,
            spot: float,
            strike: float,
            maturity: float,
            rate: float,
            option_type: str = "call",
            **kwargs
    ) -> float:
        """Exact Black-Scholes analytical formula."""
        sigma = self.parameters["vol"]

        d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * maturity) / (sigma * np.sqrt(maturity))
        d2 = d1 - sigma * np.sqrt(maturity)

        if option_type.lower() == "call":
            return spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)
        else:
            return strike * np.exp(-rate * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1)