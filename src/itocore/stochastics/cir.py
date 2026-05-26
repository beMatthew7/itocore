from __future__ import annotations
import numpy as np
from typing import Any
from .base import StochasticModel, SimulationSpec, FloatArray


class CoxIngersollRoss(StochasticModel):
    """Cox-Ingersoll-Ross (CIR) model implementation."""

    name: str = "cir"

    def validate_parameters(self) -> None:
        """Verify that all CIR parameters are positive."""
        speed = self.parameters.get("speed", -1)
        mean = self.parameters.get("mean", -1)
        vol = self.parameters.get("vol", -1)

        if speed <= 0:
            raise ValueError("CIR requires a positive 'speed' (mean reversion speed) parameter.")
        if mean <= 0:
            raise ValueError("CIR requires a positive 'mean' (long-term mean) parameter.")
        if vol <= 0:
            raise ValueError("CIR requires a positive 'vol' (volatility) parameter.")

    def simulate(self, spec: SimulationSpec) -> FloatArray:
        """Simulate interest rate paths using either the fast C++ core or Python fallback."""
        try:
            # Try to import the native C++ compiled engine
            from itocore import _core

            # Pre-allocate a contiguous NumPy array for C++ zero-copy mapping
            paths = np.empty((spec.paths, spec.steps + 1), dtype=np.float64)

            # Extract parameters
            speed_val = self.parameters.get("speed", 0.1)
            mean_val = self.parameters.get("mean", 0.05)
            vol_val = self.parameters.get("vol", 0.02)
            seed_val = spec.seed if spec.seed is not None else 42

            # Execute calculation natively in C++
            _core.simulate_cir(paths, spec.spot, speed_val, mean_val, vol_val, spec.dt, seed_val)
            return paths

        except ImportError:
            # Fallback to standard Python/NumPy implementation if binary is not compiled
            rng = self.rng(spec.seed)
            paths = np.zeros((spec.paths, spec.steps + 1))
            paths[:, 0] = spec.spot

            speed_val = self.parameters.get("speed", 0.1)
            mean_val = self.parameters.get("mean", 0.05)
            vol_val = self.parameters.get("vol", 0.02)
            dt = spec.dt

            for i in range(1, spec.steps + 1):
                z = rng.standard_normal(spec.paths)
                prev = paths[:, i-1]

                # Drift term: speed * (mean - X_t) * dt
                drift = speed_val * (mean_val - prev) * dt

                # Diffusion term: vol * sqrt(max(X_t, 0.0)) * dW_t
                # Guard against negative values due to discretization
                diffusion = vol_val * np.sqrt(np.maximum(prev, 0.0)) * np.sqrt(dt) * z

                paths[:, i] = np.maximum(prev + drift + diffusion, 0.0)

            return paths

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
        """Price a European vanilla option under the CIR model.

        For interest rate models, we treat 'spot' as the current interest rate,
        'strike' as the strike rate, and price a bond option.
        """
        # For simplicity, we'll use the same approach as GBM but adapted for rates
        # In practice, CIR has closed-form solutions for bond options
        # This is a simplified implementation for consistency with the interface

        # Using the analytical formula for zero-coupon bond options in CIR model
        # Based on Cox, Ingersoll, and Ross (1985)

        speed = self.parameters.get("speed", 0.1)
        mean = self.parameters.get("mean", 0.05)
        vol = self.parameters.get("vol", 0.02)

        # Calculate bond prices
        B = self._zero_coupon_bond_price(spot, maturity, speed, mean, vol)
        P_strike = self._zero_coupon_bond_price(strike, maturity, speed, mean, vol)

        # For a call option on a zero-coupon bond
        if option_type.lower() == "call":
            # Simplified: treat as option on the bond price
            # In practice, would need the full CIR bond option formula
            return max(B - P_strike, 0.0) * np.exp(-rate * maturity)
        else:
            # Put option
            return max(P_strike - B, 0.0) * np.exp(-rate * maturity)

    def _zero_coupon_bond_price(self, r: float, T: float, speed: float, mean: float, vol: float) -> float:
        """Calculate zero-coupon bond price in CIR model."""
        # CIR model bond price formula
        gamma = np.sqrt(speed**2 + 2 * vol**2)
        B = (2 * (np.exp(gamma * T) - 1)) / (
            (gamma + speed) * (np.exp(gamma * T) - 1) + 2 * gamma
        )
        A = (
            (2 * gamma * np.exp((speed + gamma) * T / 2)) /
            ((gamma + speed) * (np.exp(gamma * T) - 1) + 2 * gamma)
        ) ** (2 * speed * mean / vol**2)

        return A * np.exp(-B * r)

    def drift(self, state: FloatArray, time: float) -> FloatArray:
        # dr_t = speed * (mean - r_t) * dt
        speed = self.parameters.get("speed", 0.1)
        mean = self.parameters.get("mean", 0.05)
        return speed * (mean - state)

    def diffusion(self, state: FloatArray, time: float) -> FloatArray:
        # vol * sqrt(max(r_t, 0.0))
        vol = self.parameters.get("vol", 0.02)
        return vol * np.sqrt(np.maximum(state, 0.0))