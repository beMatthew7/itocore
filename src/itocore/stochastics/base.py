from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

@dataclass(frozen = True, slots = True)
class SimulationSpec:
    """Numerical grid and sample settings for path generation."""

    spot: float
    maturity: float
    steps: int
    paths: int = 1
    seed: int | None = None

    @property
    def dt(self) -> float:
        return self.maturity / self.steps

    def __post_init__(self) -> None:
        if not np.isfinite(self.spot):
            raise ValueError("spot must be finite")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.paths <= 0:
            raise ValueError("paths must be positive")


class StochasticModel(ABC):
    """Abstract contract implemented by every stochastic process.

        Subclasses should keep this public API stable even when the implementation
        is accelerated in C++, Cython, Numba, or vectorized NumPy.
    """

    name: str = "stochastic_model"

    def __init__(self, **parameters: float) -> None:
        self._parameters = MappingProxyType(dict(parameters))
        self.validate_parameters()


    @property
    def parameters(self) -> Mapping[str, float]:
        """Model parameters as a read-only mapping."""
        return self._parameters

    @abstractmethod
    def validate_parameters(self) -> None:
        """Raise ``ValueError`` when model parameters are outside the domain."""

    @abstractmethod
    def drift(self, state: FloatArray, time: float) -> FloatArray:
        """Instantaneous drift term under the chosen measure."""

    @abstractmethod
    def diffusion(self, state: FloatArray, time: float) -> FloatArray:
        """Instantaneous diffusion term under the chosen measure."""

    @abstractmethod
    def simulate(self, spec: SimulationSpec) -> FloatArray:
        """Generate paths with shape ``(paths, steps + 1)``."""

    @abstractmethod
    def price_vanilla(
            self,
            *,
            spot: float,
            strike: float,
            maturity: float,
            rate: float,
            option_type: str = "call",
            **kwargs: Any,) -> float:
        """Price a European vanilla option under this model."""

    def calibrate(self, market_data: Any, **kwargs: Any) -> "StochasticModel":
        """Fit model parameters to market data and return the calibrated model.

        Concrete models should override this when calibration is supported.
        """

        raise NotImplementedError(f"{self.name} does not implement calibration")

    def rng(self, seed: int | None = None) -> np.random.Generator:
        """Create a reproducible random generator for Python fallback engines."""

        return np.random.default_rng(seed)
