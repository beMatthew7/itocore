# itocore

High-performance quantitative finance library in Python, accelerated with C++ through pybind11.

itocore focuses on stochastic simulation and Monte Carlo pricing. The public API stays Pythonic, while compute-heavy loops run in native C++ when the extension module is available.

## Features

- Geometric Brownian Motion (GBM)
- Cox-Ingersoll-Ross (CIR) short-rate model
- Vasicek short-rate model
- Ornstein-Uhlenbeck (OU) process
- European vanilla option pricing by Monte Carlo
- Arithmetic Asian option pricing by Monte Carlo
- Black-Scholes analytical pricing for GBM validation
- NumPy-first arrays with zero-copy C++ kernels where possible

## Installation

From a local checkout:

```bash
python -m pip install .
```

For editable development:

```bash
python -m pip install -e ".[test]"
```

Core runtime dependencies are NumPy and SciPy. Building the native extension requires a C++ compiler, CMake, scikit-build-core, and pybind11.

## Quick Start

```python
from itocore.stochastics import GeometricBrownianMotion, SimulationSpec
from itocore.pricing import price_vanilla_monte_carlo

gbm = GeometricBrownianMotion(rate=0.05, vol=0.20)
spec = SimulationSpec(spot=100.0, maturity=1.0, steps=252, paths=10_000, seed=42)

paths = gbm.simulate(spec)
print(paths.shape)

result = price_vanilla_monte_carlo(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    vol=0.20,
    steps=252,
    paths=50_000,
    option_type="call",
    seed=42,
)

print(result.price, result.standard_error)
```

## Stochastic Models

### Geometric Brownian Motion

GBM is the standard equity model behind Black-Scholes:

```python
from itocore.stochastics import GeometricBrownianMotion, SimulationSpec

model = GeometricBrownianMotion(rate=0.05, vol=0.20)
spec = SimulationSpec(spot=100.0, maturity=1.0, steps=252, paths=5_000, seed=1)

paths = model.simulate(spec)
black_scholes_call = model.price_vanilla(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    option_type="call",
)
```

### Cox-Ingersoll-Ross

CIR is a mean-reverting short-rate model with non-negative paths under the implemented truncation scheme:

```python
from itocore.stochastics import CoxIngersollRoss, SimulationSpec

model = CoxIngersollRoss(speed=0.5, mean=0.04, vol=0.15)
spec = SimulationSpec(spot=0.01, maturity=2.0, steps=200, paths=1_000, seed=123)

rates = model.simulate(spec)
```

### Vasicek

Vasicek is a Gaussian mean-reverting short-rate model. It supports negative rates:

```python
from itocore.stochastics import VasicekModel, SimulationSpec

model = VasicekModel(speed=0.7, mean=0.02, vol=0.03)
spec = SimulationSpec(spot=-0.005, maturity=1.5, steps=96, paths=1_000, seed=99)

rates = model.simulate(spec)
bond_price = model.zero_coupon_bond_price(short_rate=0.025, maturity=5.0)
```

### Ornstein-Uhlenbeck

OU is useful for spreads, pairs trading, and statistical arbitrage signals:

```python
from itocore.stochastics import OrnsteinUhlenbeckProcess, SimulationSpec

model = OrnsteinUhlenbeckProcess(speed=1.5, mean=0.0, vol=0.8)
spec = SimulationSpec(spot=2.0, maturity=2.0, steps=128, paths=1_000, seed=202)

spreads = model.simulate(spec)
print(model.half_life())
print(model.stationary_variance())
```

## Pricing

### European Vanilla Options

```python
from itocore.pricing import price_vanilla_monte_carlo

result = price_vanilla_monte_carlo(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    vol=0.20,
    steps=252,
    paths=50_000,
    option_type="call",
    seed=42,
)

print(result.price)
print(result.standard_error)
```

You can also price from precomputed paths:

```python
from itocore.pricing import price_vanilla_from_paths

result = price_vanilla_from_paths(
    paths,
    strike=100.0,
    rate=0.05,
    maturity=1.0,
    option_type="put",
)
```

### Arithmetic Asian Options

Asian options depend on the average path value, so Monte Carlo pricing is a natural fit:

```python
from itocore.pricing import price_asian_monte_carlo

result = price_asian_monte_carlo(
    spot=100.0,
    strike=100.0,
    maturity=1.0,
    rate=0.05,
    vol=0.20,
    steps=252,
    paths=50_000,
    option_type="call",
    include_initial=True,
    seed=7,
)

print(result.price, result.standard_error)
```

From precomputed paths:

```python
from itocore.pricing import price_asian_from_paths

result = price_asian_from_paths(
    paths,
    strike=100.0,
    rate=0.05,
    maturity=1.0,
    option_type="call",
    include_initial=False,
)
```

## Development

Run tests:

```bash
python -m pytest -q
```

Build a wheel:

```bash
python -m build
```

Install the built wheel locally:

```bash
python -m pip install dist/itocore-0.1.0-*.whl
```

## Release Status

Version `0.1.0` is an alpha release intended for experimentation, learning, and portfolio demonstration. APIs may still evolve.

