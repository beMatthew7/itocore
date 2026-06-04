import numpy as np
import pytest

from itocore.pricing import price_vanilla_from_paths, price_vanilla_monte_carlo, vanilla_payoff
from itocore.stochastics import GeometricBrownianMotion


def test_vanilla_payoff_call_and_put():
    terminal = np.array([80.0, 100.0, 120.0], dtype=np.float64)

    np.testing.assert_array_equal(vanilla_payoff(terminal, 100.0, "call"), np.array([0.0, 0.0, 20.0]))
    np.testing.assert_array_equal(vanilla_payoff(terminal, 100.0, "put"), np.array([20.0, 0.0, 0.0]))


def test_price_vanilla_from_paths_matches_manual_discounted_payoff():
    paths = np.array(
        [
            [100.0, 110.0],
            [100.0, 90.0],
            [100.0, 120.0],
        ],
        dtype=np.float64,
    )

    result = price_vanilla_from_paths(
        paths,
        strike=100.0,
        rate=0.05,
        maturity=1.0,
        option_type="call",
    )

    expected = np.exp(-0.05) * np.mean(np.array([10.0, 0.0, 20.0]))
    np.testing.assert_allclose(result.price, expected)
    assert result.standard_error > 0.0
    assert result.paths == 3


def test_monte_carlo_vanilla_call_matches_black_scholes_within_sampling_error():
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    vol = 0.2

    result = price_vanilla_monte_carlo(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        vol=vol,
        steps=1,
        paths=50_000,
        option_type="call",
        seed=123,
    )
    analytical = GeometricBrownianMotion(rate=rate, vol=vol).price_vanilla(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        option_type="call",
    )

    assert abs(result.price - analytical) < 4.0 * result.standard_error


def test_monte_carlo_vanilla_put_matches_black_scholes_within_sampling_error():
    spot = 100.0
    strike = 95.0
    maturity = 1.0
    rate = 0.03
    vol = 0.25

    result = price_vanilla_monte_carlo(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        vol=vol,
        steps=1,
        paths=50_000,
        option_type="put",
        seed=321,
    )
    analytical = GeometricBrownianMotion(rate=rate, vol=vol).price_vanilla(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        option_type="put",
    )

    assert abs(result.price - analytical) < 4.0 * result.standard_error


def test_pricing_validates_inputs():
    paths = np.array([[100.0, 110.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="strike must be positive"):
        vanilla_payoff(paths[:, -1], 0.0)

    with pytest.raises(ValueError, match="option_type must be either"):
        vanilla_payoff(paths[:, -1], 100.0, "digital")

    with pytest.raises(ValueError, match="paths must be a 2D array"):
        price_vanilla_from_paths(paths[:, -1], strike=100.0, rate=0.05, maturity=1.0)

    with pytest.raises(ValueError, match="maturity must be positive"):
        price_vanilla_from_paths(paths, strike=100.0, rate=0.05, maturity=0.0)
