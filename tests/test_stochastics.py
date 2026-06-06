import numpy as np
import pytest

from itocore.stochastics import (
    CoxIngersollRoss,
    GeometricBrownianMotion,
    OrnsteinUhlenbeckProcess,
    SimulationSpec,
    VasicekModel,
)


def test_gbm_simulation_shape_and_determinism():
    """Test GBM simulation output shape and determinism with fixed seed."""
    # Instantiate GBM model
    gbm = GeometricBrownianMotion(vol=0.20, rate=0.05)

    # Define simulation specification
    spec = SimulationSpec(
        spot=100.0,
        maturity=1.0,
        steps=100,
        paths=50,
        seed=42
    )

    # First simulation run
    paths1 = gbm.simulate(spec)

    # Second simulation run with identical spec
    paths2 = gbm.simulate(spec)

    # Assert output shape is (paths, steps + 1)
    assert paths1.shape == (50, 101), f"Expected shape (50, 101), got {paths1.shape}"
    assert paths2.shape == (50, 101), f"Expected shape (50, 101), got {paths2.shape}"

    # Assert initial column values equal spot price
    np.testing.assert_array_equal(paths1[:, 0], 100.0)
    np.testing.assert_array_equal(paths2[:, 0], 100.0)

    # Assert perfect determinism between runs
    np.testing.assert_array_equal(paths1, paths2)


def test_cir_simulation_non_negative():
    """Test CIR simulation ensures non-negative interest rates."""
    # Instantiate CIR model with parameters that could produce negative values without guard
    cir = CoxIngersollRoss(speed=0.5, mean=0.04, vol=0.15)

    # Define simulation specification with low spot to stress-test non-negativity
    spec = SimulationSpec(
        spot=0.01,
        maturity=2.0,
        steps=200,
        paths=100,
        seed=123
    )

    # Run simulation
    paths = cir.simulate(spec)

    # Assert output shape is (paths, steps + 1)
    assert paths.shape == (100, 201), f"Expected shape (100, 201), got {paths.shape}"

    # Assert all values are non-negative (>= 0.0)
    assert np.all(paths >= 0.0), "Found negative values in CIR simulation paths"


def test_cir_simulation_shape_initial_value_and_determinism():
    """Test CIR output shape, initial value preservation, and fixed-seed determinism."""
    cir = CoxIngersollRoss(speed=1.2, mean=0.03, vol=0.35)

    spec = SimulationSpec(
        spot=0.02,
        maturity=1.0,
        steps=64,
        paths=32,
        seed=7,
    )

    paths1 = cir.simulate(spec)
    paths2 = cir.simulate(spec)

    assert paths1.shape == (32, 65)
    assert paths2.shape == (32, 65)
    np.testing.assert_array_equal(paths1[:, 0], 0.02)
    np.testing.assert_array_equal(paths1, paths2)


def test_cir_drift_and_diffusion_terms():
    """Test CIR instantaneous coefficients match the mathematical process definition."""
    cir = CoxIngersollRoss(speed=0.5, mean=0.04, vol=0.2)
    state = np.array([0.01, 0.04, 0.09], dtype=np.float64)

    np.testing.assert_allclose(cir.drift(state, 0.0), np.array([0.015, 0.0, -0.025]))
    np.testing.assert_allclose(cir.diffusion(state, 0.0), 0.2 * np.sqrt(state))


def test_vasicek_simulation_shape_initial_value_and_determinism():
    """Test Vasicek output shape, negative-rate support, and fixed-seed determinism."""
    vasicek = VasicekModel(speed=0.7, mean=0.02, vol=0.03)

    spec = SimulationSpec(
        spot=-0.005,
        maturity=1.5,
        steps=96,
        paths=40,
        seed=99,
    )

    paths1 = vasicek.simulate(spec)
    paths2 = vasicek.simulate(spec)

    assert paths1.shape == (40, 97)
    assert paths2.shape == (40, 97)
    np.testing.assert_array_equal(paths1[:, 0], -0.005)
    np.testing.assert_array_equal(paths1, paths2)
    assert np.any(paths1 < 0.0)


def test_vasicek_drift_and_diffusion_terms():
    """Test Vasicek instantaneous coefficients match the mathematical process definition."""
    vasicek = VasicekModel(speed=0.5, mean=0.04, vol=0.2)
    state = np.array([-0.01, 0.04, 0.09], dtype=np.float64)

    np.testing.assert_allclose(vasicek.drift(state, 0.0), np.array([0.025, 0.0, -0.025]))
    np.testing.assert_allclose(vasicek.diffusion(state, 0.0), np.full(3, 0.2))


def test_vasicek_zero_coupon_bond_price_is_positive():
    """Test Vasicek analytical zero-coupon bond pricing produces finite positive prices."""
    vasicek = VasicekModel(speed=0.8, mean=0.03, vol=0.015)

    price = vasicek.zero_coupon_bond_price(short_rate=0.025, maturity=5.0)

    assert np.isfinite(price)
    assert price > 0.0


def test_ou_simulation_shape_initial_value_and_determinism():
    """Test OU output shape, spread support, and fixed-seed determinism."""
    ou = OrnsteinUhlenbeckProcess(speed=1.5, mean=0.0, vol=0.8)

    spec = SimulationSpec(
        spot=2.0,
        maturity=2.0,
        steps=128,
        paths=48,
        seed=202,
    )

    paths1 = ou.simulate(spec)
    paths2 = ou.simulate(spec)

    assert paths1.shape == (48, 129)
    assert paths2.shape == (48, 129)
    np.testing.assert_array_equal(paths1[:, 0], 2.0)
    np.testing.assert_array_equal(paths1, paths2)
    assert np.any(paths1 < 0.0)


def test_ou_drift_and_diffusion_terms():
    """Test OU instantaneous coefficients match the mathematical process definition."""
    ou = OrnsteinUhlenbeckProcess(speed=0.25, mean=1.0, vol=0.4)
    state = np.array([-1.0, 1.0, 3.0], dtype=np.float64)

    np.testing.assert_allclose(ou.drift(state, 0.0), np.array([0.5, 0.0, -0.5]))
    np.testing.assert_allclose(ou.diffusion(state, 0.0), np.full(3, 0.4))


def test_ou_half_life_and_stationary_variance():
    """Test OU statistical helper methods."""
    ou = OrnsteinUhlenbeckProcess(speed=0.5, mean=0.0, vol=0.2)

    np.testing.assert_allclose(ou.half_life(), np.log(2.0) / 0.5)
    np.testing.assert_allclose(ou.stationary_variance(), 0.2**2 / (2.0 * 0.5))


def test_ou_vanilla_option_price_is_finite_and_positive():
    """Test OU Gaussian terminal-value option pricing."""
    ou = OrnsteinUhlenbeckProcess(speed=1.0, mean=0.0, vol=0.5)

    call = ou.price_vanilla(spot=0.2, strike=0.0, maturity=1.0, rate=0.03, option_type="call")
    put = ou.price_vanilla(spot=0.2, strike=0.0, maturity=1.0, rate=0.03, option_type="put")

    assert np.isfinite(call)
    assert np.isfinite(put)
    assert call > 0.0
    assert put > 0.0


def test_invalid_parameters():
    """Test that validate_parameters catches invalid model configurations."""
    # Test GBM with invalid volatility
    with pytest.raises(ValueError, match="GBM requires a positive 'vol'"):
        GeometricBrownianMotion(vol=-0.1)

    with pytest.raises(ValueError, match="GBM requires a positive 'vol'"):
        GeometricBrownianMotion(vol=0.0)

    # Test CIR with invalid speed
    with pytest.raises(ValueError, match="CIR requires a positive 'speed'"):
        CoxIngersollRoss(speed=-0.1, mean=0.04, vol=0.15)

    with pytest.raises(ValueError, match="CIR requires a positive 'speed'"):
        CoxIngersollRoss(speed=0.0, mean=0.04, vol=0.15)

    # Test CIR with invalid mean
    with pytest.raises(ValueError, match="CIR requires a positive 'mean'"):
        CoxIngersollRoss(speed=0.5, mean=-0.01, vol=0.15)

    with pytest.raises(ValueError, match="CIR requires a positive 'mean'"):
        CoxIngersollRoss(speed=0.5, mean=0.0, vol=0.15)

    # Test CIR with invalid vol
    with pytest.raises(ValueError, match="CIR requires a positive 'vol'"):
        CoxIngersollRoss(speed=0.5, mean=0.04, vol=-0.1)

    with pytest.raises(ValueError, match="CIR requires a positive 'vol'"):
        CoxIngersollRoss(speed=0.5, mean=0.04, vol=0.0)

    # Test Vasicek with invalid speed
    with pytest.raises(ValueError, match="Vasicek requires a positive 'speed'"):
        VasicekModel(speed=-0.1, mean=0.04, vol=0.15)

    with pytest.raises(ValueError, match="Vasicek requires a positive 'speed'"):
        VasicekModel(speed=0.0, mean=0.04, vol=0.15)

    # Test Vasicek with invalid mean
    with pytest.raises(ValueError, match="Vasicek requires a finite 'mean'"):
        VasicekModel(speed=0.5, mean=np.nan, vol=0.15)

    # Test Vasicek with invalid vol
    with pytest.raises(ValueError, match="Vasicek requires a positive 'vol'"):
        VasicekModel(speed=0.5, mean=0.04, vol=-0.1)

    with pytest.raises(ValueError, match="Vasicek requires a positive 'vol'"):
        VasicekModel(speed=0.5, mean=0.04, vol=0.0)

    # Test OU with invalid speed
    with pytest.raises(ValueError, match="OU requires a positive 'speed'"):
        OrnsteinUhlenbeckProcess(speed=-0.1, mean=0.0, vol=1.0)

    with pytest.raises(ValueError, match="OU requires a positive 'speed'"):
        OrnsteinUhlenbeckProcess(speed=0.0, mean=0.0, vol=1.0)

    # Test OU with invalid mean
    with pytest.raises(ValueError, match="OU requires a finite 'mean'"):
        OrnsteinUhlenbeckProcess(speed=0.5, mean=np.nan, vol=1.0)

    # Test OU with invalid vol
    with pytest.raises(ValueError, match="OU requires a positive 'vol'"):
        OrnsteinUhlenbeckProcess(speed=0.5, mean=0.0, vol=-1.0)

    with pytest.raises(ValueError, match="OU requires a positive 'vol'"):
        OrnsteinUhlenbeckProcess(speed=0.5, mean=0.0, vol=0.0)
