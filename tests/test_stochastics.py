import pytest
import numpy as np
from itocore.stochastics import GeometricBrownianMotion, CoxIngersollRoss, SimulationSpec


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