import time
from src.itocore.stochastics.gbm import GeometricBrownianMotion
from src.itocore.stochastics.base import SimulationSpec


def test_performance():
    # Instantiating the model with a 5% risk-free rate and 20% volatility
    model = GeometricBrownianMotion(vol=0.20, rate=0.05)

    # 500,000 paths, 252 trading days (1 year), explicit seed for reproducibility
    spec = SimulationSpec(spot=100.0, maturity=1.0, steps=252, paths=500_000, seed=42)

    print("Launching native C++ Monte Carlo engine...")
    start_time = time.time()
    paths = model.simulate(spec)
    end_time = time.time()

    print("\n" + "=" * 40)
    print("SIMULATION COMPLETED SUCCESSFULLY!")
    print("=" * 40)
    print(f"Matrix shape: {paths.shape[0]} paths x {paths.shape[1]} steps")
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Final average price: {paths[:, -1].mean():.2f} (Expected: ~105.13)")
    print("=" * 40)


if __name__ == "__main__":
    test_performance()