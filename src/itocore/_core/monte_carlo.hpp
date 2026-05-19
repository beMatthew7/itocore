#ifndef ITOCORE_MONTE_CARLO_HPP
#define ITOCORE_MONTE_CARLO_HPP

#include <random>
#include <cmath>
#include <cstdint>

/**
 * @brief Simulate Geometric Brownian Motion (GBM) paths using the Euler-Maruyama scheme.
 *
 * @param output Direct pointer to the NumPy array memory buffer (row-major: [paths][steps+1])
 * @param n_paths Total number of parallel simulation paths
 * @param n_steps Number of discrete time steps per path
 * @param S0 Initial asset spot price
 * @param mu Drift coefficient (risk-free rate / expected return)
 * @param sigma Diffusion coefficient (asset volatility)
 * @param dt Time step size (T / n_steps)
 * @param seed Numerical seed for explicit pseudorandom generation reproducibility
 */
inline void simulate_gbm_cpp(
    double* output,
    size_t n_paths,
    size_t n_steps,
    double S0,
    double mu,
    double sigma,
    double dt,
    uint64_t seed
) {
    // Precompute mathematical constants outside the loops for maximum efficiency
    const double drift = (mu - 0.5 * sigma * sigma) * dt;
    const double diffusion = sigma * std::sqrt(dt);

    // Instantiate the 64-bit Mersenne Twister engine using the Python-injected seed
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Iterate through each path using raw C-style memory strides
    for (size_t p = 0; p < n_paths; ++p) {
        // Calculate the memory offset for the beginning of the current row
        size_t row_offset = p * (n_steps + 1);
        output[row_offset] = S0; // S(0)

        // Generate consecutive chronological time steps
        for (size_t s = 1; s <= n_steps; ++s) {
            const double z = dist(rng); // Gaussian white noise
            const double prev = output[row_offset + s - 1];
            output[row_offset + s] = prev * std::exp(drift + diffusion * z);
        }
    }
}

#endif // ITOCORE_MONTE_CARLO_HPP