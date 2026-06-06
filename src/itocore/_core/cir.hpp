
#ifndef ITOCORE_CIR_HPP
#define ITOCORE_CIR_HPP

#include <random>
#include <cmath>
#include <cstdint>
#include <algorithm>

/**
 * @brief Simulate Cox-Ingersoll-Ross (CIR) process paths using the Euler-Maruyama scheme.
 *
 * @param output Direct pointer to the NumPy array memory buffer (row-major: [paths][steps+1])
 * @param n_paths Total number of parallel simulation paths
 * @param n_steps Number of discrete time steps per path
 * @param X0 Initial interest rate (spot)
 * @param a Speed of mean reversion (must be positive)
 * @param b Long-term mean level (must be positive)
 * @param sigma Volatility parameter (must be positive)
 * @param dt Time step size (T / n_steps)
 * @param seed Numerical seed for explicit pseudorandom generation reproducibility
 */
inline void simulate_cir_cpp(
    double* output,
    size_t n_paths,
    size_t n_steps,
    double X0,
    double a,
    double b,
    double sigma,
    double dt,
    uint64_t seed
) {
    // Precompute constants for efficiency
    const double a_dt = a * dt;
    const double a_b_dt = a * b * dt;
    const double sigma_sqrt_dt = sigma * std::sqrt(dt);

    // Instantiate the 64-bit Mersenne Twister engine using the Python-injected seed
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Iterate through each path using raw C-style memory strides
    for (size_t p = 0; p < n_paths; ++p) {
        // Calculate the memory offset for the beginning of the current row
        size_t row_offset = p * (n_steps + 1);
        output[row_offset] = X0; // X(0)

        // Generate consecutive chronological time steps
        for (size_t s = 1; s <= n_steps; ++s) {
            const double z = dist(rng); // Gaussian white noise
            const double prev = output[row_offset + s - 1];

            // Drift term: a*(b - X_t)*dt = a*b*dt - a*X_t*dt
            const double drift = a_b_dt - a_dt * prev;

            // Diffusion term: sigma * sqrt(max(X_t, 0.0)) * dW_t
            // Guard against negative values due to discretization
            const double diffusion = sigma_sqrt_dt * std::sqrt(std::max(prev, 0.0)) * z;

            output[row_offset + s] = std::max(prev + drift + diffusion, 0.0);
        }
    }
}

#endif // ITOCORE_CIR_HPP
