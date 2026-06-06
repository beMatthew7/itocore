#ifndef ITOCORE_OU_HPP
#define ITOCORE_OU_HPP

#include <cmath>
#include <cstdint>
#include <random>

/**
 * @brief Simulate Ornstein-Uhlenbeck paths using the exact Gaussian transition.
 *
 * The process is dX_t = speed * (mean - X_t) dt + vol dW_t.
 */
inline void simulate_ou_cpp(
    double* output,
    size_t n_paths,
    size_t n_steps,
    double x0,
    double speed,
    double mean,
    double vol,
    double dt,
    uint64_t seed
) {
    const double exp_neg_speed_dt = std::exp(-speed * dt);
    const double conditional_std = vol * std::sqrt((1.0 - std::exp(-2.0 * speed * dt)) / (2.0 * speed));

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t p = 0; p < n_paths; ++p) {
        const size_t row_offset = p * (n_steps + 1);
        output[row_offset] = x0;

        for (size_t s = 1; s <= n_steps; ++s) {
            const double z = dist(rng);
            const double prev = output[row_offset + s - 1];
            output[row_offset + s] = mean + (prev - mean) * exp_neg_speed_dt + conditional_std * z;
        }
    }
}

#endif // ITOCORE_OU_HPP
