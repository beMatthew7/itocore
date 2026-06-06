#ifndef ITOCORE_VASICEK_HPP
#define ITOCORE_VASICEK_HPP

#include <cmath>
#include <cstdint>
#include <random>

/**
 * @brief Simulate Vasicek paths using the exact Gaussian transition.
 *
 * The model is dr_t = a * (b - r_t) dt + sigma dW_t.
 */
inline void simulate_vasicek_cpp(
    double* output,
    size_t n_paths,
    size_t n_steps,
    double r0,
    double a,
    double b,
    double sigma,
    double dt,
    uint64_t seed
) {
    const double exp_neg_a_dt = std::exp(-a * dt);
    const double conditional_std = sigma * std::sqrt((1.0 - std::exp(-2.0 * a * dt)) / (2.0 * a));

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t p = 0; p < n_paths; ++p) {
        const size_t row_offset = p * (n_steps + 1);
        output[row_offset] = r0;

        for (size_t s = 1; s <= n_steps; ++s) {
            const double z = dist(rng);
            const double prev = output[row_offset + s - 1];
            output[row_offset + s] = b + (prev - b) * exp_neg_a_dt + conditional_std * z;
        }
    }
}

#endif // ITOCORE_VASICEK_HPP
