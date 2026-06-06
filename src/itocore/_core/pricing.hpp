#ifndef ITOCORE_PRICING_HPP
#define ITOCORE_PRICING_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

struct MonteCarloPriceResult {
    double price;
    double standard_error;
};

inline MonteCarloPriceResult price_vanilla_terminal_cpp(
    const double* paths,
    size_t n_paths,
    size_t n_steps_plus_one,
    double strike,
    double rate,
    double maturity,
    const std::string& option_type
) {
    if (n_paths == 0) {
        throw std::runtime_error("paths must contain at least one path");
    }
    if (n_steps_plus_one < 2) {
        throw std::runtime_error("paths must contain at least one time step");
    }
    if (strike <= 0.0) {
        throw std::runtime_error("strike must be positive");
    }
    if (maturity <= 0.0) {
        throw std::runtime_error("maturity must be positive");
    }

    const bool is_call = option_type == "call";
    const bool is_put = option_type == "put";
    if (!is_call && !is_put) {
        throw std::runtime_error("option_type must be either 'call' or 'put'");
    }

    double sum = 0.0;
    double sum_sq = 0.0;

    for (size_t p = 0; p < n_paths; ++p) {
        const double terminal = paths[p * n_steps_plus_one + (n_steps_plus_one - 1)];
        const double intrinsic = is_call ? terminal - strike : strike - terminal;
        const double payoff = intrinsic > 0.0 ? intrinsic : 0.0;
        sum += payoff;
        sum_sq += payoff * payoff;
    }

    const double discount = std::exp(-rate * maturity);
    const double mean = sum / static_cast<double>(n_paths);
    double standard_error = 0.0;

    if (n_paths > 1) {
        const double mean_sq = sum_sq / static_cast<double>(n_paths);
        const double variance = (mean_sq - mean * mean) * static_cast<double>(n_paths) /
                                static_cast<double>(n_paths - 1);
        standard_error = discount * std::sqrt(std::fmax(variance, 0.0) / static_cast<double>(n_paths));
    }

    return MonteCarloPriceResult{discount * mean, standard_error};
}

inline MonteCarloPriceResult price_asian_arithmetic_cpp(
    const double* paths,
    size_t n_paths,
    size_t n_steps_plus_one,
    double strike,
    double rate,
    double maturity,
    const std::string& option_type,
    bool include_initial
) {
    if (n_paths == 0) {
        throw std::runtime_error("paths must contain at least one path");
    }
    if (n_steps_plus_one < 2) {
        throw std::runtime_error("paths must contain at least one time step");
    }
    if (strike <= 0.0) {
        throw std::runtime_error("strike must be positive");
    }
    if (maturity <= 0.0) {
        throw std::runtime_error("maturity must be positive");
    }

    const bool is_call = option_type == "call";
    const bool is_put = option_type == "put";
    if (!is_call && !is_put) {
        throw std::runtime_error("option_type must be either 'call' or 'put'");
    }

    const size_t start_index = include_initial ? 0 : 1;
    const size_t averaging_count = n_steps_plus_one - start_index;
    if (averaging_count == 0) {
        throw std::runtime_error("averaging window must contain at least one value");
    }

    double sum = 0.0;
    double sum_sq = 0.0;

    for (size_t p = 0; p < n_paths; ++p) {
        const size_t row_offset = p * n_steps_plus_one;
        double path_sum = 0.0;

        for (size_t s = start_index; s < n_steps_plus_one; ++s) {
            path_sum += paths[row_offset + s];
        }

        const double average = path_sum / static_cast<double>(averaging_count);
        const double intrinsic = is_call ? average - strike : strike - average;
        const double payoff = intrinsic > 0.0 ? intrinsic : 0.0;
        sum += payoff;
        sum_sq += payoff * payoff;
    }

    const double discount = std::exp(-rate * maturity);
    const double mean = sum / static_cast<double>(n_paths);
    double standard_error = 0.0;

    if (n_paths > 1) {
        const double mean_sq = sum_sq / static_cast<double>(n_paths);
        const double variance = (mean_sq - mean * mean) * static_cast<double>(n_paths) /
                                static_cast<double>(n_paths - 1);
        standard_error = discount * std::sqrt(std::fmax(variance, 0.0) / static_cast<double>(n_paths));
    }

    return MonteCarloPriceResult{discount * mean, standard_error};
}

#endif // ITOCORE_PRICING_HPP
