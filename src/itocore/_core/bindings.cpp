#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <cstdint>
#include "monte_carlo.hpp"
#include "cir.hpp"

namespace py = pybind11;

void simulate_gbm(py::array_t<double, py::array::c_style | py::array::forcecast> output,
                  double S0, double mu, double sigma, double dt, uint64_t seed) {

    // Validate that the incoming Python NumPy array is two-dimensional
    if (output.ndim() != 2) {
        throw std::runtime_error("Output array must be 2-dimensional (n_paths, n_steps+1)");
    }

    auto buffer = output.request();
    size_t n_paths = buffer.shape[0];
    size_t n_steps_plus_one = buffer.shape[1];

    if (n_steps_plus_one < 2) {
        throw std::runtime_error("Number of steps must be at least 1");
    }
    size_t n_steps = n_steps_plus_one - 1;

    // Delegate computational hot-path execution via zero-copy raw pointer abstraction
    simulate_gbm_cpp(static_cast<double*>(buffer.ptr),
                     n_paths, n_steps, S0, mu, sigma, dt, seed);
}

void simulate_cir(py::array_t<double, py::array::c_style | py::array::forcecast> output,
                  double X0, double a, double b, double sigma, double dt, uint64_t seed) {

    // Validate that the incoming Python NumPy array is two-dimensional
    if (output.ndim() != 2) {
        throw std::runtime_error("Output array must be 2-dimensional (n_paths, n_steps+1)");
    }

    auto buffer = output.request();
    size_t n_paths = buffer.shape[0];
    size_t n_steps_plus_one = buffer.shape[1];

    if (n_steps_plus_one < 2) {
        throw std::runtime_error("Number of steps must be at least 1");
    }
    size_t n_steps = n_steps_plus_one - 1;

    // Delegate computational hot-path execution via zero-copy raw pointer abstraction
    simulate_cir_cpp(static_cast<double*>(buffer.ptr),
                     n_paths, n_steps, X0, a, b, sigma, dt, seed);
}

// Generate the native binary module named '_core' accessible via Python imports
PYBIND11_MODULE(_core, m) {
    m.doc() = "ItoCore native high-performance computational engine";

    m.def("simulate_gbm", &simulate_gbm,
          py::arg("output"),
          py::arg("S0"),
          py::arg("mu"),
          py::arg("sigma"),
          py::arg("dt"),
          py::arg("seed"));

    m.def("simulate_cir", &simulate_cir,
          py::arg("output"),
          py::arg("X0"),
          py::arg("a"),
          py::arg("b"),
          py::arg("sigma"),
          py::arg("dt"),
          py::arg("seed"));
}