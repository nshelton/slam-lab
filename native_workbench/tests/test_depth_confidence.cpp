// DepthConfidenceModel recovers a known heteroscedastic noise model.
#include "slam_native/depth_confidence.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace slam_native;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

// True network sigma: 0.08 base, x3 at strong edges, growing with depth.
double true_sigma(const DepthCues& c) {
  return 0.08 * std::exp(0.5 * (2.2 * c.edge + 0.8 * (c.log_depth - std::log(2.0))));
}
}  // namespace

int main() {
  try {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0, 1);
    std::normal_distribution<double> normal(0, 1);
    DepthConfidenceModel model;
    require(std::abs(model.sigma({}) - 0.2) < 1e-12, "prior sigma before fitting");
    for (int i = 0; i < 8000; ++i) {
      DepthCues c{u(rng) < 0.2 ? static_cast<float>(0.5 + u(rng)) : 0.0F, static_cast<float>(std::log(0.5 + 6 * u(rng))),
                  static_cast<float>(u(rng)), static_cast<float>(0.2 * u(rng))};
      const double geo = 0.01;
      model.add_reference(c, std::sqrt(true_sigma(c) * true_sigma(c) + geo * geo) * normal(rng), geo);
    }
    model.fit(20);
    const DepthCues flat{0, static_cast<float>(std::log(2.0)), 0.5F, 0.05F};
    const DepthCues edge{1.0F, static_cast<float>(std::log(2.0)), 0.5F, 0.05F};
    const DepthCues far{0, static_cast<float>(std::log(6.0)), 0.5F, 0.05F};
    std::cout << "depth confidence: sigma flat " << model.sigma(flat) << " (true " << true_sigma(flat) << "), edge "
              << model.sigma(edge) << " (" << true_sigma(edge) << "), far " << model.sigma(far) << " ("
              << true_sigma(far) << "); confidence flat " << model.confidence(flat) << " edge " << model.confidence(edge)
              << "\n";
    const auto close = [](double a, double b) { return std::abs(std::log(a / b)) < 0.15; };  // within 15 %
    require(close(model.sigma(flat), true_sigma(flat)), "flat sigma not recovered");
    require(close(model.sigma(edge), true_sigma(edge)), "edge sigma not recovered");
    require(close(model.sigma(far), true_sigma(far)), "far sigma not recovered");
    require(model.confidence(flat) > model.confidence(edge), "edges must be less confident");
    // Gross residuals are ignored.
    DepthConfidenceModel robust;
    for (int i = 0; i < 1000; ++i) robust.add_reference(flat, i % 10 == 0 ? 3.0 : 0.05 * normal(rng), 0.0);
    robust.fit(20);
    require(robust.references() == 900 && robust.sigma(flat) < 0.07, "gross residuals leaked into the fit");
    std::cout << "depth confidence checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
