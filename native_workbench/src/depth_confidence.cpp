#include "slam_native/depth_confidence.hpp"

#include <algorithm>
#include <cmath>

namespace slam_native {
namespace {
constexpr std::size_t N = DepthConfidenceModel::kTerms;

// Solves A x = b for a small symmetric positive-definite A (Cholesky).
bool solve(std::array<std::array<double, N>, N> A, std::array<double, N>& b) {
  for (std::size_t j = 0; j < N; ++j) {
    double d = A[j][j];
    for (std::size_t k = 0; k < j; ++k) d -= A[j][k] * A[j][k];
    if (!(d > 1e-12)) return false;
    A[j][j] = std::sqrt(d);
    for (std::size_t i = j + 1; i < N; ++i) {
      double s = A[i][j];
      for (std::size_t k = 0; k < j; ++k) s -= A[i][k] * A[j][k];
      A[i][j] = s / A[j][j];
    }
  }
  for (std::size_t i = 0; i < N; ++i) {  // L y = b
    for (std::size_t k = 0; k < i; ++k) b[i] -= A[i][k] * b[k];
    b[i] /= A[i][i];
  }
  for (std::size_t i = N; i-- > 0;) {  // L^T x = y
    for (std::size_t k = i + 1; k < N; ++k) b[i] -= A[k][i] * b[k];
    b[i] /= A[i][i];
  }
  return true;
}
}  // namespace

DepthConfidenceModel::DepthConfidenceModel(DepthConfidenceConfig config) : config_(config) {
  w_[0] = 2 * std::log(config_.prior_sigma);
}

std::array<double, N> DepthConfidenceModel::features(const DepthCues& c) const {
  // log depth enters relative to a typical 2 m, so the bias stays interpretable.
  return {1.0, c.edge, c.log_depth - std::log(2.0), c.radius, c.nearest_landmark};
}

void DepthConfidenceModel::add_reference(const DepthCues& cues, double residual, double geometric_sigma) {
  if (!std::isfinite(residual) || std::abs(residual) > config_.gross_log) return;
  const double geo = std::isfinite(geometric_sigma) ? geometric_sigma : 0.0;
  Reference reference{features(cues), residual * residual, geo * geo};
  if (references_.size() < config_.max_references) references_.push_back(reference);
  else references_[next_] = reference;
  next_ = (next_ + 1) % config_.max_references;
  count_ = references_.size();
}

void DepthConfidenceModel::fit(int iterations) {
  if (count_ < config_.min_references) return;
  // NLL = 0.5 sum log v + r^2 / v, v = exp(w.x) + g. Fisher scoring:
  //   grad = 0.5 sum (1/v - r^2/v^2) e x,  I = 0.5 sum (e/v)^2 x x^T  (e = exp(w.x)),
  // plus a ridge toward the constant prior on the cue weights.
  const double prior_bias = 2 * std::log(config_.prior_sigma);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    std::array<double, N> grad{};
    std::array<std::array<double, N>, N> info{};
    for (const auto& ref : references_) {
      double z = 0;
      for (std::size_t k = 0; k < N; ++k) z += w_[k] * ref.x[k];
      const double e = std::exp(std::clamp(z, -30.0, 10.0));
      const double v = e + ref.geo2;
      const double g = 0.5 * (1 / v - ref.r2 / (v * v)) * e;
      const double h = 0.5 * (e / v) * (e / v);
      for (std::size_t a = 0; a < N; ++a) {
        grad[a] += g * ref.x[a];
        for (std::size_t b = 0; b <= a; ++b) info[a][b] += h * ref.x[a] * ref.x[b];
      }
    }
    for (std::size_t a = 0; a < N; ++a)
      for (std::size_t b = a + 1; b < N; ++b) info[a][b] = info[b][a];
    for (std::size_t k = 1; k < N; ++k) {  // ridge on cue weights
      grad[k] += config_.ridge * w_[k];
      info[k][k] += config_.ridge;
    }
    grad[0] += 1e-3 * config_.ridge * (w_[0] - prior_bias);
    info[0][0] += 1e-3 * config_.ridge;
    if (!solve(info, grad)) break;
    double step = 0;
    for (std::size_t k = 0; k < N; ++k) {
      const double delta = std::clamp(grad[k], -2.0, 2.0);  // damped
      w_[k] -= delta;
      step = std::max(step, std::abs(delta));
    }
    if (step < 1e-6) break;
  }
  fitted_ = true;
}

double DepthConfidenceModel::sigma(const DepthCues& cues) const {
  if (!fitted_) return config_.prior_sigma;
  const auto x = features(cues);
  double z = 0;
  for (std::size_t k = 0; k < N; ++k) z += w_[k] * x[k];
  return std::sqrt(std::exp(std::clamp(z, -30.0, 10.0)));
}

double DepthConfidenceModel::confidence(const DepthCues& cues) const {
  const double q = sigma(cues) / config_.prior_sigma;
  return 1.0 / (1.0 + q * q);
}

}  // namespace slam_native
