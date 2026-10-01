#include "lsd/se3_tracker.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>

namespace slam_native::lsd {

InverseDepthMap downsample(const InverseDepthMap& map) {
  const int w = map.idepth.width / 2, h = map.idepth.height / 2;
  InverseDepthMap out{ImageF(w, h), ImageF(w, h)};
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      double weighted = 0, information = 0;
      int n = 0;
      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          if (!map.valid(2 * x + dx, 2 * y + dy)) continue;
          const double ivar = 1.0 / map.variance.at(2 * x + dx, 2 * y + dy);
          weighted += ivar * map.idepth.at(2 * x + dx, 2 * y + dy);
          information += ivar;
          ++n;
        }
      }
      if (n == 0) continue;
      out.idepth.at(x, y) = static_cast<float>(weighted / information);
      out.variance.at(x, y) = static_cast<float>(n / information);
    }
  }
  return out;
}

TrackingReference::TrackingReference(const Pyramid& pyramid, const InverseDepthMap& depth) {
  InverseDepthMap level_depth = depth;
  for (int l = 0; l < pyramid.size(); ++l) {
    if (l > 0) level_depth = downsample(level_depth);
    const Level& level = pyramid.levels[l];
    std::vector<Point> points;
    for (int y = 0; y < level.image.height; ++y) {
      for (int x = 0; x < level.image.width; ++x) {
        if (x >= level_depth.idepth.width || y >= level_depth.idepth.height || !level_depth.valid(x, y)) continue;
        Point p;
        p.ray = level.camera.ray(x, y);
        p.idepth = level_depth.idepth.at(x, y);
        p.variance = level_depth.variance.at(x, y);
        p.x = p.ray / p.idepth;
        p.intensity = level.image.at(x, y);
        p.u = static_cast<float>(x);
        p.v = static_cast<float>(y);
        points.push_back(p);
      }
    }
    points_.push_back(std::move(points));
  }
}

namespace {
using Vec6d = Eigen::Matrix<double, 6, 1>;

struct Linearization {
  Mat6 H{Mat6::Zero()};
  Vec6d b{Vec6d::Zero()};
  double energy{};
  int used{};
  int inside{};
};

double huber_weight(double rn, double k) {
  const double a = std::abs(rn);
  return a <= k ? 1.0 : k / a;
}
double huber_cost(double rn, double k) {
  const double a = std::abs(rn);
  return a <= k ? rn * rn : k * (2 * a - k);
}

// Residuals, Huber weights and the normal equations of one level at `pose`.
Linearization linearize(const std::vector<TrackingReference::Point>& points, const Level& level, const SE3& pose,
                        const Se3TrackerConfig& config, Se3TrackingDebug* debug) {
  Linearization lin;
  const Camera& cam = level.camera;
  const double two_sigma2 = 2 * config.sigma_intensity * config.sigma_intensity;
  if (debug) {
    debug->residual.assign(points.size(), std::numeric_limits<float>::quiet_NaN());
    debug->weight.assign(points.size(), std::numeric_limits<float>::quiet_NaN());
  }
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto& p = points[i];
    const Vec3 X = pose * p.x;
    if (X.z() <= 1e-9) continue;
    const Vec2 uv = cam.project(X);
    if (!level.image.inside(uv.x(), uv.y(), config.border) || !level.measured(uv.x(), uv.y())) continue;
    ++lin.inside;
    const double I = level.image.sample(uv.x(), uv.y());
    const double gx = level.gx.sample(uv.x(), uv.y()), gy = level.gy.sample(uv.x(), uv.y());
    const double r = I - p.intensity;

    const double iz = 1.0 / X.z();
    // dI / dX' (image gradient through the projection).
    const Vec3 g{gx * cam.fx * iz, gy * cam.fy * iz, -(gx * cam.fx * X.x() + gy * cam.fy * X.y()) * iz * iz};
    // dr / d(idepth): X' = R ray / d + t  ->  dX'/dd = -R ray / d^2.
    const double dr_dd = -g.dot(pose.R * p.ray) / (p.idepth * p.idepth);
    const double sigma2 = two_sigma2 + dr_dd * dr_dd * p.variance;
    const double rn = r / std::sqrt(sigma2);
    const double hw = huber_weight(rn, config.huber);
    const double w = hw / sigma2;

    Vec6d J;
    J.head<3>() = X.cross(g);  // dI/domega for X' <- exp(omega) X'
    J.tail<3>() = g;
    lin.H.noalias() += w * J * J.transpose();
    lin.b.noalias() += w * r * J;
    lin.energy += huber_cost(rn, config.huber);
    ++lin.used;
    if (debug) {
      debug->residual[i] = static_cast<float>(rn);
      debug->weight[i] = static_cast<float>(hw);
    }
  }
  return lin;
}
}  // namespace

Se3TrackingResult track_se3(const TrackingReference& reference, const Pyramid& frame, const SE3& initial,
                            const Se3TrackerConfig& config, Se3TrackingDebug* debug) {
  Se3TrackingResult result;
  result.pose = initial;
  const int coarsest = std::min({reference.levels(), frame.size(), static_cast<int>(config.max_iterations.size())}) - 1;
  const int first = config.first_level < 0 ? coarsest : std::min(config.first_level, coarsest);
  const int last = std::clamp(config.last_level, 0, first);

  SE3 pose = initial;
  for (int l = first; l >= last; --l) {
    const auto& points = reference.points(l);
    const Level& level = frame.levels[l];
    Linearization lin = linearize(points, level, pose, config, nullptr);
    double lambda = config.initial_lambda;
    for (int it = 0; it < config.max_iterations[l]; ++it) {
      if (lin.used < 6) break;
      ++result.iterations;
      // Inner loop: increase damping until a step lowers the energy.
      bool accepted = false;
      for (int attempt = 0; attempt < 10; ++attempt) {
        Mat6 A = lin.H;
        A.diagonal() *= 1 + lambda;
        const Vec6d delta = -A.ldlt().solve(lin.b);
        if (!delta.allFinite()) break;
        const SE3 candidate = vo::perturb(pose, delta);
        Linearization next = linearize(points, level, candidate, config, nullptr);
        // Compare energies per used point; a step must not win by pushing points out.
        const double before = lin.energy / std::max(lin.used, 1);
        const double after = next.used >= 6 ? next.energy / next.used : std::numeric_limits<double>::infinity();
        if (after < before) {
          const bool converged = delta.norm() < config.convergence_eps || after / before > config.min_relative_decrease;
          pose = candidate;
          lin = std::move(next);
          lambda = lambda <= 0.2 ? 0 : lambda * 0.5;
          accepted = true;
          if (converged) it = config.max_iterations[l];
          break;
        }
        if (delta.norm() < config.convergence_eps) break;
        lambda = lambda <= 0 ? 0.2 : lambda * 4;
      }
      if (!accepted) break;
    }
    if (l == last) {
      if (debug) linearize(points, level, pose, config, debug);
      result.hessian = lin.H;
      result.used_points = lin.used;
      result.mean_energy = lin.used > 0 ? lin.energy / lin.used : 0;
      result.usage = points.empty() ? 0 : static_cast<double>(lin.inside) / static_cast<double>(points.size());
    }
  }
  result.pose = pose;
  result.success = result.used_points >= 6 && result.usage >= config.min_usage;
  return result;
}

}  // namespace slam_native::lsd
