#include "lsd/sim3_tracker.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>

namespace slam_native::lsd {

std::vector<InverseDepthMap> depth_pyramid(const InverseDepthMap& level0, int levels) {
  std::vector<InverseDepthMap> out{level0};
  for (int l = 1; l < levels; ++l) out.push_back(downsample(out.back()));
  return out;
}

namespace {
double huber_weight(double rn, double k) {
  const double a = std::abs(rn);
  return a <= k ? 1.0 : k / a;
}
double huber_cost(double rn, double k) {
  const double a = std::abs(rn);
  return a <= k ? rn * rn : k * (2 * a - k);
}

Sim3 perturb(const Sim3& S, const Vec7& d) {
  const Sim3 step{std::exp(d(6)), vo::exp_so3(d.head<3>()), d.segment<3>(3)};
  return step * S;
}

struct Linearization {
  Mat7 H{Mat7::Zero()};
  Vec7 b{Vec7::Zero()};
  double energy{};
  int used{}, inside{}, with_depth{};
};

Linearization linearize(const std::vector<TrackingReference::Point>& points, const Level& level,
                        const InverseDepthMap& depth, const Sim3& S, const Sim3TrackerConfig& config) {
  Linearization lin;
  const Camera& cam = level.camera;
  const double two_sigma2 = 2 * config.sigma_intensity * config.sigma_intensity;
  const Mat3 sR = S.s * S.R;
  for (const auto& p : points) {
    const Vec3 X = S * p.x;
    if (X.z() <= 1e-9) continue;
    const Vec2 uv = cam.project(X);
    if (!level.image.inside(uv.x(), uv.y(), config.border) || !level.measured(uv.x(), uv.y())) continue;
    ++lin.inside;
    const double iz = 1.0 / X.z();
    // Photometric residual.
    const double gx = level.gx.sample(uv.x(), uv.y()), gy = level.gy.sample(uv.x(), uv.y());
    const double rp = level.image.sample(uv.x(), uv.y()) - p.intensity;
    const Vec3 g{gx * cam.fx * iz, gy * cam.fy * iz, -(gx * cam.fx * X.x() + gy * cam.fy * X.y()) * iz * iz};
    const Vec3 dX_dd = -(sR * p.ray) / (p.idepth * p.idepth);
    const double drp_dd = g.dot(dX_dd);
    const double var_p = two_sigma2 + drp_dd * drp_dd * p.variance;
    Vec7 Jp;
    Jp.head<3>() = X.cross(g);
    Jp.segment<3>(3) = g;
    Jp(6) = g.dot(X);
    double e2 = rp * rp / var_p;
    // Depth residual where keyframe j has depth at the nearest pixel.
    const int u = static_cast<int>(std::lround(uv.x())), v = static_cast<int>(std::lround(uv.y()));
    bool has_depth = u >= 0 && v >= 0 && u < depth.idepth.width && v < depth.idepth.height && depth.valid(u, v);
    double rd = 0, var_d = 1;
    Vec7 Jd = Vec7::Zero();
    if (has_depth) {
      rd = iz - depth.idepth.at(u, v);
      const double drd_dd = -iz * iz * dX_dd.z();  // d(1/z')/dD_i
      var_d = depth.variance.at(u, v) + drd_dd * drd_dd * p.variance;
      // d(1/z')/d delta = -(1/z'^2) dz'/d delta; dz'/domega = (X'y, -X'x, 0), dz'/dv = e_z, dz'/dlog s = z'.
      Jd << X.y(), -X.x(), 0, 0, 0, 1, X.z();
      Jd *= -iz * iz;
      e2 += rd * rd / var_d;
      ++lin.with_depth;
    }
    const double w = huber_weight(std::sqrt(e2), config.huber);
    lin.H.noalias() += (w / var_p) * Jp * Jp.transpose();
    lin.b.noalias() += (w * rp / var_p) * Jp;
    if (has_depth) {
      lin.H.noalias() += (w / var_d) * Jd * Jd.transpose();
      lin.b.noalias() += (w * rd / var_d) * Jd;
    }
    lin.energy += huber_cost(std::sqrt(e2), config.huber);
    ++lin.used;
  }
  return lin;
}
}  // namespace

Sim3TrackingResult track_sim3(const TrackingReference& reference, const Pyramid& frame,
                              const std::vector<InverseDepthMap>& frame_depth, const Sim3& initial,
                              const Sim3TrackerConfig& config) {
  Sim3TrackingResult result;
  const int coarsest = std::min({reference.levels(), frame.size(), static_cast<int>(frame_depth.size()),
                                 static_cast<int>(config.max_iterations.size())}) - 1;
  const int first = config.first_level < 0 ? coarsest : std::min(config.first_level, coarsest);
  const int last = std::clamp(config.last_level, 0, first);
  Sim3 pose = initial;
  for (int l = first; l >= last; --l) {
    const auto& points = reference.points(l);
    const Level& level = frame.levels[l];
    Linearization lin = linearize(points, level, frame_depth[l], pose, config);
    double lambda = 0;
    for (int it = 0; it < config.max_iterations[l]; ++it) {
      if (lin.used < 7) break;
      ++result.iterations;
      bool accepted = false;
      for (int attempt = 0; attempt < 10; ++attempt) {
        Mat7 A = lin.H;
        A.diagonal() *= 1 + lambda;
        const Vec7 delta = -A.ldlt().solve(lin.b);
        if (!delta.allFinite()) break;
        const Sim3 candidate = perturb(pose, delta);
        Linearization next = linearize(points, level, frame_depth[l], candidate, config);
        const double before = lin.energy / std::max(lin.used, 1);
        const double after = next.used >= 7 ? next.energy / next.used : std::numeric_limits<double>::infinity();
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
      result.hessian = lin.H;
      result.used_points = lin.used;
      result.mean_energy = lin.used > 0 ? lin.energy / lin.used : 0;
      result.usage = points.empty() ? 0 : static_cast<double>(lin.inside) / static_cast<double>(points.size());
      result.depth_fraction = lin.used > 0 ? static_cast<double>(lin.with_depth) / lin.used : 0;
    }
  }
  result.pose = pose;
  result.success = result.used_points >= 7 && result.usage >= config.min_usage && pose.s > 0 && std::isfinite(pose.s);
  return result;
}

Vec7 sim3_error(const Sim3& S) {
  Vec7 r;
  r.head<3>() = vo::log_so3(S.R);
  r.segment<3>(3) = S.t;
  r(6) = std::log(S.s);
  return r;
}

Mat7 sim3_adjoint(const Sim3& S) {
  // S exp(w, v, sigma) S^-1 = exp(R w, [t]x R w + s R v - sigma t, sigma) to first order.
  Mat3 tx;
  tx << 0, -S.t.z(), S.t.y(), S.t.z(), 0, -S.t.x(), -S.t.y(), S.t.x(), 0;
  Mat7 A = Mat7::Zero();
  A.block<3, 3>(0, 0) = S.R;
  A.block<3, 3>(3, 0) = tx * S.R;
  A.block<3, 3>(3, 3) = S.s * S.R;
  A.block<3, 1>(3, 6) = -S.t;
  A(6, 6) = 1;
  return A;
}

double reciprocal_error(const Sim3& S_ji, const Mat7& cov_ji, const Sim3& S_ij, const Mat7& cov_ij) {
  const Vec7 r = sim3_error(S_ji * S_ij);
  const Mat7 A = sim3_adjoint(S_ji);
  const Mat7 cov = cov_ji + A * cov_ij * A.transpose();
  return r.dot(cov.ldlt().solve(r));
}

}  // namespace slam_native::lsd
