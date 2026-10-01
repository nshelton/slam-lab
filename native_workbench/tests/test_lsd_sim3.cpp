// Direct Sim(3) keyframe alignment (src/lsd/sim3_tracker) on the ray-cast
// scene: two keyframes with true semi-dense depth, keyframe j in different
// depth units (scale 1.3), so S_ji carries a scale the photometric term alone
// cannot see. Also the adjoint and the reciprocal check.
#include "lsd/sim3_tracker.hpp"
#include "lsd_scene.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace slam_native::lsd;
using namespace lsd_scene;
using slam_native::vo::exp_so3;
using slam_native::vo::log_so3;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

InverseDepthMap semi_dense(const ImageF& image, const ImageF& idepth, double units) {
  ImageF gx, gy;
  gradients(image, gx, gy);
  InverseDepthMap map{ImageF(kW, kH), ImageF(kW, kH)};
  for (int y = 0; y < kH; ++y)
    for (int x = 0; x < kW; ++x)
      if (std::hypot(gx.at(x, y), gy.at(x, y)) > 4) {
        map.idepth.at(x, y) = static_cast<float>(idepth.at(x, y) / units);  // depth in `units`
        const double sigma = 0.01 * map.idepth.at(x, y);
        map.variance.at(x, y) = static_cast<float>(sigma * sigma);
      }
  return map;
}

struct Errors {
  double t_mm, r_deg, s_rel;
};
Errors compare(const Sim3& a, const Sim3& b) {
  const Sim3 e = a * b.inverse();
  return {1000 * e.t.norm(), log_so3(e.R).norm() * 180 / M_PI, std::abs(e.s - 1)};
}
}  // namespace

int main() {
  try {
    constexpr double kUnitsJ = 1.3;  // keyframe j's depth = 1.3 x metric
    const SE3 pose_i{};              // camera <- world
    const SE3 pose_j{exp_so3(Vec3{0.02, -0.06, 0.01}), Vec3{-0.15, 0.03, 0.08}};
    ImageF depth_i, depth_j;
    const ImageF image_i = render(pose_i, &depth_i), image_j = render(pose_j, &depth_j);
    const Pyramid pyr_i = build_pyramid(image_i, kCamera, 5), pyr_j = build_pyramid(image_j, kCamera, 5);
    const InverseDepthMap map_i = semi_dense(image_i, depth_i, 1.0), map_j = semi_dense(image_j, depth_j, kUnitsJ);
    const TrackingReference ref_i(pyr_i, map_i), ref_j(pyr_j, map_j);
    const auto dp_i = depth_pyramid(map_i, 5), dp_j = depth_pyramid(map_j, 5);

    // Truth: X_j(units j) = units_j * (R X_i + t).
    const SE3 rel = pose_j * pose_i.inverse();
    const Sim3 truth_ji{kUnitsJ, rel.R, kUnitsJ * rel.t};
    const Sim3 truth_ij = truth_ji.inverse();
    Sim3TrackerConfig config;

    // 1. From the rigid guess (scale 1, translation unscaled, rotation off by ~1 deg).
    const Sim3 initial{1.0, exp_so3(Vec3{0.01, 0.01, 0}) * rel.R, rel.t};
    const auto r_ji = track_sim3(ref_i, pyr_j, dp_j, initial, config);
    const Errors e = compare(r_ji.pose, truth_ji);
    std::cout << "j<-i: t " << e.t_mm << " mm, r " << e.r_deg << " deg, scale " << 100 * e.s_rel << " %, usage "
              << r_ji.usage << ", depth residuals on " << 100 * r_ji.depth_fraction << " %, " << r_ji.iterations
              << " it\n";
    require(r_ji.success, "j<-i: failed");
    require(e.s_rel < 0.01 && e.r_deg < 0.1 && e.t_mm < 5, "j<-i: pose error too large");

    // 2. The reverse direction and the reciprocal check.
    const auto r_ij = track_sim3(ref_j, pyr_i, dp_i, initial.inverse(), config);
    const Errors e2 = compare(r_ij.pose, truth_ij);
    std::cout << "i<-j: t " << e2.t_mm << " mm, r " << e2.r_deg << " deg, scale " << 100 * e2.s_rel << " %\n";
    require(r_ij.success && e2.s_rel < 0.01, "i<-j: failed");
    const Mat7 cov_ji = r_ji.hessian.inverse(), cov_ij = r_ij.hessian.inverse();
    const double consistent = reciprocal_error(r_ji.pose, cov_ji, r_ij.pose, cov_ij);
    const Sim3 wrong{1.0, r_ij.pose.R, r_ij.pose.t};  // a reverse estimate that missed the scale
    const double inconsistent = reciprocal_error(r_ji.pose, cov_ji, wrong, cov_ij);
    std::cout << "reciprocal error: consistent " << consistent << ", scale missed " << inconsistent << "\n";
    require(inconsistent > 100 * std::max(consistent, 1.0), "reciprocal check should separate the two");

    // 3. Adjoint: S exp(d) == exp(Ad d) S to first order.
    {
      const Sim3 S{1.7, exp_so3(Vec3{0.3, -0.2, 0.5}), Vec3{0.4, -1.1, 2.0}};
      Vec7 d;
      d << 1e-4, -2e-4, 3e-4, 2e-4, 1e-4, -1e-4, 3e-4;
      const auto step = [](const Vec7& v) { return Sim3{std::exp(v(6)), exp_so3(v.head<3>()), v.segment<3>(3)}; };
      const Vec7 lhs = sim3_error(S * step(d) * (step(sim3_adjoint(S) * d) * S).inverse());
      require(lhs.norm() < 1e-6, "adjoint mismatch: " + std::to_string(lhs.norm()));
    }
    std::cout << "lsd sim3 tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
}
