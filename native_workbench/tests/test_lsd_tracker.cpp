// Direct SE(3) tracking (src/lsd) on a ray-cast scene: a textured back wall
// and a nearer box, rendered with 3x3 supersampling. The keyframe carries the
// true inverse depth on high-gradient pixels.
#include "lsd/se3_tracker.hpp"
#include "lsd_scene.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace slam_native::lsd;
using slam_native::vo::exp_so3;
using slam_native::vo::log_so3;
using namespace lsd_scene;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

// Semi-dense keyframe depth: true inverse depth where the gradient is large.
InverseDepthMap semi_dense(const ImageF& image, const ImageF& idepth, double variance) {
  ImageF gx, gy;
  gradients(image, gx, gy);
  InverseDepthMap map{ImageF(kW, kH), ImageF(kW, kH)};
  for (int y = 0; y < kH; ++y)
    for (int x = 0; x < kW; ++x)
      if (std::hypot(gx.at(x, y), gy.at(x, y)) > 4) {
        map.idepth.at(x, y) = idepth.at(x, y);
        map.variance.at(x, y) = static_cast<float>(variance);
      }
  return map;
}

SE3 motion(const Vec3& omega_deg, const Vec3& t) {
  return {exp_so3(omega_deg * M_PI / 180), t};
}

struct Error {
  double t_mm, r_deg;
};
Error error(const SE3& estimate, const SE3& truth) {
  const SE3 e = estimate * truth.inverse();
  return {1000 * e.t.norm(), log_so3(e.R).norm() * 180 / M_PI};
}

int count_valid(const InverseDepthMap& m) {
  int n = 0;
  for (int y = 0; y < kH; ++y)
    for (int x = 0; x < kW; ++x) n += m.valid(x, y);
  return n;
}
}  // namespace

int main() {
  try {
    ImageF key_idepth;
    const ImageF key = render(SE3{}, &key_idepth);
    const InverseDepthMap depth = semi_dense(key, key_idepth, 1e-6);
    const Pyramid key_pyramid = build_pyramid(key, kCamera, 5);
    require(key_pyramid.size() == 5 && key_pyramid.levels[4].image.width == 20, "pyramid: 5 levels down to 20x15");
    const TrackingReference reference(key_pyramid, depth);
    std::cout << "keyframe points: " << count_valid(depth) << " of " << kW * kH << " (level 4: "
              << reference.points(4).size() << ")\n";
    Se3TrackerConfig config;

    // 1. Small and 2. large motion from identity.
    for (const auto& [name, truth] :
         {std::pair{"small", motion({0.5, -0.8, 0.3}, {0.03, 0.01, -0.02})},
          std::pair{"large", motion({3, -4, 2}, {0.18, -0.06, 0.12})}}) {
      const Pyramid frame = build_pyramid(render(truth), kCamera, 5);
      const auto t0 = std::chrono::steady_clock::now();
      const Se3TrackingResult r = track_se3(reference, frame, SE3{}, config);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      const Error e = error(r.pose, truth);
      std::cout << name << ": t " << e.t_mm << " mm, r " << e.r_deg << " deg, usage " << r.usage << ", energy "
                << r.mean_energy << ", " << r.iterations << " it, " << ms << " ms\n";
      require(r.success, std::string(name) + ": tracking failed");
      // ~1.2 mm remains from occlusion at the box edges (0.04-0.3 mm on the wall alone).
      require(e.t_mm < 2 && e.r_deg < 0.05, std::string(name) + ": pose error too large");
    }

    // 3. Noisy depth on 40% of points. With the true variance the noisy points
    // are down-weighted (paper eq. 14); with a constant variance they are not.
    {
      std::mt19937 rng(7);
      std::uniform_real_distribution<double> pick(0, 1);
      std::normal_distribution<double> normal(0, 1);
      const double noisy_sigma = 0.08;  // inverse depth (~30% at 4 m)
      InverseDepthMap weighted = depth, unweighted = depth;
      for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
          if (!depth.valid(x, y) || pick(rng) > 0.4) continue;
          const float d = std::max(0.02F, static_cast<float>(depth.idepth.at(x, y) + noisy_sigma * normal(rng)));
          weighted.idepth.at(x, y) = unweighted.idepth.at(x, y) = d;
          weighted.variance.at(x, y) = static_cast<float>(noisy_sigma * noisy_sigma);
        }
      const SE3 truth = motion({1, -1.5, 0.5}, {0.08, 0.02, 0.05});
      const Pyramid frame = build_pyramid(render(truth), kCamera, 5);
      const Error ew = error(track_se3(TrackingReference(key_pyramid, weighted), frame, SE3{}, config).pose, truth);
      const Error eu = error(track_se3(TrackingReference(key_pyramid, unweighted), frame, SE3{}, config).pose, truth);
      std::cout << "noisy depth: variance-weighted t " << ew.t_mm << " mm r " << ew.r_deg << " deg; constant-variance t "
                << eu.t_mm << " mm r " << eu.r_deg << " deg\n";
      // Measured 2026-10-01: 6.8 mm weighted vs 31 mm constant.
      require(ew.t_mm < 10, "noisy depth: weighted tracking error too large");
      require(ew.t_mm < 0.5 * eu.t_mm, "noisy depth: variance weighting should at least halve the error");
    }

    // 4. Diagnostics are aligned with the finest level's points.
    {
      Se3TrackingDebug debug;
      const SE3 truth = motion({0.2, 0.2, 0}, {0.01, 0, 0});
      track_se3(reference, build_pyramid(render(truth), kCamera, 5), SE3{}, config, &debug);
      require(debug.residual.size() == reference.points(0).size(), "debug buffers aligned with level-0 points");
    }
    std::cout << "lsd tracker tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
}
