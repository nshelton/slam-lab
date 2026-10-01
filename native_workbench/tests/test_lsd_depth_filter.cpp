// Semi-dense depth filter (src/lsd/semi_dense_depth) on the ray-cast scene
// with exact poses: refinement of a biased prior, stereo-only initialisation,
// propagation to a new keyframe, and the stereo variance's calibration.
#include "lsd/semi_dense_depth.hpp"
#include "lsd_scene.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace slam_native::lsd;
using namespace lsd_scene;
using slam_native::vo::exp_so3;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

// Camera k of a sideways-and-forward move with a slight turn (camera <- world).
SE3 pose_at(int k) {
  const SE3 world_from_camera{exp_so3(Vec3{0, 0.002 * k, 0}), Vec3{0.012 * k, 0.003 * k, 0.01 * k}};
  return world_from_camera.inverse();
}

struct Accuracy {
  int valid{};
  double median_rel{};  // |d - d_true| / d_true
  double z_robust{};    // median |d - d_true| / sigma / 0.6745: ~1 when the variance is calibrated
};

Accuracy accuracy(const SemiDenseDepth& filter, const ImageF& truth) {
  std::vector<double> rel, z;
  for (int y = 0; y < filter.height(); ++y)
    for (int x = 0; x < filter.width(); ++x) {
      const auto& h = filter.at(x, y);
      if (!h.valid) continue;
      const double t = truth.at(x, y), e = h.idepth_smoothed - t;
      rel.push_back(std::abs(e) / t);
      z.push_back(std::abs(e) / std::sqrt(h.variance_smoothed));
    }
  Accuracy a;
  a.valid = static_cast<int>(rel.size());
  if (rel.empty()) return a;
  std::nth_element(rel.begin(), rel.begin() + static_cast<long>(rel.size() / 2), rel.end());
  a.median_rel = rel[rel.size() / 2];
  std::nth_element(z.begin(), z.begin() + static_cast<long>(z.size() / 2), z.end());
  a.z_robust = z[z.size() / 2] / 0.6745;
  return a;
}

void print(const char* name, const Accuracy& a) {
  std::cout << name << ": " << a.valid << " px, median error " << 100 * a.median_rel << " %, error/sigma " << a.z_robust
            << "\n";
}
}  // namespace

int main() {
  try {
    ImageF truth;
    const ImageF key_image = render(pose_at(0), &truth);
    const Level key = build_pyramid(key_image, kCamera, 1).levels[0];
    std::vector<Level> frames;
    for (int k = 1; k <= 12; ++k) frames.push_back(build_pyramid(render(pose_at(k)), kCamera, 1).levels[0]);
    const auto relative = [&](int k) { return pose_at(k) * pose_at(0).inverse(); };  // frame <- keyframe
    DepthFilterConfig config;

    // 1. A biased, noisy prior (20 % per-pixel noise + 15 % global bias) is
    //    refined by stereo with the 12 frames.
    {
      std::mt19937 rng(3);
      std::normal_distribution<double> normal(0, 1);
      ImageF prior(kW, kH);
      for (std::size_t i = 0; i < prior.data.size(); ++i)
        prior.data[i] = static_cast<float>(truth.data[i] * 1.15 * std::exp(0.2 * normal(rng)));
      SemiDenseDepth filter(key, config);
      const int seeded = filter.seed(prior, 0.3);
      filter.regularize();
      const Accuracy before = accuracy(filter, truth);
      print("prior", before);
      StereoStats last;
      for (int k = 1; k <= 12; ++k) {
        last = filter.observe(frames[k - 1], relative(k));
        filter.regularize();
      }
      const Accuracy after = accuracy(filter, truth);
      print("refined", after);
      std::cout << "  last update: " << last.candidates << " candidates, " << last.matched << " matched, "
                << last.created << " created, " << last.inconsistent << " inconsistent, " << last.failed
                << " failed, " << last.skipped << " skipped, " << last.ms << " ms\n";
      require(seeded > 10000, "seeding: too few pixels");
      require(after.median_rel < 0.03, "refinement: median error should drop below 3 %");
      require(after.valid > 0.7 * before.valid, "refinement: too many hypotheses lost");
      require(after.z_robust < 3, "refinement: variance badly overconfident");
    }

    // 2. No prior: hypotheses created by full-range search, then refined.
    {
      SemiDenseDepth filter(key, config);
      for (int k = 1; k <= 12; ++k) {
        filter.observe(frames[k - 1], relative(k));
        filter.regularize();
      }
      const Accuracy a = accuracy(filter, truth);
      print("stereo only", a);
      require(a.valid > 8000, "stereo only: too few hypotheses");
      require(a.median_rel < 0.05, "stereo only: median error above 5 %");
    }

    // 3. Propagation of a refined map to a new keyframe at pose 12.
    {
      SemiDenseDepth filter(key, config);
      ImageF exact = truth;
      filter.seed(exact, 0.02);
      for (int k = 1; k <= 12; ++k) {
        filter.observe(frames[k - 1], relative(k));
        filter.regularize();
      }
      ImageF truth12;
      const Level key12 = build_pyramid(render(pose_at(12), &truth12), kCamera, 1).levels[0];
      SemiDenseDepth next(key12, config);
      const int placed = next.propagate(filter, relative(12));
      next.regularize();
      const Accuracy a = accuracy(next, truth12);
      std::cout << "propagated " << placed << " -> ";
      print("new keyframe", a);
      require(placed > 0.6 * filter.valid_count(), "propagation: too few hypotheses placed");
      require(a.median_rel < 0.03, "propagation: median error above 3 %");
    }
    std::cout << "lsd depth filter tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAILED: " << e.what() << "\n";
    return 1;
  }
}
