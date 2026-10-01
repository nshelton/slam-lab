#pragma once
// Direct SE(3) image alignment of a frame against a keyframe with a
// semi-dense inverse-depth map (LSD-SLAM, ECCV 2014, Sec. 3.3).
//
// Minimises sum_p huber(r_p / sigma_p) with
//   r_p       = I_frame(w(p, d_p, T)) - I_key(p)
//   sigma_p^2 = 2 sigma_I^2 + (dr_p / dd_p)^2 var(d_p)
// by Levenberg-Marquardt on left perturbations T <- exp(delta) T,
// delta = (omega, v), coarse to fine.
#include "lsd/image.hpp"

#include <Eigen/Core>
#include <array>
#include <vector>

namespace slam_native::lsd {
using Mat6 = Eigen::Matrix<double, 6, 6>;

// Inverse depth and its variance on the level-0 grid. A pixel is valid when
// idepth > 0 and variance > 0.
struct InverseDepthMap {
  ImageF idepth, variance;
  [[nodiscard]] bool valid(int x, int y) const { return idepth.at(x, y) > 0 && variance.at(x, y) > 0; }
};

// Inverse-variance-weighted 2x2 reduction; the variance is the harmonic mean
// of the children's (neighbours are correlated, so averaging does not shrink it).
InverseDepthMap downsample(const InverseDepthMap& map);

// A keyframe prepared for tracking: its semi-dense points per pyramid level.
class TrackingReference {
 public:
  struct Point {
    Vec3 x;          // 3D point in the keyframe camera (ray / idepth)
    Vec3 ray;        // pixel ray with z = 1
    double idepth{};
    double variance{};
    float intensity{};
    float u{}, v{};  // keyframe pixel
  };

  TrackingReference() = default;
  TrackingReference(const Pyramid& pyramid, const InverseDepthMap& depth);
  [[nodiscard]] int levels() const { return static_cast<int>(points_.size()); }
  [[nodiscard]] const std::vector<Point>& points(int level) const { return points_[level]; }

 private:
  std::vector<std::vector<Point>> points_;
};

struct Se3TrackerConfig {
  double sigma_intensity{4};  // image noise, 0-255 scale
  double huber{3};            // on the normalised residual r / sigma
  // Iterations per level, index = level (0 finest).
  std::array<int, 6> max_iterations{5, 20, 50, 100, 100, 100};
  double initial_lambda{0};
  double convergence_eps{1e-6};  // stop when |delta| is below this
  double min_relative_decrease{0.999};
  int border{1};                // projected points must stay this far inside the image (pixels)
  double min_usage{0.3};        // fraction of keyframe points that must project inside for success
  int first_level{-1};          // coarsest level used; -1: the coarsest available
  int last_level{0};            // finest level used
};

struct Se3TrackingResult {
  SE3 pose;  // frame <- keyframe
  bool success{};
  Mat6 hessian{Mat6::Zero()};  // J^T W J at the finest level used (inverse = covariance)
  double mean_energy{};        // robust normalised energy per used point, finest level
  double usage{};              // projected-inside fraction of keyframe points, finest level
  int used_points{};
  int iterations{};
};

// Optional per-point diagnostics at the finest level, aligned with
// TrackingReference::points(last_level). NaN where a point was not used.
struct Se3TrackingDebug {
  std::vector<float> residual;    // r / sigma
  std::vector<float> weight;      // Huber weight in [0, 1]
};

Se3TrackingResult track_se3(const TrackingReference& reference, const Pyramid& frame, const SE3& initial,
                            const Se3TrackerConfig& config, Se3TrackingDebug* debug = nullptr);

}  // namespace slam_native::lsd
