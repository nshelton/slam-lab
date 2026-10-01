#pragma once
// Semi-dense inverse-depth filter of one keyframe (Engel, Sturm, Cremers,
// "Semi-Dense Visual Odometry for a Monocular Camera", ICCV 2013, Sec. 2;
// used per keyframe by LSD-SLAM, ECCV 2014, Sec. 3.4).
//
// One Gaussian hypothesis (inverse depth d, variance) per keyframe pixel:
// - observe(): small-baseline stereo against a tracked frame. A 5-sample SSD
//   search along the epipolar line, limited to d +- 2 sigma when a
//   hypothesis exists; observation variance alpha^2 (sigma_geo^2 + sigma_photo^2);
//   Kalman fusion.
// - propagate(): forward-warps a previous keyframe's hypotheses into a new
//   keyframe (variance * (d1/d0)^4), resolving collisions by fusion or occlusion.
// - seed(): fills pixels without a hypothesis from a depth prior.
// - regularize(): inverse-variance-weighted neighbourhood mean (neighbours
//   further than 2 sigma excluded; variances unchanged), plus hole filling and
//   outlier removal from the neighbourhood's validity counts.
// Constants without a value in the papers follow lsd_slam's defaults
// (inverse depth units: lsd_slam normalises keyframes to mean inverse depth 1;
// ours are metric, which for indoor scenes is the same order).
#include "lsd/image.hpp"
#include "lsd/se3_tracker.hpp"

#include <cstdint>
#include <vector>

namespace slam_native::lsd {

struct DepthFilterConfig {
  double gradient_threshold{5};   // |grad I| on the keyframe for stereo and seeding
  double sigma_intensity{4};      // image noise (0-255)
  double sigma_epipolar_px{2};    // epipolar line position error (pose, calibration, rolling shutter); 2 px calibrates err/sigma ~1.2 on TUM fr1 (2026-10-01)
  double max_ssd{1300};           // best 5-sample SSD must be below this
  double min_second_ratio{1.5};   // non-adjacent second-best SSD / best SSD
  double min_epl_angle_cos2{0.09};  // (gradient . epipolar direction)^2 / |gradient|^2
  double min_epl_gradient2{4};    // squared intensity change along the epipolar line, per pixel
  double min_epl_length_px{1};    // epipolar motion of a unit inverse-depth change, else no baseline
  double max_search_px{30};       // longer search segments are cropped (around the prior when there is one)
  double min_search_px{3};        // shorter ones are extended
  double max_idepth{10};          // search range for pixels without a hypothesis: (0, max]
  double max_variance{0.25};      // hypotheses above this are dropped
  double success_variance_factor{1.01};  // prediction inflation per fused observation
  double fail_variance_factor{1.1};      // after an inconsistent or failed observation
  double consistency_sigmas2{1};  // (d_obs - d)^2 <= this * (var_obs + var): consistent
  double propagation_variance{0};  // sigma_p^2 added on propagation
  double max_propagation_intensity_diff{40};
  double regularize_sigmas{2};    // neighbours further than this many sigma do not contribute
  float validity_initial{5};
  float validity_increment{5};
  float validity_max{5};          // plus validity_max_gradient * |grad| / 255
  float validity_max_gradient{250};
  float validity_sum_create{30};  // 5x5 validity sum to fill a hole
  float validity_sum_keep{24};    // 5x5 validity sum below which a hypothesis is removed
};

struct DepthHypothesis {
  float idepth{}, variance{};
  float idepth_smoothed{}, variance_smoothed{};
  float validity{};
  bool valid{};
};

struct StereoStats {
  int candidates{};    // pixels considered (gradient, mask)
  int skipped{};       // no baseline, bad angle, low gradient along the line, outside
  int matched{};       // fused into an existing hypothesis
  int created{};       // new hypotheses from a full-range search
  int inconsistent{};  // a match far from the hypothesis
  int failed{};        // SSD too high or ambiguous
  int removed{};       // dropped (variance above max)
  double ms{};
};

class SemiDenseDepth {
 public:
  // keyframe: level 0 of the keyframe pyramid. mask (optional, same size):
  // pixels with mask == 0 never get a hypothesis.
  SemiDenseDepth(const Level& keyframe, const DepthFilterConfig& config, const ImageF* mask = nullptr);

  // Stereo with a tracked frame (its level 0), frame_from_keyframe = tracked pose.
  StereoStats observe(const Level& frame, const SE3& frame_from_keyframe);
  // This keyframe's hypotheses from a previous keyframe's. Returns the number placed.
  int propagate(const SemiDenseDepth& previous, const SE3& this_from_previous);
  // Hypotheses for high-gradient pixels without one: prior inverse depth (<= 0
  // or NaN: unknown) with variance (relative_sigma * d)^2. Returns the number seeded.
  int seed(const ImageF& prior_idepth, double relative_sigma);
  void regularize(bool fill_holes = true);

  // Smoothed inverse depth and variance (what tracking uses).
  [[nodiscard]] InverseDepthMap tracking_map() const;
  [[nodiscard]] const std::vector<DepthHypothesis>& pixels() const { return pixels_; }
  [[nodiscard]] const DepthHypothesis& at(int x, int y) const { return pixels_[index(x, y)]; }
  [[nodiscard]] int width() const { return width_; }
  [[nodiscard]] int height() const { return height_; }
  [[nodiscard]] int valid_count() const;
  [[nodiscard]] DepthFilterConfig& config() { return config_; }

 private:
  [[nodiscard]] std::size_t index(int x, int y) const { return static_cast<std::size_t>(y) * width_ + x; }
  [[nodiscard]] bool usable(int x, int y) const;  // gradient and mask
  [[nodiscard]] float validity_cap(int x, int y) const;
  // Stereo for one pixel over inverse depths [d_min, d_max]; prior > 0 centres a cropped search.
  enum class Search { ok, skipped, failed };
  Search search(int x, int y, const Level& frame, const SE3& T, double d_min, double d_max, double prior,
                float& idepth, float& variance) const;

  Level keyframe_;  // a copy, so the filter does not depend on the keyframe's storage
  DepthFilterConfig config_;
  int width_{}, height_{};
  std::vector<DepthHypothesis> pixels_;
  std::vector<std::uint8_t> mask_;  // 1 = may hold a hypothesis
};

}  // namespace slam_native::lsd
