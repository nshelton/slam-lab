#pragma once
// Field of view and radial distortion self-calibration from 2D point tracks.
//
// Distortion (k1, division model; see undistort_point): lens distortion bends
// epipolar lines, so with the wrong k1 no fundamental matrix fits a pair's
// tracks well (Fitzgibbon, 2001). For each candidate k1 the tracks are
// undistorted, F is refitted, and the robust epipolar residual is recorded.
// The k1 with the smallest mean residual over all pairs is the estimate. Any
// pair with enough motion informs k1, including pure translation.
//
// Focal length (field of view): with the true intrinsics K, the essential
// matrix E = K^T F K has two equal non-zero singular values; with a wrong focal
// length it does not (Mendonca & Cipolla, 1999). Assuming square pixels and a
// centred principal point, each pair gives a cost curve over candidate fields
// of view for every k1:
//     cost(fov) = (s1 - s2) / (s1 + s2),   s1 >= s2 singular values of E(fov)
// Curves are averaged over pairs; the FOV estimate is the minimum of the curve
// at the estimated k1. Pure translation carries no focal information (flat
// curve) and pairs a homography explains (pure rotation or a single plane) are
// skipped entirely: their F is undetermined.
#include "slam_native/visual_odometry.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace slam_native {

struct FocalEstimatorConfig {
  double min_hfov_degrees{30.0};
  double max_hfov_degrees{150.0};
  double hfov_step_degrees{1.0};
  double min_k1{-0.6};
  double max_k1{0.3};
  double k1_step{0.02};
  double min_displacement_fraction{0.02};  // median track displacement / width for a pair
  int max_gap_frames{90};                  // longest pair span searched
  int pair_stride_frames{10};              // frames between successive pairs
  int min_common_tracks{80};
  int min_inliers{60};
  // Candidate correspondences come from a loose RANSAC at k1 = 0 (edge tracks
  // are the most distorted and must not be rejected before k1 is searched);
  // the residual is then truncated at the tight threshold.
  double candidate_threshold_px{6.0};
  double residual_threshold_px{2.0};
  int ransac_iterations{300};
  double homography_ratio{0.8};            // skip a pair when H inliers > ratio * F inliers
  double min_curve_depth{0.02};            // pair FOV curve max - min below this: uninformative
};

struct FocalEstimate {
  std::vector<double> hfov_degrees;  // FOV grid
  std::vector<double> k1_values;     // k1 grid
  // Mean truncated epipolar residual (RMS pixels) per k1 over all used pairs.
  std::vector<double> k1_residual_px;
  // Mean FOV cost, row-major [k1 index][hfov index], over informative pairs.
  std::vector<double> cost_table;
  std::vector<double> cost;          // the row at the estimated k1 (for display)
  double best_k1{};
  double best_hfov_degrees{};        // 0 until estimated
  double best_focal_px{};            // for the frame width seen
  // Individual pairs' FOV minima at the estimated k1 and their percentiles.
  std::vector<double> pair_best_degrees;
  double pair_p25_degrees{}, pair_median_degrees{}, pair_p75_degrees{};
  int pairs_distortion{};  // pairs informing k1
  int pairs_used{};        // pairs informing the FOV
  int pairs_flat{};        // too little rotation for the FOV
  int pairs_homography{};  // rotation-only or planar
  int pairs_weak{};        // too few F inliers
  int width{}, height{};

  // FOV cost curve at any k1 (linear between grid rows) and its minimum.
  [[nodiscard]] std::vector<double> cost_at(double k1) const;
  [[nodiscard]] double best_hfov_at(double k1) const;
};

class FocalEstimator {
 public:
  explicit FocalEstimator(FocalEstimatorConfig config = {});
  // Returns true when this frame closed a new pair (the estimate changed).
  bool add(const TrackedFrame& frame);
  void reset();
  // Track IDs restart (seek, new tracker): forget buffered frames but keep the
  // accumulated evidence, since the lens belongs to the camera.
  void restart_tracks();
  [[nodiscard]] const FocalEstimate& estimate() const { return estimate_; }
  [[nodiscard]] const FocalEstimatorConfig& config() const { return config_; }

 private:
  struct Stored {
    std::uint64_t frame_index{};
    std::unordered_map<std::uint64_t, std::pair<float, float>> points;
  };
  void add_pair(const Stored& a, const Stored& b);
  void refresh();

  FocalEstimatorConfig config_;
  std::deque<Stored> frames_;
  std::uint64_t last_pair_frame_{};
  bool have_pair_{};
  std::vector<double> residual_sum_;         // per k1
  std::vector<double> cost_sum_;             // per k1 x hfov
  std::vector<std::vector<float>> pair_min_; // per informative pair: FOV minimum per k1
  FocalEstimate estimate_;
};

// Runs a FocalEstimator on a worker thread (a pair costs tens of milliseconds)
// so the caller's frame loop never waits. Frames and commands are applied in
// order; if the worker falls far behind, the oldest queued frames are dropped.
class BackgroundFocalEstimator {
 public:
  explicit BackgroundFocalEstimator(FocalEstimatorConfig config = {});
  ~BackgroundFocalEstimator();
  BackgroundFocalEstimator(const BackgroundFocalEstimator&) = delete;
  BackgroundFocalEstimator& operator=(const BackgroundFocalEstimator&) = delete;

  void add(TrackedFrame frame);
  void reset();
  void restart_tracks();
  // Copies the latest estimate into `out` if it changed since `version`;
  // returns true (and updates `version`) when it did.
  bool latest(FocalEstimate& out, std::uint64_t& version) const;

 private:
  enum class Command { frame, reset, restart_tracks };
  struct Item {
    Command command{};
    TrackedFrame frame;
  };
  void push(Item item);
  void run();

  FocalEstimator estimator_;
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Item> queue_;
  bool stop_{};
  FocalEstimate published_;
  std::uint64_t version_{};
  std::thread worker_;
};

}  // namespace slam_native
