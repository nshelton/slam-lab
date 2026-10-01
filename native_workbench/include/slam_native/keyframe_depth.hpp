#pragma once
// Dense depth clouds of keyframes (DEPTH_INTEGRATION.md Phase 2). App/bench
// side: the VO never sees depth maps. A cloud is the keyframe's network depth
// back-projected on a subsampled grid of the network's output, in the
// keyframe's camera frame, coloured from the frame. Viewers draw it with the
// keyframe's current pose (it follows bundle adjustment), dividing by
// metres_per_unit.
//
// Consistency (measured with tools/depth_consistency.py):
// - Alignment: each point is corrected by the keyframe's ScaleGrid (log
//   metres per VO unit, fitted by the VO to its landmarks), not just one
//   scale, which removes the network's low-frequency shape errors. The VO's
//   provisional grid is replaced by the settled one (settle()) when the
//   keyframe leaves the BA window.
// - Multi-view check: a new keyframe's points are projected into the last few
//   keyframes' depth maps and vice versa. A point agrees where the other view
//   sees the same depth (within consistency_tolerance, log), disagrees where
//   the other view sees through it (farther), and is unchecked where the
//   other view sees something nearer (occlusion) or doesn't see it. Points
//   that were checked and never confirmed are hidden (`shown`).
#include "slam_native/depth_confidence.hpp"
#include "slam_native/depth_estimator.hpp"
#include "slam_native/visual_odometry.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace slam_native {

struct KeyframeCloud {
  std::uint64_t frame_index{};  // the keyframe's frame (TrajectorySample::frame_index)
  int segment{};
  double metres_per_unit{};  // reference scale; 0 = unknown (viewers use the segment's)
  // Keyframe camera frame (x right, y down, z forward). Divided by
  // metres_per_unit they are in VO units with the scale grid applied.
  std::vector<std::array<float, 3>> points;
  std::vector<std::array<std::uint8_t, 3>> colors;
  std::vector<std::uint8_t> shown;  // multi-view check: 0 = hidden
  std::uint32_t version{};          // changes whenever points or shown change
  bool settled{};                   // scale from the keyframe's final geometry
  // Unscaled network points (metres) and their source pixels, to re-apply a grid.
  std::vector<std::array<float, 3>> raw;
  std::vector<std::array<float, 2>> pixels;
  std::vector<std::uint8_t> agree, disagree;  // multi-view votes per point
  // Predicted network error per point (log depth, DepthConfidenceModel at the
  // time the cloud was built); the prior sigma until the model has data.
  std::vector<float> sigma;
};

// A well-triangulated landmark seen in the keyframe: a reference for the
// network-depth confidence model (geometry-only covariance, so it is not
// itself derived from network depth; see DEPTH_INTEGRATION.md, Phase 0b).
struct DepthReference {
  float x{}, y{};       // native source pixel of its observation
  double depth{};       // camera-frame z in the keyframe, VO units
  double geometric_sigma{};  // MapPoint::depth_sigma_ratio (log depth)
};

struct KeyframeDepthConfig {
  int stride{4};            // every stride-th output pixel in each direction
  float edge_ratio{0.10F};  // drop pixels whose 3x3 max/min exceeds 1 + this ("flying pixels")
  float max_depth_m{15.0F};
  float max_median_ratio{3.0F};  // and beyond this many times the map's median depth
  std::size_t max_clouds{2000};  // oldest dropped beyond this
  int neighbours{4};             // multi-view check against this many previous keyframes
  float consistency_tolerance{0.06F};  // log depth; also the tolerance until the adaptive model is fitted
  // Adaptive tolerance: a DepthConfidenceModel of the *between-view* residual
  // (cues: edge strength in the other view's map, log depth, image radius
  // there, angle between the two viewing rays), trained only on residuals of
  // points an earlier keyframe already confirmed (so see-through outliers
  // cannot loosen it). Tolerance = clamp(sigmas x predicted sigma, min, max).
  bool adaptive_tolerance{true};
  float tolerance_sigmas{2.5F};
  float tolerance_min{0.03F};
  float tolerance_max{0.15F};
};

// What the VO says about a keyframe when it is created.
struct KeyframeDepthInput {
  std::uint64_t frame_index{};
  int segment{};
  double log_scale{std::numeric_limits<double>::quiet_NaN()};  // log metres per unit; NaN = unknown
  const ScaleGrid* grid{};  // optional (OdometryFrameResult::keyframe_scale_grid)
  Pose pose;                // world -> camera, VO units
  std::vector<DepthReference> references;  // optional: trains the per-point sigma model
};

class KeyframeDepthStore {
 public:
  using PoseLookup = std::function<std::optional<Pose>(std::uint64_t frame_index)>;
  explicit KeyframeDepthStore(KeyframeDepthConfig config = {}) : config_(config) {}

  // Native source pixels of the grid for maps with `like`'s geometry (cached).
  // Sample the frame's colour there while the frame is alive and pass the
  // colours to add() (the keyframe decision comes later).
  const std::vector<Keypoint>& grid(const DepthMap& like);
  // Builds and stores the keyframe's cloud, then cross-checks it with recent
  // keyframes. `colors`: sampled at grid(depth) (grey if the sizes differ).
  // K, k1: the VO's lens. pose_of: current poses of keyframes (after BA);
  // falls back to input.pose. Returns the number of points kept.
  std::size_t add(const KeyframeDepthInput& input, const DepthMap& depth,
                  const std::vector<std::array<std::uint8_t, 3>>& colors, const CameraIntrinsics& K, double k1,
                  const PoseLookup& pose_of = {});
  // The keyframe's settled scale (OdometryFrameResult::settled_*). Re-applies
  // it to the cloud's points. False if the keyframe has no cloud.
  bool settle(std::uint64_t frame_index, double log_scale, const ScaleGrid* grid);

  [[nodiscard]] const std::vector<KeyframeCloud>& clouds() const { return clouds_; }
  [[nodiscard]] const DepthConfidenceModel& confidence_model() const { return confidence_; }
  // Between-view residual model behind the adaptive tolerance (its
  // nearest_landmark cue holds the viewing angle between the two keyframes).
  [[nodiscard]] const DepthConfidenceModel& difference_model() const { return difference_; }
  void clear();
  // Changes whenever clouds are removed (clear, dropping the oldest), so
  // renderers drop their uploaded copies. Per-cloud changes: KeyframeCloud::version.
  [[nodiscard]] std::uint64_t generation() const { return generation_; }

 private:
  struct Cell {
    int u, v;  // output pixel
  };
  // A recent keyframe's map, kept for the multi-view check.
  struct Recent {
    std::uint64_t frame_index{};
    int segment{};
    DepthMap depth;
    double log_scale{};
    bool has_grid{};
    ScaleGrid grid{};
  };
  bool same_geometry(const DepthMap& map) const;
  KeyframeCloud* find(std::uint64_t frame_index);
  void apply_scale(KeyframeCloud& cloud, double log_scale, const ScaleGrid* grid) const;
  // Votes for `cloud`'s points against `target` (both poses world -> camera).
  // `samples` (optional) receives (cues, residual) of points already confirmed
  // by an earlier keyframe, to train difference_.
  struct DifferenceSample {
    DepthCues cues;
    double residual;
  };
  bool check(KeyframeCloud& cloud, const Pose& pose, const Recent& target, const Pose& target_pose,
             std::vector<DifferenceSample>* samples) const;
  // Adds the keyframe's references to the confidence model and refits.
  void learn(const KeyframeDepthInput& input, const DepthMap& depth);
  KeyframeDepthConfig config_;
  std::vector<KeyframeCloud> clouds_;
  std::deque<Recent> recent_;
  std::uint64_t generation_{};
  CameraIntrinsics K_{};
  double k1_{};
  DepthMap geometry_;  // metres empty: only the transform fields are used
  std::vector<Keypoint> grid_;
  std::vector<Cell> cells_;
  DepthConfidenceModel confidence_;
  DepthConfidenceModel difference_{{20000, 200, 0.04, 0.3, 1.0}};
};

}  // namespace slam_native
