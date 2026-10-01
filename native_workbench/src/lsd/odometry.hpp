#pragma once
// Keyframe-based direct monocular SLAM (LSD_SLAM.md): frames are tracked on
// SE(3) against the current keyframe's semi-dense inverse-depth map, and
// every tracked frame refines that map by stereo (SemiDenseDepth). A new
// keyframe inherits the previous keyframe's hypotheses by propagation; the
// depth network seeds high-gradient pixels that propagation did not reach,
// rescaled to agree with the previous keyframe. The replaced keyframe is
// finalised into the keyframe graph: Sim(3) alignment to its parent and
// nearby keyframes, reciprocal check, pose-graph optimisation.
//
// Frames are stored relative to their keyframe, so the whole trajectory
// follows the keyframes when the graph moves them.
//
// Usage per frame: track(); if the result asks for a keyframe, run the depth
// network on that frame and call add_keyframe() with its depth.
#include "lsd/keyframe.hpp"
#include "lsd/keyframe_graph.hpp"
#include "lsd/place_recognition.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace slam_native::lsd {

struct OdometryConfig {
  int pyramid_levels{6};  // 640x480 -> 20x15, the paper's coarsest level for Sim(3) alignment
  bool stereo{true};  // refine keyframe depth by stereo and propagate it; off: network depth per keyframe only
  bool graph{true};   // Sim(3) constraints and pose-graph optimisation when a keyframe is finalised
  double prior_relative_sigma{0.25};  // depth-network error as a fraction of depth -> inverse-depth sigma
  // New keyframe when (|t| * mean keyframe inverse depth * distance_weight)^2
  // + ((1 - usage) * usage_weight)^2 > 1 (lsd_slam's criterion).
  double keyframe_distance_weight{4};
  double keyframe_usage_weight{3};
  bool constant_velocity{true};
  bool scale_lock{true};  // rescale each new keyframe's prior to the previous keyframe's depth
  Se3TrackerConfig tracker;
  DepthFilterConfig filter;  // gradient_threshold also selects the pixels seeded from the prior
  ConstraintConfig constraints;
  LoopConfig loops;  // needs keyframe features (add_keyframe)
};

enum class TrackingState { waiting, tracking, lost };

struct FrameResult {
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  TrackingState state{TrackingState::waiting};
  Sim3 pose;  // camera <- world at the time of tracking (last good pose when lost)
  Se3TrackingResult tracking;  // pose: this frame <- keyframe
  int keyframe{-1};          // the keyframe tracked against
  bool keyframe_wanted{};    // call add_keyframe() for this frame
  double keyframe_score{};   // the keyframe criterion (>= 1 triggers)
  StereoStats stereo;        // the depth update with this frame (candidates = 0: none)
  double ms{};               // tracking
  double mapping_ms{};       // stereo, regularisation and the new tracking reference
};

struct TrajectoryEntry {
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  int keyframe{};     // the keyframe the frame was tracked against (or is)
  SE3 relative;       // camera <- that keyframe
  Sim3 pose;          // camera <- world, kept up to date with the graph
  bool is_keyframe{};
};

class Odometry {
 public:
  Odometry(Camera camera, OdometryConfig config);

  // Pixels with mask == 0 (e.g. undistortion samples outside the input) never get depth.
  void set_mask(ImageF mask) { mask_ = std::move(mask); }
  // Track a frame (level-0 grayscale at camera()'s resolution).
  const FrameResult& track(std::uint64_t frame_index, std::int64_t timestamp_ns, const ImageF& image,
                           Se3TrackingDebug* debug = nullptr);
  // Make the last tracked frame a keyframe. depth_m: metric depth prior on the
  // same grid (<= 0 or NaN where unknown; may be empty). features: appearance
  // features for loop candidates (may be empty). Returns false if too few pixels.
  bool add_keyframe(const ImageF& depth_m, KeyframeFeatures features = {});

  [[nodiscard]] const Camera& camera() const { return camera_; }
  [[nodiscard]] OdometryConfig& config() { return config_; }
  [[nodiscard]] const FrameResult& last() const { return last_; }
  [[nodiscard]] const std::vector<std::unique_ptr<Keyframe>>& keyframes() const { return keyframes_; }
  [[nodiscard]] const Keyframe* current_keyframe() const {
    return keyframes_.empty() ? nullptr : keyframes_.back().get();
  }
  [[nodiscard]] const std::vector<TrajectoryEntry>& trajectory() const { return trajectory_; }
  [[nodiscard]] const KeyframeGraph& graph() const { return graph_; }
  [[nodiscard]] const ConstraintReport& last_constraints() const { return last_constraints_; }
  // Loop candidates of the last finalised keyframe, and totals.
  [[nodiscard]] const std::vector<LoopCandidate>& last_loop_candidates() const { return last_loop_candidates_; }
  [[nodiscard]] int loop_candidates_total() const { return loop_candidates_total_; }
  // Changes whenever the graph moved keyframes (views re-read poses).
  [[nodiscard]] std::uint64_t graph_generation() const { return graph_generation_; }
  [[nodiscard]] const std::string& event() const { return event_; }

 private:
  void refresh(Keyframe& keyframe) const;  // depth, tracking reference and cloud from the filter
  void finalize(Keyframe& keyframe);       // release the filter, build its depth pyramid, add to the graph
  void update_poses();                     // current keyframe and trajectory from the graph

  Camera camera_;
  OdometryConfig config_;
  ImageF mask_;
  std::vector<std::unique_ptr<Keyframe>> keyframes_;
  std::vector<TrajectoryEntry> trajectory_;
  KeyframeGraph graph_;
  ConstraintReport last_constraints_;
  std::vector<LoopCandidate> last_loop_candidates_;
  int loop_candidates_total_{};
  std::uint64_t graph_generation_{};
  FrameResult last_;
  ImageF last_image_;
  // Constant velocity, relative to the current keyframe.
  SE3 last_relative_, previous_relative_;
  bool has_previous_{};
  std::string event_;
};

}  // namespace slam_native::lsd
