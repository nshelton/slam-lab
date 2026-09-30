#pragma once
// Live monocular visual odometry from 2D point tracks.
//
// The solver is tracker-agnostic: anything that can say "track T was observed
// at pixel (x, y) in frame F" can drive it (the GPU flow tracker, the older
// descriptor tracker, tracks loaded from CSV, another program's output).
// Track IDs must be unique within a run and must not be reused after a track
// ends. Only real observations belong in a TrackedFrame; predicted or
// coasted positions must be left out.
#include "slam_native/types.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace slam_native {

struct TrackObservation {
  std::uint64_t track_id{};
  float x{};  // pixels, in the frame's own resolution
  float y{};
};

struct TrackedFrame {
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  int width{};
  int height{};
  std::vector<TrackObservation> observations;
};

// Supported (SuperPoint-observed) tracks of a tracker output frame.
TrackedFrame tracked_frame_from(const FrameFeatures& frame);

struct CameraIntrinsics {
  double fx{}, fy{}, cx{}, cy{};  // pixels of the tracked frames
  static CameraIntrinsics from_horizontal_fov(int width, int height, double degrees);
};

// World -> camera transform: x_camera = R * X_world + t. Row-major R.
struct Pose {
  std::array<double, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::array<double, 3> translation{};
  [[nodiscard]] std::array<double, 3> center() const;  // camera centre in world
};

struct VisualOdometryConfig {
  // Intrinsics in the tracked frames' pixels; when unset, derived from the
  // horizontal field of view with a centred principal point.
  std::optional<CameraIntrinsics> intrinsics;
  double horizontal_fov_degrees{60.0};

  // Initialization (two-view, essential matrix).
  int min_init_tracks{100};             // common tracks with the reference frame
  double init_min_flow_fraction{0.01};  // median track displacement / image width
  // At least min_init_points must individually exceed
  // min_triangulation_parallax_degrees, and their median must reach this.
  double init_min_parallax_degrees{1.0};
  int min_init_points{60};
  double homography_ratio{0.85};        // H/E inlier ratio above which to wait
  double ransac_threshold_px{1.5};
  int ransac_iterations{300};
  int max_init_frames{180};             // restart the reference after this long

  // Tracking.
  int min_tracked_points{20};
  double reprojection_threshold_px{3.0};

  // Keyframes and mapping.
  int keyframe_min_interval{6};
  int keyframe_max_interval{20};
  double keyframe_track_ratio{0.75};    // tracked landmarks vs last keyframe
  double min_triangulation_parallax_degrees{0.5};
  int window_keyframes{8};
  int bundle_iterations{8};

  void validate() const;
};

enum class OdometryState { initializing, tracking, lost };
const char* to_string(OdometryState state);

struct OdometryFrameResult {
  std::uint64_t frame_index{};
  OdometryState state{OdometryState::initializing};
  bool has_pose{};
  Pose pose;
  bool keyframe{};
  int segment{};             // increments after each loss / reinitialization
  int correspondences{};     // observed tracks with a 3D landmark
  int inliers{};
  std::size_t map_points{};
  std::size_t keyframes{};   // in the current segment
  double median_reprojection_px{};
  double ms{};               // processing time of this frame (incl. mapping)
  std::string event;         // human-readable note (why waiting, lost, ...)
};

struct TrajectorySample {
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  int segment{};
  bool keyframe{};
  Pose pose;  // latest estimate: follows keyframe bundle-adjustment updates
};

class VisualOdometry {
 public:
  explicit VisualOdometry(VisualOdometryConfig config = {});
  ~VisualOdometry();
  VisualOdometry(const VisualOdometry&) = delete;
  VisualOdometry& operator=(const VisualOdometry&) = delete;

  OdometryFrameResult process(const TrackedFrame& frame);
  void reset();

  [[nodiscard]] std::vector<TrajectorySample> trajectory() const;
  // Samples [first, end). Samples before stable_prefix() can no longer change
  // (their keyframes left the bundle-adjustment window), so viewers can cache
  // them and refresh only the tail.
  [[nodiscard]] std::vector<TrajectorySample> trajectory(std::size_t first) const;
  [[nodiscard]] std::size_t stable_prefix() const;
  [[nodiscard]] std::size_t trajectory_size() const;
  // 3D points of the current segment's active map.
  [[nodiscard]] std::vector<std::array<float, 3>> map_points() const;
  // Track IDs that currently have a 3D landmark (for display).
  [[nodiscard]] bool has_landmark(std::uint64_t track_id) const;
  [[nodiscard]] std::optional<std::array<double, 3>> landmark(std::uint64_t track_id) const;
  [[nodiscard]] const OdometryFrameResult& last() const;
  [[nodiscard]] const VisualOdometryConfig& config() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace slam_native
