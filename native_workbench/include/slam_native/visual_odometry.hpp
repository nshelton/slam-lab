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
  bool has_color{};
  std::array<std::uint8_t, 3> color{};  // RGB of the image at (x, y), when has_color
};

struct TrackedFrame {
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  int width{};
  int height{};
  std::vector<TrackObservation> observations;
  // Optional appearance: one L2-normalised row per observation
  // (observations.size() x descriptor_dimension), or empty. Used to
  // re-associate landmarks after tracking is lost (relocalization).
  int descriptor_dimension{};
  std::vector<float> descriptors;
};

// Supported (SuperPoint-observed) tracks of a tracker output frame. `colors`,
// when given, is aligned with frame.keypoints (see ColorSampler). Descriptors
// come from frame.track_descriptors when present.
TrackedFrame tracked_frame_from(const FrameFeatures& frame,
                                const std::vector<std::array<std::uint8_t, 3>>& colors = {});

// One-parameter radial lens distortion (division model). With the radius
// normalised by the half-diagonal, r = |p - c| / (0.5 * hypot(width, height)),
// and c the image centre, the undistorted (pinhole) pixel is
//     p_u = c + (p - c) / (1 + k1 * r^2).
// k1 < 0 is barrel distortion (wide-angle lenses); 0 is none. Near the centre
// p_u ~ p, so focal length and field of view keep their central meaning.
// Valid for k1 > -1 (the denominator stays positive inside the frame).
void undistort_point(float& x, float& y, int width, int height, double k1);
// Inverse of undistort_point (pinhole pixel -> image pixel).
void distort_point(float& x, float& y, int width, int height, double k1);
TrackedFrame undistorted(const TrackedFrame& frame, double k1);

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
  double distortion_k1{0.0};  // see undistort_point; tracks are undistorted on input

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

  // Relocalization (needs descriptors on the tracked frames). When a segment
  // with at least relocalization_min_keyframes is lost, its landmarks are held
  // for relocalization_max_frames. Each frame, their projections under the
  // constant-velocity prediction are matched to observations within
  // relocalization_radius_px by descriptor, a P3P RANSAC pose is fitted, and a
  // guided search (guided_search_radius_px) adds matches. With enough inliers
  // tracking resumes in the old segment (same frame and scale), and the new
  // tracks take over the matched landmarks.
  bool relocalization{true};
  int relocalization_max_frames{90};
  int relocalization_min_keyframes{3};
  double relocalization_radius_px{256.0};
  double relocalization_min_similarity{0.7};
  double relocalization_threshold_px{4.0};
  int relocalization_iterations{300};
  int relocalization_min_inliers{30};
  double guided_search_radius_px{16.0};

  // Landmark persistence and re-association while tracking (needs
  // descriptors). A landmark whose track ended is kept "dormant" for
  // dormant_max_frames (at most max_dormant_landmarks, oldest retired first).
  // Each tracked frame, every landmark without an observation that projects
  // into view is reported (OdometryFrameResult::untracked_landmarks); with
  // reassociation on, it takes over a track that has no landmark yet when
  // that track lies within reassociation_radius_px of its projection and the
  // descriptors are mutual best matches (cosine >= relocalization_min_similarity).
  bool reassociation{true};
  // Measured on dreamworks.MOV: re-associations within 2 px of the projection
  // fit the next pose 92% of the time, 2-3 px 72%, beyond 3 px under 20%.
  double reassociation_radius_px{2.0};
  int dormant_max_frames{900};
  int max_dormant_landmarks{20000};

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
  bool relocalized{};        // this frame resumed a lost segment
  // Landmarks that project into this frame but have no observation in it,
  // in the input frame's (distorted) pixels. track_id: the landmark's own
  // (ended or missing) track. `reassociated`: it took over new_track_id this
  // frame (and is known by that ID from now on).
  struct ProjectedLandmark {
    std::uint64_t track_id{};
    float x{}, y{};
    bool dormant{};
    bool reassociated{};
    std::uint64_t new_track_id{};
  };
  std::vector<ProjectedLandmark> untracked_landmarks;
  int reassociated{};
  int relocalization_candidates{};  // descriptor matches tried this frame (0 if no attempt)
  // Track IDs with a landmark this frame, split by the pose fit (sorted; for
  // display). Inliers reproject within reprojection_threshold_px.
  std::vector<std::uint64_t> pose_inliers, pose_outliers;
};

// A triangulated landmark. The ID is the track ID that produced it; with the
// frame range and a track export, its observations (and, through the feature
// cache, the SuperPoint descriptors at those pixels) can be recovered later.
struct MapPoint {
  std::uint64_t track_id{};
  std::array<float, 3> position{};      // world, segment's arbitrary scale
  std::array<std::uint8_t, 3> color{};  // mean RGB over its keyframe observations
  bool has_color{};
  int segment{};
  int keyframe_observations{};
  std::uint64_t first_frame{};  // first keyframe that observed it
  std::uint64_t last_frame{};   // last frame it was an inlier (or keyframe observation)
  bool dormant{};               // track ended; kept for re-association (active_map only)
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
  // Landmarks that are not final: those of the current segment still being
  // refined (tracked or in the bundle-adjustment window), plus, with
  // relocalization, recently ended ones and a lost segment's map, which may
  // still be re-found (see relocalization_max_frames).
  [[nodiscard]] std::vector<MapPoint> active_map() const;
  // Landmarks that left the window after their track ended, or whose segment
  // ended; all segments, in retirement order. Append-only until reset(), and
  // final (never refined again), so viewers can fetch only new ones.
  [[nodiscard]] std::size_t retired_count() const;
  [[nodiscard]] std::vector<MapPoint> retired_map(std::size_t first = 0) const;
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
