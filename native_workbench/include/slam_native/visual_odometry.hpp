#pragma once
// Live monocular visual odometry from 2D point tracks.
//
// The solver is tracker-agnostic: anything that can say "track T was observed
// at pixel (x, y) in frame F" can drive it (the GPU flow tracker, tracks
// loaded from CSV, another program's output). Track IDs must be unique within
// a run and must not be reused after a track ends. Only real observations
// belong in a TrackedFrame; predicted or coasted positions must be left out.
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
  // (observations.size() x descriptor_dimension), or empty. Lets a landmark
  // whose track ended be found again by a new track (re-association).
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

  // Tracking. When neither motion prediction converges, the pose is
  // re-estimated by P3P RANSAC (recovery_threshold_px) before declaring loss.
  int min_tracked_points{20};
  double reprojection_threshold_px{3.0};
  double recovery_threshold_px{4.0};

  // Keyframes and mapping.
  int keyframe_min_interval{6};
  int keyframe_max_interval{20};
  double keyframe_track_ratio{0.75};    // tracked landmarks vs last keyframe
  // Emergency keyframe, ignoring keyframe_min_interval: when the tracked
  // landmarks collapse below this fraction of the last keyframe's, a keyframe
  // refills the map before tracking is lost. In fast motion tracked landmarks
  // halve every frame (fr1_room), faster than the minimum interval allows.
  double keyframe_emergency_ratio{0.4};
  double min_triangulation_parallax_degrees{0.5};
  int window_keyframes{8};
  int bundle_iterations{8};

  // Local map and re-association. A landmark stays in the local map while one
  // of the segment's last local_map_keyframes keyframes saw it, whether or not
  // a track still observes it; then it is retired. Each tracked frame, local
  // landmarks without a live track that project into view are matched to this
  // frame's tracks within reassociation_radius_px by descriptor (mutual best,
  // cosine >= min_descriptor_similarity). A matched track without a landmark
  // takes it over; a track whose own landmark is a pose inlier carries a
  // duplicate of the same point, and the two landmarks are merged.
  // Measured on dreamworks.MOV: re-associations within 2 px of the projection
  // fit the next pose 92% of the time, 2-3 px 72%, beyond 3 px under 20%.
  int local_map_keyframes{30};
  double reassociation_radius_px{2.0};
  double min_descriptor_similarity{0.7};

  // Coasting and relocalization. When the live tracks do not give a pose
  // (occlusion, sudden motion), the frame's pose is the constant-velocity
  // prediction, reported with the motion model's uncertainty
  // (OdometryFrameResult::predicted, confidence), and the local map is kept.
  // Its landmarks are searched by descriptor around their predicted
  // projections, within 3 sigma of that uncertainty (at least
  // relocalization_min_radius_px, at most relocalization_radius_px); P3P RANSAC
  // (recovery_threshold_px) needs relocalization_min_inliers to resume the
  // segment, and the matched tracks take over their landmarks. The segment
  // ends when the uncertainty exceeds relocalization_radius_px (the prediction
  // no longer constrains the search) or after coast_max_frames.
  int coast_max_frames{90};
  double relocalization_min_radius_px{16.0};
  double relocalization_radius_px{256.0};
  int relocalization_min_inliers{30};

  void validate() const;
};

enum class OdometryState { initializing, tracking, coasting, lost };
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
  int reassociated{};        // landmarks taken over by a new track this frame
  int merged{};              // of those, duplicates folded into the older landmark
  bool predicted{};          // coasting: the pose is the motion model's prediction
  bool relocalized{};        // the local map was found again by descriptor search
  // Pose uncertainty, when has_pose. The error is the perturbation delta =
  // (omega, v) with T_true = exp(delta) * T: omega (radians) rotates about the
  // camera centre, v (map units) moves the camera, both in camera axes.
  // Row-major 6x6. A tracked pose's comes from its inliers (residual variance
  // x inverse Gauss-Newton Hessian, landmarks taken as exact); a predicted
  // pose's is that propagated by the constant-velocity model, whose
  // acceleration noise is estimated from the segment's own prediction errors.
  std::array<double, 36> pose_covariance{};
  // The same in the image: RMS per-axis standard deviation of the landmarks'
  // projections under pose_covariance, in pixels.
  double pose_sigma_px{};
  // Probability that a landmark projects within reprojection_threshold_px of
  // where it is: 1 - exp(-threshold^2 / (2 pose_sigma_px^2)).
  double confidence{};
  // Track IDs with a landmark this frame, split by the pose fit (sorted; for
  // display). Inliers reproject within reprojection_threshold_px.
  std::vector<std::uint64_t> pose_inliers, pose_outliers;
};

// A triangulated landmark. Landmark IDs are unique per run; track_id is the
// latest track that observed it (several can over its life).
struct MapPoint {
  std::uint64_t landmark_id{};
  std::uint64_t track_id{};
  std::array<float, 3> position{};      // world, segment's arbitrary scale
  std::array<std::uint8_t, 3> color{};  // mean RGB over its keyframe observations
  bool has_color{};
  int segment{};
  int keyframe_observations{};
  std::uint64_t first_frame{};  // first keyframe that observed it
  std::uint64_t last_frame{};   // last frame it was an inlier (or keyframe observation)
};

struct TrajectorySample {
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  int segment{};
  bool keyframe{};
  Pose pose;  // latest estimate: follows keyframe bundle-adjustment updates
  // Coasted frame (see OdometryFrameResult::predicted). Once the segment is
  // found again, its pose is interpolated between the tracked poses on either
  // side; if it never is, coasted frames are removed.
  bool predicted{};
  double confidence{};  // OdometryFrameResult::confidence when the frame was processed
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
  // Landmarks of the current segment's local map (see local_map_keyframes).
  [[nodiscard]] std::vector<MapPoint> active_map() const;
  // Landmarks that left the local map, or whose segment ended; all segments,
  // in retirement order. Append-only until reset() and
  // final, so viewers can fetch only new ones.
  [[nodiscard]] std::size_t retired_count() const;
  [[nodiscard]] std::vector<MapPoint> retired_map(std::size_t first = 0) const;
  // Track IDs that currently have a 3D landmark (for display).
  [[nodiscard]] bool has_landmark(std::uint64_t track_id) const;
  [[nodiscard]] const OdometryFrameResult& last() const;
  [[nodiscard]] const VisualOdometryConfig& config() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace slam_native
