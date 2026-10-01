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
#include <limits>
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
  // 1-sigma localisation noise per axis (pixels); <= 0: unknown. Weights the
  // solvers with VisualOdometryConfig::use_observation_sigma.
  float sigma_px{};
  // Network depth at (x, y): z-depth in metres, and its 1-sigma; <= 0:
  // unknown. Optional, like colour and descriptors (DEPTH_INTEGRATION.md).
  float depth_m{};
  float depth_sigma_m{};
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

// A smooth per-keyframe depth correction: log metres per VO unit on a
// kScaleGridX x kScaleGridY grid of nodes spanning the image (node (0,0) at
// pixel (0,0), the last at (width-1, height-1)), bilinear in between. Indexed
// row-major (y * kScaleGridX + x), in the frame's own (distorted) pixels.
inline constexpr int kScaleGridX = 4;
inline constexpr int kScaleGridY = 3;
using ScaleGrid = std::array<double, kScaleGridX * kScaleGridY>;
double scale_grid_at(const ScaleGrid& grid, int width, int height, double x, double y);

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
  // Emergency keyframe, ignoring keyframe_min_interval: when the tracked
  // landmarks collapse (below keyframe_emergency_ratio x the last keyframe's,
  // or below keyframe_emergency_points) a keyframe refills the map before
  // tracking is lost. In fast motion tracked landmarks halve every frame
  // (fr1_room), faster than the minimum interval allows. 0 disables each test.
  double keyframe_emergency_ratio{0.4};
  int keyframe_emergency_points{0};
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
  // Let re-association merge a landmark into the one a matching track
  // already observes (that track must be a pose inlier): the same point
  // triangulated twice under two tracks.
  bool merge_landmarks{true};

  // Loop closure (needs descriptors). At each keyframe, older keyframes of the
  // segment (>= loop_min_keyframe_gap keyframes back) whose camera centre is
  // within loop_max_distance_ratio x the keyframe's median landmark depth and
  // whose optical axis is within loop_max_angle_degrees are candidates. Their
  // landmarks (active, dormant or retired) are matched to the keyframe's by
  // descriptor (mutual best, cosine >= relocalization_min_similarity); a
  // RANSAC similarity on the 3D-3D matches (|S X - Y| <= loop_distance_
  // tolerance x depth) with >= loop_min_inliers that also reprojects closes
  // the loop. A Sim3 pose graph over the segment's keyframes (consecutive,
  // covisibility and loop edges) is optimized; the correction moves keyframes,
  // landmarks and frames (map_generation() changes) and the loop's duplicate
  // landmarks are merged. Trivial corrections only merge.
  bool loop_closure{true};
  int loop_min_keyframe_gap{30};
  double loop_max_distance_ratio{1.0};
  double loop_max_angle_degrees{45.0};
  int loop_min_inliers{30};
  double loop_search_radius_px{160.0};  // descriptor search around old landmarks' projections
  double loop_distance_tolerance{0.05};
  int loop_covisibility_min{30};   // shared landmarks for a covisibility edge
  int loop_cooldown_keyframes{10};
  double loop_edge_weight{10.0};
  // Appearance (global) search, no pose prior.
  // The keyframe's landmarks are matched by descriptor against every older
  // landmark of the segment (mutual best, cosine >= relocalization_min_similarity,
  // cosine distance <= loop_global_ratio x the runner-up's), a P3P RANSAC
  // pose needs loop_global_min_inliers; the correction is that pose, scaled by
  // the median old/current landmark depth ratio. Then as above. Measured
  // offline on fr1_room: no wrong pose had more than 10 P3P inliers; true
  // revisits a median of 48. Runs at every keyframe (also during the loop
  // cooldown, for its statistics; loop_global_interval_frames > 0 thins it)
  // next to the spatial search; the loop with more verified pairs wins. It
  // brute-forces every old descriptor (match_all_descriptors): the baseline
  // to compare an indexed retrieval against.
  bool loop_global_search{true};
  int loop_global_interval_frames{0};
  int loop_global_min_inliers{20};
  double loop_global_ratio{0.9};
  int loop_global_iterations{1000};
  // The scale (old / current depth over the pose inliers that carry a
  // current landmark) is accepted when p75 <= loop_global_scale_spread x p25,
  // from at least loop_global_min_pairs pairs within 25% of the median.
  // fr1_room: without these gates a loop applied a 1.63x scale read from 18
  // pairs spread 1.45-2.03 (ATE 37.7 cm); with them 24.7 cm.
  double loop_global_scale_spread{1.3};
  int loop_global_min_pairs{20};
  // Solve the pose graph on the calling thread, applying the correction in
  // the same frame (deterministic: bench, tests). Otherwise a worker thread
  // solves it and a later frame applies it.
  bool loop_synchronous{false};
  // Global bundle adjustment (VisualOdometry::global_bundle_adjust): all
  // keyframes and landmarks of a segment, with every sighting. Keeps all
  // retired landmarks' sightings for it when > 0. Tools run it before
  // exporting (TUM suite, 20 iterations at the end: mean ATE 10.0 -> 8.4 cm,
  // fr2_desk 12.3 -> 6.4, no sequence worse).
  int final_bundle_iterations{20};
  // Global bundle adjustment of the segment right after a loop correction is
  // applied (iterations; 0: off). The pose graph only moves keyframes as
  // rigid blocks; this re-fits every keyframe and landmark to all sightings,
  // including the merged loop landmarks. fr1_room, global loop search: ATE
  // 51.5 -> 37.7 cm without the loop_global scale gates, 24.7 cm with them.
  // Runs on the calling thread (~0.1-4 s per loop on fr1_room).
  int loop_bundle_iterations{20};
  // Cull also re-checks observations from keyframes outside the BA window
  // (dropping those that no longer reproject within 2x the threshold). Off:
  // only windowed observations are checked, as before.
  bool cull_stale_observations{true};

  // Metric scale from network depth (DEPTH_INTEGRATION.md Phases 0-1; output
  // only, the solver is unchanged). At each keyframe, landmarks with
  // >= depth_scale_min_keyframes observations and >= depth_scale_min_parallax_
  // degrees whose track has a depth sample with sigma / depth <=
  // depth_scale_max_relative_sigma (edge samples are excluded) give
  // r = log(depth) - log(z in the segment's units). The keyframe's median r
  // updates a per-segment log-scale Kalman filter (drift per keyframe, error
  // floor: network errors are correlated, so many samples don't average out).
  bool use_depth{true};
  double depth_relative_sigma{0.15};    // when an observation has depth but no depth_sigma_m
  int depth_scale_min_samples{20};
  int depth_scale_min_keyframes{3};
  double depth_scale_min_parallax_degrees{2.0};
  double depth_scale_max_relative_sigma{0.2};
  double depth_scale_drift_sigma{0.02};  // log scale, per keyframe
  double depth_scale_floor_sigma{0.05};  // log scale, per keyframe measurement
  int dormant_max_frames{900};
  int max_dormant_landmarks{20000};

  // Observation noise (CONFIDENCE_DESIGN.md). Pose optimization and BA weight
  // each residual by 1 / sigma^2; thresholds stay in pixels. Without
  // use_observation_sigma (or for observations without sigma_px) every
  // observation gets observation_sigma_px, which at 1 reproduces the
  // unweighted solver exactly. The tracker's sigma is floored at
  // min_observation_sigma_px: its Kalman posterior ignores temporally
  // correlated errors, so it is optimistic for long tracks.
  bool use_observation_sigma{true};
  // Pose tracking: add each landmark's projected (geometry-only) covariance
  // to its observation's noise, anisotropically (Mahalanobis residual), so a
  // poorly triangulated landmark stops pulling the pose along its uncertain
  // direction but still constrains it across (CONFIDENCE_DESIGN.md, phase 2).
  bool pose_landmark_uncertainty{true};
  // With it: still reject outliers by plain pixel error (the covariance only
  // shapes the weights). A wrongly matched young landmark is uncertain along
  // its epipolar line, so a Mahalanobis gate would let it slide there.
  bool pose_landmark_pixel_gate{true};
  // Multiplies the landmark covariance used for those weights. The
  // geometry-only covariance holds the keyframes fixed and so is optimistic:
  // actual / predicted error ~1.4 on synthetic data (variance ~2).
  double pose_landmark_covariance_scale{2.0};
  double observation_sigma_px{1.0};
  double min_observation_sigma_px{0.5};
  // MapPoint::confidence (diagnostic) = precision x verification. Precision
  // is 0.5 at a relative depth sigma of confidence_depth_ratio. Verification
  // is confidence_two_view for a landmark seen by 2 keyframes (two rays always
  // fit, so a wrong match is invisible); each further keyframe halves the
  // remaining doubt (3: 0.63, 4: 0.81 at the default).
  double confidence_depth_ratio{0.05};
  double confidence_two_view{0.25};
  // Consistency, for landmarks with 3+ views: q = windowed reprojection RMS /
  // the map's median, factor 1 / (1 + (max(0, q - 1) / scale)^2).
  double confidence_consistency_scale{0.75};

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
  // in the input frame's (distorted) pixels. track_id: the landmark's latest
  // (ended or missing) track. `reassociated`: new_track_id now observes it
  // (an untracked track was attached, or that track's own, younger landmark
  // was merged into this one).
  struct ProjectedLandmark {
    std::uint64_t landmark_id{};
    std::uint64_t track_id{};
    float x{}, y{};
    bool dormant{};
    bool reassociated{};
    std::uint64_t new_track_id{};
  };
  std::vector<ProjectedLandmark> untracked_landmarks;
  int reassociated{};
  int merged{};              // duplicate landmarks folded into an older one this frame
  // Metric scale of this frame's segment: metres per segment unit (0 =
  // unknown) and its relative 1-sigma. On keyframes, also this keyframe's own
  // estimate (log scale, NaN without enough samples), its sample count and the
  // robust spread of log(depth / z) around it: the network's relative error
  // on reference landmarks.
  double metric_scale{};
  double metric_scale_sigma{};
  double keyframe_log_scale{std::numeric_limits<double>::quiet_NaN()};
  int keyframe_scale_samples{};
  double keyframe_scale_spread{std::numeric_limits<double>::quiet_NaN()};
  // On keyframes with enough samples: the keyframe's log-scale grid, fitted
  // robustly to the same reference landmarks (pulled towards
  // keyframe_log_scale where they are sparse). Corrects the network's
  // low-frequency shape errors as well as its scale (DEPTH_INTEGRATION.md).
  bool keyframe_scale_grid_valid{};
  ScaleGrid keyframe_scale_grid{};
  // Loop closure. Detected: a verified loop this frame (with an asynchronous
  // solve the correction arrives in a later frame). Closed: a correction was
  // applied this frame; keyframe_scale_changes lists (frame index, factor) for
  // the keyframes whose camera units changed (multiply their metres-per-unit).
  bool loop_detected{};
  bool loop_closed{};
  std::int64_t loop_keyframe{-1}, loop_candidate{-1};  // frame indices
  int loop_inliers{};
  bool loop_global{};  // the detected loop came from the appearance search
  // Appearance (global) search statistics, on keyframes where it ran:
  // queries (this keyframe's landmarks) x database (older landmarks),
  // mutual-best matches, the matcher's time, P3P inliers, and whether it
  // verified a loop (closed only when no correction is pending/cooling down).
  bool global_search{};
  int global_queries{}, global_database{}, global_matches{}, global_pose_inliers{};
  double global_match_ms{};
  bool global_found{};
  std::vector<std::pair<std::uint64_t, double>> keyframe_scale_changes;
  // On keyframes: the keyframe that just left the bundle-adjustment window
  // (-1 = none) and its scale refitted at its now-final pose and landmarks.
  // Consumers should replace that keyframe's provisional scale with it.
  std::int64_t settled_keyframe{-1};
  double settled_log_scale{std::numeric_limits<double>::quiet_NaN()};
  bool settled_scale_grid_valid{};
  ScaleGrid settled_scale_grid{};
  int relocalization_candidates{};  // descriptor matches tried this frame (0 if no attempt)
  // Track IDs with a landmark this frame, split by the pose fit (sorted; for
  // display). Inliers reproject within reprojection_threshold_px.
  std::vector<std::uint64_t> pose_inliers, pose_outliers;
  // Pose uncertainty from the inliers (observation_sigma_px noise, landmarks
  // taken as exact, so optimistic). Scale-free: the camera centre's largest
  // sigma over the inliers' median depth. NaN without a tracked pose.
  double rotation_sigma_degrees{std::numeric_limits<double>::quiet_NaN()};
  double translation_sigma_ratio{std::numeric_limits<double>::quiet_NaN()};
};

// A triangulated landmark. Landmarks have their own IDs (unique per run, never
// reused): tracks come and go, and several tracks (a re-found point, merged
// duplicates) can observe one landmark. track_id is the latest track that
// observed it; with the frame range and a track export, its observations (and,
// through the feature cache, the SuperPoint descriptors at those pixels) can
// be recovered later.
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
  bool dormant{};               // track ended; kept for re-association (active_map only)
  // Uncertainty from the keyframe observations with the keyframes held fixed
  // (geometry only; optimistic, but ranks points correctly), refreshed after
  // each keyframe and frozen at retirement. NaN: not computed yet.
  // depth_sigma_ratio: sigma along the viewing ray of the last keyframe that
  // saw it, over that distance (scale-free); confidence = verification(views)
  // x consistency(reprojection_rms) / (1 + (ratio / confidence_depth_ratio)^2),
  // see VisualOdometryConfig. reprojection_rms: whitened, each observation
  // measured while its keyframe was in the BA window (NaN until measured).
  // covariance: xx xy xz yy yz zz, world units^2.
  float depth_sigma_ratio{std::numeric_limits<float>::quiet_NaN()};
  float max_parallax_degrees{};
  float reprojection_rms{std::numeric_limits<float>::quiet_NaN()};
  float confidence{std::numeric_limits<float>::quiet_NaN()};
  std::array<float, 6> covariance{};
};

struct TrajectorySample {
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  int segment{};
  bool keyframe{};
  Pose pose;  // latest estimate: follows keyframe bundle-adjustment updates
};

// Two keyframes (by frame index) that observe `shared` common landmarks.
struct CovisibilityEdge {
  std::uint64_t frame_a{}, frame_b{};  // frame_a < frame_b
  int shared{};
};

class VisualOdometry {
 public:
  explicit VisualOdometry(VisualOdometryConfig config = {});
  ~VisualOdometry();
  VisualOdometry(const VisualOdometry&) = delete;
  VisualOdometry& operator=(const VisualOdometry&) = delete;

  OdometryFrameResult process(const TrackedFrame& frame);
  // Global bundle adjustment of every segment (iterations: Levenberg-Marquardt
  // steps; <= 0 uses config final_bundle_iterations). Gauge: each segment's
  // first two keyframes fixed. Moves keyframes, landmarks (all states) and the
  // retired map, refreshes their uncertainty, keeps tracking continuous and
  // bumps map_generation(). Returns false (nothing done) while an asynchronous
  // loop correction is pending.
  bool global_bundle_adjust(int iterations = 0);
  void reset();

  [[nodiscard]] std::vector<TrajectorySample> trajectory() const;
  // Samples [first, end). Samples before stable_prefix() can no longer change
  // (their keyframes left the bundle-adjustment window), so viewers can cache
  // them and refresh only the tail.
  [[nodiscard]] std::vector<TrajectorySample> trajectory(std::size_t first) const;
  [[nodiscard]] std::size_t stable_prefix() const;
  [[nodiscard]] std::size_t trajectory_size() const;
  // Changes when existing trajectory samples or retired points change other
  // than by appending (a loop correction): viewers then refetch everything.
  [[nodiscard]] std::uint64_t map_generation() const;
  // Landmarks that are not final: those of the current segment still being
  // refined (tracked or in the bundle-adjustment window), plus, with
  // relocalization, recently ended ones and a lost segment's map, which may
  // still be re-found (see relocalization_max_frames).
  [[nodiscard]] std::vector<MapPoint> active_map() const;
  // Landmarks that left the window after their track ended, or whose segment
  // ended; all segments, in retirement order. Append-only until reset() or a
  // map_generation() change (a loop correction moves them), and
  // final (never refined again), so viewers can fetch only new ones.
  [[nodiscard]] std::size_t retired_count() const;
  [[nodiscard]] std::vector<MapPoint> retired_map(std::size_t first = 0) const;
  // Track IDs that currently have a 3D landmark (for display).
  [[nodiscard]] bool has_landmark(std::uint64_t track_id) const;
  // Metres per unit of this segment from network depth; 0 = unknown.
  [[nodiscard]] double segment_scale(int segment) const;
  // The landmark a live track currently observes, if any.
  [[nodiscard]] std::optional<std::uint64_t> landmark_id(std::uint64_t track_id) const;
  [[nodiscard]] std::optional<std::array<double, 3>> landmark(std::uint64_t track_id) const;
  // Keyframe pairs sharing at least min_shared landmarks (active, dormant,
  // suspended and archived ones; retired landmarks without an archive copy
  // have no sightings), at least min_keyframe_gap keyframes apart. Cost grows
  // with sightings per landmark squared: for display, not per frame.
  [[nodiscard]] std::vector<CovisibilityEdge> covisibility(int min_shared, int min_keyframe_gap = 1) const;
  [[nodiscard]] const OdometryFrameResult& last() const;
  [[nodiscard]] const VisualOdometryConfig& config() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace slam_native
