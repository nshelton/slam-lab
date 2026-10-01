#pragma once
// Temporal fusion of network depth (DEPTH_FUSION.md, step 1). A per-pixel
// Kalman filter on the network's output grid, anchored to the current
// keyframe (as LSD-SLAM's keyframe depth maps); each new network map is just
// an observation.
//
// State per pixel of the anchor keyframe, in log depth in the network's
// metres. Monocular VO units drift (TUM fr1_desk: metres per unit falls 25 %
// within 50 frames), so a state in VO units goes stale against the poses;
// instead the pose translations are converted to metres with a smoothed
// metres-per-unit (DepthFilterFrame::log_scale), and the scale grid only
// corrects the map's shape:
//     z  the surface's log depth in the anchor's camera,
//     b  the network's bias there (log; the network measures log depth + b).
// The network's error is mostly *shared* between neighbouring views (15-20 %
// vs geometry, but only 2-4 % between views after scale-grid alignment), so a
// scalar filter that treats each map as independent would shrink sigma like
// 1/sqrt(N) while the real error stays. With b in the state, network maps
// only pin z + b down; z keeps the bias' uncertainty until a geometric
// measurement (flow triangulation, step 3) that sees z alone separates them.
// b is a stationary (Ornstein-Uhlenbeck) process: variance bias_sigma^2,
// correlation time bias_correlation_frames.
//
// Update (every frame): each anchor pixel is projected into the frame (lens k1
// as the VO). If no other anchor surface is in front of it there (z-buffer),
// the network map is sampled at that position (bilinear in log depth, nearest
// across depth edges) and fused with the measurement row H = [J, 1], J =
// d log z_frame / d log z_anchor. The state itself never moves: no resampling
// error accumulates between keyframes. The frame's pose error enters as
// measurement noise: translation sigma over depth, and the map's gradient
// times the rotation's pixel error. An innovation beyond gate_sigmas is not
// fused; reset_after such frames in a row reinitialise the pixel from the
// measurement (moving objects, wrong prior).
//
// Re-anchor (at VO keyframes): the state is splatted once into the new
// keyframe's grid with a z-buffer (nearest surface wins; candidates within
// splat_tolerance compete on sub-pixel distance), with process noise for
// the resampling (depth gradient times distance), then updated in place with
// that frame's map. Unreached pixels (disocclusions, new image area) take the
// measurement.
//
// Measurement: log(metres) - (L(pixel) - log_scale), L the latest keyframe's
// scale grid (its shape only).
//
// Host-only (no CUDA, no VO internals); a GPU port follows once the numbers
// are in. Network maps of one geometry only: a new geometry clears the state.
#include "slam_native/depth_estimator.hpp"
#include "slam_native/depth_confidence.hpp"
#include "slam_native/visual_odometry.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace slam_native {

struct DepthFilterConfig {
  double prior_sigma{0.2};             // network log-depth sigma without a per-pixel map
  double independent_fraction{0.2};    // of the network's sigma: the part independent between frames
  double bias_sigma{0.18};             // stationary sigma of b
  double bias_correlation_frames{100};
  double drift_sigma{0.005};           // log z per frame (unmodelled change)
  double gate_sigmas{3.0};
  int reset_after{3};
  // Resampling noise at re-anchoring: (|grad log z| - gradient_floor)+ per
  // output px, times the resampling distance. The floor keeps the state's own
  // pixel noise from reading as slope.
  double gradient_floor{0.03};
  // Floor on the pose's pointing error (output px). Misregistration from pose
  // error is what corrupts the state, at depth edges: gradient x this enters
  // the measurement noise and the re-anchoring process noise. The VO's own
  // rotation sigma is optimistic. Measured on fr1_desk with ground-truth depth
  // and VO poses (DEPTH_FUSION.md).
  double min_pointing_sigma_px{1.5};
  // Log sigma of the translations' scale (VO units -> metres through the
  // network-fitted scale: per-keyframe noise plus the network's own metric
  // error). The state is in metres, so a scale error turns every motion
  // toward a surface into a depth error, which a confident state would
  // otherwise lock in at each re-anchoring.
  double scale_sigma{0.1};
  // Floor the measurement noise at each frame's measured innovation spread
  // (DepthFilterStats::frame_noise).
  bool adaptive_noise{true};
  // Estimate b's mean. Network maps alone cannot separate z from b, so each
  // pixel's split of an innovation depends only on its history, and z comes
  // out speckled (fr1_desk: 5x the raw map's roughness). Off: z's mean is a
  // scalar filter's and b stays 0, while the covariance keeps b (honest
  // sigma). Turn on once a geometric measurement observes z alone.
  bool estimate_bias{false};
  float splat_tolerance{0.02F};        // log z: candidates this close to the nearest compete on distance
  float visibility_tolerance{0.05F};   // log z: an anchor pixel this far behind the nearest one is occluded
  float edge_log_ratio{0.1F};          // bilinear samples spanning more than this use the nearest pixel
  double scale_smoothing{0.05};        // per frame: exponential smoothing of log metres per VO unit
  float max_metres{0};                 // network samples at or beyond this are invalid (model clamp); 0: none
  float flicker_max_gradient{0.02F};   // flicker is measured where |grad log z| (per output px) is below this
  // Align each map to the state before fusing it: a smooth 6 x 4 grid of
  // robust log offsets (median innovation per cell over pixels seen at least
  // twice, bilinear between cell centres; cells with fewer than
  // align_min_samples take the global median). The network's scale and its
  // low-frequency shape wobble from frame to frame, and pixels entering the
  // view at the border would otherwise leave steps against the fused state.
  // High-pass: only each cell's deviation from its running mean (over
  // align_memory_frames) is removed, so persistent differences still reach
  // the state; removing them all lets the state drift away from the network.
  bool align_frames{false};  // off: with the noise floor and bias handling it costs accuracy (fr1_desk 0.057 vs 0.053)
  std::size_t align_min_samples{200};
  double align_memory_frames{20};
};

// The fused state rendered into some frame (DepthFilter::render), on the
// network's output grid.
struct DepthFilterView {
  std::uint64_t frame_index{};  // the anchor the state belongs to
  int width{}, height{};
  std::vector<float> metres;    // fused depth (network metres); 0 = none
  std::vector<float> sigma;     // log-depth sigma of z (bias uncertainty included); NaN = none
  std::vector<float> consistency;  // sigma of z + b: how well the network maps agree (log); NaN = none
  std::vector<float> bias;      // estimated network bias b (log)
  std::vector<std::uint16_t> observations;
};

// The VO's view of one frame.
struct DepthFilterFrame {
  const DepthMap* depth{};  // this frame's network map (required)
  Pose pose;                // world -> camera, VO units
  int segment{};            // a new segment clears the state
  bool keyframe{};          // re-anchor the state here
  CameraIntrinsics K{};     // pinhole intrinsics in source pixels (the VO's)
  double k1{};
  double log_scale{};                // log metres per VO unit (keyframe_log_scale): converts translations
  const ScaleGrid* grid{};           // optional: its grid; its shape (grid - log_scale) corrects the map
  const std::vector<float>* sigma{}; // optional: network log sigma per output pixel (network_sigma_map)
  double rotation_sigma_degrees{0};  // OdometryFrameResult; NaN treated as 0
  double translation_sigma_ratio{0};
};

struct DepthFilterStats {
  std::uint64_t frame_index{};
  bool reanchored{};        // the state was splatted into this frame (a keyframe)
  int valid{};              // state pixels after the update
  int carried{};            // re-anchoring: pixels that received a state
  int initialised{};        // empty pixels filled from the measurement
  int updated{};            // fused
  int occluded{};           // anchor pixels hidden behind another anchor surface in this frame
  int gated{};              // innovation beyond the gate, not fused
  int resets{};             // reinitialised after reset_after gated frames
  double scale_offset{};    // log offset removed from this map (align_frames)
  double frame_noise{};     // robust sigma (MAD) of this map's innovations (log)
  // Median |delta log z| of the same surface between consecutive updates
  // (flat pixels): the raw network vs the filter's output.
  double flicker_raw{};
  double flicker_fused{};
  int flicker_pixels{};
  double median_sigma{};      // sqrt(P_zz) over valid pixels
  double median_sum_sigma{};  // sigma of z + b (what network maps pin down)
};

class DepthFilter {
 public:
  explicit DepthFilter(DepthFilterConfig config = {}) : config_(config) {}

  // Fuses frame.depth into the anchored state; re-anchors first on keyframes
  // (and anchors on the first frame of a segment).
  DepthFilterStats process(const DepthFilterFrame& frame);
  // The segment's units changed: lengths in the new units = old / factor
  // (metres per unit multiplied by factor, OdometryFrameResult::
  // keyframe_scale_changes). Adjusts the anchor pose and the smoothed scale;
  // the state (metres) is unchanged.
  void rescale(double factor);
  // Replaces the anchor's pose (e.g. after a loop correction moved it).
  void set_anchor_pose(const Pose& pose) { anchor_pose_ = pose; }
  void clear();
  // Splats the anchored state into the frame with this pose (world -> camera,
  // VO units): nearest covering surface per pixel, cracks filled. For display
  // and for predicting depth in a frame other than the anchor.
  void render(const Pose& pose, DepthFilterView& view);

  [[nodiscard]] bool empty() const { return !has_state_; }
  [[nodiscard]] int width() const { return width_; }
  [[nodiscard]] int height() const { return height_; }
  // The anchor keyframe's frame: the state's pixels are its pixels.
  [[nodiscard]] std::uint64_t frame_index() const { return frame_index_; }
  [[nodiscard]] const Pose& anchor_pose() const { return anchor_pose_; }
  // Per output pixel of the anchor; NaN where empty.
  [[nodiscard]] float log_depth(std::size_t i) const;  // z, log metres
  [[nodiscard]] float sigma(std::size_t i) const;      // sqrt(P_zz)
  [[nodiscard]] float bias(std::size_t i) const;
  [[nodiscard]] int observations(std::size_t i) const { return state_[i].count; }
  // The fused map of the anchor as a network-like DepthMap (same geometry,
  // frame_index()): exp(z + L(pixel) - log_scale), the grid's shape put back
  // so that KeyframeDepthStore::add with the same scale and grid recovers it.
  // 0 where empty.
  [[nodiscard]] DepthMap to_depth_map(double log_scale, const ScaleGrid* grid) const;

  [[nodiscard]] const DepthFilterConfig& config() const { return config_; }

 private:
  struct Pixel {
    float z{std::numeric_limits<float>::quiet_NaN()};
    float b{};
    float pzz{}, pzb{}, pbb{};
    float raw{std::numeric_limits<float>::quiet_NaN()};  // last raw measurement in anchor log depth (flicker)
    std::uint16_t count{};
    std::uint8_t outliers{};
    [[nodiscard]] bool valid() const { return z == z; }
  };
  // Per-frame measurement inputs on the output grid.
  struct Measurement {
    std::vector<float> log_depth;  // log(metres) - L; NaN where invalid
    std::vector<float> gradient;   // |grad| of log_depth per output px
    std::vector<float> sigma;      // network log sigma
  };
  void set_geometry(const DepthMap& depth, const CameraIntrinsics& K, double k1);
  bool same_geometry(const DepthMap& depth, const CameraIntrinsics& K, double k1) const;
  void prepare(const DepthFilterFrame& frame);
  // Relative motion anchor -> frame: x_frame = R x_anchor + t.
  void relative(const Pose& pose, double R[9], double t[3]) const;
  // Projects every anchor pixel into the frame (target_*), z-buffered (zbuffer_).
  void project(const double R[9], const double t[3], bool four_neighbours);
  // After project(..., false): best_source_ per output pixel (covering
  // z-buffer, cracks filled by the closest candidate).
  void apply_gain(Pixel& p, double kz, double kb, double innovation) const;
  void select_sources();
  [[nodiscard]] float resampled(std::size_t source, std::size_t target) const;
  void reanchor(const DepthFilterFrame& frame, DepthFilterStats& stats);
  void update_in_place(DepthFilterStats& stats);
  void update_anchored(const DepthFilterFrame& frame, DepthFilterStats& stats);
  void decay();  // one frame of bias (OU) and drift process noise
  void finish(const DepthFilterFrame& frame, DepthFilterStats& stats);

  DepthFilterConfig config_;
  // Geometry (DepthMap transform fields; metres empty), lens and per-pixel
  // pinhole rays (x/z, y/z; NaN in the letterbox) and source pixels.
  DepthMap geometry_;
  CameraIntrinsics K_{};
  double k1_{};
  int width_{}, height_{};
  std::vector<float> ray_x_, ray_y_, source_x_, source_y_;
  std::vector<Pixel> state_, next_;
  Measurement measurement_;
  std::vector<float> zbuffer_, best_distance_;
  std::vector<std::int32_t> best_source_;
  std::vector<float> target_u_, target_v_, target_z_;
  std::vector<float> half_;             // splat footprint half-width (output px)
  std::vector<float> sampled_;          // anchored update: the measurement at each anchor pixel
  std::vector<std::int32_t> nearest_;   // and its nearest output pixel
  // Per-frame alignment of the map to the state (align_frames).
  static constexpr std::size_t kAlignX = 6, kAlignY = 4;
  void add_offset(std::size_t pixel, double value);  // pixel: output pixel of the frame
  void fit_alignment(DepthFilterStats& stats);
  [[nodiscard]] float alignment_at(std::size_t pixel) const;
  std::vector<double> offsets_, deviations_;
  double frame_noise_{};
  [[nodiscard]] double noise_floor2() const { return config_.adaptive_noise ? frame_noise_ * frame_noise_ : 0.0; }
  std::array<std::vector<double>, kAlignX * kAlignY> cell_offsets_;
  std::array<float, kAlignX * kAlignY> alignment_{};
  std::array<double, kAlignX * kAlignY> alignment_mean_{};
  bool alignment_started_{};
  bool has_state_{};
  std::uint64_t frame_index_{};
  int segment_{};
  Pose anchor_pose_;
  double median_z_{};  // log metres
  double log_scale_{std::numeric_limits<double>::quiet_NaN()};  // smoothed log metres per VO unit
  std::vector<double> flicker_raw_, flicker_fused_;
};

// Network log-depth sigma per output pixel from the confidence model, with
// the cues it was trained on (DepthCues). landmarks: source pixels of the
// frame's landmark observations (the nearest_landmark cue); empty: 0.5.
// Letterbox pixels get NaN.
std::vector<float> network_sigma_map(const DepthMap& depth, const DepthConfidenceModel& model,
                                     const std::vector<Keypoint>& landmarks);

}  // namespace slam_native
