#pragma once
// Temporal fusion of network depth (DEPTH_FUSION.md, step 1). A per-pixel
// Kalman filter on the network's output grid, pushed from frame to frame with
// the VO pose; each new network map is just an observation.
//
// State per pixel, in log depth (VO units, so the warp uses VO poses as is):
//     z  the surface's log depth in this frame's camera,
//     b  the network's bias there (log; the network measures z + b).
// The network's error is mostly *shared* between neighbouring views (15-20 %
// vs geometry, but only 2-4 % between views after scale-grid alignment), so a
// scalar filter that treats each map as independent would shrink sigma like
// 1/sqrt(N) while the real error stays. With b in the state, network maps
// only pin z + b down; z keeps the bias' uncertainty until a geometric
// measurement (flow triangulation, step 3) that sees z alone separates them.
// b is a stationary (Ornstein-Uhlenbeck) process: variance bias_sigma^2,
// correlation time bias_correlation_frames.
//
// Predict: every state pixel is back-projected (lens k1 as the VO), moved by
// the relative pose and splatted with a z-buffer into the next frame's grid;
// unreached pixels (disocclusions, new image area) are empty. Process noise:
// pose uncertainty (OdometryFrameResult's rotation / translation sigmas), a
// per-frame drift, and the depth gradient times the resampling distance
// (large at depth edges, where a small position error is a large depth error).
//
// Update: measurement log(metres) - L(pixel), L = log metres per VO unit from
// the latest keyframe's scale grid. Empty pixels take the measurement. An
// innovation beyond gate_sigmas is not fused; reset_after such frames in a row
// reinitialise the pixel from the measurement (moving objects, wrong prior).
//
// Host-only (no CUDA, no VO internals); a GPU port follows once the numbers
// are in. Network maps of one geometry only: a new geometry clears the state.
#include "slam_native/depth_estimator.hpp"
#include "slam_native/depth_confidence.hpp"
#include "slam_native/visual_odometry.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace slam_native {

struct DepthFilterConfig {
  double prior_sigma{0.2};             // network log-depth sigma without a per-pixel map
  double independent_fraction{0.2};    // of the network's sigma: the part independent between frames
  double bias_sigma{0.18};             // stationary sigma of b
  double bias_correlation_frames{100};
  double drift_sigma{0.005};           // log z per frame (pose drift, unmodelled motion)
  double gate_sigmas{3.0};
  int reset_after{3};
  // Resampling noise: (|grad log z| - gradient_floor)+ per output px, times the
  // resampling distance. The floor keeps the state's own pixel noise from
  // reading as slope (which would feed back as process noise).
  double gradient_floor{0.03};
  float splat_tolerance{0.02F};        // log z: candidates this close to the nearest compete on distance
  float max_metres{0};                 // network samples at or beyond this are invalid (model clamp); 0: none
  float flicker_max_gradient{0.02F};   // flicker is measured where |grad log z| (per output px) is below this
};

// The VO's view of one frame.
struct DepthFilterFrame {
  const DepthMap* depth{};  // this frame's network map (required)
  Pose pose;                // world -> camera, VO units
  int segment{};            // a new segment clears the state
  CameraIntrinsics K{};     // pinhole intrinsics in source pixels (the VO's)
  double k1{};
  double log_scale{};                // log metres per VO unit (keyframe_log_scale)
  const ScaleGrid* grid{};           // optional: its grid (overrides log_scale per pixel)
  const std::vector<float>* sigma{}; // optional: network log sigma per output pixel (network_sigma_map)
  double rotation_sigma_degrees{0};  // OdometryFrameResult; NaN treated as 0
  double translation_sigma_ratio{0};
};

struct DepthFilterStats {
  std::uint64_t frame_index{};
  bool predicted{};         // the state was carried from the previous frame
  int valid{};              // state pixels after the update
  int carried{};            // pixels that received a predicted state
  int initialised{};        // empty pixels filled from the measurement
  int updated{};            // fused
  int gated{};              // innovation beyond the gate, not fused
  int resets{};             // reinitialised after reset_after gated frames
  // Median |delta log z| of the same surface between consecutive frames
  // (flat pixels seen in both): the raw network vs the filter's output.
  double flicker_raw{};
  double flicker_fused{};
  int flicker_pixels{};
  double median_sigma{};    // sqrt(P_zz) over valid pixels
  double median_sum_sigma{};  // sigma of z + b (what network maps pin down)
};

class DepthFilter {
 public:
  explicit DepthFilter(DepthFilterConfig config = {}) : config_(config) {}

  // Predicts from the previous frame (if any, same segment and geometry),
  // then fuses frame.depth.
  DepthFilterStats process(const DepthFilterFrame& frame);
  // The segment's units changed: lengths in the new units = old / factor
  // (metres per unit multiplied by factor, OdometryFrameResult::
  // keyframe_scale_changes). Shifts z; also adjusts the stored previous pose.
  void rescale(double factor);
  // Replaces the stored previous pose (e.g. after a loop correction moved it).
  void set_previous_pose(const Pose& pose) { previous_pose_ = pose; }
  void clear();

  [[nodiscard]] bool empty() const { return !has_state_; }
  [[nodiscard]] int width() const { return width_; }
  [[nodiscard]] int height() const { return height_; }
  [[nodiscard]] std::uint64_t frame_index() const { return frame_index_; }
  // Per output pixel; NaN where empty.
  [[nodiscard]] float log_depth(std::size_t i) const;  // z, VO units
  [[nodiscard]] float sigma(std::size_t i) const;      // sqrt(P_zz)
  [[nodiscard]] float bias(std::size_t i) const;
  [[nodiscard]] int observations(std::size_t i) const { return state_[i].count; }
  // The fused map as a network-like DepthMap (same geometry and frame), in
  // metres through log_scale / grid: exp(z + L(pixel)). 0 where empty. Fed
  // to KeyframeDepthStore::add with the same scale, its points come back as
  // exp(z) in VO units.
  [[nodiscard]] DepthMap to_depth_map(double log_scale, const ScaleGrid* grid) const;

  [[nodiscard]] const DepthFilterConfig& config() const { return config_; }

 private:
  struct Pixel {
    float z{std::numeric_limits<float>::quiet_NaN()};
    float b{};
    float pzz{}, pzb{}, pbb{};
    float raw{std::numeric_limits<float>::quiet_NaN()};  // last raw measurement, carried for flicker
    std::uint16_t count{};
    std::uint8_t outliers{};
    [[nodiscard]] bool valid() const { return z == z; }
  };
  void set_geometry(const DepthMap& depth, const CameraIntrinsics& K, double k1);
  bool same_geometry(const DepthMap& depth, const CameraIntrinsics& K, double k1) const;
  void predict(const DepthFilterFrame& frame, DepthFilterStats& stats);

  DepthFilterConfig config_;
  // Geometry (DepthMap transform fields; metres empty), lens and per-pixel
  // pinhole rays (x/z, y/z; NaN in the letterbox) and source pixels.
  DepthMap geometry_;
  CameraIntrinsics K_{};
  double k1_{};
  int width_{}, height_{};
  std::vector<float> ray_x_, ray_y_, source_x_, source_y_;
  std::vector<Pixel> state_, next_;
  std::vector<float> zbuffer_, best_distance_;
  std::vector<std::int32_t> best_source_;
  std::vector<float> target_u_, target_v_, target_z_;
  bool has_state_{};
  std::uint64_t frame_index_{};
  int segment_{};
  Pose previous_pose_;
  double previous_median_z_{};
};

// Network log-depth sigma per output pixel from the confidence model, with
// the cues it was trained on (DepthCues). landmarks: source pixels of the
// frame's landmark observations (the nearest_landmark cue); empty: 0.5.
// Letterbox pixels get NaN.
std::vector<float> network_sigma_map(const DepthMap& depth, const DepthConfidenceModel& model,
                                     const std::vector<Keypoint>& landmarks);

}  // namespace slam_native
