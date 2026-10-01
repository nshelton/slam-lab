#pragma once
// Live depth image (LIVE_DEPTH.md): an inverse depth and its variance for
// every source pixel of the current frame, on the GPU. It runs beside the
// visual odometry and only reads from it: the frame's pose, and the depths of
// the landmarks the frame's tracks observe ("anchors").
//
// Built so far: the per-frame measurement, an anchors-only estimate that
// serves as the floor to beat, and the filter that fuses the measurements
// over time.
#include "slam_native/optical_flow.hpp"
#include "slam_native/types.hpp"
#include "slam_native/visual_odometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace slam_native {

// The pinhole camera of the source pixels plus the VO's one-parameter lens
// distortion (undistort_point).
struct LiveDepthCamera {
  int width{};
  int height{};
  float fx{}, fy{}, cx{}, cy{};
  float k1{};
};

// A landmark seen in the frame: where (source pixels) and how deep.
struct DepthAnchor {
  float x{};
  float y{};
  float inverse_depth{};  // 1 / z in the camera frame, VO units
};

enum class LiveDepthMode {
  // Every pixel takes the inverse depth of the anchors around it (inverse
  // distance weighting). No image evidence: what the keypoints alone give.
  anchors,
  // Every pixel is matched in an earlier frame (reference_gap back): the two
  // poses give its epipolar line, the optical flow, chained through the frames
  // in between, where on that line to look, and a short search for the best
  // patch match along the line gives the disparity.
  measurement,
  // The measurements of all frames so far in one estimate: carried into each
  // new frame with the camera motion, rescaled to the anchors under it, and
  // fused with the frame's measurement.
  filter,
  // The filter's image combined with the anchors' interpolation, which also
  // fills the pixels the filter has nothing for. The anchors are not folded
  // into the filter's state, so they are not counted again every frame.
  fused,
};

struct LiveDepthConfig {
  LiveDepthMode mode{LiveDepthMode::measurement};
  // Frames between the current frame and the one it is matched against (at
  // most 31). More gives more parallax, less overlap and a longer flow chain.
  // A frame without that much history uses what it has.
  int reference_gap{10};
  // The patch is a strip along the epipolar line.
  int patch_half_length{2};  // samples along the line: 2n + 1, one pixel apart
  int patch_half_width{1};   // rows across it: 2n + 1
  float search_radius_px{6.0F};  // around the flow's guess, along the line
  float search_step_px{0.5F};
  // Rejection.
  float max_epipolar_distance_px{5.0F};  // the flow's guess off the line: moving object or bad flow
  float min_gradient{3.0F};      // RMS luma step per pixel along the line, in the patch
  float max_residual{12.0F};     // RMS luma difference of the best match
  // Disparity noise (pixels): sigma^2 = floor^2 + 2 image_noise^2 / gradient^2.
  float disparity_floor_px{0.2F};
  float image_noise{3.0F};       // luma levels
  // The anchors' interpolation: relative sigma = base + per_px * distance to the nearest anchor.
  float anchor_sigma{0.03F};
  float anchor_sigma_per_px{0.0025F};
  // Filter. Sigmas are relative (of the inverse depth).
  float process_sigma{0.003F};             // added per frame carried
  float propagation_tolerance_px{1.5F};    // a carried point must land this close to its pixel
  float measurement_sigma{0.06F};          // added to every measurement: pose and model error
  float min_sigma{0.01F};                  // the fused estimate is never trusted beyond this
  float outlier_sigmas{2.5F};              // estimates further apart than this are not averaged
  bool scale_to_anchors{true};
};

// Why pixels got no measurement (LiveDepthMode::measurement), last frame.
struct LiveDepthStats {
  std::size_t pixels{};
  std::size_t estimated{};      // pixels with an estimate in the image
  std::size_t measured{};       // pixels this frame's measurement gave
  float scale{1.0F};            // filter: landmarks / state under them, before it was applied
  std::size_t outside{};        // ray, flow or patch leaves the image
  std::size_t off_line{};       // flow guess too far from the epipolar line
  std::size_t weak_gradient{};
  std::size_t at_search_edge{};  // best match at the end of the search range
  std::size_t poor_match{};
  std::size_t behind{};          // matched, but at a negative depth
  double ms{};                   // GPU work of the last process(), wall time
};

// Host copy of the image. A pixel without an estimate has inverse_depth 0 and
// an infinite variance.
struct LiveDepthImage {
  std::uint64_t frame_index{};
  int width{};
  int height{};
  std::vector<float> inverse_depth;
  std::vector<float> variance;
};

// The image on the device (null when the last frame gave none), valid until
// the next process(); the work that wrote it has finished.
struct LiveDepthDevice {
  const float* inverse_depth{};
  const float* variance{};
  int width{};
  int height{};
};

class LiveDepth {
 public:
  explicit LiveDepth(const LiveDepthConfig& config = {});
  ~LiveDepth();
  LiveDepth(const LiveDepth&) = delete;
  LiveDepth& operator=(const LiveDepth&) = delete;

  // Forgets the earlier frames and the filter's state. Call it on a seek or a
  // lens change, and when the odometry moved the segment's poses (loop
  // correction, segments merged: OdometryFrameResult::loop_closed,
  // segments_merged): the earlier frames' poses and the state's scale no
  // longer fit the current ones.
  void reset();

  // One frame, after the odometry processed it. `flow` is the previous ->
  // current field of this frame (null on the first frame, which starts the
  // history over); `pose` is null when the odometry has none. A frame is only
  // matched against one with a pose in the same segment: other segments have
  // other frames and scales.
  void process(const GpuFrame& frame, const DeviceFlowField* flow, const LiveDepthCamera& camera,
               const Pose* pose, int segment, const std::vector<DepthAnchor>& anchors);

  // True when the last process() produced an image.
  [[nodiscard]] bool valid() const;
  // Frames back to the reference of the last measurement (0: none).
  [[nodiscard]] int reference_gap() const;
  [[nodiscard]] LiveDepthImage download() const;
  [[nodiscard]] LiveDepthDevice device() const;
  // One pixel of the image; false when it has no estimate there.
  bool at(int x, int y, float* inverse_depth, float* variance) const;
  [[nodiscard]] const LiveDepthConfig& config() const;
  [[nodiscard]] const LiveDepthStats& stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The anchors of a frame: the observed tracks that hold a landmark of the
// odometry's current segment in front of the camera.
std::vector<DepthAnchor> depth_anchors(const TrackedFrame& frame, const VisualOdometry& odometry,
                                       const OdometryFrameResult& result);
LiveDepthCamera live_depth_camera(const VisualOdometryConfig& config, int width, int height);

}  // namespace slam_native
