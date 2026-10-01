#pragma once
// Monocular metric depth from a TensorRT engine (Depth Anything V2, Metric3D
// v2, ...). A model is an engine plus a DepthModelSpec that says how to feed it
// and how to read its output; adding a model means exporting an engine and
// adding a preset (depth_model_presets()).
//
// The network sees the *upright* frame (the display rotation applied), since
// depth models are trained on upright images; DepthMap maps results back to
// native source pixels, the coordinates every other component uses.
//
// Inference is asynchronous on its own CUDA stream: submit() returns once the
// frame has been copied into the network input, poll() picks up the result
// on a later UI frame.
#include "slam_native/types.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace slam_native {

struct DepthModelSpec {
  std::string name;         // UI label and --depth-model value
  std::string engine_stem;  // <models>/<stem>-<W>x<H>.engine, W x H = upright input size
  int input_width{};        // landscape input; a portrait display uses <H>x<W>
  int input_height{};
  std::string input_tensor{"image"};   // 1x3xHxW, float32 or float16
  std::string output_tensor{"depth"};  // [1x[1x]]HxW (any size), float32 or float16
  // Network input = (RGB in [0, 255] - mean) / std, per channel.
  std::array<float, 3> mean{0, 0, 0};
  std::array<float, 3> std{1, 1, 1};
  std::array<float, 3> pad{0, 0, 0};  // letterbox fill, RGB in [0, 255]
  // > 0: the output is depth for a camera of this focal length (Metric3D's
  // canonical camera, 1000 px) and is scaled by focal_in_input_px / this.
  float canonical_focal_px{};
  float max_depth_m{80};  // outputs are clamped to [0, max]
};

const std::vector<DepthModelSpec>& depth_model_presets();
const DepthModelSpec* find_depth_model(const std::string& name);
// The engine for this spec when the display is portrait (rotation 90/270 of a
// landscape video, or an unrotated portrait one) or landscape.
std::filesystem::path depth_engine_path(const std::filesystem::path& models_dir,
                                        const DepthModelSpec& spec, bool portrait);

// Depth in metres on the network's output grid (upright, letterboxed);
// 0 in the letterbox padding.
struct DepthMap {
  std::uint64_t frame_index{};
  int width{};
  int height{};
  std::vector<float> metres;
  // Source pixel (native frame, pixel centres at integers) -> output pixel:
  // rotate clockwise by `rotation` (continuous coordinates, x + 0.5), then
  // out = upright * scale + offset - 0.5 per axis.
  int source_width{};
  int source_height{};
  int rotation{};
  float scale_x{};
  float scale_y{};
  float offset_x{};
  float offset_y{};
  double gpu_ms{};  // preprocessing + inference + postprocessing

  [[nodiscard]] bool empty() const { return metres.empty(); }
  // Output pixel of a source pixel.
  void to_output(float x, float y, float& u, float& v) const;
  // Bilinear depth at a source pixel; NaN outside the image or in padding.
  [[nodiscard]] float at_source(float x, float y) const;
};

class DepthEstimator {
 public:
  // rotation: clockwise display rotation (0, 90, 180, 270).
  DepthEstimator(DepthModelSpec spec, const std::filesystem::path& engine_path, int rotation);
  ~DepthEstimator();
  DepthEstimator(const DepthEstimator&) = delete;
  DepthEstimator& operator=(const DepthEstimator&) = delete;

  [[nodiscard]] const DepthModelSpec& spec() const;
  [[nodiscard]] int rotation() const;
  [[nodiscard]] int input_width() const;   // upright, as the engine was built
  [[nodiscard]] int input_height() const;
  // True while the last submission is still running.
  [[nodiscard]] bool busy() const;
  // Queues preprocessing and inference for `frame`, returning once the frame
  // has been read (it may be released afterwards). focal_px: the camera's
  // focal length in source pixels (used by canonical-camera models). Ignored
  // (returns false) while busy().
  bool submit(const GpuFrame& frame, float focal_px);
  // Moves a finished result into `out`; false if none is ready.
  bool poll(DepthMap& out);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace slam_native
