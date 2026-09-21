#pragma once

#include "slam_native/types.hpp"

#include <filesystem>
#include <memory>

namespace slam_native {

struct SuperPointConfig {
  int input_width{1024};
  int input_height{576};
  int max_keypoints{2048};
  float detection_threshold{0.0005F};
};

class SuperPoint {
 public:
  virtual ~SuperPoint() = default;
  virtual FrameFeatures infer(const GpuFrame& frame) = 0;
};

std::unique_ptr<SuperPoint> make_tensorrt_superpoint(
    const std::filesystem::path& engine_path, const SuperPointConfig& config);

}  // namespace slam_native

