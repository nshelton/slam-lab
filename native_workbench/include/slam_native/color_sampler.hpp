#pragma once
// Samples the RGB colour of a decoded GPU frame at a set of pixel positions
// (3x3 mean, BT.709 limited range, like the display conversion). One small
// upload, one kernel and a 3-bytes-per-point download.
#include "slam_native/types.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace slam_native {

class ColorSampler {
 public:
  ColorSampler();
  ~ColorSampler();
  ColorSampler(const ColorSampler&) = delete;
  ColorSampler& operator=(const ColorSampler&) = delete;

  std::vector<std::array<std::uint8_t, 3>> sample(const GpuFrame& frame, const std::vector<Keypoint>& points);

 private:
  struct Device;
  std::unique_ptr<Device> device_;
};

}  // namespace slam_native
