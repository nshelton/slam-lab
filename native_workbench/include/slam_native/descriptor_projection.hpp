#pragma once

#include "slam_native/types.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace slam_native {

struct ProjectedDescriptor {
  float x{};
  float y{};
};

// Fits a fixed two-component PCA basis to the first descriptor sample. The
// basis stays fixed so points remain comparable as playback advances.
class DescriptorProjection {
 public:
  void observe(const FrameFeatures& frame);
  [[nodiscard]] std::vector<ProjectedDescriptor> project(const FrameFeatures& frame) const;
  [[nodiscard]] bool ready() const { return ready_; }
  [[nodiscard]] std::size_t sample_count() const {
    return ready_ ? fitted_sample_count_ : samples_.size() / 256;
  }

 private:
  void fit();

  std::vector<float> samples_;
  std::array<float, 256> mean_{};
  std::array<float, 256> first_{};
  std::array<float, 256> second_{};
  float first_extent_{1.0F};
  float second_extent_{1.0F};
  std::size_t observed_frames_{};
  std::size_t fitted_sample_count_{};
  bool ready_{};
};

}  // namespace slam_native
