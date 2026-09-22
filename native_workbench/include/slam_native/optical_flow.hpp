#pragma once

#include "slam_native/types.hpp"

#include <memory>
#include <vector>

namespace slam_native {

struct FlowVector {
  float dx{};
  float dy{};
};

struct FlowField {
  int width{};
  int height{};
  int grid_size{4};
  std::vector<FlowVector> vectors;

  [[nodiscard]] FlowVector sample(float x, float y) const;
};

// Uses the dedicated NVIDIA optical-flow engine on two decoded CUDA frames.
class OpticalFlow {
 public:
  OpticalFlow();
  ~OpticalFlow();
  OpticalFlow(const OpticalFlow&) = delete;
  OpticalFlow& operator=(const OpticalFlow&) = delete;

  void remember(const GpuFrame& frame);
  [[nodiscard]] bool has_reference() const;
  [[nodiscard]] FlowField compute(const GpuFrame& frame);

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace slam_native
