#pragma once

#include "slam_native/types.hpp"

#include <memory>
#include <vector>

namespace slam_native {

struct FlowVector {
  float dx{};
  float dy{};
};

// Host copy of a flow field, for display, tests and CPU reference checks.
// Cell (c, r) describes the 4x4 block whose centre is (4c + 1.5, 4r + 1.5) in
// source pixels. `vectors` is forward flow (previous -> current frame);
// `backward` (current -> previous) is optional and empty when not computed.
struct FlowField {
  int width{};
  int height{};
  int grid_size{4};
  std::vector<FlowVector> vectors;
  std::vector<FlowVector> backward;

  // Bilinear interpolation between cell centres; NaN when malformed.
  [[nodiscard]] FlowVector sample(float x, float y) const;
  [[nodiscard]] FlowVector sample_backward(float x, float y) const;
};

// Device-resident flow fields produced by OpticalFlow::compute. Pointers stay
// valid until the next compute/remember call. `ready_event` (cudaEvent_t) is
// recorded after both fields are written; consumers must wait on it.
struct DeviceFlowField {
  const FlowVector* forward{};   // previous -> current, pixels, width*height
  const FlowVector* backward{};  // current -> previous, or null when disabled
  int width{};
  int height{};
  int grid_size{4};
  void* ready_event{};
};

struct OpticalFlowConfig {
  // Also compute current -> previous flow for the tracker's forward-backward
  // check. Off by default: it doubles NVOF time (~3 -> ~6 ms at 1080p) and on
  // Osaka it shortened tracks without improving match similarity.
  bool backward{false};
  // NVIDIA preset: 0 slow (best quality), 1 medium, 2 fast.
  int quality{1};
};

// Uses the dedicated NVIDIA optical-flow engine on two decoded CUDA frames.
// Nothing is copied to the host unless download() is called.
class OpticalFlow {
 public:
  explicit OpticalFlow(OpticalFlowConfig config = {});
  ~OpticalFlow();
  OpticalFlow(const OpticalFlow&) = delete;
  OpticalFlow& operator=(const OpticalFlow&) = delete;

  void remember(const GpuFrame& frame);
  [[nodiscard]] bool has_reference() const;
  // Enqueues flow between the remembered frame and `frame` (which becomes the
  // reference for the next call). Returns without host synchronization.
  [[nodiscard]] DeviceFlowField compute(const GpuFrame& frame);
  // Host copy of the most recent fields (display, tests). Synchronizes.
  [[nodiscard]] FlowField download() const;
  // GPU time of the last compute() (both directions), once it has finished.
  [[nodiscard]] double gpu_ms() const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace slam_native
