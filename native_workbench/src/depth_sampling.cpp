#include "slam_native/depth_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace slam_native {

DepthSample sample_depth(const DepthMap& depth, float x, float y, const DepthSamplingConfig& config) {
  if (depth.empty()) return {};
  float u = 0, v = 0;
  depth.to_output(x, y, u, v);
  if (!(u >= -0.5F && v >= -0.5F && u <= depth.width - 0.5F && v <= depth.height - 0.5F)) return {};
  const int cu = std::clamp(static_cast<int>(std::lround(u)), 0, depth.width - 1);
  const int cv = std::clamp(static_cast<int>(std::lround(v)), 0, depth.height - 1);
  float lowest = INFINITY, highest = 0;
  for (int dv = -1; dv <= 1; ++dv)
    for (int du = -1; du <= 1; ++du) {
      const int pu = std::clamp(cu + du, 0, depth.width - 1), pv = std::clamp(cv + dv, 0, depth.height - 1);
      const float d = depth.metres[static_cast<std::size_t>(pv) * depth.width + pu];
      if (!(d > 0)) return {};  // letterbox padding next to the sample
      lowest = std::min(lowest, d);
      highest = std::max(highest, d);
    }
  if (config.max_depth_m > 0 && highest >= 0.999F * config.max_depth_m) return {};  // at the model's clamp
  if (highest > (1 + config.edge_ratio) * lowest) {
    if (config.drop_edges) return {};
    return {lowest, config.relative_sigma * config.edge_sigma_factor * lowest, true};
  }
  const float metres = depth.at_source(x, y);
  if (!(metres > 0)) return {};
  return {metres, config.relative_sigma * metres, false};
}

int attach_depth(const DepthMap& depth, TrackedFrame& frame, const DepthSamplingConfig& config, int* edges) {
  if (depth.frame_index != frame.frame_index)
    throw std::invalid_argument("depth map of frame " + std::to_string(depth.frame_index) + " attached to frame " +
                                std::to_string(frame.frame_index));
  int valid = 0, edge_count = 0;
  for (auto& o : frame.observations) {
    const DepthSample sample = sample_depth(depth, o.x, o.y, config);
    o.depth_m = sample.metres;
    o.depth_sigma_m = sample.sigma;
    valid += sample.valid();
    edge_count += sample.valid() && sample.edge;
  }
  if (edges) *edges = edge_count;
  return valid;
}

}  // namespace slam_native
