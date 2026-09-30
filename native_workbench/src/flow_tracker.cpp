#include "slam_native/flow_tracker.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace slam_native {
namespace {
// Must match the CUDA sampler in flow_association.cu exactly.
FlowVector bilinear(const std::vector<FlowVector>& field, int width, int height, int grid,
                    float x, float y) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  if (width <= 0 || height <= 0 || grid <= 0 || !std::isfinite(x) || !std::isfinite(y) ||
      field.size() != static_cast<std::size_t>(width) * height) return {nan, nan};
  const float half = 0.5F * static_cast<float>(grid - 1);
  const float gx = std::clamp((x - half) / grid, 0.0F, float(width - 1));
  const float gy = std::clamp((y - half) / grid, 0.0F, float(height - 1));
  const int x0 = static_cast<int>(gx), y0 = static_cast<int>(gy);
  const int x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
  const float wx = gx - x0, wy = gy - y0;
  const auto at = [&](int c, int r) { return field[static_cast<std::size_t>(r) * width + c]; };
  const auto a = at(x0, y0), b = at(x1, y0), c = at(x0, y1), d = at(x1, y1);
  return {(a.dx * (1 - wx) + b.dx * wx) * (1 - wy) + (c.dx * (1 - wx) + d.dx * wx) * wy,
          (a.dy * (1 - wx) + b.dy * wx) * (1 - wy) + (c.dy * (1 - wx) + d.dy * wx) * wy};
}
}  // namespace

FlowVector FlowField::sample(float x, float y) const {
  return bilinear(vectors, width, height, grid_size, x, y);
}
FlowVector FlowField::sample_backward(float x, float y) const {
  return bilinear(backward, width, height, grid_size, x, y);
}

void FlowTrackerConfig::validate() const {
  if (!std::isfinite(association_radius) || association_radius <= 0 || association_radius > 1000 ||
      max_tracks == 0 || max_tracks > 100000 || !std::isfinite(min_descriptor_similarity) ||
      !std::isfinite(descriptor_weight) || descriptor_weight < 0 || max_coast_frames < 0 ||
      max_coast_frames > 1000 || !std::isfinite(forward_backward_threshold) ||
      assignment_rounds < 1 || assignment_rounds > 64) {
    throw std::invalid_argument("Invalid flow tracker settings");
  }
}
}  // namespace slam_native
