#include "slam_native/keyframe_depth.hpp"

#include <algorithm>
#include <cmath>

namespace slam_native {
namespace {
// Inverse of DepthMap::to_output: output pixel -> native source pixel.
void to_source(const DepthMap& map, float u, float v, float& x, float& y) {
  const float upright_x = (u + 0.5F - map.offset_x) / map.scale_x;
  const float upright_y = (v + 0.5F - map.offset_y) / map.scale_y;
  const auto w = static_cast<float>(map.source_width), h = static_cast<float>(map.source_height);
  float cx = upright_x, cy = upright_y;
  if (map.rotation == 90) { cx = upright_y; cy = h - upright_x; }
  else if (map.rotation == 180) { cx = w - upright_x; cy = h - upright_y; }
  else if (map.rotation == 270) { cx = w - upright_y; cy = upright_x; }
  x = cx - 0.5F;
  y = cy - 0.5F;
}
}  // namespace

bool KeyframeDepthStore::same_geometry(const DepthMap& map) const {
  return !grid_.empty() && map.width == geometry_.width && map.height == geometry_.height &&
         map.source_width == geometry_.source_width && map.source_height == geometry_.source_height &&
         map.rotation == geometry_.rotation && map.scale_x == geometry_.scale_x && map.scale_y == geometry_.scale_y &&
         map.offset_x == geometry_.offset_x && map.offset_y == geometry_.offset_y;
}

const std::vector<Keypoint>& KeyframeDepthStore::grid(const DepthMap& like) {
  if (same_geometry(like)) return grid_;
  geometry_ = like;
  geometry_.metres.clear();
  grid_.clear();
  cells_.clear();
  const int stride = std::max(1, config_.stride);
  for (int v = stride / 2; v < like.height; v += stride)
    for (int u = stride / 2; u < like.width; u += stride) {
      float x = 0, y = 0;
      to_source(like, static_cast<float>(u), static_cast<float>(v), x, y);
      if (x < 0 || y < 0 || x > like.source_width - 1 || y > like.source_height - 1) continue;  // letterbox
      grid_.push_back({x, y, 0});
      cells_.push_back({u, v});
    }
  return grid_;
}

std::size_t KeyframeDepthStore::add(std::uint64_t frame_index, int segment, double metres_per_unit,
                                    const DepthMap& depth, const std::vector<std::array<std::uint8_t, 3>>& colors,
                                    const CameraIntrinsics& K, double k1) {
  if (depth.empty() || K.fx <= 0 || K.fy <= 0) return 0;
  grid(depth);
  std::vector<float> valid;
  for (std::size_t i = 0; i < depth.metres.size(); i += 13)
    if (depth.metres[i] > 0) valid.push_back(depth.metres[i]);
  if (valid.empty()) return 0;
  std::nth_element(valid.begin(), valid.begin() + static_cast<std::ptrdiff_t>(valid.size() / 2), valid.end());
  const float limit = std::min(config_.max_depth_m, config_.max_median_ratio * valid[valid.size() / 2]);
  const bool colored = colors.size() == grid_.size();
  const auto at = [&](int u, int v) {
    return depth.metres[static_cast<std::size_t>(std::clamp(v, 0, depth.height - 1)) * depth.width +
                        static_cast<std::size_t>(std::clamp(u, 0, depth.width - 1))];
  };
  KeyframeCloud cloud;
  cloud.frame_index = frame_index;
  cloud.segment = segment;
  cloud.metres_per_unit = metres_per_unit > 0 && std::isfinite(metres_per_unit) ? metres_per_unit : 0;
  cloud.points.reserve(grid_.size());
  for (std::size_t i = 0; i < grid_.size(); ++i) {
    const auto [u, v] = cells_[i];
    const float d = at(u, v);
    if (!(d > 0) || d > limit) continue;
    float lowest = d, highest = d;
    for (int dv = -1; dv <= 1; ++dv)
      for (int du = -1; du <= 1; ++du) {
        const float n = at(u + du, v + dv);
        lowest = std::min(lowest, n);
        highest = std::max(highest, n);
      }
    if (!(lowest > 0) || highest > (1 + config_.edge_ratio) * lowest) continue;  // edge: flying pixels
    float x = grid_[i].x, y = grid_[i].y;
    if (k1 != 0) undistort_point(x, y, depth.source_width, depth.source_height, k1);
    cloud.points.push_back({d * static_cast<float>((x - K.cx) / K.fx), d * static_cast<float>((y - K.cy) / K.fy), d});
    cloud.colors.push_back(colored ? colors[i] : std::array<std::uint8_t, 3>{170, 170, 170});
  }
  const std::size_t kept = cloud.points.size();
  clouds_.push_back(std::move(cloud));
  if (clouds_.size() > config_.max_clouds) {
    clouds_.erase(clouds_.begin(), clouds_.begin() + static_cast<std::ptrdiff_t>(clouds_.size() - config_.max_clouds));
    ++generation_;
  }
  return kept;
}

void KeyframeDepthStore::clear() {
  clouds_.clear();
  ++generation_;
}

}  // namespace slam_native
