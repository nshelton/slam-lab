#include "slam_native/keyframe_depth.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

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

// world -> camera: x = R X + t (row-major R).
std::array<double, 3> apply(const Pose& pose, const std::array<double, 3>& X) {
  const auto& R = pose.rotation;
  return {R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + pose.translation[0],
          R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + pose.translation[1],
          R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + pose.translation[2]};
}
// camera -> world: X = R^T (x - t).
std::array<double, 3> unapply(const Pose& pose, const std::array<double, 3>& x) {
  const auto& R = pose.rotation;
  const double a = x[0] - pose.translation[0], b = x[1] - pose.translation[1], c = x[2] - pose.translation[2];
  return {R[0] * a + R[3] * b + R[6] * c, R[1] * a + R[4] * b + R[7] * c, R[2] * a + R[5] * b + R[8] * c};
}
}  // namespace

namespace {
// log(max / min) of depth over the 3x3 output neighbourhood of (u, v); 0 if invalid.
float edge_strength(const DepthMap& depth, int u, int v) {
  float lowest = std::numeric_limits<float>::max(), highest = 0;
  for (int dv = -1; dv <= 1; ++dv)
    for (int du = -1; du <= 1; ++du) {
      const int x = std::clamp(u + du, 0, depth.width - 1), y = std::clamp(v + dv, 0, depth.height - 1);
      const float n = depth.metres[static_cast<std::size_t>(y) * depth.width + static_cast<std::size_t>(x)];
      if (!(n > 0)) return 0;
      lowest = std::min(lowest, n);
      highest = std::max(highest, n);
    }
  return std::log(highest / lowest);
}

float image_radius(const DepthMap& depth, float x, float y) {
  const float half_diagonal = 0.5F * std::hypot(static_cast<float>(depth.source_width), static_cast<float>(depth.source_height));
  return std::hypot(x - 0.5F * depth.source_width, y - 0.5F * depth.source_height) / half_diagonal;
}

// Distance (source px / width) from (x, y) to the nearest reference, skipping index `self`.
float nearest_reference(const std::vector<DepthReference>& references, float x, float y, int width,
                        std::size_t self = static_cast<std::size_t>(-1)) {
  float best = std::numeric_limits<float>::max();
  for (std::size_t k = 0; k < references.size(); ++k)
    if (k != self) best = std::min(best, (references[k].x - x) * (references[k].x - x) + (references[k].y - y) * (references[k].y - y));
  return best == std::numeric_limits<float>::max() ? 0.5F : std::sqrt(best) / static_cast<float>(width);
}
}  // namespace

void KeyframeDepthStore::learn(const KeyframeDepthInput& input, const DepthMap& depth) {
  if (input.references.empty() || !std::isfinite(input.log_scale)) return;
  const int w = depth.source_width, h = depth.source_height;
  for (std::size_t k = 0; k < input.references.size(); ++k) {
    const auto& reference = input.references[k];
    if (!(reference.depth > 0)) continue;
    const float metres = depth.at_source(reference.x, reference.y);
    if (!(metres > 0)) continue;
    // Same alignment as the clouds and the multi-view check: the keyframe's grid.
    const double L = input.grid ? scale_grid_at(*input.grid, w, h, reference.x, reference.y) : input.log_scale;
    const double residual = std::log(metres) - L - std::log(reference.depth);
    float u = 0, v = 0;
    depth.to_output(reference.x, reference.y, u, v);
    const DepthCues cues{edge_strength(depth, static_cast<int>(std::lround(u)), static_cast<int>(std::lround(v))),
                         std::log(metres), image_radius(depth, reference.x, reference.y),
                         nearest_reference(input.references, reference.x, reference.y, w, k)};
    confidence_.add_reference(cues, residual, reference.geometric_sigma);
  }
  confidence_.fit(4);
}


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

KeyframeCloud* KeyframeDepthStore::find(std::uint64_t frame_index) {
  auto it = std::lower_bound(clouds_.begin(), clouds_.end(), frame_index,
                             [](const KeyframeCloud& c, std::uint64_t f) { return c.frame_index < f; });
  return it != clouds_.end() && it->frame_index == frame_index ? &*it : nullptr;
}

void KeyframeDepthStore::apply_scale(KeyframeCloud& cloud, double log_scale, const ScaleGrid* grid) const {
  cloud.metres_per_unit = std::isfinite(log_scale) ? std::exp(log_scale) : 0.0;
  cloud.points.resize(cloud.raw.size());
  for (std::size_t i = 0; i < cloud.raw.size(); ++i) {
    // p / metres_per_unit = raw / exp(L(pixel)): the grid's local scale.
    const double factor = grid && std::isfinite(log_scale) ?
        std::exp(log_scale - scale_grid_at(*grid, geometry_.source_width, geometry_.source_height,
                                           cloud.pixels[i][0], cloud.pixels[i][1])) : 1.0;
    for (int k = 0; k < 3; ++k) cloud.points[i][k] = static_cast<float>(cloud.raw[i][k] * factor);
  }
  ++cloud.version;
}

bool KeyframeDepthStore::check(KeyframeCloud& cloud, const Pose& pose, const Recent& target,
                               const Pose& target_pose, std::vector<DifferenceSample>* samples) const {
  if (!(cloud.metres_per_unit > 0) || !std::isfinite(target.log_scale)) return false;
  const int w = target.depth.source_width, h = target.depth.source_height;
  const bool adaptive = config_.adaptive_tolerance && difference_.fitted();
  const auto centre = pose.center(), target_centre = target_pose.center();
  bool changed = false;
  for (std::size_t i = 0; i < cloud.points.size(); ++i) {
    const auto& p = cloud.points[i];
    const double s = 1.0 / cloud.metres_per_unit;
    const auto X = unapply(pose, {p[0] * s, p[1] * s, p[2] * s});
    const auto x = apply(target_pose, X);
    if (x[2] <= 1e-9) continue;
    float u = static_cast<float>(K_.fx * x[0] / x[2] + K_.cx), v = static_cast<float>(K_.fy * x[1] / x[2] + K_.cy);
    if (k1_ != 0) distort_point(u, v, w, h, k1_);
    if (u < 0 || v < 0 || u > w - 1 || v > h - 1) continue;
    const float metres = target.depth.at_source(u, v);
    if (!(metres > 0)) continue;
    const double L = target.has_grid ? scale_grid_at(target.grid, w, h, u, v) : target.log_scale;
    const double r = std::log(metres) - L - std::log(x[2]);  // target's depth vs the point's, in VO units
    // Cues for the between-view model, at the target pixel.
    float ou = 0, ov = 0;
    target.depth.to_output(u, v, ou, ov);
    double cosine = 0, a2 = 0, b2 = 0;
    for (int k = 0; k < 3; ++k) {
      const double a = X[k] - centre[k], b = X[k] - target_centre[k];
      cosine += a * b;
      a2 += a * a;
      b2 += b * b;
    }
    const DepthCues cues{edge_strength(target.depth, static_cast<int>(std::lround(ou)), static_cast<int>(std::lround(ov))),
                         std::log(metres), image_radius(target.depth, u, v),
                         static_cast<float>(std::acos(std::clamp(cosine / std::sqrt(a2 * b2 + 1e-300), -1.0, 1.0)))};
    const double tolerance = adaptive ?
        std::clamp(config_.tolerance_sigmas * difference_.sigma(cues), static_cast<double>(config_.tolerance_min),
                   static_cast<double>(config_.tolerance_max)) :
        static_cast<double>(config_.consistency_tolerance);
    if (samples && cloud.agree[i] > 0) samples->push_back({cues, r});  // confirmed by an earlier view
    if (r < -tolerance) continue;  // the target sees something nearer: occluded there, no vote
    auto& votes = std::abs(r) <= tolerance ? cloud.agree[i] : cloud.disagree[i];
    if (votes < 255) ++votes;
    const std::uint8_t shown = cloud.agree[i] > 0 || cloud.disagree[i] == 0;
    if (shown != cloud.shown[i]) {
      cloud.shown[i] = shown;
      changed = true;
    }
  }
  return changed;
}

std::size_t KeyframeDepthStore::add(const KeyframeDepthInput& input, const DepthMap& depth,
                                    const std::vector<std::array<std::uint8_t, 3>>& colors, const CameraIntrinsics& K,
                                    double k1, const PoseLookup& pose_of) {
  if (depth.empty() || K.fx <= 0 || K.fy <= 0) return 0;
  K_ = K;
  k1_ = k1;
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
  learn(input, depth);
  KeyframeCloud cloud;
  cloud.frame_index = input.frame_index;
  cloud.segment = input.segment;
  cloud.raw.reserve(grid_.size());
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
    cloud.pixels.push_back({x, y});
    if (k1 != 0) undistort_point(x, y, depth.source_width, depth.source_height, k1);
    cloud.sigma.push_back(static_cast<float>(confidence_.sigma(
        {std::log(highest / lowest), std::log(d), image_radius(depth, grid_[i].x, grid_[i].y),
         nearest_reference(input.references, grid_[i].x, grid_[i].y, depth.source_width)})));
    cloud.raw.push_back({d * static_cast<float>((x - K.cx) / K.fx), d * static_cast<float>((y - K.cy) / K.fy), d});
    cloud.colors.push_back(colored ? colors[i] : std::array<std::uint8_t, 3>{170, 170, 170});
  }
  const std::size_t kept = cloud.raw.size();
  cloud.shown.assign(kept, 1);
  cloud.agree.assign(kept, 0);
  cloud.disagree.assign(kept, 0);
  apply_scale(cloud, input.log_scale, input.grid);
  clouds_.push_back(std::move(cloud));
  if (clouds_.size() > config_.max_clouds) {
    clouds_.erase(clouds_.begin(), clouds_.begin() + static_cast<std::ptrdiff_t>(clouds_.size() - config_.max_clouds));
    ++generation_;
  }

  // Multi-view check against recent keyframes of this segment, both ways.
  Recent self{input.frame_index, input.segment, depth, input.log_scale, input.grid != nullptr, {}};
  if (input.grid) self.grid = *input.grid;
  const auto current_pose = [&](std::uint64_t frame, const Pose& fallback) {
    if (pose_of)
      if (auto p = pose_of(frame)) return *p;
    return fallback;
  };
  const Pose pose = current_pose(input.frame_index, input.pose);
  std::vector<DifferenceSample> difference_samples;
  for (const auto& other : recent_) {
    if (other.segment != input.segment) continue;
    KeyframeCloud* other_cloud = find(other.frame_index);
    if (!other_cloud) continue;
    const auto other_pose = pose_of ? pose_of(other.frame_index) : std::nullopt;
    if (!other_pose) continue;
    KeyframeCloud& mine = clouds_.back();
    if (check(mine, pose, other, *other_pose, &difference_samples)) ++mine.version;
    if (check(*other_cloud, *other_pose, self, pose, &difference_samples)) ++other_cloud->version;
  }
  for (const auto& sample : difference_samples) difference_.add_reference(sample.cues, sample.residual, 0.0);
  if (!difference_samples.empty()) difference_.fit(4);
  recent_.push_back(std::move(self));
  while (static_cast<int>(recent_.size()) > std::max(0, config_.neighbours)) recent_.pop_front();
  return kept;
}

bool KeyframeDepthStore::settle(std::uint64_t frame_index, double log_scale, const ScaleGrid* grid) {
  KeyframeCloud* cloud = find(frame_index);
  if (!cloud || !std::isfinite(log_scale)) return false;
  apply_scale(*cloud, log_scale, grid);
  cloud->settled = true;
  for (auto& r : recent_)  // keep the multi-view reference consistent
    if (r.frame_index == frame_index) {
      r.log_scale = log_scale;
      r.has_grid = grid != nullptr;
      if (grid) r.grid = *grid;
    }
  return true;
}

bool KeyframeDepthStore::rescale(std::uint64_t frame_index, double factor) {
  KeyframeCloud* cloud = find(frame_index);
  if (!cloud || !(factor > 0) || !std::isfinite(factor)) return false;
  cloud->metres_per_unit *= factor;
  const double shift = std::log(factor);
  for (auto& r : recent_)
    if (r.frame_index == frame_index) {
      r.log_scale += shift;
      for (auto& g : r.grid) g += shift;
    }
  return true;
}

void KeyframeDepthStore::clear() {
  clouds_.clear();
  recent_.clear();
  ++generation_;
}

}  // namespace slam_native
