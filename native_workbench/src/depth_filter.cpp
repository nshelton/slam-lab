#include "slam_native/depth_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace slam_native {
namespace {
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr double kDegree = 3.14159265358979323846 / 180.0;

// DepthMap::to_output, repeated here so the filter needs no CUDA translation unit.
void to_output(const DepthMap& map, float x, float y, float& u, float& v) {
  const float cx = x + 0.5F, cy = y + 0.5F;
  const auto w = static_cast<float>(map.source_width), h = static_cast<float>(map.source_height);
  float upright_x = cx, upright_y = cy;
  if (map.rotation == 90) { upright_x = h - cy; upright_y = cx; }
  else if (map.rotation == 180) { upright_x = w - cx; upright_y = h - cy; }
  else if (map.rotation == 270) { upright_x = cy; upright_y = w - cx; }
  u = upright_x * map.scale_x + map.offset_x - 0.5F;
  v = upright_y * map.scale_y + map.offset_y - 0.5F;
}

// Inverse of to_output: output pixel -> native source pixel.
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

double median(std::vector<double>& values) {
  if (values.empty()) return 0;
  const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}

// |grad| of a log-depth field at (u, v): per axis the larger one-sided
// difference (so a pixel on either side of an edge sees the jump), NaN
// neighbours ignored.
template <typename At>
float gradient(const At& at, int u, int v, int width, int height) {
  const float centre = at(u, v);
  const auto axis = [&](int du, int dv) {
    float g = 0;
    for (int s = -1; s <= 1; s += 2) {
      const int x = u + s * du, y = v + s * dv;
      if (x < 0 || y < 0 || x >= width || y >= height) continue;
      const float n = at(x, y);
      if (n == n) g = std::max(g, std::abs(n - centre));
    }
    return g;
  };
  return std::hypot(axis(1, 0), axis(0, 1));
}

bool usable(float metres, float max_metres) { return metres > 0 && (max_metres <= 0 || metres < max_metres); }
}  // namespace

float DepthFilter::log_depth(std::size_t i) const { return has_state_ ? state_[i].z : kNaN; }
float DepthFilter::sigma(std::size_t i) const {
  return has_state_ && state_[i].valid() ? std::sqrt(state_[i].pzz) : kNaN;
}
float DepthFilter::bias(std::size_t i) const { return has_state_ && state_[i].valid() ? state_[i].b : kNaN; }

void DepthFilter::clear() {
  has_state_ = false;
  std::fill(state_.begin(), state_.end(), Pixel{});
}

bool DepthFilter::same_geometry(const DepthMap& map, const CameraIntrinsics& K, double k1) const {
  return width_ > 0 && map.width == geometry_.width && map.height == geometry_.height &&
         map.source_width == geometry_.source_width && map.source_height == geometry_.source_height &&
         map.rotation == geometry_.rotation && map.scale_x == geometry_.scale_x && map.scale_y == geometry_.scale_y &&
         map.offset_x == geometry_.offset_x && map.offset_y == geometry_.offset_y && K.fx == K_.fx &&
         K.fy == K_.fy && K.cx == K_.cx && K.cy == K_.cy && k1 == k1_;
}

void DepthFilter::set_geometry(const DepthMap& map, const CameraIntrinsics& K, double k1) {
  geometry_ = map;
  geometry_.metres.clear();
  K_ = K;
  k1_ = k1;
  width_ = map.width;
  height_ = map.height;
  const std::size_t n = static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
  ray_x_.assign(n, kNaN);
  ray_y_.assign(n, kNaN);
  source_x_.assign(n, kNaN);
  source_y_.assign(n, kNaN);
  for (int v = 0; v < height_; ++v)
    for (int u = 0; u < width_; ++u) {
      float x = 0, y = 0;
      to_source(map, static_cast<float>(u), static_cast<float>(v), x, y);
      if (x < 0 || y < 0 || x > map.source_width - 1 || y > map.source_height - 1) continue;  // letterbox
      const std::size_t i = static_cast<std::size_t>(v) * width_ + u;
      source_x_[i] = x;
      source_y_[i] = y;
      undistort_point(x, y, map.source_width, map.source_height, k1);
      ray_x_[i] = static_cast<float>((x - K.cx) / K.fx);
      ray_y_[i] = static_cast<float>((y - K.cy) / K.fy);
    }
  state_.assign(n, Pixel{});
  next_.assign(n, Pixel{});
  zbuffer_.assign(n, 0);
  best_distance_.assign(n, 0);
  best_source_.assign(n, -1);
  target_u_.assign(n, 0);
  target_v_.assign(n, 0);
  target_z_.assign(n, kNaN);
  has_state_ = false;
}

void DepthFilter::rescale(double factor) {
  if (!(factor > 0)) return;
  const auto shift = static_cast<float>(std::log(factor));
  for (auto& p : state_) {
    if (p.valid()) p.z -= shift;
    if (p.raw == p.raw) p.raw -= shift;
  }
  for (auto& t : previous_pose_.translation) t /= factor;
  previous_median_z_ -= shift;
}

void DepthFilter::predict(const DepthFilterFrame& frame, DepthFilterStats& stats) {
  const std::size_t n = state_.size();
  // Relative motion previous -> current camera: x1 = R x0 + t.
  const auto& R0 = previous_pose_.rotation;
  const auto& R1 = frame.pose.rotation;
  double R[9];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      R[r * 3 + c] = R1[r * 3] * R0[c * 3] + R1[r * 3 + 1] * R0[c * 3 + 1] + R1[r * 3 + 2] * R0[c * 3 + 2];
  double t[3];
  for (int r = 0; r < 3; ++r)
    t[r] = frame.pose.translation[r] -
           (R[r * 3] * previous_pose_.translation[0] + R[r * 3 + 1] * previous_pose_.translation[1] +
            R[r * 3 + 2] * previous_pose_.translation[2]);

  // Pose uncertainty: the camera centre's sigma (translation_sigma_ratio is
  // relative to the median depth) and the rotation as output pixels.
  const double translation_ratio = std::isfinite(frame.translation_sigma_ratio) ? frame.translation_sigma_ratio : 0.0;
  const double translation_sigma = translation_ratio * std::exp(previous_median_z_);
  const double rotation_degrees = std::isfinite(frame.rotation_sigma_degrees) ? frame.rotation_sigma_degrees : 0.0;
  const double output_scale = 0.5 * (geometry_.scale_x + geometry_.scale_y);
  const double rotation_px = K_.fx * rotation_degrees * kDegree * output_scale;
  const double a = std::exp(-1.0 / std::max(1.0, config_.bias_correlation_frames));
  const double bias_q = (1 - a * a) * config_.bias_sigma * config_.bias_sigma;
  const double drift2 = config_.drift_sigma * config_.drift_sigma;

  // Forward-project every state pixel.
  const int sw = geometry_.source_width, sh = geometry_.source_height;
  for (std::size_t i = 0; i < n; ++i) {
    target_z_[i] = kNaN;
    const Pixel& p = state_[i];
    if (!p.valid()) continue;
    const double d = std::exp(static_cast<double>(p.z));
    const double X = d * ray_x_[i], Y = d * ray_y_[i], Z = d;
    const double x1 = R[0] * X + R[1] * Y + R[2] * Z + t[0];
    const double y1 = R[3] * X + R[4] * Y + R[5] * Z + t[1];
    const double z1 = R[6] * X + R[7] * Y + R[8] * Z + t[2];
    if (!(z1 > 1e-3 * d)) continue;
    float x = static_cast<float>(K_.fx * x1 / z1 + K_.cx), y = static_cast<float>(K_.fy * y1 / z1 + K_.cy);
    distort_point(x, y, sw, sh, k1_);
    if (!(x >= -1 && y >= -1 && x <= sw && y <= sh)) continue;
    float u = 0, v = 0;
    to_output(geometry_, x, y, u, v);
    target_u_[i] = u;
    target_v_[i] = v;
    target_z_[i] = static_cast<float>(std::log(z1));
  }
  // Splat to the 4 nearest output pixels (fills the cracks of a magnifying
  // warp). Pass 1: nearest surface per target; pass 2: among sources within
  // splat_tolerance of it, the closest sub-pixel position wins.
  std::fill(zbuffer_.begin(), zbuffer_.end(), std::numeric_limits<float>::infinity());
  std::fill(best_distance_.begin(), best_distance_.end(), std::numeric_limits<float>::infinity());
  std::fill(best_source_.begin(), best_source_.end(), -1);
  const auto for_targets = [&](std::size_t i, const auto& visit) {
    const int u0 = static_cast<int>(std::floor(target_u_[i])), v0 = static_cast<int>(std::floor(target_v_[i]));
    for (int dv = 0; dv <= 1; ++dv)
      for (int du = 0; du <= 1; ++du) {
        const int u = u0 + du, v = v0 + dv;
        if (u < 0 || v < 0 || u >= width_ || v >= height_) continue;
        const std::size_t target = static_cast<std::size_t>(v) * width_ + u;
        if (ray_x_[target] != ray_x_[target]) continue;  // letterbox
        visit(target, static_cast<float>(u), static_cast<float>(v));
      }
  };
  for (std::size_t i = 0; i < n; ++i)
    if (target_z_[i] == target_z_[i])
      for_targets(i, [&](std::size_t target, float, float) { zbuffer_[target] = std::min(zbuffer_[target], target_z_[i]); });
  for (std::size_t i = 0; i < n; ++i)
    if (target_z_[i] == target_z_[i])
      for_targets(i, [&](std::size_t target, float u, float v) {
        if (target_z_[i] > zbuffer_[target] + config_.splat_tolerance) return;
        const float du = target_u_[i] - u, dv = target_v_[i] - v;
        const float distance = du * du + dv * dv;
        if (distance < best_distance_[target]) {
          best_distance_[target] = distance;
          best_source_[target] = static_cast<std::int32_t>(i);
        }
      });

  const auto state_at = [&](int u, int v) { return state_[static_cast<std::size_t>(v) * width_ + u].z; };
  int carried = 0;
  for (std::size_t target = 0; target < n; ++target) {
    Pixel& q = next_[target];
    q = Pixel{};
    const std::int32_t source = best_source_[target];
    if (source < 0) continue;
    const Pixel& p = state_[static_cast<std::size_t>(source)];
    const double z1 = std::exp(static_cast<double>(target_z_[source]));
    // d log z1 / d log z0 = (z1 - t_z) / z1.
    const double J = (z1 - t[2]) / z1;
    const int su = static_cast<int>(static_cast<std::size_t>(source) % width_);
    const int sv = static_cast<int>(static_cast<std::size_t>(source) / width_);
    const double g = std::max(0.0, gradient(state_at, su, sv, width_, height_) - config_.gradient_floor);
    const double position = std::sqrt(best_distance_[target]) + rotation_px;
    const double q_z = (translation_sigma / z1) * (translation_sigma / z1) + drift2 + g * g * position * position;
    // b moves with the same Jacobian as z: the network is assumed to see the
    // depth change as the truth does, so network maps alone carry no
    // triangulation (a constant log bias would let depth ratios between frames
    // pin down absolute depth, which real networks do not support). Only a
    // geometric measurement separates z from b.
    q.z = target_z_[source];
    q.b = static_cast<float>(a * J * p.b);
    q.pzz = static_cast<float>(J * J * p.pzz + q_z);
    q.pzb = static_cast<float>(J * J * a * p.pzb);
    q.pbb = static_cast<float>(a * a * J * J * p.pbb + bias_q);
    q.raw = p.raw == p.raw ? p.raw + (q.z - p.z) : kNaN;
    q.count = p.count;
    q.outliers = p.outliers;
    ++carried;
  }
  state_.swap(next_);
  stats.carried = carried;
}

DepthFilterStats DepthFilter::process(const DepthFilterFrame& frame) {
  DepthFilterStats stats;
  const DepthMap& depth = *frame.depth;
  stats.frame_index = depth.frame_index;
  if (!same_geometry(depth, frame.K, frame.k1)) set_geometry(depth, frame.K, frame.k1);
  if (has_state_ && frame.segment != segment_) clear();
  if (has_state_) {
    predict(frame, stats);
    stats.predicted = true;
  }
  // Flicker reference: the predicted (pre-update) z where it is flat.
  const auto predicted_at = [&](int u, int v) { return state_[static_cast<std::size_t>(v) * width_ + u].z; };
  std::vector<double> flicker_raw, flicker_fused;
  const int sw = depth.source_width, sh = depth.source_height;
  const double frac2 = config_.independent_fraction * config_.independent_fraction;
  for (int v = 0; v < height_; ++v)
    for (int u = 0; u < width_; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * width_ + u;
      Pixel& p = state_[i];
      const float metres = depth.metres[i];
      if (ray_x_[i] != ray_x_[i] || !usable(metres, config_.max_metres)) continue;
      const double L = frame.grid ? scale_grid_at(*frame.grid, sw, sh, source_x_[i], source_y_[i]) : frame.log_scale;
      const double m = std::log(static_cast<double>(metres)) - L;
      double total = config_.prior_sigma;
      if (frame.sigma && (*frame.sigma)[i] > 0) total = (*frame.sigma)[i];
      const double r = frac2 * total * total;  // independent part: the measurement noise
      const double b0 = (1 - frac2) * total * total;  // shared part: the bias prior
      const auto initialise = [&] {
        p.z = static_cast<float>(m);
        p.b = 0;
        p.pzz = static_cast<float>(b0 + r);
        p.pzb = static_cast<float>(-b0);
        p.pbb = static_cast<float>(b0);
        p.count = 1;
        p.outliers = 0;
      };
      if (!p.valid()) {
        initialise();
        p.raw = static_cast<float>(m);
        ++stats.initialised;
        continue;
      }
      const double z_predicted = p.z;
      const bool flat = p.raw == p.raw && gradient(predicted_at, u, v, width_, height_) < config_.flicker_max_gradient;
      if (flat) flicker_raw.push_back(std::abs(m - p.raw));
      const double s = static_cast<double>(p.pzz) + 2.0 * p.pzb + p.pbb + r;
      const double innovation = m - (static_cast<double>(p.z) + p.b);
      p.raw = static_cast<float>(m);
      if (innovation * innovation > config_.gate_sigmas * config_.gate_sigmas * s) {
        ++stats.gated;
        if (++p.outliers >= config_.reset_after) {
          initialise();
          ++stats.resets;
        }
      } else {
        const double hz = static_cast<double>(p.pzz) + p.pzb, hb = static_cast<double>(p.pzb) + p.pbb;
        const double kz = hz / s, kb = hb / s;
        p.z = static_cast<float>(p.z + kz * innovation);
        p.b = static_cast<float>(p.b + kb * innovation);
        p.pzz = static_cast<float>(p.pzz - kz * hz);
        p.pzb = static_cast<float>(p.pzb - kz * hb);
        p.pbb = static_cast<float>(p.pbb - kb * hb);
        p.outliers = 0;
        if (p.count < std::numeric_limits<std::uint16_t>::max()) ++p.count;
        ++stats.updated;
      }
      if (flat) flicker_fused.push_back(std::abs(p.z - z_predicted));
    }
  std::vector<double> sigmas, sums, depths;
  for (std::size_t i = 0; i < state_.size(); ++i) {
    const Pixel& p = state_[i];
    if (!p.valid()) continue;
    ++stats.valid;
    sigmas.push_back(std::sqrt(p.pzz));
    sums.push_back(std::sqrt(std::max(0.0, static_cast<double>(p.pzz) + 2.0 * p.pzb + p.pbb)));
    if (i % 7 == 0) depths.push_back(p.z);
  }
  stats.flicker_pixels = static_cast<int>(flicker_raw.size());
  stats.flicker_raw = median(flicker_raw);
  stats.flicker_fused = median(flicker_fused);
  stats.median_sigma = median(sigmas);
  stats.median_sum_sigma = median(sums);
  previous_median_z_ = median(depths);
  previous_pose_ = frame.pose;
  frame_index_ = depth.frame_index;
  segment_ = frame.segment;
  has_state_ = stats.valid > 0;
  return stats;
}

DepthMap DepthFilter::to_depth_map(double log_scale, const ScaleGrid* grid) const {
  DepthMap map = geometry_;
  map.frame_index = frame_index_;
  map.metres.assign(state_.size(), 0.0F);
  if (!has_state_) return map;
  for (std::size_t i = 0; i < state_.size(); ++i) {
    if (!state_[i].valid()) continue;
    const double L = grid ? scale_grid_at(*grid, map.source_width, map.source_height, source_x_[i], source_y_[i])
                          : log_scale;
    map.metres[i] = static_cast<float>(std::exp(state_[i].z + L));
  }
  return map;
}

std::vector<float> network_sigma_map(const DepthMap& depth, const DepthConfidenceModel& model,
                                     const std::vector<Keypoint>& landmarks) {
  const int w = depth.width, h = depth.height;
  const std::size_t n = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
  std::vector<float> sigma(n, kNaN);
  // Distance (output px) to the nearest landmark: two-pass chamfer transform.
  std::vector<float> distance;
  if (!landmarks.empty()) {
    distance.assign(n, std::numeric_limits<float>::infinity());
    for (const auto& k : landmarks) {
      float u = 0, v = 0;
      to_output(depth, k.x, k.y, u, v);
      const int x = static_cast<int>(std::lround(u)), y = static_cast<int>(std::lround(v));
      if (x >= 0 && y >= 0 && x < w && y < h) distance[static_cast<std::size_t>(y) * w + x] = 0;
    }
    constexpr float diagonal = 1.41421356F;
    const auto relax = [&](int x, int y, int dx, int dy, float step) {
      const int nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) return;
      float& d = distance[static_cast<std::size_t>(y) * w + x];
      d = std::min(d, distance[static_cast<std::size_t>(ny) * w + nx] + step);
    };
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        relax(x, y, -1, 0, 1); relax(x, y, 0, -1, 1); relax(x, y, -1, -1, diagonal); relax(x, y, 1, -1, diagonal);
      }
    for (int y = h - 1; y >= 0; --y)
      for (int x = w - 1; x >= 0; --x) {
        relax(x, y, 1, 0, 1); relax(x, y, 0, 1, 1); relax(x, y, 1, 1, diagonal); relax(x, y, -1, 1, diagonal);
      }
  }
  const float output_scale = 0.5F * (depth.scale_x + depth.scale_y);
  const float half_diagonal = 0.5F * std::hypot(static_cast<float>(depth.source_width), static_cast<float>(depth.source_height));
  for (int v = 0; v < h; ++v)
    for (int u = 0; u < w; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * w + u;
      const float metres = depth.metres[i];
      if (!(metres > 0)) continue;
      // Cues as KeyframeDepthStore computes them for its references.
      float lowest = std::numeric_limits<float>::max(), highest = 0;
      bool complete = true;
      for (int dv = -1; dv <= 1 && complete; ++dv)
        for (int du = -1; du <= 1; ++du) {
          const int x = std::clamp(u + du, 0, w - 1), y = std::clamp(v + dv, 0, h - 1);
          const float m = depth.metres[static_cast<std::size_t>(y) * w + x];
          if (!(m > 0)) { complete = false; break; }
          lowest = std::min(lowest, m);
          highest = std::max(highest, m);
        }
      float x = 0, y = 0;
      to_source(depth, static_cast<float>(u), static_cast<float>(v), x, y);
      DepthCues cues;
      cues.edge = complete ? std::log(highest / lowest) : 0.0F;
      cues.log_depth = std::log(metres);
      cues.radius = std::hypot(x - 0.5F * depth.source_width, y - 0.5F * depth.source_height) / half_diagonal;
      cues.nearest_landmark = distance.empty() ? 0.5F : distance[i] / output_scale / static_cast<float>(depth.source_width);
      sigma[i] = static_cast<float>(model.sigma(cues));
    }
  return sigma;
}

}  // namespace slam_native
