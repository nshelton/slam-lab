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
  alignment_started_ = false;
  alignment_.fill(0);
  log_scale_ = std::numeric_limits<double>::quiet_NaN();
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
  measurement_.log_depth.assign(n, kNaN);
  measurement_.gradient.assign(n, 0);
  measurement_.sigma.assign(n, 0);
  zbuffer_.assign(n, 0);
  best_distance_.assign(n, 0);
  best_source_.assign(n, -1);
  target_u_.assign(n, 0);
  target_v_.assign(n, 0);
  target_z_.assign(n, kNaN);
  half_.assign(n, 0.5F);
  sampled_.assign(n, kNaN);
  nearest_.assign(n, 0);
  has_state_ = false;
}

void DepthFilter::rescale(double factor) {
  if (!(factor > 0)) return;
  // The state is in metres: only the anchor pose and the units' scale change.
  for (auto& t : anchor_pose_.translation) t /= factor;
  if (std::isfinite(log_scale_)) log_scale_ += std::log(factor);
}

void DepthFilter::prepare(const DepthFilterFrame& frame) {
  const DepthMap& depth = *frame.depth;
  const int sw = depth.source_width, sh = depth.source_height;
  auto& m = measurement_;
  for (std::size_t i = 0; i < m.log_depth.size(); ++i) {
    m.log_depth[i] = kNaN;
    if (ray_x_[i] != ray_x_[i] || !usable(depth.metres[i], config_.max_metres)) continue;
    // Only the grid's shape (relative to its keyframe's scale) is applied: the state is in metres.
    const double L = frame.grid ? scale_grid_at(*frame.grid, sw, sh, source_x_[i], source_y_[i]) - frame.log_scale : 0.0;
    m.log_depth[i] = static_cast<float>(std::log(static_cast<double>(depth.metres[i])) - L);
    const float s = frame.sigma ? (*frame.sigma)[i] : 0.0F;
    m.sigma[i] = s > 0 ? s : static_cast<float>(config_.prior_sigma);
  }
  const auto at = [&](int u, int v) { return m.log_depth[static_cast<std::size_t>(v) * width_ + u]; };
  for (int v = 0; v < height_; ++v)
    for (int u = 0; u < width_; ++u) {
      const std::size_t i = static_cast<std::size_t>(v) * width_ + u;
      m.gradient[i] = m.log_depth[i] == m.log_depth[i] ? gradient(at, u, v, width_, height_) : 0.0F;
    }
}

void DepthFilter::relative(const Pose& pose, double R[9], double t[3]) const {
  const auto& R0 = anchor_pose_.rotation;
  const auto& R1 = pose.rotation;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      R[r * 3 + c] = R1[r * 3] * R0[c * 3] + R1[r * 3 + 1] * R0[c * 3 + 1] + R1[r * 3 + 2] * R0[c * 3 + 2];
  // VO units -> metres (the state's units) with the smoothed scale.
  const double metres_per_unit = std::isfinite(log_scale_) ? std::exp(log_scale_) : 1.0;
  for (int r = 0; r < 3; ++r)
    t[r] = metres_per_unit * (pose.translation[r] -
                              (R[r * 3] * anchor_pose_.translation[0] + R[r * 3 + 1] * anchor_pose_.translation[1] +
                               R[r * 3 + 2] * anchor_pose_.translation[2]));
}

void DepthFilter::project(const double R[9], const double t[3], bool four_neighbours) {
  const std::size_t n = state_.size();
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
  // Footprint: half the spacing to the projections of the neighbours on the
  // same surface (at least half a pixel), so a magnified surface stays closed
  // and a hidden one cannot show through its gaps.
  for (std::size_t i = 0; i < n; ++i) {
    half_[i] = 0.5F;
    if (target_z_[i] != target_z_[i]) continue;
    const int u = static_cast<int>(i % static_cast<std::size_t>(width_)), v = static_cast<int>(i / static_cast<std::size_t>(width_));
    float spacing = 1.0F;
    const auto neighbour = [&](int nu, int nv) {
      if (nu < 0 || nv < 0 || nu >= width_ || nv >= height_) return;
      const std::size_t k = static_cast<std::size_t>(nv) * width_ + nu;
      if (target_z_[k] != target_z_[k] || std::abs(target_z_[k] - target_z_[i]) > config_.splat_tolerance) return;
      spacing = std::max(spacing, std::max(std::abs(target_u_[k] - target_u_[i]), std::abs(target_v_[k] - target_v_[i])));
    };
    neighbour(u + 1, v);
    neighbour(u - 1, v);
    neighbour(u, v + 1);
    neighbour(u, v - 1);
    half_[i] = std::min(1.5F, 0.5F * spacing);
  }
  // Nearest surface per output pixel: over the footprint (covering), or over
  // the 4 nearest pixels (conservative, for visibility tests).
  std::fill(zbuffer_.begin(), zbuffer_.end(), std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < n; ++i) {
    if (target_z_[i] != target_z_[i]) continue;
    const float h = four_neighbours ? 1.0F : half_[i];
    const int u0 = static_cast<int>(four_neighbours ? std::floor(target_u_[i]) : std::ceil(target_u_[i] - h));
    const int u1 = static_cast<int>(four_neighbours ? std::floor(target_u_[i]) + 1 : std::floor(target_u_[i] + h));
    const int v0 = static_cast<int>(four_neighbours ? std::floor(target_v_[i]) : std::ceil(target_v_[i] - h));
    const int v1 = static_cast<int>(four_neighbours ? std::floor(target_v_[i]) + 1 : std::floor(target_v_[i] + h));
    for (int v = std::max(0, v0); v <= std::min(height_ - 1, v1); ++v)
      for (int u = std::max(0, u0); u <= std::min(width_ - 1, u1); ++u) {
        float& z = zbuffer_[static_cast<std::size_t>(v) * width_ + u];
        z = std::min(z, target_z_[i]);
      }
  }
}

void DepthFilter::decay() {
  const double a = std::exp(-1.0 / std::max(1.0, config_.bias_correlation_frames));
  const auto bias_q = static_cast<float>((1 - a * a) * config_.bias_sigma * config_.bias_sigma);
  const auto drift2 = static_cast<float>(config_.drift_sigma * config_.drift_sigma);
  const auto af = static_cast<float>(a);
  for (auto& p : state_) {
    if (!p.valid()) continue;
    p.b *= af;
    p.pzb *= af;
    p.pbb = af * af * p.pbb + bias_q;
    p.pzz += drift2;
  }
}

void DepthFilter::select_sources() {
  // Candidates: sources within one pixel. Where some source covers the pixel,
  // only those within splat_tolerance of the nearest covering one compete;
  // elsewhere (cracks of a magnifying projection) any candidate does. The
  // closest sub-pixel position wins.
  const std::size_t n = state_.size();
  std::fill(best_distance_.begin(), best_distance_.end(), std::numeric_limits<float>::infinity());
  std::fill(best_source_.begin(), best_source_.end(), -1);
  for (std::size_t i = 0; i < n; ++i) {
    if (target_z_[i] != target_z_[i]) continue;
    const float reach = std::max(1.0F, half_[i]);
    const int u0 = static_cast<int>(std::ceil(target_u_[i] - reach)), u1 = static_cast<int>(std::floor(target_u_[i] + reach));
    const int v0 = static_cast<int>(std::ceil(target_v_[i] - reach)), v1 = static_cast<int>(std::floor(target_v_[i] + reach));
    for (int v = std::max(0, v0); v <= std::min(height_ - 1, v1); ++v)
      for (int u = std::max(0, u0); u <= std::min(width_ - 1, u1); ++u) {
        const std::size_t target = static_cast<std::size_t>(v) * width_ + u;
        if (ray_x_[target] != ray_x_[target]) continue;  // letterbox
        const float ddu = target_u_[i] - static_cast<float>(u), ddv = target_v_[i] - static_cast<float>(v);
        if (std::isfinite(zbuffer_[target])) {
          if (target_z_[i] > zbuffer_[target] + config_.splat_tolerance) continue;
        } else if (std::abs(ddu) >= 1.0F || std::abs(ddv) >= 1.0F) {
          continue;  // crack fill: only from within a pixel
        }
        const float distance = ddu * ddu + ddv * ddv;
        if (distance < best_distance_[target]) {
          best_distance_[target] = distance;
          best_source_[target] = static_cast<std::int32_t>(i);
        }
      }
  }
}

float DepthFilter::resampled(std::size_t source, std::size_t target) const {
  // The source's log depth at the target pixel centre, along the local plane
  // through its neighbours' projections (same surface only). Taking the
  // source's depth as is would add gradient x sub-pixel offset at every
  // resampling: speckle on slanted surfaces that compounds over keyframes.
  const float z = target_z_[source];
  const int su = static_cast<int>(source % static_cast<std::size_t>(width_));
  const int sv = static_cast<int>(source / static_cast<std::size_t>(width_));
  const auto neighbour = [&](int du, int dv, float& u, float& v, float& dz) {
    for (const int sign : {1, -1}) {
      const int nu = su + sign * du, nv = sv + sign * dv;
      if (nu < 0 || nv < 0 || nu >= width_ || nv >= height_) continue;
      const std::size_t k = static_cast<std::size_t>(nv) * width_ + nu;
      const float nz = target_z_[k];
      if (nz != nz || std::abs(nz - z) > config_.splat_tolerance) continue;
      u = target_u_[k] - target_u_[source];
      v = target_v_[k] - target_v_[source];
      dz = nz - z;
      return true;
    }
    return false;
  };
  float au = 0, av = 0, az = 0, bu = 0, bv = 0, bz = 0;
  if (!neighbour(1, 0, au, av, az) || !neighbour(0, 1, bu, bv, bz)) return z;
  // [au av; bu bv] [gu; gv] = [az; bz]
  const float det = au * bv - av * bu;
  if (std::abs(det) < 0.05F) return z;
  const float gu = (az * bv - av * bz) / det, gv = (au * bz - az * bu) / det;
  const float tu = static_cast<float>(target % static_cast<std::size_t>(width_));
  const float tv = static_cast<float>(target / static_cast<std::size_t>(width_));
  return z + gu * (tu - target_u_[source]) + gv * (tv - target_v_[source]);
}

void DepthFilter::render(const Pose& pose, DepthFilterView& view) {
  const std::size_t n = state_.size();
  view.width = width_;
  view.height = height_;
  view.frame_index = frame_index_;
  view.metres.assign(n, 0.0F);
  view.sigma.assign(n, kNaN);
  view.consistency.assign(n, kNaN);
  view.bias.assign(n, kNaN);
  view.observations.assign(n, 0);
  if (!has_state_) return;
  double R[9], t[3];
  relative(pose, R, t);
  project(R, t, false);
  select_sources();
  for (std::size_t target = 0; target < n; ++target) {
    const std::int32_t source = best_source_[target];
    if (source < 0) continue;
    const Pixel& p = state_[static_cast<std::size_t>(source)];
    const double z1 = std::exp(static_cast<double>(resampled(static_cast<std::size_t>(source), target)));
    const double J = (z1 - t[2]) / z1;
    view.metres[target] = static_cast<float>(z1);
    view.sigma[target] = static_cast<float>(std::abs(J) * std::sqrt(p.pzz));
    view.consistency[target] =
        static_cast<float>(std::sqrt(std::max(0.0, static_cast<double>(p.pzz) + 2.0 * p.pzb + p.pbb)));
    view.bias[target] = p.b;
    view.observations[target] = p.count;
  }
}

void DepthFilter::reanchor(const DepthFilterFrame& frame, DepthFilterStats& stats) {
  const std::size_t n = state_.size();
  double R[9], t[3];
  relative(frame.pose, R, t);
  // z-buffer of *covering* sources only (those that round to the pixel), so a
  // nearer surface does not claim the neighbouring pixels it does not cover.
  project(R, t, false);
  const double translation_ratio = std::isfinite(frame.translation_sigma_ratio) ? frame.translation_sigma_ratio : 0.0;
  const double translation_sigma = translation_ratio * std::exp(median_z_);
  const double rotation_degrees = std::isfinite(frame.rotation_sigma_degrees) ? frame.rotation_sigma_degrees : 0.0;
  const double focal_px = K_.fx * 0.5 * (geometry_.scale_x + geometry_.scale_y);  // output px per unit x/z
  // The VO's rotation sigma is optimistic (landmarks taken as exact): floor it.
  const double rotation_px = std::max(config_.min_pointing_sigma_px,
                                      K_.fx * rotation_degrees * kDegree * 0.5 * (geometry_.scale_x + geometry_.scale_y));
  select_sources();
  const auto state_at = [&](int u, int v) { return state_[static_cast<std::size_t>(v) * width_ + u].z; };
  int carried = 0;
  for (std::size_t target = 0; target < n; ++target) {
    Pixel& q = next_[target];
    q = Pixel{};
    const std::int32_t source = best_source_[target];
    if (source < 0) continue;
    const Pixel& p = state_[static_cast<std::size_t>(source)];
    const float z_target = resampled(static_cast<std::size_t>(source), target);
    const double z1 = std::exp(static_cast<double>(z_target));
    const double J = (z1 - t[2]) / z1;  // d log z1 / d log z0
    const int su = static_cast<int>(static_cast<std::size_t>(source) % width_);
    const int sv = static_cast<int>(static_cast<std::size_t>(source) / width_);
    const double g = std::max(0.0, gradient(state_at, su, sv, width_, height_) - config_.gradient_floor);
    // Translation-scale uncertainty: a depth change along the view, and a
    // lateral misregistration.
    const double scale_depth = config_.scale_sigma * std::abs(t[2]) / z1;
    const double scale_px = config_.scale_sigma * focal_px * std::hypot(t[0], t[1]) / z1;
    const double position = std::sqrt(best_distance_[target]) + std::hypot(rotation_px, scale_px);
    const double q_z = (translation_sigma / z1) * (translation_sigma / z1) + scale_depth * scale_depth +
                       g * g * position * position;
    // b moves with the same Jacobian as z: the network is assumed to see the
    // depth change as the truth does, so network maps alone carry no
    // triangulation (a constant log bias would let depth ratios between views
    // pin down absolute depth, which real networks do not support).
    q.z = z_target;
    q.b = static_cast<float>(J * p.b);
    q.pzz = static_cast<float>(J * J * p.pzz + q_z);
    q.pzb = static_cast<float>(J * J * p.pzb);
    q.pbb = static_cast<float>(J * J * p.pbb);
    q.raw = p.raw == p.raw ? p.raw + (q.z - p.z) : kNaN;
    q.count = p.count;
    q.outliers = p.outliers;
    ++carried;
  }
  state_.swap(next_);
  stats.carried = carried;
}

void DepthFilter::add_offset(std::size_t pixel, double value) {
  const std::size_t u = pixel % static_cast<std::size_t>(width_), v = pixel / static_cast<std::size_t>(width_);
  const std::size_t cx = std::min<std::size_t>(kAlignX - 1, u * kAlignX / static_cast<std::size_t>(width_));
  const std::size_t cy = std::min<std::size_t>(kAlignY - 1, v * kAlignY / static_cast<std::size_t>(height_));
  offsets_.push_back(value);
  cell_offsets_[cy * kAlignX + cx].push_back(value);
}

void DepthFilter::fit_alignment(DepthFilterStats& stats) {
  // The frame's actual measurement noise: robust spread (MAD) of the
  // innovations over pixels seen before. The network's frame-to-frame noise
  // after transport is several times the modelled independent part; gating
  // against the model alone freezes, gates and resets pixels at random
  // (speckle, trails).
  frame_noise_ = 0;
  if (offsets_.size() >= 100) {
    const double centre = median(offsets_);
    deviations_.clear();
    for (const double o : offsets_) deviations_.push_back(std::abs(o - centre));
    frame_noise_ = 1.4826 * median(deviations_);
  }
  stats.frame_noise = frame_noise_;
  const bool fit = config_.align_frames && offsets_.size() >= 100;
  const double global = fit ? median(offsets_) : 0.0;
  // High-pass: only the deviation from the running mean is removed, so the
  // frame-to-frame wobble goes but persistent differences reach the state
  // (removing those too lets the state drift away from the network).
  const double a = 1.0 / std::max(1.0, config_.align_memory_frames);
  for (std::size_t c = 0; c < alignment_.size(); ++c) {
    auto& cell = cell_offsets_[c];
    const double fitted = fit && cell.size() >= config_.align_min_samples ? median(cell) : global;
    cell.clear();
    if (!fit) {
      alignment_[c] = 0;
      continue;
    }
    alignment_mean_[c] = alignment_started_ ? alignment_mean_[c] + a * (fitted - alignment_mean_[c]) : fitted;
    alignment_[c] = static_cast<float>(fitted - alignment_mean_[c]);
  }
  if (fit) alignment_started_ = true;
  offsets_.clear();
  stats.scale_offset = global;
}

float DepthFilter::alignment_at(std::size_t pixel) const {
  // Bilinear between cell centres (clamped at the borders).
  const float u = static_cast<float>(pixel % static_cast<std::size_t>(width_)) + 0.5F;
  const float v = static_cast<float>(pixel / static_cast<std::size_t>(width_)) + 0.5F;
  const float x = std::clamp(u * kAlignX / static_cast<float>(width_) - 0.5F, 0.0F, static_cast<float>(kAlignX - 1));
  const float y = std::clamp(v * kAlignY / static_cast<float>(height_) - 0.5F, 0.0F, static_cast<float>(kAlignY - 1));
  const int x0 = std::min(static_cast<int>(x), static_cast<int>(kAlignX) - 2), y0 = std::min(static_cast<int>(y), static_cast<int>(kAlignY) - 2);
  const float wx = x - static_cast<float>(x0), wy = y - static_cast<float>(y0);
  const auto at = [&](int cx, int cy) { return alignment_[static_cast<std::size_t>(cy) * kAlignX + static_cast<std::size_t>(cx)]; };
  return (at(x0, y0) * (1 - wx) + at(x0 + 1, y0) * wx) * (1 - wy) + (at(x0, y0 + 1) * (1 - wx) + at(x0 + 1, y0 + 1) * wx) * wy;
}

void DepthFilter::apply_gain(Pixel& p, double kz, double kb, double innovation) const {
  if (config_.estimate_bias) {
    p.z = static_cast<float>(p.z + kz * innovation);
    p.b = static_cast<float>(p.b + kb * innovation);
  } else {
    // b is not observable from network maps alone: the whole innovation goes
    // to z (the mean of a scalar filter); the covariance still carries b.
    p.z = static_cast<float>(p.z + (kz + kb) * innovation);
  }
}

void DepthFilter::update_in_place(DepthFilterStats& stats) {
  const double frac2 = config_.independent_fraction * config_.independent_fraction;
  const auto& m = measurement_;
  for (std::size_t i = 0; i < state_.size(); ++i)
    if (m.log_depth[i] == m.log_depth[i] && state_[i].valid() && state_[i].count >= 2)
      add_offset(i, m.log_depth[i] - (state_[i].z + state_[i].b));
  fit_alignment(stats);
  for (std::size_t i = 0; i < state_.size(); ++i) {
    if (m.log_depth[i] != m.log_depth[i]) continue;
    const float measured = m.log_depth[i] - alignment_at(i);
    Pixel& p = state_[i];
    const double total = m.sigma[i];
    const double r = std::max(frac2 * total * total, noise_floor2());  // independent part: the measurement noise
    const double b0 = (1 - frac2) * total * total;  // shared part: the bias prior
    const auto initialise = [&] {
      p = Pixel{static_cast<float>(measured), 0, static_cast<float>(b0 + r), static_cast<float>(-b0),
                static_cast<float>(b0), measured, 1, 0};
    };
    if (!p.valid()) {
      initialise();
      ++stats.initialised;
      continue;
    }
    const double s = static_cast<double>(p.pzz) + 2.0 * p.pzb + p.pbb + r;
    const double innovation = measured - (static_cast<double>(p.z) + p.b);
    p.raw = measured;
    if (innovation * innovation > config_.gate_sigmas * config_.gate_sigmas * s) {
      ++stats.gated;
      if (++p.outliers >= config_.reset_after) {
        initialise();
        ++stats.resets;
      }
      continue;
    }
    const double hz = static_cast<double>(p.pzz) + p.pzb, hb = static_cast<double>(p.pzb) + p.pbb;
    const double kz = hz / s, kb = hb / s;
    apply_gain(p, kz, kb, innovation);
    p.pzz = static_cast<float>(p.pzz - kz * hz);
    p.pzb = static_cast<float>(p.pzb - kz * hb);
    p.pbb = static_cast<float>(p.pbb - kb * hb);
    p.outliers = 0;
    if (p.count < std::numeric_limits<std::uint16_t>::max()) ++p.count;
    ++stats.updated;
  }
}

void DepthFilter::update_anchored(const DepthFilterFrame& frame, DepthFilterStats& stats) {
  decay();
  double R[9], t[3];
  relative(frame.pose, R, t);
  project(R, t, true);
  const double translation_ratio = std::isfinite(frame.translation_sigma_ratio) ? frame.translation_sigma_ratio : 0.0;
  const double translation_sigma = translation_ratio * std::exp(median_z_);
  const double rotation_degrees = std::isfinite(frame.rotation_sigma_degrees) ? frame.rotation_sigma_degrees : 0.0;
  const double focal_px = K_.fx * 0.5 * (geometry_.scale_x + geometry_.scale_y);  // output px per unit x/z
  // The VO's rotation sigma is optimistic (landmarks taken as exact): floor it.
  const double rotation_px = std::max(config_.min_pointing_sigma_px,
                                      K_.fx * rotation_degrees * kDegree * 0.5 * (geometry_.scale_x + geometry_.scale_y));
  const double frac2 = config_.independent_fraction * config_.independent_fraction;
  const auto& m = measurement_;
  for (std::size_t i = 0; i < state_.size(); ++i) {
    Pixel& p = state_[i];
    const float predicted = target_z_[i];
    if (!p.valid() || predicted != predicted) continue;
    const float u = target_u_[i], v = target_v_[i];
    const int nu = static_cast<int>(std::lround(u)), nv = static_cast<int>(std::lround(v));
    if (nu < 0 || nv < 0 || nu >= width_ || nv >= height_) continue;
    const std::size_t nearest = static_cast<std::size_t>(nv) * width_ + nu;
    if (predicted > zbuffer_[nearest] + config_.visibility_tolerance) {
      ++stats.occluded;
      continue;
    }
    // Sample the measurement: bilinear in log depth unless it spans an edge.
    float measured = m.log_depth[nearest];
    const int u0 = static_cast<int>(std::floor(u)), v0 = static_cast<int>(std::floor(v));
    if (u0 >= 0 && v0 >= 0 && u0 + 1 < width_ && v0 + 1 < height_) {
      const std::size_t k = static_cast<std::size_t>(v0) * width_ + u0;
      const float a = m.log_depth[k], b = m.log_depth[k + 1], c = m.log_depth[k + width_],
                  d = m.log_depth[k + width_ + 1];
      if (a == a && b == b && c == c && d == d &&
          std::max(std::max(a, b), std::max(c, d)) - std::min(std::min(a, b), std::min(c, d)) <=
              config_.edge_log_ratio) {
        const float wu = u - static_cast<float>(u0), wv = v - static_cast<float>(v0);
        measured = (a * (1 - wu) + b * wu) * (1 - wv) + (c * (1 - wu) + d * wu) * wv;
      }
    }
    if (measured != measured) continue;
    sampled_[i] = measured;
    nearest_[i] = static_cast<std::int32_t>(nearest);
    if (p.count >= 2) add_offset(nearest, measured - (predicted + p.b));
  }
  fit_alignment(stats);
  for (std::size_t i = 0; i < state_.size(); ++i) {
    if (sampled_[i] != sampled_[i]) continue;
    Pixel& p = state_[i];
    const float predicted = target_z_[i];
    const auto nearest = static_cast<std::size_t>(nearest_[i]);
    const float measured = sampled_[i] - alignment_at(nearest);
    sampled_[i] = kNaN;
    const double z1 = std::exp(static_cast<double>(predicted));
    const double J = (z1 - t[2]) / z1;  // d log z_frame / d log z_anchor
    const double total = m.sigma[nearest];
    const double g = m.gradient[nearest];
    // The frame's pose error as measurement noise: depth along the ray, and
    // where the map has a gradient, the pixel error from rotation.
    const double scale_depth = config_.scale_sigma * std::abs(t[2]) / z1;
    const double scale_px = config_.scale_sigma * focal_px * std::hypot(t[0], t[1]) / z1;
    const double r = std::max(frac2 * total * total, noise_floor2()) + (translation_sigma / z1) * (translation_sigma / z1) +
                     scale_depth * scale_depth + g * g * (rotation_px * rotation_px + scale_px * scale_px);
    const double s = J * J * p.pzz + 2.0 * J * p.pzb + p.pbb + r;
    const double innovation = measured - (static_cast<double>(predicted) + p.b);
    // The raw measurement in anchor log depth (flicker reference).
    const auto anchored = static_cast<float>(p.z + (measured - predicted) / J);
    const bool flat = p.raw == p.raw && g < config_.flicker_max_gradient;
    if (flat) flicker_raw_.push_back(std::abs(anchored - p.raw));
    p.raw = anchored;
    const float before = p.z;
    if (innovation * innovation > config_.gate_sigmas * config_.gate_sigmas * s) {
      ++stats.gated;
      if (++p.outliers >= config_.reset_after) {
        const double b0 = (1 - frac2) * total * total, r0 = frac2 * total * total;
        p = Pixel{anchored, 0, static_cast<float>(b0 + r0), static_cast<float>(-b0), static_cast<float>(b0),
                  anchored, 1, 0};
        ++stats.resets;
      }
    } else {
      const double hz = J * p.pzz + p.pzb, hb = J * p.pzb + p.pbb;
      const double kz = hz / s, kb = hb / s;
      apply_gain(p, kz, kb, innovation);
      p.pzz = static_cast<float>(p.pzz - kz * hz);
      p.pzb = static_cast<float>(p.pzb - kz * hb);
      p.pbb = static_cast<float>(p.pbb - kb * hb);
      p.outliers = 0;
      if (p.count < std::numeric_limits<std::uint16_t>::max()) ++p.count;
      ++stats.updated;
    }
    if (flat) flicker_fused_.push_back(std::abs(p.z - before));
  }
}

void DepthFilter::finish(const DepthFilterFrame& frame, DepthFilterStats& stats) {
  std::vector<double> sigmas, sums, depths;
  for (std::size_t i = 0; i < state_.size(); ++i) {
    const Pixel& p = state_[i];
    if (!p.valid()) continue;
    ++stats.valid;
    sigmas.push_back(std::sqrt(p.pzz));
    sums.push_back(std::sqrt(std::max(0.0, static_cast<double>(p.pzz) + 2.0 * p.pzb + p.pbb)));
    if (i % 7 == 0) depths.push_back(p.z);
  }
  stats.flicker_pixels = static_cast<int>(flicker_raw_.size());
  stats.flicker_raw = median(flicker_raw_);
  stats.flicker_fused = median(flicker_fused_);
  flicker_raw_.clear();
  flicker_fused_.clear();
  stats.median_sigma = median(sigmas);
  stats.median_sum_sigma = median(sums);
  median_z_ = median(depths);
  segment_ = frame.segment;
  has_state_ = stats.valid > 0;
}

DepthFilterStats DepthFilter::process(const DepthFilterFrame& frame) {
  DepthFilterStats stats;
  const DepthMap& depth = *frame.depth;
  stats.frame_index = depth.frame_index;
  if (!same_geometry(depth, frame.K, frame.k1)) set_geometry(depth, frame.K, frame.k1);
  if (has_state_ && frame.segment != segment_) clear();
  if (std::isfinite(frame.log_scale))
    log_scale_ = std::isfinite(log_scale_) ? log_scale_ + config_.scale_smoothing * (frame.log_scale - log_scale_)
                                          : frame.log_scale;
  prepare(frame);
  if (!has_state_ || frame.keyframe) {
    if (has_state_) {
      decay();
      reanchor(frame, stats);
    }
    update_in_place(stats);
    anchor_pose_ = frame.pose;
    frame_index_ = depth.frame_index;
    stats.reanchored = true;
  } else {
    update_anchored(frame, stats);
  }
  finish(frame, stats);
  return stats;
}

DepthMap DepthFilter::to_depth_map(double log_scale, const ScaleGrid* grid) const {
  DepthMap map = geometry_;
  map.frame_index = frame_index_;
  map.metres.assign(state_.size(), 0.0F);
  if (!has_state_) return map;
  for (std::size_t i = 0; i < state_.size(); ++i) {
    if (!state_[i].valid()) continue;
    const double L = grid ? scale_grid_at(*grid, map.source_width, map.source_height, source_x_[i], source_y_[i]) - log_scale
                          : 0.0;
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
