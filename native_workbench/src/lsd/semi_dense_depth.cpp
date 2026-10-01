#include "lsd/semi_dense_depth.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>

namespace slam_native::lsd {
namespace {
constexpr int kBorder = 3;  // stereo needs +-2 samples around the pixel
constexpr int kRadius = 2;  // regularisation neighbourhood: 5x5
}  // namespace

SemiDenseDepth::SemiDenseDepth(const Level& keyframe, const DepthFilterConfig& config, const ImageF* mask)
    : keyframe_(keyframe), config_(config), width_(keyframe.image.width), height_(keyframe.image.height),
      pixels_(static_cast<std::size_t>(width_) * height_), mask_(pixels_.size(), 1) {
  if (mask && mask->width == width_ && mask->height == height_)
    for (std::size_t i = 0; i < mask_.size(); ++i) mask_[i] = mask->data[i] > 0.5F ? 1 : 0;
}

bool SemiDenseDepth::usable(int x, int y) const {
  if (!mask_[index(x, y)]) return false;
  const double gx = keyframe_.gx.at(x, y), gy = keyframe_.gy.at(x, y);
  return gx * gx + gy * gy >= config_.gradient_threshold * config_.gradient_threshold;
}

float SemiDenseDepth::validity_cap(int x, int y) const {
  const float g = std::hypot(keyframe_.gx.at(x, y), keyframe_.gy.at(x, y));
  return config_.validity_max + config_.validity_max_gradient * g / 255.0F;
}

int SemiDenseDepth::valid_count() const {
  int n = 0;
  for (const auto& p : pixels_) n += p.valid;
  return n;
}

SemiDenseDepth::Search SemiDenseDepth::search(int x, int y, const Level& frame, const SE3& T, double d_min,
                                              double d_max, double prior, float& idepth, float& variance) const {
  const Camera& K = keyframe_.camera;
  const Vec3 q = K.ray(x, y);
  const Vec3 Rq = T.R * q;
  const Vec3& t = T.t;

  // Epipolar direction in the keyframe: from the epipole (the frame's centre
  // projected into the keyframe) through the pixel. Its length is the pixel
  // motion per unit inverse depth, i.e. the baseline in pixels.
  const Vec3 c = -(T.R.transpose() * t);
  Vec2 e{c.z() * (x - K.cx) - K.fx * c.x(), c.z() * (y - K.cy) - K.fy * c.y()};
  const double e_length = e.norm();
  if (e_length < config_.min_epl_length_px) return Search::skipped;
  e /= e_length;
  const double gx = keyframe_.gx.at(x, y), gy = keyframe_.gy.at(x, y);
  const double along = (e.x() * gx + e.y() * gy) * (e.x() * gx + e.y() * gy);
  if (along < config_.min_epl_gradient2) return Search::skipped;
  const double cos2 = along / (gx * gx + gy * gy);
  if (cos2 < config_.min_epl_angle_cos2) return Search::skipped;

  // Five reference samples along the keyframe's epipolar line.
  std::array<double, 5> ref{};
  for (int k = -2; k <= 2; ++k) {
    const double u = x + k * e.x(), v = y + k * e.y();
    if (!keyframe_.image.inside(u, v)) return Search::skipped;
    ref[k + 2] = keyframe_.image.sample(u, v);
  }
  double grad_along_line = 0;
  for (int k = 0; k < 4; ++k) grad_along_line += (ref[k + 1] - ref[k]) * (ref[k + 1] - ref[k]);

  // The frame-side segment between inverse depths d_min (far) and d_max (near).
  if (t.z() < 0) d_max = std::min(d_max, 0.9 * -Rq.z() / t.z());  // stay in front of the frame camera
  if (Rq.z() + t.z() * d_min <= 1e-6 || d_max <= d_min) return Search::skipped;
  const auto project = [&](const Vec3& ray, double d) -> Vec2 { return K.project(T.R * ray + t * d); };
  const Vec2 pa = project(q, d_min), pb = project(q, d_max);
  const double length = (pb - pa).norm();
  if (!(length > 1e-6)) return Search::skipped;
  const Vec2 u = (pb - pa) / length;
  double s0 = 0, s1 = length;
  if (length > config_.max_search_px && prior > 0) {
    const double sc = (project(q, prior) - pa).dot(u);
    s0 = std::clamp(sc - 0.5 * config_.max_search_px, 0.0, length - config_.max_search_px);
    s1 = s0 + config_.max_search_px;
  } else if (length < config_.min_search_px) {
    s0 = 0.5 * (length - config_.min_search_px);
    s1 = s0 + config_.min_search_px;
  }
  s1 = std::min(s1, s0 + 4 * config_.max_search_px);  // full-range searches without a prior

  // The keyframe's one-pixel epipolar step, seen in the frame (scale and sign).
  const double d_mid = prior > 0 ? prior : 0.5 * (d_min + d_max);
  const Vec2 step = project(K.ray(x + e.x(), y + e.y()), d_mid) - project(q, d_mid);
  const double step_length = step.norm();
  if (step_length < 0.5 || step_length > 2.0) return Search::skipped;

  const int n = static_cast<int>(std::floor(s1 - s0)) + 1;
  std::vector<double> errors(static_cast<std::size_t>(n), std::numeric_limits<double>::infinity());
  int best = -1;
  for (int i = 0; i < n; ++i) {
    const Vec2 centre = pa + u * (s0 + i);
    const Vec2 first = centre - 2 * step, last = centre + 2 * step;
    if (!frame.image.inside(first.x(), first.y(), 1) || !frame.image.inside(last.x(), last.y(), 1) ||
        !frame.measured(first.x(), first.y()) || !frame.measured(last.x(), last.y()))
      continue;
    double ssd = 0;
    for (int k = -2; k <= 2; ++k) {
      const Vec2 p = centre + k * step;
      const double r = frame.image.sample(p.x(), p.y()) - ref[k + 2];
      ssd += r * r;
    }
    errors[i] = ssd;
    if (best < 0 || ssd < errors[best]) best = i;
  }
  if (best < 0) return Search::skipped;
  if (errors[best] > config_.max_ssd) return Search::failed;
  double second = std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i)
    if (std::abs(i - best) > 1) second = std::min(second, errors[i]);
  if (second < config_.min_second_ratio * errors[best]) return Search::failed;

  // Subpixel: parabola through the best error and its neighbours.
  double offset = 0;
  if (best > 0 && best + 1 < n && std::isfinite(errors[best - 1]) && std::isfinite(errors[best + 1])) {
    const double a = errors[best - 1], b = errors[best], c2 = errors[best + 1];
    const double curvature = a - 2 * b + c2;
    if (curvature > 0) offset = std::clamp(0.5 * (a - c2) / curvature, -0.5, 0.5);
  }
  // Inverse depth of a frame pixel on the segment (along its dominant axis).
  const auto idepth_at = [&](double s) {
    const Vec2 m = pa + u * s;
    if (std::abs(u.x()) >= std::abs(u.y())) {
      const double xn = (m.x() - K.cx) / K.fx;
      return (Rq.x() - xn * Rq.z()) / (xn * t.z() - t.x());
    }
    const double yn = (m.y() - K.cy) / K.fy;
    return (Rq.y() - yn * Rq.z()) / (yn * t.z() - t.y());
  };
  const double s = s0 + best + offset;
  const double d = idepth_at(s);
  const double alpha = std::abs(idepth_at(s + 0.5) - idepth_at(s - 0.5));  // inverse depth per frame pixel
  if (!std::isfinite(d) || !std::isfinite(alpha)) return Search::skipped;

  // ICCV 2013 eqs. 6, 9, 10: geometric and photometric disparity variance.
  const double sigma_l2 = config_.sigma_epipolar_px * config_.sigma_epipolar_px;
  const double geometric = sigma_l2 / cos2;
  const double photometric = 2 * config_.sigma_intensity * config_.sigma_intensity / (grad_along_line / 4 + 1e-6);
  idepth = static_cast<float>(std::max(d, 1e-5));
  variance = static_cast<float>(alpha * alpha * (geometric + photometric));
  return Search::ok;
}

StereoStats SemiDenseDepth::observe(const Level& frame, const SE3& frame_from_keyframe) {
  const auto start = std::chrono::steady_clock::now();
  StereoStats stats;
  for (int y = kBorder; y < height_ - kBorder; ++y) {
    for (int x = kBorder; x < width_ - kBorder; ++x) {
      if (!usable(x, y)) continue;
      ++stats.candidates;
      DepthHypothesis& h = pixels_[index(x, y)];
      double d_min = 0, d_max = config_.max_idepth, prior = -1;
      if (h.valid) {
        const double sigma = std::sqrt(h.variance_smoothed);
        prior = h.idepth_smoothed;
        d_min = std::max(prior - 2 * sigma, 1e-6);
        d_max = prior + 2 * sigma;
      }
      float d = 0, variance = 0;
      const Search result = search(x, y, frame, frame_from_keyframe, d_min, d_max, prior, d, variance);
      if (result == Search::skipped) {
        ++stats.skipped;
        continue;
      }
      if (result == Search::failed) {
        ++stats.failed;
        if (h.valid) {
          h.validity -= 1;
          h.variance *= static_cast<float>(config_.fail_variance_factor);
          if (h.variance > config_.max_variance) {
            h.valid = false;
            ++stats.removed;
          }
        }
        continue;
      }
      if (!h.valid) {
        h = {d, variance, d, variance, config_.validity_initial, true};
        ++stats.created;
        continue;
      }
      const double diff = d - h.idepth_smoothed;
      if (diff * diff > config_.consistency_sigmas2 * (variance + h.variance_smoothed)) {
        ++stats.inconsistent;
        h.variance *= static_cast<float>(config_.fail_variance_factor);
        if (h.variance > config_.max_variance) {
          h.valid = false;
          ++stats.removed;
        }
        continue;
      }
      // Kalman update (ICCV 2013 eq. 13) after inflating the prior.
      const double prior_var = h.variance * config_.success_variance_factor;
      const double w = variance / (variance + prior_var);
      h.idepth = static_cast<float>((1 - w) * d + w * h.idepth);
      h.variance = static_cast<float>(prior_var * w);
      h.validity = std::min(h.validity + config_.validity_increment, validity_cap(x, y));
      ++stats.matched;
    }
  }
  stats.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return stats;
}

int SemiDenseDepth::propagate(const SemiDenseDepth& previous, const SE3& this_from_previous) {
  const Camera& K = keyframe_.camera;
  int placed = 0;
  for (int y = 0; y < previous.height_; ++y) {
    for (int x = 0; x < previous.width_; ++x) {
      const DepthHypothesis& source = previous.at(x, y);
      if (!source.valid) continue;
      const double d0 = source.idepth_smoothed;
      const Vec3 X = this_from_previous * (previous.keyframe_.camera.ray(x, y) / d0);
      if (X.z() <= 1e-6) continue;
      const Vec2 uv = K.project(X);
      const int u = static_cast<int>(std::lround(uv.x())), v = static_cast<int>(std::lround(uv.y()));
      if (u < 0 || v < 0 || u >= width_ || v >= height_ || !mask_[index(u, v)]) continue;
      // Occlusion and appearance: the point must still look like itself.
      if (!keyframe_.image.inside(uv.x(), uv.y())) continue;
      if (std::abs(keyframe_.image.sample(uv.x(), uv.y()) - previous.keyframe_.image.at(x, y)) >
          config_.max_propagation_intensity_diff)
        continue;
      const double d1 = 1.0 / X.z();
      const double ratio = d1 / d0;
      const auto variance = static_cast<float>(source.variance * ratio * ratio * ratio * ratio +
                                               config_.propagation_variance);
      DepthHypothesis& target = pixels_[index(u, v)];
      if (!target.valid) {
        target = {static_cast<float>(d1), variance, static_cast<float>(d1), variance, source.validity, true};
        ++placed;
        continue;
      }
      const double diff = d1 - target.idepth;
      if (diff * diff > config_.consistency_sigmas2 * (variance + target.variance)) {
        // Two surfaces: keep the nearer one, the other is occluded here.
        if (d1 > target.idepth)
          target = {static_cast<float>(d1), variance, static_cast<float>(d1), variance, source.validity, true};
        continue;
      }
      const double w = variance / (variance + target.variance);
      target.idepth = static_cast<float>((1 - w) * d1 + w * target.idepth);
      target.variance = static_cast<float>(target.variance * w);
      target.idepth_smoothed = target.idepth;
      target.variance_smoothed = target.variance;
      target.validity = std::min(target.validity + source.validity, validity_cap(u, v));
    }
  }
  return placed;
}

int SemiDenseDepth::seed(const ImageF& prior_idepth, double relative_sigma) {
  if (prior_idepth.width != width_ || prior_idepth.height != height_) return 0;
  int seeded = 0;
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      DepthHypothesis& h = pixels_[index(x, y)];
      const float d = prior_idepth.at(x, y);
      if (h.valid || !(d > 0) || !std::isfinite(d) || !usable(x, y)) continue;
      const auto variance = static_cast<float>((relative_sigma * d) * (relative_sigma * d));
      h = {d, variance, d, variance, config_.validity_initial, true};
      ++seeded;
    }
  }
  return seeded;
}

void SemiDenseDepth::regularize(bool fill_holes) {
  // 5x5 validity sums over the current hypotheses.
  std::vector<float> sums(pixels_.size(), 0);
  for (int y = kRadius; y < height_ - kRadius; ++y)
    for (int x = kRadius; x < width_ - kRadius; ++x) {
      float sum = 0;
      for (int dy = -kRadius; dy <= kRadius; ++dy)
        for (int dx = -kRadius; dx <= kRadius; ++dx) {
          const auto& n = pixels_[index(x + dx, y + dy)];
          if (n.valid) sum += n.validity;
        }
      sums[index(x, y)] = sum;
    }
  // Hole filling and outlier removal decide on the same snapshot.
  std::vector<DepthHypothesis> next = pixels_;
  for (int y = kRadius; y < height_ - kRadius; ++y) {
    for (int x = kRadius; x < width_ - kRadius; ++x) {
      const std::size_t i = index(x, y);
      if (pixels_[i].valid) {
        if (sums[i] < config_.validity_sum_keep) next[i].valid = false;
        continue;
      }
      if (!fill_holes || !mask_[i] || sums[i] < config_.validity_sum_create) continue;
      double weighted = 0, information = 0;
      int count = 0;
      for (int dy = -kRadius; dy <= kRadius; ++dy)
        for (int dx = -kRadius; dx <= kRadius; ++dx) {
          const auto& n = pixels_[index(x + dx, y + dy)];
          if (!n.valid) continue;
          weighted += n.idepth / n.variance;
          information += 1.0 / n.variance;
          ++count;
        }
      if (count == 0) continue;
      const auto d = static_cast<float>(weighted / information);
      const auto variance = static_cast<float>(count / information);
      next[i] = {d, variance, d, variance, 0, true};
    }
  }
  // Inverse-variance-weighted mean over statistically compatible neighbours.
  const double k2 = config_.regularize_sigmas * config_.regularize_sigmas;
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      DepthHypothesis& h = next[index(x, y)];
      if (!h.valid) continue;
      double weighted = 0, information = 0;
      for (int dy = -kRadius; dy <= kRadius; ++dy) {
        for (int dx = -kRadius; dx <= kRadius; ++dx) {
          const int nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width_ || ny >= height_) continue;
          const auto& n = next[index(nx, ny)];
          if (!n.valid) continue;
          const double diff = n.idepth - h.idepth;
          if (diff * diff > k2 * (n.variance + h.variance)) continue;
          weighted += n.idepth / n.variance;
          information += 1.0 / n.variance;
        }
      }
      h.idepth_smoothed = static_cast<float>(weighted / information);
      h.variance_smoothed = h.variance;
    }
  }
  pixels_ = std::move(next);
}

InverseDepthMap SemiDenseDepth::tracking_map() const {
  InverseDepthMap map{ImageF(width_, height_), ImageF(width_, height_)};
  for (std::size_t i = 0; i < pixels_.size(); ++i) {
    if (!pixels_[i].valid) continue;
    map.idepth.data[i] = pixels_[i].idepth_smoothed;
    map.variance.data[i] = pixels_[i].variance_smoothed;
  }
  return map;
}

}  // namespace slam_native::lsd
