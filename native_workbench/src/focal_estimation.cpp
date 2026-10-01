#include "slam_native/focal_estimation.hpp"

#include "vo_geometry.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>

namespace slam_native {
using namespace vo;

namespace {
constexpr double kDegree = 3.14159265358979323846 / 180.0;

double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<std::size_t>(q * (v.size() - 1) + 0.5))];
}

// Grid minimum refined by a parabola through its neighbours.
double refined_minimum(const std::vector<double>& x, const double* y) {
  const auto n = x.size();
  const auto k = static_cast<std::size_t>(std::min_element(y, y + n) - y);
  if (k == 0 || k + 1 >= n) return x[k];
  const double denominator = y[k - 1] - 2 * y[k] + y[k + 1];
  if (denominator <= 0) return x[k];
  const double offset = 0.5 * (y[k - 1] - y[k + 1]) / denominator;
  return x[k] + std::clamp(offset, -1.0, 1.0) * (x[k + 1] - x[k]);
}

// Least-squares rank-2 F (eight-point) on the selected correspondences.
bool fit_f(const std::vector<Vec2>& a, const std::vector<Vec2>& b, const std::vector<char>& use, Mat3& F) {
  Eigen::Matrix<double, 9, 9> AtA = Eigen::Matrix<double, 9, 9>::Zero();
  int count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!use[i]) continue;
    const Vec3 x1(a[i].x(), a[i].y(), 1), x2(b[i].x(), b[i].y(), 1);
    Eigen::Matrix<double, 9, 1> row;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) row(r * 3 + c) = x2(r) * x1(c);
    AtA += row * row.transpose();
    ++count;
  }
  if (count < 8) return false;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> solver(AtA);
  if (solver.info() != Eigen::Success) return false;
  const Eigen::Matrix<double, 9, 1> f = solver.eigenvectors().col(0);
  Mat3 raw;
  raw << f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8);
  Eigen::JacobiSVD<Mat3> svd(raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d sigma = svd.singularValues();
  sigma(2) = 0;
  F = svd.matrixU() * sigma.asDiagonal() * svd.matrixV().transpose();
  return F.allFinite();
}

// Squared Sampson distance (same units as the coordinates).
double sampson2(const Mat3& F, const Vec2& a, const Vec2& b) {
  const Vec3 x1(a.x(), a.y(), 1), x2(b.x(), b.y(), 1);
  const Vec3 Fx1 = F * x1, Ftx2 = F.transpose() * x2;
  const double value = x2.dot(Fx1);
  const double denominator = Fx1.x() * Fx1.x() + Fx1.y() * Fx1.y() + Ftx2.x() * Ftx2.x() + Ftx2.y() * Ftx2.y();
  return denominator > 0 ? value * value / denominator : INFINITY;
}
}  // namespace

std::vector<double> FocalEstimate::cost_at(double k1) const {
  const std::size_t columns = hfov_degrees.size();
  if (k1_values.empty() || cost_table.size() != k1_values.size() * columns || pairs_used == 0) return {};
  const double step = k1_values.size() > 1 ? k1_values[1] - k1_values[0] : 1.0;
  const double position = std::clamp((k1 - k1_values.front()) / step, 0.0, double(k1_values.size() - 1));
  const auto row = std::min(static_cast<std::size_t>(position), k1_values.size() - 1);
  const auto next = std::min(row + 1, k1_values.size() - 1);
  const double t = position - double(row);
  std::vector<double> curve(columns);
  for (std::size_t c = 0; c < columns; ++c)
    curve[c] = (1 - t) * cost_table[row * columns + c] + t * cost_table[next * columns + c];
  return curve;
}

double FocalEstimate::best_hfov_at(double k1) const {
  const auto curve = cost_at(k1);
  return curve.empty() ? 0.0 : refined_minimum(hfov_degrees, curve.data());
}

FocalEstimator::FocalEstimator(FocalEstimatorConfig config) : config_(config) { reset(); }

void FocalEstimator::reset() {
  frames_.clear();
  have_pair_ = false;
  last_pair_frame_ = 0;
  const int width = estimate_.width, height = estimate_.height;
  estimate_ = {};
  estimate_.width = width;
  estimate_.height = height;
  for (double h = config_.min_hfov_degrees; h <= config_.max_hfov_degrees + 1e-9; h += config_.hfov_step_degrees)
    estimate_.hfov_degrees.push_back(h);
  for (double k = config_.min_k1; k <= config_.max_k1 + 1e-9; k += config_.k1_step)
    estimate_.k1_values.push_back(std::abs(k) < 1e-9 ? 0.0 : k);
  residual_sum_.assign(estimate_.k1_values.size(), 0.0);
  cost_sum_.assign(estimate_.k1_values.size() * estimate_.hfov_degrees.size(), 0.0);
  pair_min_.clear();
}

void FocalEstimator::restart_tracks() {
  frames_.clear();
  have_pair_ = false;
}

bool FocalEstimator::add(const TrackedFrame& frame) {
  if (frame.width <= 0) return false;
  if (estimate_.width != frame.width || estimate_.height != frame.height) {
    estimate_.width = frame.width;
    estimate_.height = frame.height;
    reset();
  }
  Stored stored{frame.frame_index, {}};
  stored.points.reserve(frame.observations.size());
  for (const auto& o : frame.observations) stored.points.emplace(o.track_id, std::make_pair(o.x, o.y));
  while (!frames_.empty() && (frames_.front().frame_index + config_.max_gap_frames < frame.frame_index ||
                              frames_.front().frame_index >= frame.frame_index))
    frames_.pop_front();

  bool added = false;
  if (!have_pair_ || frame.frame_index >= last_pair_frame_ + config_.pair_stride_frames) {
    // Most recent earlier frame with enough motion (shortest usable gap).
    const double needed = config_.min_displacement_fraction * frame.width;
    for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
      std::vector<double> displacement;
      for (const auto& [id, p] : it->points) {
        auto q = stored.points.find(id);
        if (q != stored.points.end())
          displacement.push_back(std::hypot(q->second.first - p.first, q->second.second - p.second));
      }
      if (static_cast<int>(displacement.size()) < config_.min_common_tracks) break;  // older is worse
      if (percentile(displacement, 0.5) < needed) continue;
      add_pair(*it, stored);
      have_pair_ = true;
      last_pair_frame_ = frame.frame_index;
      added = true;
      break;
    }
  }
  frames_.push_back(std::move(stored));
  return added;
}

void FocalEstimator::add_pair(const Stored& first, const Stored& second) {
  // Work in pixels centred on the image and divided by the width: a camera
  // with horizontal FOV h then has focal a = 0.5 / tan(h / 2).
  const int width = estimate_.width, height = estimate_.height;
  const double w = width;
  const double cx = 0.5 * (width - 1), cy = 0.5 * (height - 1);
  std::vector<std::pair<float, float>> pa, pb;  // raw (distorted) pixels
  for (const auto& [id, p] : first.points) {
    auto q = second.points.find(id);
    if (q == second.points.end()) continue;
    pa.push_back(p);
    pb.push_back(q->second);
  }
  const auto to_unit = [&](float x, float y) { return Vec2((x - cx) / w, (y - cy) / w); };
  std::vector<Vec2> a(pa.size()), b(pb.size());
  for (std::size_t i = 0; i < pa.size(); ++i) {
    a[i] = to_unit(pa[i].first, pa[i].second);
    b[i] = to_unit(pb[i].first, pb[i].second);
  }
  const auto seed = static_cast<std::uint32_t>(second.frame_index * 2654435761U);
  const auto F0 = ransac_fundamental(a, b, config_.candidate_threshold_px / w, config_.ransac_iterations, seed);
  if (F0.inlier_count < config_.min_inliers) {
    ++estimate_.pairs_weak;
    return;
  }
  const double tight = config_.residual_threshold_px / w;
  const auto Ftight = ransac_fundamental(a, b, tight, config_.ransac_iterations, seed + 2);
  const auto H = ransac_homography(a, b, tight, config_.ransac_iterations, seed + 1);
  if (H.inlier_count > config_.homography_ratio * std::max(Ftight.inlier_count, 1)) {
    ++estimate_.pairs_homography;
    return;
  }

  const auto& k1s = estimate_.k1_values;
  const auto& grid = estimate_.hfov_degrees;
  std::vector<double> residual(k1s.size(), 0.0);
  std::vector<double> curves(k1s.size() * grid.size(), 0.0);
  std::vector<float> minima(k1s.size());
  const double tight2 = tight * tight;
  std::vector<Vec2> ua(a.size()), ub(b.size());
  std::vector<double> magnification2(a.size());  // (undistorted / image distance)^2 near the pair
  const double half_diagonal2 = 0.25 * (w * w + double(height) * height);
  // Local scale of the undistortion at radius^2 r2 (geometric mean of the
  // radial and tangential stretch). Residuals are divided by it so they stay
  // in image pixels: otherwise k1 > 0, which shrinks the whole point cloud,
  // would look like a better fit.
  const auto scale = [&](float x, float y, double k1) {
    const double r2 = ((x - cx) * (x - cx) + (y - cy) * (y - cy)) / half_diagonal2;
    const double d = 1 + k1 * r2;
    return std::sqrt(std::abs(1 - k1 * r2) / (d * d * d));
  };
  for (std::size_t j = 0; j < k1s.size(); ++j) {
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double m = 0.5 * (scale(pa[i].first, pa[i].second, k1s[j]) + scale(pb[i].first, pb[i].second, k1s[j]));
      magnification2[i] = m * m;
      float xa = pa[i].first, ya = pa[i].second, xb = pb[i].first, yb = pb[i].second;
      undistort_point(xa, ya, width, height, k1s[j]);
      undistort_point(xb, yb, width, height, k1s[j]);
      ua[i] = to_unit(xa, ya);
      ub[i] = to_unit(xb, yb);
    }
    // Fit on the loose candidates, then refit on those within the tight
    // threshold, so the residual measures this k1's best achievable fit.
    std::vector<char> use(F0.inliers.begin(), F0.inliers.end());
    Mat3 F;
    bool ok = fit_f(ua, ub, use, F);
    for (int pass = 0; ok && pass < 2; ++pass) {
      for (std::size_t i = 0; i < a.size(); ++i)
        use[i] = F0.inliers[i] && sampson2(F, ua[i], ub[i]) / magnification2[i] < 4 * tight2;
      ok = fit_f(ua, ub, use, F);
    }
    if (!ok) {
      residual[j] = tight2;
      continue;
    }
    double sum = 0;
    int count = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (!F0.inliers[i]) continue;
      sum += std::min(sampson2(F, ua[i], ub[i]) / magnification2[i], tight2);
      ++count;
    }
    residual[j] = sum / count;
    for (std::size_t k = 0; k < grid.size(); ++k) {
      const double focal = 0.5 / std::tan(0.5 * grid[k] * kDegree);
      const Eigen::Vector3d diagonal(focal, focal, 1.0);
      const Mat3 E = diagonal.asDiagonal() * F * diagonal.asDiagonal();
      const Eigen::Vector3d s = Eigen::JacobiSVD<Mat3>(E).singularValues();
      curves[j * grid.size() + k] = s(0) + s(1) > 0 ? (s(0) - s(1)) / (s(0) + s(1)) : 1.0;
    }
    minima[j] = static_cast<float>(refined_minimum(grid, &curves[j * grid.size()]));
  }
  for (std::size_t j = 0; j < k1s.size(); ++j) residual_sum_[j] += residual[j];
  ++estimate_.pairs_distortion;

  // FOV information: judge the curve at this pair's own best k1.
  const auto own = static_cast<std::size_t>(std::min_element(residual.begin(), residual.end()) - residual.begin());
  const auto row = curves.begin() + static_cast<std::ptrdiff_t>(own * grid.size());
  const auto [low, high] = std::minmax_element(row, row + static_cast<std::ptrdiff_t>(grid.size()));
  if (*high - *low < config_.min_curve_depth) {
    ++estimate_.pairs_flat;
  } else {
    for (std::size_t i = 0; i < curves.size(); ++i) cost_sum_[i] += curves[i];
    pair_min_.push_back(std::move(minima));
    ++estimate_.pairs_used;
  }
  refresh();
}

void FocalEstimator::refresh() {
  auto& e = estimate_;
  const auto& k1s = e.k1_values;
  e.k1_residual_px.resize(k1s.size());
  for (std::size_t j = 0; j < k1s.size(); ++j)
    e.k1_residual_px[j] = std::sqrt(residual_sum_[j] / e.pairs_distortion) * e.width;
  e.best_k1 = refined_minimum(k1s, e.k1_residual_px.data());
  if (e.pairs_used == 0) return;
  e.cost_table.resize(cost_sum_.size());
  for (std::size_t i = 0; i < cost_sum_.size(); ++i) e.cost_table[i] = cost_sum_[i] / e.pairs_used;
  e.cost = e.cost_at(e.best_k1);
  e.best_hfov_degrees = refined_minimum(e.hfov_degrees, e.cost.data());
  e.best_focal_px = 0.5 * e.width / std::tan(0.5 * e.best_hfov_degrees * kDegree);
  const double step = k1s.size() > 1 ? k1s[1] - k1s[0] : 1.0;
  const auto nearest = static_cast<std::size_t>(
      std::clamp(std::lround((e.best_k1 - k1s.front()) / step), 0L, static_cast<long>(k1s.size() - 1)));
  e.pair_best_degrees.clear();
  for (const auto& minima : pair_min_) e.pair_best_degrees.push_back(minima[nearest]);
  e.pair_p25_degrees = percentile(e.pair_best_degrees, 0.25);
  e.pair_median_degrees = percentile(e.pair_best_degrees, 0.5);
  e.pair_p75_degrees = percentile(e.pair_best_degrees, 0.75);
}

BackgroundFocalEstimator::BackgroundFocalEstimator(FocalEstimatorConfig config)
    : estimator_(config), worker_([this] { run(); }) {}

BackgroundFocalEstimator::~BackgroundFocalEstimator() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  wake_.notify_one();
  worker_.join();
}

void BackgroundFocalEstimator::push(Item item) {
  {
    std::lock_guard lock(mutex_);
    constexpr std::size_t kMaxQueued = 600;
    if (item.command == Command::frame && queue_.size() >= kMaxQueued) {
      auto oldest = std::find_if(queue_.begin(), queue_.end(), [](const Item& i) { return i.command == Command::frame; });
      if (oldest != queue_.end()) queue_.erase(oldest);
    }
    queue_.push_back(std::move(item));
  }
  wake_.notify_one();
}

void BackgroundFocalEstimator::add(TrackedFrame frame) { push({Command::frame, std::move(frame)}); }
void BackgroundFocalEstimator::reset() { push({Command::reset, {}}); }
void BackgroundFocalEstimator::restart_tracks() { push({Command::restart_tracks, {}}); }

bool BackgroundFocalEstimator::latest(FocalEstimate& out, std::uint64_t& version) const {
  std::lock_guard lock(mutex_);
  if (version == version_) return false;
  out = published_;
  version = version_;
  return true;
}

void BackgroundFocalEstimator::run() {
  for (;;) {
    Item item;
    {
      std::unique_lock lock(mutex_);
      wake_.wait(lock, [&] { return stop_ || !queue_.empty(); });
      if (stop_) return;
      item = std::move(queue_.front());
      queue_.pop_front();
    }
    bool changed = false;
    switch (item.command) {
      case Command::frame: changed = estimator_.add(item.frame); break;
      case Command::reset: estimator_.reset(); changed = true; break;
      case Command::restart_tracks: estimator_.restart_tracks(); break;
    }
    if (changed) {
      std::lock_guard lock(mutex_);
      published_ = estimator_.estimate();
      ++version_;
    }
  }
}

}  // namespace slam_native
