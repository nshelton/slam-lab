// Synthetic checks for field-of-view self-calibration (CPU).
#include "slam_native/focal_estimation.hpp"

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

using namespace slam_native;
namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

// Inverse of undistort_point: the distorted pixel whose undistortion is (x, y).
void distort(double& x, double& y, int width, int height, double k1) {
  if (k1 == 0) return;
  const double cx = 0.5 * (width - 1), cy = 0.5 * (height - 1);
  const double half_diagonal = 0.5 * std::hypot(width, height);
  const double dx = x - cx, dy = y - cy, ru = std::hypot(dx, dy) / half_diagonal;
  if (ru < 1e-12) return;
  const double discriminant = 1 - 4 * k1 * ru * ru;
  if (discriminant < 0) { x = y = -1; return; }  // outside the lens model
  const double rd = (1 - std::sqrt(discriminant)) / (2 * k1 * ru);
  x = cx + dx * rd / ru;
  y = cy + dy * rd / ru;
}

// Camera walks through a box of points at 60 fps. `rotate` scales the yaw and
// pitch swing; zero gives pure translation.
std::vector<TrackedFrame> make_frames(double hfov, double rotate, std::uint32_t seed, double k1 = 0) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0, 1);
  std::normal_distribution<double> noise(0, 0.7);
  const int width = 1920, height = 1080;
  const auto K = CameraIntrinsics::from_horizontal_fov(width, height, hfov);
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 3000; ++i) points.emplace_back(-8 + 16 * u(rng), -3 + 6 * u(rng), 2 + 14 * u(rng));
  std::vector<TrackedFrame> frames;
  for (int f = 0; f < 300; ++f) {
    const double s = f / 60.0;
    const Eigen::Vector3d c(0.6 * std::sin(0.7 * s), 0.05 * std::sin(2 * s), 0.8 * s);
    const double yaw = rotate * 0.4 * std::sin(0.9 * s), pitch = rotate * 0.1 * std::sin(1.7 * s);
    const Eigen::Matrix3d R = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY()) *
                               Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX())).toRotationMatrix().transpose();
    TrackedFrame frame{static_cast<std::uint64_t>(f), static_cast<std::int64_t>(f * 16666667LL), width, height, {}, 0, {}};
    for (std::size_t i = 0; i < points.size(); ++i) {
      const Eigen::Vector3d xc = R * (points[i] - c);
      if (xc.z() < 0.3) continue;
      double x = K.fx * xc.x() / xc.z() + K.cx, y = K.fy * xc.y() / xc.z() + K.cy;
      if (std::abs(x - K.cx) > 3 * width || std::abs(y - K.cy) > 3 * height) continue;
      distort(x, y, width, height, k1);
      x += noise(rng);
      y += noise(rng);
      if (x < 0 || y < 0 || x >= width || y >= height) continue;
      frame.observations.push_back({i, static_cast<float>(x), static_cast<float>(y)});
    }
    frames.push_back(std::move(frame));
  }
  return frames;
}

FocalEstimate run(const std::vector<TrackedFrame>& frames) {
  FocalEstimator estimator;
  for (const auto& frame : frames) estimator.add(frame);
  return estimator.estimate();
}
}  // namespace

int main() {
  try {
    struct Case { double hfov, k1; };
    for (const Case c : {Case{55, 0}, Case{80, 0}, Case{110, 0}, Case{90, -0.25}, Case{110, -0.4}, Case{70, 0.1}}) {
      const auto e = run(make_frames(c.hfov, 1.0, 3, c.k1));
      std::cout << "true " << c.hfov << " deg, k1 " << c.k1 << ": estimate " << e.best_hfov_degrees << " deg, k1 "
                << e.best_k1 << " from " << e.pairs_used << "/" << e.pairs_distortion << " pairs (flat "
                << e.pairs_flat << ", homography " << e.pairs_homography << ")\n";
      require(e.pairs_used >= 5, "too few informative pairs");
      require(std::abs(e.best_k1 - c.k1) < 0.02, "k1 estimate off");
      require(std::abs(e.best_hfov_degrees - c.hfov) < 2.0, "field of view estimate off");
    }
    // Pure translation: E = [t]x for every focal length, so no pair informs.
    const auto flat = run(make_frames(80.0, 0.0, 4));
    std::cout << "pure translation: used " << flat.pairs_used << ", flat " << flat.pairs_flat << '\n';
    require(flat.pairs_used == 0 && flat.pairs_flat > 0, "pure translation must be uninformative");
    // ...but it still constrains distortion.
    const auto flat_distorted = run(make_frames(80.0, 0.0, 4, -0.3));
    std::cout << "pure translation, k1 -0.3: k1 estimate " << flat_distorted.best_k1 << '\n';
    require(std::abs(flat_distorted.best_k1 + 0.3) < 0.02, "k1 from pure translation off");
    // The background worker must reach the same estimate as the synchronous one.
    {
      const auto frames = make_frames(90.0, 1.0, 3, -0.25);
      const auto expected = run(frames);
      BackgroundFocalEstimator background;
      for (const auto& frame : frames) background.add(frame);
      FocalEstimate latest;
      std::uint64_t version = 0;
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
      while (latest.pairs_distortion < expected.pairs_distortion && std::chrono::steady_clock::now() < deadline) {
        background.latest(latest, version);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      require(latest.pairs_distortion == expected.pairs_distortion && latest.best_k1 == expected.best_k1 &&
                  latest.best_hfov_degrees == expected.best_hfov_degrees,
              "background estimator differs from synchronous");
      background.reset();
      const auto reset_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
      while (latest.pairs_distortion != 0 && std::chrono::steady_clock::now() < reset_deadline) {
        background.latest(latest, version);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      require(latest.pairs_distortion == 0, "background reset not published");
      std::cout << "background estimator matches synchronous\n";
    }
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "focal estimation checks passed\n";
  return 0;
}
