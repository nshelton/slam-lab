// Synthetic checks for the tracker-agnostic visual odometry (CPU).
#include "slam_native/track_io.hpp"
#include "slam_native/visual_odometry.hpp"
#include "vo_geometry.hpp"  // internal geometry, unit-tested directly

#include <Eigen/Geometry>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>

using namespace slam_native;
namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

struct Scene {
  std::vector<TrackedFrame> frames;
  std::vector<Eigen::Vector3d> centers;   // ground-truth camera centres
  std::vector<Eigen::Matrix3d> rotations; // ground-truth world->camera
};

// Camera walks forward with lateral sway and yaw at 60 fps through a field of
// points. Tracks break on every missed observation (new ID), and include
// independently moving points and gross outliers.
struct Difficulty {
  double noise_px{0.7}, dropout{0.08}, breaks{0.01}, outliers{0.01};
  int movers{80};
};
Scene make_scene(bool translate, int frame_count = 240, std::uint32_t seed = 5, Difficulty difficulty = {}) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0, 1);
  std::normal_distribution<double> noise(0, std::max(difficulty.noise_px, 1e-9));
  const int width = 1920, height = 1080;
  const auto K = CameraIntrinsics::from_horizontal_fov(width, height, 60);
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 2500; ++i) points.emplace_back(-10 + 20 * u(rng), -4 + 8 * u(rng), 5 + 20 * u(rng));
  const int movers = difficulty.movers;
  std::vector<Eigen::Vector3d> mover_velocity;
  for (int i = 0; i < movers; ++i) mover_velocity.emplace_back(0.04 * (u(rng) - 0.5), 0, 0.04 * (u(rng) - 0.5));
  std::vector<std::uint64_t> id(points.size());
  std::uint64_t next_id = 0;
  for (auto& value : id) value = next_id++;
  std::vector<char> seen(points.size(), 0);
  Scene scene;
  for (int f = 0; f < frame_count; ++f) {
    const double s = f / 60.0;
    const Eigen::Vector3d c = translate ? Eigen::Vector3d(1.2 * std::sin(0.8 * s), 0.1 * std::sin(2 * s), 1.1 * s)
                                        : Eigen::Vector3d::Zero();
    const double yaw = 0.25 * std::sin(0.6 * s), pitch = 0.04 * std::sin(1.3 * s);
    const Eigen::Matrix3d Rwc = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY()) *
                                 Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX())).toRotationMatrix();
    const Eigen::Matrix3d R = Rwc.transpose();
    const Eigen::Vector3d t = -R * c;
    scene.centers.push_back(c);
    scene.rotations.push_back(R);
    TrackedFrame frame{static_cast<std::uint64_t>(f), f * 16'666'667LL, width, height, {}};
    for (std::size_t p = 0; p < points.size(); ++p) {
      Eigen::Vector3d X = points[p];
      if (static_cast<int>(p) < movers) X += mover_velocity[p] * f;  // moving "people"
      const Eigen::Vector3d xc = R * X + t;
      bool visible = xc.z() > 0.5;
      double x = 0, y = 0;
      if (visible) {
        x = K.fx * xc.x() / xc.z() + K.cx + noise(rng);
        y = K.fy * xc.y() / xc.z() + K.cy + noise(rng);
        visible = x >= 0 && y >= 0 && x < width && y < height && u(rng) >= difficulty.dropout;
      }
      if (!visible || u(rng) < difficulty.breaks) {  // missed detection or tracker loss -> new ID later
        if (seen[p]) id[p] = next_id++;
        seen[p] = 0;
        if (!visible) continue;
      }
      seen[p] = 1;
      if (u(rng) < difficulty.outliers) { x += 40 * (u(rng) - 0.5); y += 40 * (u(rng) - 0.5); }  // gross outlier
      frame.observations.push_back({id[p], static_cast<float>(x), static_cast<float>(y)});
    }
    scene.frames.push_back(std::move(frame));
  }
  return scene;
}

struct Evaluation {
  int posed{}, segments{}, first_pose{-1};
  double ate_fraction{}, rotation_deg{};
};
Evaluation evaluate(const Scene& scene, const std::vector<TrajectorySample>& trajectory) {
  Evaluation result;
  std::map<int, std::vector<const TrajectorySample*>> by_segment;
  for (const auto& sample : trajectory) by_segment[sample.segment].push_back(&sample);
  result.segments = static_cast<int>(by_segment.size());
  result.posed = static_cast<int>(trajectory.size());
  if (trajectory.empty()) return result;
  result.first_pose = static_cast<int>(trajectory.front().frame_index);
  // Largest segment, similarity-aligned (monocular scale is arbitrary).
  const auto& samples = std::max_element(by_segment.begin(), by_segment.end(),
      [](const auto& a, const auto& b) { return a.second.size() < b.second.size(); })->second;
  Eigen::Matrix3Xd estimated(3, samples.size()), truth(3, samples.size());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto c = samples[i]->pose.center();
    estimated.col(i) = Eigen::Vector3d(c[0], c[1], c[2]);
    truth.col(i) = scene.centers[samples[i]->frame_index];
  }
  const Eigen::Matrix4d S = Eigen::umeyama(estimated, truth, true);
  double squared = 0, length = 0;
  std::vector<double> rotation_errors;
  const Eigen::Matrix3d align = S.topLeftCorner<3, 3>() / std::cbrt(S.topLeftCorner<3, 3>().determinant());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const Eigen::Vector3d aligned = S.topLeftCorner<3, 3>() * estimated.col(i) + S.topRightCorner<3, 1>();
    squared += (aligned - truth.col(i)).squaredNorm();
    if (i) length += (truth.col(i) - truth.col(i - 1)).norm();
    Eigen::Matrix3d R;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) R(r, c) = samples[i]->pose.rotation[r * 3 + c];
    // Estimated camera->world rotation mapped into the ground-truth frame.
    const Eigen::Matrix3d estimated_wc = align * R.transpose();
    const Eigen::Matrix3d truth_wc = scene.rotations[samples[i]->frame_index].transpose();
    rotation_errors.push_back(Eigen::AngleAxisd(estimated_wc.transpose() * truth_wc).angle() * 180 / M_PI);
  }
  result.ate_fraction = std::sqrt(squared / samples.size()) / std::max(length, 1e-9);
  std::sort(rotation_errors.begin(), rotation_errors.end());
  result.rotation_deg = rotation_errors[rotation_errors.size() / 2];
  return result;
}

std::vector<TrajectorySample> run(const std::vector<TrackedFrame>& frames, double* ms = nullptr,
                                  VisualOdometryConfig config = {}) {
  VisualOdometry odometry(config);
  double total = 0;
  const bool verbose = std::getenv("SLAM_VO_VERBOSE") != nullptr;
  for (const auto& frame : frames) {
    const auto r = odometry.process(frame);
    total += r.ms;
    if (verbose && (!r.event.empty() || r.keyframe || r.frame_index % 10 == 0))
      std::cout << "  frame " << r.frame_index << ' ' << to_string(r.state) << " seg " << r.segment
                << " corr " << r.correspondences << " inl " << r.inliers << " map " << r.map_points
                << " kf " << r.keyframes << (r.keyframe ? " KF" : "") << " med " << r.median_reprojection_px
                << " ms " << r.ms << ' ' << r.event << '\n';
  }
  if (ms) *ms = total / frames.size();
  return odometry.trajectory();
}
}  // namespace

// Pose-only optimization must converge to the exact pose from a nearby start.
void pose_optimizer_converges() {
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> u(-1, 1);
  const vo::Intrinsics K{1662.8, 1662.8, 959.5, 539.5};
  vo::SE3 truth{vo::exp_so3(vo::Vec3(0.02, -0.1, 0.03)), vo::Vec3(0.3, -0.1, 0.5)};
  std::vector<vo::PoseObservation> observations;
  for (int i = 0; i < 500; ++i) {
    const vo::Vec3 X(4 * u(rng), 2 * u(rng), 8 + 4 * u(rng));
    observations.push_back({X, K.project(truth * X)});
  }
  for (double size : {1e-4, 1e-3, 1e-2, 3e-2}) {
    vo::Vec6 delta;
    for (int k = 0; k < 6; ++k) delta(k) = size * u(rng);
    vo::SE3 pose = vo::perturb(truth, delta);
    std::vector<char> inliers;
    vo::optimize_pose(K, observations, pose, 3.0, inliers);
    const double rotation = vo::log_so3(pose.R * truth.R.transpose()).norm();
    const double translation = (pose.t - truth.t).norm();
    std::cout << "pose optimizer from " << size << ": rotation error " << rotation << ", translation error "
              << translation << '\n';
    require(rotation < 1e-7 && translation < 1e-6, "Pose optimization did not converge");
  }
}

int main() {
  try {
    pose_optimizer_converges();
    // Noise-free data must be reproduced essentially exactly (regression for
    // rotation drift off SO(3), which made poses diverge geometrically).
    const auto clean = make_scene(true, 240, 5, {0, 0, 0, 0, 0});
    const auto exact = evaluate(clean, run(clean.frames));
    std::cout << "noise-free walk: " << exact.segments << " segment(s), ATE " << 100 * exact.ate_fraction
              << "% of path, median rotation error " << exact.rotation_deg << " deg\n";
    require(exact.segments == 1 && exact.ate_fraction < 1e-5 && exact.rotation_deg < 1e-4,
            "Noise-free trajectory is not exact");
    const auto scene = make_scene(true);
    double ms = 0;
    const auto trajectory = run(scene.frames, &ms);
    const auto result = evaluate(scene, trajectory);
    std::cout << "walk: posed " << result.posed << "/" << scene.frames.size() << " frames from frame "
              << result.first_pose << ", " << result.segments << " segment(s), ATE " << 100 * result.ate_fraction
              << "% of path, median rotation error " << result.rotation_deg << " deg, " << ms << " ms/frame\n";
    require(result.segments == 1, "Visual odometry lost tracking on a clean synthetic walk");
    require(result.first_pose >= 0 && result.first_pose < 60, "Initialization took too long");
    require(result.posed >= static_cast<int>(scene.frames.size()) - 60, "Too few posed frames");
    require(result.ate_fraction < 0.01, "Trajectory error above 1% of path length");
    require(result.rotation_deg < 0.5, "Rotation error above 0.5 degrees");

    const auto rotation = make_scene(false, 150);
    const auto spun = run(rotation.frames);
    std::cout << "pure rotation: posed " << spun.size() << " frames (expected 0: no baseline)\n";
    require(spun.empty(), "Pure rotation must not initialize a map");

    // External-track hook: the same tracks through CSV give the same result.
    const auto path = std::filesystem::temp_directory_path() / "slam-native-vo-test-tracks.csv";
    {
      std::ofstream out(path);
      write_tracks_header(out, 1920, 1080);
      for (const auto& frame : scene.frames) write_tracks(out, frame);
    }
    const auto loaded = read_tracks_csv(path);
    std::filesystem::remove(path);
    require(loaded.size() == scene.frames.size(), "CSV lost frames");
    const auto replay = run(loaded);
    require(replay.size() == trajectory.size(), "CSV replay changed the trajectory length");
    double difference = 0;
    for (std::size_t i = 0; i < replay.size(); ++i) {
      const auto a = replay[i].pose.center(), b = trajectory[i].pose.center();
      difference = std::max({difference, std::abs(a[0] - b[0]), std::abs(a[1] - b[1]), std::abs(a[2] - b[2])});
    }
    std::cout << "CSV replay max centre difference " << difference << '\n';
    require(difference < 1e-3, "CSV replay differs (float32 round trip should be tiny)");
    std::cout << "visual odometry checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
