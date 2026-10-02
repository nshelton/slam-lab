// Checks for the milestone-0 landmark depth measurement (SPLINE_BA.md): the
// fixed-pose re-solve, the attribution of observations to landmarks through
// merges and deletions, and the export on a noise-free synthetic walk through
// the real visual odometry.
#include "slam_native/landmark_depth.hpp"
#include "slam_native/visual_odometry.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace slam_native;
namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

Pose pose_from(const Eigen::Matrix3d& R, const Eigen::Vector3d& centre) {  // world -> camera
  Pose pose;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) pose.rotation[static_cast<std::size_t>(3 * r + c)] = R(r, c);
  const Eigen::Vector3d t = -R * centre;
  pose.translation = {t.x(), t.y(), t.z()};
  return pose;
}
Eigen::Vector2d project(const CameraIntrinsics& K, const Pose& pose, const Eigen::Vector3d& X) {
  Eigen::Matrix3d R;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) R(r, c) = pose.rotation[static_cast<std::size_t>(3 * r + c)];
  const Eigen::Vector3d xc = R * X + Eigen::Vector3d(pose.translation[0], pose.translation[1], pose.translation[2]);
  return {K.fx * xc.x() / xc.z() + K.cx, K.fy * xc.y() / xc.z() + K.cy};
}

void spread_indices_are_even() {
  require(spread_indices(10, 2) == std::vector<std::size_t>{0, 9}, "k = 2 is not first and last");
  require(spread_indices(10, 4) == std::vector<std::size_t>{0, 3, 6, 9}, "k = 4 is not spread evenly");
  require(spread_indices(9, 3) == std::vector<std::size_t>{0, 4, 8}, "k = 3 is not spread evenly");
  require(spread_indices(3, 4).empty(), "n < k must give nothing");
  require(spread_indices(4, 4) == std::vector<std::size_t>{0, 1, 2, 3}, "n == k must give everything");
  require(spread_indices(5, 0).size() == 5, "k = 0 must give everything");
  require(spread_indices(0, 0).empty(), "nothing from nothing");
}

// A camera translating sideways past random points, exact poses. The re-solve
// from all frames must beat the one from the two ends, be exact without
// noise, and shrug off a gross outlier.
void refit_is_exact_and_improves_with_frames() {
  const int width = 1920, height = 1080, frames = 30;
  const auto K = CameraIntrinsics::from_horizontal_fov(width, height, 60);
  std::vector<Pose> poses;
  for (int f = 0; f < frames; ++f)
    poses.push_back(pose_from(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.02 * f, 0.004 * f, 0)));
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(0, 1);
  std::normal_distribution<double> noise(0, 0.7);
  std::vector<double> error_two, error_all, error_robust;
  double exact_error = 0;
  int points = 0;
  for (int attempt = 0; attempt < 2000 && points < 300; ++attempt) {
    const Eigen::Vector3d X(-2 + 4 * u(rng), -1.5 + 3 * u(rng), 4 + 8 * u(rng));
    std::vector<FixedPoseObservation> clean, noisy;
    bool visible = true;
    for (const auto& pose : poses) {
      const Eigen::Vector2d p = project(K, pose, X);
      visible = visible && p.x() >= 0 && p.x() < width && p.y() >= 0 && p.y() < height;
      clean.push_back({pose, p.x(), p.y()});
      noisy.push_back({pose, p.x() + noise(rng), p.y() + noise(rng)});
    }
    if (!visible) continue;
    ++points;
    const std::array<double, 3> initial{X.x() * 1.05, X.y() * 1.05, X.z() * 1.05};  // a 5% depth error to start from
    const auto exact = refit_landmark(K, clean, initial);
    require(exact.has_value(), "noise-free re-solve failed");
    exact_error = std::max(exact_error, (Eigen::Vector3d((*exact)[0], (*exact)[1], (*exact)[2]) - X).norm() / X.norm());
    const auto two = refit_landmark(K, {noisy.front(), noisy.back()}, initial);
    const auto all = refit_landmark(K, noisy, initial);
    require(two.has_value() && all.has_value(), "noisy re-solve failed");
    error_two.push_back((Eigen::Vector3d((*two)[0], (*two)[1], (*two)[2]) - X).norm() / X.norm());
    error_all.push_back((Eigen::Vector3d((*all)[0], (*all)[1], (*all)[2]) - X).norm() / X.norm());
    auto spoiled = noisy;
    spoiled[frames / 2].ux += 40;  // one gross outlier in the middle
    const auto robust = refit_landmark(K, spoiled, initial);
    require(robust.has_value(), "robust re-solve failed");
    error_robust.push_back((Eigen::Vector3d((*robust)[0], (*robust)[1], (*robust)[2]) - X).norm() / X.norm());
  }
  require(points >= 200, "too few visible points");
  const auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const double two = median(error_two), all = median(error_all), robust = median(error_robust);
  std::cout << "re-solve of " << points << " points, 30 frames, 0.7 px noise: median relative error from two ends "
            << 100 * two << "%, from all " << 100 * all << "%, all with one 40 px outlier " << 100 * robust
            << "%; noise-free worst " << exact_error << '\n';
  require(exact_error < 1e-9, "noise-free re-solve is not exact");
  // Information grows as the spread of camera positions: 30 even frames give
  // about 2.2x less depth error than the two ends (SPLINE_BA.md).
  require(all < 0.7 * two, "all frames do not beat the two ends");
  require(robust < 1.5 * all, "the Huber loss did not absorb the outlier");

  // Degenerate and invalid cases.
  const Eigen::Vector3d X(0.3, -0.2, 6);
  const std::array<double, 3> initial{X.x(), X.y(), X.z()};
  std::vector<FixedPoseObservation> clean;
  for (const auto& pose : poses) {
    const Eigen::Vector2d p = project(K, pose, X);
    clean.push_back({pose, p.x(), p.y()});
  }
  require(!refit_landmark(K, {clean.front()}, initial).has_value(), "one observation must not solve");
  std::vector<FixedPoseObservation> behind = clean;
  behind.push_back({pose_from(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 50)), K.cx, K.cy});
  require(!refit_landmark(K, behind, initial).has_value(), "a point behind a camera must not solve");
}

// Observations follow their tracks to the landmark each track ends on, through
// merges; tracks whose landmark was deleted (or never existed) drop out.
void attribution_follows_tracks_and_merges() {
  LandmarkObservationLog log;
  const auto frame = [](std::uint64_t index, std::vector<std::uint64_t> tracks) {
    TrackedFrame f;
    f.frame_index = index;
    f.width = 640;
    f.height = 480;
    for (auto t : tracks) f.observations.push_back({t, 10.0F * float(t), 20.0F, false, {}});
    return f;
  };
  using Pairs = std::vector<std::pair<std::uint64_t, std::uint64_t>>;
  const std::vector<std::array<float, 2>> no_raw;
  log.record(frame(0, {1, 2, 3}), no_raw, 0.0, Pairs{}, {});
  log.record(frame(1, {1, 2, 3}), no_raw, 0.0, Pairs{{1, 100}, {2, 101}}, {1, 2});
  log.record(frame(2, {1, 2, 3, 4}), no_raw, 0.0, Pairs{{1, 100}, {2, 101}, {4, 102}}, {1, 4});
  // Landmark 102 (track 4) merged into 100; track 1 lost its entry; 101 deleted.
  log.record(frame(3, {1, 3, 4}), no_raw, 0.0, Pairs{{4, 100}}, {4});
  require(log.size() == 13, "log did not keep every observation");
  std::vector<MapPoint> map(1);
  map[0].landmark_id = 100;
  const auto by_landmark = log.attribute(map, Pairs{{102, 100}});
  require(by_landmark.size() == 1 && by_landmark.count(100), "only the surviving landmark has observations");
  const auto& list = by_landmark.at(100);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> got;
  for (const auto& o : list) got.emplace_back(o.frame_index, o.track_id);
  const std::vector<std::pair<std::uint64_t, std::uint64_t>> expected{{0, 1}, {1, 1}, {2, 1}, {2, 4}, {3, 1}, {3, 4}};
  require(got == expected, "observations are not attributed through the merge in frame order");
  require(!list[0].judged && list[1].judged && list[1].inlier && list[3].judged && list[3].inlier,
          "judged/inlier flags are wrong");
  require(!list[4].judged, "a track without a landmark on a frame is unjudged there");
  require(std::isnan(list[0].raw_x), "missing raw detections must be NaN");

  // Raw detections and undistortion are recorded per observation.
  LandmarkObservationLog distorted;
  const std::vector<std::array<float, 2>> raw{{100.0F, 200.0F}};
  distorted.record(frame(5, {7}), raw, -0.1, Pairs{}, {});
  std::vector<MapPoint> one(1);
  one[0].landmark_id = 5;
  require(distorted.attribute(one, {}).empty(), "a track that never had a landmark contributes nothing");
  const auto all = distorted.attribute({}, {});
  require(all.empty(), "no landmarks, no attribution");
}

// Noise-free walk through the real VO: the export's keyframes-only depths and
// the re-solved depths both match the truth up to the segment's scale.
void export_matches_truth_on_noise_free_walk() {
  const int width = 1920, height = 1080, frame_count = 150;
  const auto K = CameraIntrinsics::from_horizontal_fov(width, height, 60);
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> u(0, 1);
  std::vector<Eigen::Vector3d> points;
  for (int i = 0; i < 1500; ++i) points.emplace_back(-8 + 16 * u(rng), -4 + 8 * u(rng), 3 + 17 * u(rng));
  std::vector<Pose> truth;
  std::vector<TrackedFrame> frames;
  for (int f = 0; f < frame_count; ++f) {
    const double s = f / 60.0;
    const Eigen::Vector3d c(1.5 * s, 0.1 * std::sin(2 * s), 0.6 * s);
    const Eigen::Matrix3d R = Eigen::AngleAxisd(-0.15 * std::sin(0.8 * s), Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Pose pose = pose_from(R, c);
    truth.push_back(pose);
    TrackedFrame frame;
    frame.frame_index = static_cast<std::uint64_t>(f);
    frame.timestamp_ns = static_cast<std::int64_t>(f * 1e9 / 60);
    frame.width = width;
    frame.height = height;
    for (std::size_t i = 0; i < points.size(); ++i) {
      const auto& X = points[i];
      const Eigen::Vector3d xc = R * X - R * c;
      if (xc.z() < 0.5) continue;
      const Eigen::Vector2d p = project(K, pose, X);
      if (p.x() < 0 || p.x() >= width || p.y() < 0 || p.y() >= height) continue;
      frame.observations.push_back({i, float(p.x()), float(p.y()), false, {}});
    }
    frames.push_back(std::move(frame));
  }
  VisualOdometryConfig config;
  config.place_synchronous = true;
  VisualOdometry odometry(config);
  LandmarkObservationLog log;
  for (const auto& frame : frames) {
    odometry.process(frame);
    std::vector<std::array<float, 2>> raw;
    for (const auto& o : frame.observations) raw.push_back({o.x, o.y});
    log.record(frame, raw, odometry);
  }
  require(odometry.trajectory_size() > 60, "the VO did not track the synthetic walk");
  std::stringstream csv;
  write_landmark_depth_csv(csv, odometry, log, {2, 4, 8, 0});
  std::string line;
  require(static_cast<bool>(std::getline(csv, line)), "no header");
  std::vector<std::string> columns;
  for (std::stringstream header(line); std::getline(header, line, ',');) columns.push_back(line);
  const auto column = [&](const std::string& name) {
    const auto it = std::find(columns.begin(), columns.end(), name);
    require(it != columns.end(), "missing column " + name);
    return static_cast<std::size_t>(it - columns.begin());
  };
  const std::size_t c_frame = column("frame_index"), c_track = column("track_id"), c_kf = column("depth_keyframes"),
                    c_all = column("depth_fused_all"), c_two = column("depth_fused_2"), c_raw = column("depth_raw_all"),
                    c_obs = column("observations");
  // Per keyframe, log(estimate / truth) per variant; the per-frame median is the scale.
  std::map<std::uint64_t, std::vector<std::array<double, 4>>> by_frame;
  std::size_t rows = 0, solved_all = 0, long_tracks = 0;
  while (std::getline(csv, line)) {
    std::vector<std::string> fields;
    for (std::stringstream row(line); std::getline(row, line, ',');) fields.push_back(line);
    require(fields.size() == columns.size(), "row width");
    ++rows;
    const auto frame = std::stoull(fields[c_frame]);
    const auto track = std::stoull(fields[c_track]);
    require(frame < truth.size() && track < points.size(), "row indices");
    const double z_true = depth_in(truth[frame], {points[track].x(), points[track].y(), points[track].z()});
    const double kf = std::stod(fields[c_kf]), all = std::stod(fields[c_all]), two = std::stod(fields[c_two]),
                 raw = std::stod(fields[c_raw]);
    require(std::isfinite(kf) && kf > 0, "keyframes-only depth missing");
    solved_all += std::isfinite(all);
    long_tracks += std::stoul(fields[c_obs]) > 8;
    if (!std::isfinite(all) || !std::isfinite(two) || !std::isfinite(raw)) continue;
    by_frame[frame].push_back({std::log(kf / z_true), std::log(all / z_true), std::log(two / z_true), std::log(raw / z_true)});
  }
  std::size_t scored = 0, bad = 0;
  double worst = 0;
  for (auto& [frame, list] : by_frame) {
    if (list.size() < 20) continue;
    for (int v = 0; v < 4; ++v) {
      std::vector<double> values;
      for (const auto& e : list) values.push_back(e[static_cast<std::size_t>(v)]);
      std::sort(values.begin(), values.end());
      const double scale = values[values.size() / 2];
      for (double value : values) {
        const double error = std::abs(value - scale);
        worst = std::max(worst, error);
        bad += error > 1e-3;
        ++scored;
      }
    }
  }
  std::cout << "noise-free export: " << rows << " (landmark, keyframe) rows, " << solved_all
            << " re-solved from all observations, " << long_tracks << " with more than 8 usable; " << scored
            << " depth values scored, worst log error " << worst << ", " << bad << " above 1e-3\n";
  require(rows > 500, "too few rows exported");
  require(solved_all > 0.95 * rows, "the re-solve from all observations failed too often");
  require(long_tracks > rows / 2, "every-frame observations were not kept");
  require(scored > 0 && bad == 0, "re-solved or keyframes-only depths do not match the truth");
}
}  // namespace

int main() {
  try {
    spread_indices_are_even();
    refit_is_exact_and_improves_with_frames();
    attribution_follows_tracks_and_merges();
    export_matches_truth_on_noise_free_walk();
    std::cout << "landmark depth checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
