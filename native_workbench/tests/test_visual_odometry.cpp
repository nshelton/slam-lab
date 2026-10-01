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
#include <unordered_map>

using namespace slam_native;
namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

struct Scene {
  std::vector<TrackedFrame> frames;
  std::vector<Eigen::Vector3d> centers;   // ground-truth camera centres
  std::vector<Eigen::Matrix3d> rotations; // ground-truth world->camera
  std::vector<Eigen::Vector3d> points;    // ground-truth landmarks (index encoded in colour)
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
    TrackedFrame frame{static_cast<std::uint64_t>(f), f * 16'666'667LL, width, height, {}, 0, {}};
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
      // Colour encodes the point index, so map points can be checked against truth.
      frame.observations.push_back({id[p], static_cast<float>(x), static_cast<float>(y), true,
                                    {static_cast<std::uint8_t>(p % 256), static_cast<std::uint8_t>(p / 256), 77}});
    }
    scene.frames.push_back(std::move(frame));
  }
  scene.points = points;
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

// The place search on the calling thread, so runs are deterministic.
VisualOdometryConfig synchronous() {
  VisualOdometryConfig config;
  config.place_synchronous = true;
  return config;
}

std::vector<TrajectorySample> run(const std::vector<TrackedFrame>& frames, double* ms = nullptr,
                                  VisualOdometryConfig config = synchronous()) {
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

// distort_point inverts undistort_point.
void distortion_round_trip() {
  for (const double k1 : {-0.3, -0.05, 0.05, 0.2})
    for (const float x : {0.0F, 400.0F, 959.5F, 1919.0F})
      for (const float y : {0.0F, 539.5F, 1079.0F}) {
        float u = x, v = y;
        undistort_point(u, v, 1920, 1080, k1);
        distort_point(u, v, 1920, 1080, k1);
        require(std::abs(u - x) < 1e-2F && std::abs(v - y) < 1e-2F, "distort_point does not invert undistort_point");
      }
}

// One random unit descriptor per scene point, attached to its observations.
void add_descriptors(Scene& scene) {
  constexpr int kDimension = 32;
  std::mt19937 rng(9);
  std::normal_distribution<float> gauss;
  std::vector<Eigen::VectorXf> descriptor(scene.points.size());
  for (auto& d : descriptor) d = Eigen::VectorXf::NullaryExpr(kDimension, [&] { return gauss(rng); }).normalized();
  for (auto& frame : scene.frames) {
    frame.descriptor_dimension = kDimension;
    for (const auto& o : frame.observations) {
      const auto& d = descriptor[o.color[0] + 256U * o.color[1]];
      frame.descriptors.insert(frame.descriptors.end(), d.data(), d.data() + kDimension);
    }
  }
}

// Largest distance between a map point and the scene point its colour
// encodes, after aligning the map to the scene by a similarity.
double worst_point_error(const Scene& scene, const std::vector<MapPoint>& map) {
  Eigen::Matrix3Xd estimated(3, map.size()), truth(3, map.size());
  for (std::size_t i = 0; i < map.size(); ++i) {
    estimated.col(i) = Eigen::Vector3d(map[i].position[0], map[i].position[1], map[i].position[2]);
    truth.col(i) = scene.points[map[i].color[0] + 256U * map[i].color[1]];
  }
  const Eigen::Matrix4d S = Eigen::umeyama(estimated, truth, true);
  double worst = 0;
  for (std::size_t i = 0; i < map.size(); ++i)
    worst = std::max(worst, (S.topLeftCorner<3, 3>() * estimated.col(i) + S.topRightCorner<3, 1>() - truth.col(i)).norm());
  return worst;
}

// The covariance of an optimized pose must match the scatter of its errors:
// the normalized squared error (NEES) of a 6-d estimate averages 6.
void pose_covariance_is_consistent() {
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> u(-1, 1);
  std::normal_distribution<double> noise(0, 0.7);
  const vo::Intrinsics K{1662.8, 1662.8, 959.5, 539.5};
  const vo::SE3 truth{vo::exp_so3(vo::Vec3(0.02, -0.1, 0.03)), vo::Vec3(0.3, -0.1, 0.5)};
  double nees = 0;
  const int trials = 400;
  for (int trial = 0; trial < trials; ++trial) {
    std::vector<vo::PoseObservation> observations;
    for (int i = 0; i < 60; ++i) {
      const vo::Vec3 X(4 * u(rng), 2 * u(rng), 8 + 4 * u(rng));
      observations.push_back({X, K.project(truth * X) + vo::Vec2(noise(rng), noise(rng))});
    }
    vo::SE3 pose = truth;
    std::vector<char> inliers;
    vo::optimize_pose(K, observations, pose, 5.0, inliers);
    const vo::Vec6 error = vo::pose_difference(truth, pose);
    nees += error.dot(vo::pose_covariance(K, observations, pose, inliers).ldlt().solve(error));
  }
  std::cout << "pose covariance: mean NEES " << nees / trials << " (6 when consistent)\n";
  require(nees / trials > 5 && nees / trials < 7.5, "Pose covariance does not match the pose errors");
}

// The motion model recovers the acceleration noise from one-step prediction
// errors despite measurement jitter, and its n-step covariance matches a
// simulation of the model.
void motion_model_is_consistent() {
  std::mt19937 rng(13);
  std::normal_distribution<double> gauss;
  // Acceleration and measurement sigma per axis. Q = 4 S, as on real clips
  // (lag-1 correlation of the errors about -0.4); with far more jitter than
  // acceleration the estimate is a small difference of large terms and errs high.
  const double q = 2e-3, r = 1e-3;
  const vo::Mat6 R = r * r * vo::Mat6::Identity();
  vo::MotionModel model;
  model.reset(R, 1);
  const auto draw = [&](double sigma) {
    vo::Vec6 v;
    for (int k = 0; k < 6; ++k) v(k) = sigma * gauss(rng);
    return v;
  };
  // Scalar-per-axis simulation: true pose x, velocity v; measured m = x + noise.
  vo::Vec6 x = vo::Vec6::Zero(), v = vo::Vec6::Zero();
  vo::Vec6 m1 = x + draw(r), m0 = m1;
  vo::Mat6 mean = vo::Mat6::Zero();
  int samples = 0;
  for (int k = 0; k < 4000; ++k) {
    v += draw(q);
    x += v;
    const vo::Vec6 m = x + draw(r);
    model.predict();
    model.correct(m - (2 * m1 - m0), R, 1);
    m0 = m1;
    m1 = m;
    if (k > 500) { mean += model.acceleration(); ++samples; }
  }
  const double estimated = std::sqrt((mean / samples).trace() / 6);
  std::cout << "motion model: acceleration sigma " << estimated << " (true " << q << ", measurement " << r << ")\n";
  require(estimated > 0.8 * q && estimated < 1.3 * q, "Acceleration noise estimate is off");
  // n frames without a measurement: P + n (C + C^T) + n^2 Pv + Q n(n+1)(2n+1)/6.
  const vo::Mat6 Q = model.acceleration(), P = model.pose, Pv = model.velocity, C = model.cross;
  const int n = 20;
  for (int i = 0; i < n; ++i) model.predict();
  const vo::Mat6 expected = P + n * (C + C.transpose()) + double(n) * n * Pv + Q * (n * (n + 1) * (2 * n + 1) / 6.0);
  require((model.pose - expected).norm() < 1e-9 * expected.norm(), "n-step covariance does not match the closed form");
}

// An occluder hides the scene for `gap` frames and every track restarts
// behind it. The odometry must coast on the motion model, with a pose whose
// confidence falls, find the local map again by descriptor, and keep one
// segment.
void coasts_through_occlusion() {
  auto scene = make_scene(true, 240, 5, {0, 0, 0, 0, 0});
  add_descriptors(scene);
  const std::size_t first = 120, gap = 20;
  for (std::size_t f = first; f < scene.frames.size(); ++f) {
    auto& frame = scene.frames[f];
    if (f < first + gap) {
      frame.observations.resize(10);  // too few to track
      frame.descriptors.resize(10 * static_cast<std::size_t>(frame.descriptor_dimension));
    }
    for (auto& o : frame.observations) o.track_id += f < first + gap ? 2'000'000 : 1'000'000;
  }
  VisualOdometry odometry(synchronous());
  int coasted = 0, relocalized = 0;
  double last_confidence = 1, last_sigma = 0, tracked_confidence = 1;
  for (const auto& frame : scene.frames) {
    const auto r = odometry.process(frame);
    if (r.predicted) {
      require(r.state == OdometryState::coasting && r.has_pose, "coasted frame without a pose");
      if (std::getenv("SLAM_VO_VERBOSE"))
        std::cout << "  coasting frame " << r.frame_index << ": sigma " << r.pose_sigma_px << " px, confidence "
                  << r.confidence << '\n';
      require(r.pose_sigma_px > last_sigma && r.confidence <= last_confidence, "uncertainty must grow while coasting");
      last_sigma = r.pose_sigma_px;
      last_confidence = r.confidence;
      ++coasted;
    } else if (r.has_pose) {
      tracked_confidence = std::min(tracked_confidence, r.confidence);
    }
    if (r.relocalized) std::cout << "occlusion: frame " << r.frame_index << ' ' << r.event << '\n';
    relocalized += r.relocalized;
  }
  const auto trajectory = odometry.trajectory();
  const auto result = evaluate(scene, trajectory);
  int predicted = 0;
  for (const auto& sample : trajectory) predicted += sample.predicted;
  std::cout << "occlusion: " << coasted << " coasted frames (confidence down to " << last_confidence
            << "), tracked confidence >= " << tracked_confidence << ", " << result.segments << " segment(s), ATE "
            << 100 * result.ate_fraction << "% of path\n";
  require(coasted == static_cast<int>(gap) && predicted == coasted, "occluded frames were not coasted");
  require(last_confidence < 0.5, "a pose predicted for many frames must not be confident");
  require(relocalized == 1 && result.segments == 1, "the map was not found again after the occlusion");
  require(tracked_confidence > 0.99, "tracked poses on exact data must be confident");
  require(result.ate_fraction < 2e-3, "trajectory error through the occlusion (coasted frames included)");

  // Never relocalized (the scene does not come back): the segment ends and
  // the unconfirmed coasted frames leave the trajectory.
  VisualOdometryConfig config = synchronous();
  config.coast_max_frames = 15;
  VisualOdometry bounded(config);
  std::uint64_t lost_at = 0;
  for (std::size_t f = 0; f < first + gap; ++f)
    if (bounded.process(scene.frames[f]).state == OdometryState::lost) lost_at = f;
  require(lost_at == first + 15, "coasting did not end after coast_max_frames");
  for (const auto& sample : bounded.trajectory()) require(!sample.predicted, "unconfirmed coasted frames were kept");
}

// The scene is hidden for longer than the odometry coasts, so the segment
// ends, and every track restarts. Nothing is dropped from the map, and the
// search without a pose prior finds it again: tracking resumes in the same
// segment. Without that search a second segment starts and the first one's
// landmarks stay in the map, outside the local map.
void resumes_segment_without_prior() {
  auto scene = make_scene(true, 240, 5, {0, 0, 0, 0, 0});
  add_descriptors(scene);
  const std::size_t first = 120, gap = 40;
  for (std::size_t f = first; f < scene.frames.size(); ++f) {
    auto& frame = scene.frames[f];
    if (f < first + gap) {
      frame.observations.resize(10);  // too few to track
      frame.descriptors.resize(10 * static_cast<std::size_t>(frame.descriptor_dimension));
    }
    for (auto& o : frame.observations) o.track_id += f < first + gap ? 2'000'000 : 1'000'000;
  }
  VisualOdometryConfig config = synchronous();
  config.coast_max_frames = 5;
  {
    VisualOdometry odometry(config);
    int lost = 0, resumed = 0;
    std::size_t before = 0;
    for (const auto& frame : scene.frames) {
      const auto r = odometry.process(frame);
      lost += r.state == OdometryState::lost;
      if (r.state == OdometryState::lost) before = odometry.map_size();
      if (r.relocalized) {
        std::cout << "long occlusion: frame " << r.frame_index << ' ' << r.event << '\n';
        require(r.frame_index == first + gap && r.segment == 0 && r.place_segment == 0 && r.keyframe,
                "the first frame after the occlusion did not resume segment 0");
        ++resumed;
      }
    }
    const auto trajectory = odometry.trajectory();
    const auto result = evaluate(scene, trajectory);
    const auto map = odometry.map();
    int old = 0;  // landmarks from before the occlusion, observed again after it
    for (const auto& point : map) old += point.first_frame < first && point.last_frame >= first + gap;
    std::cout << "long occlusion: " << lost << " loss, " << resumed << " resume, " << result.segments
              << " segment(s), ATE " << 100 * result.ate_fraction << "% of path, " << old << " of " << map.size()
              << " landmarks span the occlusion, worst point error " << worst_point_error(scene, map) << '\n';
    require(lost == 1 && resumed == 1 && result.segments == 1, "the segment was not resumed after the occlusion");
    require(map.size() >= before && old > 100, "landmarks from before the occlusion were not kept and re-observed");
    require(result.ate_fraction < 2e-3, "trajectory error across the resumed segment");
    require(worst_point_error(scene, map) < 1e-3, "the resumed map does not match the true points");
    for (const auto& sample : trajectory)
      require(sample.frame_index < first || sample.frame_index >= first + gap, "a frame without a pose was kept");
    for (const auto& tracked : odometry.tracked_landmarks())
      require(odometry.has_landmark(tracked.track_id) && tracked.first_frame <= scene.frames.back().frame_index,
              "tracked_landmarks() lists a track without a landmark");
    require(odometry.tracked_landmarks().size() > 100, "too few tracked landmarks at the end of the walk");
    std::map<std::uint64_t, int> observers;
    for (const auto& o : odometry.landmark_observations()) ++observers[o.landmark_id];
    for (const auto& point : map)
      require(observers[point.landmark_id] == point.keyframe_observations, "landmark_observations() does not match the map");
  }
  {
    // The same on the worker thread: the result arrives a few frames later
    // and is used through the tracks still observed then.
    VisualOdometryConfig threaded = config;
    threaded.place_synchronous = false;
    VisualOdometry odometry(threaded);
    std::uint64_t resumed_at = 0;
    for (const auto& frame : scene.frames) {
      const auto r = odometry.process(frame);
      if (r.relocalized) resumed_at = r.frame_index;
    }
    const auto result = evaluate(scene, odometry.trajectory());
    std::cout << "long occlusion, search on the worker: resumed at frame " << resumed_at << ", " << result.segments
              << " segment(s), ATE " << 100 * result.ate_fraction << "% of path\n";
    require(resumed_at >= first + gap && resumed_at < first + gap + 30 && result.segments == 1,
            "the worker's search did not resume the segment");
    require(result.ate_fraction < 2e-3, "trajectory error across the segment resumed from the worker");
  }
  config.place_search_interval = -1;  // no search without a prior: a new segment
  VisualOdometry odometry(config);
  for (const auto& frame : scene.frames) odometry.process(frame);
  int kept = 0, local = 0, second = 0;
  for (const auto& point : odometry.map()) {
    kept += point.segment == 0;
    local += point.segment == 0 && point.local;
    second += point.segment != 0;
  }
  std::cout << "long occlusion, no place search: " << kept << " landmarks of the ended segment kept (" << local
            << " local), " << second << " in the new one\n";
  require(evaluate(scene, odometry.trajectory()).segments == 2, "expected a second segment without the place search");
  require(kept > 300 && local == 0 && second > 100, "the ended segment's landmarks must stay, outside the local map");
}

// The camera loses the map while it looks at a part of the scene it has not
// seen (a second segment starts there), then the first part comes back into
// view. The search finds the first segment's landmarks from the second, and
// the second is moved into the first's map: one segment, one consistent map.
void merges_segments() {
  auto scene = make_scene(true, 420, 5, {0, 0, 0, 0, 0});
  add_descriptors(scene);
  const std::size_t first = 120, second = 240;
  constexpr std::size_t kSplit = 1500;  // points below: the first part of the scene
  for (std::size_t f = 0; f < scene.frames.size(); ++f) {
    auto& frame = scene.frames[f];
    TrackedFrame kept = frame;
    kept.observations.clear();
    kept.descriptors.clear();
    for (std::size_t i = 0; i < frame.observations.size(); ++i) {
      auto o = frame.observations[i];
      const std::size_t point = o.color[0] + 256U * o.color[1];
      const bool shown = f < first ? point < kSplit : f < second ? point >= kSplit : true;
      if (!shown) continue;
      if (f >= second && point < kSplit) o.track_id += 1'000'000;  // the first part's tracks restart
      kept.observations.push_back(o);
      const auto row = frame.descriptors.begin() + static_cast<std::ptrdiff_t>(i * frame.descriptor_dimension);
      kept.descriptors.insert(kept.descriptors.end(), row, row + frame.descriptor_dimension);
    }
    frame = std::move(kept);
  }
  VisualOdometryConfig config = synchronous();
  config.coast_max_frames = 5;
  config.place_search_interval = 10;
  for (const bool correct : {false, true}) {
    config.place_correction = correct;
    VisualOdometry odometry(config);
    int merged_at = -1, seen_from_other = 0;
    for (const auto& frame : scene.frames) {
      const auto r = odometry.process(frame);
      seen_from_other += r.place_inliers > 0 && r.place_segment != r.segment;
      if (r.segments_merged) {
        std::cout << "two parts: frame " << r.frame_index << ' ' << r.event << '\n';
        merged_at = static_cast<int>(r.frame_index);
      }
    }
    const auto trajectory = odometry.trajectory();
    const auto result = evaluate(scene, trajectory);
    const auto map = odometry.map();
    std::size_t in_first = 0;
    for (const auto& point : map) in_first += point.segment == 0;
    std::cout << "two parts, correction " << (correct ? "on" : "off") << ": " << result.segments << " segment(s), "
              << trajectory.size() << " poses, ATE " << 100 * result.ate_fraction << "% of path, " << in_first
              << " of " << map.size() << " landmarks in segment 0";
    if (!correct) {
      std::cout << ", the first segment seen " << seen_from_other << " times from the second\n";
      require(result.segments == 2 && merged_at < 0 && seen_from_other > 0, "expected two segments, reported not merged");
      continue;
    }
    const double worst = worst_point_error(scene, map);
    std::cout << ", worst point error " << worst << '\n';
    require(merged_at >= static_cast<int>(second) && result.segments == 1 && in_first == map.size(),
            "the second segment was not merged into the first");
    require(trajectory.size() > 360, "poses were lost in the merge");
    require(result.ate_fraction < 2e-3 && worst < 5e-3, "the merged map does not match the scene");
  }
}

// A landmark that left the local map is found again by descriptor when the
// camera looks back at it, and re-associated only when it still projects where
// it is seen: the colour-encoded map stays exact.
void reassociates_outside_local_map() {
  auto scene = make_scene(true, 600, 5, {0, 0, 0, 0, 0});
  add_descriptors(scene);
  VisualOdometryConfig config = synchronous();
  config.local_map_keyframes = config.window_keyframes;  // landmarks leave the local map quickly
  VisualOdometry odometry(config);
  int reassociated = 0, from_outside = 0;
  for (const auto& frame : scene.frames) {
    const auto r = odometry.process(frame);
    reassociated += r.reassociated;
    from_outside += r.place_reassociated;
  }
  const auto map = odometry.map();
  const double worst = worst_point_error(scene, map);
  std::cout << "outside the local map: " << from_outside << " of " << reassociated << " re-associations, "
            << map.size() << " map points, worst point error " << worst << '\n';
  require(from_outside > 20, "too few landmarks found again outside the local map");
  require(worst < 1e-3, "a re-association joined different points");
}

// Broken tracks re-find their landmark (and duplicates merge) through
// per-point descriptors, and only their own: a wrong association would mix
// two points' colour-encoded indices and break the exact map.
void reassociates_broken_tracks() {
  auto scene = make_scene(true, 240, 5, {0, 0, 0.02, 0, 0});
  add_descriptors(scene);
  VisualOdometry odometry(synchronous());
  int reassociated = 0, merged = 0;
  for (const auto& frame : scene.frames) {
    const auto r = odometry.process(frame);
    reassociated += r.reassociated;
    merged += r.merged;
  }
  const auto map = odometry.map();
  const double worst = worst_point_error(scene, map);
  std::cout << "broken tracks: " << reassociated << " re-associated (" << merged << " merged), " << map.size()
            << " map points, worst point error " << worst << '\n';
  require(reassociated > 500, "too few re-associations");
  require(worst < 1e-3, "a re-association joined different points");
}

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

// P3P recovers random poses exactly, and P3P RANSAC tolerates 60% outliers.
void p3p_recovers_pose() {
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> u(-1, 1);
  const vo::Intrinsics K{1447, 1447, 959.5, 539.5};
  int solved = 0;
  const int trials = 500;
  for (int trial = 0; trial < trials; ++trial) {
    const vo::SE3 truth{vo::exp_so3(vo::Vec3(0.5 * u(rng), 0.5 * u(rng), 0.5 * u(rng))),
                        vo::Vec3(u(rng), u(rng), 2 * u(rng))};
    std::array<vo::Vec3, 3> X, f;
    for (int i = 0; i < 3; ++i) {
      const vo::Vec3 xc(3 * u(rng), 2 * u(rng), 6 + 4 * u(rng));
      X[i] = truth.inverse() * xc;
      f[i] = xc.normalized();
    }
    double best = 1e9;
    for (const auto& T : vo::p3p(X, f))
      best = std::min(best, vo::log_so3(T.R * truth.R.transpose()).norm() + (T.t - truth.t).norm());
    solved += best < 1e-6;
  }
  std::cout << "p3p: exact in " << solved << "/" << trials << " random trials\n";
  require(solved >= trials * 98 / 100, "P3P failed to recover the pose");

  const vo::SE3 truth{vo::exp_so3(vo::Vec3(0.05, -0.2, 0.02)), vo::Vec3(0.4, -0.1, 0.3)};
  std::vector<vo::PoseObservation> observations;
  std::normal_distribution<double> noise(0, 0.7);
  for (int i = 0; i < 200; ++i) {
    const vo::Vec3 X(4 * u(rng), 2 * u(rng), 8 + 4 * u(rng));
    vo::Vec2 px = K.project(truth * X) + vo::Vec2(noise(rng), noise(rng));
    if (i % 5 < 3) px = vo::Vec2(960 + 900 * u(rng), 540 + 500 * u(rng));  // 60% wrong matches
    observations.push_back({X, px});
  }
  vo::SE3 pose;
  std::vector<char> inliers;
  const int count = vo::ransac_pnp(K, observations, 4.0, 300, 7, pose, inliers);
  vo::optimize_pose(K, observations, pose, 3.0, inliers);
  const double rotation = vo::log_so3(pose.R * truth.R.transpose()).norm() * 180 / M_PI;
  std::cout << "p3p ransac: " << count << "/200 inliers (80 true), rotation error " << rotation << " deg\n";
  require(count >= 70 && rotation < 0.2, "P3P RANSAC did not find the pose");
}

int main() {
  try {
    pose_optimizer_converges();
    p3p_recovers_pose();
    distortion_round_trip();
    pose_covariance_is_consistent();
    motion_model_is_consistent();
    reassociates_broken_tracks();
    coasts_through_occlusion();
    resumes_segment_without_prior();
    merges_segments();
    reassociates_outside_local_map();
    // Noise-free data must be reproduced essentially exactly (regression for
    // rotation drift off SO(3), which made poses diverge geometrically).
    const auto clean = make_scene(true, 240, 5, {0, 0, 0, 0, 0});
    const auto exact = evaluate(clean, run(clean.frames));
    std::cout << "noise-free walk: " << exact.segments << " segment(s), ATE " << 100 * exact.ate_fraction
              << "% of path, median rotation error " << exact.rotation_deg << " deg\n";
    require(exact.segments == 1 && exact.ate_fraction < 1e-5 && exact.rotation_deg < 1e-4,
            "Noise-free trajectory is not exact");
    {
      // Landmarks that left the local map are kept, carry their observations'
      // colour, and match the true points up to a similarity. (A local map
      // as short as the BA window, so the short walk leaves landmarks behind.)
      VisualOdometryConfig config = synchronous();
      config.local_map_keyframes = config.window_keyframes;
      VisualOdometry odometry(config);
      for (const auto& frame : clean.frames) odometry.process(frame);
      const auto map = odometry.map();
      std::size_t local = 0;
      for (const auto& point : map) local += point.local;
      const std::size_t left = map.size() - local;
      require(left > 200, "too few landmarks outside the local map");
      Eigen::Matrix3Xd estimated(3, map.size()), truth(3, map.size());
      for (std::size_t i = 0; i < map.size(); ++i) {
        const auto& point = map[i];
        const std::size_t index = point.color[0] + 256U * point.color[1];
        require(point.has_color && point.color[2] == 77 && index < clean.points.size(), "landmark colour lost");
        require(point.keyframe_observations >= 2 && point.first_frame <= point.last_frame, "landmark metadata");
        estimated.col(i) = Eigen::Vector3d(point.position[0], point.position[1], point.position[2]);
        truth.col(i) = clean.points[index];
      }
      const Eigen::Matrix4d S = Eigen::umeyama(estimated, truth, true);
      double squared = 0;
      for (std::size_t i = 0; i < map.size(); ++i)
        squared += (S.topLeftCorner<3, 3>() * estimated.col(i) + S.topRightCorner<3, 1>() - truth.col(i)).squaredNorm();
      const double rms = std::sqrt(squared / map.size());
      std::cout << "noise-free map: " << left << " points outside the local map + " << local << " local, RMS error "
                << rms << " (scene units)\n";
      require(rms < 1e-3, "map does not match the true points");
    }
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
    {
      // The named r,g,b columns survive a round trip.
      TrackedFrame frame{7, 123, 1920, 1080, {}, 0, {}};
      frame.observations = {{1, 10.5F, 20.25F, true, {1, 2, 3}}, {2, 30, 40, true, {4, 5, 6}}};
      const auto color_path = std::filesystem::temp_directory_path() / "slam-native-vo-test-colors.csv";
      {
        std::ofstream out(color_path);
        write_tracks_header(out, 1920, 1080, true);
        write_tracks(out, frame, true);
      }
      const auto back = read_tracks_csv(color_path);
      std::filesystem::remove(color_path);
      require(back.size() == 1 && back[0].observations.size() == 2, "colour CSV lost rows");
      const auto& p = back[0].observations[0];
      const auto& q = back[0].observations[1];
      require(p.has_color && p.color[2] == 3 && p.x == 10.5F && q.color[0] == 4, "tracks CSV colours did not round-trip");
    }
    std::cout << "visual odometry checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
