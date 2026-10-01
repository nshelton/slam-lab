// Synthetic checks for the tracker-agnostic visual odometry (CPU).
#include "slam_native/covariance_shape.hpp"
#include "slam_native/track_io.hpp"
#include "slam_native/visual_odometry.hpp"
#include "pose_graph.hpp"   // internal loop-closure maths, unit-tested directly
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

// Occlusion: frames [begin, end) see nothing, and every track afterwards gets
// a new ID (as the tracker does). Each true point gets a random unit
// descriptor, observed with noise (cosine ~0.95).
std::vector<TrackedFrame> occluded(const Scene& scene, int begin, int end, int dimension = 32) {
  std::mt19937 rng(11);
  std::normal_distribution<float> n(0, 1);
  std::vector<std::vector<float>> truth(scene.points.size(), std::vector<float>(dimension));
  const auto normalize = [](std::vector<float>& v) {
    float norm = 0;
    for (float x : v) norm += x * x;
    for (float& x : v) x /= std::sqrt(norm);
  };
  for (auto& d : truth) {
    for (float& x : d) x = n(rng);
    normalize(d);
  }
  std::vector<TrackedFrame> frames;
  for (const auto& source : scene.frames) {
    TrackedFrame frame = source;
    frame.observations.clear();
    frame.descriptor_dimension = dimension;
    const int f = static_cast<int>(frame.frame_index);
    if (f >= begin && f < end) {
      frames.push_back(std::move(frame));
      continue;
    }
    for (auto o : source.observations) {
      if (f >= end) o.track_id += 1'000'000;
      const std::size_t index = o.color[0] + 256U * o.color[1];
      std::vector<float> d = truth[index];
      for (float& x : d) x += 0.3F / std::sqrt(float(dimension)) * n(rng);
      normalize(d);
      frame.observations.push_back(o);
      frame.descriptors.insert(frame.descriptors.end(), d.begin(), d.end());
    }
    frames.push_back(std::move(frame));
  }
  return frames;
}

// After a full occlusion the map is re-found by descriptor + P3P: tracking
// resumes in the same segment (same frame and scale) instead of starting over.
void relocalizes_after_occlusion() {
  const auto scene = make_scene(true, 240, 5, {0.5, 0.03, 0.005, 0.005, 0});
  const auto frames = occluded(scene, 120, 140);  // 1/3 s blind at 60 fps
  for (const bool enabled : {false, true}) {
    VisualOdometryConfig config;
    config.relocalization = enabled;
    VisualOdometry odometry(config);
    int relocalized = -1;
    std::string event;
    for (const auto& frame : frames) {
      const auto r = odometry.process(frame);
      if (r.relocalized && relocalized < 0) {
        relocalized = static_cast<int>(r.frame_index);
        event = r.event;
      }
    }
    const auto trajectory = odometry.trajectory();
    const auto result = evaluate(scene, trajectory);
    int after = 0, in_first = 0;
    for (const auto& sample : trajectory) {
      after += sample.frame_index >= 140;
      in_first += sample.frame_index >= 140 && sample.segment == 0;
    }
    std::cout << "occlusion 120-140, relocalization " << (enabled ? "on" : "off") << ": " << result.segments
              << " segment(s), " << in_first << "/" << after << " later poses in segment 0, ATE of largest "
              << 100 * result.ate_fraction << "% of path";
    if (enabled) std::cout << "; frame " << relocalized << ": " << event;
    std::cout << '\n';
    if (!enabled) {
      require(in_first == 0, "without relocalization the old segment cannot continue");
      continue;
    }
    require(relocalized >= 140 && relocalized <= 145, "did not relocalize promptly after the occlusion");
    require(in_first >= after - 6, "tracking did not continue in the original segment");
    require(result.ate_fraction < 0.01, "relocalized trajectory is inconsistent with the original segment");
  }
}

// Tracks that break (every missed detection gives a new ID) re-find their
// landmark instead of triangulating a duplicate, and only their own: both
// tracks of every re-association must observe the same true point.
void reassociates_broken_tracks() {
  const auto scene = make_scene(true, 240, 5, {0.5, 0.05, 0.01, 0.005, 0});
  const auto frames = occluded(scene, -1, -1);  // descriptors, no occlusion
  std::unordered_map<std::uint64_t, std::size_t> truth;  // track -> true point (colour-encoded)
  for (const auto& frame : frames)
    for (const auto& o : frame.observations) truth[o.track_id] = o.color[0] + 256U * o.color[1];
  int landmarks_off = 0, landmarks_no_merge = 0;
  for (const int mode : {0, 1, 2}) {  // off, re-association without merging, with merging
    const bool enabled = mode > 0;
    VisualOdometryConfig config;
    config.reassociation = enabled;
    config.merge_landmarks = mode == 2;
    VisualOdometry odometry(config);
    int reassociated = 0, wrong = 0, reported = 0, merged = 0;
    for (const auto& frame : frames) {
      const auto r = odometry.process(frame);
      reported += static_cast<int>(r.untracked_landmarks.size());
      for (const auto& p : r.untracked_landmarks) {
        require(p.x >= 0 && p.y >= 0 && p.x < 1920 && p.y < 1080, "untracked landmark reported outside the image");
        if (!p.reassociated) continue;
        ++reassociated;
        wrong += truth.at(p.track_id) != truth.at(p.new_track_id);
      }
      merged += r.merged;
      require(r.reassociated == static_cast<int>(std::count_if(r.untracked_landmarks.begin(),
          r.untracked_landmarks.end(), [](const auto& p) { return p.reassociated; })), "re-association count");
    }
    auto map = odometry.retired_map();
    const auto active = odometry.active_map();
    map.insert(map.end(), active.begin(), active.end());
    std::unordered_map<std::uint64_t, int> id_count;
    for (const auto& point : map) ++id_count[point.landmark_id];
    for (const auto& [id, n] : id_count) require(n == 1, "a landmark id appears twice in the map");
    const int landmarks = static_cast<int>(map.size());
    static const char* names[] = {"off", "on, no merging", "on, merging"};
    std::cout << "broken tracks, re-association " << names[mode] << ": " << landmarks << " landmarks, "
              << reassociated << " re-associated (" << merged << " merges, " << wrong << " wrong), "
              << reported / static_cast<int>(frames.size()) << " untracked in view per frame\n";
    if (!enabled) {
      landmarks_off = landmarks;
      require(reassociated == 0 && merged == 0, "re-association while disabled");
      continue;
    }
    require(reassociated > 300, "too few re-associations");
    require(wrong * 100 <= reassociated, "more than 1% of re-associations joined different points");
    require(landmarks * 10 < landmarks_off * 9, "re-association did not reduce duplicate landmarks");
    if (mode == 1) {
      landmarks_no_merge = landmarks;
      require(merged == 0, "merged while merging is disabled");
    } else {
      require(merged > 0 && landmarks < landmarks_no_merge, "merging did not reduce duplicate landmarks");
    }
  }
}

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

// Similarity alignment (exact and RANSAC) and the Sim3 pose graph: a drifted
// chain of keyframes around a circle is pulled back by one loop edge.
void loop_closure_maths() {
  std::mt19937 rng(9);
  std::uniform_real_distribution<double> u(-1, 1);
  const vo::Sim3 truth{1.7, vo::exp_so3(vo::Vec3(0.3, -0.2, 0.5)), vo::Vec3(0.4, -1, 2)};
  std::vector<vo::Vec3> X, Y;
  for (int i = 0; i < 60; ++i) {
    X.emplace_back(3 * u(rng), 3 * u(rng), 3 * u(rng));
    Y.push_back(truth * X.back());
  }
  const vo::Sim3 fit = vo::align_similarity(X, Y);
  require(std::abs(fit.s - truth.s) < 1e-9 && (fit.t - truth.t).norm() < 1e-9 &&
          vo::log_so3(fit.R * truth.R.transpose()).norm() < 1e-9, "align_similarity is not exact");
  for (int i = 0; i < 60; i += 2) Y[static_cast<std::size_t>(i)] += vo::Vec3(5 * u(rng), 5 * u(rng), 5 * u(rng));
  vo::Sim3 model;
  std::vector<char> inliers;
  const int count = vo::ransac_similarity(X, Y, std::vector<double>(X.size(), 0.01), 300, 3, model, inliers);
  require(count == 30 && std::abs(model.s - truth.s) < 1e-6, "ransac_similarity with 50% outliers");

  // Ground truth: 40 cameras on a circle looking outwards (world -> camera).
  const int n = 40;
  std::vector<vo::Sim3> poses;
  for (int i = 0; i < n; ++i) {
    const double a = 2 * M_PI * i / n;
    const vo::Mat3 Rwc = Eigen::AngleAxisd(-a, vo::Vec3::UnitY()).toRotationMatrix();
    const vo::Vec3 c(3 * std::sin(a), 0, 3 * std::cos(a));
    poses.push_back({1, Rwc.transpose(), -(Rwc.transpose() * c)});
  }
  // Odometry: true relative motions with a little rotation, translation and
  // scale error each; integrated, the chain drifts.
  vo::PoseGraphProblem problem;
  problem.nodes.push_back(poses[0]);
  for (int i = 1; i < n; ++i) {
    vo::Sim3 relative = poses[i] * poses[i - 1].inverse();
    relative = vo::Sim3{1.01, vo::exp_so3(vo::Vec3(0.002, 0.004, -0.002)), vo::Vec3(0.004, 0.002, 0)} * relative;
    problem.nodes.push_back(relative * problem.nodes.back());
  }
  for (int i = 1; i < n; ++i)
    problem.edges.push_back({i, i - 1, problem.nodes[i] * problem.nodes[i - 1].inverse(), 1.0});
  // Loop: the last camera relative to the first, as truth says.
  problem.edges.push_back({n - 1, 0, poses[n - 1] * poses[0].inverse(), 10.0});
  problem.fixed.assign(n, 0);
  problem.fixed[0] = 1;
  const auto centre_error = [&](const std::vector<vo::Sim3>& nodes) {
    double worst = 0;
    for (int i = 0; i < n; ++i) {
      const vo::Vec3 c = -(nodes[i].R.transpose() * nodes[i].t) / nodes[i].s;
      const vo::Vec3 t = -(poses[i].R.transpose() * poses[i].t);
      worst = std::max(worst, (c - t).norm());
    }
    return worst;
  };
  const double before = centre_error(problem.nodes);
  const auto report = vo::optimize_pose_graph(problem);
  const double after = centre_error(problem.nodes);
  std::cout << "pose graph: worst camera-centre error " << before << " -> " << after << " (cost "
            << report.initial_cost << " -> " << report.final_cost << ", " << report.iterations << " iterations)\n";
  require(after < 0.2 * before, "pose graph did not pull the drifted chain back");
}

// A camera circling inside a cylinder of points (1.2 laps, so the start is
// revisited with new track IDs), with a slightly wrong focal length so the
// VO drifts. Colour encodes the point index, as in make_scene.
Scene make_loop_scene(int frame_count = 700, int lap = 480) {
  std::mt19937 rng(21);
  std::uniform_real_distribution<double> u(0, 1);
  std::normal_distribution<double> noise(0, 0.5);
  const int width = 1920, height = 1080;
  const auto K = CameraIntrinsics::from_horizontal_fov(width, height, 60);
  Scene scene;
  for (int i = 0; i < 3000; ++i) {
    const double a = 2 * M_PI * u(rng);
    scene.points.emplace_back(6 * std::sin(a), -2 + 4 * u(rng), 6 * std::cos(a));
  }
  std::vector<std::uint64_t> id(scene.points.size());
  std::uint64_t next_id = 0;
  for (auto& v : id) v = next_id++;
  std::vector<char> seen(scene.points.size(), 0);
  for (int f = 0; f < frame_count; ++f) {
    const double theta = 2 * M_PI * f / lap;
    const Eigen::Vector3d c(2 * std::sin(theta), 0.05 * std::sin(5 * theta), 2 * std::cos(theta));
    const Eigen::Matrix3d Rwc = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY()).toRotationMatrix();  // looks outwards
    const Eigen::Matrix3d R = Rwc.transpose();
    const Eigen::Vector3d t = -R * c;
    scene.centers.push_back(c);
    scene.rotations.push_back(R);
    TrackedFrame frame{static_cast<std::uint64_t>(f), f * 16'666'667LL, width, height, {}, 0, {}};
    for (std::size_t p = 0; p < scene.points.size(); ++p) {
      const Eigen::Vector3d xc = R * scene.points[p] + t;
      bool visible = xc.z() > 0.5;
      double x = 0, y = 0;
      if (visible) {
        x = K.fx * xc.x() / xc.z() + K.cx + noise(rng);
        y = K.fy * xc.y() / xc.z() + K.cy + noise(rng);
        visible = x >= 0 && y >= 0 && x < width && y < height && u(rng) >= 0.03;
      }
      if (!visible) {
        if (seen[p]) id[p] = next_id++;
        seen[p] = 0;
        continue;
      }
      seen[p] = 1;
      frame.observations.push_back({id[p], static_cast<float>(x), static_cast<float>(y), true,
                                    {static_cast<std::uint8_t>(p % 256), static_cast<std::uint8_t>(p / 256), 77}});
    }
    scene.frames.push_back(std::move(frame));
  }
  return scene;
}

void closes_a_loop() {
  const auto scene = make_loop_scene();
  const auto frames = occluded(scene, -1, -1);  // descriptors, no occlusion
  double ate_without = 0;
  for (const int mode : {0, 1, 2}) {  // off, synchronous, worker thread
    VisualOdometryConfig config;
    // True 60: a 0.5% focal error drifts ~1.5 deg of rotation per lap (larger
    // drift needs appearance retrieval; spatial candidates search 160 px).
    config.horizontal_fov_degrees = 60.3;
    config.loop_closure = mode > 0;
    config.loop_synchronous = mode == 1;
    VisualOdometry odometry(config);
    int detected = -1, closed = -1;
    std::string event;
    for (const auto& frame : frames) {
      const auto r = odometry.process(frame);
      if (r.loop_detected && detected < 0) detected = static_cast<int>(r.frame_index);
      if (r.loop_closed && closed < 0) {
        closed = static_cast<int>(r.frame_index);
        event = r.event;
        require(!r.keyframe_scale_changes.empty(), "a correction must report keyframe scale changes");
      }
    }
    const auto result = evaluate(scene, odometry.trajectory());
    static const char* names[] = {"off", "synchronous", "worker"};
    std::cout << "loop scene, loop closure " << names[mode] << ": " << result.segments << " segment(s), ATE "
              << 100 * result.ate_fraction << "% of path";
    if (mode > 0) std::cout << "; detected at " << detected << ", closed at " << closed << ": " << event;
    std::cout << '\n';
    if (mode == 0) {
      require(detected < 0 && closed < 0 && odometry.map_generation() == 0, "loop closure while disabled");
      ate_without = result.ate_fraction;
      continue;
    }
    require(detected >= 480 && closed >= detected, "the revisit was not detected and closed");
    require(odometry.map_generation() > 0, "a loop correction must change the map generation");
    require(mode == 1 ? closed == detected : closed >= detected, "synchronous corrections apply in the same frame");
    require(result.ate_fraction <= 1.05 * ate_without, "loop closure made the trajectory worse");
  }
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

// Uncertainty diagnostics (CONFIDENCE_DESIGN.md phase 0): every final map
// point carries one, low-confidence points really are worse, and the
// predicted relative depth sigma is of the right order (normalized error ~1).
void reports_uncertainty() {
  const auto scene = make_scene(true, 240, 5, {0.7, 0.08, 0.01, 0, 0});  // pixel noise, no movers/outliers
  VisualOdometry odometry;
  int tracked = 0, with_sigma = 0;
  for (const auto& frame : scene.frames) {
    const auto r = odometry.process(frame);
    if (r.state != OdometryState::tracking || !r.has_pose) continue;
    ++tracked;
    with_sigma += std::isfinite(r.rotation_sigma_degrees) && r.rotation_sigma_degrees > 0 &&
                  std::isfinite(r.translation_sigma_ratio) && r.translation_sigma_ratio > 0;
  }
  require(tracked > 100 && with_sigma >= tracked - 2, "pose uncertainty missing on tracked frames");
  auto map = odometry.retired_map();
  require(map.size() > 200, "too few retired landmarks");
  Eigen::Matrix3Xd estimated(3, map.size()), truth(3, map.size());
  for (std::size_t i = 0; i < map.size(); ++i) {
    const auto& point = map[i];
    require(!std::isnan(point.depth_sigma_ratio) && point.confidence >= 0 && point.confidence <= 1 &&
            point.max_parallax_degrees > 0, "retired landmark without uncertainty");
    estimated.col(i) = Eigen::Vector3d(point.position[0], point.position[1], point.position[2]);
    truth.col(i) = scene.points[point.color[0] + 256U * point.color[1]];
  }
  // Align by the camera trajectory (a few wild low-parallax points would
  // dominate a fit on the map itself).
  const auto trajectory = odometry.trajectory();
  Eigen::Matrix3Xd centers(3, trajectory.size()), true_centers(3, trajectory.size());
  for (std::size_t i = 0; i < trajectory.size(); ++i) {
    const auto c = trajectory[i].pose.center();
    centers.col(i) = Eigen::Vector3d(c[0], c[1], c[2]);
    true_centers.col(i) = scene.centers[trajectory[i].frame_index];
  }
  const Eigen::Matrix4d S = Eigen::umeyama(centers, true_centers, true);
  std::vector<double> confident, doubtful, normalized;
  for (std::size_t i = 0; i < map.size(); ++i) {
    const Eigen::Vector3d X = S.topLeftCorner<3, 3>() * estimated.col(i) + S.topRightCorner<3, 1>();
    const double relative = (X - truth.col(i)).norm() / (truth.col(i) - scene.centers[map[i].last_frame]).norm();
    // Precision only (this scene has no wrong matches, so the view-count
    // verification in `confidence` carries no information here): the depth
    // sigma bands where precision 1 / (1 + (r / 0.05)^2) is > 0.9 and < 0.5.
    if (map[i].depth_sigma_ratio < 0.05F / 3) confident.push_back(relative);
    if (map[i].depth_sigma_ratio > 0.05F) doubtful.push_back(relative);
    if (std::isfinite(map[i].depth_sigma_ratio)) normalized.push_back(relative / map[i].depth_sigma_ratio);
  }
  const auto median = [](std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
    return v[v.size() / 2];
  };
  require(confident.size() > 50 && doubtful.size() > 20, "confidence does not spread over the map");
  const double good = median(confident), bad = median(doubtful), ratio = median(normalized);
  std::cout << "uncertainty: " << map.size() << " retired points, relative error median " << good
            << " (precise, n=" << confident.size() << ") vs " << bad << " (imprecise, n=" << doubtful.size()
            << "), error / predicted sigma median " << ratio << ", pose sigma on " << with_sigma << "/" << tracked
            << " tracked frames\n";
  require(bad > 1.5 * good, "imprecise points are not worse than precise ones");  // measured 2.2x
  require(ratio > 0.2 && ratio < 5, "predicted depth sigma is off by more than 5x");
  // The covariance's longest axis (what the viewer draws as the ellipsoid's
  // long axis) lies along the viewing ray: depth is the weak direction.
  std::vector<double> alignment;
  for (const auto& point : map) {
    const auto shape = covariance_shape(point.covariance, std::numeric_limits<float>::max());
    int longest = 0;
    double best = -1;
    for (int k = 0; k < 3; ++k) {
      const double length = Eigen::Vector3d(shape[3 * k], shape[3 * k + 1], shape[3 * k + 2]).norm();
      if (length > best) { best = length; longest = k; }
    }
    if (!(best > 0)) continue;
    const Eigen::Vector3d axis = Eigen::Vector3d(shape[3 * longest], shape[3 * longest + 1], shape[3 * longest + 2]) / best;
    // Ray from the camera that last saw it, in the VO's own frame.
    const auto it = std::lower_bound(trajectory.begin(), trajectory.end(), point.last_frame,
                                     [](const TrajectorySample& s, std::uint64_t f) { return s.frame_index < f; });
    if (it == trajectory.end()) continue;
    const auto c = it->pose.center();
    const Eigen::Vector3d ray = (Eigen::Vector3d(point.position[0], point.position[1], point.position[2]) -
                                 Eigen::Vector3d(c[0], c[1], c[2])).normalized();
    alignment.push_back(std::abs(axis.dot(ray)));
  }
  const double aligned = median(alignment);
  std::cout << "uncertainty ellipsoids: median |cos(long axis, viewing ray)| " << aligned << " over "
            << alignment.size() << " points\n";
  require(alignment.size() > 200 && aligned > 0.95, "covariance long axis is not along the viewing ray");
}

// Verification and consistency: on the default scene, the 80 independently
// moving points produce wrong landmarks (their observations agree with no
// static point). They die young, so they are over-represented among landmarks
// seen by only two keyframes, and when they do get 3+ views their windowed
// reprojection RMS is about twice the map's (median 1.04 vs 0.56). Their
// confidence (precision x verification x consistency) is much lower. Note: by gross position error alone the view count is confounded:
// in a forward walk the longest-seen static points sit near the focus of
// expansion, with the least parallax (the precision term covers that).
void confidence_flags_unverified_points() {
  const auto scene = make_scene(true);
  constexpr std::size_t kMovers = 80;  // make_scene: points [0, movers) move
  VisualOdometry odometry;
  for (const auto& frame : scene.frames) odometry.process(frame);
  int two = 0, two_movers = 0, many = 0, many_movers = 0;
  std::vector<double> mover_confidence, static_confidence, mover_precision, static_precision;
  for (const auto& point : odometry.retired_map()) {
    const bool mover = point.color[0] + 256U * point.color[1] < kMovers;
    if (point.keyframe_observations == 2) { ++two; two_movers += mover; }
    if (point.keyframe_observations >= 4) { ++many; many_movers += mover; }
    (mover ? mover_confidence : static_confidence).push_back(point.confidence);
    const double q = point.depth_sigma_ratio / 0.05;  // precision term alone (no view count)
    (mover ? mover_precision : static_precision).push_back(std::isfinite(q) ? 1 / (1 + q * q) : 0.0);
  }
  const auto rate = [](int k, int n) { return n ? 100.0 * k / n : 0.0; };
  const auto median = [](std::vector<double> v) {
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
    return v[v.size() / 2];
  };
  require(two > 50 && many > 50 && mover_confidence.size() > 20, "too few points per class");
  std::cout << "verification: moving-point landmarks are " << rate(two_movers, two) << "% of 2-view points (n="
            << two << ") vs " << rate(many_movers, many) << "% of >=4-view (n=" << many << "); median confidence "
            << median(mover_confidence) << " (moving, n=" << mover_confidence.size() << ") vs "
            << median(static_confidence) << " (static); precision alone " << median(mover_precision) << " vs "
            << median(static_precision) << "\n";
  require(rate(two_movers, two) > 2 * rate(many_movers, many), "wrong landmarks are not concentrated in 2-view points");
  // Measured: moving / static median 0.91 with precision alone, 0.76 adding
  // the view-count verification, 0.38 adding the reprojection consistency.
  const double with_views = median(mover_confidence) / median(static_confidence);
  const double precision_only = median(mover_precision) / median(static_precision);
  require(with_views < 0.6 && with_views < precision_only - 0.2,
          "verification and consistency do not flag wrong landmarks");
}

int main() {
  try {
    reports_uncertainty();
    confidence_flags_unverified_points();
    pose_optimizer_converges();
    p3p_recovers_pose();
    loop_closure_maths();
    relocalizes_after_occlusion();
    distortion_round_trip();
    reassociates_broken_tracks();
    closes_a_loop();
    // Noise-free data must be reproduced essentially exactly (regression for
    // rotation drift off SO(3), which made poses diverge geometrically).
    const auto clean = make_scene(true, 240, 5, {0, 0, 0, 0, 0});
    const auto exact = evaluate(clean, run(clean.frames));
    std::cout << "noise-free walk: " << exact.segments << " segment(s), ATE " << 100 * exact.ate_fraction
              << "% of path, median rotation error " << exact.rotation_deg << " deg\n";
    require(exact.segments == 1 && exact.ate_fraction < 1e-5 && exact.rotation_deg < 1e-4,
            "Noise-free trajectory is not exact");
    {
      // Retired landmarks are kept, carry their observations' colour, and
      // match the true points up to a similarity.
      VisualOdometry odometry;
      for (const auto& frame : clean.frames) odometry.process(frame);
      auto map = odometry.retired_map();
      const std::size_t retired = map.size();
      const auto active = odometry.active_map();
      map.insert(map.end(), active.begin(), active.end());
      require(retired > 200, "too few retired landmarks");
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
      std::cout << "noise-free map: " << retired << " retired + " << active.size() << " active points, RMS error "
                << rms << " (scene units)\n";
      require(rms < 1e-3, "retired map does not match the true points");
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
      // Named optional columns: colour, sigma and depth survive a round trip.
      TrackedFrame frame{7, 123, 1920, 1080, {}, 0, {}};
      TrackObservation a{1, 10.5F, 20.25F, true, {1, 2, 3}};
      a.sigma_px = 0.8F;
      a.depth_m = 4.5F;
      a.depth_sigma_m = 0.6F;
      TrackObservation b{2, 30, 40, true, {4, 5, 6}};  // depth unknown
      frame.observations = {a, b};
      const auto depth_path = std::filesystem::temp_directory_path() / "slam-native-vo-test-depth.csv";
      {
        std::ofstream out(depth_path);
        write_tracks_header(out, 1920, 1080, TrackColumns{true, true, true});
        write_tracks(out, frame, TrackColumns{true, true, true});
      }
      const auto back = read_tracks_csv(depth_path);
      std::filesystem::remove(depth_path);
      require(back.size() == 1 && back[0].observations.size() == 2, "depth CSV lost rows");
      const auto& p = back[0].observations[0];
      const auto& q = back[0].observations[1];
      require(p.has_color && p.color[2] == 3 && p.sigma_px == 0.8F && p.depth_m == 4.5F && p.depth_sigma_m == 0.6F &&
              q.depth_m == 0 && q.color[0] == 4, "named tracks CSV columns did not round-trip");
    }
    std::cout << "visual odometry checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
