#include "slam_native/landmark_depth.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_set>

namespace slam_native {
namespace {
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

Eigen::Matrix3d rotation_of(const Pose& pose) {
  Eigen::Matrix3d R;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) R(r, c) = pose.rotation[static_cast<std::size_t>(3 * r + c)];
  return R;
}
Eigen::Vector3d translation_of(const Pose& pose) {
  return {pose.translation[0], pose.translation[1], pose.translation[2]};
}

// Sorted-vector lookups (the VO reports inliers and tracked landmarks sorted by track id).
std::optional<std::uint64_t> landmark_of(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& track_landmark,
                                         std::uint64_t track) {
  auto it = std::lower_bound(track_landmark.begin(), track_landmark.end(), track,
                             [](const auto& entry, std::uint64_t value) { return entry.first < value; });
  if (it == track_landmark.end() || it->first != track) return std::nullopt;
  return it->second;
}

double huber_cost(double e, double delta) { return e <= delta ? 0.5 * e * e : delta * (e - 0.5 * delta); }
}  // namespace

void LandmarkObservationLog::record(const TrackedFrame& frame, const std::vector<std::array<float, 2>>& raw,
                                    const VisualOdometry& odometry) {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> track_landmark;
  for (const auto& t : odometry.tracked_landmarks()) track_landmark.emplace_back(t.track_id, t.landmark_id);
  record(frame, raw, odometry.config().distortion_k1, track_landmark, odometry.last().pose_inliers);
}

void LandmarkObservationLog::record(const TrackedFrame& frame, const std::vector<std::array<float, 2>>& raw,
                                    double k1,
                                    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& track_landmark,
                                    const std::vector<std::uint64_t>& inliers) {
  if (!raw.empty() && raw.size() != frame.observations.size())
    throw std::invalid_argument("LandmarkObservationLog: raw detections do not match the frame's observations");
  width_ = frame.width;
  height_ = frame.height;
  observations_.reserve(observations_.size() + frame.observations.size());
  for (std::size_t i = 0; i < frame.observations.size(); ++i) {
    const auto& o = frame.observations[i];
    Record record;
    auto& obs = record.observation;
    obs.frame_index = frame.frame_index;
    obs.track_id = o.track_id;
    obs.x = obs.ux = o.x;
    obs.y = obs.uy = o.y;
    if (k1 != 0) undistort_point(obs.ux, obs.uy, frame.width, frame.height, k1);
    obs.raw_x = obs.raw_ux = raw.empty() ? kNaN : raw[i][0];
    obs.raw_y = obs.raw_uy = raw.empty() ? kNaN : raw[i][1];
    if (k1 != 0 && std::isfinite(obs.raw_x) && std::isfinite(obs.raw_y))
      undistort_point(obs.raw_ux, obs.raw_uy, frame.width, frame.height, k1);
    record.landmark = landmark_of(track_landmark, o.track_id);
    obs.judged = record.landmark.has_value();
    obs.inlier = obs.judged && std::binary_search(inliers.begin(), inliers.end(), o.track_id);
    observations_.push_back(record);
  }
}

std::unordered_map<std::uint64_t, std::vector<LandmarkDepthObservation>> LandmarkObservationLog::attribute(
    const std::vector<MapPoint>& map, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& merges) const {
  std::unordered_map<std::uint64_t, std::uint64_t> alias;  // absorbed -> kept
  for (const auto& [absorbed, kept] : merges) alias[absorbed] = kept;
  const auto resolve = [&](std::uint64_t id) {
    for (auto it = alias.find(id); it != alias.end(); it = alias.find(id)) id = it->second;
    return id;
  };
  std::unordered_set<std::uint64_t> final_ids;
  for (const auto& point : map) final_ids.insert(point.landmark_id);
  // The landmark each track ends on: its last association, followed through merges.
  std::unordered_map<std::uint64_t, std::uint64_t> track_final;
  for (const auto& record : observations_)
    if (record.landmark) track_final[record.observation.track_id] = *record.landmark;
  for (auto& [track, id] : track_final) id = resolve(id);
  std::unordered_map<std::uint64_t, std::vector<LandmarkDepthObservation>> result;
  for (const auto& record : observations_) {
    auto it = track_final.find(record.observation.track_id);
    if (it == track_final.end() || !final_ids.count(it->second)) continue;
    result[it->second].push_back(record.observation);
  }
  for (auto& [id, list] : result)  // records are in frame order already; several tracks interleave
    std::stable_sort(list.begin(), list.end(), [](const LandmarkDepthObservation& l, const LandmarkDepthObservation& r) {
      return l.frame_index < r.frame_index;
    });
  return result;
}

double depth_in(const Pose& pose, const std::array<double, 3>& X) {
  const auto& R = pose.rotation;
  return R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + pose.translation[2];
}

std::optional<std::array<double, 3>> refit_landmark(const CameraIntrinsics& K,
                                                    const std::vector<FixedPoseObservation>& observations,
                                                    const std::array<double, 3>& initial, double huber_px,
                                                    int iterations) {
  if (observations.size() < 2) return std::nullopt;
  struct View {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Vector2d u;
  };
  std::vector<View> views;
  views.reserve(observations.size());
  for (const auto& o : observations) views.push_back({rotation_of(o.pose), translation_of(o.pose), {o.ux, o.uy}});

  // Returns false when the point is behind a camera.
  const auto evaluate = [&](const Eigen::Vector3d& X, double& cost, Eigen::Matrix3d* H, Eigen::Vector3d* g) {
    cost = 0;
    if (H) H->setZero();
    if (g) g->setZero();
    for (const auto& v : views) {
      const Eigen::Vector3d xc = v.R * X + v.t;
      if (!(xc.z() > 1e-9)) return false;
      const Eigen::Vector2d r(K.fx * xc.x() / xc.z() + K.cx - v.u.x(), K.fy * xc.y() / xc.z() + K.cy - v.u.y());
      const double e = r.norm();
      cost += huber_cost(e, huber_px);
      if (!H) continue;
      const double w = e <= huber_px ? 1.0 : huber_px / e;
      Eigen::Matrix<double, 2, 3> J;
      const double iz = 1.0 / xc.z();
      J << K.fx * iz, 0, -K.fx * xc.x() * iz * iz, 0, K.fy * iz, -K.fy * xc.y() * iz * iz;
      J = J * v.R;
      *H += w * J.transpose() * J;
      *g += w * J.transpose() * r;
    }
    return true;
  };

  Eigen::Vector3d X(initial[0], initial[1], initial[2]);
  double cost = 0;
  Eigen::Matrix3d H;
  Eigen::Vector3d g;
  if (!X.allFinite() || !evaluate(X, cost, &H, &g)) return std::nullopt;
  double lambda = 1e-3;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    bool stepped = false;
    for (int attempt = 0; attempt < 12 && !stepped; ++attempt) {
      Eigen::Matrix3d A = H;
      for (int k = 0; k < 3; ++k) A(k, k) += lambda * std::max(H(k, k), 1e-12);
      const Eigen::Vector3d d = A.ldlt().solve(-g);
      if (!d.allFinite()) return std::nullopt;
      const Eigen::Vector3d candidate = X + d;
      double trial = 0;
      if (evaluate(candidate, trial, nullptr, nullptr) && trial <= cost) {
        const bool converged = d.norm() <= 1e-10 * (1.0 + X.norm()) || cost - trial <= 1e-12 * (1.0 + cost);
        X = candidate;
        if (!evaluate(X, cost, &H, &g)) return std::nullopt;
        lambda = std::max(lambda / 3, 1e-9);
        stepped = true;
        if (converged) iteration = iterations;
      } else {
        lambda *= 10;
      }
    }
    if (!stepped) break;  // no decrease found: at a minimum within precision
  }
  if (!X.allFinite()) return std::nullopt;
  for (const auto& v : views)
    if (!((v.R * X + v.t).z() > 1e-9)) return std::nullopt;
  return std::array<double, 3>{X.x(), X.y(), X.z()};
}

std::vector<std::size_t> spread_indices(std::size_t n, std::size_t k) {
  std::vector<std::size_t> indices;
  if (k == 0) {
    for (std::size_t i = 0; i < n; ++i) indices.push_back(i);
    return indices;
  }
  if (n < k) return indices;
  if (k == 1) return {0};
  for (std::size_t i = 0; i < k; ++i)
    indices.push_back(static_cast<std::size_t>(std::llround(double(i) * double(n - 1) / double(k - 1))));
  return indices;
}

void write_landmark_depth_csv(std::ostream& out, const VisualOdometry& odometry, const LandmarkObservationLog& log,
                              const std::vector<std::size_t>& ks, double huber_px) {
  const auto& config = odometry.config();
  const CameraIntrinsics K = config.intrinsics ? *config.intrinsics :
      CameraIntrinsics::from_horizontal_fov(log.width(), log.height(), config.horizontal_fov_degrees);
  struct FramePose {
    Pose pose;
    int segment;
    bool predicted;
  };
  std::unordered_map<std::uint64_t, FramePose> frame_pose;
  for (const auto& sample : odometry.trajectory())
    frame_pose[sample.frame_index] = {sample.pose, sample.segment, sample.predicted};
  const auto map = odometry.map();
  const auto by_landmark = log.attribute(map, odometry.landmark_merges());
  std::map<std::uint64_t, std::vector<std::uint64_t>> sightings;  // landmark -> keyframes
  for (const auto& o : odometry.landmark_observations()) sightings[o.landmark_id].push_back(o.frame_index);

  out.precision(10);
  const auto name_of = [](std::size_t k) { return k == 0 ? std::string("all") : std::to_string(k); };
  out << "frame_index,landmark_id,segment,track_id,x,y,ux,uy,raw_x,raw_y,raw_ux,raw_uy,inlier,observations,"
         "depth_keyframes";
  for (bool raw : {false, true})
    for (auto k : ks) out << ",depth_" << (raw ? "raw_" : "fused_") << name_of(k);
  out << '\n';
  const auto number = [&](double value) {
    if (!std::isfinite(value)) out << "nan";
    else out << value;
  };

  for (const auto& point : map) {
    auto obs_it = by_landmark.find(point.landmark_id);
    auto kf_it = sightings.find(point.landmark_id);
    if (obs_it == by_landmark.end() || kf_it == sightings.end()) continue;
    const auto& observations = obs_it->second;
    const std::array<double, 3> X0{point.position[0], point.position[1], point.position[2]};
    // Observations usable for the re-solve, in frame order.
    std::vector<const LandmarkDepthObservation*> usable;
    for (const auto& o : observations) {
      auto fp = frame_pose.find(o.frame_index);
      if (fp == frame_pose.end() || fp->second.predicted || fp->second.segment != point.segment) continue;
      if (o.judged && !o.inlier) continue;
      usable.push_back(&o);
    }
    // Variants: [raw][k] -> re-solved position.
    std::vector<std::vector<std::optional<std::array<double, 3>>>> solved(2);
    for (int raw = 0; raw < 2; ++raw) {
      for (auto k : ks) {
        std::vector<FixedPoseObservation> fixed;
        for (auto index : spread_indices(usable.size(), k)) {
          const auto& o = *usable[index];
          const double ux = raw ? o.raw_ux : o.ux, uy = raw ? o.raw_uy : o.uy;
          if (!std::isfinite(ux) || !std::isfinite(uy)) continue;
          fixed.push_back({frame_pose.at(o.frame_index).pose, ux, uy});
        }
        // A raw subset with fewer than k observations (coasted frames) is not the variant asked for.
        const bool complete = k == 0 ? !fixed.empty() : fixed.size() == k;
        solved[raw].push_back(complete ? refit_landmark(K, fixed, X0, huber_px) : std::nullopt);
      }
    }
    for (auto frame : kf_it->second) {
      auto fp = frame_pose.find(frame);
      if (fp == frame_pose.end()) continue;
      // The log's observation of this landmark on that keyframe (one track per frame).
      auto o = std::find_if(observations.begin(), observations.end(),
                            [&](const LandmarkDepthObservation& c) { return c.frame_index == frame; });
      if (o == observations.end()) continue;
      out << frame << ',' << point.landmark_id << ',' << point.segment << ',' << o->track_id << ',' << o->x << ','
          << o->y << ',' << o->ux << ',' << o->uy << ',';
      number(o->raw_x); out << ',';
      number(o->raw_y); out << ',';
      number(o->raw_ux); out << ',';
      number(o->raw_uy); out << ',';
      out << (o->inlier ? 1 : 0) << ',' << usable.size() << ',';
      number(depth_in(fp->second.pose, X0));
      for (int raw = 0; raw < 2; ++raw)
        for (const auto& X : solved[raw]) {
          out << ',';
          number(X ? depth_in(fp->second.pose, *X) : std::numeric_limits<double>::quiet_NaN());
        }
      out << '\n';
    }
  }
}

}  // namespace slam_native
