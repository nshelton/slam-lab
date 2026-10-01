#include "slam_native/visual_odometry.hpp"

#include "vo_geometry.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace slam_native {
using namespace vo;

namespace {
constexpr double kDegree = 3.14159265358979323846 / 180.0;

Pose to_pose(const SE3& T) {
  Pose pose;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) pose.rotation[r * 3 + c] = T.R(r, c);
    pose.translation[r] = T.t(r);
  }
  return pose;
}
double median(std::vector<double> values) {
  if (values.empty()) return 0;
  const auto middle = values.begin() + values.size() / 2;
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}
}  // namespace

TrackedFrame tracked_frame_from(const FrameFeatures& frame, const std::vector<std::array<std::uint8_t, 3>>& colors) {
  TrackedFrame result;
  result.frame_index = frame.frame_index;
  result.timestamp_ns = frame.timestamp_ns;
  result.width = frame.width;
  result.height = frame.height;
  const std::size_t count = std::min(frame.keypoints.size(), frame.landmark_ids.size());
  constexpr int kDimension = 256;
  const bool descriptors = frame.track_descriptors.size() == frame.keypoints.size() * kDimension && count > 0;
  result.observations.reserve(count);
  if (descriptors) {
    result.descriptor_dimension = kDimension;
    result.descriptors.reserve(count * kDimension);
  }
  for (std::size_t k = 0; k < count; ++k) {
    if (k < frame.superpoint_supported.size() && !frame.superpoint_supported[k]) continue;
    TrackObservation o{frame.landmark_ids[k], frame.keypoints[k].x, frame.keypoints[k].y};
    if (k < frame.track_sigmas_px.size()) o.sigma_px = frame.track_sigmas_px[k];
    if (k < colors.size()) {
      o.has_color = true;
      o.color = colors[k];
    }
    result.observations.push_back(o);
    if (descriptors) {
      const auto row = frame.track_descriptors.begin() + static_cast<std::ptrdiff_t>(k * kDimension);
      result.descriptors.insert(result.descriptors.end(), row, row + kDimension);
    }
  }
  return result;
}

void undistort_point(float& x, float& y, int width, int height, double k1) {
  const double cx = 0.5 * (width - 1), cy = 0.5 * (height - 1);
  const double half_diagonal2 = 0.25 * (double(width) * width + double(height) * height);
  const double dx = x - cx, dy = y - cy;
  const double scale = 1.0 / (1.0 + k1 * (dx * dx + dy * dy) / half_diagonal2);
  x = static_cast<float>(cx + dx * scale);
  y = static_cast<float>(cy + dy * scale);
}

void distort_point(float& x, float& y, int width, int height, double k1) {
  const double cx = 0.5 * (width - 1), cy = 0.5 * (height - 1);
  const double half_diagonal2 = 0.25 * (double(width) * width + double(height) * height);
  const double dx = x - cx, dy = y - cy, ru = std::hypot(dx, dy);
  if (k1 == 0 || ru < 1e-12) return;
  // Solve ru = rd / (1 + k1 rd^2 / h^2) for the root continuous with rd = ru at k1 = 0.
  const double a = k1 * ru / half_diagonal2;
  const double rd = 2 * ru / (1 + std::sqrt(std::max(0.0, 1 - 4 * a * ru)));
  x = static_cast<float>(cx + dx * rd / ru);
  y = static_cast<float>(cy + dy * rd / ru);
}

TrackedFrame undistorted(const TrackedFrame& frame, double k1) {
  TrackedFrame result = frame;
  if (k1 != 0)
    for (auto& o : result.observations) undistort_point(o.x, o.y, frame.width, frame.height, k1);
  return result;
}

CameraIntrinsics CameraIntrinsics::from_horizontal_fov(int width, int height, double degrees) {
  const double f = 0.5 * width / std::tan(0.5 * degrees * kDegree);
  return {f, f, 0.5 * (width - 1), 0.5 * (height - 1)};
}

std::array<double, 3> Pose::center() const {
  std::array<double, 3> c{};
  for (int i = 0; i < 3; ++i)
    for (int r = 0; r < 3; ++r) c[i] -= rotation[r * 3 + i] * translation[r];
  return c;
}

void VisualOdometryConfig::validate() const {
  const bool ok = horizontal_fov_degrees > 1 && horizontal_fov_degrees < 179 && min_init_tracks >= 8 &&
      init_min_flow_fraction >= 0 && init_min_parallax_degrees >= 0 && min_init_points >= 8 &&
      homography_ratio > 0 && distortion_k1 > -1 && ransac_threshold_px > 0 && ransac_iterations > 0 && max_init_frames > 0 &&
      min_tracked_points >= 6 && reprojection_threshold_px > 0 && keyframe_min_interval >= 1 &&
      keyframe_max_interval >= keyframe_min_interval && keyframe_track_ratio > 0 &&
      min_triangulation_parallax_degrees >= 0 && window_keyframes >= 2 && bundle_iterations >= 0 &&
      (!intrinsics || (intrinsics->fx > 0 && intrinsics->fy > 0)) && relocalization_max_frames >= 1 &&
      relocalization_min_keyframes >= 2 && relocalization_radius_px > 0 && relocalization_threshold_px > 0 &&
      relocalization_iterations >= 1 && relocalization_min_inliers >= 6 && guided_search_radius_px >= 0 &&
      reassociation_radius_px > 0 && dormant_max_frames >= 0 && max_dormant_landmarks >= 0 &&
      depth_relative_sigma > 0 && depth_scale_min_samples >= 1 && depth_scale_min_keyframes >= 2 &&
      depth_scale_min_parallax_degrees >= 0 && depth_scale_max_relative_sigma > 0 && depth_scale_drift_sigma >= 0 &&
      depth_scale_floor_sigma > 0 &&
      observation_sigma_px > 0 && min_observation_sigma_px > 0 && confidence_depth_ratio > 0 &&
      confidence_two_view >= 0 && confidence_two_view <= 1 && confidence_consistency_scale > 0 &&
      pose_landmark_covariance_scale > 0;
  if (!ok) throw std::invalid_argument("Invalid visual odometry settings");
}

const char* to_string(OdometryState state) {
  switch (state) {
    case OdometryState::initializing: return "initializing";
    case OdometryState::tracking: return "tracking";
    case OdometryState::lost: return "lost";
  }
  return "unknown";
}

struct VisualOdometry::Impl {
  struct KeyFrame {
    int id{};
    std::uint64_t frame_index{};
    int segment{};
    SE3 pose;
    std::unordered_map<std::uint64_t, Vec2> observations;
    std::unordered_map<std::uint64_t, double> sigmas;  // resolved pixel sigma (see sigma_map), by track
  };
  // A landmark's observation in a keyframe. The sigma is stored with it, as
  // several tracks can observe one landmark over its life.
  struct Sighting {
    int keyframe;
    Vec2 u;
    double sigma;
    // Whitened reprojection error, last measured while this keyframe was in
    // the BA window (see cull). Old keyframes drift relative to the window,
    // so measuring them against the current point would penalise age, not
    // inconsistency. NaN until first measured.
    double error{std::numeric_limits<double>::quiet_NaN()};
  };
  struct Landmark {
    Vec3 X;
    std::vector<Sighting> observations;
    std::array<double, 3> color_sum{};
    int color_count{};
    std::uint64_t first_frame{}, last_frame{};
    int segment{};
    std::vector<float> descriptor;  // latest observed; empty without descriptors
    std::uint64_t track{};               // latest track observing it
    std::vector<std::uint64_t> tracks;   // every track that has observed it
    // Diagnostics (update_quality): geometry-only covariance, keyframes fixed.
    Mat3 covariance{Mat3::Zero()};
    double depth_sigma_ratio{std::numeric_limits<double>::quiet_NaN()};
    double max_parallax{};  // radians
    double reprojection_rms{std::numeric_limits<double>::quiet_NaN()};  // observation sigmas
  };
  // A lost segment kept for relocalization.
  struct LostSegment {
    int segment{};
    std::vector<int> keyframes;
    std::unordered_map<std::uint64_t, Landmark> landmarks;
    SE3 last_pose, velocity;
    std::uint64_t last_frame{};  // frame of last_pose
  };
  struct FrameRecord {
    std::uint64_t frame_index{};
    std::int64_t timestamp_ns{};
    int segment{};
    int keyframe{};   // reference keyframe id
    SE3 relative;     // frame pose = relative * keyframe pose
    bool is_keyframe{};
  };

  VisualOdometryConfig config;
  bool calibrated{};
  Intrinsics K;
  int width{}, height{};
  OdometryState state{OdometryState::initializing};
  int segment{};
  int next_segment{1};                   // segment ids are never reused
  std::vector<KeyFrame> keyframes;       // all segments; id == index
  std::vector<int> segment_keyframes;    // ids in the current segment
  // All landmark maps (landmarks, dormant, lost->landmarks) are keyed by
  // landmark id, never by track id.
  std::unordered_map<std::uint64_t, Landmark> landmarks;
  // Live track -> the active landmark it observes. One live track per
  // landmark; dormant and lost landmarks are not indexed (their tracks ended).
  std::unordered_map<std::uint64_t, std::uint64_t> track_landmark;
  std::uint64_t next_landmark_id{};
  // Landmarks of the current segment whose tracks ended, kept (with their
  // latest descriptor) for relocalization_max_frames so a relocalization can
  // re-find them; then retired. Only with relocalization enabled.
  std::unordered_map<std::uint64_t, Landmark> dormant;
  std::vector<MapPoint> retired;
  std::unordered_map<std::uint64_t, std::array<std::uint8_t, 3>> colors;  // this frame's, by track
  std::unordered_map<std::uint64_t, double> sigmas;  // this frame's resolved pixel sigma, by track
  std::unordered_map<std::uint64_t, std::pair<double, double>> depths;  // this frame's (depth, sigma) m, by track
  struct DepthScale {
    double log_scale{}, variance{};
    int keyframes{};
  };
  std::unordered_map<int, DepthScale> depth_scales;  // by segment
  std::optional<TrackedFrame> reference;
  int reference_age{};
  std::vector<FrameRecord> frames;
  SE3 last_pose, previous_pose;
  std::uint64_t last_pose_frame{};
  std::optional<LostSegment> lost;
  std::unordered_map<std::uint64_t, const float*> descriptors;  // this frame's, by track
  int descriptor_dimension{};
  int frames_since_keyframe{};
  int keyframe_landmarks{};
  double typical_rms{};  // see update_typical_rms; 0 = not known yet
  OdometryFrameResult last;

  static std::unordered_map<std::uint64_t, Vec2> observation_map(const TrackedFrame& frame) {
    std::unordered_map<std::uint64_t, Vec2> map;
    map.reserve(frame.observations.size());
    for (const auto& o : frame.observations) map.emplace(o.track_id, Vec2(o.x, o.y));
    return map;
  }
  Landmark new_landmark(const Vec3& X, std::uint64_t first_frame, std::uint64_t last_frame,
                        std::uint64_t track) const {
    Landmark landmark;
    landmark.X = X;
    landmark.first_frame = first_frame;
    landmark.last_frame = last_frame;
    landmark.track = track;
    landmark.tracks = {track};
    return landmark;
  }
  // Active landmarks and track_landmark change together through these.
  std::uint64_t add_active(Landmark landmark) {
    const std::uint64_t id = next_landmark_id++;
    track_landmark[landmark.track] = id;
    landmarks.emplace(id, std::move(landmark));
    return id;
  }
  void unindex(std::uint64_t id, const Landmark& landmark) {
    for (auto track : landmark.tracks) {
      auto it = track_landmark.find(track);
      if (it != track_landmark.end() && it->second == id) track_landmark.erase(it);
    }
  }
  std::unordered_map<std::uint64_t, Landmark>::iterator erase_active(
      std::unordered_map<std::uint64_t, Landmark>::iterator it) {
    unindex(it->first, it->second);
    return landmarks.erase(it);
  }
  // `track` becomes the (only) live track of active landmark `id`.
  void attach(std::uint64_t id, std::uint64_t track) {
    Landmark& landmark = landmarks.at(id);
    unindex(id, landmark);
    landmark.track = track;
    if (std::find(landmark.tracks.begin(), landmark.tracks.end(), track) == landmark.tracks.end())
      landmark.tracks.push_back(track);
    track_landmark[track] = id;
  }
  // Fold active landmark `absorbed` into active landmark `kept`: keyframe
  // observations (kept's wins where both saw a keyframe), tracks, colour and
  // frame range. The absorbed id disappears; kept's position stays (BA and
  // cull() then refine it and drop observations that do not fit).
  void merge(std::uint64_t kept_id, std::uint64_t absorbed_id) {
    auto node = landmarks.extract(absorbed_id);
    const Landmark& absorbed = node.mapped();
    unindex(absorbed_id, absorbed);
    Landmark& kept = landmarks.at(kept_id);
    for (const auto& o : absorbed.observations) {
      const bool seen = std::any_of(kept.observations.begin(), kept.observations.end(),
                                    [&](const Sighting& k) { return k.keyframe == o.keyframe; });
      if (!seen) kept.observations.push_back(o);
    }
    std::sort(kept.observations.begin(), kept.observations.end(),
              [](const Sighting& l, const Sighting& r) { return l.keyframe < r.keyframe; });
    for (int c = 0; c < 3; ++c) kept.color_sum[c] += absorbed.color_sum[c];
    kept.color_count += absorbed.color_count;
    kept.first_frame = std::min(kept.first_frame, absorbed.first_frame);
    kept.last_frame = std::max(kept.last_frame, absorbed.last_frame);
    for (auto track : absorbed.tracks)
      if (std::find(kept.tracks.begin(), kept.tracks.end(), track) == kept.tracks.end()) kept.tracks.push_back(track);
  }
  // Pixel noise used to weight an observation: the tracker's sigma_px (floored)
  // with use_observation_sigma, else the uniform observation_sigma_px.
  std::unordered_map<std::uint64_t, double> sigma_map(const TrackedFrame& frame) const {
    std::unordered_map<std::uint64_t, double> map;
    map.reserve(frame.observations.size());
    for (const auto& o : frame.observations)
      map.emplace(o.track_id, config.use_observation_sigma && o.sigma_px > 0 ?
                                  std::max<double>(o.sigma_px, config.min_observation_sigma_px) :
                                  config.observation_sigma_px);
    return map;
  }
  double sigma_of(std::uint64_t track_id) const {
    auto it = sigmas.find(track_id);
    return it != sigmas.end() ? it->second : config.observation_sigma_px;
  }
  Sighting sighting(int kf, std::uint64_t track_id) const {
    const auto& keyframe = keyframes[kf];
    auto it = keyframe.sigmas.find(track_id);
    return {kf, keyframe.observations.at(track_id),
            it != keyframe.sigmas.end() ? it->second : config.observation_sigma_px};
  }
  Vec2 normalized(const Vec2& u) const { return K.unproject(u).head<2>(); }
  bool reprojects(const SE3& T, const Vec3& X, const Vec2& u, double threshold) const {
    const Vec3 xc = T * X;
    return xc.z() > 1e-6 && (K.project(xc) - u).squaredNorm() <= threshold * threshold;
  }

  void add_color(Landmark& landmark, std::uint64_t track_id,
                 const std::unordered_map<std::uint64_t, std::array<std::uint8_t, 3>>& source) const {
    auto it = source.find(track_id);
    if (it == source.end()) return;
    for (int c = 0; c < 3; ++c) landmark.color_sum[c] += it->second[c];
    ++landmark.color_count;
  }
  static std::unordered_map<std::uint64_t, std::array<std::uint8_t, 3>> color_map(const TrackedFrame& frame) {
    std::unordered_map<std::uint64_t, std::array<std::uint8_t, 3>> map;
    for (const auto& o : frame.observations)
      if (o.has_color) map.emplace(o.track_id, o.color);
    return map;
  }
  MapPoint map_point(std::uint64_t id, const Landmark& landmark) const {
    MapPoint point;
    point.landmark_id = id;
    point.track_id = landmark.track;
    point.position = {float(landmark.X.x()), float(landmark.X.y()), float(landmark.X.z())};
    point.has_color = landmark.color_count > 0;
    for (int c = 0; c < 3 && point.has_color; ++c)
      point.color[c] = static_cast<std::uint8_t>(std::lround(landmark.color_sum[c] / landmark.color_count));
    point.segment = landmark.segment;
    point.keyframe_observations = static_cast<int>(landmark.observations.size());
    point.first_frame = landmark.first_frame;
    point.last_frame = landmark.last_frame;
    point.depth_sigma_ratio = static_cast<float>(landmark.depth_sigma_ratio);
    point.max_parallax_degrees = static_cast<float>(landmark.max_parallax / kDegree);
    point.reprojection_rms = static_cast<float>(landmark.reprojection_rms);
    if (std::isfinite(landmark.depth_sigma_ratio)) {
      // Precision (depth sigma if the matches are right) times verification:
      // two views always fit almost exactly, so a wrong match is invisible;
      // every further consistent view halves the remaining doubt.
      const double q = landmark.depth_sigma_ratio / config.confidence_depth_ratio;
      const int extra_views = std::max(0, static_cast<int>(landmark.observations.size()) - 2);
      const double verification =
          1.0 - (1.0 - config.confidence_two_view) * std::pow(0.5, static_cast<double>(extra_views));
      // Consistency (3+ views): windowed reprojection RMS relative to the
      // map's typical RMS, which self-calibrates for a mis-scaled sigma.
      double consistency = 1.0;
      if (extra_views > 0 && std::isfinite(landmark.reprojection_rms) && typical_rms > 0) {
        const double excess = std::max(0.0, landmark.reprojection_rms / typical_rms - 1.0) /
                              config.confidence_consistency_scale;
        consistency = 1.0 / (1.0 + excess * excess);
      }
      point.confidence = static_cast<float>(verification * consistency / (1.0 + q * q));
    } else if (std::isinf(landmark.depth_sigma_ratio)) {
      point.confidence = 0;  // unconstrained depth
    }
    const auto& C = landmark.covariance;
    point.covariance = {float(C(0, 0)), float(C(0, 1)), float(C(0, 2)), float(C(1, 1)), float(C(1, 2)), float(C(2, 2))};
    return point;
  }
  void retire(std::uint64_t id, const Landmark& landmark) { retired.push_back(map_point(id, landmark)); }
  // Pose tracking treats landmarks as exact; their position error then acts as
  // extra, direction-dependent pixel noise. Attach each landmark's covariance
  // projected through the predicted pose (J Sigma_X J^T, px^2); optimize_pose
  // then measures the residual as a Mahalanobis distance. Landmarks with
  // unknown/unconstrained covariance are left as they are.
  void inflate_by_landmark(std::vector<PoseObservation>& observations, const std::vector<std::uint64_t>& tracks,
                           const SE3& predicted) const {
    for (std::size_t i = 0; i < observations.size(); ++i) {
      const Mat3& covariance = landmarks.at(track_landmark.at(tracks[i])).covariance;
      if (covariance.isZero()) continue;
      const Vec3 xc = predicted * observations[i].X;
      if (xc.z() <= 1e-6) continue;
      const double z = xc.z();
      Eigen::Matrix<double, 2, 3> J;
      J << K.fx / z, 0, -K.fx * xc.x() / (z * z), 0, K.fy / z, -K.fy * xc.y() / (z * z);
      J = J * predicted.R;
      observations[i].landmark_covariance = config.pose_landmark_covariance_scale * (J * covariance * J.transpose());
    }
  }
  void report_pose_uncertainty(const std::vector<PoseObservation>& observations, const SE3& pose,
                               const std::vector<char>& inliers, OdometryFrameResult& result) const {
    const Eigen::Matrix<double, 6, 6> information = pose_information(K, observations, pose, inliers);
    const Eigen::Matrix<double, 6, 6> covariance = information.inverse();
    if (!covariance.allFinite()) return;
    std::vector<double> depths;
    for (std::size_t i = 0; i < observations.size(); ++i)
      if (inliers[i]) depths.push_back((pose * observations[i].X).z());
    const auto largest = [](const Mat3& block) {
      return std::sqrt(std::max(Eigen::SelfAdjointEigenSolver<Mat3>(block).eigenvalues()(2), 0.0));
    };
    // perturb() is a left perturbation: delta v moves the camera centre by -R^T v.
    result.rotation_sigma_degrees = largest(covariance.topLeftCorner<3, 3>()) / kDegree;
    if (!depths.empty()) result.translation_sigma_ratio = largest(covariance.bottomRightCorner<3, 3>()) / median(depths);
  }
  // Uncertainty diagnostics from the landmark's keyframe observations at the
  // keyframes' current poses. Reported only; no decision reads them.
  // Median windowed reprojection RMS over active landmarks with 3+ views:
  // the reference for MapPoint::confidence's consistency term.
  void update_typical_rms() {
    std::vector<double> values;
    for (const auto& [id, landmark] : landmarks)
      if (landmark.observations.size() >= 3 && std::isfinite(landmark.reprojection_rms))
        values.push_back(landmark.reprojection_rms);
    if (values.size() >= 20) typical_rms = median(std::move(values));
  }
  void update_quality(Landmark& landmark) const {
    const auto& obs = landmark.observations;
    if (obs.empty()) return;
    std::vector<PointView> views;
    views.reserve(obs.size());
    double squared = 0;
    int measured = 0;
    int latest = obs.front().keyframe;
    for (const auto& [kf, u, sigma, error] : obs) {
      views.push_back({keyframes[kf].pose, u, sigma});
      if (std::isfinite(error)) {
        squared += error * error;
        ++measured;
      }
      latest = std::max(latest, kf);
    }
    landmark.reprojection_rms = measured ? std::sqrt(squared / measured) : std::numeric_limits<double>::quiet_NaN();
    landmark.max_parallax = 0;
    for (std::size_t a = 0; a < views.size(); ++a)
      for (std::size_t b = a + 1; b < views.size(); ++b)
        landmark.max_parallax = std::max(landmark.max_parallax, parallax(views[a].camera, views[b].camera, landmark.X));
    const Mat3 information = point_information(K, landmark.X, views);
    const Eigen::SelfAdjointEigenSolver<Mat3> eigen(information);
    const auto& values = eigen.eigenvalues();  // ascending
    if (eigen.info() != Eigen::Success || !(values(2) > 0) || values(0) <= 1e-12 * values(2)) {
      landmark.covariance.setZero();
      landmark.depth_sigma_ratio = std::numeric_limits<double>::infinity();  // depth unconstrained
      return;
    }
    landmark.covariance = eigen.eigenvectors() * values.cwiseInverse().asDiagonal() * eigen.eigenvectors().transpose();
    const Vec3 ray = landmark.X - keyframes[latest].pose.center();
    const double distance = ray.norm();
    const Vec3 direction = ray / distance;
    landmark.depth_sigma_ratio = std::sqrt(direction.dot(landmark.covariance * direction)) / distance;
  }
  // New landmark bookkeeping: segment and the current frame's descriptor.
  void stamp(Landmark& landmark, std::uint64_t track_id) {
    landmark.segment = segment;
    update_descriptor(landmark, track_id);
  }
  void update_descriptor(Landmark& landmark, std::uint64_t track_id) const {
    auto it = descriptors.find(track_id);
    if (it != descriptors.end()) landmark.descriptor.assign(it->second, it->second + descriptor_dimension);
  }
  void expire_lost() {
    if (!lost) return;
    retire_sorted(lost->landmarks);
    lost.reset();
  }
  void retire_sorted(const std::unordered_map<std::uint64_t, Landmark>& source) {
    std::vector<std::uint64_t> ids;
    for (const auto& [id, landmark] : source) ids.push_back(id);
    std::sort(ids.begin(), ids.end());
    for (auto id : ids) retire(id, source.at(id));
  }
  // Retire dormant landmarks unseen for dormant_max_frames, and the oldest
  // beyond max_dormant_landmarks.
  void expire_dormant(std::uint64_t frame_index) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> by_age;  // (last frame, id)
    for (const auto& [id, landmark] : dormant) by_age.emplace_back(landmark.last_frame, id);
    std::sort(by_age.begin(), by_age.end());
    const std::size_t excess = by_age.size() > static_cast<std::size_t>(config.max_dormant_landmarks) ?
        by_age.size() - static_cast<std::size_t>(config.max_dormant_landmarks) : 0;
    std::vector<std::uint64_t> ids;
    for (std::size_t k = 0; k < by_age.size(); ++k)
      if (k < excess || by_age[k].first + static_cast<std::uint64_t>(config.dormant_max_frames) < frame_index)
        ids.push_back(by_age[k].second);
    std::sort(ids.begin(), ids.end());
    for (auto id : ids) {
      retire(id, dormant.at(id));
      dormant.erase(id);
    }
  }

  // `keep`: retire the segment's landmarks (they stay in the map) rather than
  // discarding them (a rejected initialization).
  void clear_segment(bool keep = true) {
    if (keep) {
      for (const auto& [id, landmark] : landmarks)
        if (landmark.observations.size() >= 2) retire(id, landmark);
      retire_sorted(dormant);
    }
    landmarks.clear();
    track_landmark.clear();
    dormant.clear();
    segment_keyframes.clear();
    reference.reset();
    reference_age = 0;
  }

  OdometryFrameResult process(const TrackedFrame& input) {
    const auto started = std::chrono::steady_clock::now();
    std::optional<TrackedFrame> corrected;
    if (config.distortion_k1 != 0) corrected = undistorted(input, config.distortion_k1);
    const TrackedFrame& frame = corrected ? *corrected : input;
    colors = color_map(frame);
    sigmas = sigma_map(frame);
    depths.clear();
    if (config.use_depth)
      for (const auto& o : frame.observations)
        if (o.depth_m > 0)
          depths.emplace(o.track_id, std::make_pair(double(o.depth_m), o.depth_sigma_m > 0 ? double(o.depth_sigma_m) :
                                                                       config.depth_relative_sigma * o.depth_m));
    descriptors.clear();
    descriptor_dimension = frame.descriptor_dimension;
    if (descriptor_dimension > 0 &&
        frame.descriptors.size() == frame.observations.size() * static_cast<std::size_t>(descriptor_dimension))
      for (std::size_t i = 0; i < frame.observations.size(); ++i)
        descriptors.emplace(frame.observations[i].track_id, frame.descriptors.data() + i * descriptor_dimension);
    else
      descriptor_dimension = 0;
    if (frame.width <= 0 || frame.height <= 0) throw std::invalid_argument("Invalid tracked frame size");
    if (!calibrated || frame.width != width || frame.height != height) {
      if (calibrated) {  // resolution change: start over
        clear_segment();
        expire_lost();
        state = OdometryState::initializing;
        segment = next_segment++;
      }
      width = frame.width;
      height = frame.height;
      const auto intrinsics = config.intrinsics ? *config.intrinsics :
          CameraIntrinsics::from_horizontal_fov(width, height, config.horizontal_fov_degrees);
      K = {intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy};
      calibrated = true;
    }
    OdometryFrameResult result;
    result.frame_index = frame.frame_index;
    if (lost && frame.frame_index > lost->last_frame + static_cast<std::uint64_t>(config.relocalization_max_frames))
      expire_lost();
    expire_dormant(frame.frame_index);
    if (!(lost && relocalize(frame, result))) {
      if (state == OdometryState::tracking) track(frame, result);
      else initialize(frame, result);
    }
    result.segment = segment;
    if (auto it = depth_scales.find(segment); it != depth_scales.end()) {
      result.metric_scale = std::exp(it->second.log_scale);
      result.metric_scale_sigma = std::sqrt(it->second.variance);
    }
    result.map_points = landmarks.size();
    result.keyframes = segment_keyframes.size();
    result.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    last = result;
    return result;
  }

  void initialize(const TrackedFrame& frame, OdometryFrameResult& result) {
    state = OdometryState::initializing;
    result.state = state;
    if (!reference) {
      reference = frame;
      reference_age = 0;
      result.event = "reference frame set";
      return;
    }
    ++reference_age;
    const auto restart = [&](const char* why) {
      reference = frame;
      reference_age = 0;
      result.event = why;
    };
    const auto current = observation_map(frame);
    std::vector<std::uint64_t> ids;
    std::vector<Vec2> ua, ub, a, b;
    std::vector<double> displacement;
    for (const auto& o : reference->observations) {
      auto it = current.find(o.track_id);
      if (it == current.end()) continue;
      ids.push_back(o.track_id);
      ua.emplace_back(o.x, o.y);
      ub.push_back(it->second);
      a.push_back(normalized(ua.back()));
      b.push_back(normalized(ub.back()));
      displacement.push_back((ub.back() - ua.back()).norm());
    }
    if (static_cast<int>(ids.size()) < config.min_init_tracks) return restart("too few common tracks; new reference");
    const double flow = median(displacement);
    if (flow < config.init_min_flow_fraction * width) {
      result.event = "waiting for motion (median flow " + std::to_string(static_cast<int>(flow)) + " px)";
      if (reference_age > config.max_init_frames) restart("no motion; new reference");
      return;
    }
    const double threshold = config.ransac_threshold_px / K.fx;
    const std::uint32_t seed = static_cast<std::uint32_t>(frame.frame_index * 2654435761U);
    const auto E = ransac_essential(a, b, threshold, config.ransac_iterations, seed);
    if (E.inlier_count < config.min_init_points) {
      result.event = "essential matrix: too few inliers";
      if (reference_age > config.max_init_frames) restart("initialization timeout; new reference");
      return;
    }
    const auto H = ransac_homography(a, b, threshold, config.ransac_iterations, seed + 1);
    if (H.inlier_count > config.homography_ratio * E.inlier_count) {
      result.event = "homography explains motion (rotation or plane); waiting";
      if (reference_age > config.max_init_frames) restart("initialization timeout; new reference");
      return;
    }
    // Choose the (R, t) with the most points in front of both cameras.
    const SE3 first;
    int best = -1, best_count = 0, second_count = 0;
    std::vector<std::vector<char>> good_sets;
    std::vector<std::vector<Vec3>> point_sets;
    const auto candidates = decompose_essential(E.model);
    for (std::size_t c = 0; c < candidates.size(); ++c) {
      std::vector<char> good(ids.size(), 0);
      std::vector<Vec3> X(ids.size(), Vec3::Zero());
      int count = 0;
      for (std::size_t i = 0; i < ids.size(); ++i) {
        if (!E.inliers[i] || !triangulate(first, a[i], candidates[c], b[i], X[i])) continue;
        good[i] = reprojects(first, X[i], ua[i], 2 * config.ransac_threshold_px) &&
                  reprojects(candidates[c], X[i], ub[i], 2 * config.ransac_threshold_px);
        count += good[i];
      }
      if (count > best_count) {
        second_count = best_count;
        best_count = count;
        best = static_cast<int>(c);
      } else {
        second_count = std::max(second_count, count);
      }
      good_sets.push_back(std::move(good));
      point_sets.push_back(std::move(X));
    }
    if (best < 0 || best_count < config.min_init_points || second_count > 0.7 * best_count) {
      result.event = "ambiguous two-view geometry; waiting";
      if (reference_age > config.max_init_frames) restart("initialization timeout; new reference");
      return;
    }
    SE3 second = candidates[best];
    auto& good = good_sets[best];
    auto& X = point_sets[best];
    std::vector<double> parallaxes, depths;  // of well-conditioned points
    int well_conditioned = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (!good[i]) continue;
      const double angle = parallax(first, second, X[i]) / kDegree;
      // Near the focus of expansion depth is unobservable; keep such points
      // out of the map (they would pass the reprojection test at any depth).
      if (angle < config.min_triangulation_parallax_degrees) {
        good[i] = 0;
        continue;
      }
      parallaxes.push_back(angle);
      depths.push_back(X[i].z());
      ++well_conditioned;
    }
    const double median_parallax = median(parallaxes);
    if (well_conditioned < config.min_init_points) {
      result.event = "too few well-conditioned points; waiting";
      if (reference_age > config.max_init_frames) restart("initialization timeout; new reference");
      return;
    }
    if (median_parallax < config.init_min_parallax_degrees) {
      result.event = "waiting for parallax (" + std::to_string(median_parallax).substr(0, 4) + " deg)";
      if (reference_age > config.max_init_frames) restart("initialization timeout; new reference");
      return;
    }
    // Arbitrary monocular scale: median scene depth of the first keyframe = 1.
    const double scale = 1.0 / median(depths);
    second.t *= scale;
    const int k0 = add_keyframe(reference->frame_index, first, observation_map(*reference), sigma_map(*reference));
    const int k1 = add_keyframe(frame.frame_index, second, current, sigmas);
    const auto reference_colors = color_map(*reference);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (!good[i]) continue;
      Landmark landmark = new_landmark(X[i] * scale, reference->frame_index, frame.frame_index, ids[i]);
      landmark.observations = {sighting(k0, ids[i]), sighting(k1, ids[i])};
      add_color(landmark, ids[i], reference_colors);
      add_color(landmark, ids[i], colors);
      stamp(landmark, ids[i]);
      add_active(std::move(landmark));
    }
    local_bundle_adjustment();
    cull(current);
    if (static_cast<int>(landmarks.size()) < config.min_init_points) {
      keyframes.resize(keyframes.size() - 2);
      clear_segment(false);
      restart("initial map too small after refinement; new reference");
      return;
    }
    const SE3& pose0 = keyframes[k0].pose;
    const SE3& pose1 = keyframes[k1].pose;
    frames.push_back({reference->frame_index, reference->timestamp_ns, segment, k0, SE3{}, true});
    frames.push_back({frame.frame_index, frame.timestamp_ns, segment, k1, SE3{}, true});
    const SE3 velocity = scale_motion(pose1 * pose0.inverse(), 1.0 / std::max(reference_age, 1));
    last_pose = pose1;
    last_pose_frame = frame.frame_index;
    previous_pose = velocity.inverse() * pose1;
    frames_since_keyframe = 0;
    keyframe_landmarks = static_cast<int>(landmarks.size());
    reference.reset();
    state = OdometryState::tracking;
    result.state = state;
    result.has_pose = true;
    result.keyframe = true;
    result.pose = to_pose(pose1);
    result.correspondences = result.inliers = static_cast<int>(landmarks.size());
    for (const auto& [id, landmark] : landmarks) result.pose_inliers.push_back(landmark.track);
    std::sort(result.pose_inliers.begin(), result.pose_inliers.end());
    result.event = "initialized: " + std::to_string(landmarks.size()) + " points, " +
                   std::to_string(median_parallax).substr(0, 4) + " deg parallax";
  }

  void track(const TrackedFrame& frame, OdometryFrameResult& result) {
    const auto current = observation_map(frame);
    std::vector<PoseObservation> observations;
    std::vector<std::uint64_t> ids;  // tracks
    for (const auto& [id, u] : current) {
      auto it = track_landmark.find(id);
      if (it == track_landmark.end()) continue;
      observations.push_back({landmarks.at(it->second).X, u, sigma_of(id)});
      ids.push_back(id);
    }
    // Deterministic order (unordered_map iteration is not).
    std::vector<std::size_t> order(ids.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto l, auto r) { return ids[l] < ids[r]; });
    {
      std::vector<PoseObservation> sorted_obs;
      std::vector<std::uint64_t> sorted_ids;
      for (auto k : order) { sorted_obs.push_back(observations[k]); sorted_ids.push_back(ids[k]); }
      observations = std::move(sorted_obs);
      ids = std::move(sorted_ids);
    }
    result.correspondences = static_cast<int>(observations.size());
    if (result.correspondences < config.min_tracked_points) return lose(frame, result, "too few tracked landmarks");
    const SE3 velocity = last_pose * previous_pose.inverse();
    SE3 pose = velocity * last_pose;
    if (config.pose_landmark_uncertainty) inflate_by_landmark(observations, ids, pose);
    std::vector<char> inliers;
    int count = optimize_pose(K, observations, pose, config.reprojection_threshold_px, inliers, 10,
                              config.pose_landmark_pixel_gate);
    if (count < std::max(config.min_tracked_points, result.correspondences / 2)) {
      SE3 fallback = last_pose;  // constant position instead of constant velocity
      std::vector<char> fallback_inliers;
      const int fallback_count = optimize_pose(K, observations, fallback, config.reprojection_threshold_px,
                                               fallback_inliers, 10, config.pose_landmark_pixel_gate);
      if (fallback_count > count) {
        pose = fallback;
        inliers = std::move(fallback_inliers);
        count = fallback_count;
      }
    }
    if (count < std::max(config.min_tracked_points, result.correspondences / 2)) {
      // Sudden motion: neither prediction converges. Re-estimate from scratch
      // (P3P RANSAC on the same correspondences) before declaring loss.
      SE3 recovered = pose;
      std::vector<char> recovered_inliers;
      const std::uint32_t seed = static_cast<std::uint32_t>(frame.frame_index * 3266489917U);
      if (ransac_pnp(K, observations, config.relocalization_threshold_px, config.relocalization_iterations, seed,
                     recovered, recovered_inliers) > count) {
        const int recovered_count = optimize_pose(K, observations, recovered, config.reprojection_threshold_px,
                                                  recovered_inliers, 10, config.pose_landmark_pixel_gate);
        if (recovered_count > count) {
          pose = recovered;
          inliers = std::move(recovered_inliers);
          count = recovered_count;
          result.event = "pose re-estimated (P3P RANSAC)";
        }
      }
    }
    if (count < config.min_tracked_points) return lose(frame, result, "pose optimization lost the map");
    report_pose_uncertainty(observations, pose, inliers, result);
    std::vector<double> errors;
    std::vector<std::uint64_t> inlier_ids, outlier_ids;
    for (std::size_t i = 0; i < observations.size(); ++i) {
      if (!inliers[i]) {
        outlier_ids.push_back(ids[i]);
        continue;
      }
      inlier_ids.push_back(ids[i]);
      auto& landmark = landmarks.at(track_landmark.at(ids[i]));
      landmark.last_frame = frame.frame_index;
      update_descriptor(landmark, ids[i]);
      errors.push_back((K.project(pose * observations[i].X) - observations[i].u).norm());
    }
    result.inliers = count;
    result.median_reprojection_px = median(errors);
    result.pose_inliers = inlier_ids;    // ids were sorted above
    result.pose_outliers = outlier_ids;
    previous_pose = last_pose;
    last_pose = pose;
    last_pose_frame = frame.frame_index;
    // Before the keyframe step, so a re-found landmark is not triangulated again.
    reassociate(frame, pose, current, std::unordered_set<std::uint64_t>(inlier_ids.begin(), inlier_ids.end()), result);
    ++frames_since_keyframe;
    const bool keyframe = frames_since_keyframe >= config.keyframe_min_interval &&
        (count < config.keyframe_track_ratio * keyframe_landmarks ||
         frames_since_keyframe >= config.keyframe_max_interval);
    if (keyframe) {
      const int id = add_keyframe(frame.frame_index, pose, current, sigmas);
      for (auto track_id : inlier_ids) {
        auto it = track_landmark.find(track_id);
        if (it == track_landmark.end()) continue;
        auto& landmark = landmarks.at(it->second);
        landmark.observations.push_back(sighting(id, track_id));
        add_color(landmark, track_id, colors);
      }
      // A live track whose landmark no longer fits usually has a poorly
      // conditioned depth from a short baseline. Drop the point and let it be
      // re-triangulated from the widest keyframe pair that saw the track.
      for (auto track_id : outlier_ids) {
        auto it = track_landmark.find(track_id);
        if (it != track_landmark.end()) erase_active(landmarks.find(it->second));
      }
      triangulate_new(id);
      local_bundle_adjustment();
      cull(current);
      update_depth_scale(id, result);
      last_pose = keyframes[id].pose;
      frames_since_keyframe = 0;
      keyframe_landmarks = 0;
      for (const auto& [landmark_id, landmark] : landmarks)
        for (const auto& o : landmark.observations) keyframe_landmarks += o.keyframe == id;
      frames.push_back({frame.frame_index, frame.timestamp_ns, segment, id, SE3{}, true});
      result.keyframe = true;
    } else {
      const int reference_keyframe = segment_keyframes.back();
      frames.push_back({frame.frame_index, frame.timestamp_ns, segment, reference_keyframe,
                        pose * keyframes[reference_keyframe].pose.inverse(), false});
    }
    result.state = state;
    result.has_pose = true;
    result.pose = to_pose(last_pose);
  }

  // Mutual-best descriptor matches between projected points (pixel,
  // descriptor) and this frame's observations accepted by `eligible` (track
  // id), within `radius` px and above the similarity threshold. Returns
  // (point index, observation index) pairs in point order.
  template <class Eligible>
  std::vector<std::pair<std::size_t, int>> match_projected(
      const TrackedFrame& frame, const std::vector<std::pair<Vec2, const std::vector<float>*>>& points,
      double radius, Eligible eligible) const {
    std::vector<std::pair<std::size_t, int>> matches;
    if (descriptor_dimension == 0 || points.empty()) return matches;
    const int cols = static_cast<int>(width / radius) + 1, rows = static_cast<int>(height / radius) + 1;
    std::vector<std::vector<int>> grid(static_cast<std::size_t>(cols) * rows);
    const auto cell = [&](double value, int count) { return std::clamp(static_cast<int>(value / radius), 0, count - 1); };
    for (int i = 0; i < static_cast<int>(frame.observations.size()); ++i) {
      const auto& o = frame.observations[i];
      if (!descriptors.count(o.track_id) || !eligible(o.track_id)) continue;
      grid[static_cast<std::size_t>(cell(o.y, rows)) * cols + cell(o.x, cols)].push_back(i);
    }
    const float min_similarity = static_cast<float>(config.relocalization_min_similarity);
    std::vector<std::pair<std::size_t, int>> best_of_point;
    std::unordered_map<int, std::pair<float, std::size_t>> best_of_observation;
    for (std::size_t k = 0; k < points.size(); ++k) {
      const auto& [p, descriptor] = points[k];
      if (static_cast<int>(descriptor->size()) != descriptor_dimension) continue;
      const Eigen::Map<const Eigen::VectorXf> a(descriptor->data(), descriptor_dimension);
      float best = min_similarity;
      int best_index = -1;
      for (int r = cell(p.y() - radius, rows); r <= cell(p.y() + radius, rows); ++r)
        for (int c = cell(p.x() - radius, cols); c <= cell(p.x() + radius, cols); ++c)
          for (int i : grid[static_cast<std::size_t>(r) * cols + c]) {
            const auto& o = frame.observations[i];
            if ((Vec2(o.x, o.y) - p).squaredNorm() > radius * radius) continue;
            const float similarity =
                a.dot(Eigen::Map<const Eigen::VectorXf>(descriptors.at(o.track_id), descriptor_dimension));
            if (similarity > best) { best = similarity; best_index = i; }
          }
      if (best_index < 0) continue;
      best_of_point.emplace_back(k, best_index);
      auto& slot = best_of_observation[best_index];
      if (best > slot.first) slot = {best, k};
    }
    for (const auto& [k, i] : best_of_point)
      if (best_of_observation[i].second == k) matches.emplace_back(k, i);
    return matches;
  }

  // The lost segment's landmarks projected with T, matched within `radius`.
  // Sorted by landmark id (deterministic RANSAC input).
  struct Match {
    std::uint64_t landmark, track;
    Vec2 u;
  };
  std::vector<Match> match_lost(const TrackedFrame& frame, const SE3& T, double radius,
                                const std::unordered_set<std::uint64_t>& used_landmarks,
                                const std::unordered_set<std::uint64_t>& used_tracks) const {
    // Only landmarks seen shortly before the loss: older dormant ones have
    // drifted and, in this wide window, steal mutual-best matches.
    const auto horizon = static_cast<std::uint64_t>(config.relocalization_max_frames);
    std::vector<std::uint64_t> ids;
    for (const auto& [id, landmark] : lost->landmarks)
      if (!used_landmarks.count(id) && landmark.last_frame + horizon >= lost->last_frame) ids.push_back(id);
    std::sort(ids.begin(), ids.end());
    std::vector<std::uint64_t> kept;
    std::vector<std::pair<Vec2, const std::vector<float>*>> points;
    for (auto id : ids) {
      const auto& landmark = lost->landmarks.at(id);
      const Vec3 xc = T * landmark.X;
      if (xc.z() <= 1e-6) continue;
      const Vec2 p = K.project(xc);
      if (p.x() < -radius || p.y() < -radius || p.x() > width + radius || p.y() > height + radius) continue;
      kept.push_back(id);
      points.emplace_back(p, &landmark.descriptor);
    }
    std::vector<Match> matches;
    for (const auto& [k, i] : match_projected(frame, points, radius,
                                              [&](std::uint64_t track) { return !used_tracks.count(track); })) {
      const auto& o = frame.observations[i];
      matches.push_back({kept[k], o.track_id, Vec2(o.x, o.y)});
    }
    return matches;
  }

  // Landmarks in view without an observation this frame: report them, and
  // let each take over a matching track. A track without a landmark is
  // attached; a track whose own landmark is a pose inlier this frame carries
  // a duplicate of the same point, and the two landmarks are merged (the one
  // with more keyframe observations is kept).
  void reassociate(const TrackedFrame& frame, const SE3& pose, const std::unordered_map<std::uint64_t, Vec2>& current,
                   const std::unordered_set<std::uint64_t>& inlier_tracks, OdometryFrameResult& result) {
    struct Candidate {
      std::uint64_t id, track;
      bool dormant;
      Vec2 p;
    };
    std::vector<Candidate> candidates;
    const auto consider = [&](const std::unordered_map<std::uint64_t, Landmark>& source, bool is_dormant) {
      for (const auto& [id, landmark] : source) {
        if (!is_dormant && current.count(landmark.track)) continue;  // observed this frame
        const Vec3 xc = pose * landmark.X;
        if (xc.z() <= 1e-6) continue;
        const Vec2 p = K.project(xc);
        if (p.x() < 0 || p.y() < 0 || p.x() > width - 1 || p.y() > height - 1) continue;
        candidates.push_back({id, landmark.track, is_dormant, p});
      }
    };
    consider(landmarks, false);
    consider(dormant, true);
    std::sort(candidates.begin(), candidates.end(), [](const auto& l, const auto& r) { return l.id < r.id; });
    std::vector<std::optional<std::uint64_t>> taken(candidates.size());
    if (config.reassociation) {
      std::vector<std::pair<Vec2, const std::vector<float>*>> points;
      for (const auto& c : candidates)
        points.emplace_back(c.p, c.dormant ? &dormant.at(c.id).descriptor : &landmarks.at(c.id).descriptor);
      const auto matches = match_projected(frame, points, config.reassociation_radius_px, [&](std::uint64_t track) {
        return !track_landmark.count(track) || (config.merge_landmarks && inlier_tracks.count(track));
      });
      for (const auto& [k, i] : matches) {
        const auto& c = candidates[k];
        const std::uint64_t track = frame.observations[i].track_id;
        if (c.dormant) landmarks.insert(dormant.extract(c.id));
        std::uint64_t kept = c.id;
        auto existing = track_landmark.find(track);
        if (existing != track_landmark.end()) {
          const std::uint64_t other = existing->second;
          const auto size = [&](std::uint64_t id) { return landmarks.at(id).observations.size(); };
          kept = size(c.id) >= size(other) ? c.id : other;
          merge(kept, kept == c.id ? other : c.id);
          ++result.merged;
        }
        attach(kept, track);
        auto& landmark = landmarks.at(kept);
        landmark.last_frame = frame.frame_index;
        update_descriptor(landmark, track);
        taken[k] = track;
      }
      result.reassociated = static_cast<int>(matches.size());
    }
    result.untracked_landmarks.reserve(candidates.size());
    for (std::size_t k = 0; k < candidates.size(); ++k) {
      float x = static_cast<float>(candidates[k].p.x()), y = static_cast<float>(candidates[k].p.y());
      if (config.distortion_k1 != 0) distort_point(x, y, width, height, config.distortion_k1);
      result.untracked_landmarks.push_back({candidates[k].id, candidates[k].track, x, y, candidates[k].dormant,
                                            taken[k].has_value(), taken[k].value_or(0)});
    }
  }

  // Try to resume the lost segment in this frame: predicted pose -> descriptor
  // matches -> P3P RANSAC -> refine -> guided search -> refine. On success the
  // frame is handled here (returns true); otherwise it falls through to
  // tracking/initialization of the current segment.
  bool relocalize(const TrackedFrame& frame, OdometryFrameResult& result) {
    if (descriptor_dimension == 0 || frame.frame_index <= lost->last_frame) return false;
    const double gap = static_cast<double>(frame.frame_index - lost->last_frame);
    const SE3 predicted = scale_motion(lost->velocity, gap) * lost->last_pose;
    auto matches = match_lost(frame, predicted, config.relocalization_radius_px, {}, {});
    result.relocalization_candidates = static_cast<int>(matches.size());
    if (static_cast<int>(matches.size()) < config.relocalization_min_inliers) return false;
    const auto observations_of = [&](const std::vector<Match>& list) {
      std::vector<PoseObservation> observations;
      for (const auto& m : list) observations.push_back({lost->landmarks.at(m.landmark).X, m.u, sigma_of(m.track)});
      return observations;
    };
    SE3 pose = predicted;
    std::vector<char> inliers;
    const std::uint32_t seed = static_cast<std::uint32_t>(frame.frame_index * 2246822519U);
    int count = ransac_pnp(K, observations_of(matches), config.relocalization_threshold_px,
                           config.relocalization_iterations, seed, pose, inliers);
    if (count < config.relocalization_min_inliers) return false;
    count = optimize_pose(K, observations_of(matches), pose, config.reprojection_threshold_px, inliers);
    if (count < config.relocalization_min_inliers) return false;
    // Guided search with the recovered pose for the landmarks not yet matched.
    std::vector<Match> accepted;
    for (std::size_t i = 0; i < matches.size(); ++i)
      if (inliers[i]) accepted.push_back(matches[i]);
    if (config.guided_search_radius_px > 0) {
      std::unordered_set<std::uint64_t> used_landmarks, used_tracks;
      for (const auto& m : accepted) {
        used_landmarks.insert(m.landmark);
        used_tracks.insert(m.track);
      }
      const auto more = match_lost(frame, pose, config.guided_search_radius_px, used_landmarks, used_tracks);
      accepted.insert(accepted.end(), more.begin(), more.end());
      count = optimize_pose(K, observations_of(accepted), pose, config.reprojection_threshold_px, inliers);
      if (count < config.relocalization_min_inliers) return false;
      std::vector<Match> kept;
      for (std::size_t i = 0; i < accepted.size(); ++i)
        if (inliers[i]) kept.push_back(accepted[i]);
      accepted = std::move(kept);
    }
    resume_lost(frame, pose, accepted, result);
    return true;
  }

  // Continue the lost segment from `pose`: its landmarks come back, matched
  // ones under the new track IDs, and a keyframe ties this frame in.
  void resume_lost(const TrackedFrame& frame, const SE3& pose, const std::vector<Match>& matches,
                   OdometryFrameResult& result) {
    const int young = segment;
    const bool young_had_map = !landmarks.empty();
    const std::uint64_t gap = frame.frame_index - lost->last_frame;
    clear_segment();  // whatever was (re)initialized meanwhile is retired as its own segment
    segment = lost->segment;
    segment_keyframes = lost->keyframes;
    landmarks = std::move(lost->landmarks);
    const SE3 velocity = lost->velocity;
    lost.reset();
    const auto current = observation_map(frame);
    // Landmarks whose own track is still live (it survived the gap) keep it.
    std::vector<std::uint64_t> ids;
    for (const auto& [id, landmark] : landmarks) ids.push_back(id);
    std::sort(ids.begin(), ids.end());
    for (auto id : ids)
      if (current.count(landmarks.at(id).track) && !track_landmark.count(landmarks.at(id).track))
        track_landmark[landmarks.at(id).track] = id;
    std::vector<std::uint64_t> matched;
    for (const auto& m : matches) {
      auto existing = track_landmark.find(m.track);
      if (existing != track_landmark.end() && existing->second != m.landmark) continue;  // track has another landmark
      attach(m.landmark, m.track);
      matched.push_back(m.track);
    }
    std::sort(matched.begin(), matched.end());
    const int id = add_keyframe(frame.frame_index, pose, current, sigmas);
    for (auto track_id : matched) {
      auto& landmark = landmarks.at(track_landmark.at(track_id));
      landmark.observations.push_back(sighting(id, track_id));
      landmark.last_frame = frame.frame_index;
      update_descriptor(landmark, track_id);
      add_color(landmark, track_id, colors);
    }
    local_bundle_adjustment();
    cull(current);
    update_depth_scale(id, result);
    last_pose = keyframes[id].pose;
    previous_pose = velocity.inverse() * last_pose;
    last_pose_frame = frame.frame_index;
    frames.push_back({frame.frame_index, frame.timestamp_ns, segment, id, SE3{}, true});
    frames_since_keyframe = 0;
    keyframe_landmarks = 0;
    for (const auto& [landmark_id, landmark] : landmarks)
      for (const auto& o : landmark.observations) keyframe_landmarks += o.keyframe == id;
    state = OdometryState::tracking;
    result.state = state;
    result.has_pose = true;
    result.keyframe = true;
    result.relocalized = true;
    result.pose = to_pose(last_pose);
    result.correspondences = static_cast<int>(matches.size());
    result.inliers = static_cast<int>(matched.size());
    result.pose_inliers = matched;
    std::vector<double> errors;
    for (auto track_id : matched)
      errors.push_back((K.project(last_pose * landmarks.at(track_landmark.at(track_id)).X) - current.at(track_id)).norm());
    result.median_reprojection_px = median(errors);
    result.event = "relocalized into segment " + std::to_string(segment) + ": " + std::to_string(matched.size()) +
                   " landmarks re-found (" + std::to_string(gap) + " frames after its last pose)" + (young_had_map ? "; segment " + std::to_string(young) + " retired" : "");
  }

  // Per-segment metric scale from the network depth of reference landmarks
  // observed at keyframe `id` (just refined by BA). See the config comment.
  void update_depth_scale(int id, OdometryFrameResult& result) {
    if (depths.empty()) return;
    const SE3& T = keyframes[id].pose;
    std::vector<double> r;
    for (const auto& [landmark_id, landmark] : landmarks) {
      if (static_cast<int>(landmark.observations.size()) < config.depth_scale_min_keyframes ||
          landmark.max_parallax < config.depth_scale_min_parallax_degrees * kDegree ||
          landmark.observations.back().keyframe != id)
        continue;
      auto it = depths.find(landmark.track);
      if (it == depths.end() || it->second.second > config.depth_scale_max_relative_sigma * it->second.first) continue;
      const double z = (T * landmark.X).z();
      if (z > 1e-9) r.push_back(std::log(it->second.first) - std::log(z));
    }
    result.keyframe_scale_samples = static_cast<int>(r.size());
    if (static_cast<int>(r.size()) < config.depth_scale_min_samples) return;
    const double centre = median(r);
    std::vector<double> deviation;
    for (double v : r) deviation.push_back(std::abs(v - centre));
    const double spread = 1.4826 * median(deviation);
    result.keyframe_log_scale = centre;
    result.keyframe_scale_spread = spread;
    const double measurement = config.depth_scale_floor_sigma * config.depth_scale_floor_sigma +
                               spread * spread / static_cast<double>(r.size());
    auto& scale = depth_scales[segment];
    if (scale.keyframes == 0) {
      scale.log_scale = centre;
      scale.variance = measurement;
    } else {
      const double predicted = scale.variance + config.depth_scale_drift_sigma * config.depth_scale_drift_sigma;
      const double gain = predicted / (predicted + measurement);
      scale.log_scale += gain * (centre - scale.log_scale);
      scale.variance = (1 - gain) * predicted;
    }
    ++scale.keyframes;
  }

  void lose(const TrackedFrame& frame, OdometryFrameResult& result, const char* why) {
    state = OdometryState::initializing;
    result.state = OdometryState::lost;
    result.event = why;
    int described = 0;
    for (const auto& [id, landmark] : landmarks) described += !landmark.descriptor.empty();
    if (config.relocalization && static_cast<int>(segment_keyframes.size()) >= config.relocalization_min_keyframes &&
        described >= config.relocalization_min_inliers) {
      // Keep this segment's map for relocalization (it replaces an older one).
      expire_lost();
      lost = LostSegment{segment, segment_keyframes, {}, last_pose, last_pose * previous_pose.inverse(),
                         last_pose_frame};
      for (auto& [id, landmark] : landmarks)
        if (landmark.observations.size() >= 2) lost->landmarks.emplace(id, std::move(landmark));
      for (auto& [id, landmark] : dormant) lost->landmarks.emplace(id, std::move(landmark));
      landmarks.clear();
      dormant.clear();
    }
    clear_segment();
    segment = next_segment++;
    reference = frame;  // reinitialize from here
  }

  int add_keyframe(std::uint64_t frame_index, const SE3& pose, std::unordered_map<std::uint64_t, Vec2> observations,
                   std::unordered_map<std::uint64_t, double> observation_sigmas) {
    const int id = static_cast<int>(keyframes.size());
    keyframes.push_back({id, frame_index, segment, pose, std::move(observations), std::move(observation_sigmas)});
    segment_keyframes.push_back(id);
    return id;
  }

  std::vector<int> window() const {
    const int count = std::min<int>(config.window_keyframes, static_cast<int>(segment_keyframes.size()));
    return {segment_keyframes.end() - count, segment_keyframes.end()};
  }

  void triangulate_new(int id) {
    const auto& kf = keyframes[id];
    const auto active = window();
    const double threshold = config.reprojection_threshold_px;
    std::vector<std::uint64_t> fresh;
    for (const auto& [track_id, u] : kf.observations)
      if (!track_landmark.count(track_id)) fresh.push_back(track_id);
    std::sort(fresh.begin(), fresh.end());
    for (auto track_id : fresh) {
      const Vec2 u = kf.observations.at(track_id);
      // Widest baseline: the oldest window keyframe that saw this track.
      for (int other : active) {
        if (other == id) break;
        auto it = keyframes[other].observations.find(track_id);
        if (it == keyframes[other].observations.end()) continue;
        Vec3 X;
        const SE3& T0 = keyframes[other].pose;
        if (!triangulate(T0, normalized(it->second), kf.pose, normalized(u), X)) break;
        if (!reprojects(T0, X, it->second, threshold) || !reprojects(kf.pose, X, u, threshold) ||
            parallax(T0, kf.pose, X) < config.min_triangulation_parallax_degrees * kDegree) break;
        Landmark landmark = new_landmark(X, keyframes[other].frame_index, kf.frame_index, track_id);
        add_color(landmark, track_id, colors);
        stamp(landmark, track_id);
        for (int k : active) {
          auto seen = keyframes[k].observations.find(track_id);
          if (seen != keyframes[k].observations.end() && reprojects(keyframes[k].pose, X, seen->second, threshold))
            landmark.observations.push_back(sighting(k, track_id));
        }
        if (landmark.observations.size() >= 2) add_active(std::move(landmark));
        break;
      }
    }
  }

  void local_bundle_adjustment() {
    const auto active = window();
    if (active.size() < 2 || config.bundle_iterations == 0) return;
    BundleProblem problem;
    std::unordered_map<int, int> camera_of;
    for (std::size_t c = 0; c < active.size(); ++c) {
      camera_of[active[c]] = static_cast<int>(c);
      problem.cameras.push_back(keyframes[active[c]].pose);
      // Gauge: the oldest keyframe fixes the frame; a second fixed keyframe
      // (when available) holds the monocular scale.
      problem.fixed.push_back(c == 0 || (c == 1 && active.size() >= 3));
    }
    std::vector<std::uint64_t> point_ids;
    for (const auto& [id, landmark] : landmarks) {
      int seen = 0;
      for (const auto& o : landmark.observations) seen += camera_of.count(o.keyframe) > 0;
      if (seen >= 2) point_ids.push_back(id);
    }
    std::sort(point_ids.begin(), point_ids.end());
    for (std::size_t p = 0; p < point_ids.size(); ++p) {
      const auto& landmark = landmarks.at(point_ids[p]);
      problem.points.push_back(landmark.X);
      for (const auto& o : landmark.observations) {
        auto it = camera_of.find(o.keyframe);
        if (it != camera_of.end()) problem.observations.push_back({it->second, static_cast<int>(p), o.u, o.sigma});
      }
    }
    bundle_adjust(K, problem, config.bundle_iterations, config.reprojection_threshold_px);
    for (std::size_t c = 0; c < active.size(); ++c) keyframes[active[c]].pose = problem.cameras[c];
    for (std::size_t p = 0; p < point_ids.size(); ++p) landmarks.at(point_ids[p]).X = problem.points[p];
  }

  // Drop bad observations/points and points that can never be observed again.
  void cull(const std::unordered_map<std::uint64_t, Vec2>& current) {
    update_typical_rms();
    const auto active = window();
    const std::unordered_set<int> in_window(active.begin(), active.end());
    for (auto it = landmarks.begin(); it != landmarks.end();) {
      auto& obs = it->second.observations;
      obs.erase(std::remove_if(obs.begin(), obs.end(), [&](const auto& o) {
        const bool windowed = in_window.count(o.keyframe) > 0;
        if (!windowed && !config.cull_stale_observations) return false;
        return !reprojects(keyframes[o.keyframe].pose, it->second.X, o.u, 2 * config.reprojection_threshold_px);
      }), obs.end());
      for (auto& o : obs) {
        if (!in_window.count(o.keyframe)) continue;
        const Vec3 xc = keyframes[o.keyframe].pose * it->second.X;
        o.error = xc.z() > 0 ? (K.project(xc) - o.u).norm() / o.sigma : std::numeric_limits<double>::quiet_NaN();
      }
      update_quality(it->second);
      bool windowed = false;
      for (const auto& o : obs) windowed |= in_window.count(o.keyframe) > 0;
      const bool dead = !current.count(it->second.track) && !windowed;
      const bool enough = obs.size() >= 2;
      if (enough && dead) {
        unindex(it->first, it->second);
        if ((config.relocalization || config.reassociation) && !it->second.descriptor.empty())
          dormant.emplace(it->first, std::move(it->second));
        else retire(it->first, it->second);  // final: keep it in the map
      }
      if (!enough || dead) it = erase_active(it);
      else ++it;
    }
  }
};

VisualOdometry::VisualOdometry(VisualOdometryConfig config) : impl_(std::make_unique<Impl>()) {
  config.validate();
  impl_->config = std::move(config);
}
VisualOdometry::~VisualOdometry() = default;

OdometryFrameResult VisualOdometry::process(const TrackedFrame& frame) { return impl_->process(frame); }

void VisualOdometry::reset() {
  auto config = impl_->config;
  impl_ = std::make_unique<Impl>();
  impl_->config = std::move(config);
}

std::vector<TrajectorySample> VisualOdometry::trajectory() const { return trajectory(0); }

std::vector<TrajectorySample> VisualOdometry::trajectory(std::size_t first) const {
  std::vector<TrajectorySample> samples;
  const auto& frames = impl_->frames;
  if (first >= frames.size()) return samples;
  samples.reserve(frames.size() - first);
  for (std::size_t i = first; i < frames.size(); ++i) {
    const auto& record = frames[i];
    const SE3 pose = record.relative * impl_->keyframes[record.keyframe].pose;
    samples.push_back({record.frame_index, record.timestamp_ns, record.segment, record.is_keyframe,
                       to_pose(pose)});
  }
  return samples;
}

std::size_t VisualOdometry::stable_prefix() const {
  const auto& frames = impl_->frames;
  if (impl_->state != OdometryState::tracking) return frames.size();
  const auto window = impl_->window();
  if (window.empty()) return frames.size();
  // Reference keyframe ids are non-decreasing along the frame records.
  std::size_t prefix = frames.size();
  while (prefix > 0 && frames[prefix - 1].keyframe >= window.front()) --prefix;
  return prefix;
}

std::size_t VisualOdometry::trajectory_size() const { return impl_->frames.size(); }

std::vector<MapPoint> VisualOdometry::active_map() const {
  std::vector<MapPoint> points;
  points.reserve(impl_->landmarks.size() + impl_->dormant.size());
  for (const auto& [id, landmark] : impl_->landmarks) points.push_back(impl_->map_point(id, landmark));
  for (const auto& [id, landmark] : impl_->dormant) {
    points.push_back(impl_->map_point(id, landmark));
    points.back().dormant = true;
  }
  if (impl_->lost)  // held for relocalization: neither refined nor retired yet
    for (const auto& [id, landmark] : impl_->lost->landmarks) {
      points.push_back(impl_->map_point(id, landmark));
      points.back().dormant = true;
    }
  return points;
}
std::size_t VisualOdometry::retired_count() const { return impl_->retired.size(); }
std::vector<MapPoint> VisualOdometry::retired_map(std::size_t first) const {
  const auto& retired = impl_->retired;
  if (first >= retired.size()) return {};
  return {retired.begin() + static_cast<std::ptrdiff_t>(first), retired.end()};
}

double VisualOdometry::segment_scale(int segment) const {
  auto it = impl_->depth_scales.find(segment);
  return it == impl_->depth_scales.end() ? 0.0 : std::exp(it->second.log_scale);
}
bool VisualOdometry::has_landmark(std::uint64_t track_id) const { return impl_->track_landmark.count(track_id) > 0; }
std::optional<std::uint64_t> VisualOdometry::landmark_id(std::uint64_t track_id) const {
  auto it = impl_->track_landmark.find(track_id);
  if (it == impl_->track_landmark.end()) return std::nullopt;
  return it->second;
}
std::optional<std::array<double, 3>> VisualOdometry::landmark(std::uint64_t track_id) const {
  auto it = impl_->track_landmark.find(track_id);
  if (it == impl_->track_landmark.end()) return std::nullopt;
  const auto& X = impl_->landmarks.at(it->second).X;
  return std::array<double, 3>{X.x(), X.y(), X.z()};
}
const OdometryFrameResult& VisualOdometry::last() const { return impl_->last; }
const VisualOdometryConfig& VisualOdometry::config() const { return impl_->config; }
}  // namespace slam_native
