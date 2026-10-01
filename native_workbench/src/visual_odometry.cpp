#include "slam_native/visual_odometry.hpp"

#include "vo_geometry.hpp"

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
      min_tracked_points >= 6 && reprojection_threshold_px > 0 && recovery_threshold_px > 0 &&
      keyframe_min_interval >= 1 && keyframe_max_interval >= keyframe_min_interval && keyframe_track_ratio > 0 &&
      keyframe_emergency_ratio >= 0 && min_triangulation_parallax_degrees >= 0 && window_keyframes >= 2 &&
      bundle_iterations >= 0 && local_map_keyframes >= window_keyframes && reassociation_radius_px >= 0 &&
      min_descriptor_similarity <= 1 && (!intrinsics || (intrinsics->fx > 0 && intrinsics->fy > 0));
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
    std::unordered_map<std::uint64_t, Vec2> observations;  // by track
  };
  struct Sighting {
    int keyframe;
    Vec2 u;
  };
  struct Landmark {
    Vec3 X;
    std::vector<Sighting> observations;
    std::array<double, 3> color_sum{};
    int color_count{};
    std::uint64_t first_frame{}, last_frame{};
    int segment{};
    std::uint64_t track{};          // latest track observing it
    std::vector<float> descriptor;  // that track's latest; empty without descriptors
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
  std::unordered_map<std::uint64_t, Landmark> landmarks;  // the local map, by landmark id
  // Track -> the landmark it observes (one track per landmark). Kept while
  // the landmark lives: a track missing from a frame may resume.
  std::unordered_map<std::uint64_t, std::uint64_t> track_landmark;
  std::uint64_t next_landmark_id{};
  std::vector<MapPoint> retired;
  std::unordered_map<std::uint64_t, std::array<std::uint8_t, 3>> colors;  // this frame's, by track
  std::unordered_map<std::uint64_t, const float*> descriptors;            // this frame's, by track
  int descriptor_dimension{};
  std::optional<TrackedFrame> reference;
  int reference_age{};
  std::vector<FrameRecord> frames;
  SE3 last_pose, previous_pose;
  int frames_since_keyframe{};
  int keyframe_landmarks{};
  OdometryFrameResult last;

  static std::unordered_map<std::uint64_t, Vec2> observation_map(const TrackedFrame& frame) {
    std::unordered_map<std::uint64_t, Vec2> map;
    map.reserve(frame.observations.size());
    for (const auto& o : frame.observations) map.emplace(o.track_id, Vec2(o.x, o.y));
    return map;
  }
  static std::unordered_map<std::uint64_t, std::array<std::uint8_t, 3>> color_map(const TrackedFrame& frame) {
    std::unordered_map<std::uint64_t, std::array<std::uint8_t, 3>> map;
    for (const auto& o : frame.observations)
      if (o.has_color) map.emplace(o.track_id, o.color);
    return map;
  }
  Landmark new_landmark(const Vec3& X, std::uint64_t first_frame, std::uint64_t last_frame,
                        std::uint64_t track) const {
    Landmark landmark;
    landmark.X = X;
    landmark.first_frame = first_frame;
    landmark.last_frame = last_frame;
    landmark.segment = segment;
    landmark.track = track;
    update_descriptor(landmark);
    return landmark;
  }
  // The landmark's track's descriptor in this frame, when it has one.
  void update_descriptor(Landmark& landmark) const {
    if (auto it = descriptors.find(landmark.track); it != descriptors.end())
      landmark.descriptor.assign(it->second, it->second + descriptor_dimension);
  }
  void add_active(Landmark landmark) {
    const std::uint64_t id = next_landmark_id++;
    track_landmark[landmark.track] = id;
    landmarks.emplace(id, std::move(landmark));
  }
  std::unordered_map<std::uint64_t, Landmark>::iterator erase_active(
      std::unordered_map<std::uint64_t, Landmark>::iterator it) {
    if (auto t = track_landmark.find(it->second.track); t != track_landmark.end() && t->second == it->first)
      track_landmark.erase(t);
    return landmarks.erase(it);
  }
  Sighting sighting(int kf, std::uint64_t track_id) const { return {kf, keyframes[kf].observations.at(track_id)}; }
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
    return point;
  }

  // `keep`: retire the segment's landmarks (they stay in the map) rather than
  // discarding them (a rejected initialization).
  void clear_segment(bool keep = true) {
    if (keep) {
      std::vector<std::uint64_t> ids;
      for (const auto& [id, landmark] : landmarks)
        if (landmark.observations.size() >= 2) ids.push_back(id);
      std::sort(ids.begin(), ids.end());
      for (auto id : ids) retired.push_back(map_point(id, landmarks.at(id)));
    }
    landmarks.clear();
    track_landmark.clear();
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
    if (state == OdometryState::tracking) track(frame, result);
    else initialize(frame, result);
    result.segment = segment;
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
    const int k0 = add_keyframe(reference->frame_index, first, observation_map(*reference));
    const int k1 = add_keyframe(frame.frame_index, second, current);
    const auto reference_colors = color_map(*reference);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (!good[i]) continue;
      Landmark landmark = new_landmark(X[i] * scale, reference->frame_index, frame.frame_index, ids[i]);
      landmark.observations = {sighting(k0, ids[i]), sighting(k1, ids[i])};
      add_color(landmark, ids[i], reference_colors);
      add_color(landmark, ids[i], colors);
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
    // Sorted by track (deterministic order; unordered_map iteration is not).
    std::vector<std::uint64_t> ids;
    for (const auto& [id, u] : current)
      if (track_landmark.count(id)) ids.push_back(id);
    std::sort(ids.begin(), ids.end());
    std::vector<PoseObservation> observations;
    for (auto id : ids) observations.push_back({landmarks.at(track_landmark.at(id)).X, current.at(id)});
    result.correspondences = static_cast<int>(observations.size());
    if (result.correspondences < config.min_tracked_points) return lose(frame, result, "too few tracked landmarks");
    const SE3 velocity = last_pose * previous_pose.inverse();
    SE3 pose = velocity * last_pose;
    std::vector<char> inliers;
    int count = optimize_pose(K, observations, pose, config.reprojection_threshold_px, inliers);
    if (count < std::max(config.min_tracked_points, result.correspondences / 2)) {
      SE3 fallback = last_pose;  // constant position instead of constant velocity
      std::vector<char> fallback_inliers;
      const int fallback_count = optimize_pose(K, observations, fallback, config.reprojection_threshold_px,
                                               fallback_inliers);
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
      if (ransac_pnp(K, observations, config.recovery_threshold_px, config.ransac_iterations, seed, recovered,
                     recovered_inliers) > count) {
        const int recovered_count = optimize_pose(K, observations, recovered, config.reprojection_threshold_px,
                                                  recovered_inliers);
        if (recovered_count > count) {
          pose = recovered;
          inliers = std::move(recovered_inliers);
          count = recovered_count;
          result.event = "pose re-estimated (P3P RANSAC)";
        }
      }
    }
    if (count < config.min_tracked_points) return lose(frame, result, "pose optimization lost the map");
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
      update_descriptor(landmark);
      errors.push_back((K.project(pose * observations[i].X) - observations[i].u).norm());
    }
    result.inliers = count;
    result.median_reprojection_px = median(errors);
    result.pose_inliers = inlier_ids;
    result.pose_outliers = outlier_ids;
    previous_pose = last_pose;
    last_pose = pose;
    // Before the keyframe step, so a re-found landmark is not triangulated again.
    reassociate(frame, pose, current, std::unordered_set<std::uint64_t>(inlier_ids.begin(), inlier_ids.end()), result);
    ++frames_since_keyframe;
    const bool emergency = count < config.keyframe_emergency_ratio * keyframe_landmarks;
    const bool keyframe = (frames_since_keyframe >= config.keyframe_min_interval &&
                           (count < config.keyframe_track_ratio * keyframe_landmarks ||
                            frames_since_keyframe >= config.keyframe_max_interval)) ||
                          emergency;
    if (keyframe) {
      const int id = add_keyframe(frame.frame_index, pose, current);
      for (auto track_id : inlier_ids) {
        auto& landmark = landmarks.at(track_landmark.at(track_id));
        landmark.observations.push_back(sighting(id, track_id));
        add_color(landmark, track_id, colors);
      }
      // A live track whose landmark no longer fits usually has a poorly
      // conditioned depth from a short baseline. Drop the point and let it be
      // re-triangulated from the widest keyframe pair that saw the track.
      for (auto track_id : outlier_ids) erase_active(landmarks.find(track_landmark.at(track_id)));
      triangulate_new(id);
      local_bundle_adjustment();
      cull(current);
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

  // Local landmarks that no track observes in this frame are projected and
  // matched by descriptor to the frame's tracks that either have no landmark
  // or whose landmark is a pose inlier (see local_map_keyframes).
  void reassociate(const TrackedFrame& frame, const SE3& pose, const std::unordered_map<std::uint64_t, Vec2>& current,
                   const std::unordered_set<std::uint64_t>& inlier_tracks, OdometryFrameResult& result) {
    const double radius = config.reassociation_radius_px;
    if (descriptor_dimension == 0 || radius <= 0) return;
    struct Candidate {
      std::uint64_t id;
      Vec2 p;
    };
    std::vector<Candidate> candidates;
    for (const auto& [id, landmark] : landmarks) {
      if (current.count(landmark.track) || static_cast<int>(landmark.descriptor.size()) != descriptor_dimension) continue;
      const Vec3 xc = pose * landmark.X;
      if (xc.z() <= 1e-6) continue;
      const Vec2 p = K.project(xc);
      if (p.x() < 0 || p.y() < 0 || p.x() > width - 1 || p.y() > height - 1) continue;
      candidates.push_back({id, p});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& l, const auto& r) { return l.id < r.id; });
    // Eligible observations on a grid of radius-sized cells.
    const int cols = static_cast<int>(width / radius) + 1, rows = static_cast<int>(height / radius) + 1;
    const auto cell = [&](double value, int count) { return std::clamp(static_cast<int>(value / radius), 0, count - 1); };
    std::vector<std::vector<int>> grid(static_cast<std::size_t>(cols) * rows);
    for (int i = 0; i < static_cast<int>(frame.observations.size()); ++i) {
      const auto track = frame.observations[i].track_id;
      if (!descriptors.count(track) || (track_landmark.count(track) && !inlier_tracks.count(track))) continue;
      grid[static_cast<std::size_t>(cell(frame.observations[i].y, rows)) * cols + cell(frame.observations[i].x, cols)]
          .push_back(i);
    }
    // Mutual best matches above the similarity threshold.
    std::vector<std::pair<std::size_t, int>> best_of_candidate;
    std::unordered_map<int, std::pair<float, std::size_t>> best_of_observation;
    for (std::size_t k = 0; k < candidates.size(); ++k) {
      const Vec2& p = candidates[k].p;
      const Eigen::Map<const Eigen::VectorXf> a(landmarks.at(candidates[k].id).descriptor.data(), descriptor_dimension);
      float best = static_cast<float>(config.min_descriptor_similarity);
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
      best_of_candidate.emplace_back(k, best_index);
      auto& slot = best_of_observation[best_index];
      if (best > slot.first) slot = {best, k};
    }
    for (const auto& [k, i] : best_of_candidate)
      if (best_of_observation[i].second == k) associate(candidates[k].id, frame.observations[i].track_id, frame, result);
  }

  // `track` observes landmark `id`. A track that already has a landmark
  // carries a duplicate of the same point: the one with fewer keyframe
  // sightings is folded into the other (keyframes both saw keep the
  // survivor's sighting). BA and cull() then refine the survivor.
  void associate(std::uint64_t id, std::uint64_t track, const TrackedFrame& frame, OdometryFrameResult& result) {
    std::uint64_t kept = id;
    if (auto it = track_landmark.find(track); it != track_landmark.end()) {
      std::uint64_t other = it->second;
      if (landmarks.at(other).observations.size() > landmarks.at(id).observations.size()) std::swap(kept, other);
      auto node = landmarks.extract(other);
      Landmark& absorbed = node.mapped();
      if (auto t = track_landmark.find(absorbed.track); t != track_landmark.end() && t->second == other)
        track_landmark.erase(t);
      Landmark& survivor = landmarks.at(kept);
      for (const auto& o : absorbed.observations)
        if (std::none_of(survivor.observations.begin(), survivor.observations.end(),
                         [&](const Sighting& k) { return k.keyframe == o.keyframe; }))
          survivor.observations.push_back(o);
      std::sort(survivor.observations.begin(), survivor.observations.end(),
                [](const Sighting& l, const Sighting& r) { return l.keyframe < r.keyframe; });
      for (int c = 0; c < 3; ++c) survivor.color_sum[c] += absorbed.color_sum[c];
      survivor.color_count += absorbed.color_count;
      survivor.first_frame = std::min(survivor.first_frame, absorbed.first_frame);
      ++result.merged;
    }
    Landmark& landmark = landmarks.at(kept);
    if (auto it = track_landmark.find(landmark.track); it != track_landmark.end() && it->second == kept)
      track_landmark.erase(it);
    landmark.track = track;
    landmark.last_frame = frame.frame_index;
    update_descriptor(landmark);
    track_landmark[track] = kept;
    ++result.reassociated;
  }

  // The segment ends: its landmarks are retired and a new segment initializes
  // from this frame.
  void lose(const TrackedFrame& frame, OdometryFrameResult& result, const char* why) {
    state = OdometryState::initializing;
    result.state = OdometryState::lost;
    result.event = why;
    clear_segment();
    segment = next_segment++;
    reference = frame;
  }

  int add_keyframe(std::uint64_t frame_index, const SE3& pose, std::unordered_map<std::uint64_t, Vec2> observations) {
    const int id = static_cast<int>(keyframes.size());
    keyframes.push_back({id, frame_index, segment, pose, std::move(observations)});
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
        if (it != camera_of.end()) problem.observations.push_back({it->second, static_cast<int>(p), o.u});
      }
    }
    bundle_adjust(K, problem, config.bundle_iterations, config.reprojection_threshold_px);
    for (std::size_t c = 0; c < active.size(); ++c) keyframes[active[c]].pose = problem.cameras[c];
    for (std::size_t p = 0; p < point_ids.size(); ++p) landmarks.at(point_ids[p]).X = problem.points[p];
  }

  // Drop observations that no longer reproject and points left with fewer
  // than two; retire points that left the local map (no live track, no
  // sighting in the last local_map_keyframes keyframes).
  void cull(const std::unordered_map<std::uint64_t, Vec2>& current) {
    const std::size_t local = std::min<std::size_t>(config.local_map_keyframes, segment_keyframes.size());
    const int oldest_local = segment_keyframes[segment_keyframes.size() - local];  // ids increase in a segment
    std::vector<std::uint64_t> ended;
    for (auto it = landmarks.begin(); it != landmarks.end();) {
      auto& obs = it->second.observations;
      obs.erase(std::remove_if(obs.begin(), obs.end(), [&](const Sighting& o) {
        return !reprojects(keyframes[o.keyframe].pose, it->second.X, o.u, 2 * config.reprojection_threshold_px);
      }), obs.end());
      const bool seen = !obs.empty() && obs.back().keyframe >= oldest_local;  // sightings sorted by keyframe
      const bool ended_here = !current.count(it->second.track) && !seen;
      const bool enough = obs.size() >= 2;
      if (enough && ended_here) ended.push_back(it->first);
      if (!enough) it = erase_active(it);
      else ++it;
    }
    std::sort(ended.begin(), ended.end());
    for (auto id : ended) {
      auto it = landmarks.find(id);
      retired.push_back(map_point(id, it->second));
      erase_active(it);
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
  points.reserve(impl_->landmarks.size());
  for (const auto& [id, landmark] : impl_->landmarks) points.push_back(impl_->map_point(id, landmark));
  return points;
}
std::size_t VisualOdometry::retired_count() const { return impl_->retired.size(); }
std::vector<MapPoint> VisualOdometry::retired_map(std::size_t first) const {
  const auto& retired = impl_->retired;
  if (first >= retired.size()) return {};
  return {retired.begin() + static_cast<std::ptrdiff_t>(first), retired.end()};
}
bool VisualOdometry::has_landmark(std::uint64_t track_id) const { return impl_->track_landmark.count(track_id) > 0; }
const OdometryFrameResult& VisualOdometry::last() const { return impl_->last; }
const VisualOdometryConfig& VisualOdometry::config() const { return impl_->config; }
}  // namespace slam_native
