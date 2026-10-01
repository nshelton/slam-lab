#include "slam_native/visual_odometry.hpp"

#include "place_index.hpp"
#include "pose_graph.hpp"
#include "vo_geometry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <limits>
#include <map>
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
      min_descriptor_similarity <= 1 && coast_max_frames >= 0 && relocalization_min_radius_px > 0 &&
      relocalization_radius_px >= relocalization_min_radius_px && relocalization_min_inliers >= 6 &&
      place_ratio > 0 && place_ransac_iterations > 0 && place_min_inliers >= 6 && loop_scale_spread >= 1 &&
      loop_min_pairs >= 3 && loop_min_keyframe_gap >= 1 && loop_covisibility_min >= 1 && loop_cooldown_keyframes >= 0 &&
      loop_edge_weight > 0 && (!intrinsics || (intrinsics->fx > 0 && intrinsics->fy > 0));
  if (!ok) throw std::invalid_argument("Invalid visual odometry settings");
}

const char* to_string(OdometryState state) {
  switch (state) {
    case OdometryState::initializing: return "initializing";
    case OdometryState::tracking: return "tracking";
    case OdometryState::coasting: return "coasting";
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
    // No odometry ties it to the keyframe before it: the first keyframe of a
    // segment, or one that resume() placed from the map (see `links`).
    bool starts_stretch{};
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
    bool predicted{};
    double confidence{};
  };

  VisualOdometryConfig config;
  bool calibrated{};
  Intrinsics K;
  int width{}, height{};
  OdometryState state{OdometryState::initializing};
  int segment{};
  int next_segment{1};                   // segment ids are never reused
  std::vector<KeyFrame> keyframes;       // all segments; id == index
  std::vector<int> segment_keyframes;    // ids in the current segment, in time order
  // The keyframes the local map and the bundle-adjustment window are taken
  // from (their tail): the segment's own, or after resume() the old keyframes
  // that saw the landmarks found, followed by the new ones.
  std::vector<int> local_keyframes;
  // Every landmark of the run, by id: all segments, never retired.
  std::unordered_map<std::uint64_t, Landmark> landmarks;
  // The local map: the current segment's landmarks that a live track observes
  // or that one of the last local_map_keyframes local keyframes saw. Only
  // these are projected into frames, re-associated at a pose and
  // bundle-adjusted; the others are found again by descriptor (`places`).
  std::unordered_set<std::uint64_t> local;
  PlaceIndex places;  // every landmark's latest descriptor
  int frames_since_place_search{};
  // Keyframe pairs tied by a place match rather than by odometry: a resumed
  // keyframe with the old keyframe that saw most of its landmarks, the two
  // ends of a closed loop, the two segments of a merge. Edges of later pose graphs.
  std::vector<std::pair<int, int>> links;
  int keyframes_since_correction{1 << 30};
  std::string note;            // this frame's correction, for the event text
  std::size_t changed_from{std::numeric_limits<std::size_t>::max()};  // first frame record a correction moved this frame
  // Track -> the landmark it observes (one track per landmark). Kept while
  // the landmark is local: a track missing from a frame may resume.
  std::unordered_map<std::uint64_t, std::uint64_t> track_landmark;
  std::uint64_t next_landmark_id{};
  std::unordered_map<std::uint64_t, std::array<std::uint8_t, 3>> colors;  // this frame's, by track
  std::unordered_map<std::uint64_t, const float*> descriptors;            // this frame's, by track
  int descriptor_dimension{};
  std::optional<TrackedFrame> reference;
  int reference_age{};
  std::vector<FrameRecord> frames;
  SE3 last_pose, previous_pose;
  MotionModel motion;
  int coasted{};    // frames since the last tracked pose
  SE3 coast_origin; // that pose
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
    if (auto it = descriptors.find(track); it != descriptors.end())
      landmark.descriptor.assign(it->second, it->second + descriptor_dimension);
    return landmark;
  }
  // The landmark's track's descriptor in this frame, when it has one.
  void update_descriptor(std::uint64_t id, Landmark& landmark) {
    auto it = descriptors.find(landmark.track);
    if (it == descriptors.end()) return;
    landmark.descriptor.assign(it->second, it->second + descriptor_dimension);
    index_set(id, it->second);
  }
  // A new landmark, in the local map.
  void add_landmark(Landmark landmark) {
    const std::uint64_t id = next_landmark_id++;
    track_landmark[landmark.track] = id;
    if (!landmark.descriptor.empty()) index_set(id, landmark.descriptor.data());
    local.insert(id);
    landmarks.emplace(id, std::move(landmark));
  }
  // Remove a landmark from the map altogether (a point that did not hold up).
  void erase_landmark(std::uint64_t id) {
    auto it = landmarks.find(id);
    if (auto t = track_landmark.find(it->second.track); t != track_landmark.end() && t->second == id)
      track_landmark.erase(t);
    local.erase(id);
    index_erase(id);
    landmarks.erase(it);
  }
  std::vector<std::uint64_t> sorted_local() const {  // deterministic order (the set's is not)
    std::vector<std::uint64_t> ids(local.begin(), local.end());
    std::sort(ids.begin(), ids.end());
    return ids;
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
    point.local = local.count(id) > 0;
    return point;
  }

  // The segment ends. Its landmarks stay in the map, outside the local map,
  // where resume() can find them again; with !keep they are discarded (a
  // rejected initialization).
  void end_segment(bool keep = true) {
    for (auto id : sorted_local())
      if (!keep || landmarks.at(id).observations.size() < 2) erase_landmark(id);
    local.clear();
    track_landmark.clear();
    segment_keyframes.clear();
    local_keyframes.clear();
    reference.reset();
    reference_age = 0;
    coasted = 0;
  }

  // RMS per-axis pixel uncertainty of the projections of `points` (world)
  // under a pose with error covariance C, and the confidence it implies.
  double image_sigma(const SE3& pose, const Mat6& C, const std::vector<Vec3>& points) const {
    double sum = 0;
    int count = 0;
    for (const auto& X : points) {
      const Vec3 xc = pose * X;
      if (xc.z() <= 1e-6) continue;
      const auto J = pose_jacobian(K, xc);
      sum += (J * C * J.transpose()).trace();
      ++count;
    }
    return count ? std::sqrt(sum / (2 * count)) : std::numeric_limits<double>::infinity();
  }
  double confidence_of(double sigma) const {
    const double t = config.reprojection_threshold_px;
    return sigma > 0 ? 1 - std::exp(-t * t / (2 * sigma * sigma)) : 1;
  }
  void report_uncertainty(double sigma, OdometryFrameResult& result) const {
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(result.pose_covariance.data()) = motion.pose;
    result.pose_sigma_px = sigma;
    result.confidence = confidence_of(sigma);
  }
  // Local landmarks, in id order, that are in front of the camera and in the image.
  std::vector<Vec3> visible_landmarks(const SE3& pose) const {
    std::vector<std::pair<std::uint64_t, Vec3>> found;
    for (auto id : local) {
      const Landmark& landmark = landmarks.at(id);
      const Vec3 xc = pose * landmark.X;
      if (xc.z() <= 1e-6) continue;
      const Vec2 p = K.project(xc);
      if (p.x() >= 0 && p.y() >= 0 && p.x() <= width - 1 && p.y() <= height - 1) found.emplace_back(id, landmark.X);
    }
    std::sort(found.begin(), found.end(), [](const auto& l, const auto& r) { return l.first < r.first; });
    std::vector<Vec3> points;
    for (const auto& [id, X] : found) points.push_back(X);
    return points;
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
        end_segment();
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
    note.clear();
    changed_from = std::numeric_limits<std::size_t>::max();
    poll(frame, result);
    if (state == OdometryState::initializing) initialize(frame, result);
    else track(frame, result);
    if (!note.empty()) result.event = result.event.empty() ? note : note + "; " + result.event;
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
    // A place the map already has: tracking resumes there instead.
    {
      std::vector<std::uint64_t> ids;
      std::vector<PoseObservation> observations;
      SE3 pose;
      std::vector<char> inliers;
      if (find_place(frame, observation_map(frame), nullptr, 0, ids, observations, pose, inliers, result) < 0) return;
    }
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
    keyframes[k0].starts_stretch = true;
    const int k1 = add_keyframe(frame.frame_index, second, current);
    const auto reference_colors = color_map(*reference);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      if (!good[i]) continue;
      Landmark landmark = new_landmark(X[i] * scale, reference->frame_index, frame.frame_index, ids[i]);
      landmark.observations = {sighting(k0, ids[i]), sighting(k1, ids[i])};
      add_color(landmark, ids[i], reference_colors);
      add_color(landmark, ids[i], colors);
      add_landmark(std::move(landmark));
    }
    local_bundle_adjustment();
    cull(current);
    if (static_cast<int>(local.size()) < config.min_init_points) {
      keyframes.resize(keyframes.size() - 2);
      end_segment(false);
      restart("initial map too small after refinement; new reference");
      return;
    }
    const SE3& pose0 = keyframes[k0].pose;
    const SE3& pose1 = keyframes[k1].pose;
    std::vector<std::uint64_t> mapped;
    for (auto id : local) mapped.push_back(landmarks.at(id).track);
    std::sort(mapped.begin(), mapped.end());
    std::vector<PoseObservation> observations;
    std::vector<Vec3> points;
    for (auto track : mapped) {
      observations.push_back({landmarks.at(track_landmark.at(track)).X, current.at(track)});
      points.push_back(observations.back().X);
    }
    motion.reset(pose_covariance(K, observations, pose1, std::vector<char>(observations.size(), 1)),
                 std::max(reference_age, 1));
    report_uncertainty(image_sigma(pose1, motion.pose, points), result);
    // The first keyframe defines the map's frame: its pose is exact.
    frames.push_back({reference->frame_index, reference->timestamp_ns, segment, k0, SE3{}, true, false, 1.0});
    frames.push_back({frame.frame_index, frame.timestamp_ns, segment, k1, SE3{}, true, false, result.confidence});
    const SE3 velocity = scale_motion(pose1 * pose0.inverse(), 1.0 / std::max(reference_age, 1));
    last_pose = pose1;
    previous_pose = velocity.inverse() * pose1;
    frames_since_keyframe = 0;
    frames_since_place_search = 0;
    keyframe_landmarks = static_cast<int>(local.size());
    reference.reset();
    state = OdometryState::tracking;
    result.state = state;
    result.has_pose = true;
    result.keyframe = true;
    result.pose = to_pose(pose1);
    result.correspondences = result.inliers = static_cast<int>(local.size());
    result.pose_inliers = mapped;
    result.event = "initialized: " + std::to_string(local.size()) + " points, " +
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
    const SE3 velocity = last_pose * previous_pose.inverse();
    const SE3 predicted = velocity * last_pose;
    motion.predict();
    SE3 pose = predicted;
    std::vector<char> inliers;
    int count = 0;
    if (result.correspondences >= config.min_tracked_points) {
      count = optimize_pose(K, observations, pose, config.reprojection_threshold_px, inliers);
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
        // (P3P RANSAC on the same correspondences).
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
    }
    if (count < config.min_tracked_points) {
      const double sigma = image_sigma(predicted, motion.pose, visible_landmarks(predicted));
      count = relocalize(frame, current, predicted, sigma, nullptr, ids, observations, pose, inliers, result);
      if (count < config.min_tracked_points) {
        count = find_place(frame, current, &predicted, sigma, ids, observations, pose, inliers, result);
        if (count < 0) return;  // resumed in another segment
        if (count < config.min_tracked_points) return coast(frame, predicted, sigma, result);
      }
    }
    std::vector<double> errors;
    std::vector<std::uint64_t> inlier_ids, outlier_ids;
    for (std::size_t i = 0; i < observations.size(); ++i) {
      if (!inliers[i]) {
        outlier_ids.push_back(ids[i]);
        continue;
      }
      inlier_ids.push_back(ids[i]);
      const std::uint64_t landmark_id = track_landmark.at(ids[i]);
      auto& landmark = landmarks.at(landmark_id);
      landmark.last_frame = frame.frame_index;
      update_descriptor(landmark_id, landmark);
      errors.push_back((K.project(pose * observations[i].X) - observations[i].u).norm());
    }
    std::sort(inlier_ids.begin(), inlier_ids.end());
    std::sort(outlier_ids.begin(), outlier_ids.end());
    result.inliers = count;
    result.median_reprojection_px = median(errors);
    result.pose_inliers = inlier_ids;
    result.pose_outliers = outlier_ids;
    motion.correct(pose_difference(pose, predicted), pose_covariance(K, observations, pose, inliers), coasted + 1);
    std::vector<Vec3> points;
    for (std::size_t i = 0; i < observations.size(); ++i)
      if (inliers[i]) points.push_back(observations[i].X);
    report_uncertainty(image_sigma(pose, motion.pose, points), result);
    if (coasted > 0) {
      // Tracking resumes: the coasted frames' poses become the constant-velocity
      // path between the tracked poses on either side of the gap.
      const SE3 gap = pose * coast_origin.inverse();
      for (int i = 1; i <= coasted; ++i) {
        auto& record = frames[frames.size() - coasted + i - 1];
        record.relative = scale_motion(gap, double(i) / (coasted + 1)) * coast_origin *
                          keyframes[record.keyframe].pose.inverse();
      }
      previous_pose = scale_motion(gap, 1.0 / (coasted + 1)).inverse() * pose;
      state = OdometryState::tracking;
    } else {
      previous_pose = last_pose;
    }
    const bool resumed = coasted > 0 || result.relocalized;
    coasted = 0;
    last_pose = pose;
    // Before the keyframe step, so a re-found landmark is not triangulated again.
    const std::unordered_set<std::uint64_t> inlier_tracks(inlier_ids.begin(), inlier_ids.end());
    reassociate(frame, pose, current, inlier_tracks, result);
    if (place_ready) {
      if (place_ready->tracking) use_places(frame, pose, current, inlier_tracks, result);
      else place_ready.reset();  // made when there was no pose
      pose = last_pose;  // a correction moves the map, and this pose with it
    }
    ++frames_since_place_search;
    ++frames_since_keyframe;
    const bool emergency = count < config.keyframe_emergency_ratio * keyframe_landmarks;
    const bool keyframe = (frames_since_keyframe >= config.keyframe_min_interval &&
                           (count < config.keyframe_track_ratio * keyframe_landmarks ||
                            frames_since_keyframe >= config.keyframe_max_interval)) ||
                          emergency || resumed;  // a keyframe ties the re-found map to this frame
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
      for (auto track_id : outlier_ids) erase_landmark(track_landmark.at(track_id));
      triangulate_new(id);
      local_bundle_adjustment();
      cull(current);
      last_pose = keyframes[id].pose;
      frames_since_keyframe = 0;
      keyframe_landmarks = sightings_in(id);
      frames.push_back({frame.frame_index, frame.timestamp_ns, segment, id, SE3{}, true, false, result.confidence});
      result.keyframe = true;
      // Look for places mapped before, among the landmarks outside the local map.
      if (config.place_search_interval > 0 && frames_since_place_search >= config.place_search_interval &&
          can_search()) {
        frames_since_place_search = 0;
        request_search(frame, id, [&](std::uint64_t landmark, const Landmark&) { return !local.count(landmark); },
                       [](std::uint64_t) { return true; });
      }
    } else {
      const int reference_keyframe = segment_keyframes.back();
      frames.push_back({frame.frame_index, frame.timestamp_ns, segment, reference_keyframe,
                        pose * keyframes[reference_keyframe].pose.inverse(), false, false, result.confidence});
    }
    result.state = state;
    result.has_pose = true;
    result.pose = to_pose(last_pose);
  }

  // Local landmarks that no track observes in this frame, projected with
  // `pose` and matched by descriptor to the `eligible` tracks within `radius`
  // of the projection (mutual best, cosine >= min_descriptor_similarity).
  // Projections up to `margin` outside the image count. Returns (landmark id,
  // observation index), by landmark id.
  template <typename Eligible>
  std::vector<std::pair<std::uint64_t, int>> match_local(const TrackedFrame& frame, const SE3& pose, double radius,
                                                          double margin,
                                                          const std::unordered_map<std::uint64_t, Vec2>& current,
                                                          Eligible eligible) const {
    std::vector<std::pair<std::uint64_t, int>> matches;
    if (descriptor_dimension == 0 || radius <= 0) return matches;
    struct Candidate {
      std::uint64_t id;
      Vec2 p;
    };
    std::vector<Candidate> candidates;
    for (auto id : local) {
      const Landmark& landmark = landmarks.at(id);
      if (current.count(landmark.track) || static_cast<int>(landmark.descriptor.size()) != descriptor_dimension) continue;
      const Vec3 xc = pose * landmark.X;
      if (xc.z() <= 1e-6) continue;
      const Vec2 p = K.project(xc);
      if (p.x() < -margin || p.y() < -margin || p.x() > width - 1 + margin || p.y() > height - 1 + margin) continue;
      candidates.push_back({id, p});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& l, const auto& r) { return l.id < r.id; });
    // Eligible observations on a grid of radius-sized cells.
    const int cols = static_cast<int>(width / radius) + 1, rows = static_cast<int>(height / radius) + 1;
    const auto cell = [&](double value, int count) { return std::clamp(static_cast<int>(value / radius), 0, count - 1); };
    std::vector<std::vector<int>> grid(static_cast<std::size_t>(cols) * rows);
    for (int i = 0; i < static_cast<int>(frame.observations.size()); ++i) {
      const auto track = frame.observations[i].track_id;
      if (!descriptors.count(track) || !eligible(track)) continue;
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
      if (best_of_observation[i].second == k) matches.emplace_back(candidates[k].id, i);
    return matches;
  }

  // Re-association at a tracked pose: within reassociation_radius_px, to
  // tracks that either have no landmark or whose landmark is a pose inlier
  // (see local_map_keyframes).
  void reassociate(const TrackedFrame& frame, const SE3& pose, const std::unordered_map<std::uint64_t, Vec2>& current,
                   const std::unordered_set<std::uint64_t>& inlier_tracks, OdometryFrameResult& result) {
    const auto matches = match_local(frame, pose, config.reassociation_radius_px, 0, current, [&](std::uint64_t track) {
      return !track_landmark.count(track) || inlier_tracks.count(track);
    });
    for (const auto& [id, i] : matches) associate(id, frame.observations[i].track_id, frame, result);
  }

  // ------------------------------------------------------------------------
  // Search without a pose prior (place recognition). A search matches a
  // frame's tracks to landmarks by descriptor alone and fits a P3P pose per
  // segment. It works from a snapshot (PlaceQuery) on a worker thread, unless
  // place_synchronous, and its result is used on a later frame through the
  // tracks that are still observed then.
  static constexpr std::uint64_t kNone = std::numeric_limits<std::uint64_t>::max();
  struct PlaceQuery {
    std::uint64_t frame_index{};
    Intrinsics K;
    VisualOdometryConfig config;
    std::vector<float> descriptors;  // one row per track
    std::vector<std::uint64_t> tracks;
    std::vector<Vec2> pixels;
    // A search made while tracking, at keyframe `keyframe` of `segment` with
    // pose `pose`: per track, its landmark (kNone: none) and its position.
    bool tracking{};
    int segment{}, keyframe{-1};
    SE3 pose;
    std::vector<std::uint64_t> landmarks;
    std::vector<Vec3> positions;
    // Per index entry (PlaceIndex::ids() order): searched or not, its position and segment.
    std::vector<char> accept;
    std::vector<Vec3> X;
    std::vector<int> segments;
  };
  struct PlaceMatch {
    std::uint64_t landmark, track;
    bool inlier;  // of the group's pose
  };
  using Pairs = std::vector<std::pair<std::uint64_t, std::uint64_t>>;
  struct PlaceGroup {
    int segment{};
    std::vector<PlaceMatch> matches;  // by landmark id
    int inliers{};                    // of the P3P pose; 0: none
    SE3 pose;                         // the query frame in this segment's map
    // Tracking only. offset_px: how far the tracked pose puts the inliers from
    // their matches (median; this segment). S: the similarity taking the
    // tracked map around the query frame onto this group's landmarks, and the
    // (tracked, found) landmark pairs it rests on; no pairs: no alignment.
    double offset_px{};
    Sim3 S;
    Pairs pairs;
  };
  struct PlaceResult {
    std::uint64_t frame_index{};
    bool tracking{};
    int segment{}, keyframe{-1};
    std::vector<PlaceGroup> groups;  // most matches first
  };

  // P3P RANSAC and refinement over 2D-3D matches without a starting pose.
  // Returns the inlier count (0 below `needed`).
  static int fit_pose(const Intrinsics& K, const VisualOdometryConfig& config, const std::vector<PoseObservation>& all,
                      std::uint32_t seed, int iterations, int needed, SE3& found, std::vector<char>& fit) {
    if (static_cast<int>(all.size()) < needed) return 0;
    if (ransac_pnp(K, all, config.recovery_threshold_px, iterations, seed, found, fit) < needed) return 0;
    // Refine on the inliers only, twice: most descriptor matches without a
    // prior are wrong, and a robust fit over all of them is pulled off the pose.
    const double limit = config.reprojection_threshold_px * config.reprojection_threshold_px;
    int count = 0;
    for (int round = 0; round < 2; ++round) {
      std::vector<PoseObservation> kept;
      for (std::size_t i = 0; i < all.size(); ++i)
        if (fit[i]) kept.push_back(all[i]);
      std::vector<char> unused;
      optimize_pose(K, kept, found, config.reprojection_threshold_px, unused);
      count = 0;
      for (std::size_t i = 0; i < all.size(); ++i) {
        const Vec3 xc = found * all[i].X;
        count += fit[i] = xc.z() > 1e-6 && (K.project(xc) - all[i].u).squaredNorm() <= limit;
      }
      if (count < needed) return 0;
    }
    return count;
  }

  // The similarity between the tracked map and a group's landmarks, from the
  // group's pose and the depth ratio of the inliers whose track also has a
  // tracked landmark (a 3D-3D fit is too noisy: young landmarks have poor
  // depth). The ratio must be consistent: p75 <= loop_scale_spread x p25, and
  // at least loop_min_pairs pairs within 25% of the median.
  static void align(const PlaceQuery& q, const std::vector<std::pair<std::size_t, int>>& found,
                    const std::vector<PoseObservation>& all, const std::vector<char>& fit, PlaceGroup& group) {
    std::vector<double> ratios, offsets;
    std::vector<std::size_t> used;
    for (std::size_t m = 0; m < all.size(); ++m) {
      if (!fit[m]) continue;
      if (group.segment == q.segment)
        if (const Vec3 xc = q.pose * all[m].X; xc.z() > 1e-6) offsets.push_back((q.K.project(xc) - all[m].u).norm());
      const int query = found[m].second;
      if (q.landmarks[query] == kNone) continue;
      const double z_found = (group.pose * all[m].X).z(), z_tracked = (q.pose * q.positions[query]).z();
      if (z_found <= 1e-9 || z_tracked <= 1e-9) continue;
      ratios.push_back(z_found / z_tracked);
      used.push_back(m);
    }
    group.offset_px = median(offsets);
    const std::size_t needed = static_cast<std::size_t>(q.config.loop_min_pairs);
    if (ratios.size() < needed) return;
    std::vector<double> sorted = ratios;
    std::sort(sorted.begin(), sorted.end());
    if (sorted[3 * sorted.size() / 4] > q.config.loop_scale_spread * sorted[sorted.size() / 4]) return;
    const double scale = sorted[sorted.size() / 2];  // found units per tracked unit
    Pairs pairs;
    for (std::size_t j = 0; j < used.size(); ++j)
      if (std::abs(std::log(ratios[j] / scale)) < 0.25)
        pairs.emplace_back(q.landmarks[found[used[j]].second], group.matches[used[j]].landmark);
    if (pairs.size() < needed) return;
    // Found map -> query camera in tracked units; S takes the tracked map onto the found one.
    const Sim3 corrected = Sim3{1 / scale, Mat3::Identity(), Vec3::Zero()} * Sim3::from(group.pose);
    group.S = corrected.inverse() * Sim3::from(q.pose);
    group.pairs = std::move(pairs);
  }

  // The search itself: reads only its arguments (worker thread).
  static PlaceResult search(const PlaceIndex& index, const PlaceQuery& q) {
    PlaceResult result{q.frame_index, q.tracking, q.segment, q.keyframe, {}};
    const auto& ids = index.ids();
    std::map<int, std::vector<std::pair<std::size_t, int>>> by_segment;  // (index row, query)
    for (const auto& m : index.match(q.descriptors.data(), static_cast<int>(q.tracks.size()),
                                     q.config.min_descriptor_similarity, q.config.place_ratio, q.accept))
      by_segment[q.segments[m.row]].emplace_back(m.row, m.query);
    std::vector<int> order;
    for (const auto& [id, found] : by_segment) order.push_back(id);
    std::stable_sort(order.begin(), order.end(), [&](int l, int r) { return by_segment[l].size() > by_segment[r].size(); });
    for (int id : order) {
      const auto& found = by_segment[id];
      PlaceGroup group;
      group.segment = id;
      std::vector<PoseObservation> all;
      for (const auto& [row, query] : found) {
        group.matches.push_back({ids[row], q.tracks[query], false});
        all.push_back({q.X[row], q.pixels[query]});
      }
      std::vector<char> fit;
      const std::uint32_t seed = static_cast<std::uint32_t>(q.frame_index * 2246822519U + 2 + id);
      group.inliers = fit_pose(q.K, q.config, all, seed, q.config.place_ransac_iterations, q.config.place_min_inliers,
                               group.pose, fit);
      if (group.inliers > 0) {
        for (std::size_t m = 0; m < fit.size(); ++m) group.matches[m].inlier = fit[m];
        if (q.tracking) align(q, found, all, fit, group);
      }
      result.groups.push_back(std::move(group));
    }
    return result;
  }

  // The index is read by a running search: changes wait in a log until it finishes.
  void index_set(std::uint64_t id, const float* descriptor) {
    if (place_job.valid()) place_log.emplace_back(id, std::vector<float>(descriptor, descriptor + descriptor_dimension));
    else places.set(id, descriptor, descriptor_dimension);
  }
  void index_erase(std::uint64_t id) {
    if (place_job.valid()) place_log.emplace_back(id, std::vector<float>{});
    else places.erase(id);
  }
  [[nodiscard]] bool can_search() const {
    return config.place_search_interval >= 0 && !place_job.valid() && !place_ready && !correction &&
           descriptor_dimension > 0 && descriptor_dimension == places.dimension() && places.size() > 0;
  }

  // Start a search for this frame's `eligible` tracks over the landmarks
  // `accept` allows. `keyframe` >= 0: made while tracking, at that keyframe.
  template <typename Accept, typename Eligible>
  void request_search(const TrackedFrame& frame, int keyframe, Accept accept, Eligible eligible) {
    PlaceQuery q;
    q.frame_index = frame.frame_index;
    q.K = K;
    q.config = config;
    q.tracking = keyframe >= 0;
    q.segment = segment;
    q.keyframe = keyframe;
    if (q.tracking) q.pose = keyframes[keyframe].pose;
    for (std::size_t i = 0; i < frame.observations.size(); ++i) {
      const auto& o = frame.observations[i];
      if (!eligible(o.track_id)) continue;
      const float* d = frame.descriptors.data() + i * static_cast<std::size_t>(descriptor_dimension);
      q.descriptors.insert(q.descriptors.end(), d, d + descriptor_dimension);
      q.tracks.push_back(o.track_id);
      q.pixels.emplace_back(o.x, o.y);
      auto mapped = track_landmark.find(o.track_id);
      q.landmarks.push_back(mapped == track_landmark.end() ? kNone : mapped->second);
      q.positions.push_back(mapped == track_landmark.end() ? Vec3(Vec3::Zero()) : landmarks.at(mapped->second).X);
    }
    const auto& ids = places.ids();
    q.accept.reserve(ids.size());
    q.X.reserve(ids.size());
    q.segments.reserve(ids.size());
    for (auto id : ids) {
      const Landmark& landmark = landmarks.at(id);
      q.accept.push_back(accept(id, landmark));
      q.X.push_back(landmark.X);
      q.segments.push_back(landmark.segment);
    }
    if (config.place_synchronous) {
      place_ready = search(places, q);
      return;
    }
    place_job = std::async(std::launch::async, [this, q = std::move(q)] { return search(places, q); });
  }

  // Collect what the workers finished, at the start of a frame.
  void poll(const TrackedFrame& frame, OdometryFrameResult& result) {
    using namespace std::chrono_literals;
    if (place_job.valid() && place_job.wait_for(0s) == std::future_status::ready) {
      place_ready = place_job.get();
      for (const auto& [id, descriptor] : place_log) {
        if (descriptor.empty()) places.erase(id);
        else places.set(id, descriptor.data(), static_cast<int>(descriptor.size()));
      }
      place_log.clear();
    }
    // Its tracks are gone after a while.
    if (place_ready && frame.frame_index > place_ready->frame_index + 30) place_ready.reset();
    if (correction && correction->solution.wait_for(0s) == std::future_status::ready) apply_correction(frame, result);
  }

  // A finished search's inlier matches of `group` that still hold on this
  // frame: the landmark exists and the track is observed (and `eligible`).
  // (landmark id, observation index), by landmark id.
  using Matches = std::vector<std::pair<std::uint64_t, int>>;
  template <typename Eligible>
  Matches hits(const TrackedFrame& frame, const PlaceGroup& group, Eligible eligible) const {
    std::unordered_map<std::uint64_t, int> index_of;
    for (int i = 0; i < static_cast<int>(frame.observations.size()); ++i) index_of.emplace(frame.observations[i].track_id, i);
    Matches found;
    for (const auto& m : group.matches) {
      if (!m.inlier || !landmarks.count(m.landmark)) continue;
      auto it = index_of.find(m.track);
      if (it != index_of.end() && eligible(m.track)) found.emplace_back(m.landmark, it->second);
    }
    return found;
  }

  // The live tracks do not give a pose. `place` null: search the local map
  // around the predicted pose, as wide as its uncertainty (`sigma`, pixels),
  // among the tracks without a landmark. Otherwise: take a finished search
  // without a pose prior, whose matches in this segment are `place`, as far
  // as they are in the local map. P3P
  // RANSAC over those matches and the live correspondences. On success the
  // matched tracks take over their landmarks, the correspondences and `pose`
  // describe the fit, and the inlier count is returned; otherwise 0, with
  // nothing changed.
  int relocalize(const TrackedFrame& frame, const std::unordered_map<std::uint64_t, Vec2>& current,
                 const SE3& predicted, double sigma, const PlaceGroup* place, std::vector<std::uint64_t>& ids,
                 std::vector<PoseObservation>& observations, SE3& pose, std::vector<char>& inliers,
                 OdometryFrameResult& result) {
    const double radius = std::clamp(3 * sigma, config.relocalization_min_radius_px, config.relocalization_radius_px);
    const auto unmapped = [&](std::uint64_t track) { return !track_landmark.count(track); };
    // A landmark in view may be predicted outside the image, by as much as the search is wide.
    Matches matches = place ? hits(frame, *place, unmapped) :
                              match_local(frame, predicted, radius, radius, current, unmapped);
    // An old part of the segment may have drifted from here: that is resume()'s case.
    if (place)
      matches.erase(std::remove_if(matches.begin(), matches.end(),
                                   [&](const auto& match) { return !local.count(match.first); }),
                    matches.end());
    std::vector<PoseObservation> all = observations;
    for (const auto& [id, i] : matches)
      all.push_back({landmarks.at(id).X, Vec2(frame.observations[i].x, frame.observations[i].y)});
    const std::size_t live = observations.size();
    SE3 found = predicted;
    std::vector<char> fit;
    const std::uint32_t seed = static_cast<std::uint32_t>(frame.frame_index * 2246822519U);
    const int count = place ? fit_pose(K, config, all, seed + 1, config.place_ransac_iterations,
                                       config.place_min_inliers, found, fit) :
                              fit_pose(K, config, all, seed, config.ransac_iterations,
                                       config.relocalization_min_inliers, found, fit);
    if (count == 0) return 0;
    // Without a prior the evidence has to come from the descriptor matches:
    // the live correspondences agree with any pose near the last one.
    if (place && std::count(fit.begin() + static_cast<std::ptrdiff_t>(live), fit.end(), 1) < config.place_min_inliers)
      return 0;
    std::vector<double> offsets;  // how far the prediction was, in the image
    for (std::size_t m = 0; m < matches.size(); ++m) {
      if (!fit[live + m]) continue;
      const auto& [id, i] = matches[m];
      const Vec3& X = landmarks.at(id).X;
      if ((predicted * X).z() > 1e-6) offsets.push_back((K.project(predicted * X) - K.project(found * X)).norm());
      associate(id, frame.observations[i].track_id, frame, result);
      ids.push_back(frame.observations[i].track_id);
      observations.push_back(all[live + m]);
    }
    inliers.assign(fit.begin(), fit.begin() + static_cast<std::ptrdiff_t>(live));
    inliers.resize(observations.size(), 1);
    pose = found;
    result.correspondences = static_cast<int>(all.size());
    result.relocalized = true;
    char text[240];
    if (place)
      std::snprintf(text, sizeof text,
                    "relocalized after %d coasted frames by descriptor, without the predicted pose: %d inliers of %zu "
                    "matches; prediction was off by %.1f px (median), sigma %.1f px",
                    coasted, count, all.size(), median(offsets), sigma);
    else
      std::snprintf(text, sizeof text,
                    "relocalized after %d coasted frames: %d inliers of %zu matches within %.0f px; prediction was off "
                    "by %.1f px (median), sigma %.1f px",
                    coasted, count, all.size(), radius, median(offsets), sigma);
    result.event = text;
    if (place) {
      result.place_segment = segment;
      result.place_inliers = count;
    }
    return count;
  }

  // A frame without a pose from its tracks or from the search around the
  // predicted pose (`predicted` set), or without a map at all (initializing):
  // use a finished search without a pose prior, or start one. Returns the
  // relocalized inlier count (in the local map), or -1 when tracking resumed
  // elsewhere in the map (the frame is done), or 0.
  int find_place(const TrackedFrame& frame, const std::unordered_map<std::uint64_t, Vec2>& current,
                 const SE3* predicted, double sigma, std::vector<std::uint64_t>& ids,
                 std::vector<PoseObservation>& observations, SE3& pose, std::vector<char>& inliers,
                 OdometryFrameResult& result) {
    if (config.place_search_interval < 0 || landmarks.empty()) return 0;
    if (place_ready && place_ready->tracking) place_ready.reset();  // made for a pose that is gone
    for (int attempt = 0; attempt < 2; ++attempt) {
      if (place_ready) {
        const PlaceResult ready = std::move(*place_ready);
        place_ready.reset();
        if (predicted)
          for (const auto& group : ready.groups)
            if (group.segment == segment && group.inliers > 0)
              if (const int count = relocalize(frame, current, *predicted, sigma, &group, ids, observations, pose,
                                               inliers, result))
                return count;
        if (resume(frame, current, ready, result)) return -1;
      }
      if (attempt == 1 || !can_search()) break;
      // Landmarks a live track observes are already correspondences.
      request_search(frame, -1, [&](std::uint64_t id, const Landmark& landmark) {
        return !(local.count(id) && current.count(landmark.track));
      }, [&](std::uint64_t track) { return !track_landmark.count(track); });
      if (!place_ready) break;  // running on the worker: a later frame takes it
    }
    return 0;
  }

  int sightings_in(int keyframe) const {
    int count = 0;
    for (auto id : local)
      for (const auto& o : landmarks.at(id).observations) count += o.keyframe == keyframe;
    return count;
  }

  // Tracking continues, from this frame, at a place mapped before: in the
  // segment with the most matches in the finished search `ready` whose inlier
  // matches still give a P3P pose on this frame (another segment, or an old
  // part of this one). The local map is dropped; the frame becomes a keyframe
  // whose local keyframes are the old keyframes that saw the landmarks found.
  // It is tied to the most covisible of them (`links`), not to the keyframe
  // before it: the camera's motion in between is unknown. Returns false, with
  // nothing changed, when no segment gives a pose.
  bool resume(const TrackedFrame& frame, const std::unordered_map<std::uint64_t, Vec2>& current,
              const PlaceResult& ready, OdometryFrameResult& result) {
    int target = -1, count = 0;
    SE3 pose;
    std::vector<char> fit;
    std::vector<PoseObservation> all;
    Matches group;
    for (const auto& candidate : ready.groups) {
      if (candidate.inliers == 0) continue;
      group = hits(frame, candidate, [](std::uint64_t) { return true; });
      all.clear();
      for (const auto& [landmark, i] : group)
        all.push_back({landmarks.at(landmark).X, Vec2(frame.observations[i].x, frame.observations[i].y)});
      const std::uint32_t seed = static_cast<std::uint32_t>(frame.frame_index * 2246822519U + 2 + candidate.segment);
      count = fit_pose(K, config, all, seed, config.place_ransac_iterations, config.place_min_inliers, pose, fit);
      if (count > 0) {
        target = candidate.segment;
        break;
      }
    }
    if (target < 0) return false;
    const std::size_t matched = group.size();

    if (state == OdometryState::coasting) frames.resize(frames.size() - coasted);  // never confirmed
    end_segment();
    segment = target;
    for (const auto& kf : keyframes)
      if (kf.segment == segment) segment_keyframes.push_back(kf.id);
    // Local keyframes: those with the most sightings of the landmarks found, most last.
    std::map<int, int> sightings;
    for (std::size_t m = 0; m < group.size(); ++m)
      if (fit[m])
        for (const auto& o : landmarks.at(group[m].first).observations) ++sightings[o.keyframe];
    std::vector<std::pair<int, int>> ranked;  // (sightings, keyframe)
    for (const auto& [keyframe, n] : sightings) ranked.emplace_back(n, keyframe);
    std::sort(ranked.begin(), ranked.end());
    if (ranked.size() > static_cast<std::size_t>(config.local_map_keyframes))
      ranked.erase(ranked.begin(), ranked.end() - config.local_map_keyframes);
    std::unordered_set<int> recent;
    for (const auto& [n, keyframe] : ranked) {
      local_keyframes.push_back(keyframe);
      recent.insert(keyframe);
    }
    for (const auto& [id, landmark] : landmarks)
      if (landmark.segment == segment &&
          std::any_of(landmark.observations.begin(), landmark.observations.end(),
                      [&](const Sighting& o) { return recent.count(o.keyframe) > 0; }))
        local.insert(id);

    std::vector<PoseObservation> observations;
    std::vector<Vec3> points;
    std::vector<std::uint64_t> tracks;
    for (std::size_t m = 0; m < group.size(); ++m) {
      if (!fit[m]) continue;
      const auto& [id, i] = group[m];
      associate(id, frame.observations[i].track_id, frame, result);
      observations.push_back(all[m]);
      points.push_back(all[m].X);
      tracks.push_back(frame.observations[i].track_id);
    }
    motion.reset(pose_covariance(K, observations, pose, std::vector<char>(observations.size(), 1)), 1);
    report_uncertainty(image_sigma(pose, motion.pose, points), result);
    const int id = add_keyframe(frame.frame_index, pose, current);
    keyframes[id].starts_stretch = true;
    links.emplace_back(id, ranked.back().second);
    for (auto track : tracks) {
      auto& landmark = landmarks.at(track_landmark.at(track));
      landmark.observations.push_back(sighting(id, track));
      add_color(landmark, track, colors);
    }
    triangulate_new(id);
    local_bundle_adjustment();
    cull(current);
    last_pose = previous_pose = keyframes[id].pose;  // no velocity yet
    frames_since_keyframe = 0;
    frames_since_place_search = 0;
    keyframe_landmarks = sightings_in(id);
    frames.push_back({frame.frame_index, frame.timestamp_ns, segment, id, SE3{}, true, false, result.confidence});
    state = OdometryState::tracking;
    std::sort(tracks.begin(), tracks.end());
    result.state = state;
    result.has_pose = true;
    result.keyframe = true;
    result.relocalized = true;
    result.pose = to_pose(last_pose);
    result.correspondences = static_cast<int>(matched);
    result.inliers = count;
    result.pose_inliers = tracks;
    result.place_segment = segment;
    result.place_inliers = count;
    result.event = "resumed segment " + std::to_string(segment) + " at keyframe " +
                   std::to_string(keyframes[ranked.back().second].frame_index) +
                   " by descriptor, without a pose prior: " +
                   std::to_string(count) + " inliers of " + std::to_string(matched) + " matches";
    return true;
  }

  // While tracking: use a finished search (made at a keyframe of this segment,
  // over the landmarks outside the local map). Matches in this segment that
  // project within reassociation_radius_px of their track under the tracked
  // pose are re-associated like local ones. A group with an alignment is a
  // place mapped before: with drift in this segment (a loop, corrected by a
  // pose graph, see close_loop()), or in another segment (merged, see
  // merge_segments()).
  void use_places(const TrackedFrame& frame, const SE3& pose, const std::unordered_map<std::uint64_t, Vec2>& current,
                  const std::unordered_set<std::uint64_t>& inlier_tracks, OdometryFrameResult& result) {
    const PlaceResult ready = std::move(*place_ready);
    place_ready.reset();
    if (ready.segment != segment) return;  // the segment changed meanwhile
    for (const auto& group : ready.groups) {
      if (group.segment == segment && config.reassociation_radius_px > 0)
        for (const auto& m : group.matches) {
          auto landmark = landmarks.find(m.landmark);
          auto u = current.find(m.track);
          if (landmark == landmarks.end() || u == current.end() || local.count(m.landmark)) continue;
          if (!reprojects(pose, landmark->second.X, u->second, config.reassociation_radius_px)) continue;
          if (track_landmark.count(m.track) && !inlier_tracks.count(m.track)) continue;
          associate(m.landmark, m.track, frame, result);
          ++result.place_reassociated;
        }
      if (group.inliers > result.place_inliers) {
        result.place_segment = group.segment;
        result.place_inliers = group.inliers;
        result.place_offset_px = group.offset_px;
      }
    }
    if (!config.place_correction || correction || keyframes_since_correction < config.loop_cooldown_keyframes) return;
    for (const auto& group : ready.groups) {
      if (group.pairs.empty()) continue;
      if (group.segment != segment) merge_segments(frame, ready, group, result);
      else if (!close_loop(frame, ready, group, result)) continue;
      break;
    }
  }

  // Absorb landmark `absorbed` into `kept` (the same point): sightings of
  // keyframes both saw keep `kept`'s.
  void merge_landmarks(std::uint64_t kept, std::uint64_t absorbed) {
    auto node = landmarks.extract(absorbed);
    Landmark& gone = node.mapped();
    if (auto t = track_landmark.find(gone.track); t != track_landmark.end() && t->second == absorbed)
      track_landmark.erase(t);
    local.erase(absorbed);
    index_erase(absorbed);
    Landmark& survivor = landmarks.at(kept);
    for (const auto& o : gone.observations)
      if (std::none_of(survivor.observations.begin(), survivor.observations.end(),
                       [&](const Sighting& k) { return k.keyframe == o.keyframe; }))
        survivor.observations.push_back(o);
    std::sort(survivor.observations.begin(), survivor.observations.end(),
              [](const Sighting& l, const Sighting& r) { return l.keyframe < r.keyframe; });
    for (int c = 0; c < 3; ++c) survivor.color_sum[c] += gone.color_sum[c];
    survivor.color_count += gone.color_count;
    survivor.first_frame = std::min(survivor.first_frame, gone.first_frame);
  }

  // Each (tracked, found) pair of a correction is one point: merge them, the
  // live track observing the survivor. Returns the number merged.
  int merge_pairs(const Pairs& pairs, const TrackedFrame& frame, OdometryFrameResult& result) {
    int merged = 0;
    for (const auto& [tracked, found] : pairs) {
      if (tracked == found || !landmarks.count(tracked) || !landmarks.count(found)) continue;
      const std::uint64_t track = landmarks.at(tracked).track;
      auto live = track_landmark.find(track);
      if (live != track_landmark.end() && live->second == tracked && descriptors.count(track)) {
        associate(found, track, frame, result);  // merges, and keeps the track on the survivor
      } else {
        const bool keep_found = landmarks.at(found).observations.size() >= landmarks.at(tracked).observations.size();
        const bool mapped = live != track_landmark.end() && live->second == tracked;
        const std::uint64_t kept = keep_found ? found : tracked;
        merge_landmarks(kept, keep_found ? tracked : found);
        if (mapped) {
          landmarks.at(kept).track = track;
          track_landmark[track] = kept;
        }
        local.insert(kept);
        ++result.merged;
      }
      ++merged;
    }
    return merged;
  }

  // The old keyframes that saw the landmarks a correction found join the
  // local keyframes, behind the bundle-adjustment window, with what they saw:
  // now that the map agrees there, those landmarks re-associate at a pose.
  void extend_local(const Pairs& pairs) {
    std::map<int, int> votes;
    for (const auto& [tracked, found] : pairs)
      if (auto it = landmarks.find(found); it != landmarks.end())
        for (const auto& o : it->second.observations) ++votes[o.keyframe];
    const std::unordered_set<int> present(local_keyframes.begin(), local_keyframes.end());
    std::vector<std::pair<int, int>> ranked;  // (votes, keyframe)
    for (const auto& [keyframe, n] : votes)
      if (!present.count(keyframe) && keyframes[keyframe].segment == segment) ranked.emplace_back(n, keyframe);
    std::sort(ranked.begin(), ranked.end());
    const std::size_t room = static_cast<std::size_t>(config.local_map_keyframes - config.window_keyframes);
    if (ranked.size() > room) ranked.erase(ranked.begin(), ranked.end() - static_cast<std::ptrdiff_t>(room));
    std::unordered_set<int> added;
    std::vector<int> block;
    for (const auto& [n, keyframe] : ranked) {
      block.push_back(keyframe);
      added.insert(keyframe);
    }
    const std::size_t window = std::min<std::size_t>(config.window_keyframes, local_keyframes.size());
    local_keyframes.insert(local_keyframes.end() - static_cast<std::ptrdiff_t>(window), block.begin(), block.end());
    for (const auto& [id, landmark] : landmarks)
      if (landmark.segment == segment &&
          std::any_of(landmark.observations.begin(), landmark.observations.end(),
                      [&](const Sighting& o) { return added.count(o.keyframe) > 0; }))
        local.insert(id);
  }

  // Move segment `seg` by `correction(keyframe)`: the similarity (old map ->
  // new map) of whatever is anchored at that keyframe. Keyframes become rigid
  // poses in the new units, landmarks move with their latest keyframe, and
  // the frames stored relative to a keyframe are rescaled to its new units.
  template <typename Correction>
  void move_segment(int seg, Correction correction) {
    std::unordered_map<int, double> unit_scale;
    int newest = -1;
    for (auto& kf : keyframes) {
      if (kf.segment != seg) continue;
      const Sim3 corrected = Sim3::from(kf.pose) * correction(kf.id).inverse();
      kf.pose = corrected.rigid();
      unit_scale[kf.id] = corrected.s;
      newest = kf.id;
    }
    if (newest < 0) return;
    for (auto& [id, landmark] : landmarks)
      if (landmark.segment == seg)
        landmark.X = correction(landmark.observations.empty() ? newest : landmark.observations.back().keyframe) *
                     landmark.X;
    for (std::size_t i = 0; i < frames.size(); ++i) {
      if (frames[i].segment != seg) continue;
      frames[i].relative.t /= unit_scale.at(frames[i].keyframe);
      changed_from = std::min(changed_from, i);
    }
    if (seg == segment)
      for (SE3* pose : {&last_pose, &previous_pose, &coast_origin})
        *pose = (Sim3::from(*pose) * correction(newest).inverse()).rigid();
  }

  // A loop being corrected: the pose graph over the segment's keyframes is
  // solved on a worker (unless place_synchronous) and applied as a delta.
  struct Correction {
    int segment{};
    std::vector<int> nodes;     // keyframe ids, as in the problem
    std::vector<Sim3> before;
    std::future<PoseGraphProblem> solution;
    Pairs pairs;
    int keyframe{}, candidate{};  // the loop's two ends
  };

  // The place found is in this segment: the tracked map has drifted from it.
  // The alignment becomes an edge between the search's keyframe and the old
  // keyframe that saw most of the landmarks found, in a pose graph over the
  // segment's keyframes (consecutive ones, and pairs sharing >=
  // loop_covisibility_min landmarks). Returns false when the old keyframe is
  // not at least loop_min_keyframe_gap keyframes back.
  bool close_loop(const TrackedFrame& frame, const PlaceResult& ready, const PlaceGroup& group,
                  OdometryFrameResult& result) {
    const int k = ready.keyframe;
    std::unordered_map<int, int> node_of;
    for (std::size_t i = 0; i < segment_keyframes.size(); ++i) node_of[segment_keyframes[i]] = static_cast<int>(i);
    if (!node_of.count(k)) return false;
    std::map<int, int> votes;
    for (const auto& [tracked, found] : group.pairs)
      if (auto it = landmarks.find(found); it != landmarks.end())
        for (const auto& o : it->second.observations) ++votes[o.keyframe];
    int c = -1, most = 0;
    for (const auto& [keyframe, n] : votes) {
      auto node = node_of.find(keyframe);
      if (node == node_of.end() || node->second > node_of.at(k) - config.loop_min_keyframe_gap) continue;
      if (n > most) {
        most = n;
        c = keyframe;
      }
    }
    if (c < 0) return false;
    const SE3& Tk = keyframes[k].pose;
    result.loop_keyframe = static_cast<std::int64_t>(keyframes[k].frame_index);
    result.loop_candidate = static_cast<std::int64_t>(keyframes[c].frame_index);
    keyframes_since_correction = 0;
    // The map already agrees there: only join the duplicates.
    std::vector<double> depths;
    for (const auto& [tracked, found] : group.pairs)
      if (auto it = landmarks.find(tracked); it != landmarks.end()) depths.push_back((Tk * it->second.X).z());
    const double shift = (group.S * Tk.center() - Tk.center()).norm();
    if (std::abs(std::log(group.S.s)) < 0.005 && log_so3(group.S.R).norm() < 0.2 * kDegree &&
        shift < 0.005 * median(depths)) {
      extend_local(group.pairs);
      const int merged = merge_pairs(group.pairs, frame, result);
      note = "revisit of keyframe " + std::to_string(keyframes[c].frame_index) + ": the map agrees, " +
             std::to_string(merged) + " landmarks merged";
      return true;
    }
    Correction job;
    job.segment = segment;
    job.pairs = group.pairs;
    job.keyframe = k;
    job.candidate = c;
    PoseGraphProblem problem;
    for (int id : segment_keyframes) {
      job.nodes.push_back(id);
      job.before.push_back(Sim3::from(keyframes[id].pose));
      problem.nodes.push_back(job.before.back());
      problem.fixed.push_back(id == c);
    }
    const auto add_edge = [&](int a, int b, const Sim3& measured, double weight) {
      PoseGraphEdge edge;
      edge.a = a;
      edge.b = b;
      edge.measured = measured;
      edge.weight = weight;
      problem.edges.push_back(edge);
    };
    const auto relative = [&](int a, int b) { return problem.nodes[a] * problem.nodes[b].inverse(); };
    for (int i = 1; i < static_cast<int>(job.nodes.size()); ++i)
      if (!keyframes[job.nodes[i]].starts_stretch) add_edge(i, i - 1, relative(i, i - 1), 1.0);
    for (const auto& [a, b] : links)
      if (node_of.count(a) && node_of.count(b))
        add_edge(node_of.at(a), node_of.at(b), relative(node_of.at(a), node_of.at(b)), config.loop_edge_weight);
    std::map<std::pair<int, int>, int> shared;
    for (const auto& [id, landmark] : landmarks) {
      if (landmark.segment != segment) continue;
      std::vector<int> seen;
      for (const auto& o : landmark.observations)
        if (auto it = node_of.find(o.keyframe); it != node_of.end()) seen.push_back(it->second);
      std::sort(seen.begin(), seen.end());
      seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
      for (std::size_t a = 0; a < seen.size(); ++a)
        for (std::size_t b = a + 1; b < seen.size(); ++b)
          if (seen[b] - seen[a] > 1) ++shared[{seen[b], seen[a]}];
    }
    for (const auto& [edge, n] : shared)
      if (n >= config.loop_covisibility_min) add_edge(edge.first, edge.second, relative(edge.first, edge.second), 1.0);
    // The search's keyframe as the found map sees it, against the old keyframe.
    const Sim3 k_corrected = Sim3::from(Tk) * group.S.inverse();
    add_edge(node_of.at(k), node_of.at(c), k_corrected * problem.nodes[node_of.at(c)].inverse(),
             config.loop_edge_weight);
    if (config.place_synchronous) {
      std::promise<PoseGraphProblem> done;
      optimize_pose_graph(problem);
      done.set_value(std::move(problem));
      job.solution = done.get_future();
      correction = std::move(job);
      apply_correction(frame, result);
    } else {
      job.solution = std::async(std::launch::async, [problem = std::move(problem)]() mutable {
        optimize_pose_graph(problem);
        return problem;
      });
      correction = std::move(job);
      note = "loop found with keyframe " + std::to_string(keyframes[c].frame_index) + " (" +
             std::to_string(group.pairs.size()) + " landmarks); correcting on the worker";
    }
    return true;
  }

  // A solved loop correction, applied as a delta to the current estimates
  // (bundle adjustment may have moved the window meanwhile; keyframes added
  // since take the newest node's correction).
  void apply_correction(const TrackedFrame& frame, OdometryFrameResult& result) {
    Correction job = std::move(*correction);
    correction.reset();
    const PoseGraphProblem solved = job.solution.get();
    if (job.segment != segment || state == OdometryState::initializing) return;  // the segment ended meanwhile
    std::unordered_map<int, Sim3> W;  // old map -> new map, for what is anchored at each keyframe
    for (std::size_t i = 0; i < job.nodes.size(); ++i) W[job.nodes[i]] = solved.nodes[i].inverse() * job.before[i];
    const Sim3 newest = W.at(job.nodes.back());
    move_segment(segment, [&](int keyframe) -> const Sim3& {
      auto it = W.find(keyframe);
      return it != W.end() ? it->second : newest;
    });
    extend_local(job.pairs);
    const int merged = merge_pairs(job.pairs, frame, result);
    links.emplace_back(job.keyframe, job.candidate);
    keyframes_since_correction = 0;
    result.loop_closed = true;
    result.loop_keyframe = static_cast<std::int64_t>(keyframes[job.keyframe].frame_index);
    result.loop_candidate = static_cast<std::int64_t>(keyframes[job.candidate].frame_index);
    note = "loop closed: keyframe " + std::to_string(keyframes[job.keyframe].frame_index) + " with " +
           std::to_string(keyframes[job.candidate].frame_index) + ", " + std::to_string(job.nodes.size()) +
           " keyframes corrected, " + std::to_string(merged) + " landmarks merged";
  }

  // The place found belongs to another segment: the tracked segment is moved
  // into that segment's map by the alignment and becomes part of it.
  void merge_segments(const TrackedFrame& frame, const PlaceResult& ready, const PlaceGroup& group,
                      OdometryFrameResult& result) {
    const int from = segment, into = group.segment;
    std::map<int, int> votes;  // the found segment's keyframes, by sightings of the paired landmarks
    for (const auto& [tracked, found] : group.pairs)
      if (auto it = landmarks.find(found); it != landmarks.end())
        for (const auto& o : it->second.observations) ++votes[o.keyframe];
    int anchor = -1, most = 0;
    for (const auto& [keyframe, n] : votes)
      if (n > most && keyframes[keyframe].segment == into) {
        most = n;
        anchor = keyframe;
      }
    if (anchor < 0) return;
    links.emplace_back(ready.keyframe, anchor);
    const Sim3 S = group.S;
    move_segment(from, [&](int) -> const Sim3& { return S; });
    for (auto& kf : keyframes)
      if (kf.segment == from) kf.segment = into;
    for (auto& [id, landmark] : landmarks)
      if (landmark.segment == from) landmark.segment = into;
    for (auto& record : frames)
      if (record.segment == from) record.segment = into;  // move_segment() marked them changed
    segment = into;
    segment_keyframes.clear();
    for (const auto& kf : keyframes)
      if (kf.segment == segment) segment_keyframes.push_back(kf.id);
    extend_local(group.pairs);
    const int merged = merge_pairs(group.pairs, frame, result);
    keyframes_since_correction = 0;
    result.segments_merged = true;
    note = "segment " + std::to_string(from) + " merged into segment " + std::to_string(into) + ": scale " +
           std::to_string(S.s).substr(0, 5) + ", " + std::to_string(merged) + " landmarks merged";
  }

  // Declared after `places`, which a running search reads: destroyed (and
  // so waited for) first.
  std::future<PlaceResult> place_job;
  std::vector<std::pair<std::uint64_t, std::vector<float>>> place_log;  // index changes; empty descriptor: erase
  std::optional<PlaceResult> place_ready;  // a finished search, not used yet
  std::optional<Correction> correction;

  // No pose from the map this frame: report the motion model's prediction
  // with its uncertainty and keep the local map for relocalize(). The segment
  // ends once the prediction no longer constrains that search.
  void coast(const TrackedFrame& frame, const SE3& predicted, double sigma, OdometryFrameResult& result) {
    if (coasted >= config.coast_max_frames || !(sigma <= config.relocalization_radius_px)) {
      const std::string why = coasted == 0 ? "tracking lost" :
          "tracking lost: not relocalized in " + std::to_string(coasted) + " coasted frames";
      frames.resize(frames.size() - coasted);  // predictions that were never confirmed
      return lose(frame, result, why);
    }
    if (coasted == 0) coast_origin = last_pose;
    ++coasted;
    previous_pose = last_pose;
    last_pose = predicted;
    state = OdometryState::coasting;
    report_uncertainty(sigma, result);
    const int reference_keyframe = segment_keyframes.back();
    frames.push_back({frame.frame_index, frame.timestamp_ns, segment, reference_keyframe,
                      predicted * keyframes[reference_keyframe].pose.inverse(), false, true, result.confidence});
    result.state = state;
    result.has_pose = true;
    result.predicted = true;
    result.pose = to_pose(predicted);
    result.inliers = 0;
    result.event = "coasting (" + std::to_string(coasted) + " frames): pose predicted by constant velocity";
  }

  // `track` observes landmark `id`. A track that already has a landmark
  // carries a duplicate of the same point: the one with fewer keyframe
  // sightings is folded into the other (keyframes both saw keep the
  // survivor's sighting). BA and cull() then refine the survivor.
  void associate(std::uint64_t id, std::uint64_t track, const TrackedFrame& frame, OdometryFrameResult& result) {
    std::uint64_t kept = id;
    if (auto it = track_landmark.find(track); it != track_landmark.end() && it->second != id) {
      std::uint64_t other = it->second;
      if (landmarks.at(other).observations.size() > landmarks.at(id).observations.size()) std::swap(kept, other);
      merge_landmarks(kept, other);
      ++result.merged;
    }
    Landmark& landmark = landmarks.at(kept);
    if (auto it = track_landmark.find(landmark.track); it != track_landmark.end() && it->second == kept)
      track_landmark.erase(it);
    landmark.track = track;
    landmark.last_frame = frame.frame_index;
    update_descriptor(kept, landmark);
    track_landmark[track] = kept;
    local.insert(kept);
    ++result.reassociated;
  }

  // The segment ends (its landmarks stay in the map) and a new segment
  // initializes from this frame.
  void lose(const TrackedFrame& frame, OdometryFrameResult& result, const std::string& why) {
    state = OdometryState::initializing;
    result.state = OdometryState::lost;
    result.event = why;
    end_segment();
    segment = next_segment++;
    reference = frame;
  }

  int add_keyframe(std::uint64_t frame_index, const SE3& pose, std::unordered_map<std::uint64_t, Vec2> observations) {
    const int id = static_cast<int>(keyframes.size());
    keyframes.push_back({id, frame_index, segment, pose, std::move(observations)});
    segment_keyframes.push_back(id);
    local_keyframes.push_back(id);
    ++keyframes_since_correction;
    return id;
  }

  std::vector<int> window() const {
    const int count = std::min<int>(config.window_keyframes, static_cast<int>(local_keyframes.size()));
    return {local_keyframes.end() - count, local_keyframes.end()};
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
        if (landmark.observations.size() >= 2) add_landmark(std::move(landmark));
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
    for (auto id : local) {
      int seen = 0;
      for (const auto& o : landmarks.at(id).observations) seen += camera_of.count(o.keyframe) > 0;
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

  // Over the local map: drop observations that no longer reproject and points
  // left with fewer than two; points with no live track and no sighting in
  // the last local_map_keyframes local keyframes leave the local map (they
  // stay in the map).
  void cull(const std::unordered_map<std::uint64_t, Vec2>& current) {
    const std::size_t count = std::min<std::size_t>(config.local_map_keyframes, local_keyframes.size());
    const std::unordered_set<int> recent(local_keyframes.end() - static_cast<std::ptrdiff_t>(count),
                                         local_keyframes.end());
    for (auto id : sorted_local()) {
      Landmark& landmark = landmarks.at(id);
      auto& obs = landmark.observations;
      obs.erase(std::remove_if(obs.begin(), obs.end(), [&](const Sighting& o) {
        return !reprojects(keyframes[o.keyframe].pose, landmark.X, o.u, 2 * config.reprojection_threshold_px);
      }), obs.end());
      if (obs.size() < 2) {
        erase_landmark(id);
        continue;
      }
      const bool seen = std::any_of(obs.begin(), obs.end(), [&](const Sighting& o) { return recent.count(o.keyframe) > 0; });
      if (seen || current.count(landmark.track)) continue;
      if (auto t = track_landmark.find(landmark.track); t != track_landmark.end() && t->second == id)
        track_landmark.erase(t);
      local.erase(id);
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
                       to_pose(pose), record.predicted, record.confidence});
  }
  return samples;
}

std::size_t VisualOdometry::stable_prefix() const {
  const auto& frames = impl_->frames;
  if (impl_->state == OdometryState::initializing) return std::min(frames.size(), impl_->changed_from);
  const auto window = impl_->window();
  if (window.empty()) return std::min(frames.size(), impl_->changed_from);
  // Reference keyframe ids are non-decreasing along the frame records.
  const int oldest = *std::min_element(window.begin(), window.end());
  std::size_t prefix = frames.size();
  while (prefix > 0 && frames[prefix - 1].keyframe >= oldest) --prefix;
  return std::min(prefix, impl_->changed_from);
}

std::size_t VisualOdometry::trajectory_size() const { return impl_->frames.size(); }

std::vector<MapPoint> VisualOdometry::map() const {
  std::vector<MapPoint> points;
  points.reserve(impl_->landmarks.size());
  for (const auto& [id, landmark] : impl_->landmarks) points.push_back(impl_->map_point(id, landmark));
  std::sort(points.begin(), points.end(),
            [](const MapPoint& l, const MapPoint& r) { return l.landmark_id < r.landmark_id; });
  return points;
}
std::size_t VisualOdometry::map_size() const { return impl_->landmarks.size(); }
std::vector<LandmarkObservation> VisualOdometry::landmark_observations() const {
  std::vector<LandmarkObservation> observations;
  for (const auto& [id, landmark] : impl_->landmarks)
    for (const auto& o : landmark.observations)
      observations.push_back({id, impl_->keyframes[o.keyframe].frame_index});
  std::sort(observations.begin(), observations.end(), [](const LandmarkObservation& l, const LandmarkObservation& r) {
    return l.landmark_id != r.landmark_id ? l.landmark_id < r.landmark_id : l.frame_index < r.frame_index;
  });
  return observations;
}
bool VisualOdometry::has_landmark(std::uint64_t track_id) const { return impl_->track_landmark.count(track_id) > 0; }
std::vector<TrackedLandmark> VisualOdometry::tracked_landmarks() const {
  std::vector<TrackedLandmark> tracked;
  tracked.reserve(impl_->track_landmark.size());
  for (const auto& [track, id] : impl_->track_landmark) {
    const auto& landmark = impl_->landmarks.at(id);
    tracked.push_back({track, id, {float(landmark.X.x()), float(landmark.X.y()), float(landmark.X.z())},
                       landmark.first_frame});
  }
  std::sort(tracked.begin(), tracked.end(),
            [](const TrackedLandmark& l, const TrackedLandmark& r) { return l.track_id < r.track_id; });
  return tracked;
}
const OdometryFrameResult& VisualOdometry::last() const { return impl_->last; }
const VisualOdometryConfig& VisualOdometry::config() const { return impl_->config; }
}  // namespace slam_native
