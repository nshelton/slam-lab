// Headless tracking benchmark on real video (GPU). Not part of CTest.
// usage: slam-native-tracking-bench VIDEO ENGINE [START_SECONDS] [FRAMES]
#include "slam_native/color_sampler.hpp"
#include "slam_native/depth_estimator.hpp"
#include "slam_native/depth_sampling.hpp"
#include "slam_native/keyframe_depth.hpp"
#include "slam_native/feature_store.hpp"
#include "slam_native/flow_tracker.hpp"
#include "slam_native/optical_flow.hpp"
#include "slam_native/superpoint.hpp"
#include "slam_native/track_io.hpp"
#include "slam_native/video_decoder.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <stdexcept>
#include <thread>
#include <unordered_map>

using namespace slam_native;
namespace {
float percentile(std::vector<float> values, double q) {
  if (values.empty()) return NAN;
  std::sort(values.begin(), values.end());
  return values[std::min(values.size() - 1, static_cast<std::size_t>(q * (values.size() - 1) + 0.5))];
}
float nearest(const std::vector<Keypoint>& points, float x, float y) {
  float best = INFINITY;
  for (const auto& p : points) best = std::min(best, std::hypot(p.x - x, p.y - y));
  return best;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: slam-native-tracking-bench VIDEO ENGINE [START_SECONDS] [FRAMES]\n";
    return 2;
  }
  const double start = argc > 3 ? std::stod(argv[3]) : 30.0;
  const int frames = argc > 4 ? std::stoi(argv[4]) : 600;
  try {
    auto decoder = make_ffmpeg_cuda_decoder();
    decoder->open(argv[1]);
    if (start > 0) decoder->seek_ns(decoder->start_time_ns() + static_cast<std::int64_t>(start * 1e9));
    FlowTrackerConfig config;
    OpticalFlowConfig flow_config;
    SuperPointConfig sp_config;
    std::string export_tracks, export_trajectory, export_map, export_features, depth_model, export_keyframe_depth;
    std::string export_clouds;
    DepthSamplingConfig depth_sampling;
    bool dense_clouds = false;
    KeyframeDepthStore keyframe_depth;
    VisualOdometryConfig vo_config;
    for (int a = 5; a + 1 < argc; a += 2) {
      const std::string key = argv[a];
      if (key == "export-tracks") { export_tracks = argv[a + 1]; continue; }
      if (key == "export-trajectory") { export_trajectory = argv[a + 1]; continue; }
      if (key == "export-map") { export_map = argv[a + 1]; continue; }
      // Raw SuperPoint detections + descriptors (f16), in the app's feature-cache format.
      if (key == "export-features") { export_features = argv[a + 1]; continue; }
      // Depth network (preset name or engine stem, e.g. depth-anything-v2-s-vkitti), run synchronously
      // on every frame and sampled at the tracks (DEPTH_INTEGRATION.md); "off" disables.
      if (key == "depth") { depth_model = argv[a + 1] == std::string("off") ? "" : argv[a + 1]; continue; }
      if (key == "depth-edge") { depth_sampling.edge_ratio = std::stof(argv[a + 1]); continue; }
      if (key == "depth-drop-edges") { depth_sampling.drop_edges = std::stof(argv[a + 1]) != 0; continue; }
      // Each keyframe's raw depth map (DIR/kf_<frame>.f32, row-major float32) and
      // DIR/keyframes.csv (geometry and scale estimates), for tools/depth_consistency.py.
      if (key == "export-keyframe-depth") { export_keyframe_depth = argv[a + 1]; continue; }
      // Build the app's keyframe depth clouds (alignment + multi-view check) and report them.
      if (key == "dense") { dense_clouds = std::stof(argv[a + 1]) != 0; continue; }
      // With dense 1: each cloud's points at the end, DIR/clouds.csv
      // (frame_index,x,y,depth_m,shown,agree,disagree,sigma), x/y native source pixels.
      if (key == "export-clouds") { export_clouds = argv[a + 1]; continue; }
      // Multi-view check tolerance: 0 fixed 6 % (default), 1 adaptive (between-view model).
      // dense-adaptive 2: adaptive but never looser than the fixed tolerance (tightening only).
      if (key == "dense-adaptive") {
        KeyframeDepthConfig dense_config;
        const float mode = std::stof(argv[a + 1]);
        dense_config.adaptive_tolerance = mode != 0;
        if (mode == 2) dense_config.tolerance_max = dense_config.consistency_tolerance;
        keyframe_depth = KeyframeDepthStore(dense_config);
        continue;
      }
      const float value = std::stof(argv[a + 1]);
      if (key == "hfov") { vo_config.horizontal_fov_degrees = value; continue; }
      if (key == "k1") { vo_config.distortion_k1 = value; continue; }
      if (key == "fx" || key == "fy" || key == "cx" || key == "cy") {  // full intrinsics (all four)
        if (!vo_config.intrinsics) vo_config.intrinsics = CameraIntrinsics{};
        (key == "fx" ? vo_config.intrinsics->fx : key == "fy" ? vo_config.intrinsics->fy :
         key == "cx" ? vo_config.intrinsics->cx : vo_config.intrinsics->cy) = value;
        continue;
      }
      if (key == "reloc") { vo_config.relocalization = value != 0; continue; }
      // Loop closure on/off, and solving it synchronously (deterministic) vs on a worker.
      if (key == "loop") { vo_config.loop_closure = value != 0; continue; }
      if (key == "loop-sync") { vo_config.loop_synchronous = value != 0; continue; }
      if (key == "reassoc") { vo_config.reassociation = value != 0; continue; }
      if (key == "reassoc-radius") { vo_config.reassociation_radius_px = value; continue; }
      if (key == "merge") { vo_config.merge_landmarks = value != 0; continue; }
      if (key == "dormant") { vo_config.dormant_max_frames = static_cast<int>(value); continue; }
      if (key == "radius") config.association_radius = value;
      else if (key == "min-sim") config.min_descriptor_similarity = value;
      else if (key == "weight") config.descriptor_weight = value;
      else if (key == "coast") config.max_coast_frames = static_cast<int>(value);
      else if (key == "fb") config.forward_backward_threshold = value;
      else if (key == "rounds") config.assignment_rounds = static_cast<int>(value);
      else if (key == "flow-sigma") config.flow_sigma_px = value;
      else if (key == "det-sigma") config.detection_sigma_px = value;
      else if (key == "tracks") config.max_tracks = static_cast<std::size_t>(value);
      else if (key == "quality") flow_config.quality = static_cast<int>(value);
      else if (key == "backward") flow_config.backward = value != 0;
      else if (key == "k") sp_config.max_keypoints = static_cast<int>(value);
      else throw std::invalid_argument("unknown option " + key);
    }
    auto superpoint = make_tensorrt_superpoint(argv[2], sp_config);
    std::unique_ptr<DepthEstimator> depth;  // created on the first frame (needs its size)
    const DepthModelSpec* depth_spec = nullptr;
    double depth_ms = 0;
    std::size_t depth_samples = 0, depth_valid = 0, depth_edges = 0;
    if (!depth_model.empty()) {
      depth_spec = find_depth_model(depth_model);
      if (!depth_spec) throw std::invalid_argument("unknown depth model " + depth_model);
      depth_sampling.max_depth_m = depth_spec->max_depth_m;
    }
    OpticalFlow optical_flow(flow_config);
    FlowTracker tracker(config);
    VisualOdometry odometry(vo_config);
    ColorSampler color_sampler;
    std::unique_ptr<FeatureStore> feature_store;
    if (!export_features.empty()) {
      feature_store = std::make_unique<FeatureStore>(export_features, StoreConfig{});
      feature_store->set_session(argv[1], argv[2], sp_config.input_width, sp_config.input_height,
                                 sp_config.max_keypoints, sp_config.detection_threshold);
    }
    std::ofstream tracks_out;
    if (!export_tracks.empty()) tracks_out.open(export_tracks);
    bool tracks_header = false;
    int vo_posed = 0, vo_keyframes = 0, vo_losses = 0, vo_relocalized = 0;
    // Re-associated tracks: does the landmark then fit the pose (inlier) under its new track?
    struct Pending { std::uint64_t frame; int bin; };
    std::unordered_map<std::uint64_t, Pending> reassoc_pending;  // new track -> when, diagnostic bin
    std::map<int, std::array<int, 3>> reassoc_bins;  // bin -> inlier, outlier, unresolved
    std::size_t merged_total = 0;
    struct KeyframeScale { std::uint64_t frame; double log_scale, spread; int samples; };
    std::map<int, std::vector<KeyframeScale>> keyframe_scales;  // by segment
    std::size_t reassoc_total = 0, reassoc_inlier = 0, reassoc_outlier = 0, untracked_in_view = 0;
    // What lies at an untracked landmark's projection (nearest within 3 px):
    // [0] nothing, [1] raw detection only, [2] track without landmark, [3] track with another landmark.
    std::array<std::size_t, 4> untracked_class{};
    std::vector<float> untracked_nearest, random_nearest, tracked_landmark_nearest;
    std::mt19937 untracked_rng(1);
    double vo_ms = 0, vo_max_ms = 0;
    std::vector<float> vo_inlier_ratio, vo_reprojection;

    std::unordered_map<std::uint64_t, std::uint64_t> length;                // id -> observations
    std::vector<std::uint64_t> finished_lengths;
    std::vector<float> similarities, corrections, plus, minus, zero;
    std::size_t previous_live = 0, continued = 0, observed = 0, detections = 0;
    double tracking_ms = 0, tracking_gpu_ms = 0, flow_ms = 0, sp_ms = 0;
    std::size_t coasted = 0;
    int processed = 0;
    std::map<int, std::size_t> cohort;  // frames since start -> survivors from frame-0 cohort
    std::vector<std::uint64_t> first_ids;
    std::vector<Keypoint> previous_points;
    std::vector<std::uint64_t> previous_ids;
    std::size_t lost_none = 0, lost_taken = 0, lost_free = 0, lost_bad_flow = 0;
    std::vector<float> lost_nearest;
    for (int index = 0; index < frames; ++index) {
      GpuFrame image;
      if (!decoder->next(image)) break;
      const auto t0 = std::chrono::steady_clock::now();
      auto features = superpoint->infer(image);
      const auto t1 = std::chrono::steady_clock::now();
      std::optional<DeviceFlowField> device_field;
      if (optical_flow.has_reference()) device_field = optical_flow.compute(image);
      else optical_flow.remember(image);
      const DeviceDetections device = superpoint->device_detections();
      const auto raw = features.keypoints;
      const auto raw_descriptors = features.descriptors;
      detections += raw.size();
      if (feature_store) feature_store->enqueue(features);
      // Hot path first (timed); host diagnostics afterwards.
      tracker.associate(features, device_field ? &*device_field : nullptr, &device);
      auto tracked = tracked_frame_from(features, color_sampler.sample(image, features.keypoints));
      DepthMap frame_depth;
      if (depth_spec && !depth) {
        const int rotation = decoder->display_rotation();
        const bool sideways = rotation == 90 || rotation == 270;
        const bool portrait = (sideways ? image.width : image.height) > (sideways ? image.height : image.width);
        depth = std::make_unique<DepthEstimator>(
            *depth_spec, depth_engine_path(std::filesystem::path(argv[2]).parent_path(), *depth_spec, portrait),
            rotation);
      }
      if (depth) {
        // Synchronous, so results never depend on GPU timing.
        const auto d0 = std::chrono::steady_clock::now();
        const float focal = static_cast<float>(vo_config.intrinsics ? vo_config.intrinsics->fx :
            CameraIntrinsics::from_horizontal_fov(image.width, image.height, vo_config.horizontal_fov_degrees).fx);
        if (!depth->submit(image, focal)) throw std::runtime_error("depth estimator busy in synchronous mode");
        DepthMap& map = frame_depth;
        while (!depth->poll(map)) std::this_thread::sleep_for(std::chrono::microseconds(100));
        int edges = 0;
        depth_valid += static_cast<std::size_t>(attach_depth(map, tracked, depth_sampling, &edges));
        depth_edges += static_cast<std::size_t>(edges);
        depth_samples += tracked.observations.size();
        depth_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d0).count();
      }
      if (tracks_out) {
        const TrackColumns columns{true, true, depth != nullptr};
        if (!tracks_header) { write_tracks_header(tracks_out, tracked.width, tracked.height, columns); tracks_header = true; }
        write_tracks(tracks_out, tracked, columns);
      }
      const auto pose = odometry.process(tracked);
      if (dense_clouds) {
        if (pose.settled_keyframe >= 0)
          keyframe_depth.settle(static_cast<std::uint64_t>(pose.settled_keyframe), pose.settled_log_scale,
                                pose.settled_scale_grid_valid ? &pose.settled_scale_grid : nullptr);
        if (pose.keyframe && pose.has_pose && !frame_depth.empty()) {
          KeyframeDepthInput input;
          input.frame_index = pose.frame_index;
          input.segment = pose.segment;
          input.log_scale = std::isfinite(pose.keyframe_log_scale) ? pose.keyframe_log_scale :
              pose.metric_scale > 0 ? std::log(pose.metric_scale) : std::numeric_limits<double>::quiet_NaN();
          input.grid = pose.keyframe_scale_grid_valid ? &pose.keyframe_scale_grid : nullptr;
          input.pose = pose.pose;
          const auto K = vo_config.intrinsics ? *vo_config.intrinsics :
              CameraIntrinsics::from_horizontal_fov(image.width, image.height, vo_config.horizontal_fov_degrees);
          const auto samples = odometry.trajectory();
          keyframe_depth.add(input, frame_depth, {}, K, vo_config.distortion_k1,
                             [&](std::uint64_t f) -> std::optional<Pose> {
                               auto it = std::lower_bound(samples.begin(), samples.end(), f,
                                   [](const TrajectorySample& s, std::uint64_t x) { return s.frame_index < x; });
                               if (it == samples.end() || it->frame_index != f) return std::nullopt;
                               return it->pose;
                             });
        }
      }
      if (!export_keyframe_depth.empty() && pose.settled_keyframe >= 0 && pose.settled_scale_grid_valid) {
        const std::filesystem::path dir(export_keyframe_depth);
        std::filesystem::create_directories(dir);
        const bool fresh = !std::filesystem::exists(dir / "settled.csv");
        std::ofstream settled(dir / "settled.csv", std::ios::app);
        if (fresh) settled << "frame_index,log_scale,g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11\n";
        settled << std::setprecision(9) << pose.settled_keyframe << ',' << pose.settled_log_scale;
        for (double g : pose.settled_scale_grid) settled << ',' << g;
        settled << '\n';
      }
      if (!export_keyframe_depth.empty() && pose.keyframe && pose.has_pose && !frame_depth.empty()) {
        const std::filesystem::path dir(export_keyframe_depth);
        std::filesystem::create_directories(dir);
        std::ofstream raw(dir / ("kf_" + std::to_string(pose.frame_index) + ".f32"), std::ios::binary);
        raw.write(reinterpret_cast<const char*>(frame_depth.metres.data()),
                  static_cast<std::streamsize>(frame_depth.metres.size() * sizeof(float)));
        const bool fresh = !std::filesystem::exists(dir / "keyframes.csv");
        std::ofstream index(dir / "keyframes.csv", std::ios::app);
        if (fresh)
          index << "frame_index,segment,width,height,source_width,source_height,rotation,scale_x,scale_y,offset_x,"
                   "offset_y,keyframe_log_scale,metric_scale,g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11\n";
        index << std::setprecision(9) << pose.frame_index << ',' << pose.segment << ',' << frame_depth.width << ','
              << frame_depth.height << ',' << frame_depth.source_width << ',' << frame_depth.source_height << ','
              << frame_depth.rotation << ',' << frame_depth.scale_x << ',' << frame_depth.scale_y << ','
              << frame_depth.offset_x << ',' << frame_depth.offset_y << ',' << pose.keyframe_log_scale << ','
              << pose.metric_scale;
        for (double g : pose.keyframe_scale_grid)
          index << ',' << (pose.keyframe_scale_grid_valid ? g : std::numeric_limits<double>::quiet_NaN());
        index << '\n';
      }
      vo_posed += pose.has_pose;
      vo_keyframes += pose.keyframe;
      vo_losses += pose.state == OdometryState::lost;
      vo_relocalized += pose.relocalized;
      untracked_in_view += pose.untracked_landmarks.size();
      merged_total += static_cast<std::size_t>(pose.merged);
      if (pose.keyframe && std::isfinite(pose.keyframe_log_scale))
        keyframe_scales[pose.segment].push_back(
            {pose.frame_index, pose.keyframe_log_scale, pose.keyframe_scale_spread, pose.keyframe_scale_samples});
      if (index % 10 == 0 && !raw.empty()) {
        std::uniform_real_distribution<float> ux(0, float(image.width)), uy(0, float(image.height));
        for (const auto& p : pose.untracked_landmarks) {
          if (!p.reassociated) untracked_nearest.push_back(nearest(raw, p.x, p.y));
          random_nearest.push_back(nearest(raw, ux(untracked_rng), uy(untracked_rng)));
        }
      }
      for (const auto& p : pose.untracked_landmarks) {
        if (p.reassociated) continue;
        float best = 3.0F;
        int kind = 0;
        for (const auto& o : tracked.observations) {
          const float d = std::hypot(o.x - p.x, o.y - p.y);
          if (d < best) { best = d; kind = odometry.has_landmark(o.track_id) ? 3 : 2; }
        }
        if (kind == 0)
          for (const auto& k : raw)
            if (std::hypot(k.x - p.x, k.y - p.y) < 3.0F) { kind = 1; break; }
        ++untracked_class[kind];
      }
      for (auto it = reassoc_pending.begin(); it != reassoc_pending.end();) {
        const bool in = std::binary_search(pose.pose_inliers.begin(), pose.pose_inliers.end(), it->first);
        const bool out = std::binary_search(pose.pose_outliers.begin(), pose.pose_outliers.end(), it->first);
        if (in || out || pose.frame_index > it->second.frame + 10) {
          reassoc_inlier += in;
          reassoc_outlier += out;
          ++reassoc_bins[it->second.bin][in ? 0 : out ? 1 : 2];
          it = reassoc_pending.erase(it);
        } else {
          ++it;
        }
      }
      for (const auto& p : pose.untracked_landmarks)
        if (p.reassociated) {
          ++reassoc_total;
          float d = NAN;
          for (const auto& o : tracked.observations)
            if (o.track_id == p.new_track_id) d = std::hypot(o.x - p.x, o.y - p.y);
          // bin: dormant * 10 + distance bucket (0-1, 1-2, 2-3, 3-5, 5+ px)
          const int bucket = d < 1 ? 0 : d < 2 ? 1 : d < 3 ? 2 : d < 5 ? 3 : 4;
          reassoc_pending[p.new_track_id] = {pose.frame_index, (p.dormant ? 10 : 0) + bucket};
        }
      if (std::getenv("SLAM_VO_TRACE") && pose.has_pose)
        std::cout << "  trace " << pose.frame_index << " inl " << pose.inliers << "/" << pose.correspondences
                  << " reassoc " << pose.reassociated << " med " << pose.median_reprojection_px
                  << (pose.keyframe ? " KF" : "") << ' ' << pose.event << '\n';
      if (std::getenv("SLAM_VO_VERBOSE") && pose.relocalization_candidates > 0 && !pose.relocalized)
        std::cout << "  vo frame " << pose.frame_index << ": relocalization tried, " << pose.relocalization_candidates
                  << " descriptor matches\n";
      if (pose.loop_detected || pose.loop_closed)
        std::cout << "  vo frame " << pose.frame_index << ": " << pose.event << '\n';
      if (pose.state == OdometryState::lost || pose.relocalized || pose.event == "pose re-estimated (P3P RANSAC)")
        std::cout << "  vo frame " << pose.frame_index << ": " << pose.event << '\n';
      vo_ms += pose.ms;
      vo_max_ms = std::max(vo_max_ms, pose.ms);
      if (pose.has_pose && pose.correspondences > 0) {
        vo_inlier_ratio.push_back(float(pose.inliers) / pose.correspondences);
        vo_reprojection.push_back(float(pose.median_reprojection_px));
      }
      std::optional<FlowField> field;
      if (device_field) field = optical_flow.download();
      // Direction sanity: distance from each previous track moved by +flow/-flow/0 to nearest detection.
      if (field) {
        for (const auto& p : previous_points) {
          const auto v = field->sample(p.x, p.y);
          if (!std::isfinite(v.dx)) continue;
          plus.push_back(nearest(raw, p.x + v.dx, p.y + v.dy));
          minus.push_back(nearest(raw, p.x - v.dx, p.y - v.dy));
          zero.push_back(nearest(raw, p.x, p.y));
        }
      }
      if (index > 0) {
        sp_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        flow_ms += optical_flow.gpu_ms();
        tracking_ms += features.tracking_ms;
        tracking_gpu_ms += features.tracking_gpu_ms;
        coasted += features.coasted_landmarks;
        ++processed;
      }
      if (field) {
        std::unordered_map<std::uint64_t, int> now;
        for (std::size_t k = 0; k < features.keypoints.size(); ++k) now[features.landmark_ids[k]] = int(k);
        for (std::size_t t = 0; t < previous_points.size(); ++t) {
          if (now.count(previous_ids[t])) continue;
          const auto p = previous_points[t];
          const auto v = field->sample(p.x, p.y);
          if (!std::isfinite(v.dx)) { ++lost_bad_flow; continue; }
          float best = INFINITY; std::size_t r = 0;
          for (std::size_t j = 0; j < raw.size(); ++j) {
            const float d = std::hypot(raw[j].x - p.x - v.dx, raw[j].y - p.y - v.dy);
            if (d < best) { best = d; r = j; }
          }
          lost_nearest.push_back(best);
          if (best > 6) { ++lost_none; continue; }
          bool taken = false;
          for (const auto& q : features.keypoints) taken |= q.x == raw[r].x && q.y == raw[r].y;
          ++(taken ? lost_taken : lost_free);
        }
      }
      std::unordered_map<std::uint64_t, std::uint64_t> next_length;
      std::size_t matched_now = 0;
      // Matched = a previously live track supported by a detection this frame.
      // (Positions cannot identify the detection: with position fusion a
      // track sits between its flow prediction and the detection.)
      for (std::size_t k = 0; k < features.keypoints.size(); ++k) {
        const auto id = features.landmark_ids[k];
        const bool supported = features.superpoint_supported.empty() || features.superpoint_supported[k];
        if (supported) ++observed;
        if (supported && length.count(id)) {
          ++matched_now;
          if (k < features.landmark_similarities.size() && std::isfinite(features.landmark_similarities[k]))
            similarities.push_back(features.landmark_similarities[k]);
        }
        next_length[id] = (length.count(id) ? length[id] : 0) + (supported ? 1 : 0);
        if (k < features.correction_distances.size() && std::isfinite(features.correction_distances[k]))
          corrections.push_back(features.correction_distances[k]);
      }
      for (const auto& [id, n] : length) if (!next_length.count(id)) finished_lengths.push_back(n);
      if (index > 0) { continued += matched_now; }
      if (index == 0) first_ids = features.landmark_ids;
      else {
        std::size_t alive = 0;
        for (auto id : first_ids) alive += next_length.count(id);
        cohort[index] = alive;
      }
      if (index > 0) previous_live += length.size();
      length = std::move(next_length);
      previous_points = features.keypoints;
      previous_ids = features.landmark_ids;
    }
    for (const auto& [id, n] : length) finished_lengths.push_back(n);
    std::vector<float> lengths(finished_lengths.begin(), finished_lengths.end());
    std::size_t over10 = 0, over30 = 0, over100 = 0;
    for (auto n : finished_lengths) { over10 += n >= 10; over30 += n >= 30; over100 += n >= 100; }
    std::size_t weak = 0;
    for (auto s : similarities) weak += s < 0.7F;
    std::printf("frames=%d start=%.1fs mean_detections=%.0f\n", processed + 1, start,
                double(detections) / (processed + 1));
    std::printf("continuation_rate=%.4f (matched %zu of %zu previous live tracks)\n",
                double(continued) / std::max<std::size_t>(1, previous_live), continued, previous_live);
    std::printf("tracks=%zu mean_len=%.2f median_len=%.0f p90_len=%.0f >=10:%zu >=30:%zu >=100:%zu\n",
                lengths.size(), [&] { double s = 0; for (auto l : lengths) s += l; return s / std::max<std::size_t>(1, lengths.size()); }(),
                percentile(lengths, 0.5), percentile(lengths, 0.9), over10, over30, over100);
    std::printf("frame0_cohort=%zu survivors@10=%zu @30=%zu @60=%zu @120=%zu @300=%zu\n", first_ids.size(),
                cohort[10], cohort[30], cohort[60], cohort[120], cohort[300]);
    std::printf("match_descriptor_cos median=%.3f p10=%.3f frac<0.7=%.4f\n", percentile(similarities, 0.5),
                percentile(similarities, 0.1), double(weak) / std::max<std::size_t>(1, similarities.size()));
    std::printf("correction px median=%.2f p90=%.2f\n", percentile(corrections, 0.5), percentile(corrections, 0.9));
    std::printf("nearest-detection px from prev track: +flow median=%.2f p90=%.2f | -flow median=%.2f | zero median=%.2f p90=%.2f\n",
                percentile(plus, 0.5), percentile(plus, 0.9), percentile(minus, 0.5), percentile(zero, 0.5),
                percentile(zero, 0.9));
    std::printf("lost tracks: no detection within 6px=%zu, nearest taken by other track=%zu, nearest left free=%zu, bad flow=%zu; lost nearest px median=%.2f p75=%.2f\n",
                lost_none, lost_taken, lost_free, lost_bad_flow, percentile(lost_nearest, 0.5), percentile(lost_nearest, 0.75));
    std::map<int, int> segment_poses;
    for (const auto& sample : odometry.trajectory()) ++segment_poses[sample.segment];
    int largest = 0;
    for (const auto& [id, n] : segment_poses) largest = std::max(largest, n);
    std::printf("camera pose: posed %d/%d frames, %d keyframes, %d losses, %d relocalized, %zu segments "
                "(largest %d poses), inlier ratio median %.3f, "
                "median reprojection %.2f px, %.2f ms/frame (max %.1f)\n", vo_posed, processed + 1, vo_keyframes,
                vo_losses, vo_relocalized, segment_poses.size(), largest, percentile(vo_inlier_ratio, 0.5),
                percentile(vo_reprojection, 0.5), vo_ms / (processed + 1), vo_max_ms);
    std::printf("merged duplicate landmarks: %zu\n", merged_total);
    if (dense_clouds) {
      std::size_t points = 0, hidden = 0, confirmed = 0, unchecked = 0, settled = 0;
      for (const auto& c : keyframe_depth.clouds()) {
        settled += c.settled;
        for (std::size_t i = 0; i < c.points.size(); ++i) {
          ++points;
          hidden += !c.shown[i];
          confirmed += c.agree[i] > 0;
          unchecked += c.agree[i] == 0 && c.disagree[i] == 0;
        }
      }
      std::printf("dense clouds: %zu keyframes (%zu settled), %zu points: confirmed %.3f, hidden %.3f, unchecked %.3f\n",
                  keyframe_depth.clouds().size(), settled, points, double(confirmed) / std::max<std::size_t>(1, points),
                  double(hidden) / std::max<std::size_t>(1, points), double(unchecked) / std::max<std::size_t>(1, points));
      if (!export_clouds.empty()) {
        std::filesystem::create_directories(export_clouds);
        std::ofstream out(std::filesystem::path(export_clouds) / "clouds.csv");
        out << "frame_index,x,y,depth_m,shown,agree,disagree,sigma\n" << std::setprecision(7);
        for (const auto& c : keyframe_depth.clouds())
          for (std::size_t i = 0; i < c.raw.size(); ++i)
            out << c.frame_index << ',' << c.pixels[i][0] << ',' << c.pixels[i][1] << ',' << c.raw[i][2] << ','
                << int(c.shown[i]) << ',' << int(c.agree[i]) << ',' << int(c.disagree[i]) << ','
                << (i < c.sigma.size() ? c.sigma[i] : -1.0F) << '\n';
      }
      const auto& difference = keyframe_depth.difference_model();
      const auto& w = difference.weights();
      std::printf("dense between-view model: %s, %zu refs, sigma at 2 m flat centre %.4f; weights bias %.2f edge %.2f "
                  "log-depth %.2f radius %.2f angle %.2f\n",
                  difference.fitted() ? "fitted" : "not fitted", difference.references(),
                  difference.sigma({0, static_cast<float>(std::log(2.0)), 0, 0}), w[0], w[1], w[2], w[3], w[4]);
    }
    if (depth) {
      // Per segment: fused scale, the network's relative error on reference landmarks (keyframe spreads),
      // scale drift (log-scale slope), and the implied camera speed (walking ~1.2-1.5 m/s).
      const auto trajectory = odometry.trajectory();
      for (const auto& [seg, list] : keyframe_scales) {
        std::vector<float> logs, spreads;
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (const auto& k : list) {
          logs.push_back(float(k.log_scale));
          spreads.push_back(float(k.spread));
          const double x = double(k.frame) / 100.0;
          sx += x; sy += k.log_scale; sxx += x * x; sxy += x * k.log_scale;
        }
        const double n = double(list.size());
        const double slope = n > 1 ? (n * sxy - sx * sy) / std::max(1e-12, n * sxx - sx * sx) : 0.0;
        const double scale = odometry.segment_scale(seg);
        std::vector<float> speeds;
        for (std::size_t i = 1; i < trajectory.size(); ++i) {
          const auto& a = trajectory[i - 1];
          const auto& b = trajectory[i];
          if (a.segment != seg || b.segment != seg || b.frame_index != a.frame_index + 1) continue;
          const auto ca = a.pose.center(), cb = b.pose.center();
          const double dt = double(b.timestamp_ns - a.timestamp_ns) * 1e-9;
          if (dt > 0) speeds.push_back(float(std::hypot(cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]) * scale / dt));
        }
        std::printf("depth scale segment %d: %zu keyframes, %.4f m/unit (keyframe log-scale p10/p50/p90 %.3f/%.3f/%.3f, "
                    "drift %+.3f per 100 frames), network error on reference landmarks p50 %.3f p90 %.3f (log), "
                    "camera speed median %.2f m/s\n", seg, list.size(), scale, percentile(logs, .1), percentile(logs, .5),
                    percentile(logs, .9), slope, percentile(spreads, .5), percentile(spreads, .9), percentile(speeds, .5));
      }
      std::printf("depth: %s, %.2f ms/frame (synchronous), samples valid %.3f, edges %.3f of valid\n",
                  depth->spec().name.c_str(), depth_ms / (processed + 1),
                  double(depth_valid) / std::max<std::size_t>(1, depth_samples),
                  double(depth_edges) / std::max<std::size_t>(1, depth_valid));
    }
    std::printf("landmarks: %zu (%zu retired), re-associated %zu: next pose fit inlier %zu / outlier %zu / "
                "unresolved %zu; untracked landmarks in view %.1f per frame\n",
                odometry.retired_count() + odometry.active_map().size(), odometry.retired_count(), reassoc_total,
                reassoc_inlier, reassoc_outlier, reassoc_total - reassoc_inlier - reassoc_outlier - reassoc_pending.size(),
                double(untracked_in_view) / (processed + 1));
    {
      const double n = std::max<std::size_t>(1, untracked_class[0] + untracked_class[1] + untracked_class[2] + untracked_class[3]);
      std::printf("nearest raw detection (px) p25/p50/p75: untracked landmarks %.1f/%.1f/%.1f, random points %.1f/%.1f/%.1f\n",
                  percentile(untracked_nearest, .25), percentile(untracked_nearest, .5), percentile(untracked_nearest, .75),
                  percentile(random_nearest, .25), percentile(random_nearest, .5), percentile(random_nearest, .75));
      std::printf("untracked in view, what is within 3 px: nothing %.2f, raw detection only %.2f, "
                  "track without landmark %.2f, track with another landmark %.2f\n",
                  untracked_class[0] / n, untracked_class[1] / n, untracked_class[2] / n, untracked_class[3] / n);
    }
    for (const auto& [bin, n] : reassoc_bins) {
      static const char* names[] = {"0-1", "1-2", "2-3", "3-5", "5+"};
      std::printf("  re-associated %s, projection %s px: inlier %d outlier %d unresolved %d\n",
                  bin >= 10 ? "dormant" : "active ", names[bin % 10], n[0], n[1], n[2]);
    }
    if (!export_map.empty()) {
      std::ofstream out(export_map);
      write_map_csv(out, odometry);
    }
    if (!export_trajectory.empty()) {
      std::ofstream out(export_trajectory);
      write_trajectory_csv(out, odometry.trajectory());
    }
    std::printf("mean coasted tracks/frame=%.1f\n", double(coasted) / processed);
    std::printf("ms/frame: superpoint=%.2f flow_gpu=%.2f tracking_wall=%.2f (incl. flow wait) tracking_gpu=%.3f\n",
                sp_ms / processed, flow_ms / processed, tracking_ms / processed, tracking_gpu_ms / processed);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
