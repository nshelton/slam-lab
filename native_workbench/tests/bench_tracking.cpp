// Headless pipeline benchmark on real video (GPU): decode, SuperPoint, flow
// tracking and visual odometry, as the app runs them. Not part of CTest.
// usage: slam-native-tracking-bench VIDEO ENGINE [START_SECONDS] [FRAMES] [KEY VALUE]...
#include "slam_native/color_sampler.hpp"
#include "slam_native/feature_store.hpp"
#include "slam_native/flow_tracker.hpp"
#include "slam_native/optical_flow.hpp"
#include "slam_native/superpoint.hpp"
#include "slam_native/track_io.hpp"
#include "slam_native/video_decoder.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <stdexcept>
#include <unordered_map>

using namespace slam_native;
namespace {
float percentile(std::vector<float> values, double q) {
  if (values.empty()) return NAN;
  std::sort(values.begin(), values.end());
  return values[std::min(values.size() - 1, static_cast<std::size_t>(q * (values.size() - 1) + 0.5))];
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: slam-native-tracking-bench VIDEO ENGINE [START_SECONDS] [FRAMES] [KEY VALUE]...\n";
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
    VisualOdometryConfig vo_config;
    std::string export_tracks, export_trajectory, export_map, export_features;
    int skip = 0;  // frames decoded and dropped first (frame-exact, unlike a seek); indices count them
    for (int a = 5; a + 1 < argc; a += 2) {
      const std::string key = argv[a];
      if (key == "export-tracks") { export_tracks = argv[a + 1]; continue; }
      if (key == "export-trajectory") { export_trajectory = argv[a + 1]; continue; }
      if (key == "export-map") { export_map = argv[a + 1]; continue; }
      // Raw SuperPoint detections + descriptors (f16), in the app's feature-cache format.
      if (key == "export-features") { export_features = argv[a + 1]; continue; }
      const float value = std::stof(argv[a + 1]);
      if (key == "skip") skip = static_cast<int>(value);
      else if (key == "hfov") vo_config.horizontal_fov_degrees = value;
      else if (key == "k1") vo_config.distortion_k1 = value;
      else if (key == "fx" || key == "fy" || key == "cx" || key == "cy") {  // full intrinsics (all four)
        if (!vo_config.intrinsics) vo_config.intrinsics = CameraIntrinsics{};
        (key == "fx" ? vo_config.intrinsics->fx : key == "fy" ? vo_config.intrinsics->fy :
         key == "cx" ? vo_config.intrinsics->cx : vo_config.intrinsics->cy) = value;
      }
      else if (key == "kf-min") vo_config.keyframe_min_interval = static_cast<int>(value);
      else if (key == "kf-max") vo_config.keyframe_max_interval = static_cast<int>(value);
      else if (key == "kf-emergency-ratio") vo_config.keyframe_emergency_ratio = value;
      else if (key == "window") vo_config.window_keyframes = static_cast<int>(value);
      else if (key == "local-map") vo_config.local_map_keyframes = static_cast<int>(value);
      else if (key == "reassoc-radius") vo_config.reassociation_radius_px = value;
      else if (key == "radius") config.association_radius = value;
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

    int vo_posed = 0, vo_keyframes = 0, vo_losses = 0, vo_reassociated = 0, vo_merged = 0;
    double vo_ms = 0, vo_max_ms = 0;
    std::vector<float> vo_inlier_ratio, vo_reprojection;
    std::unordered_map<std::uint64_t, std::uint64_t> length;  // live track -> supported observations
    std::vector<std::uint64_t> finished_lengths;
    std::vector<float> similarities;
    std::size_t previous_live = 0, continued = 0, detections = 0;
    double tracking_ms = 0, tracking_gpu_ms = 0, flow_ms = 0, sp_ms = 0;
    std::size_t coasted = 0;
    int processed = 0;
    std::map<int, std::size_t> cohort;  // frames since start -> survivors from frame-0 cohort
    std::vector<std::uint64_t> first_ids;
    for (int index = 0; index < skip; ++index) {
      GpuFrame image;
      if (!decoder->next(image)) break;
    }
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
      detections += features.keypoints.size();
      if (feature_store) feature_store->enqueue(features);
      tracker.associate(features, device_field ? &*device_field : nullptr, &device);
      const auto tracked = tracked_frame_from(features, color_sampler.sample(image, features.keypoints));
      if (tracks_out) {
        if (!tracks_header) { write_tracks_header(tracks_out, tracked.width, tracked.height, true); tracks_header = true; }
        write_tracks(tracks_out, tracked, true);
      }
      const auto pose = odometry.process(tracked);
      vo_posed += pose.has_pose;
      vo_keyframes += pose.keyframe;
      vo_losses += pose.state == OdometryState::lost;
      vo_reassociated += pose.reassociated;
      vo_merged += pose.merged;
      if (pose.state == OdometryState::lost || pose.event == "pose re-estimated (P3P RANSAC)")
        std::cout << "  vo frame " << pose.frame_index << ": " << pose.event << '\n';
      vo_ms += pose.ms;
      vo_max_ms = std::max(vo_max_ms, pose.ms);
      if (pose.has_pose && pose.correspondences > 0) {
        vo_inlier_ratio.push_back(float(pose.inliers) / pose.correspondences);
        vo_reprojection.push_back(float(pose.median_reprojection_px));
      }
      if (index > 0) {
        sp_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        flow_ms += optical_flow.gpu_ms();
        tracking_ms += features.tracking_ms;
        tracking_gpu_ms += features.tracking_gpu_ms;
        coasted += features.coasted_landmarks;
        ++processed;
      }
      // Continued = a previously live track supported by a detection this frame.
      std::unordered_map<std::uint64_t, std::uint64_t> next_length;
      std::size_t matched_now = 0;
      for (std::size_t k = 0; k < features.keypoints.size(); ++k) {
        const auto id = features.landmark_ids[k];
        const bool supported = features.superpoint_supported.empty() || features.superpoint_supported[k];
        if (supported && length.count(id)) {
          ++matched_now;
          if (k < features.landmark_similarities.size() && std::isfinite(features.landmark_similarities[k]))
            similarities.push_back(features.landmark_similarities[k]);
        }
        next_length[id] = (length.count(id) ? length[id] : 0) + (supported ? 1 : 0);
      }
      for (const auto& [id, n] : length) if (!next_length.count(id)) finished_lengths.push_back(n);
      if (index == 0) {
        first_ids = features.landmark_ids;
      } else {
        continued += matched_now;
        previous_live += length.size();
        std::size_t alive = 0;
        for (auto id : first_ids) alive += next_length.count(id);
        cohort[index] = alive;
      }
      length = std::move(next_length);
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
    std::map<int, int> segment_poses;
    for (const auto& sample : odometry.trajectory()) ++segment_poses[sample.segment];
    int largest = 0;
    for (const auto& [id, n] : segment_poses) largest = std::max(largest, n);
    std::printf("camera pose: posed %d/%d frames, %d keyframes, %d losses, %zu segments (largest %d poses), "
                "inlier ratio median %.3f, median reprojection %.2f px, %.2f ms/frame (max %.1f)\n",
                vo_posed, processed + 1, vo_keyframes, vo_losses, segment_poses.size(), largest,
                percentile(vo_inlier_ratio, 0.5), percentile(vo_reprojection, 0.5), vo_ms / (processed + 1), vo_max_ms);
    std::printf("landmarks: %zu (%zu retired), re-associated %d (%d merged)\n",
                odometry.retired_count() + odometry.active_map().size(), odometry.retired_count(), vo_reassociated,
                vo_merged);
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
