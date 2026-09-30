// Headless tracking benchmark on real video (GPU). Not part of CTest.
// usage: slam-native-tracking-bench VIDEO ENGINE [START_SECONDS] [FRAMES]
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
    std::string export_tracks, export_trajectory;
    VisualOdometryConfig vo_config;
    for (int a = 5; a + 1 < argc; a += 2) {
      const std::string key = argv[a];
      if (key == "export-tracks") { export_tracks = argv[a + 1]; continue; }
      if (key == "export-trajectory") { export_trajectory = argv[a + 1]; continue; }
      const float value = std::stof(argv[a + 1]);
      if (key == "hfov") { vo_config.horizontal_fov_degrees = value; continue; }
      if (key == "radius") config.association_radius = value;
      else if (key == "min-sim") config.min_descriptor_similarity = value;
      else if (key == "weight") config.descriptor_weight = value;
      else if (key == "coast") config.max_coast_frames = static_cast<int>(value);
      else if (key == "fb") config.forward_backward_threshold = value;
      else if (key == "rounds") config.assignment_rounds = static_cast<int>(value);
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
    std::ofstream tracks_out;
    if (!export_tracks.empty()) tracks_out.open(export_tracks);
    bool tracks_header = false;
    int vo_posed = 0, vo_keyframes = 0, vo_losses = 0;
    double vo_ms = 0, vo_max_ms = 0;
    std::vector<float> vo_inlier_ratio, vo_reprojection;

    std::unordered_map<std::uint64_t, std::vector<float>> last_descriptor;  // id -> descriptor
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
      // Hot path first (timed); host diagnostics afterwards.
      tracker.associate(features, device_field ? &*device_field : nullptr, &device);
      const auto tracked = tracked_frame_from(features);
      if (tracks_out) {
        if (!tracks_header) { write_tracks_header(tracks_out, tracked.width, tracked.height); tracks_header = true; }
        write_tracks(tracks_out, tracked);
      }
      const auto pose = odometry.process(tracked);
      vo_posed += pose.has_pose;
      vo_keyframes += pose.keyframe;
      vo_losses += pose.state == OdometryState::lost;
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
      std::unordered_map<std::uint64_t, std::vector<float>> next_descriptor;
      std::unordered_map<std::uint64_t, std::uint64_t> next_length;
      std::size_t matched_now = 0;
      for (std::size_t k = 0; k < features.keypoints.size(); ++k) {
        const auto id = features.landmark_ids[k];
        const bool supported = features.superpoint_supported.empty() || features.superpoint_supported[k];
        const auto point = features.keypoints[k];
        std::size_t r = 0;
        while (r < raw.size() && (raw[r].x != point.x || raw[r].y != point.y)) ++r;
        auto it = last_descriptor.find(id);
        if (supported && r < raw.size()) {
          std::vector<float> d(raw_descriptors.begin() + r * 256, raw_descriptors.begin() + (r + 1) * 256);
          if (it != last_descriptor.end()) {
            float dot = 0;
            for (int c = 0; c < 256; ++c) dot += d[c] * it->second[c];
            similarities.push_back(dot);
            ++matched_now;
          }
          next_descriptor[id] = std::move(d);
          ++observed;
        } else if (it != last_descriptor.end()) {
          next_descriptor[id] = it->second;
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
      last_descriptor = std::move(next_descriptor);
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
    std::printf("camera pose: posed %d/%d frames, %d keyframes, %d losses, %d segments, inlier ratio median %.3f, "
                "median reprojection %.2f px, %.2f ms/frame (max %.1f)\n", vo_posed, processed + 1, vo_keyframes,
                vo_losses, odometry.last().segment + 1, percentile(vo_inlier_ratio, 0.5),
                percentile(vo_reprojection, 0.5), vo_ms / (processed + 1), vo_max_ms);
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
