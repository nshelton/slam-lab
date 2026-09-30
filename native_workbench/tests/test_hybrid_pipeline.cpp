#include "slam_native/feature_store.hpp"
#include "slam_native/flow_tracker.hpp"
#include "slam_native/optical_flow.hpp"
#include "slam_native/superpoint.hpp"
#include "slam_native/video_decoder.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

// Optional real-GPU integration check, not part of CPU-only CTest.
int main(int argc, char** argv) {
  using namespace slam_native;
  if (argc != 3) {
    std::cerr << "usage: slam-native-hybrid-test VIDEO ENGINE\n";
    return 2;
  }
  const auto root = std::filesystem::temp_directory_path() /
      ("slam-hybrid-gpu-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  try {
    std::vector<std::vector<Keypoint>> positions;
    std::vector<std::vector<std::uint64_t>> identities;
    std::vector<std::vector<Keypoint>> raw_points;
    std::vector<std::optional<FlowField>> fields;
    std::size_t supported = 0;
    double tracking_ms = 0;
    std::size_t timed_frames = 0;
    for (int pass = 0; pass < 2; ++pass) {
      auto decoder = make_ffmpeg_cuda_decoder();
      decoder->open(argv[1]);
      auto superpoint = make_tensorrt_superpoint(argv[2], {});
      OpticalFlow optical_flow;
      FlowTracker tracker({});
      FeatureStore store(root / "features.db", {});
      store.set_session(argv[1], argv[2], 1024, 576, 2048, 0.0005F);
      for (int index = 0; index < 90; ++index) {
        GpuFrame image;
        if (!decoder->next(image)) throw std::runtime_error("Video needs at least 90 frames");
        auto cached = store.load_frame(image);
        if (cached.has_value() != (pass == 1)) throw std::runtime_error("Unexpected cache status");
        auto features = cached ? std::move(*cached) : superpoint->infer(image);
        if (!cached) store.enqueue(features);
        std::optional<DeviceFlowField> device_field;
        if (optical_flow.has_reference()) device_field = optical_flow.compute(image);
        else optical_flow.remember(image);
        std::optional<FlowField> field;
        if (device_field) field = optical_flow.download();
        if (pass == 0) {
          raw_points.push_back(features.keypoints);
          fields.push_back(field);
        } else {
          if (features.keypoints.size() != raw_points[index].size())
            throw std::runtime_error("Cache changed detection count");
          for (std::size_t p = 0; p < features.keypoints.size(); ++p) {
            const auto a = features.keypoints[p], b = raw_points[index][p];
            if (a.x != b.x || a.y != b.y || a.score != b.score)
              throw std::runtime_error("Cache changed detection coordinates/scores");
          }
          if (field) {
            std::size_t differences = 0;
            for (std::size_t p = 0; p < field->vectors.size(); ++p) {
              auto a = field->vectors[p], b = fields[index]->vectors[p];
              differences += a.dx != b.dx || a.dy != b.dy;
            }
            if (differences) throw std::runtime_error("Recomputed flow differs at frame " +
                                                      std::to_string(index));
          }
        }
        // Live pass: device detections straight from TensorRT. Replay: upload.
        const DeviceDetections device = cached ? DeviceDetections{} : superpoint->device_detections();
        tracker.associate(features, device_field ? &*device_field : nullptr,
                          cached ? nullptr : &device);
        if (index > 0) { tracking_ms += features.tracking_ms; ++timed_frames; }
        if (pass == 0) {
          positions.push_back(features.keypoints);
          identities.push_back(features.landmark_ids);
          supported += features.matched_landmarks;
          if (features.keypoints.size() > 1000) throw std::runtime_error("Track cap exceeded");
        } else {
          if (features.landmark_ids != identities[index])
            throw std::runtime_error("Replay changed identities at frame " + std::to_string(index));
          for (std::size_t p = 0; p < features.keypoints.size(); ++p) {
            if (features.keypoints[p].x != positions[index][p].x ||
                features.keypoints[p].y != positions[index][p].y)
              throw std::runtime_error("Replay changed track positions");
          }
        }
      }
      store.flush();
      if (store.persisted() != 90) throw std::runtime_error("Replay wrote extra cache rows");
    }
    if (!supported) throw std::runtime_error("Video did not exercise hybrid tracking");
    std::cout << "90 frames plus cached replay: " << supported << " supported associations, "
              << (tracking_ms / timed_frames) << " ms mean tracking; replay agrees.\n";
    std::filesystem::remove_all(root);
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\nTemporary cache: " << root << '\n';
    return 1;
  }
}
