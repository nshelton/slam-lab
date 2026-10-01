#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace slam_native {

enum class PixelFormat { nv12, p010 };

// A reference to a decoded frame which remains owned by the decoder. The
// reference is valid until the next call to VideoDecoder::next().
struct GpuFrame {
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  std::int64_t pts{};
  int width{};
  int height{};
  PixelFormat format{PixelFormat::nv12};
  std::uintptr_t luma{};
  std::uintptr_t chroma{};
  std::size_t luma_pitch{};
  std::size_t chroma_pitch{};
  // CUevent (== cudaEvent_t) recorded on the decoder's stream after the surface
  // is complete. Consumers on their own streams must cudaStreamWaitEvent on it
  // before reading luma/chroma; null means the surface is already complete.
  void* ready_event{};
};

struct Keypoint {
  float x{};
  float y{};
  float score{};
};

// Device-resident detections for the GPU tracker, in the same order as the
// host FrameFeatures::keypoints of the same frame. Valid until the producer's
// next frame. `ready_event` (cudaEvent_t) marks completion; null = complete.
struct DeviceDetections {
  const Keypoint* points{};     // source-image pixels
  const float* descriptors{};   // count x 256, L2-normalized float32
  int count{};
  void* ready_event{};
};

struct LandmarkState {
  std::uint64_t id{};
  std::array<float, 256> resultant{};
  std::uint64_t observation_count{};
  std::uint64_t first_frame{};
  std::uint64_t last_frame{};
  float concentration{};
  float confidence{1.0F};
};

// Descriptor rows are contiguous. Each row has descriptor_dimension values.
// The inference result stays in float32 for online tracking. FeatureStore
// chooses its on-disk encoding independently.
struct FrameFeatures {
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  std::int64_t pts{};
  int width{};
  int height{};
  std::vector<Keypoint> keypoints;
  std::vector<float> descriptors;
  std::vector<std::uint64_t> landmark_ids;
  std::vector<float> landmark_similarities;
  std::vector<LandmarkState> landmark_updates;
  // Runtime-only diagnostics, aligned with tracked keypoints.
  std::vector<float> track_confidences;
  std::vector<bool> superpoint_supported;
  std::vector<Keypoint> flow_predictions;
  std::vector<float> correction_distances;
  // After tracking, when the detections' descriptors were on the host: the
  // descriptor of the detection each track matched this frame (keypoints.size()
  // x 256, aligned with keypoints; zero rows for coasted tracks). Empty otherwise.
  std::vector<float> track_descriptors;
  std::uint32_t new_landmarks{};
  std::uint32_t matched_landmarks{};
  std::uint32_t coasted_landmarks{};  // live tracks carried by flow without a detection
  double tracking_ms{};      // host wall time, including waits for flow/detections
  double tracking_gpu_ms{};  // GPU time of the tracking kernels and record copy
  double decode_ms{};
  double preprocess_ms{};
  double inference_ms{};
  double readback_ms{};
};

struct PipelineStats {
  std::uint64_t decoded_frames{};
  std::uint64_t inferred_frames{};
  std::uint64_t persisted_frames{};
  double decode_ms{};
  double preprocess_ms{};
  double inference_ms{};
  double readback_ms{};
  double storage_queue_ms{};
};

}  // namespace slam_native
