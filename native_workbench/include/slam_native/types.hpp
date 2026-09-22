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
};

struct Keypoint {
  float x{};
  float y{};
  float score{};
};

struct LandmarkState {
  std::uint64_t id{};
  std::array<float, 256> resultant{};
  std::uint64_t observation_count{};
  std::uint64_t first_frame{};
  std::uint64_t last_frame{};
  float concentration{};
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
  std::uint32_t new_landmarks{};
  std::uint32_t matched_landmarks{};
  double tracking_ms{};
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
