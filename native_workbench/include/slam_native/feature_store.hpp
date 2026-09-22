#pragma once

#include "slam_native/types.hpp"
#include "slam_native/online_tracker.hpp"
#include "slam_native/flow_tracker.hpp"

#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

struct sqlite3;

namespace slam_native {

enum class DescriptorEncoding { float16, float32, none };

struct StoreConfig {
  DescriptorEncoding descriptor_encoding{DescriptorEncoding::float16};
  std::size_t transaction_frames{16};
  std::size_t max_queued_frames{64};
};

class FeatureStore {
 public:
  FeatureStore(const std::filesystem::path& path, StoreConfig config);
  ~FeatureStore();

  FeatureStore(const FeatureStore&) = delete;
  FeatureStore& operator=(const FeatureStore&) = delete;

  void set_session(const std::filesystem::path& source_video,
                   const std::filesystem::path& engine_path,
                   int input_width,
                   int input_height,
                   int max_keypoints,
                   float threshold,
                   const TrackerConfig& tracker,
                   bool optical_flow = false);
  [[nodiscard]] std::vector<LandmarkState> load_active_landmarks(
      std::uint32_t max_inactive_frames) const;
  [[nodiscard]] std::uint64_t next_landmark_id() const;
  [[nodiscard]] TrackLengthHistogram load_length_histogram() const;
  [[nodiscard]] std::vector<FlowTrackPosition> load_latest_positions() const;
  void enqueue(FrameFeatures features);
  void flush();
  [[nodiscard]] std::size_t queued() const;
  [[nodiscard]] std::uint64_t persisted() const;
  [[nodiscard]] std::uint64_t landmark_rows() const;
  [[nodiscard]] std::uint64_t observation_count() const;
  [[nodiscard]] std::int64_t last_frame_index() const;

 private:
  void writer_loop();
  void write_batch(std::vector<FrameFeatures>& batch);
  void throw_writer_error();

  sqlite3* db_{};
  StoreConfig config_;
  mutable std::mutex mutex_;
  std::condition_variable readable_;
  std::condition_variable writable_;
  std::condition_variable drained_;
  std::queue<FrameFeatures> queue_;
  std::thread writer_;
  bool stopping_{false};
  bool writing_{false};
  std::exception_ptr writer_error_;
  std::uint64_t persisted_{};
  std::uint64_t landmark_rows_{};
  std::uint64_t observation_count_{};
  std::int64_t last_frame_index_{-1};
};

}  // namespace slam_native
