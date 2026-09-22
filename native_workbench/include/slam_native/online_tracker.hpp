#pragma once

#include "slam_native/types.hpp"

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <vector>

namespace slam_native {

struct TrackerConfig {
  float min_similarity{0.82F};
  float min_margin{0.02F};
  std::uint32_t max_inactive_frames{15};
};

constexpr std::size_t kTrackLengthBins = 24;
using TrackLengthHistogram = std::array<std::uint64_t, kTrackLengthBins>;

constexpr std::size_t track_length_bin(std::uint64_t observations) {
  if (observations <= 16) return observations ? observations - 1 : 0;
  std::size_t bin = 16;
  std::uint64_t upper = 32;
  while (observations > upper && bin + 1 < kTrackLengthBins) {
    upper *= 2;
    ++bin;
  }
  return bin;
}

class OnlineTracker {
 public:
  OnlineTracker(TrackerConfig config, std::vector<LandmarkState> active,
                std::uint64_t next_id, TrackLengthHistogram histogram);
  ~OnlineTracker();
  OnlineTracker(const OnlineTracker&) = delete;
  OnlineTracker& operator=(const OnlineTracker&) = delete;

  void associate(FrameFeatures& frame);
  [[nodiscard]] std::size_t active_count() const;
  [[nodiscard]] std::uint64_t landmark_count() const;
  [[nodiscard]] const LandmarkState* find(std::uint64_t id) const;
  [[nodiscard]] const TrackLengthHistogram& length_histogram() const;

 private:
  struct Gpu;
  TrackerConfig config_;
  std::vector<LandmarkState> active_;
  std::uint64_t next_id_{};
  TrackLengthHistogram histogram_{};
  std::unique_ptr<Gpu> gpu_;
};

}  // namespace slam_native
