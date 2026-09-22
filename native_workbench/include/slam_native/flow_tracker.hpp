#pragma once

#include "slam_native/online_tracker.hpp"
#include "slam_native/optical_flow.hpp"

#include <optional>
#include <unordered_map>

namespace slam_native {

struct FlowTrackPosition {
  std::uint64_t id{};
  float x{};
  float y{};
};

class FlowTracker {
 public:
  FlowTracker(std::vector<LandmarkState> active,
              std::vector<FlowTrackPosition> positions,
              std::uint64_t next_id, TrackLengthHistogram histogram);
  void associate(FrameFeatures& frame, const std::optional<FlowField>& flow);
  [[nodiscard]] std::size_t active_count() const { return active_.size(); }
  [[nodiscard]] std::uint64_t landmark_count() const { return next_id_; }
  [[nodiscard]] const LandmarkState* find(std::uint64_t id) const;
  [[nodiscard]] const TrackLengthHistogram& length_histogram() const { return histogram_; }

 private:
  struct Track { LandmarkState landmark; float x{}, y{}; };
  std::vector<Track> active_;
  std::uint64_t next_id_{};
  TrackLengthHistogram histogram_{};
};

}  // namespace slam_native
