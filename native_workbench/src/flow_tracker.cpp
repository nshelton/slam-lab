#include "slam_native/flow_tracker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace slam_native {

FlowVector FlowField::sample(float x, float y) const {
  if (width <= 0 || height <= 0 || grid_size <= 0 ||
      vectors.size() != static_cast<std::size_t>(width) * height) return {};
  const int column = std::clamp(static_cast<int>(x / grid_size), 0, width - 1);
  const int row = std::clamp(static_cast<int>(y / grid_size), 0, height - 1);
  return vectors[static_cast<std::size_t>(row) * width + column];
}

FlowTracker::FlowTracker(std::vector<LandmarkState> active,
                         std::vector<FlowTrackPosition> positions,
                         std::uint64_t next_id, TrackLengthHistogram histogram)
    : next_id_(next_id), histogram_(histogram) {
  std::unordered_map<std::uint64_t, FlowTrackPosition> last_positions;
  for (const auto& position : positions) last_positions[position.id] = position;
  for (auto& landmark : active) {
    if (const auto found = last_positions.find(landmark.id); found != last_positions.end()) {
      active_.push_back({std::move(landmark), found->second.x, found->second.y});
    }
  }
}

void FlowTracker::associate(FrameFeatures& frame, const std::optional<FlowField>& flow) {
  const auto started = std::chrono::steady_clock::now();
  constexpr std::size_t target_tracks = 1600;
  constexpr std::size_t max_tracks = 2000;
  if (frame.width <= 0 || frame.height <= 0) throw std::invalid_argument("Invalid flow frame size");
  int spacing = std::max(24, static_cast<int>(std::ceil(std::sqrt(
      static_cast<double>(frame.width) * frame.height / target_tracks))));
  auto cell_columns = [&] { return (frame.width + spacing - 1) / spacing; };
  auto cell_rows = [&] { return (frame.height + spacing - 1) / spacing; };
  while (static_cast<std::size_t>(cell_columns()) * cell_rows() > max_tracks) ++spacing;
  if (flow) {
    for (auto& track : active_) {
      const FlowVector vector = flow->sample(track.x, track.y);
      track.x += vector.dx;
      track.y += vector.dy;
      track.landmark.last_frame = frame.frame_index;
    }
    active_.erase(std::remove_if(active_.begin(), active_.end(), [&](const Track& track) {
                    return !std::isfinite(track.x) || !std::isfinite(track.y) ||
                           track.x < 0 || track.y < 0 ||
                           track.x >= frame.width || track.y >= frame.height;
                  }), active_.end());
  } else if (frame.frame_index != 0) {
    // A resumed session starts from its last saved positions. The next decoded
    // frame must supply flow before those positions can advance.
    active_.clear();
  }
  frame.keypoints.clear();
  frame.landmark_ids.clear();
  frame.landmark_similarities.clear();
  frame.landmark_updates.clear();
  frame.descriptors.clear();
  frame.new_landmarks = 0;
  const int cells_x = cell_columns();
  const int cells_y = cell_rows();
  const auto cell_index = [&](const Track& track) {
    const int column = std::clamp(static_cast<int>(track.x) / spacing, 0, cells_x - 1);
    const int row = std::clamp(static_cast<int>(track.y) / spacing, 0, cells_y - 1);
    return static_cast<std::size_t>(row) * cells_x + column;
  };
  // Flow can bring many tracks into the same region. Reserve one long-lived
  // track per cell so the point budget can cover newly visible image areas.
  std::vector<int> winner(static_cast<std::size_t>(cells_x) * cells_y, -1);
  for (std::size_t index = 0; index < active_.size(); ++index) {
    int& previous = winner[cell_index(active_[index])];
    if (previous < 0 ||
        active_[index].landmark.observation_count >
            active_[static_cast<std::size_t>(previous)].landmark.observation_count ||
        (active_[index].landmark.observation_count ==
             active_[static_cast<std::size_t>(previous)].landmark.observation_count &&
         active_[index].landmark.id <
             active_[static_cast<std::size_t>(previous)].landmark.id)) {
      previous = static_cast<int>(index);
    }
  }
  std::vector<bool> occupied(static_cast<std::size_t>(cells_x) * cells_y);
  std::vector<Track> kept;
  kept.reserve(std::min(active_.size(), occupied.size()));
  for (std::size_t index = 0; index < active_.size(); ++index) {
    const std::size_t cell = cell_index(active_[index]);
    if (winner[cell] != static_cast<int>(index)) continue;
    occupied[cell] = true;
    kept.push_back(std::move(active_[index]));
  }
  active_ = std::move(kept);
  frame.matched_landmarks = static_cast<std::uint32_t>(active_.size());
  for (int row = 0; row < cells_y; ++row) {
    for (int column = 0; column < cells_x; ++column) {
      if (active_.size() >= max_tracks) break;
      if (occupied[static_cast<std::size_t>(row) * cells_x + column]) continue;
      const float x = (column * spacing +
                       std::min((column + 1) * spacing, frame.width)) * 0.5F;
      const float y = (row * spacing +
                       std::min((row + 1) * spacing, frame.height)) * 0.5F;
      LandmarkState landmark;
      landmark.id = next_id_++;
      landmark.observation_count = 0;
      landmark.first_frame = frame.frame_index;
      landmark.last_frame = frame.frame_index;
      active_.push_back({landmark, x, y});
      ++frame.new_landmarks;
    }
  }
  frame.keypoints.reserve(active_.size());
  frame.landmark_ids.reserve(active_.size());
  frame.landmark_similarities.reserve(active_.size());
  frame.landmark_updates.reserve(active_.size());
  for (auto& track : active_) {
    if (track.landmark.observation_count) {
      --histogram_[track_length_bin(track.landmark.observation_count)];
    }
    ++track.landmark.observation_count;
    ++histogram_[track_length_bin(track.landmark.observation_count)];
    frame.keypoints.push_back({track.x, track.y, 1.0F});
    frame.landmark_ids.push_back(track.landmark.id);
    frame.landmark_similarities.push_back(std::numeric_limits<float>::quiet_NaN());
    frame.landmark_updates.push_back(track.landmark);
  }
  frame.tracking_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
}

const LandmarkState* FlowTracker::find(std::uint64_t id) const {
  for (const auto& track : active_) {
    if (track.landmark.id == id) return &track.landmark;
  }
  return nullptr;
}

}  // namespace slam_native
