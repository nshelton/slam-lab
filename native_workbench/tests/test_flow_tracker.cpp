#include "slam_native/feature_store.hpp"
#include "slam_native/flow_tracker.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  const fs::path root = fs::temp_directory_path() /
      ("slam-flow-tracker-test-" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  const fs::path database = root / "flow.sqlite3";
  try {
    slam_native::StoreConfig settings;
    settings.descriptor_encoding = slam_native::DescriptorEncoding::none;
    slam_native::TrackerConfig tracker_settings;
    slam_native::FlowField flow;
    flow.width = 24;
    flow.height = 24;
    flow.vectors.assign(24 * 24, {2.0F, 1.0F});
    std::uint64_t first_id{};
    {
      slam_native::FeatureStore store(database, settings);
      store.set_session(root / "test.mp4", {}, 0, 0, 0, 0, tracker_settings, true);
      slam_native::FlowTracker tracker({}, {}, 0, {});
      slam_native::FrameFeatures first;
      first.frame_index = 0;
      first.width = 96;
      first.height = 96;
      tracker.associate(first, std::nullopt);
      require(first.keypoints.size() == 16 && first.new_landmarks == 16,
              "Flow tracker did not seed the first frame");
      first_id = first.landmark_ids.front();
      const float first_x = first.keypoints.front().x;
      store.enqueue(std::move(first));
      slam_native::FrameFeatures second;
      second.frame_index = 1;
      second.width = 96;
      second.height = 96;
      tracker.associate(second, flow);
      require(second.landmark_ids.front() == first_id &&
              second.keypoints.front().x == first_x + 2.0F &&
              second.matched_landmarks == 16 && second.descriptors.empty(),
              "Flow did not advance the existing track independently of descriptors");
      store.enqueue(std::move(second));
      store.flush();
    }
    {
      slam_native::FeatureStore store(database, settings);
      store.set_session(root / "test.mp4", {}, 0, 0, 0, 0, tracker_settings, true);
      require(store.last_frame_index() == 1 && store.persisted() == 2,
              "Flow frames were not persisted");
      const auto positions = store.load_latest_positions();
      require(positions.size() >= 16 && positions.front().id == first_id,
              "Persisted flow positions were not restored");
      slam_native::FlowTracker resumed(store.load_active_landmarks(1), positions,
                                       store.next_landmark_id(),
                                       store.load_length_histogram());
      slam_native::FrameFeatures third;
      third.frame_index = 2;
      third.width = 96;
      third.height = 96;
      resumed.associate(third, flow);
      require(third.landmark_ids.front() == first_id &&
              third.keypoints.front().x == positions.front().x + 2.0F,
              "Resumed flow track did not retain its identity and position");
    }
    {
      // A camera move can crowd the full track budget into a small part of the
      // image. Vacant cells must still receive new points.
      std::vector<slam_native::LandmarkState> crowded(2000);
      std::vector<slam_native::FlowTrackPosition> positions;
      positions.reserve(crowded.size());
      for (std::size_t index = 0; index < crowded.size(); ++index) {
        crowded[index].id = index;
        crowded[index].observation_count = 1;
        crowded[index].last_frame = 0;
        positions.push_back({index, 12.0F, 12.0F});
      }
      slam_native::TrackLengthHistogram histogram{};
      histogram[0] = crowded.size();
      slam_native::FlowTracker tracker(std::move(crowded), std::move(positions),
                                       2000, histogram);
      slam_native::FlowField still;
      still.width = 24;
      still.height = 24;
      still.vectors.resize(24 * 24);
      slam_native::FrameFeatures frame;
      frame.frame_index = 1;
      frame.width = 96;
      frame.height = 96;
      tracker.associate(frame, still);
      require(frame.matched_landmarks == 1 && frame.new_landmarks == 15 &&
              frame.keypoints.size() == 16 && frame.landmark_ids.front() == 0,
              "Crowded tracks blocked reseeding of uncovered image cells");
    }
    fs::remove_all(root);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
