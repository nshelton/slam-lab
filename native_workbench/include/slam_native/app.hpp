#pragma once

#include "slam_native/feature_store.hpp"
#include "slam_native/flow_tracker.hpp"
#include "slam_native/superpoint.hpp"
#include "slam_native/visual_odometry.hpp"

#include <filesystem>
#include <optional>

namespace slam_native {

struct AppConfig {
  std::filesystem::path video;
  std::filesystem::path engine;
  std::filesystem::path database;
  SuperPointConfig superpoint;
  StoreConfig store;
  FlowTrackerConfig flow_tracker;
  OpticalFlowConfig optical_flow;
  VisualOdometryConfig odometry;
  std::optional<int> display_rotation;  // clockwise degrees; unset: from the video's metadata
  bool start_immediately{};
};

int run_app(const AppConfig& config);

}  // namespace slam_native
