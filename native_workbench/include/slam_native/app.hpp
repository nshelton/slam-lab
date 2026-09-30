#pragma once

#include "slam_native/feature_store.hpp"
#include "slam_native/flow_tracker.hpp"
#include "slam_native/online_tracker.hpp"
#include "slam_native/superpoint.hpp"
#include "slam_native/visual_odometry.hpp"

#include <filesystem>

namespace slam_native {

struct AppConfig {
  std::filesystem::path video;
  std::filesystem::path engine;
  std::filesystem::path database;
  SuperPointConfig superpoint;
  StoreConfig store;
  TrackerConfig tracker;
  FlowTrackerConfig flow_tracker;
  OpticalFlowConfig optical_flow;
  VisualOdometryConfig odometry;
  bool start_immediately{};
};

int run_app(const AppConfig& config);

}  // namespace slam_native
