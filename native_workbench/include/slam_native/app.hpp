#pragma once

#include "slam_native/feature_store.hpp"
#include "slam_native/flow_tracker.hpp"
#include "slam_native/online_tracker.hpp"
#include "slam_native/superpoint.hpp"
#include "slam_native/visual_odometry.hpp"

#include <filesystem>
#include <optional>
#include <string>

namespace slam_native {

struct AppConfig {
  std::filesystem::path video;
  std::filesystem::path engine;
  std::filesystem::path database;
  std::filesystem::path models_dir;  // depth engines; default: <native_workbench>/models
  std::string depth_model{"Depth Anything V2-S (indoor)"};  // a depth_model_presets() name; empty: off
  SuperPointConfig superpoint;
  StoreConfig store;
  TrackerConfig tracker;
  FlowTrackerConfig flow_tracker;
  OpticalFlowConfig optical_flow;
  VisualOdometryConfig odometry;
  std::optional<int> display_rotation;  // clockwise degrees; unset: from the video's metadata
  bool start_immediately{};
};

int run_app(const AppConfig& config);

}  // namespace slam_native
