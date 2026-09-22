#pragma once

#include "slam_native/feature_store.hpp"
#include "slam_native/online_tracker.hpp"
#include "slam_native/superpoint.hpp"

#include <filesystem>

namespace slam_native {

enum class TrackingMethod { superpoint, optical_flow };

struct AppConfig {
  std::filesystem::path video;
  std::filesystem::path engine;
  std::filesystem::path database;
  SuperPointConfig superpoint;
  StoreConfig store;
  TrackerConfig tracker;
  TrackingMethod tracking_method{TrackingMethod::superpoint};
  bool start_immediately{};
};

int run_app(const AppConfig& config);

}  // namespace slam_native
