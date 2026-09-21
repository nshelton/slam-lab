#pragma once

#include "slam_native/feature_store.hpp"
#include "slam_native/superpoint.hpp"

#include <filesystem>

namespace slam_native {

struct AppConfig {
  std::filesystem::path video;
  std::filesystem::path engine;
  std::filesystem::path database{"features.sqlite3"};
  SuperPointConfig superpoint;
  StoreConfig store;
};

int run_app(const AppConfig& config);

}  // namespace slam_native

