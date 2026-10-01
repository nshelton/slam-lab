#pragma once
// The LSD workbench (LSD_SLAM.md): a separate GUI for the direct pipeline.
// Same launcher as the SuperPoint workbench (video only), then the video,
// the keyframe depth and the pose graph.
#include "lsd/odometry.hpp"
#include "slam_native/app.hpp"

#include <filesystem>
#include <string>

namespace slam_native {

struct LsdAppConfig {
  AppConfig base;      // video, engine, odometry.intrinsics / hfov, display_rotation
  std::filesystem::path models_dir;  // depth engines; default: <native_workbench>/models
  std::string depth_model{"Depth Anything V2-S (indoor)"};  // a depth_model_presets() name; empty: off
  int max_width{640};  // the luma image is halved until it is at most this wide
  bool undistort{true};  // remap with calibration.json's "distortion" when present
  lsd::OdometryConfig odometry;
};

int run_lsd_app(const LsdAppConfig& config);

}  // namespace slam_native
