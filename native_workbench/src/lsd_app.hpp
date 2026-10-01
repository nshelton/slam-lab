#pragma once
// The LSD workbench (LSD_SLAM.md): a separate GUI for the direct pipeline.
// Same launcher as the SuperPoint workbench (video only), then the video,
// the keyframe depth and the pose graph.
#include "lsd/odometry.hpp"
#include "slam_native/app.hpp"

namespace slam_native {

struct LsdAppConfig {
  AppConfig base;      // video, models_dir, depth_model, odometry.intrinsics / hfov, display_rotation
  int max_width{640};  // the luma image is halved until it is at most this wide
  bool undistort{true};  // remap with calibration.json's "distortion" when present
  lsd::OdometryConfig odometry;
};

int run_lsd_app(const LsdAppConfig& config);

}  // namespace slam_native
