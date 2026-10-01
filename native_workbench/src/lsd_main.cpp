#include "lsd_app.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
void usage() {
  std::cout << "usage: slam-native-lsd-workbench [options]\n"
               "\n"
               "Direct monocular SLAM after LSD-SLAM (native_workbench/LSD_SLAM.md).\n"
               "Launch without options to choose a video in the GUI; --video starts directly.\n"
               "\n"
               "options:\n"
               "  --video PATH              source video\n"
               "  --depth-model NAME        keyframe depth prior: 'Depth Anything V2-S (indoor)' (default),\n"
               "                            'Depth Anything V2-S (outdoor)' or 'Metric3D v2 ViT-S'\n"
               "  --max-width N             working image width limit; the luma is halved until it fits (default: 640)\n"
               "  --hfov DEG                horizontal field of view without calibration.json (default: 60)\n"
               "  --intrinsics FX FY CX CY  camera intrinsics in source-video pixels\n"
               "  --rotate DEG              display rotation, clockwise (default: from video metadata)\n"
               "  --no-undistort            ignore calibration.json's lens distortion\n";
}
}  // namespace

int main(int argc, char** argv) {
  slam_native::LsdAppConfig config;
  try {
    for (int index = 1; index < argc; ++index) {
      const std::string_view option = argv[index];
      const auto value = [&]() -> const char* {
        if (++index >= argc) throw std::invalid_argument("Missing value for " + std::string(option));
        return argv[index];
      };
      if (option == "--video") {
        config.base.video = value();
        config.base.start_immediately = true;
      } else if (option == "--depth-model") {
        config.base.depth_model = value();
      } else if (option == "--max-width") {
        config.max_width = std::stoi(value());
        if (config.max_width < 32) throw std::invalid_argument("--max-width must be at least 32");
      } else if (option == "--hfov") {
        config.base.odometry.horizontal_fov_degrees = std::stod(value());
      } else if (option == "--intrinsics") {
        slam_native::CameraIntrinsics k;
        k.fx = std::stod(value());
        k.fy = std::stod(value());
        k.cx = std::stod(value());
        k.cy = std::stod(value());
        config.base.odometry.intrinsics = k;
      } else if (option == "--rotate") {
        const int degrees = std::stoi(value());
        if (degrees % 90 != 0) throw std::invalid_argument("--rotate expects 0, 90, 180 or 270");
        config.base.display_rotation = ((degrees / 90) % 4 + 4) % 4 * 90;
      } else if (option == "--no-undistort") {
        config.undistort = false;
      } else if (option == "--help" || option == "-h") {
        usage();
        return 0;
      } else {
        throw std::invalid_argument("Unknown option: " + std::string(option));
      }
    }
    return slam_native::run_lsd_app(config);
  } catch (const std::exception& error) {
    std::cerr << "slam-native-lsd-workbench: " << error.what() << '\n';
    return 1;
  }
}
