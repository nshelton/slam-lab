#include "slam_native/app.hpp"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

int integer(const char* value, const char* option) {
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || parsed <= 0) {
    throw std::invalid_argument(std::string("Invalid value for ") + option);
  }
  return static_cast<int>(parsed);
}

void usage() {
  std::cout
      << "usage: slam-native-workbench [options]\n"
         "\n"
         "Launch without options to choose a video and database in the GUI.\n"
         "Supplying video, engine, and database paths starts directly.\n"
         "\n"
         "options:\n"
         "  --video PATH              source video\n"
         "  --engine PATH             TensorRT SuperPoint engine\n"
         "  --db PATH                 feature database\n"
         "  --input-width N           TensorRT input width (default: 1024)\n"
         "  --input-height N          TensorRT input height (default: 576)\n"
         "  --max-keypoints N         fixed engine output count (default: 2048)\n"
         "  --threshold F             minimum retained score (default: 0.0005)\n"
         "  --track-radius F          SuperPoint association radius in pixels (default: 6)\n"
         "  --max-tracks N            live-track cap; continuations first (default: 1000)\n"
         "  --max-coast N             frames a missed track is carried by flow (default: 5; 0 = die)\n"
         "  --min-similarity F        descriptor cosine gate for association (default: 0.7)\n"
         "  --assignment-rounds N     propose/accept rounds; 1 = no second choice (default: 4)\n"
         "  --flow-backward F         compute backward flow; forward-backward limit in px\n"
         "                            for coasting (default: off; costs one more NVOF pass)\n"
         "  --hfov DEG                camera horizontal field of view for pose (default: 60)\n"
         "  --intrinsics FX FY CX CY  camera intrinsics in source-video pixels (overrides --hfov)\n"
         "  --descriptor-storage TYPE f16 (default) or f32\n";
}

}  // namespace

int main(int argc, char** argv) {
  slam_native::AppConfig config;
  bool has_video = false;
  bool has_engine = false;
  bool has_database = false;
  try {
    for (int index = 1; index < argc; ++index) {
      const std::string_view option = argv[index];
      const auto value = [&]() -> const char* {
        if (++index >= argc) throw std::invalid_argument("Missing value for " + std::string(option));
        return argv[index];
      };
      if (option == "--video") {
        config.video = value();
        has_video = true;
      } else if (option == "--engine") {
        config.engine = value();
        has_engine = true;
      } else if (option == "--db") {
        config.database = value();
        has_database = true;
      } else if (option == "--input-width") {
        config.superpoint.input_width = integer(value(), "--input-width");
      } else if (option == "--input-height") {
        config.superpoint.input_height = integer(value(), "--input-height");
      } else if (option == "--max-keypoints") {
        config.superpoint.max_keypoints = integer(value(), "--max-keypoints");
      } else if (option == "--threshold") {
        config.superpoint.detection_threshold = std::stof(value());
      } else if (option == "--track-radius") {
        config.flow_tracker.association_radius = std::stof(value());
      } else if (option == "--max-tracks") {
        config.flow_tracker.max_tracks = integer(value(), "--max-tracks");
      } else if (option == "--max-coast") {
        const std::string text = value();
        std::size_t used = 0;
        const int frames = std::stoi(text, &used);
        if (used != text.size() || frames < 0) throw std::invalid_argument("Invalid value for --max-coast");
        config.flow_tracker.max_coast_frames = frames;
      } else if (option == "--min-similarity") {
        config.flow_tracker.min_descriptor_similarity = std::stof(value());
      } else if (option == "--assignment-rounds") {
        config.flow_tracker.assignment_rounds = integer(value(), "--assignment-rounds");
      } else if (option == "--flow-backward") {
        config.optical_flow.backward = true;
        config.flow_tracker.forward_backward_threshold = std::stof(value());
      } else if (option == "--hfov") {
        config.odometry.horizontal_fov_degrees = std::stod(value());
      } else if (option == "--intrinsics") {
        slam_native::CameraIntrinsics k;
        k.fx = std::stod(value());
        k.fy = std::stod(value());
        k.cx = std::stod(value());
        k.cy = std::stod(value());
        config.odometry.intrinsics = k;
      } else if (option == "--descriptor-storage") {
        const std::string_view encoding = value();
        if (encoding == "f16") {
          config.store.descriptor_encoding = slam_native::DescriptorEncoding::float16;
        } else if (encoding == "f32") {
          config.store.descriptor_encoding = slam_native::DescriptorEncoding::float32;
        } else {
          throw std::invalid_argument("--descriptor-storage must be f16 or f32");
        }
      } else if (option == "--help" || option == "-h") {
        usage();
        return 0;
      } else {
        throw std::invalid_argument("Unknown option: " + std::string(option));
      }
    }
    config.flow_tracker.validate();
    config.odometry.validate();
    config.start_immediately = has_video && has_database && has_engine;
    return slam_native::run_app(config);
  } catch (const std::exception& error) {
    std::cerr << "slam-native-workbench: " << error.what() << '\n';
    return 1;
  }
}
