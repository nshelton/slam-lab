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
         "  --tracker TYPE            superpoint (default) or optical-flow\n"
         "  --input-width N           TensorRT input width (default: 1024)\n"
         "  --input-height N          TensorRT input height (default: 576)\n"
         "  --max-keypoints N         fixed engine output count (default: 2048)\n"
         "  --threshold F             minimum retained score (default: 0.0005)\n"
         "  --track-similarity F      minimum cosine similarity (default: 0.82)\n"
         "  --track-margin F          best-versus-second margin (default: 0.02)\n"
         "  --track-inactive N        frames before a landmark expires (default: 15)\n"
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
      } else if (option == "--tracker") {
        const std::string_view method = value();
        if (method == "superpoint") config.tracking_method = slam_native::TrackingMethod::superpoint;
        else if (method == "optical-flow") config.tracking_method = slam_native::TrackingMethod::optical_flow;
        else throw std::invalid_argument("--tracker must be superpoint or optical-flow");
      } else if (option == "--input-width") {
        config.superpoint.input_width = integer(value(), "--input-width");
      } else if (option == "--input-height") {
        config.superpoint.input_height = integer(value(), "--input-height");
      } else if (option == "--max-keypoints") {
        config.superpoint.max_keypoints = integer(value(), "--max-keypoints");
      } else if (option == "--threshold") {
        config.superpoint.detection_threshold = std::stof(value());
      } else if (option == "--track-similarity") {
        config.tracker.min_similarity = std::stof(value());
      } else if (option == "--track-margin") {
        config.tracker.min_margin = std::stof(value());
      } else if (option == "--track-inactive") {
        config.tracker.max_inactive_frames =
            static_cast<std::uint32_t>(integer(value(), "--track-inactive"));
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
    config.start_immediately = has_video && has_database &&
        (config.tracking_method == slam_native::TrackingMethod::optical_flow || has_engine);
    return slam_native::run_app(config);
  } catch (const std::exception& error) {
    std::cerr << "slam-native-workbench: " << error.what() << '\n';
    return 1;
  }
}
