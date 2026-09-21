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
      << "usage: slam-native-workbench --video VIDEO --engine ENGINE_PATH [options]\n"
         "\n"
         "options:\n"
         "  --db PATH                 feature database (default: features.sqlite3)\n"
         "  --input-width N           TensorRT input width (default: 1024)\n"
         "  --input-height N          TensorRT input height (default: 576)\n"
         "  --max-keypoints N         fixed engine output count (default: 2048)\n"
         "  --threshold F             minimum retained score (default: 0.0005)\n"
         "  --descriptor-storage TYPE f16 (default) or f32\n";
}

}  // namespace

int main(int argc, char** argv) {
  slam_native::AppConfig config;
  try {
    for (int index = 1; index < argc; ++index) {
      const std::string_view option = argv[index];
      const auto value = [&]() -> const char* {
        if (++index >= argc) throw std::invalid_argument("Missing value for " + std::string(option));
        return argv[index];
      };
      if (option == "--video") {
        config.video = value();
      } else if (option == "--engine") {
        config.engine = value();
      } else if (option == "--db") {
        config.database = value();
      } else if (option == "--input-width") {
        config.superpoint.input_width = integer(value(), "--input-width");
      } else if (option == "--input-height") {
        config.superpoint.input_height = integer(value(), "--input-height");
      } else if (option == "--max-keypoints") {
        config.superpoint.max_keypoints = integer(value(), "--max-keypoints");
      } else if (option == "--threshold") {
        config.superpoint.detection_threshold = std::stof(value());
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
    if (config.video.empty() || config.engine.empty()) {
      usage();
      return 2;
    }
    return slam_native::run_app(config);
  } catch (const std::exception& error) {
    std::cerr << "slam-native-workbench: " << error.what() << '\n';
    return 1;
  }
}
