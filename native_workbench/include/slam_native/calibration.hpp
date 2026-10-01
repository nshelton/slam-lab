#pragma once
// Intrinsics from a calibration.json next to the video (the TUM and KITTI
// conversions write one: "fx", "fy", "cx", "cy" in source-video pixels).
#include "slam_native/visual_odometry.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>

namespace slam_native {

inline std::optional<CameraIntrinsics> read_calibration(const std::filesystem::path& video) {
  std::ifstream in(video.parent_path() / "calibration.json");
  if (!in) return std::nullopt;
  std::stringstream text;
  text << in.rdbuf();
  const std::string json = text.str();
  const auto number = [&](const char* key) -> std::optional<double> {
    std::smatch match;
    if (!std::regex_search(json, match, std::regex(std::string("\"") + key + "\"\\s*:\\s*([-+0-9.eE]+)")))
      return std::nullopt;
    return std::stod(match[1]);
  };
  const auto fx = number("fx"), fy = number("fy"), cx = number("cx"), cy = number("cy");
  if (!fx || !fy || !cx || !cy || *fx <= 0 || *fy <= 0) return std::nullopt;
  return CameraIntrinsics{*fx, *fy, *cx, *cy};
}

// OpenCV radial-tangential coefficients (k1, k2, p1, p2, k3) from the same
// file's "distortion" array; nullopt when absent or all zero.
inline std::optional<std::array<double, 5>> read_distortion(const std::filesystem::path& video) {
  std::ifstream in(video.parent_path() / "calibration.json");
  if (!in) return std::nullopt;
  std::stringstream text;
  text << in.rdbuf();
  const std::string json = text.str();
  std::smatch match;
  if (!std::regex_search(json, match, std::regex("\"distortion\"\\s*:\\s*\\[([^\\]]*)\\]"))) return std::nullopt;
  std::array<double, 5> d{};
  std::stringstream list(match[1].str());
  std::string item;
  bool any = false;
  for (std::size_t i = 0; i < d.size() && std::getline(list, item, ','); ++i) {
    d[i] = std::stod(item);
    any = any || d[i] != 0;
  }
  return any ? std::optional(d) : std::nullopt;
}

}  // namespace slam_native
