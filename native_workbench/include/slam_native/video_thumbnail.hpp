#pragma once
// Small RGB preview images of videos, for the launcher's recent-video grid.
// Decoded with the same NVDEC path as sessions (one frame ~10 % into the
// video), converted on the CPU, rotated upright, and cached as binary PPM.
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace slam_native {

struct Thumbnail {
  int width{};
  int height{};
  std::vector<std::uint8_t> rgb;  // width * height * 3, rows top to bottom
};

// Decodes one frame of `video` and returns it upright, scaled so its longer
// side is at most `max_side` pixels. Throws on decode errors. Safe to call
// from a worker thread (it needs CUDA, not OpenGL).
Thumbnail make_thumbnail(const std::filesystem::path& video, int max_side = 320);

void save_ppm(const std::filesystem::path& path, const Thumbnail& thumbnail);
std::optional<Thumbnail> load_ppm(const std::filesystem::path& path);

}  // namespace slam_native
