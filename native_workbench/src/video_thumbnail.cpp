#include "slam_native/video_thumbnail.hpp"

#include "slam_native/video_decoder.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace slam_native {
namespace {

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) throw std::runtime_error(std::string("thumbnail: ") + what + ": " + cudaGetErrorString(status));
}

std::uint8_t clamp_byte(double value) { return static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L)); }

}  // namespace

Thumbnail make_thumbnail(const std::filesystem::path& video, int max_side) {
  auto decoder = make_ffmpeg_cuda_decoder();
  decoder->open(video);
  if (decoder->duration_ns() > 0) decoder->seek_ns(decoder->start_time_ns() + decoder->duration_ns() / 10);
  GpuFrame frame;
  if (!decoder->next(frame)) {  // very short video (or a failed seek): take the first frame
    decoder = make_ffmpeg_cuda_decoder();
    decoder->open(video);
    if (!decoder->next(frame)) throw std::runtime_error("thumbnail: no frames in " + video.string());
  }
  if (frame.ready_event) check(cudaEventSynchronize(static_cast<cudaEvent_t>(frame.ready_event)), "wait for frame");

  // Copy NV12 / P010 planes to the host (P010: 16-bit samples, data in the high byte).
  const int bytes = frame.format == PixelFormat::p010 ? 2 : 1;
  const std::size_t row = static_cast<std::size_t>(frame.width) * bytes;
  std::vector<std::uint8_t> luma(row * frame.height), chroma(row * (frame.height / 2));
  check(cudaMemcpy2D(luma.data(), row, reinterpret_cast<const void*>(frame.luma), frame.luma_pitch, row,
                     frame.height, cudaMemcpyDeviceToHost), "copy luma");
  check(cudaMemcpy2D(chroma.data(), row, reinterpret_cast<const void*>(frame.chroma), frame.chroma_pitch, row,
                     frame.height / 2, cudaMemcpyDeviceToHost), "copy chroma");
  const auto sample = [&](const std::vector<std::uint8_t>& plane, int x, int y) -> double {
    const std::size_t index = static_cast<std::size_t>(y) * row + static_cast<std::size_t>(x) * bytes;
    return bytes == 2 ? plane[index + 1] + plane[index] / 256.0 : plane[index];
  };

  // Box-filtered downscale of the stored (unrotated) frame, BT.709 limited range.
  const double factor = std::max(1.0, static_cast<double>(std::max(frame.width, frame.height)) / max_side);
  const int w = std::max(1, static_cast<int>(frame.width / factor)), h = std::max(1, static_cast<int>(frame.height / factor));
  std::vector<std::uint8_t> stored(static_cast<std::size_t>(w) * h * 3);
  for (int ty = 0; ty < h; ++ty) {
    for (int tx = 0; tx < w; ++tx) {
      const int x0 = static_cast<int>(tx * factor), x1 = std::max(x0 + 1, static_cast<int>((tx + 1) * factor));
      const int y0 = static_cast<int>(ty * factor), y1 = std::max(y0 + 1, static_cast<int>((ty + 1) * factor));
      double r = 0, g = 0, b = 0;
      int count = 0;
      for (int y = y0; y < std::min(y1, frame.height); y += 2) {      // every 2nd source pixel is plenty
        for (int x = x0; x < std::min(x1, frame.width); x += 2) {
          const double Y = 1.164 * (sample(luma, x, y) - 16);
          const int cx = (x / 2) * 2;
          const double U = sample(chroma, cx, y / 2) - 128, V = sample(chroma, cx + 1, y / 2) - 128;
          r += Y + 1.793 * V;
          g += Y - 0.213 * U - 0.533 * V;
          b += Y + 2.112 * U;
          ++count;
        }
      }
      auto* out = &stored[(static_cast<std::size_t>(ty) * w + tx) * 3];
      out[0] = clamp_byte(r / count);
      out[1] = clamp_byte(g / count);
      out[2] = clamp_byte(b / count);
    }
  }

  // Rotate clockwise by the container's display rotation.
  const int rotation = decoder->display_rotation();
  Thumbnail result;
  result.width = rotation == 90 || rotation == 270 ? h : w;
  result.height = rotation == 90 || rotation == 270 ? w : h;
  result.rgb.resize(stored.size());
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      int u = x, v = y;  // destination pixel
      if (rotation == 90) { u = h - 1 - y; v = x; }
      else if (rotation == 180) { u = w - 1 - x; v = h - 1 - y; }
      else if (rotation == 270) { u = y; v = w - 1 - x; }
      std::copy_n(&stored[(static_cast<std::size_t>(y) * w + x) * 3], 3,
                  &result.rgb[(static_cast<std::size_t>(v) * result.width + u) * 3]);
    }
  }
  return result;
}

void save_ppm(const std::filesystem::path& path, const Thumbnail& thumbnail) {
  std::filesystem::create_directories(path.parent_path());
  const auto temporary = path.string() + ".tmp";
  {
    std::ofstream out(temporary, std::ios::binary);
    out << "P6\n" << thumbnail.width << ' ' << thumbnail.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(thumbnail.rgb.data()), static_cast<std::streamsize>(thumbnail.rgb.size()));
    if (!out) throw std::runtime_error("Cannot write thumbnail " + path.string());
  }
  std::filesystem::rename(temporary, path);  // readers never see a partial file
}

std::optional<Thumbnail> load_ppm(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::string magic;
  int max_value = 0;
  Thumbnail thumbnail;
  if (!(in >> magic >> thumbnail.width >> thumbnail.height >> max_value) || magic != "P6" || max_value != 255 ||
      thumbnail.width <= 0 || thumbnail.height <= 0 || thumbnail.width > 4096 || thumbnail.height > 4096)
    return std::nullopt;
  in.get();  // the single whitespace after the header
  thumbnail.rgb.resize(static_cast<std::size_t>(thumbnail.width) * thumbnail.height * 3);
  if (!in.read(reinterpret_cast<char*>(thumbnail.rgb.data()), static_cast<std::streamsize>(thumbnail.rgb.size())))
    return std::nullopt;
  return thumbnail;
}

}  // namespace slam_native
