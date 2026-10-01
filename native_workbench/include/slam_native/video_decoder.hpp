#pragma once

#include "slam_native/types.hpp"

#include <filesystem>
#include <cstdint>
#include <memory>
#include <string>

namespace slam_native {

class VideoDecoder {
 public:
  virtual ~VideoDecoder() = default;
  virtual void open(const std::filesystem::path& path) = 0;
  virtual bool next(GpuFrame& frame) = 0;
  virtual void seek_ns(std::int64_t timestamp_ns) = 0;
  [[nodiscard]] virtual std::int64_t duration_ns() const = 0;
  [[nodiscard]] virtual std::int64_t start_time_ns() const = 0;
  [[nodiscard]] virtual double nominal_fps() const = 0;
  [[nodiscard]] virtual std::string codec_name() const = 0;
  // Clockwise rotation (0, 90, 180 or 270 degrees) that displays the video
  // upright, from the container's display matrix (phones record portrait
  // video as rotated landscape frames). Decoded frames are not rotated.
  [[nodiscard]] virtual int display_rotation() const { return 0; }
};

std::unique_ptr<VideoDecoder> make_ffmpeg_cuda_decoder();

}  // namespace slam_native
