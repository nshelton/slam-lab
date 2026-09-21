#pragma once

#include "slam_native/types.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace slam_native {

class VideoDecoder {
 public:
  virtual ~VideoDecoder() = default;
  virtual void open(const std::filesystem::path& path) = 0;
  virtual bool next(GpuFrame& frame) = 0;
  [[nodiscard]] virtual double nominal_fps() const = 0;
  [[nodiscard]] virtual std::string codec_name() const = 0;
};

std::unique_ptr<VideoDecoder> make_ffmpeg_cuda_decoder();

}  // namespace slam_native
