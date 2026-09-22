#include "slam_native/video_decoder.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    throw std::invalid_argument("Expected a video path and optional seek time in seconds");
  }
  auto decoder = slam_native::make_ffmpeg_cuda_decoder();
  decoder->open(argv[1]);
  slam_native::GpuFrame frame;
  if (!decoder->next(frame)) throw std::runtime_error("No first frame");
  const auto first_time = frame.timestamp_ns;
  auto preview = slam_native::make_ffmpeg_cuda_decoder();
  preview->open(argv[1]);
  const auto target = argc == 3
      ? preview->start_time_ns() + static_cast<std::int64_t>(std::stod(argv[2]) * 1e9)
      : preview->start_time_ns() + preview->duration_ns() / 2;
  preview->seek_ns(target);
  bool reached = false;
  for (int decoded = 0; decoded < 300 && preview->next(frame); ++decoded) {
    if (frame.timestamp_ns >= target) {
      reached = true;
      break;
    }
  }
  if (!reached) throw std::runtime_error("Seek did not reach target time");
  std::cout << "duration_ns=" << preview->duration_ns()
            << " target_ns=" << target
            << " reached_ns=" << frame.timestamp_ns
            << " pts=" << frame.pts << '\n';
  preview->seek_ns(preview->start_time_ns());
  if (!preview->next(frame) || frame.timestamp_ns != first_time) {
    throw std::runtime_error("Seek back to first frame failed");
  }
  if (!decoder->next(frame) || frame.timestamp_ns <= first_time) {
    throw std::runtime_error("Processing decoder moved during preview seek");
  }
}
