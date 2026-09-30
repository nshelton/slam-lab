// GPU optical-flow checks: known synthetic shift (direction, S10.5 units, grid
// mapping) and repeatability on a real decoded frame pair when a video is given.
// usage: slam-native-optical-flow-test [VIDEO]
#include "slam_native/optical_flow.hpp"
#include "slam_native/video_decoder.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace slam_native;
namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct DeviceImage {
  std::uint8_t* data{};
  std::size_t pitch{};
  int width{}, height{};
  DeviceImage(int w, int h) : width(w), height(h) {
    check(cudaMallocPitch(reinterpret_cast<void**>(&data), &pitch, w, h));
  }
  ~DeviceImage() { cudaFree(data); }
  void upload(const std::vector<std::uint8_t>& host) {
    check(cudaMemcpy2D(data, pitch, host.data(), width, width, height, cudaMemcpyHostToDevice));
  }
  GpuFrame frame(std::uint64_t index) const {
    GpuFrame f;
    f.frame_index = index;
    f.timestamp_ns = static_cast<std::int64_t>(index) * 16'666'667;
    f.width = width;
    f.height = height;
    f.format = PixelFormat::nv12;
    f.luma = reinterpret_cast<std::uintptr_t>(data);
    f.luma_pitch = pitch;
    return f;
  }
};

// Smooth random texture; `shift` moves content right/down by (sx, sy) pixels.
std::vector<std::uint8_t> texture(int w, int h, float sx, float sy) {
  std::mt19937 rng(7);
  const int n = 48;
  std::vector<float> ax(n), ay(n), fx(n), fy(n), ph(n);
  std::uniform_real_distribution<float> u(0, 1);
  for (int i = 0; i < n; ++i) {
    fx[i] = 0.02F + 0.25F * u(rng); fy[i] = 0.02F + 0.25F * u(rng);
    ph[i] = 6.283F * u(rng); ax[i] = u(rng);
  }
  std::vector<std::uint8_t> image(static_cast<std::size_t>(w) * h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float v = 0;
      const float X = x - sx, Y = y - sy;
      for (int i = 0; i < n; ++i) v += ax[i] * std::sin(fx[i] * X + ph[i]) * std::cos(fy[i] * Y - ph[i]);
      image[static_cast<std::size_t>(y) * w + x] =
          static_cast<std::uint8_t>(std::clamp(128.0F + 22.0F * v, 16.0F, 235.0F));
    }
  return image;
}

void synthetic_shift() {
  const int w = 640, h = 384;
  const float sx = 5.0F, sy = -3.0F;
  DeviceImage a(w, h), b(w, h);
  a.upload(texture(w, h, 0, 0));
  b.upload(texture(w, h, sx, sy));
  OpticalFlowConfig config;
  config.backward = true;
  OpticalFlow flow(config);
  flow.remember(a.frame(0));
  (void)flow.compute(b.frame(1));
  const auto field = flow.download();
  require(!field.backward.empty(), "Backward flow was requested");
  require(field.grid_size == 4 && field.width == w / 4 && field.height == h / 4, "Unexpected flow grid");
  std::vector<float> dx, dy;
  for (int gy = 8; gy < field.height - 8; ++gy)
    for (int gx = 8; gx < field.width - 8; ++gx) {
      const auto v = field.vectors[static_cast<std::size_t>(gy) * field.width + gx];
      dx.push_back(v.dx);
      dy.push_back(v.dy);
    }
  std::nth_element(dx.begin(), dx.begin() + dx.size() / 2, dx.end());
  std::nth_element(dy.begin(), dy.begin() + dy.size() / 2, dy.end());
  const float mx = dx[dx.size() / 2], my = dy[dy.size() / 2];
  std::cout << "synthetic shift (" << sx << ", " << sy << ") -> forward median (" << mx << ", " << my << ")\n";
  require(std::abs(mx - sx) < 0.5F && std::abs(my - sy) < 0.5F,
          "Forward flow must map previous-frame pixels to their current-frame location");
  // Sample through the public API at a keypoint location as well.
  const auto v = field.sample(320.3F, 190.9F);
  require(std::abs(v.dx - sx) < 0.75F && std::abs(v.dy - sy) < 0.75F, "Sampled flow disagrees");
  // Backward field, when computed, must point the other way.
  if (!field.backward.empty()) {
    const auto back = field.sample_backward(320.3F + sx, 190.9F + sy);
    std::cout << "backward sample (" << back.dx << ", " << back.dy << ")\n";
    require(std::abs(back.dx + sx) < 0.75F && std::abs(back.dy + sy) < 0.75F, "Backward flow direction");
  }
}

void repeatability(const char* video) {
  auto decoder = make_ffmpeg_cuda_decoder();
  decoder->open(video);
  decoder->seek_ns(decoder->start_time_ns() + 120'000'000'000LL);
  GpuFrame frame;
  require(decoder->next(frame), "Decode first frame");
  DeviceImage a(frame.width, frame.height), b(frame.width, frame.height);
  require(frame.format == PixelFormat::nv12, "Repeatability check expects 8-bit video");
  check(cudaDeviceSynchronize());
  check(cudaMemcpy2D(a.data, a.pitch, reinterpret_cast<void*>(frame.luma), frame.luma_pitch,
                     frame.width, frame.height, cudaMemcpyDeviceToDevice));
  require(decoder->next(frame), "Decode second frame");
  check(cudaDeviceSynchronize());
  check(cudaMemcpy2D(b.data, b.pitch, reinterpret_cast<void*>(frame.luma), frame.luma_pitch,
                     frame.width, frame.height, cudaMemcpyDeviceToDevice));
  std::vector<FlowVector> first;
  std::size_t worst = 0;
  for (int run = 0; run < 6; ++run) {
    OpticalFlow flow;
    flow.remember(a.frame(0));
    (void)flow.compute(b.frame(1));
    const auto field = flow.download();
    if (run == 0) { first = field.vectors; continue; }
    std::size_t differences = 0;
    for (std::size_t i = 0; i < first.size(); ++i)
      differences += first[i].dx != field.vectors[i].dx || first[i].dy != field.vectors[i].dy;
    worst = std::max(worst, differences);
  }
  std::cout << "real pair: max cells differing across 6 identical runs = " << worst << " of "
            << first.size() << '\n';
  require(worst == 0, "Optical flow is not repeatable on identical input");
}
// The app feeds decoder surfaces straight into the flow engine. Two identical
// decode passes must produce identical fields frame by frame.
std::size_t sequence_differences(const char* video, bool synchronize) {
  std::vector<std::vector<FlowVector>> passes[2];
  for (auto& pass : passes) {
    auto decoder = make_ffmpeg_cuda_decoder();
    decoder->open(video);
    decoder->seek_ns(decoder->start_time_ns() + 120'000'000'000LL);
    OpticalFlow flow;
    GpuFrame frame;
    for (int i = 0; i < 40 && decoder->next(frame); ++i) {
      if (synchronize) check(cudaDeviceSynchronize());
      if (!flow.has_reference()) { flow.remember(frame); continue; }
      (void)flow.compute(frame);
      pass.push_back(flow.download().vectors);
    }
  }
  std::size_t differences = 0;
  for (std::size_t f = 0; f < passes[0].size(); ++f)
    for (std::size_t i = 0; i < passes[0][f].size(); ++i)
      differences += passes[0][f][i].dx != passes[1][f][i].dx || passes[0][f][i].dy != passes[1][f][i].dy;
  return differences;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    // FFmpeg must create the CUDA primary context before any runtime API use.
    if (argc > 1) {
      repeatability(argv[1]);
      const auto unsynchronized = sequence_differences(argv[1], false);
      const auto synchronized = sequence_differences(argv[1], true);
      std::cout << "sequence differences across two passes: unsynchronized=" << unsynchronized
                << " synchronized=" << synchronized << '\n';
      require(unsynchronized == 0, "Streaming optical flow is not repeatable");
    }
    synthetic_shift();
    std::cout << "optical flow checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
