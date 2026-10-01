#include "slam_native/color_sampler.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace slam_native {
namespace {
void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

// 8-bit sample of plane `plane` (0 luma, 1 U, 2 V) at pixel (x, y).
__device__ float sample_plane(const GpuFrame& f, int plane, int x, int y) {
  if (plane == 0) {
    const auto* row = reinterpret_cast<const std::uint8_t*>(f.luma) + y * f.luma_pitch;
    return f.format == PixelFormat::nv12 ? row[x] : static_cast<float>(reinterpret_cast<const std::uint16_t*>(row)[x] >> 8U);
  }
  const auto* row = reinterpret_cast<const std::uint8_t*>(f.chroma) + (y / 2) * f.chroma_pitch;
  const int index = (x / 2) * 2 + (plane - 1);
  return f.format == PixelFormat::nv12 ? row[index] :
      static_cast<float>(reinterpret_cast<const std::uint16_t*>(row)[index] >> 8U);
}

__device__ unsigned char clamp_byte(float v) { return static_cast<unsigned char>(fminf(255.0F, fmaxf(0.0F, v + 0.5F))); }

__global__ void sample_colors(GpuFrame frame, const float2* points, int count, uchar3* output) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int cx = static_cast<int>(points[i].x + 0.5F), cy = static_cast<int>(points[i].y + 0.5F);
  float y = 0, u = 0, v = 0;
  int n = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      const int x = min(max(cx + dx, 0), frame.width - 1), yy = min(max(cy + dy, 0), frame.height - 1);
      y += sample_plane(frame, 0, x, yy);
      u += sample_plane(frame, 1, x, yy);
      v += sample_plane(frame, 2, x, yy);
      ++n;
    }
  y = 1.164383F * (y / n - 16.0F);
  u = u / n - 128.0F;
  v = v / n - 128.0F;
  output[i] = make_uchar3(clamp_byte(y + 1.792741F * v), clamp_byte(y - 0.213249F * u - 0.532909F * v),
                          clamp_byte(y + 2.112402F * u));
}
}  // namespace

struct ColorSampler::Device {
  cudaStream_t stream{};
  float2* points{};
  uchar3* colors{};
  int capacity{};
  ~Device() {
    cudaFree(points);
    cudaFree(colors);
    if (stream) cudaStreamDestroy(stream);
  }
};

// CUDA objects are created on first use: constructing them here could create
// the primary context before the video decoder configures it.
ColorSampler::ColorSampler() : device_(std::make_unique<Device>()) {}
ColorSampler::~ColorSampler() = default;

std::vector<std::array<std::uint8_t, 3>> ColorSampler::sample(const GpuFrame& frame,
                                                               const std::vector<Keypoint>& points) {
  std::vector<std::array<std::uint8_t, 3>> result(points.size());
  if (points.empty() || frame.luma == 0) return result;
  auto& d = *device_;
  if (!d.stream) cuda_check(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking), "create colour stream");
  const int count = static_cast<int>(points.size());
  if (count > d.capacity) {
    cudaFree(d.points);
    cudaFree(d.colors);
    d.points = nullptr;
    d.colors = nullptr;
    d.capacity = 0;
    cuda_check(cudaMalloc(&d.points, sizeof(float2) * count), "allocate colour points");
    cuda_check(cudaMalloc(&d.colors, sizeof(uchar3) * count), "allocate colours");
    d.capacity = count;
  }
  std::vector<float2> host(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) host[i] = make_float2(points[i].x, points[i].y);
  if (frame.ready_event)
    cuda_check(cudaStreamWaitEvent(d.stream, static_cast<cudaEvent_t>(frame.ready_event), 0), "wait for frame");
  cuda_check(cudaMemcpyAsync(d.points, host.data(), sizeof(float2) * count, cudaMemcpyHostToDevice, d.stream),
             "upload colour points");
  sample_colors<<<(count + 127) / 128, 128, 0, d.stream>>>(frame, d.points, count, d.colors);
  cuda_check(cudaGetLastError(), "sample colours");
  static_assert(sizeof(uchar3) == 3 && sizeof(std::array<std::uint8_t, 3>) == 3);
  cuda_check(cudaMemcpyAsync(result.data(), d.colors, 3 * points.size(), cudaMemcpyDeviceToHost, d.stream),
             "download colours");
  cuda_check(cudaStreamSynchronize(d.stream), "colour stream");
  return result;
}

}  // namespace slam_native
