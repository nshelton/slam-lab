#define GL_GLEXT_PROTOTYPES
#include "slam_native/cuda_frame_presenter.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace slam_native {
namespace {

void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

__device__ unsigned char clamp_byte(float value) {
  return static_cast<unsigned char>(fminf(255.0F, fmaxf(0.0F, value)));
}

__device__ uchar4 convert_yuv(float y, float u, float v) {
  // BT.709 limited-range conversion. Color metadata handling can select the
  // exact matrix later; this path does not change the inference input.
  const float yy = 1.164383F * (y - 16.0F);
  const float uu = u - 128.0F;
  const float vv = v - 128.0F;
  return make_uchar4(clamp_byte(yy + 1.792741F * vv),
                     clamp_byte(yy - 0.213249F * uu - 0.532909F * vv),
                     clamp_byte(yy + 2.112402F * uu), 255);
}

__global__ void nv12_to_rgba(const std::uint8_t* luma,
                             const std::uint8_t* chroma,
                             std::size_t luma_pitch,
                             std::size_t chroma_pitch,
                             uchar4* output,
                             int width,
                             int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const auto* uv = chroma + (y / 2) * chroma_pitch + (x / 2) * 2;
  output[y * width + x] = convert_yuv(luma[y * luma_pitch + x], uv[0], uv[1]);
}

__global__ void p010_to_rgba(const std::uint16_t* luma,
                             const std::uint16_t* chroma,
                             std::size_t luma_pitch,
                             std::size_t chroma_pitch,
                             uchar4* output,
                             int width,
                             int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const auto* y_row = reinterpret_cast<const std::uint16_t*>(
      reinterpret_cast<const std::uint8_t*>(luma) + y * luma_pitch);
  const auto* uv_row = reinterpret_cast<const std::uint16_t*>(
      reinterpret_cast<const std::uint8_t*>(chroma) + (y / 2) * chroma_pitch);
  const float yy = static_cast<float>(y_row[x] >> 8U);
  const float uu = static_cast<float>(uv_row[(x / 2) * 2] >> 8U);
  const float vv = static_cast<float>(uv_row[(x / 2) * 2 + 1] >> 8U);
  output[y * width + x] = convert_yuv(yy, uu, vv);
}

class CudaGlPresenter final : public CudaFramePresenter {
 public:
  ~CudaGlPresenter() override { release(); }

  unsigned int present(const GpuFrame& frame) override {
    ensure_size(frame.width, frame.height);
    cuda_check(cudaGraphicsMapResources(1, &cuda_pbo_), "map display PBO");
    void* output = nullptr;
    std::size_t bytes = 0;
    cuda_check(cudaGraphicsResourceGetMappedPointer(&output, &bytes, cuda_pbo_),
               "get display PBO pointer");
    const dim3 block(16, 16);
    const dim3 grid((frame.width + block.x - 1) / block.x,
                    (frame.height + block.y - 1) / block.y);
    if (frame.format == PixelFormat::nv12) {
      nv12_to_rgba<<<grid, block>>>(reinterpret_cast<const std::uint8_t*>(frame.luma),
                                   reinterpret_cast<const std::uint8_t*>(frame.chroma),
                                   frame.luma_pitch, frame.chroma_pitch,
                                   static_cast<uchar4*>(output), frame.width, frame.height);
    } else {
      p010_to_rgba<<<grid, block>>>(reinterpret_cast<const std::uint16_t*>(frame.luma),
                                   reinterpret_cast<const std::uint16_t*>(frame.chroma),
                                   frame.luma_pitch, frame.chroma_pitch,
                                   static_cast<uchar4*>(output), frame.width, frame.height);
    }
    cuda_check(cudaGetLastError(), "convert decoded frame for display");
    cuda_check(cudaGraphicsUnmapResources(1, &cuda_pbo_), "unmap display PBO");

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, frame.width, frame.height, GL_RGBA,
                    GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    return texture_;
  }

 private:
  void ensure_size(int width, int height) {
    if (width == width_ && height == height_) return;
    release();
    width_ = width;
    height_ = height;
    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 nullptr);
    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 static_cast<GLsizeiptr>(width) * height * sizeof(uchar4), nullptr,
                 GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cuda_check(cudaGraphicsGLRegisterBuffer(&cuda_pbo_, pbo_, cudaGraphicsRegisterFlagsWriteDiscard),
               "register OpenGL display PBO with CUDA");
  }

  void release() {
    if (cuda_pbo_) {
      cudaGraphicsUnregisterResource(cuda_pbo_);
      cuda_pbo_ = nullptr;
    }
    if (pbo_) {
      glDeleteBuffers(1, &pbo_);
      pbo_ = 0;
    }
    if (texture_) {
      glDeleteTextures(1, &texture_);
      texture_ = 0;
    }
    width_ = 0;
    height_ = 0;
  }

  GLuint texture_{};
  GLuint pbo_{};
  cudaGraphicsResource* cuda_pbo_{};
  int width_{};
  int height_{};
};

}  // namespace

std::unique_ptr<CudaFramePresenter> make_cuda_gl_presenter() {
  return std::make_unique<CudaGlPresenter>();
}

}  // namespace slam_native

