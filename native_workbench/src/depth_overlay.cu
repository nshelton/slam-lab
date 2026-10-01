#define GL_GLEXT_PROTOTYPES
#include "slam_native/depth_overlay.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <string>

namespace slam_native {
namespace {

void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

__device__ unsigned char byte(float value) {
  return static_cast<unsigned char>(fminf(255.0F, fmaxf(0.0F, 255.0F * value + 0.5F)));
}

// Google's "turbo" ramp (polynomial fit by A. Mikhailov): blue at 0, red at 1.
__device__ uchar4 turbo(float t, unsigned char alpha) {
  t = fminf(fmaxf(t, 0.0F), 1.0F);
  const float t2 = t * t, t3 = t2 * t, t4 = t2 * t2, t5 = t4 * t;
  const float r = 0.13572138F + 4.61539260F * t - 42.66032258F * t2 + 132.13108234F * t3 - 152.94239396F * t4 +
                  59.28637943F * t5;
  const float g = 0.09140261F + 2.19418839F * t + 4.84296658F * t2 - 14.18503333F * t3 + 4.27729857F * t4 +
                  2.82956604F * t5;
  const float b = 0.10667330F + 12.64194608F * t - 60.58204836F * t2 + 110.36276771F * t3 - 89.90310912F * t4 +
                  27.34824973F * t5;
  return make_uchar4(byte(r), byte(g), byte(b), alpha);
}

__global__ void colorize(const float* inverse_depth, const float* variance, int width, int height,
                         DepthOverlayStyle style, uchar4* output) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int index = y * width + x;
  const float rho = inverse_depth[index], var = variance[index];
  if (!(var < INFINITY) || !(rho > 0.0F)) {
    output[index] = make_uchar4(0, 0, 0, 0);
    return;
  }
  const float sigma = sqrtf(var) / rho;
  const float alpha = (style.max_sigma - sigma) / fmaxf(style.max_sigma - style.full_sigma, 1e-6F);
  if (style.color == DepthOverlayColor::sigma) {
    // Certain (blue) to uncertain (red), on a logarithmic scale from 1% to 100%.
    output[index] = turbo(log10f(fmaxf(sigma, 0.01F)) / 2.0F + 1.0F, 255);
    return;
  }
  // Logarithmic in depth, so that near and far both keep contrast.
  const float t = logf(rho / style.far_inverse_depth) /
                  logf(fmaxf(style.near_inverse_depth / style.far_inverse_depth, 1.0001F));
  output[index] = turbo(t, byte(alpha));
}

}  // namespace

struct DepthOverlay::Impl {
  GLuint texture{};
  GLuint pbo{};
  cudaGraphicsResource* resource{};
  int width{}, height{};

  ~Impl() { release(); }
  void release() {
    if (resource) cudaGraphicsUnregisterResource(resource);
    if (pbo) glDeleteBuffers(1, &pbo);
    if (texture) glDeleteTextures(1, &texture);
    resource = nullptr;
    pbo = texture = 0;
    width = height = 0;
  }
  void ensure(int w, int h) {
    if (w == width && h == height) return;
    release();
    GLint previous = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(previous));
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(w) * h * sizeof(uchar4), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cuda_check(cudaGraphicsGLRegisterBuffer(&resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard),
               "register the depth overlay buffer with CUDA");
    width = w;
    height = h;
  }
};

DepthOverlay::DepthOverlay() : impl_(std::make_unique<Impl>()) {}
DepthOverlay::~DepthOverlay() = default;

unsigned int DepthOverlay::present(const LiveDepth& depth, const DepthOverlayStyle& style) {
  const auto image = depth.device();
  if (!image.inverse_depth) return 0;
  auto& m = *impl_;
  m.ensure(image.width, image.height);
  cuda_check(cudaGraphicsMapResources(1, &m.resource), "map the depth overlay buffer");
  void* output = nullptr;
  std::size_t bytes = 0;
  cudaError_t status = cudaGraphicsResourceGetMappedPointer(&output, &bytes, m.resource);
  if (status == cudaSuccess) {
    const dim3 block(16, 16);
    const dim3 grid((image.width + 15) / 16, (image.height + 15) / 16);
    colorize<<<grid, block>>>(image.inverse_depth, image.variance, image.width, image.height, style,
                             static_cast<uchar4*>(output));
    status = cudaGetLastError();
  }
  cuda_check(cudaGraphicsUnmapResources(1, &m.resource), "unmap the depth overlay buffer");
  cuda_check(status, "colour the depth image");
  GLint previous = 0;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m.pbo);
  glBindTexture(GL_TEXTURE_2D, m.texture);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.width, image.height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(previous));
  return m.texture;
}

}  // namespace slam_native
