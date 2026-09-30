#pragma once

#include "slam_native/types.hpp"

#include <memory>

namespace slam_native {

// Requires a current OpenGL context; fails before opening videos or caches.
void validate_cuda_gl_display();

class CudaFramePresenter {
 public:
  virtual ~CudaFramePresenter() = default;
  virtual unsigned int present(const GpuFrame& frame) = 0;
};

std::unique_ptr<CudaFramePresenter> make_cuda_gl_presenter();

}  // namespace slam_native
