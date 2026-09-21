#pragma once

#include "slam_native/types.hpp"

#include <memory>

namespace slam_native {

class CudaFramePresenter {
 public:
  virtual ~CudaFramePresenter() = default;
  virtual unsigned int present(const GpuFrame& frame) = 0;
};

std::unique_ptr<CudaFramePresenter> make_cuda_gl_presenter();

}  // namespace slam_native

