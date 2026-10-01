#pragma once
// The process-wide TensorRT logger. TensorRT keeps the first logger it is
// given and warns when another one is passed, so every engine shares this one.
#include <NvInfer.h>

#include <cstdio>

namespace slam_native {

inline nvinfer1::ILogger& tensorrt_logger() {
  class Logger final : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* message) noexcept override {
      if (severity <= Severity::kWARNING) fprintf(stderr, "TensorRT: %s\n", message);
    }
  };
  static Logger logger;
  return logger;
}

}  // namespace slam_native
