#include "slam_native/superpoint.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace slam_native {
namespace {

void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      fprintf(stderr, "TensorRT: %s\n", message);
    }
  }
};

template <typename T>
struct TrtDelete {
  void operator()(T* value) const { delete value; }
};

float event_ms(cudaEvent_t start, cudaEvent_t end) {
  float elapsed = 0.0F;
  cuda_check(cudaEventElapsedTime(&elapsed, start, end), "measure CUDA event interval");
  return elapsed;
}

__device__ float normalized_luma(std::uint8_t value) {
  return fminf(1.0F, fmaxf(0.0F, (static_cast<float>(value) - 16.0F) / 219.0F));
}

__device__ float normalized_luma(std::uint16_t value) {
  const float ten_bit = static_cast<float>(value >> 6U);
  return fminf(1.0F, fmaxf(0.0F, (ten_bit - 64.0F) / 876.0F));
}

template <typename Pixel>
__global__ void resize_luma_letterbox(const Pixel* luma,
                                      std::size_t pitch,
                                      int source_width,
                                      int source_height,
                                      float* destination,
                                      int destination_width,
                                      int destination_height,
                                      float scale,
                                      int offset_x,
                                      int offset_y,
                                      int scaled_width,
                                      int scaled_height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= destination_width || y >= destination_height) return;
  if (x < offset_x || y < offset_y || x >= offset_x + scaled_width ||
      y >= offset_y + scaled_height) {
    destination[y * destination_width + x] = 0.0F;
    return;
  }
  const float source_x = (static_cast<float>(x - offset_x) + 0.5F) / scale - 0.5F;
  const float source_y = (static_cast<float>(y - offset_y) + 0.5F) / scale - 0.5F;
  const int x0 = max(0, min(source_width - 1, static_cast<int>(floorf(source_x))));
  const int y0 = max(0, min(source_height - 1, static_cast<int>(floorf(source_y))));
  const int x1 = min(source_width - 1, x0 + 1);
  const int y1 = min(source_height - 1, y0 + 1);
  const float wx = source_x - floorf(source_x);
  const float wy = source_y - floorf(source_y);
  const auto* row0 = reinterpret_cast<const Pixel*>(
      reinterpret_cast<const std::uint8_t*>(luma) + y0 * pitch);
  const auto* row1 = reinterpret_cast<const Pixel*>(
      reinterpret_cast<const std::uint8_t*>(luma) + y1 * pitch);
  const float top = normalized_luma(row0[x0]) * (1.0F - wx) +
                    normalized_luma(row0[x1]) * wx;
  const float bottom = normalized_luma(row1[x0]) * (1.0F - wx) +
                       normalized_luma(row1[x1]) * wx;
  destination[y * destination_width + x] = top * (1.0F - wy) + bottom * wy;
}

class TensorRtSuperPoint final : public SuperPoint {
 public:
  TensorRtSuperPoint(const std::filesystem::path& engine_path, SuperPointConfig config)
      : config_(config) {
    if (config.input_width % 8 || config.input_height % 8) {
      throw std::invalid_argument("SuperPoint input dimensions must be divisible by 8");
    }
    std::ifstream input(engine_path, std::ios::binary | std::ios::ate);
    if (!input) {
      throw std::runtime_error("Cannot open TensorRT engine: " +
                               std::filesystem::absolute(engine_path).string() +
                               " (build it with 'Native: build SuperPoint engine' in VS Code)");
    }
    const auto size = input.tellg();
    input.seekg(0);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    input.read(bytes.data(), size);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("Cannot create TensorRT runtime");
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine_) throw std::runtime_error("Cannot deserialize TensorRT engine");
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Cannot create TensorRT execution context");

    require_tensor("image", nvinfer1::TensorIOMode::kINPUT);
    require_tensor("keypoints", nvinfer1::TensorIOMode::kOUTPUT);
    require_tensor("scores", nvinfer1::TensorIOMode::kOUTPUT);
    require_tensor("descriptors", nvinfer1::TensorIOMode::kOUTPUT);
    const auto image_shape = engine_->getTensorShape("image");
    if (image_shape.nbDims != 4 || image_shape.d[0] != 1 || image_shape.d[1] != 1 ||
        image_shape.d[2] != config_.input_height || image_shape.d[3] != config_.input_width) {
      throw std::runtime_error("TensorRT engine input shape does not match --input-width/height");
    }
    const auto point_shape = engine_->getTensorShape("keypoints");
    if (point_shape.nbDims != 3 || point_shape.d[0] != 1 || point_shape.d[1] != config_.max_keypoints ||
        point_shape.d[2] != 2) {
      throw std::runtime_error("TensorRT keypoint output does not match --max-keypoints");
    }
    const auto score_shape = engine_->getTensorShape("scores");
    const auto descriptor_shape = engine_->getTensorShape("descriptors");
    if (score_shape.nbDims != 2 || score_shape.d[0] != 1 ||
        score_shape.d[1] != config_.max_keypoints ||
        descriptor_shape.nbDims != 3 || descriptor_shape.d[0] != 1 ||
        descriptor_shape.d[1] != config_.max_keypoints || descriptor_shape.d[2] != 256) {
      throw std::runtime_error("TensorRT score or descriptor output has the wrong shape");
    }
    for (const char* name : {"image", "keypoints", "scores", "descriptors"}) {
      if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error(std::string("Tensor must expose float32 I/O: ") + name);
      }
    }

    keypoints_.resize(static_cast<std::size_t>(config_.max_keypoints) * 2);
    scores_.resize(config_.max_keypoints);
    descriptors_.resize(static_cast<std::size_t>(config_.max_keypoints) * 256);
    cuda_check(cudaMalloc(&device_image_, image_bytes()), "allocate SuperPoint input");
    cuda_check(cudaMalloc(&device_keypoints_, keypoints_.size() * sizeof(float)),
               "allocate keypoint output");
    cuda_check(cudaMalloc(&device_scores_, scores_.size() * sizeof(float)),
               "allocate score output");
    cuda_check(cudaMalloc(&device_descriptors_, descriptors_.size() * sizeof(float)),
               "allocate descriptor output");
    cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create inference stream");
    for (cudaEvent_t* event : {&preprocess_begin_, &preprocess_end_, &inference_end_, &readback_end_}) {
      cuda_check(cudaEventCreate(event), "create timing event");
    }
    if (!context_->setTensorAddress("image", device_image_) ||
        !context_->setTensorAddress("keypoints", device_keypoints_) ||
        !context_->setTensorAddress("scores", device_scores_) ||
        !context_->setTensorAddress("descriptors", device_descriptors_)) {
      throw std::runtime_error("Cannot bind TensorRT engine buffers");
    }
  }

  ~TensorRtSuperPoint() override {
    for (cudaEvent_t event : {preprocess_begin_, preprocess_end_, inference_end_, readback_end_}) {
      if (event) cudaEventDestroy(event);
    }
    if (stream_) cudaStreamDestroy(stream_);
    cudaFree(device_image_);
    cudaFree(device_keypoints_);
    cudaFree(device_scores_);
    cudaFree(device_descriptors_);
  }

  FrameFeatures infer(const GpuFrame& frame) override {
    const float scale = std::min(static_cast<float>(config_.input_width) / frame.width,
                                 static_cast<float>(config_.input_height) / frame.height);
    const int scaled_width = std::max(1, static_cast<int>(frame.width * scale + 0.5F));
    const int scaled_height = std::max(1, static_cast<int>(frame.height * scale + 0.5F));
    const int offset_x = (config_.input_width - scaled_width) / 2;
    const int offset_y = (config_.input_height - scaled_height) / 2;
    const dim3 block(16, 16);
    const dim3 grid((config_.input_width + block.x - 1) / block.x,
                    (config_.input_height + block.y - 1) / block.y);

    cuda_check(cudaEventRecord(preprocess_begin_, stream_), "start preprocess timer");
    if (frame.format == PixelFormat::nv12) {
      resize_luma_letterbox<<<grid, block, 0, stream_>>>(
          reinterpret_cast<const std::uint8_t*>(frame.luma), frame.luma_pitch,
          frame.width, frame.height, static_cast<float*>(device_image_), config_.input_width,
          config_.input_height, scale, offset_x, offset_y, scaled_width, scaled_height);
    } else {
      resize_luma_letterbox<<<grid, block, 0, stream_>>>(
          reinterpret_cast<const std::uint16_t*>(frame.luma), frame.luma_pitch,
          frame.width, frame.height, static_cast<float*>(device_image_), config_.input_width,
          config_.input_height, scale, offset_x, offset_y, scaled_width, scaled_height);
    }
    cuda_check(cudaGetLastError(), "preprocess SuperPoint input");
    cuda_check(cudaEventRecord(preprocess_end_, stream_), "end preprocess timer");
    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("TensorRT SuperPoint enqueue failed");
    }
    cuda_check(cudaEventRecord(inference_end_, stream_), "end inference timer");
    cuda_check(cudaMemcpyAsync(keypoints_.data(), device_keypoints_,
                               keypoints_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_),
               "read back keypoints");
    cuda_check(cudaMemcpyAsync(scores_.data(), device_scores_, scores_.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_), "read back scores");
    cuda_check(cudaMemcpyAsync(descriptors_.data(), device_descriptors_,
                               descriptors_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_),
               "read back descriptors");
    cuda_check(cudaEventRecord(readback_end_, stream_), "end readback timer");
    cuda_check(cudaEventSynchronize(readback_end_), "wait for SuperPoint result");

    FrameFeatures result;
    result.frame_index = frame.frame_index;
    result.timestamp_ns = frame.timestamp_ns;
    result.pts = frame.pts;
    result.width = frame.width;
    result.height = frame.height;
    result.preprocess_ms = event_ms(preprocess_begin_, preprocess_end_);
    result.inference_ms = event_ms(preprocess_end_, inference_end_);
    result.readback_ms = event_ms(inference_end_, readback_end_);
    result.keypoints.reserve(config_.max_keypoints);
    result.descriptors.reserve(static_cast<std::size_t>(config_.max_keypoints) * 256);
    for (int index = 0; index < config_.max_keypoints; ++index) {
      const float x = keypoints_[index * 2];
      const float y = keypoints_[index * 2 + 1];
      const float score = scores_[index];
      if (score < config_.detection_threshold || x < offset_x || y < offset_y ||
          x >= offset_x + scaled_width || y >= offset_y + scaled_height) {
        continue;
      }
      result.keypoints.push_back({(x - offset_x) / scale, (y - offset_y) / scale, score});
      const auto first = descriptors_.begin() + static_cast<std::ptrdiff_t>(index) * 256;
      result.descriptors.insert(result.descriptors.end(), first, first + 256);
    }
    return result;
  }

 private:
  void require_tensor(const char* name, nvinfer1::TensorIOMode mode) {
    if (engine_->getTensorIOMode(name) != mode) {
      throw std::runtime_error(std::string("Missing TensorRT tensor: ") + name);
    }
  }

  [[nodiscard]] std::size_t image_bytes() const {
    return static_cast<std::size_t>(config_.input_width) * config_.input_height * sizeof(float);
  }

  SuperPointConfig config_;
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, TrtDelete<nvinfer1::IRuntime>> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDelete<nvinfer1::ICudaEngine>> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDelete<nvinfer1::IExecutionContext>> context_;
  void* device_image_{};
  void* device_keypoints_{};
  void* device_scores_{};
  void* device_descriptors_{};
  cudaStream_t stream_{};
  cudaEvent_t preprocess_begin_{};
  cudaEvent_t preprocess_end_{};
  cudaEvent_t inference_end_{};
  cudaEvent_t readback_end_{};
  std::vector<float> keypoints_;
  std::vector<float> scores_;
  std::vector<float> descriptors_;
};

}  // namespace

std::unique_ptr<SuperPoint> make_tensorrt_superpoint(
    const std::filesystem::path& engine_path, const SuperPointConfig& config) {
  return std::make_unique<TensorRtSuperPoint>(engine_path, config);
}

}  // namespace slam_native
