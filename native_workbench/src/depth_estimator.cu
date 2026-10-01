#include "slam_native/depth_estimator.hpp"
#include "slam_native/tensorrt_logger.hpp"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace slam_native {

const std::vector<DepthModelSpec>& depth_model_presets() {
  // ImageNet statistics in [0, 255] RGB; padding with the mean normalizes to 0.
  constexpr std::array<float, 3> imagenet_mean{123.675F, 116.28F, 103.53F};
  constexpr std::array<float, 3> imagenet_std{58.395F, 57.12F, 57.375F};
  static const std::vector<DepthModelSpec> presets = [&] {
    DepthModelSpec indoor;
    indoor.name = "Depth Anything V2-S (indoor)";
    indoor.engine_stem = "depth-anything-v2-s-hypersim";
    indoor.input_width = 518;
    indoor.input_height = 294;
    indoor.mean = imagenet_mean;
    indoor.std = imagenet_std;
    indoor.pad = imagenet_mean;
    indoor.max_depth_m = 20;
    DepthModelSpec outdoor = indoor;
    outdoor.name = "Depth Anything V2-S (outdoor)";
    outdoor.engine_stem = "depth-anything-v2-s-vkitti";
    outdoor.max_depth_m = 80;
    // onnx-community/metric3d-vit-small: normalizes inside the graph and
    // predicts depth for a canonical 1000 px focal length.
    DepthModelSpec metric3d;
    metric3d.name = "Metric3D v2 ViT-S";
    metric3d.engine_stem = "metric3d-vit-s";
    metric3d.input_width = 1064;
    metric3d.input_height = 616;
    metric3d.input_tensor = "pixel_values";
    metric3d.output_tensor = "predicted_depth";
    metric3d.pad = imagenet_mean;
    metric3d.canonical_focal_px = 1000;
    metric3d.max_depth_m = 300;
    return std::vector<DepthModelSpec>{indoor, outdoor, metric3d};
  }();
  return presets;
}

const DepthModelSpec* find_depth_model(const std::string& name) {
  for (const auto& spec : depth_model_presets())
    if (spec.name == name || spec.engine_stem == name) return &spec;
  return nullptr;
}

std::filesystem::path depth_engine_path(const std::filesystem::path& models_dir,
                                        const DepthModelSpec& spec, bool portrait) {
  const int width = portrait ? spec.input_height : spec.input_width;
  const int height = portrait ? spec.input_width : spec.input_height;
  return models_dir / (spec.engine_stem + "-" + std::to_string(width) + "x" + std::to_string(height) +
                       ".engine");
}

void DepthMap::to_output(float x, float y, float& u, float& v) const {
  const float cx = x + 0.5F, cy = y + 0.5F;
  const auto w = static_cast<float>(source_width), h = static_cast<float>(source_height);
  float upright_x = cx, upright_y = cy;
  if (rotation == 90) { upright_x = h - cy; upright_y = cx; }
  else if (rotation == 180) { upright_x = w - cx; upright_y = h - cy; }
  else if (rotation == 270) { upright_x = cy; upright_y = w - cx; }
  u = upright_x * scale_x + offset_x - 0.5F;
  v = upright_y * scale_y + offset_y - 0.5F;
}

float DepthMap::at_source(float x, float y) const {
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  if (empty()) return nan;
  float u = 0, v = 0;
  to_output(x, y, u, v);
  u = std::clamp(u, 0.0F, static_cast<float>(width - 1));
  v = std::clamp(v, 0.0F, static_cast<float>(height - 1));
  const int x0 = static_cast<int>(u), y0 = static_cast<int>(v);
  const int x1 = std::min(width - 1, x0 + 1), y1 = std::min(height - 1, y0 + 1);
  const float wx = u - x0, wy = v - y0;
  const auto at = [&](int px, int py) { return metres[static_cast<std::size_t>(py) * width + px]; };
  const float d00 = at(x0, y0), d10 = at(x1, y0), d01 = at(x0, y1), d11 = at(x1, y1);
  if (d00 <= 0 || d10 <= 0 || d01 <= 0 || d11 <= 0) return nan;  // padding
  return (d00 * (1 - wx) + d10 * wx) * (1 - wy) + (d01 * (1 - wx) + d11 * wx) * wy;
}

namespace {

void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

template <typename T>
struct TrtDelete {
  void operator()(T* value) const { delete value; }
};

// Letterbox of the upright frame inside the network input, in input pixels.
struct Letterbox {
  float scale;
  float offset_x, offset_y;
  float content_width, content_height;
};

struct Normalization {
  float mean[3];
  float inv_std[3];
  float pad[3];  // already normalized
};

// 8-bit sample of plane 0 (Y), 1 (U) or 2 (V); P010 keeps its top 8 bits.
__device__ float sample_plane(const GpuFrame& f, int plane, int x, int y) {
  if (plane == 0) {
    const auto* row = reinterpret_cast<const std::uint8_t*>(f.luma) + y * f.luma_pitch;
    return f.format == PixelFormat::nv12 ? row[x]
                                         : static_cast<float>(reinterpret_cast<const std::uint16_t*>(row)[x] >> 8U);
  }
  const auto* row = reinterpret_cast<const std::uint8_t*>(f.chroma) + (y / 2) * f.chroma_pitch;
  const int index = (x / 2) * 2 + (plane - 1);
  return f.format == PixelFormat::nv12 ? row[index]
                                       : static_cast<float>(reinterpret_cast<const std::uint16_t*>(row)[index] >> 8U);
}

__device__ void store(float* target, int index, float value) { target[index] = value; }
__device__ void store(__half* target, int index, float value) { target[index] = __float2half(value); }
__device__ float load(const float* source, int index) { return source[index]; }
__device__ float load(const __half* source, int index) { return __half2float(source[index]); }

// Upright, letterboxed, normalized planar RGB. Each input pixel averages up to
// 4x4 nearest-pixel taps over its source footprint (BT.709 limited range, as
// ColorSampler and the display).
template <typename T>
__global__ void preprocess(GpuFrame frame, int rotation, Letterbox box, Normalization norm,
                           T* input, int width, int height) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height) return;
  const int plane = width * height;
  const int index = j * width + i;
  const float upright_x = (i + 0.5F - box.offset_x) / box.scale;
  const float upright_y = (j + 0.5F - box.offset_y) / box.scale;
  if (upright_x < 0 || upright_y < 0 || i + 0.5F >= box.offset_x + box.content_width ||
      j + 0.5F >= box.offset_y + box.content_height) {
    for (int c = 0; c < 3; ++c) store(input, c * plane + index, norm.pad[c]);
    return;
  }
  const float footprint = 1.0F / box.scale;
  const int taps = min(4, max(1, static_cast<int>(ceilf(footprint))));
  const auto w = static_cast<float>(frame.width), h = static_cast<float>(frame.height);
  float rgb[3] = {0, 0, 0};
  for (int ty = 0; ty < taps; ++ty)
    for (int tx = 0; tx < taps; ++tx) {
      const float ux = upright_x + ((tx + 0.5F) / taps - 0.5F) * footprint;
      const float uy = upright_y + ((ty + 0.5F) / taps - 0.5F) * footprint;
      float cx = ux, cy = uy;  // source, continuous
      if (rotation == 90) { cx = uy; cy = h - ux; }
      else if (rotation == 180) { cx = w - ux; cy = h - uy; }
      else if (rotation == 270) { cx = w - uy; cy = ux; }
      const int x = min(max(static_cast<int>(cx), 0), frame.width - 1);
      const int y = min(max(static_cast<int>(cy), 0), frame.height - 1);
      const float luma = 1.164383F * (sample_plane(frame, 0, x, y) - 16.0F);
      const float u = sample_plane(frame, 1, x, y) - 128.0F;
      const float v = sample_plane(frame, 2, x, y) - 128.0F;
      rgb[0] += luma + 1.792741F * v;
      rgb[1] += luma - 0.213249F * u - 0.532909F * v;
      rgb[2] += luma + 2.112402F * u;
    }
  const float inv_taps = 1.0F / (taps * taps);
  for (int c = 0; c < 3; ++c) {
    const float value = fminf(255.0F, fmaxf(0.0F, rgb[c] * inv_taps));
    store(input, c * plane + index, (value - norm.mean[c]) * norm.inv_std[c]);
  }
}

// Network output -> metres; 0 in the letterbox padding (and for NaN).
template <typename T>
__global__ void postprocess(const T* output, int width, int height, float ratio_x, float ratio_y,
                            Letterbox box, float depth_scale, float max_depth, float* metres) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= width || j >= height) return;
  const float input_x = (i + 0.5F) / ratio_x, input_y = (j + 0.5F) / ratio_y;
  const bool inside = input_x >= box.offset_x && input_y >= box.offset_y &&
                      input_x < box.offset_x + box.content_width && input_y < box.offset_y + box.content_height;
  const float depth = load(output, j * width + i) * depth_scale;
  metres[j * width + i] = inside && depth == depth ? fminf(max_depth, fmaxf(0.0F, depth)) : 0.0F;
}

}  // namespace

struct DepthEstimator::Impl {
  DepthModelSpec spec;
  int rotation{};
  std::unique_ptr<nvinfer1::IRuntime, TrtDelete<nvinfer1::IRuntime>> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDelete<nvinfer1::ICudaEngine>> engine;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDelete<nvinfer1::IExecutionContext>> context;
  int input_width{}, input_height{}, output_width{}, output_height{};
  bool input_half{}, output_half{};
  void* device_input{};
  void* device_output{};
  float* device_metres{};
  float* host_metres{};  // pinned
  std::vector<void*> extra_outputs;  // bound but unused outputs (e.g. Metric3D normals)
  cudaStream_t stream{};
  cudaEvent_t begin{}, preprocessed{}, done{};
  bool pending{};
  DepthMap result;  // geometry of the pending submission

  ~Impl() {
    if (stream) cudaStreamSynchronize(stream);
    for (cudaEvent_t event : {begin, preprocessed, done})
      if (event) cudaEventDestroy(event);
    if (stream) cudaStreamDestroy(stream);
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_metres);
    cudaFreeHost(host_metres);
    for (void* buffer : extra_outputs) cudaFree(buffer);
  }

  bool half_tensor(const char* name) const {
    const auto type = engine->getTensorDataType(name);
    if (type == nvinfer1::DataType::kHALF) return true;
    if (type == nvinfer1::DataType::kFLOAT) return false;
    throw std::runtime_error(std::string("Depth tensor must be float32 or float16: ") + name);
  }
};

DepthEstimator::DepthEstimator(DepthModelSpec spec, const std::filesystem::path& engine_path, int rotation)
    : impl_(std::make_unique<Impl>()) {
  auto& m = *impl_;
  m.spec = std::move(spec);
  m.rotation = ((rotation % 360) + 360) % 360;
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Cannot open depth engine: " + std::filesystem::absolute(engine_path).string() +
                             " (build it with 'Native: build depth engines' in VS Code)");
  }
  const auto size = file.tellg();
  file.seekg(0);
  std::vector<char> bytes(static_cast<std::size_t>(size));
  file.read(bytes.data(), size);
  m.runtime.reset(nvinfer1::createInferRuntime(tensorrt_logger()));
  if (!m.runtime) throw std::runtime_error("Cannot create TensorRT runtime");
  m.engine.reset(m.runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
  if (!m.engine) throw std::runtime_error("Cannot deserialize depth engine " + engine_path.string());
  m.context.reset(m.engine->createExecutionContext());
  if (!m.context) throw std::runtime_error("Cannot create depth execution context");

  const char* input = m.spec.input_tensor.c_str();
  const char* output = m.spec.output_tensor.c_str();
  if (m.engine->getTensorIOMode(input) != nvinfer1::TensorIOMode::kINPUT)
    throw std::runtime_error("Depth engine has no input tensor '" + m.spec.input_tensor + "'");
  if (m.engine->getTensorIOMode(output) != nvinfer1::TensorIOMode::kOUTPUT)
    throw std::runtime_error("Depth engine has no output tensor '" + m.spec.output_tensor + "'");
  // Dynamic-shape engines (e.g. the community Metric3D export) run at the
  // optimization profile's preferred shape.
  auto input_shape = m.engine->getTensorShape(input);
  if (input_shape.nbDims != 4) throw std::runtime_error("Depth input must be 1x3xHxW");
  if (std::any_of(input_shape.d, input_shape.d + 4, [](auto d) { return d < 0; })) {
    input_shape = m.engine->getProfileShape(input, 0, nvinfer1::OptProfileSelector::kOPT);
    if (!m.context->setInputShape(input, input_shape))
      throw std::runtime_error("Cannot set the depth engine's input shape");
  }
  if (input_shape.d[0] != 1 || input_shape.d[1] != 3) throw std::runtime_error("Depth input must be 1x3xHxW");
  m.input_height = static_cast<int>(input_shape.d[2]);
  m.input_width = static_cast<int>(input_shape.d[3]);
  const auto output_shape = m.context->getTensorShape(output);
  if (output_shape.nbDims < 2) throw std::runtime_error("Depth output must end in HxW");
  for (int axis = 0; axis + 2 < output_shape.nbDims; ++axis)
    if (output_shape.d[axis] != 1) throw std::runtime_error("Depth output must be a single map");
  m.output_height = static_cast<int>(output_shape.d[output_shape.nbDims - 2]);
  m.output_width = static_cast<int>(output_shape.d[output_shape.nbDims - 1]);
  if (m.output_width <= 0 || m.output_height <= 0) throw std::runtime_error("Depth output shape is unresolved");
  m.input_half = m.half_tensor(input);
  m.output_half = m.half_tensor(output);

  const std::size_t input_count = 3ULL * m.input_width * m.input_height;
  const std::size_t output_count = static_cast<std::size_t>(m.output_width) * m.output_height;
  cuda_check(cudaMalloc(&m.device_input, input_count * (m.input_half ? 2 : 4)), "allocate depth input");
  cuda_check(cudaMalloc(&m.device_output, output_count * (m.output_half ? 2 : 4)), "allocate depth output");
  cuda_check(cudaMalloc(&m.device_metres, output_count * sizeof(float)), "allocate depth map");
  cuda_check(cudaMallocHost(&m.host_metres, output_count * sizeof(float)), "allocate host depth map");
  cuda_check(cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking), "create depth stream");
  for (cudaEvent_t* event : {&m.begin, &m.preprocessed, &m.done})
    cuda_check(cudaEventCreate(event), "create depth event");
  if (!m.context->setTensorAddress(input, m.device_input) || !m.context->setTensorAddress(output, m.device_output))
    throw std::runtime_error("Cannot bind depth engine buffers");
  // Any other outputs (Metric3D's normals) need somewhere to go.
  for (int index = 0; index < m.engine->getNbIOTensors(); ++index) {
    const char* name = m.engine->getIOTensorName(index);
    if (m.engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kOUTPUT || m.spec.output_tensor == name) continue;
    const auto shape = m.context->getTensorShape(name);
    std::size_t count = 1;
    for (int axis = 0; axis < shape.nbDims; ++axis) count *= static_cast<std::size_t>(std::max<std::int64_t>(1, shape.d[axis]));
    void* buffer = nullptr;
    cuda_check(cudaMalloc(&buffer, count * 4), "allocate auxiliary depth output");
    m.extra_outputs.push_back(buffer);
    if (!m.context->setTensorAddress(name, buffer)) throw std::runtime_error("Cannot bind depth output");
  }
}

DepthEstimator::~DepthEstimator() = default;

const DepthModelSpec& DepthEstimator::spec() const { return impl_->spec; }
int DepthEstimator::rotation() const { return impl_->rotation; }
int DepthEstimator::input_width() const { return impl_->input_width; }
int DepthEstimator::input_height() const { return impl_->input_height; }

bool DepthEstimator::busy() const {
  return impl_->pending && cudaEventQuery(impl_->done) == cudaErrorNotReady;
}

bool DepthEstimator::submit(const GpuFrame& frame, float focal_px) {
  auto& m = *impl_;
  if (busy()) return false;
  const bool sideways = m.rotation == 90 || m.rotation == 270;
  const int upright_width = sideways ? frame.height : frame.width;
  const int upright_height = sideways ? frame.width : frame.height;
  Letterbox box{};
  box.scale = std::min(static_cast<float>(m.input_width) / upright_width,
                       static_cast<float>(m.input_height) / upright_height);
  box.content_width = std::max(1.0F, std::round(upright_width * box.scale));
  box.content_height = std::max(1.0F, std::round(upright_height * box.scale));
  box.offset_x = std::floor((m.input_width - box.content_width) / 2);
  box.offset_y = std::floor((m.input_height - box.content_height) / 2);
  Normalization norm{};
  for (int c = 0; c < 3; ++c) {
    norm.mean[c] = m.spec.mean[c];
    norm.inv_std[c] = 1.0F / m.spec.std[c];
    norm.pad[c] = (m.spec.pad[c] - m.spec.mean[c]) * norm.inv_std[c];
  }
  const float ratio_x = static_cast<float>(m.output_width) / m.input_width;
  const float ratio_y = static_cast<float>(m.output_height) / m.input_height;
  const float depth_scale = m.spec.canonical_focal_px > 0 ? focal_px * box.scale / m.spec.canonical_focal_px : 1.0F;

  if (frame.ready_event)
    cuda_check(cudaStreamWaitEvent(m.stream, static_cast<cudaEvent_t>(frame.ready_event), 0), "wait for decoded frame");
  cuda_check(cudaEventRecord(m.begin, m.stream), "start depth timer");
  const dim3 block(16, 16);
  const dim3 input_grid((m.input_width + 15) / 16, (m.input_height + 15) / 16);
  if (m.input_half)
    preprocess<<<input_grid, block, 0, m.stream>>>(frame, m.rotation, box, norm, static_cast<__half*>(m.device_input),
                                                   m.input_width, m.input_height);
  else
    preprocess<<<input_grid, block, 0, m.stream>>>(frame, m.rotation, box, norm, static_cast<float*>(m.device_input),
                                                   m.input_width, m.input_height);
  cuda_check(cudaGetLastError(), "preprocess depth input");
  cuda_check(cudaEventRecord(m.preprocessed, m.stream), "mark depth preprocessing");
  if (!m.context->enqueueV3(m.stream)) throw std::runtime_error("TensorRT depth enqueue failed");
  const dim3 output_grid((m.output_width + 15) / 16, (m.output_height + 15) / 16);
  if (m.output_half)
    postprocess<<<output_grid, block, 0, m.stream>>>(static_cast<const __half*>(m.device_output), m.output_width,
                                                     m.output_height, ratio_x, ratio_y, box, depth_scale,
                                                     m.spec.max_depth_m, m.device_metres);
  else
    postprocess<<<output_grid, block, 0, m.stream>>>(static_cast<const float*>(m.device_output), m.output_width,
                                                     m.output_height, ratio_x, ratio_y, box, depth_scale,
                                                     m.spec.max_depth_m, m.device_metres);
  cuda_check(cudaGetLastError(), "postprocess depth output");
  cuda_check(cudaMemcpyAsync(m.host_metres, m.device_metres,
                             static_cast<std::size_t>(m.output_width) * m.output_height * sizeof(float),
                             cudaMemcpyDeviceToHost, m.stream), "read back depth");
  cuda_check(cudaEventRecord(m.done, m.stream), "finish depth");
  m.pending = true;

  m.result = {};
  m.result.frame_index = frame.frame_index;
  m.result.width = m.output_width;
  m.result.height = m.output_height;
  m.result.source_width = frame.width;
  m.result.source_height = frame.height;
  m.result.rotation = m.rotation;
  m.result.scale_x = box.scale * ratio_x;
  m.result.scale_y = box.scale * ratio_y;
  m.result.offset_x = box.offset_x * ratio_x;
  m.result.offset_y = box.offset_y * ratio_y;
  // The decoder may recycle the surface after this returns.
  cuda_check(cudaEventSynchronize(m.preprocessed), "wait for depth preprocessing");
  return true;
}

bool DepthEstimator::poll(DepthMap& out) {
  auto& m = *impl_;
  if (!m.pending || cudaEventQuery(m.done) != cudaSuccess) return false;
  m.pending = false;
  float elapsed = 0;
  cuda_check(cudaEventElapsedTime(&elapsed, m.begin, m.done), "time depth inference");
  m.result.gpu_ms = elapsed;
  m.result.metres.assign(m.host_metres, m.host_metres + static_cast<std::size_t>(m.output_width) * m.output_height);
  out = std::move(m.result);
  return true;
}

}  // namespace slam_native
