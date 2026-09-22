#include "slam_native/optical_flow.hpp"

#include "nvOpticalFlowCuda.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace slam_native {
namespace {

void cuda_check(cudaError_t result, const char* action) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(result));
  }
}

void of_check(NV_OF_STATUS status, const char* action) {
  if (status != NV_OF_SUCCESS) {
    throw std::runtime_error(std::string(action) + ": NVIDIA optical-flow status " +
                             std::to_string(static_cast<int>(status)));
  }
}

__global__ void p010_to_luma8(const std::uint16_t* source, std::size_t source_pitch,
                               std::uint8_t* target, std::size_t target_pitch,
                               int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const auto* source_row = reinterpret_cast<const std::uint16_t*>(
      reinterpret_cast<const std::uint8_t*>(source) + y * source_pitch);
  target[y * target_pitch + x] = static_cast<std::uint8_t>(source_row[x] >> 8);
}

}  // namespace

struct OpticalFlow::State {
  void* library{};
  NV_OF_CUDA_API_FUNCTION_LIST api{};
  NvOFHandle handle{};
  NvOFGPUBufferHandle input[2]{};
  NvOFGPUBufferHandle output{};
  cudaStream_t stream{};
  int width{};
  int height{};
  int grid_width{};
  int grid_height{};
  int previous{-1};

  ~State() {
    if (stream) cudaStreamSynchronize(stream);
    if (handle) {
      for (auto& buffer : input) if (buffer) api.nvOFDestroyGPUBufferCuda(buffer);
      if (output) api.nvOFDestroyGPUBufferCuda(output);
      api.nvOFDestroy(handle);
    }
    if (stream) cudaStreamDestroy(stream);
    if (library) dlclose(library);
  }

  void initialize(int frame_width, int frame_height) {
    if (handle) {
      if (width != frame_width || height != frame_height) {
        throw std::runtime_error("Video resolution changed during optical-flow tracking");
      }
      return;
    }
    width = frame_width;
    height = frame_height;
    grid_width = (width + 3) / 4;
    grid_height = (height + 3) / 4;
    library = dlopen("libnvidia-opticalflow.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(std::string("Load NVIDIA optical-flow driver: ") + dlerror());
    auto create = reinterpret_cast<decltype(&NvOFAPICreateInstanceCuda)>(
        dlsym(library, "NvOFAPICreateInstanceCuda"));
    if (!create) throw std::runtime_error("NVIDIA optical-flow CUDA API is unavailable");
    of_check(create(NV_OF_API_VERSION, &api), "Initialize NVIDIA optical-flow API");
    cuda_check(cudaFree(nullptr), "Initialize CUDA context");
    CUcontext context{};
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context) {
      throw std::runtime_error("CUDA primary context is unavailable for optical flow");
    }
    of_check(api.nvCreateOpticalFlowCuda(context, &handle), "Create optical-flow session");
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "Create optical-flow CUDA stream");
    of_check(api.nvOFSetIOCudaStreams(handle, reinterpret_cast<CUstream>(stream),
                                      reinterpret_cast<CUstream>(stream)),
             "Set optical-flow CUDA stream");
    NV_OF_INIT_PARAMS init{};
    init.width = width;
    init.height = height;
    init.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    init.mode = NV_OF_MODE_OPTICALFLOW;
    init.perfLevel = NV_OF_PERF_LEVEL_MEDIUM;
    of_check(api.nvOFInit(handle, &init), "Initialize optical-flow engine");
    NV_OF_BUFFER_DESCRIPTOR input_desc{};
    input_desc.width = width;
    input_desc.height = height;
    input_desc.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;
    input_desc.bufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8;
    for (auto& buffer : input) {
      of_check(api.nvOFCreateGPUBufferCuda(handle, &input_desc,
                                          NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &buffer),
               "Allocate optical-flow input");
    }
    NV_OF_BUFFER_DESCRIPTOR output_desc{};
    output_desc.width = grid_width;
    output_desc.height = grid_height;
    output_desc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;
    output_desc.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2;
    of_check(api.nvOFCreateGPUBufferCuda(handle, &output_desc,
                                        NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &output),
             "Allocate optical-flow output");
  }

  void copy_frame(const GpuFrame& frame, int slot) {
    NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
    of_check(api.nvOFGPUBufferGetStrideInfo(input[slot], &stride),
             "Query optical-flow input stride");
    const auto destination = reinterpret_cast<std::uint8_t*>(
        api.nvOFGPUBufferGetCUdeviceptr(input[slot]));
    if (!destination || stride.numPlanes < 1) {
      throw std::runtime_error("Invalid optical-flow input buffer");
    }
    const std::size_t pitch = stride.strideInfo[0].strideXInBytes;
    if (frame.format == PixelFormat::nv12) {
      cuda_check(cudaMemcpy2DAsync(destination, pitch,
                  reinterpret_cast<const void*>(frame.luma), frame.luma_pitch,
                  width, height, cudaMemcpyDeviceToDevice, stream),
                 "Copy decoded luma to optical-flow input");
    } else {
      const dim3 block(16, 16);
      const dim3 grid((width + 15) / 16, (height + 15) / 16);
      p010_to_luma8<<<grid, block, 0, stream>>>(
          reinterpret_cast<const std::uint16_t*>(frame.luma), frame.luma_pitch,
          destination, pitch, width, height);
      cuda_check(cudaGetLastError(), "Convert P010 luma for optical flow");
    }
    cuda_check(cudaStreamSynchronize(stream), "Wait for optical-flow input copy");
  }
};

OpticalFlow::OpticalFlow() : state_(std::make_unique<State>()) {}
OpticalFlow::~OpticalFlow() = default;

void OpticalFlow::remember(const GpuFrame& frame) {
  state_->initialize(frame.width, frame.height);
  const int slot = state_->previous < 0 ? 0 : 1 - state_->previous;
  state_->copy_frame(frame, slot);
  state_->previous = slot;
}

bool OpticalFlow::has_reference() const { return state_->previous >= 0; }

FlowField OpticalFlow::compute(const GpuFrame& frame) {
  if (!has_reference()) throw std::runtime_error("Optical flow needs a previous frame");
  state_->initialize(frame.width, frame.height);
  const int current = 1 - state_->previous;
  state_->copy_frame(frame, current);
  NV_OF_EXECUTE_INPUT_PARAMS input{};
  input.inputFrame = state_->input[state_->previous];
  input.referenceFrame = state_->input[current];
  NV_OF_EXECUTE_OUTPUT_PARAMS output{};
  output.outputBuffer = state_->output;
  of_check(state_->api.nvOFExecute(state_->handle, &input, &output),
           "Compute NVIDIA optical flow");
  NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
  of_check(state_->api.nvOFGPUBufferGetStrideInfo(state_->output, &stride),
           "Query optical-flow output stride");
  const auto source = reinterpret_cast<const void*>(
      state_->api.nvOFGPUBufferGetCUdeviceptr(state_->output));
  if (!source || stride.numPlanes < 1) throw std::runtime_error("Invalid optical-flow output buffer");
  std::vector<NV_OF_FLOW_VECTOR> raw(
      static_cast<std::size_t>(state_->grid_width) * state_->grid_height);
  cuda_check(cudaMemcpy2DAsync(raw.data(), state_->grid_width * sizeof(NV_OF_FLOW_VECTOR),
              source, stride.strideInfo[0].strideXInBytes,
              state_->grid_width * sizeof(NV_OF_FLOW_VECTOR), state_->grid_height,
              cudaMemcpyDeviceToHost, state_->stream), "Read optical-flow vectors");
  cuda_check(cudaStreamSynchronize(state_->stream), "Wait for optical-flow vectors");
  FlowField field;
  field.width = state_->grid_width;
  field.height = state_->grid_height;
  field.vectors.reserve(raw.size());
  for (const auto vector : raw) {
    field.vectors.push_back({vector.flowx / 32.0F, vector.flowy / 32.0F});
  }
  state_->previous = current;
  return field;
}

}  // namespace slam_native
