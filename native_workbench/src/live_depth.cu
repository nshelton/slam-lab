#include "slam_native/live_depth.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace slam_native {
namespace {

constexpr int kMaxPatch = 45;       // (2 * 4 + 1) x (2 * 2 + 1)
constexpr int kMaxCandidates = 65;  // search steps
constexpr int kRing = 32;           // frames kept as possible references

enum Counter { measured, outside, off_line, weak_gradient, at_search_edge, poor_match, behind, counter_count };

void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

// previous = rotation * current + translation, for points in camera frames.
struct Relative {
  float rotation[9];
  float translation[3];
};

struct Lens {
  int width, height;
  float fx, fy, cx, cy;
  float k1;
  // Division model about the image centre, radius in half-diagonals (undistort_point).
  __device__ void undistort(float* x, float* y) const {
    if (k1 == 0) return;
    const float mx = 0.5F * (width - 1), my = 0.5F * (height - 1);
    const float h2 = 0.25F * (float(width) * width + float(height) * height);
    const float dx = *x - mx, dy = *y - my;
    const float scale = 1.0F / (1.0F + k1 * (dx * dx + dy * dy) / h2);
    *x = mx + dx * scale;
    *y = my + dy * scale;
  }
  __device__ void distort(float* x, float* y) const {
    if (k1 == 0) return;
    const float mx = 0.5F * (width - 1), my = 0.5F * (height - 1);
    const float h2 = 0.25F * (float(width) * width + float(height) * height);
    const float dx = *x - mx, dy = *y - my, ru = sqrtf(dx * dx + dy * dy);
    if (ru < 1e-6F) return;
    const float a = k1 * ru / h2;
    const float rd = 2.0F * ru / (1.0F + sqrtf(fmaxf(0.0F, 1.0F - 4.0F * a * ru)));
    *x = mx + dx * rd / ru;
    *y = my + dy * rd / ru;
  }
};

__global__ void copy_luma(const std::uint8_t* source, std::size_t pitch, int sixteen_bit, std::uint8_t* target,
                          int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const std::uint8_t* row = source + y * pitch;
  target[y * width + x] = sixteen_bit ? static_cast<std::uint8_t>(reinterpret_cast<const std::uint16_t*>(row)[x] >> 8U)
                                      : row[x];
}

// Bilinear luma; the caller keeps (x, y) at least one pixel inside the image.
__device__ float luma(const std::uint8_t* image, int width, float x, float y) {
  const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
  const float ax = x - x0, ay = y - y0;
  const std::uint8_t* p = image + y0 * width + x0;
  return (1 - ay) * ((1 - ax) * p[0] + ax * p[1]) + ay * ((1 - ax) * p[width] + ax * p[width + 1]);
}

// Forward flow at a source pixel: bilinear between the centres of the grid cells.
__device__ FlowVector flow_at(const FlowVector* field, int width, int height, int grid, float x, float y) {
  const float half = 0.5F * (grid - 1);
  const float gx = fminf(fmaxf((x - half) / grid, 0.0F), width - 1.0F);
  const float gy = fminf(fmaxf((y - half) / grid, 0.0F), height - 1.0F);
  const int x0 = min(static_cast<int>(gx), width - 2 < 0 ? 0 : width - 2);
  const int y0 = min(static_cast<int>(gy), height - 2 < 0 ? 0 : height - 2);
  const int x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
  const float ax = gx - x0, ay = gy - y0;
  const FlowVector a = field[y0 * width + x0], b = field[y0 * width + x1], c = field[y1 * width + x0],
                   d = field[y1 * width + x1];
  return {(1 - ay) * ((1 - ax) * a.dx + ax * b.dx) + ay * ((1 - ax) * c.dx + ax * d.dx),
          (1 - ay) * ((1 - ax) * a.dy + ax * b.dy) + ay * ((1 - ax) * c.dy + ax * d.dy)};
}

__device__ void no_estimate(float* inverse_depth, float* variance, int index, unsigned* counters, Counter why) {
  inverse_depth[index] = 0.0F;
  variance[index] = INFINITY;
  atomicAdd(&counters[why], 1U);
}

// The reference frames: a ring of the last frames' luma and of the flow
// fields that lead to them (field k is frame k - 1 -> frame k).
struct Ring {
  const std::uint8_t* luma;   // size frames of width x height
  const FlowVector* flow;     // size fields of flow_width x flow_height
  int size;
  int newest;                 // slot of the current frame
  int flow_width, flow_height, flow_grid;
};

// One pixel of the current frame against a reference frame `gap` frames back.
//
// With r the pixel's ray (z = 1) and rho its inverse depth, the point is
// r / rho, and in the reference camera it lies along a + rho b, a = R r,
// b = t. Its image there moves along a straight line (in undistorted pixels)
// from P(0), the match of a point at infinity, as rho grows. The optical flow,
// chained back through the frames in between, says where on the line the
// match is; the patch search refines that.
__global__ void measure_kernel(Ring ring, int gap, Lens lens, Relative relative, LiveDepthConfig config,
                               float* inverse_depth, float* variance, unsigned* counters) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= lens.width || y >= lens.height) return;
  const int index = y * lens.width + x;
  const int half_length = config.patch_half_length, half_width = config.patch_half_width;
  const float margin = 2.0F * (half_length + half_width) + 2.0F;
  const auto inside = [&](float px, float py) {
    return px >= margin && py >= margin && px < lens.width - 1 - margin && py < lens.height - 1 - margin;
  };
  if (!inside(float(x), float(y))) return no_estimate(inverse_depth, variance, index, counters, outside);
  const std::size_t pixels = static_cast<std::size_t>(lens.width) * lens.height;
  const std::size_t cells = static_cast<std::size_t>(ring.flow_width) * ring.flow_height;
  const std::uint8_t* current = ring.luma + ring.newest * pixels;
  const std::uint8_t* reference = ring.luma + ((ring.newest - gap + ring.size) % ring.size) * pixels;

  // Where the flow puts this pixel in the reference frame. Each field is
  // older -> newer, so it is inverted by two fixed-point steps.
  float guess_x = float(x), guess_y = float(y);
  for (int k = 0; k < gap; ++k) {
    const FlowVector* field = ring.flow + ((ring.newest - k + ring.size) % ring.size) * cells;
    FlowVector f = flow_at(field, ring.flow_width, ring.flow_height, ring.flow_grid, guess_x, guess_y);
    f = flow_at(field, ring.flow_width, ring.flow_height, ring.flow_grid, guess_x - f.dx, guess_y - f.dy);
    guess_x -= f.dx;
    guess_y -= f.dy;
    if (!inside(guess_x, guess_y)) return no_estimate(inverse_depth, variance, index, counters, outside);
  }
  lens.undistort(&guess_x, &guess_y);

  const float* R = relative.rotation;
  const float* b = relative.translation;
  // Undistorted reference pixel of the source pixel (sx, sy) at inverse depth
  // rho, and optionally its derivative by rho. False behind the reference camera.
  const auto project = [&](float sx, float sy, float rho, float* px, float* py, float* dx, float* dy) {
    lens.undistort(&sx, &sy);
    const float r0 = (sx - lens.cx) / lens.fx, r1 = (sy - lens.cy) / lens.fy;
    const float a0 = R[0] * r0 + R[1] * r1 + R[2], a1 = R[3] * r0 + R[4] * r1 + R[5], a2 = R[6] * r0 + R[7] * r1 + R[8];
    const float z = a2 + rho * b[2];
    if (z < 0.1F) return false;
    const float nx = (a0 + rho * b[0]) / z, ny = (a1 + rho * b[1]) / z;
    *px = lens.fx * nx + lens.cx;
    *py = lens.fy * ny + lens.cy;
    if (dx) {
      *dx = lens.fx * (b[0] - nx * b[2]) / z;
      *dy = lens.fy * (b[1] - ny * b[2]) / z;
    }
    return true;
  };
  float far_x, far_y, line_x, line_y;
  if (!project(float(x), float(y), 0.0F, &far_x, &far_y, &line_x, &line_y))
    return no_estimate(inverse_depth, variance, index, counters, outside);
  const float sensitivity = sqrtf(line_x * line_x + line_y * line_y);  // pixels per unit of rho, at rho = 0
  if (!(sensitivity > 1e-9F)) return no_estimate(inverse_depth, variance, index, counters, off_line);
  line_x /= sensitivity;
  line_y /= sensitivity;
  // The flow's guess, as a distance along the line and off it.
  const float along = (guess_x - far_x) * line_x + (guess_y - far_y) * line_y;
  const float across = (guess_x - far_x) * line_y - (guess_y - far_y) * line_x;
  if (fabsf(across) > config.max_epipolar_distance_px)
    return no_estimate(inverse_depth, variance, index, counters, off_line);
  // Inverse depth of a point of the line, from the axis the line runs along.
  float ux = float(x), uy = float(y);
  lens.undistort(&ux, &uy);
  const float r[3] = {(ux - lens.cx) / lens.fx, (uy - lens.cy) / lens.fy, 1.0F};
  const float a[3] = {R[0] * r[0] + R[1] * r[1] + R[2], R[3] * r[0] + R[4] * r[1] + R[5],
                      R[6] * r[0] + R[7] * r[1] + R[8]};
  const auto depth_at = [&](float distance) {
    const float px = far_x + distance * line_x, py = far_y + distance * line_y;
    if (fabsf(line_x) >= fabsf(line_y)) {
      const float n = (px - lens.cx) / lens.fx;
      return (a[0] - n * a[2]) / (n * b[2] - b[0]);
    }
    const float n = (py - lens.cy) / lens.fy;
    return (a[1] - n * a[2]) / (n * b[2] - b[1]);
  };

  // How the neighbourhood of the pixel maps into the reference frame, taking
  // it as a surface facing the camera at the guessed depth: J = d(reference
  // pixel) / d(current pixel). It carries the rotation and scale between the
  // frames into the patch.
  const float guess_rho = fmaxf(depth_at(along), 0.0F);
  float base_x, base_y, right_x, right_y, down_x, down_y;
  if (!project(float(x), float(y), guess_rho, &base_x, &base_y, nullptr, nullptr) ||
      !project(x + 1.0F, float(y), guess_rho, &right_x, &right_y, nullptr, nullptr) ||
      !project(float(x), y + 1.0F, guess_rho, &down_x, &down_y, nullptr, nullptr))
    return no_estimate(inverse_depth, variance, index, counters, outside);
  const float j00 = right_x - base_x, j10 = right_y - base_y, j01 = down_x - base_x, j11 = down_y - base_y;
  const float determinant = j00 * j11 - j01 * j10;
  if (!(fabsf(determinant) > 0.25F) || !(fabsf(determinant) < 4.0F))  // more than 2x scale change: not matchable
    return no_estimate(inverse_depth, variance, index, counters, poor_match);
  // The direction in the current image that maps onto the line: e = J^-1 l.
  float ex = (j11 * line_x - j01 * line_y) / determinant, ey = (-j10 * line_x + j00 * line_y) / determinant;
  const float stretch = 1.0F / sqrtf(ex * ex + ey * ey);  // reference pixels per current pixel, along the line
  ex *= stretch;
  ey *= stretch;

  // The patch in the current frame: a strip along that direction.
  const int length = 2 * half_length + 1, rows = 2 * half_width + 1, taps = length * rows;
  float patch[kMaxPatch];
  float mean = 0.0F, gradient = 0.0F;
  for (int j = 0; j < rows; ++j) {
    for (int i = 0; i < length; ++i) {
      const float u = float(i - half_length), v = float(j - half_width);
      const float value = luma(current, lens.width, x + u * ex - v * ey, y + u * ey + v * ex);
      patch[j * length + i] = value;
      mean += value;
      if (i > 0) {
        const float step = value - patch[j * length + i - 1];
        gradient += step * step;
      }
    }
  }
  mean /= taps;
  gradient = sqrtf(gradient / (rows * (length - 1)));
  if (gradient < config.min_gradient) return no_estimate(inverse_depth, variance, index, counters, weak_gradient);

  // Candidates along the line around the guess, in source pixels of the reference frame.
  float start_x = far_x + along * line_x, start_y = far_y + along * line_y;
  lens.distort(&start_x, &start_y);
  const int reach = min(static_cast<int>(config.search_radius_px / config.search_step_px + 0.5F),
                        (kMaxCandidates - 1) / 2);
  if (!inside(start_x - reach * config.search_step_px * line_x, start_y - reach * config.search_step_px * line_y) ||
      !inside(start_x + reach * config.search_step_px * line_x, start_y + reach * config.search_step_px * line_y))
    return no_estimate(inverse_depth, variance, index, counters, outside);
  // Patch offsets in the reference frame: J applied to the current frame's.
  const float along_x = j00 * ex + j01 * ey, along_y = j10 * ex + j11 * ey;
  const float cross_x = -j00 * ey + j01 * ex, cross_y = -j10 * ey + j11 * ex;
  float cost[kMaxCandidates];
  int best = 0;
  for (int k = -reach; k <= reach; ++k) {
    const float cx = start_x + k * config.search_step_px * line_x, cy = start_y + k * config.search_step_px * line_y;
    float values[kMaxPatch];
    float other = 0.0F;
    for (int j = 0; j < rows; ++j) {
      for (int i = 0; i < length; ++i) {
        const float u = float(i - half_length), v = float(j - half_width);
        values[j * length + i] = luma(reference, lens.width, cx + u * along_x + v * cross_x, cy + u * along_y + v * cross_y);
        other += values[j * length + i];
      }
    }
    other /= taps;
    float sum = 0.0F;  // means removed: exposure changes between frames
    for (int t = 0; t < taps; ++t) {
      const float difference = (patch[t] - mean) - (values[t] - other);
      sum += difference * difference;
    }
    cost[k + reach] = sum;
    if (sum < cost[best]) best = k + reach;
  }
  if (best == 0 || best == 2 * reach) return no_estimate(inverse_depth, variance, index, counters, at_search_edge);
  if (sqrtf(cost[best] / taps) > config.max_residual)
    return no_estimate(inverse_depth, variance, index, counters, poor_match);
  // Sub-step position from a parabola through the best cost and its neighbours.
  const float curvature = cost[best - 1] - 2.0F * cost[best] + cost[best + 1];
  const float offset = curvature > 1e-6F ? 0.5F * (cost[best - 1] - cost[best + 1]) / curvature : 0.0F;
  const float matched = along + (best - reach + fminf(fmaxf(offset, -0.5F), 0.5F)) * config.search_step_px;

  const float rho = depth_at(matched);
  if (!(rho > 0.0F) || !isfinite(rho)) return no_estimate(inverse_depth, variance, index, counters, behind);
  float at_x, at_y, slope_x, slope_y;
  if (!project(float(x), float(y), rho, &at_x, &at_y, &slope_x, &slope_y))
    return no_estimate(inverse_depth, variance, index, counters, behind);
  const float slope = sqrtf(slope_x * slope_x + slope_y * slope_y);  // pixels per unit of rho, here
  if (!(slope > 1e-9F)) return no_estimate(inverse_depth, variance, index, counters, behind);
  // Disparity noise in reference pixels; the gradient there is the current one over the stretch.
  const float reference_gradient = gradient / stretch;
  const float sigma2 = config.disparity_floor_px * config.disparity_floor_px +
                       2.0F * config.image_noise * config.image_noise / (reference_gradient * reference_gradient);
  inverse_depth[index] = rho;
  variance[index] = sigma2 / (slope * slope);
  atomicAdd(&counters[measured], 1U);
}

// Inverse distance weighting (power 4) of the anchors' inverse depths.
__global__ void anchors_kernel(const DepthAnchor* anchors, int count, int width, int height, float sigma,
                               float sigma_per_px, float* inverse_depth, float* variance) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  float weights = 0.0F, sum = 0.0F, nearest = INFINITY;
  for (int k = 0; k < count; ++k) {
    const float dx = anchors[k].x - x, dy = anchors[k].y - y;
    const float d2 = dx * dx + dy * dy + 1.0F;
    const float w = 1.0F / (d2 * d2);
    weights += w;
    sum += w * anchors[k].inverse_depth;
    nearest = fminf(nearest, d2);
  }
  const float rho = sum / weights;
  const float relative = sigma + sigma_per_px * sqrtf(nearest);
  inverse_depth[y * width + x] = rho;
  variance[y * width + x] = relative * relative * rho * rho;
}

}  // namespace

struct LiveDepth::Impl {
  LiveDepthConfig config;
  LiveDepthStats stats;
  cudaStream_t stream{};
  int width{}, height{};
  // Ring of the last kRing frames: luma, the flow field that led to each
  // frame, and what the odometry said about it.
  std::uint8_t* luma{};
  FlowVector* flow{};
  int flow_width{}, flow_height{}, flow_grid{4};
  int newest{-1};
  struct Slot {
    bool filled{}, posed{}, has_flow{};
    Pose pose;
    int segment{-1};
  };
  Slot slots[kRing];
  float* inverse_depth{};
  float* variance{};
  DepthAnchor* anchors{};
  std::size_t anchor_capacity{};
  unsigned* counters{};
  bool valid{};
  int gap{};  // of the last measurement
  std::uint64_t frame_index{};

  ~Impl() {
    if (stream) cudaStreamSynchronize(stream);
    release();
    cudaFree(anchors);
    cudaFree(counters);
    if (stream) cudaStreamDestroy(stream);
  }
  void release() {
    cudaFree(luma);
    cudaFree(flow);
    cudaFree(inverse_depth);
    cudaFree(variance);
    luma = nullptr;
    flow = nullptr;
    inverse_depth = variance = nullptr;
    width = height = flow_width = flow_height = 0;
  }
  void forget() {
    for (auto& slot : slots) slot = {};
    newest = -1;
    valid = false;
  }
  void ensure(int w, int h) {
    if (!stream) {
      cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create live depth stream");
      cuda_check(cudaMalloc(&counters, counter_count * sizeof(unsigned)), "allocate live depth counters");
    }
    if (w == width && h == height) return;
    cuda_check(cudaStreamSynchronize(stream), "synchronize live depth");
    release();
    const std::size_t pixels = static_cast<std::size_t>(w) * h;
    cuda_check(cudaMalloc(&luma, kRing * pixels), "allocate live depth luma");
    cuda_check(cudaMalloc(&inverse_depth, pixels * sizeof(float)), "allocate live depth image");
    cuda_check(cudaMalloc(&variance, pixels * sizeof(float)), "allocate live depth variance");
    width = w;
    height = h;
    forget();
  }
};

LiveDepth::LiveDepth(const LiveDepthConfig& config) : impl_(std::make_unique<Impl>()) {
  if (config.patch_half_length < 1 || config.patch_half_length > 4 || config.patch_half_width < 0 ||
      config.patch_half_width > 2 || !(config.search_step_px > 0) ||
      !(config.search_radius_px >= config.search_step_px) || config.reference_gap < 1 || config.reference_gap >= kRing)
    throw std::invalid_argument("live depth: patch or search out of range");
  impl_->config = config;
}
LiveDepth::~LiveDepth() = default;

void LiveDepth::reset() { impl_->forget(); }
int LiveDepth::reference_gap() const { return impl_->gap; }

bool LiveDepth::valid() const { return impl_->valid; }
const LiveDepthStats& LiveDepth::stats() const { return impl_->stats; }

void LiveDepth::process(const GpuFrame& frame, const DeviceFlowField* flow, const LiveDepthCamera& camera,
                        const Pose* pose, int segment, const std::vector<DepthAnchor>& anchors) {
  auto& m = *impl_;
  if (camera.width != frame.width || camera.height != frame.height)
    throw std::invalid_argument("live depth: the camera is not the frame's");
  m.ensure(frame.width, frame.height);
  const auto begin = std::chrono::steady_clock::now();
  const dim3 block(16, 16);
  const dim3 grid((m.width + 15) / 16, (m.height + 15) / 16);
  // The frame takes the next slot of the ring, with the flow that led to it.
  const std::size_t pixels = static_cast<std::size_t>(m.width) * m.height;
  const bool continues = m.newest >= 0 && flow && flow->forward;
  if (!continues) m.forget();  // no flow from the previous frame: the chain is broken
  m.newest = (m.newest + 1) % kRing;
  auto& slot = m.slots[m.newest];
  slot = {};
  slot.filled = true;
  slot.posed = pose != nullptr;
  if (pose) slot.pose = *pose;
  slot.segment = segment;
  if (frame.ready_event)
    cuda_check(cudaStreamWaitEvent(m.stream, static_cast<cudaEvent_t>(frame.ready_event), 0), "wait for the frame");
  copy_luma<<<grid, block, 0, m.stream>>>(reinterpret_cast<const std::uint8_t*>(frame.luma), frame.luma_pitch,
                                         frame.format == PixelFormat::nv12 ? 0 : 1, m.luma + m.newest * pixels,
                                         m.width, m.height);
  cuda_check(cudaGetLastError(), "copy luma");
  if (continues) {
    const std::size_t cells = static_cast<std::size_t>(flow->width) * flow->height;
    if (flow->width != m.flow_width || flow->height != m.flow_height) {
      cuda_check(cudaStreamSynchronize(m.stream), "synchronize live depth");
      cudaFree(m.flow);
      m.flow = nullptr;
      cuda_check(cudaMalloc(&m.flow, kRing * cells * sizeof(FlowVector)), "allocate live depth flow");
      m.flow_width = flow->width;
      m.flow_height = flow->height;
    }
    m.flow_grid = flow->grid_size;
    if (flow->ready_event)
      cuda_check(cudaStreamWaitEvent(m.stream, static_cast<cudaEvent_t>(flow->ready_event), 0), "wait for the flow");
    cuda_check(cudaMemcpyAsync(m.flow + m.newest * cells, flow->forward, cells * sizeof(FlowVector),
                               cudaMemcpyDeviceToDevice, m.stream), "keep the flow field");
    slot.has_flow = true;
  }

  m.valid = false;
  m.gap = 0;
  m.stats = {};
  m.stats.pixels = pixels;
  m.frame_index = frame.frame_index;
  if (m.config.mode == LiveDepthMode::anchors) {
    if (pose && anchors.size() >= 8) {
      if (anchors.size() > m.anchor_capacity) {
        cudaFree(m.anchors);
        m.anchors = nullptr;
        m.anchor_capacity = 0;
        cuda_check(cudaMalloc(&m.anchors, 2 * anchors.size() * sizeof(DepthAnchor)), "allocate depth anchors");
        m.anchor_capacity = 2 * anchors.size();
      }
      cuda_check(cudaMemcpyAsync(m.anchors, anchors.data(), anchors.size() * sizeof(DepthAnchor),
                                 cudaMemcpyHostToDevice, m.stream), "upload depth anchors");
      anchors_kernel<<<grid, block, 0, m.stream>>>(m.anchors, static_cast<int>(anchors.size()), m.width, m.height,
                                                  m.config.anchor_sigma, m.config.anchor_sigma_per_px,
                                                  m.inverse_depth, m.variance);
      cuda_check(cudaGetLastError(), "interpolate depth anchors");
      m.stats.measured = m.stats.pixels;
      m.valid = true;
    }
  } else if (pose) {
    // The reference: the frame reference_gap back, or the oldest one before
    // that which the flow chain reaches and which has a pose in this segment.
    int gap = 0;
    for (int k = 1; k <= m.config.reference_gap; ++k) {
      if (!m.slots[(m.newest - k + 1 + kRing) % kRing].has_flow) break;  // the field into frame newest - k + 1
      const auto& candidate = m.slots[(m.newest - k + kRing) % kRing];
      if (!candidate.filled) break;
      if (candidate.posed && candidate.segment == segment) gap = k;
    }
    if (gap > 0) {
      // reference = R_r R_c^T current + (t_r - R_r R_c^T t_c)
      const Pose& reference = m.slots[(m.newest - gap + kRing) % kRing].pose;
      Relative relative{};
      const auto& Rr = reference.rotation;
      const auto& Rc = pose->rotation;
      double R[9];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          R[i * 3 + j] = 0;
          for (int k = 0; k < 3; ++k) R[i * 3 + j] += Rr[i * 3 + k] * Rc[j * 3 + k];
          relative.rotation[i * 3 + j] = static_cast<float>(R[i * 3 + j]);
        }
      for (int i = 0; i < 3; ++i) {
        double t = reference.translation[i];
        for (int j = 0; j < 3; ++j) t -= R[i * 3 + j] * pose->translation[j];
        relative.translation[i] = static_cast<float>(t);
      }
      const Lens lens{m.width, m.height, camera.fx, camera.fy, camera.cx, camera.cy, camera.k1};
      const Ring ring{m.luma, m.flow, kRing, m.newest, m.flow_width, m.flow_height, m.flow_grid};
      cuda_check(cudaMemsetAsync(m.counters, 0, counter_count * sizeof(unsigned), m.stream),
                 "clear live depth counters");
      measure_kernel<<<grid, block, 0, m.stream>>>(ring, gap, lens, relative, m.config, m.inverse_depth, m.variance,
                                                  m.counters);
      cuda_check(cudaGetLastError(), "measure depth");
      unsigned host[counter_count]{};
      cuda_check(cudaMemcpyAsync(host, m.counters, sizeof(host), cudaMemcpyDeviceToHost, m.stream),
                 "read live depth counters");
      cuda_check(cudaStreamSynchronize(m.stream), "synchronize live depth");
      m.stats.measured = host[measured];
      m.stats.outside = host[outside];
      m.stats.off_line = host[off_line];
      m.stats.weak_gradient = host[weak_gradient];
      m.stats.at_search_edge = host[at_search_edge];
      m.stats.poor_match = host[poor_match];
      m.stats.behind = host[behind];
      m.gap = gap;
      m.valid = true;
    }
  }
  cuda_check(cudaStreamSynchronize(m.stream), "synchronize live depth");
  m.stats.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}

LiveDepthImage LiveDepth::download() const {
  const auto& m = *impl_;
  LiveDepthImage image;
  if (!m.valid) return image;
  image.frame_index = m.frame_index;
  image.width = m.width;
  image.height = m.height;
  const std::size_t pixels = static_cast<std::size_t>(m.width) * m.height;
  image.inverse_depth.resize(pixels);
  image.variance.resize(pixels);
  cuda_check(cudaMemcpy(image.inverse_depth.data(), m.inverse_depth, pixels * sizeof(float), cudaMemcpyDeviceToHost),
             "download live depth");
  cuda_check(cudaMemcpy(image.variance.data(), m.variance, pixels * sizeof(float), cudaMemcpyDeviceToHost),
             "download live depth variance");
  return image;
}

std::vector<DepthAnchor> depth_anchors(const TrackedFrame& frame, const VisualOdometry& odometry,
                                       const OdometryFrameResult& result) {
  std::vector<DepthAnchor> anchors;
  if (!result.has_pose || result.predicted) return anchors;
  const auto landmarks = odometry.tracked_landmarks();  // sorted by track id
  const std::unordered_set<std::uint64_t> inliers(result.pose_inliers.begin(), result.pose_inliers.end());
  const auto& R = result.pose.rotation;
  const auto& t = result.pose.translation;
  for (const auto& observation : frame.observations) {
    if (!inliers.count(observation.track_id)) continue;
    const auto it = std::lower_bound(landmarks.begin(), landmarks.end(), observation.track_id,
                                     [](const TrackedLandmark& l, std::uint64_t id) { return l.track_id < id; });
    if (it == landmarks.end() || it->track_id != observation.track_id) continue;
    const double z = R[6] * it->position[0] + R[7] * it->position[1] + R[8] * it->position[2] + t[2];
    if (z > 1e-6) anchors.push_back({observation.x, observation.y, static_cast<float>(1.0 / z)});
  }
  return anchors;
}

LiveDepthCamera live_depth_camera(const VisualOdometryConfig& config, int width, int height) {
  const auto k = config.intrinsics ? *config.intrinsics :
      CameraIntrinsics::from_horizontal_fov(width, height, config.horizontal_fov_degrees);
  return {width, height, static_cast<float>(k.fx), static_cast<float>(k.fy), static_cast<float>(k.cx),
          static_cast<float>(k.cy), static_cast<float>(config.distortion_k1)};
}

}  // namespace slam_native
