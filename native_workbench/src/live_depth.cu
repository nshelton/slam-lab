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
constexpr int kAnchorGrid = 4;      // pixels per cell of the anchors' interpolation

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

// Inverse distance weighting (power 4) of the anchors' inverse depths, on a
// grid of one value per kAnchorGrid x kAnchorGrid pixels (at their centre):
// the result is smooth, and every cell visits every anchor.
__global__ void anchors_kernel(const DepthAnchor* anchors, int count, int width, int height, float sigma,
                               float sigma_per_px, float* inverse_depth, float* variance) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const float centre_x = kAnchorGrid * x + 0.5F * (kAnchorGrid - 1), centre_y = kAnchorGrid * y + 0.5F * (kAnchorGrid - 1);
  float weights = 0.0F, sum = 0.0F, nearest = INFINITY;
  for (int k = 0; k < count; ++k) {
    const float dx = anchors[k].x - centre_x, dy = anchors[k].y - centre_y;
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

// The anchor grid at full resolution (bilinear between cell centres).
__global__ void upsample_kernel(const float* coarse, const float* coarse_variance, int coarse_width, int coarse_height,
                                int width, int height, float* inverse_depth, float* variance) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const float half = 0.5F * (kAnchorGrid - 1);
  const float gx = fminf(fmaxf((x - half) / kAnchorGrid, 0.0F), coarse_width - 1.0F);
  const float gy = fminf(fmaxf((y - half) / kAnchorGrid, 0.0F), coarse_height - 1.0F);
  const int x0 = min(static_cast<int>(gx), max(coarse_width - 2, 0)), y0 = min(static_cast<int>(gy), max(coarse_height - 2, 0));
  const int x1 = min(x0 + 1, coarse_width - 1), y1 = min(y0 + 1, coarse_height - 1);
  const float ax = gx - x0, ay = gy - y0;
  const auto blend = [&](const float* image) {
    return (1 - ay) * ((1 - ax) * image[y0 * coarse_width + x0] + ax * image[y0 * coarse_width + x1]) +
           ay * ((1 - ax) * image[y1 * coarse_width + x0] + ax * image[y1 * coarse_width + x1]);
  };
  inverse_depth[y * width + x] = blend(coarse);
  variance[y * width + x] = blend(coarse_variance);
}

__global__ void fill_kernel(float* inverse_depth, float* variance, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  inverse_depth[index] = 0.0F;
  variance[index] = INFINITY;
}

// Carries the image of the previous frame into the current one. Each current
// pixel looks up where the flow says it was, takes the nearest estimate
// there, and, with a pose, moves that point by the camera motion. The lookup
// is then shifted once so that the point taken really lands on this pixel:
// the result follows the pose, the flow only starts it. `relative` maps the
// current camera to the previous one.
__global__ void propagate_kernel(const float* old_depth, const float* old_variance, const FlowVector* flow,
                                 int flow_width, int flow_height, int flow_grid, Lens lens, Relative relative,
                                 int posed, LiveDepthConfig config, float* inverse_depth, float* variance) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= lens.width || y >= lens.height) return;
  const int index = y * lens.width + x;
  inverse_depth[index] = 0.0F;
  variance[index] = INFINITY;
  FlowVector f = flow_at(flow, flow_width, flow_height, flow_grid, float(x), float(y));
  f = flow_at(flow, flow_width, flow_height, flow_grid, x - f.dx, y - f.dy);
  float qx = x - f.dx, qy = y - f.dy;
  const float* R = relative.rotation;
  const float* t = relative.translation;
  for (int attempt = 0; attempt < 2; ++attempt) {
    // The nearest pixel with an estimate among the four around (qx, qy).
    const int x0 = static_cast<int>(floorf(qx)), y0 = static_cast<int>(floorf(qy));
    int source = -1, sx = 0, sy = 0;
    float nearest = INFINITY;
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i) {
        const int px = x0 + i, py = y0 + j;
        if (px < 0 || py < 0 || px >= lens.width || py >= lens.height) continue;
        if (!(old_variance[py * lens.width + px] < INFINITY)) continue;
        const float d = (px - qx) * (px - qx) + (py - qy) * (py - qy);
        if (d < nearest) {
          nearest = d;
          source = py * lens.width + px;
          sx = px;
          sy = py;
        }
      }
    if (source < 0) return;
    const float rho = old_depth[source], var = old_variance[source];
    if (!posed) {  // no camera motion known: the estimate rides the flow and ages faster
      const float sigma = 4.0F * config.process_sigma * rho;
      inverse_depth[index] = rho;
      variance[index] = var + sigma * sigma;
      return;
    }
    // The point in the previous camera, then in the current one: X_c = R^T (X_p - t).
    float ux = float(sx), uy = float(sy);
    lens.undistort(&ux, &uy);
    const float p[3] = {(ux - lens.cx) / lens.fx / rho - t[0], (uy - lens.cy) / lens.fy / rho - t[1], 1.0F / rho - t[2]};
    const float c[3] = {R[0] * p[0] + R[3] * p[1] + R[6] * p[2], R[1] * p[0] + R[4] * p[1] + R[7] * p[2],
                        R[2] * p[0] + R[5] * p[1] + R[8] * p[2]};
    if (!(c[2] > 1e-6F)) return;
    float landed_x = lens.fx * c[0] / c[2] + lens.cx, landed_y = lens.fy * c[1] / c[2] + lens.cy;
    lens.distort(&landed_x, &landed_y);
    const float miss_x = x - landed_x, miss_y = y - landed_y;
    const float miss = sqrtf(miss_x * miss_x + miss_y * miss_y);
    if (attempt == 0 && miss > 0.6F) {  // look where a point landing here would have come from
      qx = sx + miss_x;
      qy = sy + miss_y;
      continue;
    }
    if (miss > config.propagation_tolerance_px) return;  // disoccluded, or the flow and the pose disagree
    const float moved = 1.0F / c[2];
    const float ratio = moved / rho, sigma = config.process_sigma * moved;
    inverse_depth[index] = moved;
    variance[index] = var * ratio * ratio * ratio * ratio + sigma * sigma;
    return;
  }
}

// The state at the anchors' pixels (0 where it has none), for the scale fit.
__global__ void gather_kernel(const DepthAnchor* anchors, int count, const float* inverse_depth, const float* variance,
                              int width, int height, float* out) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= count) return;
  const int x = static_cast<int>(anchors[k].x + 0.5F), y = static_cast<int>(anchors[k].y + 0.5F);
  out[k] = 0.0F;
  if (x < 0 || y < 0 || x >= width || y >= height) return;
  if (variance[y * width + x] < INFINITY) out[k] = inverse_depth[y * width + x];
}

__global__ void scale_kernel(float* inverse_depth, float* variance, int count, float scale) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count || !(variance[index] < INFINITY)) return;
  inverse_depth[index] *= scale;
  variance[index] *= scale * scale;
}

// Product of two estimates of one inverse depth that agree; when they do not,
// the more certain one.
__device__ void combine(float a, float var_a, float b, float var_b, float sigmas, float* out, float* var_out) {
  const float difference = a - b, sum = var_a + var_b;
  if (difference * difference <= sigmas * sigmas * sum) {
    *out = (a * var_b + b * var_a) / sum;
    *var_out = var_a * var_b / sum;
  } else if (var_a <= var_b) {
    *out = a;
    *var_out = var_a;
  } else {
    *out = b;
    *var_out = var_b;
  }
}

// Folds this frame's measurement into the state.
__global__ void fuse_kernel(const float* measured, const float* measured_variance, LiveDepthConfig config,
                            float* inverse_depth, float* variance, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count || !(measured_variance[index] < INFINITY)) return;
  const float rho = measured[index];
  // What the matching noise does not cover: the poses and the model.
  const float extra = config.measurement_sigma * rho;
  const float measured_var = measured_variance[index] + extra * extra;
  if (!(variance[index] < INFINITY)) {
    inverse_depth[index] = rho;
    variance[index] = measured_var;
    return;
  }
  const float state = inverse_depth[index], state_var = variance[index];
  const float difference = rho - state;
  float fused, fused_var;
  if (difference * difference <= config.outlier_sigmas * config.outlier_sigmas * (state_var + measured_var)) {
    fused = (state * measured_var + rho * state_var) / (state_var + measured_var);
    fused_var = state_var * measured_var / (state_var + measured_var);
  } else if (measured_var < state_var) {  // the newcomer is the better one
    fused = rho;
    fused_var = measured_var;
  } else {  // doubt the estimate a little: repeated disagreement ends up replacing it
    fused = state;
    fused_var = state_var * 2.0F;
  }
  const float floor = config.min_sigma * fused;
  inverse_depth[index] = fused;
  variance[index] = fmaxf(fused_var, floor * floor);
}

// The image handed out: the state, or the state combined with the anchors' interpolation.
__global__ void output_kernel(const float* state, const float* state_variance, const float* anchored,
                              const float* anchored_variance, float sigmas, float* inverse_depth, float* variance,
                              int count, unsigned* counters) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float rho = state[index], var = state_variance[index];
  if (anchored) {
    if (var < INFINITY) combine(rho, var, anchored[index], anchored_variance[index], sigmas, &rho, &var);
    else {
      rho = anchored[index];
      var = anchored_variance[index];
    }
  }
  inverse_depth[index] = rho;
  variance[index] = var;
  if (var < INFINITY) atomicAdd(&counters[measured], 1U);
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
  float* inverse_depth{};  // the image handed out
  float* variance{};
  // LiveDepthMode::filter / fused: the estimate carried from frame to frame
  // (state[0], with state[1] as the propagation target), this frame's
  // measurement, and the anchors' interpolation.
  float* state[2]{};
  float* state_variance[2]{};
  float* measured{};
  float* measured_variance{};
  float* anchored{};
  float* anchored_variance{};
  float* coarse{};           // the anchors' interpolation on its grid
  float* coarse_variance{};
  float* gathered{};
  bool state_filled{};
  int state_segment{-1};
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
    cudaFree(gathered);
    cudaFree(counters);
    if (stream) cudaStreamDestroy(stream);
  }
  void release() {
    cudaFree(luma);
    cudaFree(flow);
    for (float** image : {&inverse_depth, &variance, &state[0], &state[1], &state_variance[0], &state_variance[1],
                          &measured, &measured_variance, &anchored, &anchored_variance, &coarse, &coarse_variance}) {
      cudaFree(*image);
      *image = nullptr;
    }
    luma = nullptr;
    flow = nullptr;
    width = height = flow_width = flow_height = 0;
  }
  void forget() {
    for (auto& slot : slots) slot = {};
    newest = -1;
    valid = false;
    state_filled = false;
  }
  void upload(const std::vector<DepthAnchor>& points) {
    if (points.size() > anchor_capacity) {
      cudaFree(anchors);
      cudaFree(gathered);
      anchors = nullptr;
      gathered = nullptr;
      anchor_capacity = 0;
      cuda_check(cudaMalloc(&anchors, 2 * points.size() * sizeof(DepthAnchor)), "allocate depth anchors");
      cuda_check(cudaMalloc(&gathered, 2 * points.size() * sizeof(float)), "allocate depth anchor lookups");
      anchor_capacity = 2 * points.size();
    }
    cuda_check(cudaMemcpyAsync(anchors, points.data(), points.size() * sizeof(DepthAnchor), cudaMemcpyHostToDevice,
                               stream), "upload depth anchors");
  }
  // reference = R current + t, for points in the two camera frames.
  static Relative relative(const Pose& reference, const Pose& current) {
    Relative out{};
    double R[9];
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        R[i * 3 + j] = 0;
        for (int k = 0; k < 3; ++k) R[i * 3 + j] += reference.rotation[i * 3 + k] * current.rotation[j * 3 + k];
        out.rotation[i * 3 + j] = static_cast<float>(R[i * 3 + j]);
      }
    for (int i = 0; i < 3; ++i) {
      double t = reference.translation[i];
      for (int j = 0; j < 3; ++j) t -= R[i * 3 + j] * current.translation[j];
      out.translation[i] = static_cast<float>(t);
    }
    return out;
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
    for (float** image : {&inverse_depth, &variance, &state[0], &state[1], &state_variance[0], &state_variance[1],
                          &measured, &measured_variance, &anchored, &anchored_variance, &coarse, &coarse_variance})
      cuda_check(cudaMalloc(image, pixels * sizeof(float)), "allocate live depth image");
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
const LiveDepthConfig& LiveDepth::config() const { return impl_->config; }
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
  const Lens lens{m.width, m.height, camera.fx, camera.fy, camera.cx, camera.cy, camera.k1};
  const int count = static_cast<int>(pixels);
  const unsigned strips = static_cast<unsigned>((pixels + 255) / 256);
  const auto read_counters = [&](unsigned* host) {
    cuda_check(cudaMemcpyAsync(host, m.counters, counter_count * sizeof(unsigned), cudaMemcpyDeviceToHost, m.stream),
               "read live depth counters");
    cuda_check(cudaStreamSynchronize(m.stream), "synchronize live depth");
  };
  // Matches the frame against an earlier one into the given image; false when no frame qualifies.
  const auto measure = [&](float* out, float* out_variance) {
    // The reference: the frame reference_gap back, or the oldest one before
    // that which the flow chain reaches and which has a pose in this segment.
    int gap = 0;
    for (int k = 1; k <= m.config.reference_gap; ++k) {
      if (!m.slots[(m.newest - k + 1 + kRing) % kRing].has_flow) break;  // the field into frame newest - k + 1
      const auto& candidate = m.slots[(m.newest - k + kRing) % kRing];
      if (!candidate.filled) break;
      if (candidate.posed && candidate.segment == segment) gap = k;
    }
    if (gap == 0) return false;
    const Ring ring{m.luma, m.flow, kRing, m.newest, m.flow_width, m.flow_height, m.flow_grid};
    cuda_check(cudaMemsetAsync(m.counters, 0, counter_count * sizeof(unsigned), m.stream),
               "clear live depth counters");
    measure_kernel<<<grid, block, 0, m.stream>>>(ring, gap, lens,
                                                Impl::relative(m.slots[(m.newest - gap + kRing) % kRing].pose, *pose),
                                                m.config, out, out_variance, m.counters);
    cuda_check(cudaGetLastError(), "measure depth");
    unsigned host[counter_count]{};
    read_counters(host);
    m.stats.measured = host[measured];
    m.stats.outside = host[outside];
    m.stats.off_line = host[off_line];
    m.stats.weak_gradient = host[weak_gradient];
    m.stats.at_search_edge = host[at_search_edge];
    m.stats.poor_match = host[poor_match];
    m.stats.behind = host[behind];
    m.gap = gap;
    return true;
  };
  const auto interpolate_anchors = [&](float* out, float* out_variance) {
    const int coarse_width = (m.width + kAnchorGrid - 1) / kAnchorGrid;
    const int coarse_height = (m.height + kAnchorGrid - 1) / kAnchorGrid;
    anchors_kernel<<<dim3((coarse_width + 15) / 16, (coarse_height + 15) / 16), block, 0, m.stream>>>(
        m.anchors, static_cast<int>(anchors.size()), coarse_width, coarse_height, m.config.anchor_sigma,
        m.config.anchor_sigma_per_px, m.coarse, m.coarse_variance);
    upsample_kernel<<<grid, block, 0, m.stream>>>(m.coarse, m.coarse_variance, coarse_width, coarse_height, m.width,
                                                 m.height, out, out_variance);
    cuda_check(cudaGetLastError(), "interpolate depth anchors");
  };
  const bool enough_anchors = pose && anchors.size() >= 8;
  if (enough_anchors) m.upload(anchors);

  if (m.config.mode == LiveDepthMode::anchors) {
    if (enough_anchors) {
      interpolate_anchors(m.inverse_depth, m.variance);
      m.stats.measured = m.stats.estimated = pixels;
      m.valid = true;
    }
  } else if (m.config.mode == LiveDepthMode::measurement) {
    if (pose && measure(m.inverse_depth, m.variance)) {
      m.stats.estimated = m.stats.measured;
      m.valid = true;
    }
  } else {
    // 1. Carry the state into this frame. Another segment has another frame and scale.
    if (pose && segment != m.state_segment) m.state_filled = false;
    if (m.state_filled && continues) {
      const auto& before = m.slots[(m.newest - 1 + kRing) % kRing];
      const bool moved = pose && before.posed && before.segment == segment;
      const std::size_t cells = static_cast<std::size_t>(m.flow_width) * m.flow_height;
      propagate_kernel<<<grid, block, 0, m.stream>>>(m.state[0], m.state_variance[0], m.flow + m.newest * cells,
                                                    m.flow_width, m.flow_height, m.flow_grid, lens,
                                                    moved ? Impl::relative(before.pose, *pose) : Relative{}, moved ? 1 : 0,
                                                    m.config, m.state[1], m.state_variance[1]);
      cuda_check(cudaGetLastError(), "propagate depth");
      std::swap(m.state[0], m.state[1]);
      std::swap(m.state_variance[0], m.state_variance[1]);
    } else {
      fill_kernel<<<strips, 256, 0, m.stream>>>(m.state[0], m.state_variance[0], count);
      cuda_check(cudaGetLastError(), "clear depth state");
      m.state_filled = true;
    }
    if (pose) {
      m.state_segment = segment;
      // 2. One scale for the whole state, so that it agrees with the landmarks
      // under it: it follows bundle adjustment and scale drift.
      if (m.config.scale_to_anchors && anchors.size() >= 20) {
        gather_kernel<<<static_cast<unsigned>((anchors.size() + 255) / 256), 256, 0, m.stream>>>(
            m.anchors, static_cast<int>(anchors.size()), m.state[0], m.state_variance[0], m.width, m.height,
            m.gathered);
        cuda_check(cudaGetLastError(), "look up depth anchors");
        std::vector<float> under(anchors.size());
        cuda_check(cudaMemcpyAsync(under.data(), m.gathered, under.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   m.stream), "read depth anchor lookups");
        cuda_check(cudaStreamSynchronize(m.stream), "synchronize live depth");
        std::vector<float> ratios;
        for (std::size_t k = 0; k < under.size(); ++k)
          if (under[k] > 0) ratios.push_back(anchors[k].inverse_depth / under[k]);
        if (ratios.size() >= 20) {
          std::nth_element(ratios.begin(), ratios.begin() + ratios.size() / 2, ratios.end());
          const float scale = ratios[ratios.size() / 2];
          m.stats.scale = scale;
          if (scale > 0.5F && scale < 2.0F && std::abs(scale - 1.0F) > 1e-3F) {
            scale_kernel<<<strips, 256, 0, m.stream>>>(m.state[0], m.state_variance[0], count, scale);
            cuda_check(cudaGetLastError(), "scale depth state");
          }
        }
      }
      // 3. This frame's measurement.
      if (measure(m.measured, m.measured_variance)) {
        fuse_kernel<<<strips, 256, 0, m.stream>>>(m.measured, m.measured_variance, m.config, m.state[0],
                                                 m.state_variance[0], count);
        cuda_check(cudaGetLastError(), "fuse depth");
      }
      // 4. The image: the state, or the state combined with what the landmarks alone say.
      const bool with_anchors = m.config.mode == LiveDepthMode::fused && enough_anchors;
      if (with_anchors) interpolate_anchors(m.anchored, m.anchored_variance);
      cuda_check(cudaMemsetAsync(m.counters, 0, counter_count * sizeof(unsigned), m.stream),
                 "clear live depth counters");
      output_kernel<<<strips, 256, 0, m.stream>>>(m.state[0], m.state_variance[0], with_anchors ? m.anchored : nullptr,
                                                 m.anchored_variance, m.config.outlier_sigmas, m.inverse_depth,
                                                 m.variance, count, m.counters);
      cuda_check(cudaGetLastError(), "compose depth image");
      unsigned host[counter_count]{};
      read_counters(host);
      m.stats.estimated = host[measured];
      m.valid = true;
    }
  }
  cuda_check(cudaStreamSynchronize(m.stream), "synchronize live depth");
  m.stats.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}

LiveDepthDevice LiveDepth::device() const {
  const auto& m = *impl_;
  if (!m.valid) return {};
  return {m.inverse_depth, m.variance, m.width, m.height};
}

bool LiveDepth::at(int x, int y, float* inverse_depth, float* variance) const {
  const auto& m = *impl_;
  if (!m.valid || x < 0 || y < 0 || x >= m.width || y >= m.height) return false;
  const std::size_t index = static_cast<std::size_t>(y) * m.width + x;
  cuda_check(cudaMemcpy(inverse_depth, m.inverse_depth + index, sizeof(float), cudaMemcpyDeviceToHost),
             "read a live depth pixel");
  cuda_check(cudaMemcpy(variance, m.variance + index, sizeof(float), cudaMemcpyDeviceToHost),
             "read a live depth pixel");
  return std::isfinite(*variance);
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
