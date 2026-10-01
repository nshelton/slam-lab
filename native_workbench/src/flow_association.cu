// GPU-resident SuperPoint + optical-flow tracker. See ARCHITECTURE.md
// ("Tracking and keypoint decisions") for the rules implemented here.
#include "slam_native/flow_tracker.hpp"
#include <cub/block/block_scan.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace slam_native {
namespace {
constexpr int kDimension = 256;
constexpr int kCandidates = 8;      // best gated detections kept per track
constexpr int kBlock = 1024;        // single-block compaction kernels
constexpr unsigned kFull = 0xffffffffU;
constexpr unsigned long long kNoProposal = ~0ULL;

enum Kind : unsigned { kMatched = 0, kCoasted = 1, kNew = 2 };
enum PredictionFlags : int { kValid = 1, kReliable = 2 };

void check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
template <class T> struct Buffer {
  T* data{};
  std::size_t capacity{};
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { if (data) cudaFree(data); }
  void reserve(std::size_t count) {
    if (count <= capacity) return;
    T* allocation{};
    check(cudaMalloc(&allocation, std::max<std::size_t>(count, 1) * sizeof(T)), "Allocate tracker buffer");
    if (data) cudaFree(data);
    data = allocation;
    capacity = count;
  }
};

struct DeviceTrack {
  float x, y, score;
  float variance;    // position variance per axis (px^2), see FlowTrackerConfig
  unsigned length;   // SuperPoint observations
  unsigned misses;   // consecutive frames carried by flow only
  unsigned long long id;
  unsigned long long first_frame;
  unsigned long long last_frame;  // last frame with a SuperPoint observation
};

// Host-visible compact per-track output.
struct TrackRecord {
  float x, y, score;
  float prediction_x, prediction_y;
  float correction;   // matched: |detection - prediction|
  float similarity;   // matched: descriptor cosine
  float fb_error;     // forward-backward residual of the prediction
  unsigned length, misses, kind;
  int detection;      // index of the supporting detection this frame; -1 when coasted
  unsigned long long id, first_frame, last_frame;
};
struct Counts {
  int matched, coasted, fresh, total;
};

// Exact replica of feature_store.cpp float_to_half (round half up), so live
// float32 descriptors and cached float16 descriptors quantize identically.
__device__ __half store_half(float value) {
  const unsigned bits = __float_as_uint(value);
  const unsigned sign = (bits >> 16U) & 0x8000U;
  const unsigned exponent = (bits >> 23U) & 0xffU;
  const unsigned mantissa = bits & 0x7fffffU;
  unsigned short result;
  if (exponent == 0xffU) {
    result = sign | 0x7c00U | (mantissa ? 0x0200U : 0U);
  } else {
    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
      result = sign | 0x7c00U;
    } else if (half_exponent <= 0) {
      if (half_exponent < -10) {
        result = sign;
      } else {
        const unsigned normalized = mantissa | 0x800000U;
        const int shift = 14 - half_exponent;
        result = sign | ((normalized + (1U << (shift - 1))) >> shift);
      }
    } else {
      const unsigned rounded = mantissa + 0x1000U;
      if (rounded & 0x800000U) {
        const int incremented = half_exponent + 1;
        result = sign | (incremented >= 31 ? 0x7c00U : incremented << 10U);
      } else {
        result = sign | (half_exponent << 10U) | (rounded >> 13U);
      }
    }
  }
  return __ushort_as_half(result);
}

__global__ void quantize(const float* source, __half* target, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) target[i] = store_half(source[i]);
}

// Cell (c, r) is the 4x4 block centred at (grid*c + (grid-1)/2, ...).
// Must match FlowField::sample in flow_tracker.cpp.
__device__ float2 sample(const FlowVector* field, int columns, int rows, int grid, float x, float y) {
  const float half = 0.5F * static_cast<float>(grid - 1);
  const float gx = fminf(fmaxf((x - half) / grid, 0.0F), float(columns - 1));
  const float gy = fminf(fmaxf((y - half) / grid, 0.0F), float(rows - 1));
  const int x0 = static_cast<int>(gx), y0 = static_cast<int>(gy);
  const int x1 = min(x0 + 1, columns - 1), y1 = min(y0 + 1, rows - 1);
  const float wx = gx - x0, wy = gy - y0;
  const FlowVector a = field[y0 * columns + x0], b = field[y0 * columns + x1];
  const FlowVector c = field[y1 * columns + x0], d = field[y1 * columns + x1];
  return {(a.dx * (1 - wx) + b.dx * wx) * (1 - wy) + (c.dx * (1 - wx) + d.dx * wx) * wy,
          (a.dy * (1 - wx) + b.dy * wx) * (1 - wy) + (c.dy * (1 - wx) + d.dy * wx) * wy};
}

__global__ void predict(const DeviceTrack* tracks, int count, const FlowVector* forward,
                        const FlowVector* backward, int columns, int rows, int grid, int width,
                        int height, float fb_threshold, float2* predictions, int* flags,
                        float* fb_errors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const auto track = tracks[i];
  const float2 v = sample(forward, columns, rows, grid, track.x, track.y);
  const float px = track.x + v.x, py = track.y + v.y;
  const bool valid = isfinite(px) && isfinite(py) && px >= 0 && py >= 0 && px < width && py < height;
  bool reliable = valid;
  float fb = CUDART_NAN_F;
  if (valid && backward && fb_threshold > 0) {
    const float2 b = sample(backward, columns, rows, grid, px, py);
    fb = hypotf(px + b.x - track.x, py + b.y - track.y);
    reliable = fb <= fb_threshold;  // NaN fails
  }
  predictions[i] = valid ? float2{px, py} : float2{CUDART_NAN_F, CUDART_NAN_F};
  flags[i] = (valid ? kValid : 0) | (reliable ? kReliable : 0);
  fb_errors[i] = fb;
}

// One warp per track: scan every detection for the radius gate, then score
// in-radius detections by descriptor cosine. Keeps the kCandidates cheapest
// gated detections, ordered by (cost, detection index).
__global__ void gather_candidates(const float2* predictions, const int* flags, int track_count,
                                  const __half* track_descriptors, const Keypoint* detections,
                                  const __half* detection_descriptors, int detection_count,
                                  float radius2, float min_similarity, float weight,
                                  int* candidates, float* costs, float* similarities,
                                  int* candidate_counts) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (i >= track_count) return;
  int found = 0;
  int best[kCandidates];
  float best_cost[kCandidates], best_similarity[kCandidates];
  if (flags[i] & kValid) {
    const float2 p = predictions[i];
    float own[8];
    const __half* descriptor = track_descriptors + static_cast<std::size_t>(i) * kDimension + lane * 8;
    for (int k = 0; k < 8; ++k) own[k] = __half2float(descriptor[k]);
    for (int base = 0; base < detection_count; base += 32) {
      const int j = base + lane;
      float d2 = CUDART_INF_F;
      if (j < detection_count) {
        const float dx = detections[j].x - p.x, dy = detections[j].y - p.y;
        d2 = dx * dx + dy * dy;
      }
      unsigned inside = __ballot_sync(kFull, d2 <= radius2);
      while (inside) {
        const int bit = __ffs(inside) - 1;
        inside &= inside - 1;
        const int candidate = base + bit;
        const float distance2 = __shfl_sync(kFull, d2, bit);
        const __half* other = detection_descriptors + static_cast<std::size_t>(candidate) * kDimension + lane * 8;
        float dot = 0;
        for (int k = 0; k < 8; ++k) dot += own[k] * __half2float(other[k]);
        for (int offset = 16; offset > 0; offset /= 2) dot += __shfl_xor_sync(kFull, dot, offset);
        if (lane == 0 && dot >= min_similarity) {
          const float cost = distance2 / radius2 + weight * (1.0F - dot);
          int position = min(found, kCandidates);
          while (position > 0 && best_cost[position - 1] > cost) --position;
          if (position < kCandidates) {
            for (int k = min(found, kCandidates - 1); k > position; --k) {
              best[k] = best[k - 1];
              best_cost[k] = best_cost[k - 1];
              best_similarity[k] = best_similarity[k - 1];
            }
            best[position] = candidate;
            best_cost[position] = cost;
            best_similarity[position] = dot;
            found = min(found + 1, kCandidates);
          }
        }
      }
    }
  }
  if (lane == 0) {
    candidate_counts[i] = found;
    for (int k = 0; k < found; ++k) {
      candidates[i * kCandidates + k] = best[k];
      costs[i * kCandidates + k] = best_cost[k];
      similarities[i * kCandidates + k] = best_similarity[k];
    }
  }
}

// Round of a deterministic propose/accept assignment. Each unassigned track
// proposes to its cheapest still-free candidate; each detection keeps the
// cheapest proposal (ties: lower track index) via a packed 64-bit atomicMin,
// so the result does not depend on thread scheduling.
__global__ void propose(const int* candidates, const float* costs, const int* candidate_counts,
                        int track_count, const int* track_match, const int* detection_match,
                        unsigned long long* proposals) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= track_count || track_match[i] >= 0) return;
  for (int k = 0; k < candidate_counts[i]; ++k) {
    const int j = candidates[i * kCandidates + k];
    if (detection_match[j] >= 0) continue;
    const unsigned long long key =
        (static_cast<unsigned long long>(__float_as_uint(costs[i * kCandidates + k])) << 32) |
        static_cast<unsigned>(i);
    atomicMin(proposals + j, key);
    return;
  }
}
__global__ void accept(unsigned long long* proposals, int detection_count, int* track_match,
                       int* detection_match) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= detection_count) return;
  const unsigned long long key = proposals[j];
  if (key == kNoProposal) return;
  const int i = static_cast<int>(key & 0xffffffffULL);
  detection_match[j] = i;
  track_match[i] = j;
  proposals[j] = kNoProposal;
}

// Rank of each unmatched detection by (score desc, index asc) among unmatched.
__global__ void rank_unmatched(const Keypoint* detections, const int* detection_match, int count,
                               int* ranks) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= count) return;
  if (detection_match[j] >= 0) { ranks[j] = -1; return; }
  const float score = detections[j].score;
  int rank = 0;
  for (int k = 0; k < count; ++k) {
    if (k == j || detection_match[k] >= 0) continue;
    const float other = detections[k].score;
    rank += other > score || (other == score && k < j);
  }
  ranks[j] = rank;
}

// Single block: lays out next-frame slots as [matched (detection order) |
// coasted (previous-track order) | new (score rank)], capped at `capacity`.
__global__ void layout(const int* detection_match, const int* ranks, int detection_count,
                       const int* track_match, const int* flags, const DeviceTrack* tracks,
                       int track_count, int max_coast, int capacity, int* slot_detection,
                       int* slot_track, Counts* counts) {
  using Scan = cub::BlockScan<int, kBlock>;
  __shared__ typename Scan::TempStorage storage;
  __shared__ int base;
  if (threadIdx.x == 0) base = 0;
  __syncthreads();
  for (int start = 0; start < detection_count; start += kBlock) {
    const int j = start + threadIdx.x;
    const int flag = j < detection_count && detection_match[j] >= 0;
    int position, total;
    Scan(storage).ExclusiveSum(flag, position, total);
    if (flag) {
      slot_detection[base + position] = j;
      slot_track[base + position] = detection_match[j];
    }
    __syncthreads();
    if (threadIdx.x == 0) base += total;
    __syncthreads();
  }
  const int matched = base;
  for (int start = 0; start < track_count; start += kBlock) {
    const int i = start + threadIdx.x;
    const int flag = i < track_count && track_match[i] < 0 && (flags[i] & kReliable) &&
                     static_cast<int>(tracks[i].misses) < max_coast;
    int position, total;
    Scan(storage).ExclusiveSum(flag, position, total);
    if (flag && base + position < capacity) {
      slot_detection[base + position] = -1;
      slot_track[base + position] = i;
    }
    __syncthreads();
    if (threadIdx.x == 0) base = min(capacity, base + total);
    __syncthreads();
  }
  const int coasted = base - matched;
  const int remaining = capacity - base;
  int unmatched = 0;
  for (int start = 0; start < detection_count; start += kBlock) {
    const int j = start + threadIdx.x;
    const int rank = j < detection_count ? ranks[j] : -1;
    if (rank >= 0 && rank < remaining) {
      slot_detection[base + rank] = j;
      slot_track[base + rank] = -1;
    }
    int position, total;
    Scan(storage).ExclusiveSum(rank >= 0 ? 1 : 0, position, total);
    unmatched += total;  // identical in every thread
    __syncthreads();
  }
  const int fresh_total = min(unmatched, remaining);
  if (threadIdx.x == 0) {
    counts->matched = matched;
    counts->coasted = coasted;
    counts->fresh = fresh_total;
    counts->total = base + fresh_total;
  }
}

__global__ void build(const Counts* counts, const int* slot_detection, const int* slot_track,
                      const DeviceTrack* tracks, const float2* predictions, const float* fb_errors,
                      const Keypoint* detections, const int* candidates, const float* similarities,
                      const int* candidate_counts, int capacity, unsigned long long first_id,
                      unsigned long long frame, float detection_variance, float flow_variance,
                      DeviceTrack* next, TrackRecord* records) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= capacity || slot >= counts->total) return;
  const int j = slot_detection[slot], i = slot_track[slot];
  DeviceTrack track{};
  TrackRecord record{};
  record.prediction_x = record.prediction_y = record.correction = record.similarity =
      record.fb_error = CUDART_NAN_F;
  if (i >= 0) {
    track = tracks[i];
    record.prediction_x = predictions[i].x;
    record.prediction_y = predictions[i].y;
    record.fb_error = fb_errors[i];
  }
  record.detection = j;
  if (j >= 0) {
    const Keypoint point = detections[j];
    if (i >= 0 && flow_variance > 0) {
      // Kalman update of the flow prediction with the detection.
      const float predicted = track.variance + flow_variance;
      const float gain = predicted / (predicted + detection_variance);
      track.x = record.prediction_x + gain * (point.x - record.prediction_x);
      track.y = record.prediction_y + gain * (point.y - record.prediction_y);
      track.variance = (1 - gain) * predicted;
    } else {
      track.x = point.x;
      track.y = point.y;
      track.variance = detection_variance;
    }
    track.score = point.score;
    track.misses = 0;
    track.last_frame = frame;
    if (i >= 0) {
      ++track.length;
      record.kind = kMatched;
      record.correction = hypotf(point.x - record.prediction_x, point.y - record.prediction_y);
      for (int k = 0; k < candidate_counts[i]; ++k) {
        if (candidates[i * kCandidates + k] == j) record.similarity = similarities[i * kCandidates + k];
      }
    } else {
      const int rank = slot - (counts->matched + counts->coasted);
      track.id = first_id + static_cast<unsigned long long>(rank);
      track.length = 1;
      track.first_frame = frame;
      record.kind = kNew;
    }
  } else {
    track.x = record.prediction_x;
    track.y = record.prediction_y;
    if (flow_variance > 0) track.variance += flow_variance;
    ++track.misses;
    record.kind = kCoasted;
  }
  next[slot] = track;
  record.x = track.x;
  record.y = track.y;
  record.score = track.score;
  record.length = track.length;
  record.misses = track.misses;
  record.id = track.id;
  record.first_frame = track.first_frame;
  record.last_frame = track.last_frame;
  records[slot] = record;
}

// Matched/new tracks take the current detection descriptor; coasted tracks
// keep their last observed one. 32 threads per slot, 8 halves each.
__global__ void copy_descriptors(const Counts* counts, const int* slot_detection, const int* slot_track,
                                 const __half* previous, const __half* detections, int capacity,
                                 __half* next) {
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int slot = thread / 32, lane = thread % 32;
  if (slot >= capacity || slot >= counts->total) return;
  const int j = slot_detection[slot];
  const __half* source = j >= 0 ? detections + static_cast<std::size_t>(j) * kDimension :
                                  previous + static_cast<std::size_t>(slot_track[slot]) * kDimension;
  const auto* from = reinterpret_cast<const uint4*>(source + lane * 8);
  auto* to = reinterpret_cast<uint4*>(next + static_cast<std::size_t>(slot) * kDimension + lane * 8);
  *to = *from;
}

int blocks(std::size_t count, int threads) { return static_cast<int>((count + threads - 1) / threads); }
}  // namespace

struct FlowTracker::Device {
  cudaStream_t stream{};
  int current{};  // index of the live track buffers
  Buffer<DeviceTrack> tracks[2];
  Buffer<__half> descriptors[2];
  Buffer<float2> predictions;
  Buffer<int> flags, candidates, candidate_counts, track_match;
  Buffer<float> costs, similarities, fb_errors;
  Buffer<Keypoint> detection_upload;
  Buffer<float> descriptor_upload;
  Buffer<__half> detection_descriptors;
  Buffer<unsigned long long> proposals;
  Buffer<int> detection_match, ranks, slot_detection, slot_track;
  Buffer<Counts> counts;
  Buffer<TrackRecord> records;
  Buffer<FlowVector> flow_upload[2];
  cudaEvent_t begin{}, end{};
  Counts* host_counts{};
  TrackRecord* host_records{};
  std::size_t host_capacity{};

  explicit Device(std::size_t capacity) {
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Create tracker stream");
    check(cudaEventCreate(&begin), "Create tracker timer");
    check(cudaEventCreate(&end), "Create tracker timer");
    for (int k = 0; k < 2; ++k) {
      tracks[k].reserve(capacity);
      descriptors[k].reserve(capacity * kDimension);
    }
    predictions.reserve(capacity);
    flags.reserve(capacity);
    fb_errors.reserve(capacity);
    track_match.reserve(capacity);
    candidate_counts.reserve(capacity);
    candidates.reserve(capacity * kCandidates);
    costs.reserve(capacity * kCandidates);
    similarities.reserve(capacity * kCandidates);
    slot_detection.reserve(capacity);
    slot_track.reserve(capacity);
    records.reserve(capacity);
    counts.reserve(1);
    check(cudaMallocHost(&host_counts, sizeof(Counts)), "Allocate pinned counts");
    check(cudaMallocHost(&host_records, capacity * sizeof(TrackRecord)), "Allocate pinned records");
    host_capacity = capacity;
  }
  ~Device() {
    cudaStreamSynchronize(stream);
    cudaFreeHost(host_counts);
    cudaFreeHost(host_records);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);
  }
};

FlowTracker::FlowTracker(FlowTrackerConfig config) : config_(config) {
  config_.validate();
  device_ = std::make_unique<Device>(config_.max_tracks);
}
FlowTracker::~FlowTracker() = default;

void FlowTracker::set_association_radius(float radius) {
  auto next = config_;
  next.association_radius = radius;
  next.validate();
  config_ = next;
}

void FlowTracker::associate(FrameFeatures& frame, const std::optional<FlowField>& flow) {
  if (!flow) return associate(frame, static_cast<const DeviceFlowField*>(nullptr));
  const std::size_t cells = static_cast<std::size_t>(std::max(flow->width, 0)) * std::max(flow->height, 0);
  if (flow->width <= 0 || flow->height <= 0 || flow->grid_size <= 0 || flow->vectors.size() != cells ||
      (!flow->backward.empty() && flow->backward.size() != cells)) {
    // A malformed field must not imply zero motion: treat as missing flow.
    return associate(frame, static_cast<const DeviceFlowField*>(nullptr));
  }
  auto& d = *device_;
  DeviceFlowField field;
  field.width = flow->width;
  field.height = flow->height;
  field.grid_size = flow->grid_size;
  d.flow_upload[0].reserve(cells);
  check(cudaMemcpyAsync(d.flow_upload[0].data, flow->vectors.data(), cells * sizeof(FlowVector),
                        cudaMemcpyHostToDevice, d.stream), "Upload forward flow");
  field.forward = d.flow_upload[0].data;
  if (!flow->backward.empty()) {
    d.flow_upload[1].reserve(cells);
    check(cudaMemcpyAsync(d.flow_upload[1].data, flow->backward.data(), cells * sizeof(FlowVector),
                          cudaMemcpyHostToDevice, d.stream), "Upload backward flow");
    field.backward = d.flow_upload[1].data;
  }
  associate(frame, &field);
}

void FlowTracker::associate(FrameFeatures& frame, const DeviceFlowField* flow,
                            const DeviceDetections* detections) {
  const auto started = std::chrono::steady_clock::now();
  if (frame.width <= 0 || frame.height <= 0) throw std::invalid_argument("Invalid flow frame size");
  for (const auto& point : frame.keypoints) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.score) ||
        point.x < 0 || point.y < 0 || point.x >= frame.width || point.y >= frame.height) {
      throw std::invalid_argument("Invalid SuperPoint coordinates or score");
    }
  }
  if (detections && detections->count != static_cast<int>(frame.keypoints.size())) {
    throw std::invalid_argument("Device detections do not match the frame keypoints");
  }
  if (!detections && frame.descriptors.size() != frame.keypoints.size() * kDimension) {
    throw std::invalid_argument("Tracking requires one 256-D descriptor per keypoint");
  }
  if (frame.keypoints.size() > INT_MAX / kDimension) throw std::invalid_argument("Too many detections");
  auto& d = *device_;
  const bool usable_flow = flow && flow->forward && flow->width > 0 && flow->height > 0 &&
                           flow->grid_size > 0;
  const bool consecutive = timestamp_ns_ && frame.frame_index == frame_index_ + 1 &&
      frame.timestamp_ns > *timestamp_ns_ && frame.width == width_ && frame.height == height_;
  if (!consecutive || !usable_flow) active_.clear();
  timestamp_ns_ = frame.timestamp_ns;
  frame_index_ = frame.frame_index;
  width_ = frame.width;
  height_ = frame.height;

  const int n = static_cast<int>(active_.size());
  const int m = static_cast<int>(frame.keypoints.size());
  const int capacity = static_cast<int>(config_.max_tracks);
  const cudaStream_t stream = d.stream;
  if (usable_flow && flow->ready_event) {
    check(cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(flow->ready_event), 0), "Wait for flow");
  }
  if (detections && detections->ready_event) {
    check(cudaStreamWaitEvent(stream, static_cast<cudaEvent_t>(detections->ready_event), 0),
          "Wait for detections");
  }

  check(cudaEventRecord(d.begin, stream), "Start tracker timer");
  // Detections: device view from SuperPoint, or upload of host data.
  const std::size_t detection_slots = std::max(m, 1);
  d.detection_descriptors.reserve(detection_slots * kDimension);
  d.proposals.reserve(detection_slots);
  d.detection_match.reserve(detection_slots);
  d.ranks.reserve(detection_slots);
  const Keypoint* points = nullptr;
  const float* descriptor_source = nullptr;
  if (detections) {
    points = detections->points;
    descriptor_source = detections->descriptors;
  } else if (m > 0) {
    d.detection_upload.reserve(m);
    d.descriptor_upload.reserve(static_cast<std::size_t>(m) * kDimension);
    check(cudaMemcpyAsync(d.detection_upload.data, frame.keypoints.data(), m * sizeof(Keypoint),
                          cudaMemcpyHostToDevice, stream), "Upload detections");
    check(cudaMemcpyAsync(d.descriptor_upload.data, frame.descriptors.data(),
                          static_cast<std::size_t>(m) * kDimension * sizeof(float),
                          cudaMemcpyHostToDevice, stream), "Upload descriptors");
    points = d.detection_upload.data;
    descriptor_source = d.descriptor_upload.data;
  }
  if (m > 0) {
    quantize<<<blocks(static_cast<std::size_t>(m) * kDimension, 256), 256, 0, stream>>>(
        descriptor_source, d.detection_descriptors.data, m * kDimension);
    check(cudaMemsetAsync(d.proposals.data, 0xff, m * sizeof(unsigned long long), stream), "Clear proposals");
    check(cudaMemsetAsync(d.detection_match.data, 0xff, m * sizeof(int), stream), "Clear detections");
  }
  const int current = d.current, next = 1 - d.current;
  if (n > 0) {
    check(cudaMemsetAsync(d.track_match.data, 0xff, n * sizeof(int), stream), "Clear tracks");
    predict<<<blocks(n, 128), 128, 0, stream>>>(d.tracks[current].data, n, flow->forward,
        flow->backward, flow->width, flow->height, flow->grid_size, frame.width, frame.height,
        config_.forward_backward_threshold, d.predictions.data, d.flags.data, d.fb_errors.data);
    const float radius2 = config_.association_radius * config_.association_radius;
    gather_candidates<<<blocks(static_cast<std::size_t>(n) * 32, 128), 128, 0, stream>>>(
        d.predictions.data, d.flags.data, n, d.descriptors[current].data, points,
        d.detection_descriptors.data, m, radius2, config_.min_descriptor_similarity,
        config_.descriptor_weight, d.candidates.data, d.costs.data, d.similarities.data,
        d.candidate_counts.data);
    for (int round = 0; round < config_.assignment_rounds && m > 0; ++round) {
      propose<<<blocks(n, 128), 128, 0, stream>>>(d.candidates.data, d.costs.data,
          d.candidate_counts.data, n, d.track_match.data, d.detection_match.data, d.proposals.data);
      accept<<<blocks(m, 128), 128, 0, stream>>>(d.proposals.data, m, d.track_match.data,
                                                d.detection_match.data);
    }
  }
  if (m > 0) {
    rank_unmatched<<<blocks(m, 128), 128, 0, stream>>>(points, d.detection_match.data, m, d.ranks.data);
  }
  layout<<<1, kBlock, 0, stream>>>(d.detection_match.data, d.ranks.data, m, d.track_match.data,
      d.flags.data, d.tracks[current].data, n, config_.max_coast_frames, capacity,
      d.slot_detection.data, d.slot_track.data, d.counts.data);
  build<<<blocks(capacity, 128), 128, 0, stream>>>(d.counts.data, d.slot_detection.data,
      d.slot_track.data, d.tracks[current].data, d.predictions.data, d.fb_errors.data, points,
      d.candidates.data, d.similarities.data, d.candidate_counts.data, capacity, next_id_,
      frame.frame_index, config_.detection_sigma_px * config_.detection_sigma_px,
      config_.flow_sigma_px > 0 ? config_.flow_sigma_px * config_.flow_sigma_px : 0.0F,
      d.tracks[next].data, d.records.data);
  copy_descriptors<<<blocks(static_cast<std::size_t>(capacity) * 32, 128), 128, 0, stream>>>(
      d.counts.data, d.slot_detection.data, d.slot_track.data, d.descriptors[current].data,
      d.detection_descriptors.data, capacity, d.descriptors[next].data);
  check(cudaGetLastError(), "Launch flow tracking kernels");
  // The only device -> host transfer: compact per-track records for display/stats.
  check(cudaMemcpyAsync(d.host_counts, d.counts.data, sizeof(Counts), cudaMemcpyDeviceToHost, stream),
        "Read track counts");
  check(cudaMemcpyAsync(d.host_records, d.records.data, capacity * sizeof(TrackRecord),
                        cudaMemcpyDeviceToHost, stream), "Read track records");
  check(cudaEventRecord(d.end, stream), "Stop tracker timer");
  check(cudaStreamSynchronize(stream), "Wait for flow tracking");
  float gpu_ms = 0;
  check(cudaEventElapsedTime(&gpu_ms, d.begin, d.end), "Read tracker timer");
  frame.tracking_gpu_ms = gpu_ms;
  d.current = next;

  const Counts counts = *d.host_counts;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> detection_descriptors = std::move(frame.descriptors);
  const bool have_descriptors = !detection_descriptors.empty() &&
      detection_descriptors.size() == frame.keypoints.size() * kDimension;
  frame.keypoints.clear();
  frame.descriptors.clear();
  frame.track_descriptors.clear();
  frame.landmark_ids.clear();
  frame.landmark_similarities.clear();
  frame.landmark_updates.clear();
  frame.track_confidences.clear();
  frame.superpoint_supported.clear();
  frame.flow_predictions.clear();
  frame.correction_distances.clear();
  frame.matched_landmarks = static_cast<std::uint32_t>(counts.matched);
  frame.coasted_landmarks = static_cast<std::uint32_t>(counts.coasted);
  frame.new_landmarks = static_cast<std::uint32_t>(counts.fresh);
  active_.clear();
  active_.reserve(counts.total);
  for (int slot = 0; slot < counts.total; ++slot) {
    const auto& record = d.host_records[slot];
    frame.keypoints.push_back({record.x, record.y, record.score});
    frame.landmark_ids.push_back(record.id);
    frame.landmark_similarities.push_back(record.similarity);
    frame.superpoint_supported.push_back(record.kind != kCoasted);
    frame.flow_predictions.push_back(record.kind == kNew ? Keypoint{nan, nan, 0} :
        Keypoint{record.prediction_x, record.prediction_y, 0});
    frame.correction_distances.push_back(record.correction);
    if (have_descriptors) {
      if (record.detection >= 0) {
        const auto row = detection_descriptors.begin() + static_cast<std::ptrdiff_t>(record.detection) * kDimension;
        frame.track_descriptors.insert(frame.track_descriptors.end(), row, row + kDimension);
      } else {
        frame.track_descriptors.resize(frame.track_descriptors.size() + kDimension, 0.0F);
      }
    }
    if (record.kind == kNew) {
      ++histogram_[track_length_bin(1)];
    } else if (record.kind == kMatched) {
      --histogram_[track_length_bin(record.length - 1)];
      ++histogram_[track_length_bin(record.length)];
    }
    active_.push_back({record.id, record.length, record.first_frame, record.last_frame});
  }
  next_id_ += static_cast<std::uint64_t>(counts.fresh);
  frame.tracking_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
}

const LandmarkState* FlowTracker::find(std::uint64_t id) const {
  for (const auto& track : active_) {
    if (track.id != id) continue;
    inspection_ = {};
    inspection_.id = track.id;
    inspection_.observation_count = track.observation_count;
    inspection_.first_frame = track.first_frame;
    inspection_.last_frame = track.last_frame;
    return &inspection_;
  }
  return nullptr;
}
}  // namespace slam_native
