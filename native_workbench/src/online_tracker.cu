#include "slam_native/online_tracker.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace slam_native {
namespace {

constexpr int kDimension = 256;

void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

void cublas_check(cublasStatus_t result, const char* operation) {
  if (result != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                             std::to_string(static_cast<int>(result)));
  }
}

struct Candidate {
  int index;
  float best;
  float second;
};

__device__ void insert_score(float score, int index, float& best,
                             int& best_index, float& second) {
  if (index < 0) return;
  if (score > best || (score == best && index < best_index)) {
    second = best;
    best = score;
    best_index = index;
  } else if (index != best_index && score > second) {
    second = score;
  }
}

__global__ void reduce_scores(const float* scores, Candidate* output,
                              int landmark_count, int feature_count) {
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int feature = thread / 32;
  const int lane = thread & 31;
  if (feature >= feature_count) return;
  float best = -CUDART_INF_F;
  float second = -CUDART_INF_F;
  int best_index = -1;
  for (int landmark = lane; landmark < landmark_count; landmark += 32) {
    insert_score(scores[static_cast<std::size_t>(feature) * landmark_count + landmark],
                 landmark, best, best_index, second);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    const float other_best = __shfl_down_sync(0xffffffff, best, offset);
    const float other_second = __shfl_down_sync(0xffffffff, second, offset);
    const int other_index = __shfl_down_sync(0xffffffff, best_index, offset);
    if (lane + offset < 32) {
      insert_score(other_best, other_index, best, best_index, second);
      if (other_second > second) second = other_second;
    }
  }
  if (lane == 0) output[feature] = {best_index, best, second};
}

float normalize(const float* source, float* target) {
  double squared = 0;
  for (int component = 0; component < kDimension; ++component) {
    const float value = source[component];
    if (!std::isfinite(value)) {
      std::fill_n(target, kDimension, 0.0F);
      return 0.0F;
    }
    squared += static_cast<double>(value) * value;
  }
  const float length = static_cast<float>(std::sqrt(squared));
  if (length <= 0) {
    std::fill_n(target, kDimension, 0.0F);
    return 0.0F;
  }
  for (int component = 0; component < kDimension; ++component) {
    target[component] = source[component] / length;
  }
  return length;
}

float mean_direction(const LandmarkState& landmark, float* target) {
  return normalize(landmark.resultant.data(), target);
}

float concentration(const LandmarkState& landmark) {
  double squared = 0;
  for (float value : landmark.resultant) squared += static_cast<double>(value) * value;
  return static_cast<float>(std::sqrt(squared) / landmark.observation_count);
}

}  // namespace

struct OnlineTracker::Gpu {
  cublasHandle_t handle{};
  cudaStream_t stream{};
  float* query{};
  float* means{};
  float* scores{};
  Candidate* candidates{};
  std::size_t query_capacity{};
  std::size_t mean_capacity{};
  std::size_t score_capacity{};

  Gpu() {
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create tracker stream");
    cublas_check(cublasCreate(&handle), "create cuBLAS handle");
    cublas_check(cublasSetStream(handle, stream), "set cuBLAS stream");
  }

  ~Gpu() {
    if (candidates) cudaFree(candidates);
    if (scores) cudaFree(scores);
    if (means) cudaFree(means);
    if (query) cudaFree(query);
    if (handle) cublasDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
  }

  template <typename T>
  static void ensure(T*& pointer, std::size_t& capacity, std::size_t count) {
    if (count <= capacity) return;
    if (pointer) cuda_check(cudaFree(pointer), "resize tracker allocation");
    capacity = std::max(count, std::max<std::size_t>(capacity * 2, 1));
    cuda_check(cudaMalloc(&pointer, capacity * sizeof(T)), "allocate tracker buffer");
  }

  std::vector<Candidate> search(const std::vector<float>& descriptors,
                                const std::vector<float>& directions,
                                int feature_count, int landmark_count) {
    ensure(query, query_capacity, descriptors.size());
    ensure(means, mean_capacity, directions.size());
    ensure(scores, score_capacity,
           static_cast<std::size_t>(feature_count) * landmark_count);
    if (candidate_capacity < static_cast<std::size_t>(feature_count)) {
      if (candidates) cuda_check(cudaFree(candidates), "resize tracker candidates");
      candidate_capacity = std::max(static_cast<std::size_t>(feature_count),
                                    candidate_capacity * 2);
      cuda_check(cudaMalloc(&candidates, candidate_capacity * sizeof(Candidate)),
                 "allocate tracker candidates");
    }
    cuda_check(cudaMemcpyAsync(query, descriptors.data(), descriptors.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream), "upload descriptors");
    cuda_check(cudaMemcpyAsync(means, directions.data(), directions.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream), "upload landmark means");
    const float alpha = 1.0F;
    const float beta = 0.0F;
    cublas_check(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            landmark_count, feature_count, kDimension,
                            &alpha, means, kDimension, query, kDimension,
                            &beta, scores, landmark_count), "compute cosine similarities");
    reduce_scores<<<(feature_count * 32 + 127) / 128, 128, 0, stream>>>(
        scores, candidates, landmark_count, feature_count);
    cuda_check(cudaGetLastError(), "reduce cosine similarities");
    std::vector<Candidate> result(feature_count);
    cuda_check(cudaMemcpyAsync(result.data(), candidates,
                               result.size() * sizeof(Candidate),
                               cudaMemcpyDeviceToHost, stream), "read match candidates");
    cuda_check(cudaStreamSynchronize(stream), "wait for landmark association");
    return result;
  }

  std::size_t candidate_capacity{};
};

OnlineTracker::OnlineTracker(TrackerConfig config, std::vector<LandmarkState> active,
                             std::uint64_t next_id, TrackLengthHistogram histogram)
    : config_(config), active_(std::move(active)), next_id_(next_id),
      histogram_(histogram), gpu_(std::make_unique<Gpu>()) {
  if (config_.min_similarity < -1 || config_.min_similarity > 1 ||
      config_.min_margin < 0 || config_.min_margin > 2 ||
      config_.max_inactive_frames == 0) {
    throw std::invalid_argument("Invalid online tracker settings");
  }
}

OnlineTracker::~OnlineTracker() = default;

void OnlineTracker::associate(FrameFeatures& frame) {
  const auto started = std::chrono::steady_clock::now();
  const std::size_t count = frame.keypoints.size();
  if (frame.descriptors.size() != count * kDimension) {
    throw std::invalid_argument("Tracker expected N x 256 descriptors");
  }
  active_.erase(std::remove_if(active_.begin(), active_.end(), [&](const LandmarkState& state) {
                  return frame.frame_index < state.last_frame ||
                         frame.frame_index - state.last_frame > config_.max_inactive_frames;
                }), active_.end());
  frame.landmark_ids.resize(count);
  frame.landmark_similarities.assign(count, std::numeric_limits<float>::quiet_NaN());
  frame.landmark_updates.clear();
  frame.landmark_updates.reserve(count);
  frame.new_landmarks = 0;
  frame.matched_landmarks = 0;
  if (count == 0) {
    frame.tracking_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return;
  }

  std::vector<float> descriptors(count * kDimension);
  std::vector<bool> valid(count);
  for (std::size_t feature = 0; feature < count; ++feature) {
    valid[feature] = normalize(frame.descriptors.data() + feature * kDimension,
                               descriptors.data() + feature * kDimension) > 0;
  }
  const std::size_t old_count = active_.size();
  std::vector<Candidate> candidates(count, {-1, -std::numeric_limits<float>::infinity(),
                                             -std::numeric_limits<float>::infinity()});
  if (old_count > 0) {
    std::vector<float> means(old_count * kDimension);
    for (std::size_t landmark = 0; landmark < old_count; ++landmark) {
      mean_direction(active_[landmark], means.data() + landmark * kDimension);
    }
    candidates = gpu_->search(descriptors, means, static_cast<int>(count),
                               static_cast<int>(old_count));
  }

  std::vector<std::size_t> order;
  order.reserve(count);
  for (std::size_t feature = 0; feature < count; ++feature) {
    const auto& match = candidates[feature];
    if (!valid[feature] || match.index < 0 ||
        match.best < config_.min_similarity ||
        match.best - match.second < config_.min_margin) continue;
    order.push_back(feature);
  }
  std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    if (candidates[left].best != candidates[right].best) {
      return candidates[left].best > candidates[right].best;
    }
    return left < right;
  });
  std::vector<int> assignments(count, -1);
  std::vector<bool> used(old_count);
  for (std::size_t feature : order) {
    const int index = candidates[feature].index;
    if (!used[index]) {
      assignments[feature] = index;
      used[index] = true;
    }
  }
  for (std::size_t feature = 0; feature < count; ++feature) {
    const float* descriptor = descriptors.data() + feature * kDimension;
    LandmarkState* state;
    if (assignments[feature] >= 0) {
      state = &active_[assignments[feature]];
      --histogram_[track_length_bin(state->observation_count)];
      for (int component = 0; component < kDimension; ++component) {
        state->resultant[component] += descriptor[component];
      }
      ++state->observation_count;
      ++histogram_[track_length_bin(state->observation_count)];
      state->last_frame = frame.frame_index;
      state->concentration = concentration(*state);
      frame.landmark_similarities[feature] = candidates[feature].best;
      ++frame.matched_landmarks;
    } else {
      LandmarkState created;
      created.id = next_id_++;
      std::copy_n(descriptor, kDimension, created.resultant.begin());
      created.observation_count = 1;
      created.first_frame = frame.frame_index;
      created.last_frame = frame.frame_index;
      created.concentration = valid[feature] ? 1.0F : 0.0F;
      active_.push_back(created);
      state = &active_.back();
      ++histogram_[track_length_bin(1)];
      ++frame.new_landmarks;
    }
    frame.landmark_ids[feature] = state->id;
    frame.landmark_updates.push_back(*state);
  }
  frame.tracking_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
}

std::size_t OnlineTracker::active_count() const { return active_.size(); }
std::uint64_t OnlineTracker::landmark_count() const { return next_id_; }
const LandmarkState* OnlineTracker::find(std::uint64_t id) const {
  const auto found = std::find_if(active_.begin(), active_.end(),
                                  [id](const LandmarkState& state) { return state.id == id; });
  return found == active_.end() ? nullptr : &*found;
}

const TrackLengthHistogram& OnlineTracker::length_histogram() const {
  return histogram_;
}

}  // namespace slam_native
