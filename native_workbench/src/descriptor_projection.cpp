#include "slam_native/descriptor_projection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace slam_native {
namespace {

constexpr std::size_t kDimension = 256;
constexpr std::size_t kSampleLimit = 2048;

float quantile_extent(std::vector<float>& values) {
  std::sort(values.begin(), values.end());
  const std::size_t lower = values.size() / 100;
  const std::size_t upper = values.size() - lower - 1;
  return std::max(0.05F,
      std::max(std::abs(values[lower]), std::abs(values[upper])) * 1.2F);
}

}  // namespace

void DescriptorProjection::observe(const FrameFeatures& frame) {
  if (ready_) return;
  ++observed_frames_;
  const std::size_t count = frame.descriptors.size() / kDimension;
  samples_.reserve(kSampleLimit * kDimension);
  for (std::size_t feature = 0; feature < count && sample_count() < kSampleLimit; ++feature) {
    const float* source = frame.descriptors.data() + feature * kDimension;
    double squared = 0;
    bool finite = true;
    for (std::size_t component = 0; component < kDimension; ++component) {
      finite &= std::isfinite(source[component]);
      squared += static_cast<double>(source[component]) * source[component];
    }
    if (!finite || squared <= 0) continue;
    const float inverse = static_cast<float>(1.0 / std::sqrt(squared));
    for (std::size_t component = 0; component < kDimension; ++component) {
      samples_.push_back(source[component] * inverse);
    }
  }
  if (sample_count() >= kSampleLimit ||
      (observed_frames_ >= 10 && sample_count() >= 256)) fit();
}

void DescriptorProjection::fit() {
  const std::size_t count = sample_count();
  if (count < 2) return;
  for (std::size_t sample = 0; sample < count; ++sample) {
    for (std::size_t component = 0; component < kDimension; ++component) {
      mean_[component] += samples_[sample * kDimension + component];
    }
  }
  for (float& value : mean_) value /= static_cast<float>(count);

  auto component = [&](std::array<float, kDimension>& axis,
                       const std::array<float, kDimension>* previous,
                       float phase) {
    for (std::size_t index = 0; index < kDimension; ++index) {
      axis[index] = std::sin(static_cast<float>(index + 1) * phase);
    }
    for (int iteration = 0; iteration < 12; ++iteration) {
      std::array<double, kDimension> next{};
      for (std::size_t sample = 0; sample < count; ++sample) {
        const float* row = samples_.data() + sample * kDimension;
        double projection = 0;
        for (std::size_t index = 0; index < kDimension; ++index) {
          projection += static_cast<double>(row[index] - mean_[index]) * axis[index];
        }
        for (std::size_t index = 0; index < kDimension; ++index) {
          next[index] += (row[index] - mean_[index]) * projection;
        }
      }
      if (previous) {
        double overlap = 0;
        for (std::size_t index = 0; index < kDimension; ++index) {
          overlap += next[index] * (*previous)[index];
        }
        for (std::size_t index = 0; index < kDimension; ++index) {
          next[index] -= overlap * (*previous)[index];
        }
      }
      double squared = 0;
      for (double value : next) squared += value * value;
      if (squared <= 0) break;
      const double inverse = 1.0 / std::sqrt(squared);
      for (std::size_t index = 0; index < kDimension; ++index) {
        axis[index] = static_cast<float>(next[index] * inverse);
      }
    }
  };
  component(first_, nullptr, 0.37F);
  component(second_, &first_, 0.83F);

  std::vector<float> first_values;
  std::vector<float> second_values;
  first_values.reserve(count);
  second_values.reserve(count);
  for (std::size_t sample = 0; sample < count; ++sample) {
    const float* row = samples_.data() + sample * kDimension;
    float x = 0;
    float y = 0;
    for (std::size_t index = 0; index < kDimension; ++index) {
      const float centered = row[index] - mean_[index];
      x += centered * first_[index];
      y += centered * second_[index];
    }
    first_values.push_back(x);
    second_values.push_back(y);
  }
  first_extent_ = quantile_extent(first_values);
  second_extent_ = quantile_extent(second_values);
  fitted_sample_count_ = count;
  ready_ = true;
  samples_.clear();
  samples_.shrink_to_fit();
}

std::vector<ProjectedDescriptor> DescriptorProjection::project(
    const FrameFeatures& frame) const {
  std::vector<ProjectedDescriptor> projected;
  if (!ready_) return projected;
  const std::size_t count = frame.descriptors.size() / kDimension;
  projected.resize(count);
  for (std::size_t sample = 0; sample < count; ++sample) {
    const float* row = frame.descriptors.data() + sample * kDimension;
    double squared = 0;
    for (std::size_t index = 0; index < kDimension; ++index) {
      squared += static_cast<double>(row[index]) * row[index];
    }
    if (squared <= 0 || !std::isfinite(squared)) continue;
    const float inverse = static_cast<float>(1.0 / std::sqrt(squared));
    float x = 0;
    float y = 0;
    for (std::size_t index = 0; index < kDimension; ++index) {
      const float centered = row[index] * inverse - mean_[index];
      x += centered * first_[index];
      y += centered * second_[index];
    }
    projected[sample] = {x / first_extent_, y / second_extent_};
  }
  return projected;
}

}  // namespace slam_native
