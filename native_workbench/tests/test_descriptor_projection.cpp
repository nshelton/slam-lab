#include "slam_native/descriptor_projection.hpp"
#include "slam_native/online_tracker.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>

using namespace slam_native;

static_assert(track_length_bin(1) == 0);
static_assert(track_length_bin(16) == 15);
static_assert(track_length_bin(17) == 16);
static_assert(track_length_bin(32) == 16);
static_assert(track_length_bin(33) == 17);
static_assert(track_length_bin(2049) == 23);

int main() {
  FrameFeatures frame;
  constexpr std::size_t count = 2048;
  frame.descriptors.resize(count * 256);
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t offset = index * 256;
    frame.descriptors[offset] = index < count / 2 ? 1.0F : -1.0F;
    frame.descriptors[offset + 1] = 0.2F * std::sin(static_cast<float>(index));
    frame.descriptors[offset + 2] = 0.1F * std::cos(static_cast<float>(index) * 0.7F);
  }
  DescriptorProjection projection;
  projection.observe(frame);
  assert(projection.ready());
  assert(projection.sample_count() == count);
  const auto points = projection.project(frame);
  assert(points.size() == count);
  double first_mean = 0;
  double second_mean = 0;
  for (std::size_t index = 0; index < count; ++index) {
    assert(std::isfinite(points[index].x));
    assert(std::isfinite(points[index].y));
    if (index < count / 2) first_mean += points[index].x;
    else second_mean += points[index].x;
  }
  first_mean /= count / 2;
  second_mean /= count / 2;
  assert(std::abs(first_mean - second_mean) > 0.5);
}
