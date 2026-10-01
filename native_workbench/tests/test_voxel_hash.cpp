// GPU voxel hash checks (built with the CUDA app; needs a GPU).
#include "slam_native/voxel_hash.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>

using namespace slam_native;
namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
bool close(float a, float b, float tolerance = 1e-4F) { return std::abs(a - b) <= tolerance; }

VoxelSample point(float x, float y, float z, int segment = 0) {
  VoxelSample s{};
  s.point[0] = x, s.point[1] = y, s.point[2] = z;
  s.segment = segment;
  s.weight = 1;
  return s;
}
VoxelSample ray(std::array<float, 3> origin, std::array<float, 3> target, int segment = 0) {
  VoxelSample s = point(target[0], target[1], target[2], segment);
  std::copy(origin.begin(), origin.end(), s.origin);
  s.flags = VoxelSample::has_ray;
  return s;
}
using Coord = std::tuple<int, int, int, int>;  // segment, x, y, z
std::map<Coord, VoxelInstance> by_coord(const std::vector<VoxelInstance>& voxels) {
  std::map<Coord, VoxelInstance> out;
  for (const auto& v : voxels)
    require(out.emplace(Coord{v.segment, v.coord[0], v.coord[1], v.coord[2]}, v).second, "A voxel appears twice");
  return out;
}

// Points without rays: one voxel per occupied cell, weight = points in it,
// colour = their mean; negative coordinates round down.
void occupancy() {
  VoxelHash hash({0.5F, 2.0F, 1.0F, 4096});
  std::vector<VoxelSample> samples{point(0.1F, 0.1F, 0.1F), point(0.4F, 0.2F, 0.3F), point(-0.1F, 0.1F, 0.1F),
                                   point(0.1F, 0.1F, 0.1F, 3)};
  samples[0].flags = samples[1].flags = VoxelSample::has_color;
  samples[0].color[0] = 100, samples[1].color[0] = 200, samples[1].color[2] = 50;
  hash.integrate(samples.data(), samples.size());
  const auto voxels = by_coord(hash.download());
  require(voxels.size() == 3 && hash.stats().voxels == 3 && hash.stats().rays == 0, "Three cells are occupied");
  const auto& both = voxels.at({0, 0, 0, 0});
  require(close(both.weight, 2) && close(both.distance, 0) && both.color[0] == 150 && both.color[2] == 25 &&
          both.color[3] == 255, "A voxel sums its points' weights and averages their colours");
  const auto& negative = voxels.at({0, -1, 0, 0});
  require(close(negative.weight, 1) && negative.color[3] == 0, "Negative coordinates round down; no colour");
  require(voxels.count({3, 0, 0, 0}) == 1, "Segments have their own voxels");
}

// One ray along x with unit voxels: the band is +/- 2 voxels around the
// point, with the signed distance of each voxel centre; a voxel on the way
// is carved, and free space allocates nothing.
void band_and_carving() {
  for (const float carve : {1.0F, 0.0F, 3.0F}) {
    VoxelHash hash({1.0F, 2.0F, carve, 4096});
    const std::vector<VoxelSample> samples{ray({0.5F, 0.5F, 0.5F}, {10.5F, 0.5F, 0.5F}), point(5.5F, 0.5F, 0.5F)};
    hash.integrate(samples.data(), samples.size());
    const auto voxels = by_coord(hash.download());
    require(voxels.size() == 6, "The band has five voxels; free space allocates none");
    for (int x = 8; x <= 12; ++x) {
      const auto& v = voxels.at({0, x, 0, 0});
      require(close(v.weight, 1) && close(v.distance, static_cast<float>(10 - x)),
              "Band voxels carry the signed distance to the point along the ray");
    }
    const auto& crossed = voxels.at({0, 5, 0, 0});
    // Its own point says distance 0 (weight 1); the ray says free: +2 voxels (weight carve).
    require(close(crossed.weight, 1 + carve) && close(crossed.distance, 2 * carve / (1 + carve)),
            "A ray through an occupied voxel carves it by the carve weight");
  }
  // The walk is capped from the band toward the camera.
  VoxelHash capped({1.0F, 2.0F, 1.0F, 2});
  const std::vector<VoxelSample> samples{ray({0.5F, 0.5F, 0.5F}, {10.5F, 0.5F, 0.5F}), point(6.5F, 0.5F, 0.5F),
                                         point(5.5F, 0.5F, 0.5F)};
  capped.integrate(samples.data(), samples.size());
  const auto voxels = by_coord(capped.download());
  require(close(voxels.at({0, 6, 0, 0}).weight, 2) && close(voxels.at({0, 5, 0, 0}).weight, 1),
          "A capped ray carves the voxels next to the band first");
}

// Random rays: every band voxel touches its ray's band segment, the voxel at
// the point is there with a distance below half a diagonal, and each ray
// contributes a connected run of voxels.
void oblique_rays() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> position(-20.0F, 20.0F);
  const float voxel = 0.25F, truncation = 2.0F;
  std::vector<VoxelSample> samples;
  for (int i = 0; i < 400; ++i)
    samples.push_back(ray({position(rng), position(rng), position(rng)}, {position(rng), position(rng), position(rng)}, i));
  VoxelHash hash({voxel, truncation, 0.0F, 4096});
  hash.integrate(samples.data(), samples.size());
  const auto voxels = hash.download();
  const auto map = by_coord(voxels);
  std::vector<int> per_ray(samples.size(), 0);
  for (const auto& v : voxels) {
    const auto& s = samples[static_cast<std::size_t>(v.segment)];
    double o[3], d[3], depth = 0;
    for (int a = 0; a < 3; ++a) {
      o[a] = s.origin[a] / voxel;
      d[a] = s.point[a] / voxel - o[a];
      depth += d[a] * d[a];
    }
    depth = std::sqrt(depth);
    // Slab test of the voxel's cube against the band segment.
    double enter = std::max(0.0, depth - truncation), leave = depth + truncation;
    for (int a = 0; a < 3; ++a) {
      const double direction = d[a] / depth, low = v.coord[a] - 1e-3, high = v.coord[a] + 1 + 1e-3;
      if (std::abs(direction) < 1e-12) {
        require(o[a] >= low && o[a] <= high, "A band voxel must touch its ray");
        continue;
      }
      const double t0 = (low - o[a]) / direction, t1 = (high - o[a]) / direction;
      enter = std::max(enter, std::min(t0, t1));
      leave = std::min(leave, std::max(t0, t1));
    }
    require(enter <= leave, "A band voxel must touch its ray");
    ++per_ray[static_cast<std::size_t>(v.segment)];
  }
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto& s = samples[i];
    const Coord at{static_cast<int>(i), static_cast<int>(std::floor(s.point[0] / voxel)),
                   static_cast<int>(std::floor(s.point[1] / voxel)), static_cast<int>(std::floor(s.point[2] / voxel))};
    require(map.count(at) == 1 && std::abs(map.at(at).distance) <= 0.87F, "The voxel at the point is on the surface");
    // A segment of 4 voxels crosses at least 4 and at most 3 * 4 + 1 voxels.
    require(per_ray[i] >= 4 && per_ray[i] <= 13, "A ray's band is a short connected run of voxels");
  }
}

// Many points over several calls: the table grows (rehash) without losing or
// duplicating voxels, and resetting empties it.
void growth() {
  std::mt19937 rng(11);
  std::uniform_int_distribution<int> cell(-60, 60);
  std::map<Coord, int> reference;
  VoxelHash hash({0.1F, 2.0F, 1.0F, 4096});
  const std::size_t initial_capacity = hash.stats().capacity;
  for (int batch = 0; batch < 4; ++batch) {
    std::vector<VoxelSample> samples;
    for (int i = 0; i < 60'000; ++i) {
      const int x = cell(rng), y = cell(rng), z = cell(rng);
      samples.push_back(point((x + 0.5F) * 0.1F, (y + 0.5F) * 0.1F, (z + 0.5F) * 0.1F, batch % 2));
      ++reference[{batch % 2, x, y, z}];
    }
    hash.integrate(samples.data(), samples.size());
  }
  const auto voxels = by_coord(hash.download());
  require(hash.stats().capacity > std::max<std::size_t>(initial_capacity, 2 * reference.size() - 1),
          "The table grows to keep its load below one half");
  require(voxels.size() == reference.size() && hash.stats().voxels == reference.size() &&
          hash.stats().samples == 240'000, "Every occupied cell has exactly one voxel after growing");
  for (const auto& [coord, count] : reference)
    require(close(voxels.at(coord).weight, static_cast<float>(count)), "Weights survive rehashing");
  hash.reset({0.1F, 2.0F, 1.0F, 4096});
  require(hash.download().empty() && hash.stats().voxels == 0, "Reset empties the table");
}

// Ambient visibility. A 41 x 41 floor of voxels (z = 0) with one voxel standing
// on it. Known values for cosine-distributed rays: an open face sees
// everything; a face against a solid voxel nothing; the floor voxel next to
// the box loses the form factor between two unit squares sharing an edge at a
// right angle (0.2); the box's side sees the floor in half of its hemisphere.
void ambient_visibility() {
  std::vector<VoxelSample> samples;
  for (int x = -20; x <= 20; ++x)
    for (int y = -20; y <= 20; ++y) samples.push_back(point(x + 0.5F, y + 0.5F, 0.5F));
  samples.push_back(point(0.5F, 0.5F, 1.5F));
  VoxelShadeConfig shade;
  shade.opaque_weight = 1;  // each voxel holds one point of weight 1
  shade.max_distance = 16;
  const VoxelHashConfig grid{1.0F, 2.0F, 1.0F, 4096};
  const auto visibility = [](const VoxelInstance& v, int axis, bool positive) {
    return (positive ? v.visibility_positive : v.visibility_negative)[axis] / 255.0F;
  };
  VoxelHash hash(grid);
  hash.integrate(samples.data(), samples.size());
  {
    const auto before = by_coord(hash.download());
    require(visibility(before.at({0, 1, 0, 0}), 2, true) == 1.0F, "Unshaded voxels are fully lit");
  }
  for (int pass = 0; pass < 600; ++pass) hash.shade(shade);
  const auto voxels = by_coord(hash.download());
  const auto& box = voxels.at({0, 0, 0, 1});
  const auto& beside = voxels.at({0, 1, 0, 0});
  const auto& under = voxels.at({0, 0, 0, 0});
  const auto& open = voxels.at({0, 10, 10, 0});
  require(visibility(open, 2, true) == 1.0F && visibility(box, 2, true) == 1.0F, "Nothing above: fully visible");
  require(visibility(under, 2, true) == 0.0F && visibility(box, 2, false) == 0.0F &&
          visibility(open, 0, true) == 0.0F, "A face against a solid voxel sees nothing");
  require(close(visibility(beside, 2, true), 0.8F, 0.02F),
          "The floor next to the box loses the form factor of the box's side (0.2)");
  // Half the hemisphere looks down; the floor ends 20 voxels away and the
  // rays 16, so a little more than half gets out.
  for (int axis = 0; axis < 2; ++axis)
    for (const bool positive : {false, true})
      require(visibility(box, axis, positive) > 0.5F && visibility(box, axis, positive) < 0.56F,
              "The box's sides see the floor in half of their hemisphere");

  // A rebuild keeps the averages: one more pass moves them by at most 1 / history.
  hash.reset(grid);
  hash.integrate(samples.data(), samples.size());
  hash.shade(shade);
  const auto rebuilt = by_coord(hash.download());
  require(close(visibility(rebuilt.at({0, 1, 0, 0}), 2, true), 0.8F, 0.04F),
          "Visibility averages survive a rebuild of the same voxels");
  // A changed grid does not: the single first pass of 8 rays gives multiples of 1/8.
  const VoxelHashConfig other{0.5F, 2.0F, 1.0F, 4096};
  hash.reset(other);
  hash.integrate(samples.data(), samples.size());
  hash.shade(shade);
  for (const auto& v : hash.download())
    for (int axis = 0; axis < 3; ++axis) {
      const float eighths = 8.0F * visibility(v, axis, true);
      require(close(eighths, std::round(eighths), 0.02F), "A new voxel grid starts its averages again");
    }

  // Light voxels block in proportion to their weight; hidden ones not at all.
  VoxelHash pair(grid);
  const std::vector<VoxelSample> two{point(0.5F, 0.5F, 0.5F), point(1.5F, 0.5F, 0.5F)};
  pair.integrate(two.data(), two.size());
  shade.opaque_weight = 2;
  for (int pass = 0; pass < 20; ++pass) pair.shade(shade);
  require(close(visibility(by_coord(pair.download()).at({0, 0, 0, 0}), 0, true), 0.5F, 0.01F),
          "A voxel of half the opaque weight blocks half of each ray");
  shade.min_weight = 1.5F;  // neither voxel is drawn now
  for (int pass = 0; pass < 40; ++pass) pair.shade(shade, pass == 0);
  require(close(visibility(by_coord(pair.download()).at({0, 0, 0, 0}), 0, true), 0.5F, 0.01F),
          "Voxels that are not drawn are not shaded");
  shade.min_weight = 0;
  shade.band = 0.75F;
  shade.opaque_weight = 1e6F;  // nearly transparent
  for (int pass = 0; pass < 100; ++pass) pair.shade(shade, pass == 0);
  require(visibility(by_coord(pair.download()).at({0, 0, 0, 0}), 0, true) > 0.95F,
          "After a restart the averages follow the new solid set within a few passes");
}

void rejected_input() {
  VoxelHash hash({0.01F, 2.0F, 1.0F, 4096});
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<VoxelSample> samples{point(nan, 0, 0), point(1e9F, 0, 0), point(0, 0, 0, -1), point(0, 0, 0, 5000),
                                   ray({nan, 0, 0}, {0.005F, 0.005F, 0.005F}),
                                   ray({1e7F, 0, 0.005F}, {1.005F, 0.005F, 0.005F})};
  hash.integrate(samples.data(), samples.size());
  const auto voxels = by_coord(hash.download());
  require(hash.stats().out_of_range == 4, "Non-finite, far and bad-segment samples are counted and skipped");
  require(voxels.count({0, 0, 0, 0}) == 1, "A sample with an unusable camera keeps its point");
  require(voxels.count({0, 100, 0, 0}) == 1 && voxels.size() == 2,
          "A camera too far away to express in voxels leaves the point without a ray");
  bool rejected = false;
  try { VoxelHash bad({0.0F, 2.0F, 1.0F, 4096}); } catch (const std::exception&) { rejected = true; }
  require(rejected, "A zero voxel size is rejected");
}
}  // namespace

int main() {
  try {
    occupancy();
    band_and_carving();
    oblique_rays();
    growth();
    ambient_visibility();
    rejected_input();
  } catch (const std::exception& error) {
    std::cerr << "voxel hash test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "voxel hash tests passed\n";
  return 0;
}
