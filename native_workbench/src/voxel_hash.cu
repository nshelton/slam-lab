#include "slam_native/voxel_hash.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace slam_native {
namespace {

using Key = unsigned long long;  // the type atomicCAS takes

constexpr Key kEmpty = ~0ULL;
constexpr int kCoordBits = 18;
constexpr int kCoordBias = 1 << (kCoordBits - 1);  // voxel coordinates in [-2^17, 2^17)
constexpr int kMaxSegment = 1022;                  // 10 bits; 1023 with all-ones coordinates would be kEmpty
constexpr int kMaxProbes = 128;
constexpr std::size_t kMinCapacity = std::size_t{1} << 16;
constexpr std::size_t kMaxCapacity = std::size_t{1} << 24;  // 512 MB of table
constexpr int kBlock = 256;

enum Counter { inserted, failed, out_of_range, emitted, solid, counter_count };

// Per-voxel sums. The signed distance is kept as a fraction of the truncation
// distance (-1 .. 1), weighted by the sample weights.
struct Cell {
  float weight, distance, color_weight, r, g, b;
};

// Running Monte Carlo average of each face's ambient visibility (0 .. 65535 =
// blocked .. open), in the order -x, -y, -z, +x, +y, +z, and the number of
// samples behind it (0: never shaded).
struct Light {
  std::uint16_t visibility[6];
  std::uint16_t count;
  std::uint16_t unused;
};

constexpr std::uint16_t kRestartCount = 4;  // samples an average keeps when the solid set changed

struct Table {
  Key* keys;
  Cell* cells;
  Light* light;
  unsigned mask;  // capacity - 1 (a power of two)
};

void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

__device__ bool pack(int segment, int x, int y, int z, Key* key) {
  if (x < -kCoordBias || x >= kCoordBias || y < -kCoordBias || y >= kCoordBias || z < -kCoordBias ||
      z >= kCoordBias)
    return false;
  *key = (static_cast<Key>(segment) << (3 * kCoordBits)) | (static_cast<Key>(x + kCoordBias) << (2 * kCoordBits)) |
         (static_cast<Key>(y + kCoordBias) << kCoordBits) | static_cast<Key>(z + kCoordBias);
  return true;
}

__device__ void unpack(Key key, int* segment, int* cell) {
  const unsigned coord_mask = (1U << kCoordBits) - 1;
  cell[0] = static_cast<int>((key >> (2 * kCoordBits)) & coord_mask) - kCoordBias;
  cell[1] = static_cast<int>((key >> kCoordBits) & coord_mask) - kCoordBias;
  cell[2] = static_cast<int>(key & coord_mask) - kCoordBias;
  *segment = static_cast<int>(key >> (3 * kCoordBits));
}

__device__ Key mix(Key key) {  // MurmurHash3 finalizer
  key ^= key >> 33;
  key *= 0xff51afd7ed558ccdULL;
  key ^= key >> 33;
  key *= 0xc4ceb9fe1a85ec53ULL;
  key ^= key >> 33;
  return key;
}

__device__ unsigned home_slot(Key key, unsigned mask) { return static_cast<unsigned>(mix(key)) & mask; }

// Slot of the key, claiming an empty one if it is new; -1 when the probe
// sequence is exhausted (the caller grows the table and retries). Keys are
// never removed, so two threads inserting the same key meet in the same slot.
__device__ int insert(const Table& table, Key key, unsigned* counters) {
  unsigned slot = home_slot(key, table.mask);
  for (int probe = 0; probe < kMaxProbes; ++probe) {
    Key seen = table.keys[slot];
    if (seen == kEmpty) {
      seen = atomicCAS(&table.keys[slot], kEmpty, key);
      if (seen == kEmpty) {
        atomicAdd(&counters[inserted], 1U);
        return static_cast<int>(slot);
      }
    }
    if (seen == key) return static_cast<int>(slot);
    slot = (slot + 1) & table.mask;
  }
  atomicAdd(&counters[failed], 1U);
  return -1;
}

__device__ int find(const Table& table, Key key) {
  unsigned slot = home_slot(key, table.mask);
  for (int probe = 0; probe < kMaxProbes; ++probe) {
    const Key seen = table.keys[slot];
    if (seen == key) return static_cast<int>(slot);
    if (seen == kEmpty) return -1;
    slot = (slot + 1) & table.mask;
  }
  return -1;
}

// Voxel traversal along a ray (Amanatides & Woo), in voxel units.
struct Walker {
  int cell[3];
  int step[3];
  float next[3];   // ray parameter at which the ray leaves the current voxel, per axis
  float delta[3];  // parameter length of one voxel, per axis

  __device__ void start(const float* origin, const float* direction) {
    for (int a = 0; a < 3; ++a) {
      cell[a] = static_cast<int>(floorf(origin[a]));
      if (direction[a] > 0) {
        step[a] = 1;
        next[a] = (static_cast<float>(cell[a] + 1) - origin[a]) / direction[a];
        delta[a] = 1.0F / direction[a];
      } else if (direction[a] < 0) {
        step[a] = -1;
        next[a] = (static_cast<float>(cell[a]) - origin[a]) / direction[a];
        delta[a] = -1.0F / direction[a];
      } else {
        step[a] = 0;
        next[a] = delta[a] = INFINITY;
      }
    }
  }
  // Ray parameter at which the ray leaves the current voxel.
  __device__ float exit() const { return fminf(next[0], fminf(next[1], next[2])); }
  __device__ void advance() {
    const int a = next[0] <= next[1] ? (next[0] <= next[2] ? 0 : 2) : (next[1] <= next[2] ? 1 : 2);
    cell[a] += step[a];
    next[a] += delta[a];
  }
};

// A sample in voxel units.
struct Ray {
  float origin[3];     // camera centre
  float direction[3];  // unit, toward the point
  float point[3];
  float depth;         // camera to point
  bool has_ray;
};

// False when the sample cannot be placed (non-finite, outside the key range,
// bad segment).
__device__ bool prepare(const VoxelSample& sample, float inverse_voxel, Ray* ray) {
  if (sample.segment < 0 || sample.segment > kMaxSegment || !(sample.weight > 0)) return false;
  const float limit = static_cast<float>(kCoordBias - 1);
  for (int a = 0; a < 3; ++a) {
    ray->point[a] = sample.point[a] * inverse_voxel;
    if (!(fabsf(ray->point[a]) < limit)) return false;  // also rejects NaN
  }
  ray->has_ray = false;
  ray->depth = 0;
  if (!(sample.flags & VoxelSample::has_ray)) return true;
  float squared = 0;
  for (int a = 0; a < 3; ++a) {
    ray->origin[a] = sample.origin[a] * inverse_voxel;
    ray->direction[a] = ray->point[a] - ray->origin[a];
    squared += ray->direction[a] * ray->direction[a];
  }
  // A camera too far away to express in voxels: keep the point, drop the ray.
  if (!(squared > 1e-12F) || !(squared < 1e12F)) return true;
  ray->depth = sqrtf(squared);
  for (int a = 0; a < 3; ++a) ray->direction[a] /= ray->depth;
  ray->has_ray = true;
  return true;
}

// The band of +/- truncation around the point, along the ray.
struct Band {
  Walker walker;
  float length;
  int max_steps;

  __device__ Band(const Ray& ray, float truncation) {
    const float begin = fmaxf(0.0F, ray.depth - truncation);
    float start[3];
    for (int a = 0; a < 3; ++a) start[a] = ray.origin[a] + begin * ray.direction[a];
    walker.start(start, ray.direction);
    length = ray.depth + truncation - begin;
    max_steps = static_cast<int>(3.0F * length) + 4;  // a ray enters at most 3 voxels per unit length
  }
};

__global__ void allocate_kernel(const VoxelSample* samples, int count, Table table, float inverse_voxel,
                                float truncation, unsigned* counters) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const VoxelSample sample = samples[index];
  Ray ray;
  if (!prepare(sample, inverse_voxel, &ray)) {
    atomicAdd(&counters[out_of_range], 1U);
    return;
  }
  Key key;
  if (!ray.has_ray) {
    pack(sample.segment, static_cast<int>(floorf(ray.point[0])), static_cast<int>(floorf(ray.point[1])),
         static_cast<int>(floorf(ray.point[2])), &key);
    insert(table, key, counters);
    return;
  }
  Band band(ray, truncation);
  for (int k = 0; k < band.max_steps; ++k) {
    if (pack(sample.segment, band.walker.cell[0], band.walker.cell[1], band.walker.cell[2], &key))
      insert(table, key, counters);
    if (band.walker.exit() >= band.length) break;
    band.walker.advance();
  }
}

__device__ void add(Cell* cell, float weight, float distance) {
  atomicAdd(&cell->weight, weight);
  atomicAdd(&cell->distance, weight * distance);
}

__device__ void add_color(Cell* cell, const VoxelSample& sample, float weight) {
  if (!(sample.flags & VoxelSample::has_color)) return;
  atomicAdd(&cell->color_weight, weight);
  atomicAdd(&cell->r, weight * sample.color[0]);
  atomicAdd(&cell->g, weight * sample.color[1]);
  atomicAdd(&cell->b, weight * sample.color[2]);
}

// Runs after allocate_kernel: every band voxel exists, so this only looks up.
__global__ void integrate_kernel(const VoxelSample* samples, int count, Table table, float inverse_voxel,
                                 float truncation, float carve_weight, int max_ray_steps) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const VoxelSample sample = samples[index];
  Ray ray;
  if (!prepare(sample, inverse_voxel, &ray)) return;
  Key key;
  if (!ray.has_ray) {
    pack(sample.segment, static_cast<int>(floorf(ray.point[0])), static_cast<int>(floorf(ray.point[1])),
         static_cast<int>(floorf(ray.point[2])), &key);
    const int slot = find(table, key);
    if (slot < 0) return;
    add(&table.cells[slot], sample.weight, 0.0F);
    add_color(&table.cells[slot], sample, sample.weight);
    return;
  }
  Band band(ray, truncation);
  for (int k = 0; k < band.max_steps; ++k) {
    const int* c = band.walker.cell;
    if (pack(sample.segment, c[0], c[1], c[2], &key)) {
      const int slot = find(table, key);
      if (slot >= 0) {
        // Signed distance of the voxel centre along the ray: positive in front of the point.
        const float along = (static_cast<float>(c[0]) + 0.5F - ray.origin[0]) * ray.direction[0] +
                            (static_cast<float>(c[1]) + 0.5F - ray.origin[1]) * ray.direction[1] +
                            (static_cast<float>(c[2]) + 0.5F - ray.origin[2]) * ray.direction[2];
        const float distance = fminf(1.0F, fmaxf(-1.0F, (ray.depth - along) / truncation));
        add(&table.cells[slot], sample.weight, distance);
        // Colour comes mostly from the voxels at the point itself.
        const float near = 1.0F - fabsf(distance);
        add_color(&table.cells[slot], sample, sample.weight * fmaxf(0.02F, near * near));
      }
    }
    if (band.walker.exit() >= band.length) break;
    band.walker.advance();
  }
  // Free space: from the band back to the camera, so that a capped walk still
  // carves next to the surface. The first voxel belongs to the band.
  const float free_length = ray.depth - truncation;
  if (!(carve_weight > 0) || free_length <= 0) return;
  float start[3], backward[3];
  for (int a = 0; a < 3; ++a) {
    start[a] = ray.origin[a] + free_length * ray.direction[a];
    backward[a] = -ray.direction[a];
  }
  Walker walker;
  walker.start(start, backward);
  for (int k = 0; k < max_ray_steps; ++k) {
    if (walker.exit() >= free_length) break;
    walker.advance();
    if (!pack(sample.segment, walker.cell[0], walker.cell[1], walker.cell[2], &key)) break;
    const int slot = find(table, key);
    if (slot >= 0) add(&table.cells[slot], carve_weight * sample.weight, 1.0F);
  }
}

// Moves every voxel of `from` into the (empty, larger) table `to`.
__global__ void rehash_kernel(Table from, Table to, unsigned* counters) {
  const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot > from.mask || from.keys[slot] == kEmpty) return;
  const int target = insert(to, from.keys[slot], counters);
  if (target < 0) return;
  to.cells[target] = from.cells[slot];  // keys are unique: the slot is this thread's
  to.light[target] = from.light[slot];
}

// How much of a ray the voxel blocks: nothing unless it is drawn (enough
// weight, mean distance inside the band), then in proportion to its weight.
__device__ float opacity(const Cell& cell, float truncation, const VoxelShadeConfig& config) {
  if (!(cell.weight > 0) || cell.weight < config.min_weight) return 0.0F;
  if (fabsf(truncation * cell.distance / cell.weight) > config.band) return 0.0F;
  return fminf(1.0F, cell.weight / config.opaque_weight);
}

// Fraction of a ray that gets max_distance away from its start. `origin` is
// relative to the voxel `base` (voxel units), which keeps it exact far from
// the grid's origin.
__device__ float trace(const Table& table, float truncation, const VoxelShadeConfig& config, int segment,
                       const int* base, const float* origin, const float* direction) {
  Walker walker;
  walker.start(origin, direction);
  float through = 1.0F;
  const int max_steps = static_cast<int>(3.0F * config.max_distance) + 4;
  for (int k = 0; k < max_steps; ++k) {
    Key key;
    if (!pack(segment, base[0] + walker.cell[0], base[1] + walker.cell[1], base[2] + walker.cell[2], &key)) break;
    const int slot = find(table, key);
    if (slot >= 0) {
      through *= 1.0F - opacity(table.cells[slot], truncation, config);
      if (through < 0.01F) return 0.0F;
    }
    if (walker.exit() >= config.max_distance) break;
    walker.advance();
  }
  return through;
}

// Hands the averages of the table before the last reset to the voxels that exist again.
__global__ void inherit_kernel(Table table, Table previous, int history) {
  const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot > table.mask || table.keys[slot] == kEmpty) return;
  const int old = find(previous, table.keys[slot]);
  if (old < 0) return;
  Light light = previous.light[old];
  if (light.count > history) light.count = static_cast<std::uint16_t>(history);
  table.light[slot] = light;
}

// One pass: per solid voxel and face, a few rays from random points of the
// face, cosine-distributed about its normal, folded into the running average.
// The points come from a 4D low-discrepancy sequence (Roberts' R4), shifted
// per voxel face, so consecutive passes fill the face and the hemisphere evenly.
__global__ void shade_kernel(Table table, float truncation, VoxelShadeConfig config, unsigned sequence,
                             int restart, unsigned* counters) {
  const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot > table.mask) return;
  const Key key = table.keys[slot];
  if (key == kEmpty) return;
  Light light = table.light[slot];
  if (restart && light.count > kRestartCount) light.count = kRestartCount;
  if (!(opacity(table.cells[slot], truncation, config) > 0)) {  // not drawn: nothing to shade
    table.light[slot] = light;
    return;
  }
  atomicAdd(&counters[solid], 1U);
  int segment, base[3];
  unpack(key, &segment, base);
  const int rays = light.count == 0 ? config.first_rays : config.rays_per_face;
  const int total = min(static_cast<int>(light.count) + rays, config.max_history);
  const unsigned step[4] = {3679390609U, 3152041523U, 2700274806U, 2313257605U};  // 2^32 / phi^k, phi^5 = phi + 1
  for (int face = 0; face < 6; ++face) {
    const int axis = face % 3, u = (axis + 1) % 3, v = (axis + 2) % 3;
    const bool positive = face >= 3;
    unsigned shift[4];
    for (int j = 0; j < 4; ++j)
      shift[j] = static_cast<unsigned>(mix(key ^ (0x9E3779B97F4A7C15ULL * static_cast<Key>(face * 4 + j + 1))));
    float sum = 0.0F;
    for (int r = 0; r < rays; ++r) {
      float x[4];
      for (int j = 0; j < 4; ++j)
        x[j] = static_cast<float>(shift[j] + (sequence + static_cast<unsigned>(r)) * step[j]) * 2.3283064e-10F;
      // Start just outside the face, so the first voxel walked is the neighbour.
      float origin[3], direction[3];
      origin[axis] = positive ? 1.0F + 1e-3F : -1e-3F;
      origin[u] = fminf(fmaxf(x[0], 1e-3F), 1.0F - 1e-3F);
      origin[v] = fminf(fmaxf(x[1], 1e-3F), 1.0F - 1e-3F);
      const float radius = sqrtf(fminf(x[2], 1.0F)), angle = 6.2831853F * x[3];
      direction[axis] = (positive ? 1.0F : -1.0F) * sqrtf(fmaxf(1.0F - radius * radius, 1e-6F));
      direction[u] = radius * cosf(angle);
      direction[v] = radius * sinf(angle);
      sum += trace(table, truncation, config, segment, base, origin, direction);
    }
    const float mean = sum / static_cast<float>(rays);
    const float old = static_cast<float>(light.visibility[face]) / 65535.0F;
    const float blended = light.count == 0 ? mean :
        old + (mean - old) * fminf(1.0F, static_cast<float>(rays) / static_cast<float>(total));
    light.visibility[face] = static_cast<std::uint16_t>(fminf(fmaxf(blended, 0.0F), 1.0F) * 65535.0F + 0.5F);
  }
  light.count = static_cast<std::uint16_t>(total);
  table.light[slot] = light;
}

__global__ void compact_kernel(Table table, float truncation, VoxelInstance* out, unsigned capacity,
                               unsigned* counters) {
  const unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot > table.mask) return;
  const Key key = table.keys[slot];
  if (key == kEmpty) return;
  const unsigned index = atomicAdd(&counters[emitted], 1U);
  if (index >= capacity) return;
  const Cell cell = table.cells[slot];
  const Light light = table.light[slot];
  VoxelInstance instance;
  unpack(key, &instance.segment, instance.coord);
  for (int face = 0; face < 6; ++face) {
    const auto value = static_cast<std::uint8_t>(light.count == 0 ? 255 : (light.visibility[face] * 255U + 32767U) / 65535U);
    if (face < 3) instance.visibility_negative[face] = value;
    else instance.visibility_positive[face - 3] = value;
  }
  instance.unused[0] = instance.unused[1] = 0;
  const bool colored = cell.color_weight > 0;
  const float scale = colored ? 1.0F / cell.color_weight : 0.0F;
  instance.color[0] = static_cast<std::uint8_t>(fminf(255.0F, cell.r * scale + 0.5F));
  instance.color[1] = static_cast<std::uint8_t>(fminf(255.0F, cell.g * scale + 0.5F));
  instance.color[2] = static_cast<std::uint8_t>(fminf(255.0F, cell.b * scale + 0.5F));
  instance.color[3] = colored ? 255 : 0;
  instance.weight = cell.weight;
  instance.distance = cell.weight > 0 ? truncation * cell.distance / cell.weight : 0.0F;
  out[index] = instance;
}

unsigned blocks(std::size_t count) { return static_cast<unsigned>((count + kBlock - 1) / kBlock); }

}  // namespace

struct VoxelHash::Impl {
  VoxelHashConfig config;
  VoxelHashStats stats;
  cudaStream_t stream{};
  Table table{};
  std::size_t capacity{};
  // The table as it was before the last reset: only its keys and light are
  // read, to carry the visibility averages over (inherit_pending).
  Table previous{};
  std::size_t previous_capacity{};
  bool inherit_pending{};
  unsigned sequence{};  // next index of the shading sample sequence
  VoxelSample* samples{};
  std::size_t sample_capacity{};
  unsigned* counters{};

  Impl() {
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create voxel hash stream");
    cuda_check(cudaMalloc(&counters, counter_count * sizeof(unsigned)), "allocate voxel hash counters");
  }
  ~Impl() {
    if (stream) cudaStreamSynchronize(stream);
    release(table);
    release(previous);
    cudaFree(samples);
    cudaFree(counters);
    if (stream) cudaStreamDestroy(stream);
  }

  static void release(Table& t) {
    cudaFree(t.keys);
    cudaFree(t.cells);
    cudaFree(t.light);
    t = {};
  }

  Table allocate(std::size_t slots) {
    Table t{};
    t.mask = static_cast<unsigned>(slots - 1);
    try {
      cuda_check(cudaMalloc(&t.keys, slots * sizeof(Key)), "allocate voxel hash keys");
      cuda_check(cudaMalloc(&t.cells, slots * sizeof(Cell)), "allocate voxel hash cells");
      cuda_check(cudaMalloc(&t.light, slots * sizeof(Light)), "allocate voxel hash light");
    } catch (...) {
      release(t);
      throw;
    }
    clear(t, slots);
    return t;
  }

  void clear(const Table& t, std::size_t slots) {
    cuda_check(cudaMemsetAsync(t.keys, 0xFF, slots * sizeof(Key), stream), "clear voxel hash keys");
    cuda_check(cudaMemsetAsync(t.cells, 0, slots * sizeof(Cell), stream), "clear voxel hash cells");
    cuda_check(cudaMemsetAsync(t.light, 0, slots * sizeof(Light), stream), "clear voxel hash light");
  }

  void zero_counters() {
    cuda_check(cudaMemsetAsync(counters, 0, counter_count * sizeof(unsigned), stream), "clear voxel hash counters");
  }
  // Synchronizes the stream.
  void read_counters(unsigned* host) {
    cuda_check(cudaMemcpyAsync(host, counters, counter_count * sizeof(unsigned), cudaMemcpyDeviceToHost, stream),
               "read voxel hash counters");
    cuda_check(cudaStreamSynchronize(stream), "synchronize voxel hash");
  }

  // Grows the table to at least `slots` (a power of two), keeping its voxels.
  void grow(std::size_t slots) {
    slots = std::max(slots, kMinCapacity);
    if (slots <= capacity) return;
    if (slots > kMaxCapacity)
      throw std::runtime_error("voxel hash: more than " + std::to_string(kMaxCapacity / 2) +
                               " voxels; use a larger voxel size");
    Table larger = allocate(slots);
    if (capacity > 0 && stats.voxels > 0) {
      zero_counters();
      rehash_kernel<<<blocks(capacity), kBlock, 0, stream>>>(table, larger, counters);
      unsigned host[counter_count]{};
      try {
        cuda_check(cudaGetLastError(), "rehash voxels");
        read_counters(host);
      } catch (...) {
        release(larger);
        throw;
      }
      if (host[failed] > 0) {  // not expected: the load at most halves
        release(larger);
        throw std::runtime_error("voxel hash: rehash failed");
      }
    }
    cuda_check(cudaStreamSynchronize(stream), "synchronize voxel hash");
    release(table);
    table = larger;
    capacity = slots;
    stats.capacity = capacity;
  }

  static std::size_t power_of_two(std::size_t n) {
    std::size_t p = kMinCapacity;
    while (p < n) p *= 2;
    return p;
  }
};

VoxelHash::VoxelHash(const VoxelHashConfig& config) : impl_(std::make_unique<Impl>()) { reset(config); }
VoxelHash::~VoxelHash() = default;

const VoxelHashConfig& VoxelHash::config() const { return impl_->config; }
const VoxelHashStats& VoxelHash::stats() const { return impl_->stats; }
void* VoxelHash::stream() const { return impl_->stream; }

void VoxelHash::reset(const VoxelHashConfig& config) {
  if (!(config.voxel_size > 0) || !(config.truncation_voxels > 0) || config.max_ray_steps < 0)
    throw std::invalid_argument("voxel hash: voxel size and truncation must be positive");
  auto& m = *impl_;
  const bool same_grid = config.voxel_size == m.config.voxel_size &&
                         config.truncation_voxels == m.config.truncation_voxels;
  if (!same_grid) {
    m.inherit_pending = false;  // the old keys name other places now
  } else if (m.stats.voxels > 0) {
    // Keep the voxels aside for their visibility averages and reuse the table
    // before them. An empty table leaves an older one in place.
    std::swap(m.table, m.previous);
    std::swap(m.capacity, m.previous_capacity);
    m.inherit_pending = true;
  }
  m.config = config;
  m.stats = {};
  m.stats.capacity = m.capacity;
  if (m.capacity > 0) m.clear(m.table, m.capacity);
}

void VoxelHash::shade(const VoxelShadeConfig& requested, bool restart) {
  auto& m = *impl_;
  if (m.capacity == 0 || m.stats.voxels == 0) return;
  VoxelShadeConfig config = requested;
  if (!(config.opaque_weight > 0) || !(config.max_distance > 0))
    throw std::invalid_argument("voxel hash: opaque weight and ray distance must be positive");
  config.max_distance = std::min(config.max_distance, 4096.0F);
  config.rays_per_face = std::clamp(config.rays_per_face, 1, 256);
  config.first_rays = std::clamp(config.first_rays, config.rays_per_face, 256);
  config.max_history = std::clamp(config.max_history, config.first_rays, 65535);
  config.history = std::clamp(config.history, 1, config.max_history);
  const auto begin = std::chrono::steady_clock::now();
  if (m.inherit_pending) {
    m.inherit_pending = false;
    if (m.previous_capacity > 0) {
      inherit_kernel<<<blocks(m.capacity), kBlock, 0, m.stream>>>(m.table, m.previous, config.history);
      cuda_check(cudaGetLastError(), "inherit voxel visibility");
    }
  }
  m.zero_counters();
  shade_kernel<<<blocks(m.capacity), kBlock, 0, m.stream>>>(m.table, m.config.truncation_voxels, config, m.sequence,
                                                           restart ? 1 : 0, m.counters);
  cuda_check(cudaGetLastError(), "shade voxels");
  unsigned host[counter_count]{};
  m.read_counters(host);
  m.stats.solid = host[solid];
  // A new voxel's first rays run into the indices of the next passes; harmless.
  m.sequence += static_cast<unsigned>(config.rays_per_face);
  m.stats.shade_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}

void VoxelHash::integrate(const VoxelSample* samples, std::size_t count) {
  if (count == 0) return;
  auto& m = *impl_;
  const auto begin = std::chrono::steady_clock::now();
  if (count > m.sample_capacity) {
    cudaFree(m.samples);
    m.samples = nullptr;
    m.sample_capacity = 0;
    const std::size_t wanted = std::max(count, std::size_t{4096});
    cuda_check(cudaMalloc(&m.samples, wanted * sizeof(VoxelSample)), "allocate voxel samples");
    m.sample_capacity = wanted;
  }
  cuda_check(cudaMemcpyAsync(m.samples, samples, count * sizeof(VoxelSample), cudaMemcpyHostToDevice, m.stream),
             "upload voxel samples");
  const float inverse_voxel = 1.0F / m.config.voxel_size;
  const int n = static_cast<int>(count);
  // Allocation only inserts keys, so it can simply be repeated after the
  // table has grown; the sums are added once every voxel has its slot.
  m.grow(Impl::power_of_two(4 * (m.stats.voxels + count)));
  for (;;) {
    m.zero_counters();
    allocate_kernel<<<blocks(count), kBlock, 0, m.stream>>>(m.samples, n, m.table, inverse_voxel,
                                                           m.config.truncation_voxels, m.counters);
    cuda_check(cudaGetLastError(), "allocate voxels");
    unsigned host[counter_count]{};
    m.read_counters(host);
    m.stats.voxels += host[inserted];
    const bool full = host[failed] > 0;
    if (full || 2 * m.stats.voxels > m.capacity)
      m.grow(std::max(2 * m.capacity, Impl::power_of_two(4 * m.stats.voxels)));
    if (!full) {
      m.stats.out_of_range += host[out_of_range];
      break;
    }
  }
  integrate_kernel<<<blocks(count), kBlock, 0, m.stream>>>(m.samples, n, m.table, inverse_voxel,
                                                          m.config.truncation_voxels, m.config.carve_weight,
                                                          m.config.max_ray_steps);
  cuda_check(cudaGetLastError(), "integrate voxels");
  cuda_check(cudaStreamSynchronize(m.stream), "synchronize voxel hash");
  m.stats.samples += count;
  for (std::size_t i = 0; i < count; ++i) m.stats.rays += (samples[i].flags & VoxelSample::has_ray) ? 1 : 0;
  m.stats.integrate_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}

std::size_t VoxelHash::compact(void* device_instances, std::size_t capacity) {
  auto& m = *impl_;
  if (m.capacity == 0 || m.stats.voxels == 0) return 0;
  m.zero_counters();
  compact_kernel<<<blocks(m.capacity), kBlock, 0, m.stream>>>(m.table, m.config.truncation_voxels,
                                                             static_cast<VoxelInstance*>(device_instances),
                                                             static_cast<unsigned>(capacity), m.counters);
  cuda_check(cudaGetLastError(), "compact voxels");
  unsigned host[counter_count]{};
  m.read_counters(host);
  return host[emitted];
}

std::vector<VoxelInstance> VoxelHash::download() {
  auto& m = *impl_;
  std::vector<VoxelInstance> result(m.stats.voxels);
  if (result.empty()) return result;
  VoxelInstance* device = nullptr;
  cuda_check(cudaMalloc(&device, result.size() * sizeof(VoxelInstance)), "allocate voxel download");
  try {
    result.resize(std::min(result.size(), compact(device, result.size())));
    cuda_check(cudaMemcpy(result.data(), device, result.size() * sizeof(VoxelInstance), cudaMemcpyDeviceToHost),
               "download voxels");
  } catch (...) {
    cudaFree(device);
    throw;
  }
  cudaFree(device);
  return result;
}

}  // namespace slam_native
