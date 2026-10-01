#pragma once
// GPU voxel hash with a ray-carved truncated signed distance, for display only.
//
// Input is a list of samples: a 3D point, optionally with the camera centre
// that observed it. Each sample updates the voxels in a band of +/- the
// truncation distance around its point along the viewing ray (the usual TSDF
// update, scattered along the ray instead of gathered from a depth image) and
// then walks the ray from the camera to that band, marking the voxels it
// crosses as free. Free-space updates only touch voxels that already exist, so
// the table holds the bands and not the empty space the rays cross.
//
// shade() estimates each voxel face's ambient visibility by Monte Carlo: a
// few rays per face and call, traced through the table and averaged over
// calls. The averages survive reset() by voxel key, so a map that is rebuilt
// every frame keeps converging.
//
// Voxels live in an open-addressing hash table on the device (64-bit keys,
// linear probing, lock-free insertion). Nothing here reads or writes the SLAM
// state: the caller builds the samples.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace slam_native {

struct VoxelSample {
  enum : std::uint8_t { has_ray = 1, has_color = 2 };
  float point[3];
  float origin[3];        // camera centre that observed the point (has_ray)
  std::uint8_t color[3];  // has_color
  std::uint8_t flags;
  std::int32_t segment;   // voxels of different segments never mix; 0 .. 1022
  float weight;
};
static_assert(sizeof(VoxelSample) == 36);

// One occupied voxel, as compacted for drawing.
struct VoxelInstance {
  std::int32_t coord[3];  // the voxel spans [coord, coord + 1) * voxel_size
  std::uint8_t color[4];  // mean colour; a = 255 when any sample had one
  std::int32_t segment;
  float weight;    // summed sample weight, band and free-space updates together
  float distance;  // mean signed distance in voxels: 0 at a point, > 0 in front of it (free), < 0 behind
  // Ambient visibility of the faces toward -x, -y, -z and +x, +y, +z:
  // 255 = nothing in the way (also when the voxel was never shaded).
  std::uint8_t visibility_negative[3];
  std::uint8_t visibility_positive[3];
  std::uint8_t unused[2];
};
static_assert(sizeof(VoxelInstance) == 36);

struct VoxelHashConfig {
  float voxel_size{0.05F};        // world units
  float truncation_voxels{2.0F};  // half-width of the band around a point, along its ray
  float carve_weight{1.0F};       // weight of a free-space update relative to a band update; 0: no carving
  int max_ray_steps{4096};        // free-space walk, counted from the band toward the camera
};

// Ambient visibility of the voxel faces, see VoxelHash::shade().
struct VoxelShadeConfig {
  // Which voxels are solid: the ones a viewer draws (VoxelStyle has the same two).
  float min_weight{1.0F};
  float band{0.75F};           // |mean distance| in voxels
  float opaque_weight{2.0F};   // a solid voxel blocks weight / opaque_weight of a ray, at most all of it
  float max_distance{48.0F};   // voxels; a ray that gets this far counts as unblocked
  int rays_per_face{1};        // per call
  int first_rays{8};           // for a voxel that has no average yet
  int history{64};             // samples an average keeps across reset(): how fast it follows a changing map
  int max_history{2048};       // samples it can reach while the map is unchanged (at most 65535)
};

struct VoxelHashStats {
  std::size_t samples{};       // integrated since the last reset
  std::size_t rays{};          // of those, with a viewing ray
  std::size_t voxels{};        // allocated
  std::size_t solid{};         // of those, the ones the last shade() took as solid (drawn)
  std::size_t capacity{};      // table slots
  std::size_t out_of_range{};  // voxel updates skipped: non-finite or beyond the +/- 2^17 voxel key range
  double integrate_ms{};       // last integrate(), wall time including the upload
  double shade_ms{};           // last shade()
};

class VoxelHash {
 public:
  explicit VoxelHash(const VoxelHashConfig& config = {});
  ~VoxelHash();
  VoxelHash(const VoxelHash&) = delete;
  VoxelHash& operator=(const VoxelHash&) = delete;

  // Empties the table (keeps its memory) and sets the configuration. The
  // visibility averages are kept aside and handed to the voxels that exist
  // again after the next integrate() calls, unless the voxel grid changed.
  void reset(const VoxelHashConfig& config);
  // Adds samples to the voxels already there. The table grows as needed.
  void integrate(const VoxelSample* samples, std::size_t count);
  // One Monte Carlo pass over the faces of the solid voxels: rays_per_face
  // cosine-distributed rays from random points of each face, their mean
  // unblocked fraction folded into the face's running average. `restart`
  // (the solid set changed: other min_weight, band, ...) keeps the averages
  // but lets the next passes replace them quickly.
  void shade(const VoxelShadeConfig& config, bool restart = false);
  // Writes every allocated voxel to a device buffer of `capacity` instances
  // and returns how many there are (at most stats().voxels); nothing is
  // written past the capacity. The order is unspecified.
  std::size_t compact(void* device_instances, std::size_t capacity);
  // The same, to the host (tests and inspection).
  [[nodiscard]] std::vector<VoxelInstance> download();

  [[nodiscard]] const VoxelHashConfig& config() const;
  [[nodiscard]] const VoxelHashStats& stats() const;
  // The CUDA stream the hash works on (a cudaStream_t), for mapping buffers.
  [[nodiscard]] void* stream() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace slam_native
