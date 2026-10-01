#pragma once
// Draws a VoxelHash as solid cubes: the hash is rebuilt on the GPU when its
// samples or configuration change, compacted straight into an OpenGL instance
// buffer (CUDA-GL interop, no host copy) and drawn with one instanced call.
// Each face is darkened by its ambient visibility, which every draw() refines
// with one more Monte Carlo pass (VoxelHash::shade) until it has converged.
// draw() is meant for PointCloudRenderer's extra pass, so the cubes share the
// map points' depth buffer. All calls need the app's OpenGL 3.3 core context;
// CUDA is first used in draw().
#include "slam_native/point_cloud_renderer.hpp"
#include "slam_native/voxel_hash.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct cudaGraphicsResource;

namespace slam_native {

enum class VoxelColor { plain, image, weight, distance };

struct VoxelStyle {
  float min_weight{1.0F};  // voxels with less summed weight are not drawn
  float band{0.75F};       // nor are voxels whose mean distance is beyond this many voxels from zero
  VoxelColor color{VoxelColor::image};  // image: falls back to plain without a colour
  int segment{-1};         // only this segment's voxels; < 0 draws all
  float fill{0.92F};       // cube edge as a fraction of the voxel size
  float weight_span{8.0F};  // VoxelColor::weight: the weight at the ramp's end
  // Ambient occlusion: how much of each face's blocked ambient light is taken
  // away (0: none, and nothing is traced; 1: all of it).
  float ambient{1.0F};
  float ambient_distance{48.0F};  // VoxelShadeConfig::max_distance, voxels
  float opaque_weight{2.0F};      // VoxelShadeConfig::opaque_weight
};

class VoxelRenderer {
 public:
  VoxelRenderer();
  ~VoxelRenderer();
  VoxelRenderer(const VoxelRenderer&) = delete;
  VoxelRenderer& operator=(const VoxelRenderer&) = delete;

  // The voxels are rebuilt from these in the next draw(). No GL or CUDA calls.
  void set_samples(std::vector<VoxelSample> samples);
  void set_config(const VoxelHashConfig& config);

  // Draws into the bound, depth-tested target with the map points' projection.
  // Changes the bound program, vertex array and array buffer.
  void draw(const PointCloudCamera& camera, float logical_width, float logical_height, const VoxelStyle& style);

  [[nodiscard]] std::size_t voxels() const { return count_; }
  [[nodiscard]] VoxelHashStats stats() const;
  [[nodiscard]] double build_ms() const { return build_ms_; }  // last rebuild: integrate, one shading pass, compact
  // Samples per face since the map or the solid voxels last changed, and the
  // number at which shading stops.
  [[nodiscard]] int shade_samples() const { return shade_samples_; }
  [[nodiscard]] int shade_target() const { return VoxelShadeConfig{}.max_history; }
  [[nodiscard]] const std::string& error() const { return error_; }  // empty when the last rebuild worked

 private:
  bool ensure_gl();
  void ensure_buffer(std::size_t instances);
  void rebuild();
  bool shade(const VoxelStyle& style);
  void upload();

  std::vector<VoxelSample> samples_;
  VoxelHashConfig config_;
  bool dirty_{};
  std::unique_ptr<VoxelHash> hash_;
  std::size_t count_{};
  double build_ms_{};
  VoxelShadeConfig shade_;  // of the last pass
  int shade_samples_{};
  std::string error_;

  bool gl_ready_{}, gl_failed_{};
  unsigned int program_{}, mesh_{}, vao_{}, buffer_{};
  std::size_t capacity_{};
  cudaGraphicsResource* resource_{};
  int u_pivot_{-1}, u_right_{-1}, u_up_{-1}, u_toward_{-1}, u_eye_distance_{-1}, u_perspective_{-1}, u_focal_{-1},
      u_scale_{-1}, u_near_{-1}, u_far_{-1}, u_viewport_{-1}, u_rotation_{-1}, u_voxel_size_{-1}, u_fill_{-1},
      u_min_weight_{-1}, u_band_{-1}, u_color_mode_{-1}, u_segment_{-1}, u_weight_span_{-1}, u_ambient_{-1};
};

}  // namespace slam_native
