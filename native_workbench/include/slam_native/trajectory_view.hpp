#pragma once
// ImGui panel showing the live visual-odometry camera trajectory and map.
// Map points are drawn by PointCloudRenderer (instanced GL geometry); the
// grid, trajectory and frustum are ImGui draw-list overlays.
#include "slam_native/point_cloud_renderer.hpp"
#include "slam_native/visual_odometry.hpp"
#include "slam_native/voxel_renderer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace slam_native {

class TrajectoryView {
 public:
  // Refresh the cached trajectory after each processed frame (cheap: only
  // samples that can still change are re-read).
  void update(const VisualOdometry& odometry);
  void reset();
  // Clockwise display rotation of the video (degrees): the view's "up"
  // follows the upright image rather than the sensor's y axis.
  void set_display_rotation(int degrees) { rotation_ = degrees; }
  // Turns the voxel map on with this voxel size and the map points off or on
  // (for snapshots; the window has the controls).
  void show_voxels(float voxel_size, bool with_points, float carve = 1.0F, bool all_observers = true) {
    show_voxels_ = true;
    voxel_size_ = voxel_size;
    show_points_ = with_points;
    voxel_carve_ = carve;
    voxel_all_observers_ = all_observers;
  }
  [[nodiscard]] const VoxelRenderer& voxels() const { return voxels_; }
  // `open` toggles the window; frame size sets the drawn frustum's aspect.
  void draw(const VisualOdometry& odometry, bool* open, int frame_width, int frame_height);

 private:
  std::vector<TrajectorySample> samples_;
  // The odometry's map as of the last update(): the landmarks in its local
  // map (tracked and refined now) and the rest (kept; found again by descriptor).
  std::vector<MapPoint> local_;
  std::vector<MapPoint> outside_;
  PointCloudRenderer points_renderer_;
  int mode_{0};           // 0 orbit, 1 top, 2 side, 3 front
  float yaw_{-0.6F};
  float pitch_{0.45F};
  float zoom_{1.0F};                 // orthographic views: magnification
  float dolly_{1.0F};                // perspective orbit: eye distance / fitted distance
  std::array<float, 3> pan_{};       // orbit point offset (display space)
  double last_interaction_{-1e9};    // ImGui time of the last wheel/drag (trackball fade)
  bool follow_{false};
  bool show_points_{true};
  bool show_outside_{true};
  int color_mode_{1};            // PointColor: 0 plain, 1 image, 2 age
  float age_span_{300.0F};       // colour by age: frames from new to the ramp's end
  bool dim_outside_{true};
  // World-space diameter of the map points. Segment units: the first
  // keyframe's median scene depth is 1.
  float point_size_{0.02F};
  int rotation_{};
  bool perspective_{true};  // orbit view only; fixed views are orthographic
  bool all_segments_{false};

  // Voxel map of the landmarks (display only, see voxel_hash.hpp): rebuilt on
  // the GPU from the map points and the camera centres that saw them.
  [[nodiscard]] std::vector<VoxelSample> voxel_samples(const VisualOdometry& odometry) const;
  VoxelRenderer voxels_;
  bool show_voxels_{false};
  bool voxels_stale_{true};       // the map or the trajectory changed since the samples were built
  float voxel_size_{0.05F};       // segment units, like point_size_
  float voxel_carve_{1.0F};       // VoxelHashConfig::carve_weight
  float voxel_min_weight_{1.0F};  // VoxelStyle
  float voxel_band_{0.75F};
  int voxel_color_{1};            // VoxelColor
  bool voxel_all_observers_{true};  // rays from every keyframe that saw a landmark, not only its first and last frames
  float voxel_ambient_{1.0F};       // VoxelStyle::ambient
  float voxel_ambient_distance_{48.0F};
};

}  // namespace slam_native
