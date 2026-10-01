#pragma once
// ImGui panel showing the live visual-odometry camera trajectory and map.
// Map points are drawn by PointCloudRenderer (instanced GL geometry); the
// grid, trajectory and frustum are ImGui draw-list overlays.
#include "slam_native/point_cloud_renderer.hpp"
#include "slam_native/visual_odometry.hpp"

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
  // `open` toggles the window; frame size sets the drawn frustum's aspect.
  void draw(const VisualOdometry& odometry, bool* open, int frame_width, int frame_height);

 private:
  std::vector<TrajectorySample> samples_;
  std::vector<MapPoint> retired_;  // append-only copy of the odometry's retired map
  std::vector<MapPoint> active_;
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
  bool show_retired_{true};
  int color_mode_{1};            // PointColor: 0 plain, 1 image
  bool dim_retired_{true};
  // World-space diameter of the map points. Segment units: the first
  // keyframe's median scene depth is 1.
  float point_size_{0.02F};
  int rotation_{};
  bool perspective_{true};  // orbit view only; fixed views are orthographic
  bool all_segments_{false};
};

}  // namespace slam_native
