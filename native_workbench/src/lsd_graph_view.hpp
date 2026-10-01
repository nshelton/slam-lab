#pragma once
// ImGui panel for the LSD pipeline's keyframe pose graph: keyframe frustums,
// graph edges, the tracked trajectory and each keyframe's semi-dense cloud
// (PointCloudRenderer). Camera controls follow TrajectoryView: wheel zooms,
// left-drag orbits, right-drag pans, double-click fits.
#include "lsd/odometry.hpp"
#include "slam_native/point_cloud_renderer.hpp"

#include <array>

namespace slam_native {

class LsdGraphView {
 public:
  void set_display_rotation(int degrees) { rotation_ = degrees; }
  // Drop uploaded clouds (the odometry was restarted). No GL calls.
  void reset() { ++generation_; }
  void draw(const lsd::Odometry& odometry, bool* open);

 private:
  PointCloudRenderer renderer_;
  std::uint64_t generation_{1};
  int rotation_{};
  int mode_{0};  // 0 orbit, 1 top, 2 side, 3 front
  float yaw_{-0.6F}, pitch_{0.45F};
  float zoom_{1}, dolly_{1};
  std::array<float, 3> pan_{};
  bool perspective_{true};
  bool follow_{false};
  bool show_clouds_{true};
  bool show_trajectory_{true};
  bool show_frustums_{true};
  float cloud_point_pixels_{1.5F};
  float max_sigma_{1.0F};  // relative inverse-depth sigma; >= 1 shows all
  bool color_by_sigma_{false};
};

}  // namespace slam_native
