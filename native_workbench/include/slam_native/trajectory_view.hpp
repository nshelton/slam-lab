#pragma once
// ImGui panel showing the live visual-odometry camera trajectory and map.
#include "slam_native/visual_odometry.hpp"

#include <cstddef>
#include <vector>

namespace slam_native {

class TrajectoryView {
 public:
  // Refresh the cached trajectory after each processed frame (cheap: only
  // samples that can still change are re-read).
  void update(const VisualOdometry& odometry);
  void reset();
  // `open` toggles the window; frame size sets the drawn frustum's aspect.
  void draw(const VisualOdometry& odometry, bool* open, int frame_width, int frame_height);

 private:
  std::vector<TrajectorySample> samples_;
  std::vector<std::array<float, 3>> points_;
  int mode_{0};           // 0 orbit, 1 top, 2 side, 3 front
  float yaw_{-0.6F};
  float pitch_{0.45F};
  float zoom_{1.0F};
  float pan_x_{}, pan_y_{};
  bool follow_{false};
  bool show_points_{true};
  bool all_segments_{false};
};

}  // namespace slam_native
