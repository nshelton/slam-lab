#pragma once
// ImGui panel for the live lens estimate (field of view and radial
// distortion k1) and the values the camera pose uses.
#include "slam_native/focal_estimation.hpp"

#include <optional>

namespace slam_native {

struct LensChoice {
  double hfov_degrees{};
  double k1{};
};

class FovView {
 public:
  // `applied` is what the pose currently uses (hfov 0 when explicit
  // intrinsics override it). Returns a choice to apply when the user asks.
  // Sets `reset_estimate` when the user clears the accumulated evidence.
  std::optional<LensChoice> draw(const FocalEstimate& estimate, LensChoice applied, bool* open,
                                 bool& reset_estimate);

 private:
  float hfov_{60.0F};
  float k1_{0.0F};
  bool follow_{true};  // sliders track the estimate until the user drags one
  bool initialized_{};
};

}  // namespace slam_native
