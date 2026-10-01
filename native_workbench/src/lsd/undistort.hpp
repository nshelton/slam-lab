#pragma once
// Remaps a lens-distorted image (OpenCV radial-tangential model) to the
// pinhole image of the same intrinsics, at the working resolution.
#include "lsd/image.hpp"

#include <array>

namespace slam_native::lsd {

class Undistorter {
 public:
  Undistorter() = default;  // identity
  // camera: the working-resolution intrinsics, used for both the distorted
  // input and the pinhole output. k = (k1, k2, p1, p2, k3).
  Undistorter(const Camera& camera, const std::array<double, 5>& k);

  [[nodiscard]] bool identity() const { return map_x_.empty(); }
  // Pinhole image; pixels whose source lies outside the input are clamped and
  // marked 0 in mask().
  [[nodiscard]] ImageF apply(const ImageF& distorted) const;
  // Distorted input pixel seen by pinhole pixel (x, y) (identity: itself).
  [[nodiscard]] Vec2 source(int x, int y) const;
  // The same for any (sub)pixel, computed from the model.
  [[nodiscard]] Vec2 distort_pixel(const Vec2& pinhole) const;
  // Inverse: the pinhole pixel of a distorted pixel (fixed-point iteration).
  [[nodiscard]] Vec2 undistort_pixel(const Vec2& distorted) const;
  // 1 where the pinhole pixel samples inside the input (all 1 for identity).
  [[nodiscard]] const ImageF& mask() const { return mask_; }

 private:
  Camera camera_;
  std::array<double, 5> k_{};
  ImageF map_x_, map_y_, mask_;
};

// Distorted normalised coordinates of an undistorted normalised point.
Vec2 distort(const std::array<double, 5>& k, const Vec2& x);

}  // namespace slam_native::lsd
