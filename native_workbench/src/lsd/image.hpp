#pragma once
// Grayscale float images and pyramids for the direct (LSD) pipeline.
// Pixel centres are at integer coordinates. Intensities are on a 0-255 scale.
#include "vo_geometry.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace slam_native::lsd {
using vo::Mat3;
using vo::SE3;
using vo::Vec2;
using vo::Vec3;
using vo::Vec6;

struct ImageF {
  int width{}, height{};
  std::vector<float> data;

  ImageF() = default;
  ImageF(int w, int h, float fill = 0) : width(w), height(h), data(static_cast<std::size_t>(w) * h, fill) {}
  [[nodiscard]] float& at(int x, int y) { return data[static_cast<std::size_t>(y) * width + x]; }
  [[nodiscard]] float at(int x, int y) const { return data[static_cast<std::size_t>(y) * width + x]; }
  [[nodiscard]] bool empty() const { return data.empty(); }
  // True when bilinear sampling at (x, y) only touches pixels at least `border` inside.
  [[nodiscard]] bool inside(double x, double y, double border = 0) const {
    return x >= border && y >= border && x <= width - 1 - border && y <= height - 1 - border;
  }
  // Bilinear sample; (x, y) must satisfy inside(x, y).
  [[nodiscard]] float sample(double x, double y) const {
    const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    const int x1 = x0 + 1 < width ? x0 + 1 : x0, y1 = y0 + 1 < height ? y0 + 1 : y0;
    const float fx = static_cast<float>(x - x0), fy = static_cast<float>(y - y0);
    const float top = at(x0, y0) + fx * (at(x1, y0) - at(x0, y0));
    const float bottom = at(x0, y1) + fx * (at(x1, y1) - at(x0, y1));
    return top + fy * (bottom - top);
  }
};

// Pinhole intrinsics of one pyramid level.
struct Camera {
  double fx{1}, fy{1}, cx{0}, cy{0};
  int width{}, height{};
  [[nodiscard]] Vec2 project(const Vec3& x) const { return {fx * x.x() / x.z() + cx, fy * x.y() / x.z() + cy}; }
  // Ray through pixel (u, v) with z = 1.
  [[nodiscard]] Vec3 ray(double u, double v) const { return {(u - cx) / fx, (v - cy) / fy, 1.0}; }
  // The next level of a 2x2-average pyramid: u' = (u + 0.5) / 2 - 0.5.
  [[nodiscard]] Camera half() const {
    return {fx / 2, fy / 2, (cx + 0.5) / 2 - 0.5, (cy + 0.5) / 2 - 0.5, width / 2, height / 2};
  }
};

// One pyramid level: intensity and central-difference gradients. mask
// (optional): 0 where the image holds no measurement (e.g. undistortion
// sampled outside the input); samples there are not compared.
struct Level {
  Camera camera;
  ImageF image, gx, gy;
  ImageF mask;
  [[nodiscard]] bool measured(double x, double y) const {
    return mask.empty() || mask.at(static_cast<int>(x + 0.5), static_cast<int>(y + 0.5)) > 0.5F;
  }
};

// Image pyramid of one frame. Level 0 is the input resolution.
struct Pyramid {
  std::vector<Level> levels;
  [[nodiscard]] int size() const { return static_cast<int>(levels.size()); }
};

// Up to `levels` levels; stops before a side would drop below 4 px.
// mask (optional, same size): level 0's mask; coarser levels are measured
// only where all four children are.
Pyramid build_pyramid(const ImageF& image, const Camera& camera, int levels, const ImageF* mask = nullptr);
// 2x2 average (odd last row/column dropped).
ImageF downsample(const ImageF& image);
void gradients(const ImageF& image, ImageF& gx, ImageF& gy);

}  // namespace slam_native::lsd
