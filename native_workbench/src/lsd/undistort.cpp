#include "lsd/undistort.hpp"

#include <algorithm>

namespace slam_native::lsd {

Vec2 distort(const std::array<double, 5>& k, const Vec2& x) {
  const double r2 = x.squaredNorm(), r4 = r2 * r2, r6 = r4 * r2;
  const double radial = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6;
  const double xy = x.x() * x.y();
  return {x.x() * radial + 2 * k[2] * xy + k[3] * (r2 + 2 * x.x() * x.x()),
          x.y() * radial + k[2] * (r2 + 2 * x.y() * x.y()) + 2 * k[3] * xy};
}

Undistorter::Undistorter(const Camera& camera, const std::array<double, 5>& k)
    : camera_(camera), k_(k), map_x_(camera.width, camera.height), map_y_(camera.width, camera.height), mask_(camera.width, camera.height) {
  for (int y = 0; y < camera.height; ++y) {
    for (int x = 0; x < camera.width; ++x) {
      const Vec3 ray = camera.ray(x, y);
      const Vec2 d = distort(k, {ray.x(), ray.y()});
      const double u = camera.fx * d.x() + camera.cx, v = camera.fy * d.y() + camera.cy;
      map_x_.at(x, y) = static_cast<float>(u);
      map_y_.at(x, y) = static_cast<float>(v);
      mask_.at(x, y) = u >= 0 && v >= 0 && u <= camera.width - 1 && v <= camera.height - 1 ? 1.0F : 0.0F;
    }
  }
  // Erode by 2 px: gradients next to the edge see clamped samples.
  const ImageF inside = mask_;
  for (int y = 0; y < camera.height; ++y)
    for (int x = 0; x < camera.width; ++x)
      for (int dy = -2; dy <= 2 && mask_.at(x, y) > 0; ++dy)
        for (int dx = -2; dx <= 2; ++dx) {
          const int nx = std::clamp(x + dx, 0, camera.width - 1), ny = std::clamp(y + dy, 0, camera.height - 1);
          if (inside.at(nx, ny) == 0) {
            mask_.at(x, y) = 0;
            break;
          }
        }
}

ImageF Undistorter::apply(const ImageF& distorted) const {
  if (identity()) return distorted;
  ImageF out(map_x_.width, map_x_.height);
  const double xmax = distorted.width - 1, ymax = distorted.height - 1;
  for (std::size_t i = 0; i < out.data.size(); ++i)
    out.data[i] = distorted.sample(std::clamp<double>(map_x_.data[i], 0, xmax), std::clamp<double>(map_y_.data[i], 0, ymax));
  return out;
}

Vec2 Undistorter::source(int x, int y) const {
  if (identity()) return {double(x), double(y)};
  return {map_x_.at(x, y), map_y_.at(x, y)};
}

Vec2 Undistorter::distort_pixel(const Vec2& pinhole) const {
  if (identity()) return pinhole;
  const Vec3 ray = camera_.ray(pinhole.x(), pinhole.y());
  const Vec2 d = distort(k_, {ray.x(), ray.y()});
  return {camera_.fx * d.x() + camera_.cx, camera_.fy * d.y() + camera_.cy};
}

Vec2 Undistorter::undistort_pixel(const Vec2& distorted) const {
  if (identity()) return distorted;
  const Vec2 target{(distorted.x() - camera_.cx) / camera_.fx, (distorted.y() - camera_.cy) / camera_.fy};
  Vec2 x = target;
  for (int it = 0; it < 20; ++it) x += target - distort(k_, x);
  return {camera_.fx * x.x() + camera_.cx, camera_.fy * x.y() + camera_.cy};
}

}  // namespace slam_native::lsd
