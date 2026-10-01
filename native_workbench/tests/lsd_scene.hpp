#pragma once
// Ray-cast test scene for the LSD tests: a textured wall at z = 4 and a box
// face at z = 2.5 (world = first camera), 320x240, f = 300, 3x3 supersampling.
#include "lsd/image.hpp"

#include <cmath>
#include <cstdint>

namespace lsd_scene {
using namespace slam_native::lsd;

constexpr int kW = 320, kH = 240;
inline const Camera kCamera{300, 300, 159.5, 119.5, kW, kH};

inline double hash01(int x, int y, int seed) {
  std::uint32_t h = static_cast<std::uint32_t>(x) * 374761393U + static_cast<std::uint32_t>(y) * 668265263U +
                    static_cast<std::uint32_t>(seed) * 2147483647U;
  h = (h ^ (h >> 13)) * 1274126177U;
  return ((h ^ (h >> 16)) & 0xFFFFFF) / static_cast<double>(0xFFFFFF);
}
inline double value_noise(double x, double y, int seed) {
  const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
  const double fx = x - x0, fy = y - y0;
  const double sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const double a = hash01(x0, y0, seed), b = hash01(x0 + 1, y0, seed);
  const double c = hash01(x0, y0 + 1, seed), d = hash01(x0 + 1, y0 + 1, seed);
  return (a + sx * (b - a)) + sy * ((c + sx * (d - c)) - (a + sx * (b - a)));
}
// Multi-octave texture on a surface (metres).
inline double texture(double x, double y, int seed) {
  double v = 0, amplitude = 1, frequency = 2, total = 0;
  for (int o = 0; o < 5; ++o) {
    v += amplitude * value_noise(x * frequency, y * frequency, seed + o);
    total += amplitude;
    amplitude *= 0.6;
    frequency *= 2.2;
  }
  return 30 + 195 * v / total;
}

// World: wall z = 4, box front face z = 2.5 over x in [-0.7, 0.2], y in [-0.5, 0.4].
// Returns depth along the camera ray and intensity. pose: camera <- world.
inline void trace(const SE3& pose, double u, double v, double& depth, double& intensity) {
  const SE3 inv = pose.inverse();
  const Vec3 origin = inv.t;
  const Vec3 dir = inv.R * kCamera.ray(u, v);  // camera z component 1 before rotation
  double best = 1e9;
  int seed = 0;
  Vec3 hit = Vec3::Zero();
  auto plane = [&](double z, int s, double x0, double x1, double y0, double y1) {
    if (std::abs(dir.z()) < 1e-12) return;
    const double lambda = (z - origin.z()) / dir.z();
    if (lambda <= 0 || lambda >= best) return;
    const Vec3 p = origin + lambda * dir;
    if (p.x() < x0 || p.x() > x1 || p.y() < y0 || p.y() > y1) return;
    best = lambda;
    seed = s;
    hit = p;
  };
  plane(4.0, 11, -1e9, 1e9, -1e9, 1e9);
  plane(2.5, 23, -0.7, 0.2, -0.5, 0.4);
  depth = best;  // ray has z = 1 in the camera, so lambda is the camera depth
  intensity = texture(hit.x(), hit.y(), seed);
}

inline ImageF render(const SE3& pose, ImageF* idepth = nullptr) {
  ImageF image(kW, kH);
  if (idepth) *idepth = ImageF(kW, kH);
  for (int y = 0; y < kH; ++y) {
    for (int x = 0; x < kW; ++x) {
      double sum = 0, depth = 0, intensity = 0;
      for (int sy = -1; sy <= 1; ++sy)
        for (int sx = -1; sx <= 1; ++sx) {
          trace(pose, x + sx / 3.0, y + sy / 3.0, depth, intensity);
          sum += intensity;
        }
      image.at(x, y) = static_cast<float>(sum / 9);
      if (idepth) {
        trace(pose, x, y, depth, intensity);
        idepth->at(x, y) = static_cast<float>(1 / depth);
      }
    }
  }
  return image;
}

}  // namespace lsd_scene
