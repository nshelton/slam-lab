#pragma once
// Principal axes of a 3x3 covariance, for drawing uncertainty ellipsoids:
// the shape matrix S = R diag(sqrt(lambda)) maps the unit sphere onto the 1-sigma
// ellipsoid (S S^T = covariance). Header-only, no dependencies.
#include <algorithm>
#include <array>
#include <cmath>

namespace slam_native {

// covariance: xx xy xz yy yz zz (MapPoint::covariance). Returns S column-major
// (axis k = column k), each axis length clamped to max_axis (> 0). All-zero
// (unknown) or non-finite input gives a zero matrix.
inline std::array<float, 9> covariance_shape(const std::array<float, 6>& covariance, float max_axis) {
  double a[3][3] = {{covariance[0], covariance[1], covariance[2]},
                    {covariance[1], covariance[3], covariance[4]},
                    {covariance[2], covariance[4], covariance[5]}};
  std::array<float, 9> shape{};
  bool any = false;
  for (auto& row : a)
    for (double v : row) {
      if (!std::isfinite(v)) return shape;
      any |= v != 0;
    }
  if (!any) return shape;
  // Cyclic Jacobi: a -> diag(lambda), v accumulates the rotations (columns = axes).
  double v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int sweep = 0; sweep < 12; ++sweep) {
    const double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    if (off < 1e-30 * (a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2]) + 1e-300) break;
    for (int p = 0; p < 2; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        if (a[p][q] == 0) continue;
        const double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const double t = (theta >= 0 ? 1 : -1) / (std::abs(theta) + std::sqrt(theta * theta + 1));
        const double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < 3; ++k) {  // a <- J^T a J
          const double akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
          const double apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
          const double vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    const double length = std::min(std::sqrt(std::max(a[axis][axis], 0.0)), static_cast<double>(max_axis));
    for (int k = 0; k < 3; ++k) shape[axis * 3 + k] = static_cast<float>(v[k][axis] * length);
  }
  return shape;
}

}  // namespace slam_native
