#pragma once
// Similarity transforms and Sim3 pose-graph optimization for loop closure
// (internal to the visual odometry). Monocular drift includes scale, so loop
// corrections are similarities, not rigid motions.
#include "vo_geometry.hpp"

#include <cstdint>
#include <vector>

namespace slam_native::vo {

// x = s R X + t.
struct Sim3 {
  double s{1};
  Mat3 R{Mat3::Identity()};
  Vec3 t{Vec3::Zero()};
  [[nodiscard]] Vec3 operator*(const Vec3& X) const { return s * (R * X) + t; }
  [[nodiscard]] Sim3 operator*(const Sim3& o) const { return {s * o.s, orthonormalize(R * o.R), s * (R * o.t) + t}; }
  [[nodiscard]] Sim3 inverse() const {
    const Mat3 Rt = R.transpose();
    return {1 / s, Rt, -(Rt * t) / s};
  }
  static Sim3 from(const SE3& T) { return {1, T.R, T.t}; }
  // The rigid part with the camera frame rescaled: x' = x / s (s, R, t) -> (R, t / s).
  [[nodiscard]] SE3 rigid() const { return {R, t / s}; }
};

// Least-squares similarity with Y ~ S X (Umeyama). Needs >= 3 points.
Sim3 align_similarity(const std::vector<Vec3>& X, const std::vector<Vec3>& Y);

// RANSAC over 3-point similarities: correspondence i is an inlier when
// |S X_i - Y_i| <= threshold[i]. The best model is refitted on its inliers.
int ransac_similarity(const std::vector<Vec3>& X, const std::vector<Vec3>& Y, const std::vector<double>& threshold,
                      int iterations, std::uint32_t seed, Sim3& model, std::vector<char>& inliers);

// Nodes are world -> camera similarities. An edge (a, b) measures
// S_a * S_b^-1; its residual is the error transform's
// (log R, t, log s), weighted by sqrt(weight) * sqrt_information
// (U with U^T U = the measurement's information matrix; default identity).
struct PoseGraphEdge {
  int a{}, b{};
  Sim3 measured;
  double weight{1};
  Eigen::Matrix<double, 7, 7> sqrt_information{Eigen::Matrix<double, 7, 7>::Identity()};
};
struct PoseGraphProblem {
  std::vector<Sim3> nodes;
  std::vector<char> fixed;  // per node
  std::vector<PoseGraphEdge> edges;
};
struct PoseGraphReport {
  double initial_cost{}, final_cost{};
  int iterations{};
};
PoseGraphReport optimize_pose_graph(PoseGraphProblem& problem, int iterations = 20);

}  // namespace slam_native::vo
