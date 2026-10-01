#pragma once
// Direct Sim(3) alignment of two keyframes (LSD-SLAM, ECCV 2014, Sec. 3.5).
//
// Keyframe i's semi-dense points are warped into keyframe j by S_ji (i's
// units -> j's units) and compared photometrically and in inverse depth:
//   r_p = I_j(w(p)) - I_i(p),        sigma_p^2 = 2 sigma_I^2 + (dr_p/dD_i)^2 V_i
//   r_d = 1 / z'(p) - D_j(w(p)),     sigma_d^2 = V_j + (dr_d/dD_i)^2 V_i
// minimising sum huber(sqrt(r_p^2 / sigma_p^2 + r_d^2 / sigma_d^2)): the
// Huber norm is on the sum, since a point that is an outlier in one usually
// is in the other. The depth residual is what makes scale observable; D_j's
// gradient is taken as zero (as in the paper). Levenberg-Marquardt on left
// perturbations S <- exp(delta) S, delta = (omega, v, log s), coarse to fine.
#include "lsd/se3_tracker.hpp"
#include "pose_graph.hpp"

#include <Eigen/Core>
#include <array>
#include <vector>

namespace slam_native::lsd {
using vo::Sim3;
using Mat7 = Eigen::Matrix<double, 7, 7>;
using Vec7 = Eigen::Matrix<double, 7, 1>;

// Inverse-depth maps of every pyramid level (InverseDepthMap reduction).
std::vector<InverseDepthMap> depth_pyramid(const InverseDepthMap& level0, int levels);

struct Sim3TrackerConfig {
  double sigma_intensity{4};
  double huber{3};  // on sqrt(r_p^2 / sigma_p^2 + r_d^2 / sigma_d^2)
  std::array<int, 6> max_iterations{5, 20, 50, 100, 100, 100};
  double convergence_eps{1e-6};
  double min_relative_decrease{0.999};
  int border{1};
  double min_usage{0.2};
  int first_level{-1};  // -1: coarsest available (20x15 at 640x480 with 6 levels)
  int last_level{0};
};

struct Sim3TrackingResult {
  Sim3 pose;  // j <- i
  bool success{};
  Mat7 hessian{Mat7::Zero()};  // finest level used; inverse = covariance of a left perturbation
  double mean_energy{};
  double usage{};
  double depth_fraction{};  // of used points that had a depth residual
  int used_points{};
  int iterations{};
};

// reference: keyframe i; frame / frame_depth: keyframe j's pyramid and depth pyramid.
Sim3TrackingResult track_sim3(const TrackingReference& reference, const Pyramid& frame,
                              const std::vector<InverseDepthMap>& frame_depth, const Sim3& initial,
                              const Sim3TrackerConfig& config);

// The error transform's (log R, t, log s), the pose graph's residual layout.
Vec7 sim3_error(const Sim3& S);
// Adjoint for left perturbations: S exp(delta) = exp(adjoint(S) delta) S.
Mat7 sim3_adjoint(const Sim3& S);
// Reciprocal check (paper eq. 20): Mahalanobis distance of S_ji * S_ij from the
// identity, with covariance cov_ji + Adj(S_ji) cov_ij Adj(S_ji)^T.
double reciprocal_error(const Sim3& S_ji, const Mat7& cov_ji, const Sim3& S_ij, const Mat7& cov_ij);

}  // namespace slam_native::lsd
