#pragma once
// Internal multi-view geometry for the visual odometry (Eigen, CPU).
// Camera convention: x_camera = R * X_world + t (world -> camera).
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstdint>
#include <vector>

namespace slam_native::vo {
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

// Project onto SO(3). Composition and the constant-velocity prediction
// (A * B^-1 * A with R^T as inverse) otherwise amplify round-off shear
// geometrically, and exp(w) * R can never remove a shear.
inline Mat3 orthonormalize(const Mat3& R) { return Eigen::Quaterniond(R).normalized().toRotationMatrix(); }

struct SE3 {
  Mat3 R{Mat3::Identity()};
  Vec3 t{Vec3::Zero()};
  [[nodiscard]] Vec3 operator*(const Vec3& x) const { return R * x + t; }
  [[nodiscard]] SE3 operator*(const SE3& o) const { return {orthonormalize(R * o.R), R * o.t + t}; }
  [[nodiscard]] SE3 inverse() const { return {R.transpose(), -(R.transpose() * t)}; }
  [[nodiscard]] Vec3 center() const { return -(R.transpose() * t); }
};

Mat3 exp_so3(const Vec3& omega);
Vec3 log_so3(const Mat3& R);
// Left perturbation: T <- exp(delta) * T, delta = (omega, v).
SE3 perturb(const SE3& T, const Vec6& delta);
// The delta with a = perturb(b, delta).
Vec6 pose_difference(const SE3& a, const SE3& b);
// Power of a relative motion (for per-frame velocity from a multi-frame motion).
SE3 scale_motion(const SE3& T, double fraction);

struct Intrinsics {
  double fx{1}, fy{1}, cx{0}, cy{0};
  [[nodiscard]] Vec2 project(const Vec3& xc) const { return {fx * xc.x() / xc.z() + cx, fy * xc.y() / xc.z() + cy}; }
  [[nodiscard]] Vec3 unproject(const Vec2& u) const { return {(u.x() - cx) / fx, (u.y() - cy) / fy, 1.0}; }
};

// Two-view estimation on normalized (calibrated) coordinates.
struct TwoViewResult {
  Mat3 model{Mat3::Zero()};
  std::vector<char> inliers;
  int inlier_count{};
};
// threshold is in normalized units (pixels / focal length).
TwoViewResult ransac_essential(const std::vector<Vec2>& a, const std::vector<Vec2>& b, double threshold,
                               int iterations, std::uint32_t seed);
// Uncalibrated: any rank-2 F. Inputs should be roughly unit-scaled (for
// example pixels centred and divided by the image width).
TwoViewResult ransac_fundamental(const std::vector<Vec2>& a, const std::vector<Vec2>& b, double threshold,
                                 int iterations, std::uint32_t seed);
TwoViewResult ransac_homography(const std::vector<Vec2>& a, const std::vector<Vec2>& b, double threshold,
                                int iterations, std::uint32_t seed);
// The four (R, t) candidates of an essential matrix (|t| = 1).
std::vector<SE3> decompose_essential(const Mat3& E);

// Linear triangulation from normalized coordinates. Returns false if degenerate.
bool triangulate(const SE3& T1, const Vec2& x1, const SE3& T2, const Vec2& x2, Vec3& X);
// Angle (radians) between the viewing rays of X from two camera centres.
double parallax(const SE3& T1, const SE3& T2, const Vec3& X);

// Robust (Huber, pixels) pose-only optimization. `inliers` marks observations
// with reprojection error below `inlier_threshold` pixels after optimization.
struct PoseObservation {
  Vec3 X;
  Vec2 u;
};
int optimize_pose(const Intrinsics& K, const std::vector<PoseObservation>& observations, SE3& pose,
                  double inlier_threshold, std::vector<char>& inliers, int iterations = 10);

// d(pixel)/d(delta) of a camera-frame point under perturb.
Eigen::Matrix<double, 2, 6> pose_jacobian(const Intrinsics& K, const Vec3& xc);
// Covariance of an optimized pose's error delta (T_true = perturb(T, delta);
// rotation in radians about the camera centre, translation in map units):
// s^2 * (sum J^T J)^-1 over the inliers, with s^2 their residual variance
// (sum |e|^2 / (2n - 6)). Landmarks are taken as exact, so it is optimistic.
Mat6 pose_covariance(const Intrinsics& K, const std::vector<PoseObservation>& observations, const SE3& pose,
                     const std::vector<char>& inliers);

// Uncertainty of the constant-velocity camera motion model, per frame:
//     T[k+1] = exp(a) * V[k] * T[k],   V[k+1] = exp(a) * V[k],   a ~ N(0, Q)
// (V is the frame-to-frame motion; white-noise acceleration). Errors are
// perturbations as in pose_covariance. One frame's motion is small, so its
// adjoint is taken as the identity: errors of consecutive frames add, and n
// frames without a measurement grow the pose covariance like
//     P + n (C + C^T) + n^2 Pv + Q n(n+1)(2n+1)/6.
// Q is estimated from the one-step prediction errors e of tracked frames.
// With pose measurement noise S (white), e[k] = d[k] - 2 d[k-1] + d[k-2] + a,
// so E[e e^T] = Q + 6 S and E[e[k] e[k-1]^T] = -4 S: measurement jitter shows
// as negative correlation of consecutive errors, and
//     Q = E[e e^T] + 1.5 E[e[k] e[k-1]^T]   (symmetrized, projected to PSD),
// with both expectations exponentially weighted over about the last 60 frames.
// Without that correction the jitter would be extrapolated as acceleration.
struct MotionModel {
  Mat6 pose{Mat6::Zero()};      // P: covariance of the pose error
  Mat6 velocity{Mat6::Zero()};  // Pv: of the velocity error
  Mat6 cross{Mat6::Zero()};     // C: E[pose error * velocity error^T]
  // Start (or restart) from a measured pose; the velocity is the mean motion
  // over `steps` frames between two poses with this covariance.
  void reset(const Mat6& measured, int steps);
  // One frame forward without a measurement.
  void predict();
  // The frame's pose was measured with covariance `measured`, `error` away
  // from the prediction (pose_difference), `steps` frames after the last
  // measured pose; the new velocity is the mean motion over those frames.
  void correct(const Vec6& error, const Mat6& measured, int steps);
  [[nodiscard]] Mat6 acceleration() const;  // Q

 private:
  Mat6 measured_{Mat6::Zero()};  // the last measured pose's covariance
  Mat6 lag0_{Mat6::Zero()}, lag1_{Mat6::Zero()};  // weighted sums of e e^T and e[k] e[k-1]^T
  double weight0_{}, weight1_{};
  Vec6 previous_error_{Vec6::Zero()};
  bool has_previous_error_{};
};

// Perspective-three-point (Grunert's quartic). `bearings` are unit rays in the
// camera frame towards the world points X. Returns up to four candidate
// world -> camera poses.
std::vector<SE3> p3p(const std::array<Vec3, 3>& X, const std::array<Vec3, 3>& bearings);
// P3P RANSAC over 2D-3D matches (u in pixels). Returns the inlier count
// (reprojection error <= threshold_px) of the best hypothesis; `pose` and
// `inliers` describe it. No refinement: follow with optimize_pose.
int ransac_pnp(const Intrinsics& K, const std::vector<PoseObservation>& observations, double threshold_px,
               int iterations, std::uint32_t seed, SE3& pose, std::vector<char>& inliers);

// Sliding-window bundle adjustment with Schur complement Levenberg-Marquardt.
struct BundleObservation {
  int camera;
  int point;
  Vec2 u;
};
struct BundleProblem {
  std::vector<SE3> cameras;
  std::vector<char> fixed;  // per camera
  std::vector<Vec3> points;
  std::vector<BundleObservation> observations;
};
struct BundleReport {
  double initial_cost{}, final_cost{};
  int iterations{};
};
BundleReport bundle_adjust(const Intrinsics& K, BundleProblem& problem, int iterations, double huber_px);

}  // namespace slam_native::vo
