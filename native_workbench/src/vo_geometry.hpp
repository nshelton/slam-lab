#pragma once
// Internal multi-view geometry for the visual odometry (Eigen, CPU).
// Camera convention: x_camera = R * X_world + t (world -> camera).
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <vector>

namespace slam_native::vo {
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;

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
