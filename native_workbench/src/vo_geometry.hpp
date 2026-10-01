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
// Each residual is weighted by 1 / sigma^2 (sigma: pixel noise of u); the
// Huber knee and the inlier test stay in pixels. sigma = 1 everywhere is the
// unweighted problem, bit for bit.
// landmark_covariance (px^2): X's own uncertainty projected into the image.
// When set, the residual is measured as a Mahalanobis distance in sigma-pixels,
// d^2 = e^T M e with M = sigma^2 (sigma^2 I + C)^-1, and d replaces |e| in the
// weight, the Huber knee and the inlier test: error along the direction in
// which the landmark is uncertain is neither penalised nor rejected, while
// error across it still is. Zero (default): M = I, the code above exactly.
struct PoseObservation {
  Vec3 X;
  Vec2 u;
  double sigma{1};
  Eigen::Matrix2d landmark_covariance{Eigen::Matrix2d::Zero()};
};
// pixel_gate: classify inliers by the plain pixel error |e| even for
// observations with a landmark covariance (which then only shapes weights).
int optimize_pose(const Intrinsics& K, const std::vector<PoseObservation>& observations, SE3& pose,
                  double inlier_threshold, std::vector<char>& inliers, int iterations = 10,
                  bool pixel_gate = false);

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
  double sigma{1};  // pixel noise of u: weight 1 / sigma^2, as in optimize_pose
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

// Uncertainty: Gauss-Newton information sum J^T J / sigma^2. Diagnostics
// only: no solver uses them.
// Information of a 3D point from its observations with the cameras held fixed.
// Its inverse is the point covariance (world^2 per px^2), which ignores camera
// uncertainty and so is optimistic, but ranks points correctly.
struct PointView {
  SE3 camera;
  Vec2 u;
  double sigma{1};
};
Mat3 point_information(const Intrinsics& K, const Vec3& X, const std::vector<PointView>& views);
// Information of a pose (left perturbation delta = (omega, v), see perturb)
// from the observations marked in `inliers` (all when empty).
Eigen::Matrix<double, 6, 6> pose_information(const Intrinsics& K, const std::vector<PoseObservation>& observations,
                                             const SE3& pose, const std::vector<char>& inliers);
}  // namespace slam_native::vo
