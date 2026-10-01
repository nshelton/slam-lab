#include "vo_geometry.hpp"


#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <random>

namespace slam_native::vo {
namespace {
Mat3 skew(const Vec3& v) {
  Mat3 m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
}
Eigen::Vector3d homogeneous(const Vec2& x) { return {x.x(), x.y(), 1.0}; }

std::vector<int> sample(std::mt19937& rng, int count, int size) {
  std::vector<int> picked;
  while (static_cast<int>(picked.size()) < size) {
    const int value = static_cast<int>(rng() % static_cast<std::uint32_t>(count));
    if (std::find(picked.begin(), picked.end(), value) == picked.end()) picked.push_back(value);
  }
  return picked;
}

bool fit_essential(const std::vector<Vec2>& a, const std::vector<Vec2>& b, const std::vector<int>& indices,
                   Mat3& E) {
  Eigen::Matrix<double, 9, 9> AtA = Eigen::Matrix<double, 9, 9>::Zero();
  for (int i : indices) {
    const Vec3 x1 = homogeneous(a[i]), x2 = homogeneous(b[i]);
    Eigen::Matrix<double, 9, 1> row;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) row(r * 3 + c) = x2(r) * x1(c);
    AtA += row * row.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> solver(AtA);
  if (solver.info() != Eigen::Success) return false;
  const Eigen::Matrix<double, 9, 1> e = solver.eigenvectors().col(0);
  Mat3 raw;
  raw << e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8);
  Eigen::JacobiSVD<Mat3> svd(raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
  E = svd.matrixU() * Eigen::Vector3d(1, 1, 0).asDiagonal() * svd.matrixV().transpose();
  return E.allFinite();
}

// Eight-point fundamental matrix: like fit_essential but only rank 2 is
// enforced (the two non-zero singular values stay free).
bool fit_fundamental(const std::vector<Vec2>& a, const std::vector<Vec2>& b, const std::vector<int>& indices,
                     Mat3& F) {
  Eigen::Matrix<double, 9, 9> AtA = Eigen::Matrix<double, 9, 9>::Zero();
  for (int i : indices) {
    const Vec3 x1 = homogeneous(a[i]), x2 = homogeneous(b[i]);
    Eigen::Matrix<double, 9, 1> row;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) row(r * 3 + c) = x2(r) * x1(c);
    AtA += row * row.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> solver(AtA);
  if (solver.info() != Eigen::Success) return false;
  const Eigen::Matrix<double, 9, 1> f = solver.eigenvectors().col(0);
  Mat3 raw;
  raw << f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8);
  Eigen::JacobiSVD<Mat3> svd(raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d sigma = svd.singularValues();
  sigma(2) = 0;
  F = svd.matrixU() * sigma.asDiagonal() * svd.matrixV().transpose();
  return F.allFinite() && sigma(0) > 0;
}

double sampson(const Mat3& E, const Vec2& a, const Vec2& b) {
  const Vec3 x1 = homogeneous(a), x2 = homogeneous(b);
  const Vec3 Ex1 = E * x1, Etx2 = E.transpose() * x2;
  const double value = x2.dot(Ex1);
  const double denominator = Ex1.x() * Ex1.x() + Ex1.y() * Ex1.y() + Etx2.x() * Etx2.x() + Etx2.y() * Etx2.y();
  return denominator > 0 ? value * value / denominator : INFINITY;
}

bool fit_homography(const std::vector<Vec2>& a, const std::vector<Vec2>& b, const std::vector<int>& indices,
                    Mat3& H) {
  Eigen::MatrixXd A(2 * indices.size(), 9);
  int row = 0;
  for (int i : indices) {
    const double x = a[i].x(), y = a[i].y(), u = b[i].x(), v = b[i].y();
    A.row(row++) << -x, -y, -1, 0, 0, 0, u * x, u * y, u;
    A.row(row++) << 0, 0, 0, -x, -y, -1, v * x, v * y, v;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::VectorXd h = svd.matrixV().col(8);
  H << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
  return H.allFinite() && std::abs(H.determinant()) > 1e-12;
}

double transfer(const Mat3& H, const Vec2& a, const Vec2& b) {
  const Vec3 p = H * homogeneous(a);
  if (std::abs(p.z()) < 1e-12) return INFINITY;
  return (p.head<2>() / p.z() - b).squaredNorm();
}

template <class Fit, class Error>
TwoViewResult ransac(const std::vector<Vec2>& a, const std::vector<Vec2>& b, int minimal, double threshold,
                     int iterations, std::uint32_t seed, Fit fit, Error error) {
  TwoViewResult best;
  const int count = static_cast<int>(a.size());
  best.inliers.assign(count, 0);
  if (count < minimal) return best;
  std::mt19937 rng(seed);
  const double t2 = threshold * threshold;
  const auto score = [&](const Mat3& model, std::vector<char>& inliers) {
    int n = 0;
    for (int i = 0; i < count; ++i) n += (inliers[i] = error(model, a[i], b[i]) <= t2);
    return n;
  };
  std::vector<char> inliers(count);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    Mat3 model;
    if (!fit(a, b, sample(rng, count, minimal), model)) continue;
    const int n = score(model, inliers);
    if (n > best.inlier_count) {
      best.inlier_count = n;
      best.model = model;
      best.inliers = inliers;
    }
  }
  // Refit on all inliers (two passes).
  for (int pass = 0; pass < 2 && best.inlier_count >= minimal; ++pass) {
    std::vector<int> support;
    for (int i = 0; i < count; ++i) if (best.inliers[i]) support.push_back(i);
    Mat3 model;
    if (!fit(a, b, support, model)) break;
    const int n = score(model, inliers);
    if (n < best.inlier_count) break;
    best.inlier_count = n;
    best.model = model;
    best.inliers = inliers;
  }
  return best;
}

double huber_cost(double squared, double delta) {
  const double norm = std::sqrt(squared);
  return norm <= delta ? squared : 2 * delta * norm - delta * delta;
}
double huber_weight(double squared, double delta) {
  const double norm = std::sqrt(squared);
  return norm <= delta ? 1.0 : delta / norm;
}
// d(pixel)/d(camera point)
Eigen::Matrix<double, 2, 3> projection_jacobian(const Intrinsics& K, const Vec3& xc) {
  const double z = xc.z(), z2 = z * z;
  Eigen::Matrix<double, 2, 3> J;
  J << K.fx / z, 0, -K.fx * xc.x() / z2, 0, K.fy / z, -K.fy * xc.y() / z2;
  return J;
}
Eigen::Matrix<double, 3, 6> perturbation_jacobian(const Vec3& xc) {
  Eigen::Matrix<double, 3, 6> J;
  J.leftCols<3>() = -skew(xc);
  J.rightCols<3>() = Mat3::Identity();
  return J;
}
constexpr double kMinDepth = 1e-6;
}  // namespace

Mat3 exp_so3(const Vec3& omega) {
  const double angle = omega.norm();
  if (angle < 1e-12) return Mat3::Identity() + skew(omega);
  return Eigen::AngleAxisd(angle, omega / angle).toRotationMatrix();
}
Vec3 log_so3(const Mat3& R) {
  const Eigen::AngleAxisd aa(R);
  return aa.angle() * aa.axis();
}
SE3 perturb(const SE3& T, const Vec6& delta) {
  const Mat3 dR = exp_so3(delta.head<3>());
  return {orthonormalize(dR * T.R), dR * T.t + delta.tail<3>()};
}
SE3 scale_motion(const SE3& T, double fraction) {
  return {exp_so3(log_so3(T.R) * fraction), T.t * fraction};
}

TwoViewResult ransac_essential(const std::vector<Vec2>& a, const std::vector<Vec2>& b, double threshold,
                               int iterations, std::uint32_t seed) {
  return ransac(a, b, 8, threshold, iterations, seed, fit_essential, sampson);
}
TwoViewResult ransac_fundamental(const std::vector<Vec2>& a, const std::vector<Vec2>& b, double threshold,
                                 int iterations, std::uint32_t seed) {
  return ransac(a, b, 8, threshold, iterations, seed, fit_fundamental, sampson);
}
TwoViewResult ransac_homography(const std::vector<Vec2>& a, const std::vector<Vec2>& b, double threshold,
                                int iterations, std::uint32_t seed) {
  return ransac(a, b, 4, threshold, iterations, seed, fit_homography,
                [](const Mat3& H, const Vec2& x, const Vec2& y) {
                  const Mat3 inverse = H.inverse();
                  return std::max(transfer(H, x, y), transfer(inverse, y, x));
                });
}

std::vector<SE3> decompose_essential(const Mat3& E) {
  Eigen::JacobiSVD<Mat3> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 U = svd.matrixU(), V = svd.matrixV();
  if (U.determinant() < 0) U = -U;
  if (V.determinant() < 0) V = -V;
  Mat3 W;
  W << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  const Mat3 R1 = U * W * V.transpose(), R2 = U * W.transpose() * V.transpose();
  const Vec3 t = U.col(2).normalized();
  return {{R1, t}, {R1, -t}, {R2, t}, {R2, -t}};
}

bool triangulate(const SE3& T1, const Vec2& x1, const SE3& T2, const Vec2& x2, Vec3& X) {
  Eigen::Matrix<double, 3, 4> P1, P2;
  P1 << T1.R, T1.t;
  P2 << T2.R, T2.t;
  Eigen::Matrix4d A;
  A.row(0) = x1.x() * P1.row(2) - P1.row(0);
  A.row(1) = x1.y() * P1.row(2) - P1.row(1);
  A.row(2) = x2.x() * P2.row(2) - P2.row(0);
  A.row(3) = x2.y() * P2.row(2) - P2.row(1);
  Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
  const Eigen::Vector4d v = svd.matrixV().col(3);
  if (std::abs(v(3)) < 1e-12) return false;
  X = v.head<3>() / v(3);
  return X.allFinite();
}

double parallax(const SE3& T1, const SE3& T2, const Vec3& X) {
  const Vec3 r1 = X - T1.center(), r2 = X - T2.center();
  const double c = r1.dot(r2) / (r1.norm() * r2.norm());
  return std::acos(std::clamp(c, -1.0, 1.0));
}

std::vector<SE3> p3p(const std::array<Vec3, 3>& X, const std::array<Vec3, 3>& f) {
  // Distances along the rays s_i; s2 = u s1, s3 = v s1 (Haralick et al. 1994, Grunert).
  const double a2 = (X[1] - X[2]).squaredNorm(), b2 = (X[0] - X[2]).squaredNorm(), c2 = (X[0] - X[1]).squaredNorm();
  if (a2 < 1e-18 || b2 < 1e-18 || c2 < 1e-18) return {};
  const double ca = f[1].dot(f[2]), cb = f[0].dot(f[2]), cg = f[0].dot(f[1]);
  const double p = (a2 - c2) / b2, q = (a2 + c2) / b2, ab = a2 / b2, cbb = c2 / b2, bc = (b2 - c2) / b2,
               ba = (b2 - a2) / b2;
  Eigen::Matrix<double, 5, 1> A;  // A(k) multiplies v^k
  A(4) = (p - 1) * (p - 1) - 4 * cbb * ca * ca;
  A(3) = 4 * (p * (1 - p) * cb - (1 - q) * ca * cg + 2 * cbb * ca * ca * cb);
  A(2) = 2 * (p * p - 1 + 2 * p * p * cb * cb + 2 * bc * ca * ca - 4 * q * ca * cb * cg + 2 * ba * cg * cg);
  A(1) = 4 * (-p * (1 + p) * cb + 2 * ab * cg * cg * cb - (1 - q) * ca * cg);
  A(0) = (1 + p) * (1 + p) - 4 * ab * cg * cg;
  if (std::abs(A(4)) < 1e-12) return {};
  Eigen::Matrix4d companion = Eigen::Matrix4d::Zero();
  for (int k = 0; k < 4; ++k) companion(0, k) = -A(3 - k) / A(4);
  for (int k = 1; k < 4; ++k) companion(k, k - 1) = 1;
  const Eigen::EigenSolver<Eigen::Matrix4d> roots(companion, false);
  std::vector<SE3> poses;
  for (int r = 0; r < 4; ++r) {
    if (std::abs(roots.eigenvalues()(r).imag()) > 1e-6 * (1 + std::abs(roots.eigenvalues()(r).real()))) continue;
    double v = roots.eigenvalues()(r).real();
    for (int it = 0; it < 3; ++it) {  // Newton polish
      double value = 0, slope = 0;
      for (int k = 4; k >= 0; --k) {
        slope = slope * v + value;
        value = value * v + A(k);
      }
      if (std::abs(slope) > 1e-15) v -= value / slope;
    }
    if (v <= 0) continue;
    const double d1 = 1 + v * v - 2 * v * cb;
    if (d1 <= 1e-12) continue;
    const double s1 = std::sqrt(b2 / d1), s3 = v * s1;
    // s2 from the c-equation (two roots); keep the one satisfying the a-equation.
    const double disc = cg * cg - 1 + c2 / (s1 * s1);
    if (disc < -1e-9) continue;
    double best_error = 1e300, s2 = 0;
    for (const double sign : {-1.0, 1.0}) {
      const double u = cg + sign * std::sqrt(std::max(disc, 0.0));
      if (u <= 0) continue;
      const double candidate = u * s1;
      const double error = std::abs(candidate * candidate + s3 * s3 - 2 * candidate * s3 * ca - a2);
      if (error < best_error) { best_error = error; s2 = candidate; }
    }
    if (s2 <= 0 || best_error > 1e-3 * a2) continue;
    // Absolute orientation: camera points Q_i = s_i f_i = R X_i + t (Kabsch).
    const std::array<Vec3, 3> Q{s1 * f[0], s2 * f[1], s3 * f[2]};
    const Vec3 mx = (X[0] + X[1] + X[2]) / 3, mq = (Q[0] + Q[1] + Q[2]) / 3;
    Mat3 H = Mat3::Zero();
    for (int i = 0; i < 3; ++i) H += (X[i] - mx) * (Q[i] - mq).transpose();
    const Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 D = Mat3::Identity();
    D(2, 2) = (svd.matrixV() * svd.matrixU().transpose()).determinant() < 0 ? -1 : 1;
    const Mat3 R = svd.matrixV() * D * svd.matrixU().transpose();
    poses.push_back({R, mq - R * mx});
  }
  return poses;
}

int ransac_pnp(const Intrinsics& K, const std::vector<PoseObservation>& observations, double threshold_px,
               int iterations, std::uint32_t seed, SE3& pose, std::vector<char>& inliers) {
  const int count = static_cast<int>(observations.size());
  inliers.assign(count, 0);
  if (count < 4) return 0;
  std::vector<Vec3> bearings(count);
  for (int i = 0; i < count; ++i) bearings[i] = K.unproject(observations[i].u).normalized();
  std::mt19937 rng(seed);
  const double t2 = threshold_px * threshold_px;
  int best = 0;
  std::vector<char> current(count);
  for (int it = 0; it < iterations; ++it) {
    const auto picked = sample(rng, count, 3);
    const std::array<Vec3, 3> X{observations[picked[0]].X, observations[picked[1]].X, observations[picked[2]].X};
    const std::array<Vec3, 3> f{bearings[picked[0]], bearings[picked[1]], bearings[picked[2]]};
    for (const auto& T : p3p(X, f)) {
      int n = 0;
      for (int i = 0; i < count; ++i) {
        const Vec3 xc = T * observations[i].X;
        current[i] = xc.z() > 1e-9 && (K.project(xc) - observations[i].u).squaredNorm() <= t2;
        n += current[i];
      }
      if (n > best) {
        best = n;
        pose = T;
        inliers = current;
      }
    }
  }
  return best;
}

int optimize_pose(const Intrinsics& K, const std::vector<PoseObservation>& observations, SE3& pose,
                  double inlier_threshold, std::vector<char>& inliers, int iterations) {
  const int count = static_cast<int>(observations.size());
  inliers.assign(count, 1);
  const auto evaluate = [&](const SE3& T) {
    double cost = 0;
    for (int i = 0; i < count; ++i) {
      if (!inliers[i]) continue;
      const Vec3 xc = T * observations[i].X;
      if (xc.z() < kMinDepth) { cost += 1e6; continue; }
      cost += huber_cost((K.project(xc) - observations[i].u).squaredNorm(), inlier_threshold);
    }
    return cost;
  };
  for (int stage = 0; stage < 2; ++stage) {
    double lambda = 1e-4;
    double cost = evaluate(pose);
    for (int iteration = 0; iteration < iterations; ++iteration) {
      Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
      Vec6 g = Vec6::Zero();
      for (int i = 0; i < count; ++i) {
        if (!inliers[i]) continue;
        const Vec3 xc = pose * observations[i].X;
        if (xc.z() < kMinDepth) continue;
        const Vec2 e = K.project(xc) - observations[i].u;
        const Eigen::Matrix<double, 2, 6> J = projection_jacobian(K, xc) * perturbation_jacobian(xc);
        const double w = huber_weight(e.squaredNorm(), inlier_threshold);
        H += w * J.transpose() * J;
        g -= w * J.transpose() * e;
      }
      bool improved = false;
      for (int attempt = 0; attempt < 6 && !improved; ++attempt) {
        Eigen::Matrix<double, 6, 6> damped = H;
        damped.diagonal() += lambda * (H.diagonal().array() + 1e-9).matrix();
        const Vec6 delta = damped.ldlt().solve(g);
        if (!delta.allFinite()) break;
        const SE3 candidate = perturb(pose, delta);
        const double next = evaluate(candidate);
        if (next < cost) {
          pose = candidate;
          improved = true;
          lambda = std::max(lambda * 0.1, 1e-9);
          const bool converged = delta.norm() < 1e-9 || cost - next < 1e-9 * cost;
          cost = next;
          if (converged) iteration = iterations;
        } else {
          lambda *= 10;
        }
      }
      if (!improved) break;
    }
    // Classify, then refine on inliers only.
    for (int i = 0; i < count; ++i) {
      const Vec3 xc = pose * observations[i].X;
      inliers[i] = xc.z() > kMinDepth &&
          (K.project(xc) - observations[i].u).squaredNorm() <= inlier_threshold * inlier_threshold;
    }
  }
  return static_cast<int>(std::count(inliers.begin(), inliers.end(), 1));
}

BundleReport bundle_adjust(const Intrinsics& K, BundleProblem& problem, int iterations, double huber_px) {
  BundleReport report;
  const int cameras = static_cast<int>(problem.cameras.size());
  const int points = static_cast<int>(problem.points.size());
  std::vector<int> variable(cameras, -1);
  int free = 0;
  for (int c = 0; c < cameras; ++c) if (!problem.fixed[c]) variable[c] = free++;
  std::vector<std::vector<int>> by_point(points);
  for (int o = 0; o < static_cast<int>(problem.observations.size()); ++o)
    by_point[problem.observations[o].point].push_back(o);

  const auto evaluate = [&](const std::vector<SE3>& T, const std::vector<Vec3>& X) {
    double cost = 0;
    for (const auto& ob : problem.observations) {
      const Vec3 xc = T[ob.camera] * X[ob.point];
      if (xc.z() < kMinDepth) { cost += 2 * huber_px * 100; continue; }
      cost += huber_cost((K.project(xc) - ob.u).squaredNorm(), huber_px);
    }
    return cost;
  };
  double cost = evaluate(problem.cameras, problem.points);
  report.initial_cost = cost;
  double lambda = 1e-3;
  const int n = 6 * free;
  std::vector<Eigen::Matrix<double, 6, 6>> Hcc(cameras);
  std::vector<Vec6> bc(cameras);
  std::vector<Mat3> Hpp(points);
  std::vector<Vec3> bp(points);
  std::vector<Eigen::Matrix<double, 6, 3>> W(problem.observations.size());
  for (int iteration = 0; iteration < iterations; ++iteration) {
    for (auto& m : Hcc) m.setZero();
    for (auto& v : bc) v.setZero();
    for (auto& m : Hpp) m.setZero();
    for (auto& v : bp) v.setZero();
    for (std::size_t o = 0; o < problem.observations.size(); ++o) {
      const auto& ob = problem.observations[o];
      const SE3& T = problem.cameras[ob.camera];
      const Vec3 xc = T * problem.points[ob.point];
      W[o].setZero();
      if (xc.z() < kMinDepth) continue;
      const Vec2 e = K.project(xc) - ob.u;
      const double w = huber_weight(e.squaredNorm(), huber_px);
      const Eigen::Matrix<double, 2, 3> Jproj = projection_jacobian(K, xc);
      const Eigen::Matrix<double, 2, 3> Jp = Jproj * T.R;
      Hpp[ob.point] += w * Jp.transpose() * Jp;
      bp[ob.point] -= w * Jp.transpose() * e;
      if (variable[ob.camera] >= 0) {
        const Eigen::Matrix<double, 2, 6> Jc = Jproj * perturbation_jacobian(xc);
        Hcc[ob.camera] += w * Jc.transpose() * Jc;
        bc[ob.camera] -= w * Jc.transpose() * e;
        W[o] = w * Jc.transpose() * Jp;
      }
    }
    bool improved = false;
    for (int attempt = 0; attempt < 8 && !improved; ++attempt) {
      Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n, n);
      Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
      for (int c = 0; c < cameras; ++c) {
        if (variable[c] < 0) continue;
        Eigen::Matrix<double, 6, 6> block = Hcc[c];
        block.diagonal() += lambda * (Hcc[c].diagonal().array() + 1e-9).matrix();
        S.block<6, 6>(6 * variable[c], 6 * variable[c]) += block;
        rhs.segment<6>(6 * variable[c]) += bc[c];
      }
      std::vector<Mat3> inverse(points);
      for (int p = 0; p < points; ++p) {
        Mat3 damped = Hpp[p];
        damped.diagonal() += lambda * (Hpp[p].diagonal().array() + 1e-9).matrix();
        inverse[p] = damped.inverse();
        if (!inverse[p].allFinite()) inverse[p].setZero();
        for (int a : by_point[p]) {
          const int ca = variable[problem.observations[a].camera];
          if (ca < 0) continue;
          const Eigen::Matrix<double, 6, 3> Ta = W[a] * inverse[p];
          rhs.segment<6>(6 * ca) -= Ta * bp[p];
          for (int b : by_point[p]) {
            const int cb = variable[problem.observations[b].camera];
            if (cb >= 0) S.block<6, 6>(6 * ca, 6 * cb) -= Ta * W[b].transpose();
          }
        }
      }
      Eigen::VectorXd dx = n > 0 ? Eigen::VectorXd(S.ldlt().solve(rhs)) : Eigen::VectorXd();
      if (n > 0 && !dx.allFinite()) { lambda *= 10; continue; }
      std::vector<SE3> cameras_next = problem.cameras;
      for (int c = 0; c < cameras; ++c)
        if (variable[c] >= 0) cameras_next[c] = perturb(problem.cameras[c], dx.segment<6>(6 * variable[c]));
      std::vector<Vec3> points_next = problem.points;
      for (int p = 0; p < points; ++p) {
        Vec3 r = bp[p];
        for (int a : by_point[p]) {
          const int ca = variable[problem.observations[a].camera];
          if (ca >= 0) r -= W[a].transpose() * dx.segment<6>(6 * ca);
        }
        points_next[p] += inverse[p] * r;
      }
      const double next = evaluate(cameras_next, points_next);
      if (next < cost) {
        improved = true;
        const bool converged = cost - next < 1e-7 * cost;
        problem.cameras = std::move(cameras_next);
        problem.points = std::move(points_next);
        cost = next;
        lambda = std::max(lambda * 0.1, 1e-9);
        ++report.iterations;
        if (converged) iteration = iterations;
      } else {
        lambda *= 10;
      }
    }
    if (!improved) break;
  }
  report.final_cost = cost;
  return report;
}

}  // namespace slam_native::vo
