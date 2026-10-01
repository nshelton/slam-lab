#include "pose_graph.hpp"

#include <Eigen/SVD>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <random>

namespace slam_native::vo {
namespace {
using Vec7 = Eigen::Matrix<double, 7, 1>;

// Left perturbation: S <- exp(delta) S, delta = (omega, v, sigma).
Sim3 perturb(const Sim3& S, const Vec7& d) {
  const Sim3 step{std::exp(d(6)), exp_so3(d.head<3>()), d.segment<3>(3)};
  return step * S;
}

Vec7 residual(const PoseGraphEdge& e, const Sim3& a, const Sim3& b) {
  const Sim3 error = e.measured * b * a.inverse();
  Vec7 r;
  r.head<3>() = log_so3(error.R);
  r.segment<3>(3) = error.t;
  r(6) = std::log(error.s);
  return std::sqrt(e.weight) * r;
}

double cost(const PoseGraphProblem& p) {
  double total = 0;
  for (const auto& e : p.edges) total += residual(e, p.nodes[e.a], p.nodes[e.b]).squaredNorm();
  return total;
}
}  // namespace

Sim3 align_similarity(const std::vector<Vec3>& X, const std::vector<Vec3>& Y) {
  const std::size_t n = X.size();
  Vec3 mx = Vec3::Zero(), my = Vec3::Zero();
  for (std::size_t i = 0; i < n; ++i) {
    mx += X[i];
    my += Y[i];
  }
  mx /= static_cast<double>(n);
  my /= static_cast<double>(n);
  Mat3 C = Mat3::Zero();
  double var = 0;
  for (std::size_t i = 0; i < n; ++i) {
    C += (Y[i] - my) * (X[i] - mx).transpose();
    var += (X[i] - mx).squaredNorm();
  }
  C /= static_cast<double>(n);
  var /= static_cast<double>(n);
  const Eigen::JacobiSVD<Mat3> svd(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 D = Mat3::Identity();
  if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0) D(2, 2) = -1;
  Sim3 S;
  S.R = svd.matrixU() * D * svd.matrixV().transpose();
  S.s = var > 1e-18 ? (svd.singularValues().asDiagonal() * D).trace() / var : 1.0;
  S.t = my - S.s * (S.R * mx);
  return S;
}

int ransac_similarity(const std::vector<Vec3>& X, const std::vector<Vec3>& Y, const std::vector<double>& threshold,
                      int iterations, std::uint32_t seed, Sim3& model, std::vector<char>& inliers) {
  const int n = static_cast<int>(X.size());
  inliers.assign(n, 0);
  if (n < 3) return 0;
  std::mt19937 rng(seed);
  const auto count = [&](const Sim3& S, std::vector<char>& marks) {
    int k = 0;
    for (int i = 0; i < n; ++i) {
      marks[i] = (S * X[i] - Y[i]).norm() <= threshold[i];
      k += marks[i];
    }
    return k;
  };
  int best = 0;
  std::vector<char> marks(n);
  for (int it = 0; it < iterations; ++it) {
    int a = static_cast<int>(rng() % n), b = static_cast<int>(rng() % n), c = static_cast<int>(rng() % n);
    if (a == b || b == c || a == c) continue;
    const Sim3 S = align_similarity({X[a], X[b], X[c]}, {Y[a], Y[b], Y[c]});
    if (!(S.s > 0) || !std::isfinite(S.s)) continue;
    const int k = count(S, marks);
    if (k > best) {
      best = k;
      model = S;
      inliers = marks;
    }
  }
  // Refit on the inliers, twice (the inlier set can grow).
  for (int round = 0; round < 2 && best >= 3; ++round) {
    std::vector<Vec3> x, y;
    for (int i = 0; i < n; ++i)
      if (inliers[i]) {
        x.push_back(X[i]);
        y.push_back(Y[i]);
      }
    const Sim3 S = align_similarity(x, y);
    const int k = count(S, marks);
    if (k < best) break;
    best = k;
    model = S;
    inliers = marks;
  }
  return best;
}

PoseGraphReport optimize_pose_graph(PoseGraphProblem& problem, int iterations) {
  PoseGraphReport report;
  const int n = static_cast<int>(problem.nodes.size());
  // Free nodes get consecutive variable blocks.
  std::vector<int> block(n, -1);
  int free = 0;
  for (int i = 0; i < n; ++i)
    if (!problem.fixed[i]) block[i] = free++;
  report.initial_cost = report.final_cost = cost(problem);
  if (free == 0 || problem.edges.empty()) return report;
  double lambda = 1e-4;
  constexpr double kStep = 1e-6;
  for (int it = 0; it < iterations; ++it) {
    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::VectorXd g = Eigen::VectorXd::Zero(7 * free);
    for (const auto& e : problem.edges) {
      const Vec7 r = residual(e, problem.nodes[e.a], problem.nodes[e.b]);
      // Numerical Jacobians w.r.t. the two endpoints' left perturbations.
      Eigen::Matrix<double, 7, 7> J[2];
      const int ends[2] = {e.a, e.b};
      for (int side = 0; side < 2; ++side) {
        if (block[ends[side]] < 0) continue;
        for (int k = 0; k < 7; ++k) {
          Vec7 d = Vec7::Zero();
          d(k) = kStep;
          Sim3 plus_a = problem.nodes[e.a], plus_b = problem.nodes[e.b];
          Sim3 minus_a = plus_a, minus_b = plus_b;
          (side == 0 ? plus_a : plus_b) = perturb(side == 0 ? plus_a : plus_b, d);
          (side == 0 ? minus_a : minus_b) = perturb(side == 0 ? minus_a : minus_b, -d);
          J[side].col(k) = (residual(e, plus_a, plus_b) - residual(e, minus_a, minus_b)) / (2 * kStep);
        }
      }
      for (int p = 0; p < 2; ++p) {
        if (block[ends[p]] < 0) continue;
        g.segment<7>(7 * block[ends[p]]) += J[p].transpose() * r;
        for (int q = 0; q < 2; ++q) {
          if (block[ends[q]] < 0) continue;
          const Eigen::Matrix<double, 7, 7> H = J[p].transpose() * J[q];
          for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
              triplets.emplace_back(7 * block[ends[p]] + i, 7 * block[ends[q]] + j, H(i, j));
        }
      }
    }
    Eigen::SparseMatrix<double> H(7 * free, 7 * free);
    H.setFromTriplets(triplets.begin(), triplets.end());
    const double before = cost(problem);
    bool improved = false;
    for (int attempt = 0; attempt < 8 && !improved; ++attempt) {
      Eigen::SparseMatrix<double> A = H;
      for (int i = 0; i < 7 * free; ++i) A.coeffRef(i, i) += lambda * (1 + H.coeff(i, i));
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver(A);
      if (solver.info() != Eigen::Success) {
        lambda *= 10;
        continue;
      }
      const Eigen::VectorXd delta = solver.solve(-g);
      std::vector<Sim3> saved = problem.nodes;
      for (int i = 0; i < n; ++i)
        if (block[i] >= 0) problem.nodes[i] = perturb(problem.nodes[i], delta.segment<7>(7 * block[i]));
      const double after = cost(problem);
      if (after < before) {
        improved = true;
        lambda = std::max(lambda / 10, 1e-9);
        report.final_cost = after;
      } else {
        problem.nodes = std::move(saved);
        lambda *= 10;
      }
    }
    report.iterations = it + 1;
    if (!improved || before - report.final_cost < 1e-10 * std::max(1.0, before)) break;
  }
  return report;
}

}  // namespace slam_native::vo
