#include "lsd/keyframe_graph.hpp"

#include "pose_graph.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>

namespace slam_native::lsd {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}  // namespace

bool KeyframeGraph::connected(int a, int b) const {
  return std::any_of(edges_.begin(), edges_.end(),
                     [&](const GraphEdge& e) { return (e.i == a && e.j == b) || (e.i == b && e.j == a); });
}

KeyframeGraph::Outcome KeyframeGraph::try_pair(const Keyframe& i, const Keyframe& j, const Sim3& initial_ji,
                                               bool loop, ConstraintReport& report) {
  ++report.tried;
  const auto r_ji = track_sim3(i.reference, j.pyramid, j.depth_levels, initial_ji, config_.tracker);
  const auto r_ij = track_sim3(j.reference, i.pyramid, i.depth_levels, initial_ji.inverse(), config_.tracker);
  {
    const Vec7 f = sim3_error(r_ji.pose * initial_ji.inverse()), b = sim3_error(r_ij.pose * initial_ji);
    report.pair_details.push_back({double(i.id), double(j.id), f.head<3>().norm() * 180 / M_PI,
                                   f.segment<3>(3).norm() * j.mean_idepth, b.head<3>().norm() * 180 / M_PI,
                                   b.segment<3>(3).norm() * i.mean_idepth});
  }
  if (!r_ji.success || !r_ij.success) {
    ++report.rejected_tracking;
    return Outcome::tracking_failed;
  }
  const double error = reciprocal_error(r_ji.pose, r_ji.hessian.inverse(), r_ij.pose, r_ij.hessian.inverse());
  report.reciprocal_errors.push_back(error);
  {
    const Vec7 r = sim3_error(r_ji.pose * r_ij.pose);
    report.reciprocal_motion.push_back(
        {r.head<3>().norm() * 180 / M_PI, r.segment<3>(3).norm() * i.mean_idepth, 100 * std::abs(r(6))});
  }
  if (!(error <= config_.max_reciprocal)) {
    ++report.rejected_reciprocal;
    return Outcome::reciprocal_failed;
  }
  GraphEdge edge;
  edge.i = i.id;
  edge.j = j.id;
  edge.S_ji = r_ji.pose;
  edge.information = r_ji.hessian;
  edge.reciprocal = error;
  edge.loop = loop;
  edges_.push_back(edge);
  ++report.accepted;
  return Outcome::accepted;
}

ConstraintReport KeyframeGraph::add_keyframe(std::vector<std::unique_ptr<Keyframe>>& keyframes, int id,
                                             const std::vector<Candidate>& extra) {
  const auto start = Clock::now();
  ConstraintReport report;
  Keyframe& k = *keyframes[id];

  // The parent: aligned from the tracked relative pose (same depth units).
  if (k.parent >= 0 && keyframes[k.parent]->finalized) {
    const Keyframe& parent = *keyframes[k.parent];
    if (try_pair(parent, k, Sim3::from(k.from_parent), false, report) == Outcome::accepted) {
      last_parent_information_ = edges_.back().information;
    } else {
      // Frame tracking's relative pose (scale 1: the depth units are inherited),
      // weighted like the last successful parent alignment.
      GraphEdge edge;
      edge.i = parent.id;
      edge.j = k.id;
      edge.S_ji = Sim3::from(k.from_parent);
      edge.information = last_parent_information_;
      edge.fallback = true;
      edges_.push_back(edge);
      report.parent_fallback = true;
    }
  }

  // Nearby finalised keyframes, nearest first (distance in this keyframe's mean depths).
  const double scene_depth = k.mean_idepth > 0 ? 1.0 / k.mean_idepth : 1.0;
  const Vec3 axis = k.pose.R.transpose().col(2);
  const double min_cos = std::cos(config_.max_angle_deg * M_PI / 180);
  std::vector<std::pair<double, int>> nearby;
  for (const auto& other : keyframes) {
    if (!other || !other->finalized || other->id == k.id || other->id == k.parent) continue;
    if (std::abs(other->id - k.id) < config_.min_keyframe_gap || connected(other->id, k.id)) continue;
    const Vec3 centre_here = k.pose * other->pose.inverse().t;  // its centre in this camera
    const double distance = centre_here.norm() / scene_depth;
    if (distance > config_.max_distance) continue;
    if (axis.dot(other->pose.R.transpose().col(2)) < min_cos) continue;
    nearby.emplace_back(distance, other->id);
  }
  std::sort(nearby.begin(), nearby.end());
  if (static_cast<int>(nearby.size()) > config_.max_neighbours) nearby.resize(config_.max_neighbours);
  for (const auto& [distance, other] : nearby)
    try_pair(k, *keyframes[other], keyframes[other]->pose * k.pose.inverse(), false, report);
  for (const auto& c : extra)
    if (c.id != k.id && keyframes[c.id]->finalized && !connected(c.id, k.id))
      try_pair(k, *keyframes[c.id], c.initial, true, report);
  report.ms = ms_since(start);
  optimize(keyframes, report);
  return report;
}

void KeyframeGraph::optimize(std::vector<std::unique_ptr<Keyframe>>& keyframes, ConstraintReport& report) {
  const auto start = Clock::now();
  vo::PoseGraphProblem problem;
  std::unordered_map<int, int> node;
  for (const auto& kf : keyframes) {
    if (!kf || !kf->finalized) continue;
    node[kf->id] = static_cast<int>(problem.nodes.size());
    problem.nodes.push_back(kf->pose);
    problem.fixed.push_back(kf->id == 0 ? 1 : 0);
  }
  for (const auto& e : edges_) {
    const auto a = node.find(e.j), b = node.find(e.i);
    if (a == node.end() || b == node.end()) continue;
    vo::PoseGraphEdge edge;
    edge.a = a->second;
    edge.b = b->second;
    edge.measured = e.S_ji;
    // Regularise slightly: a degenerate alignment can leave the Hessian singular.
    Mat7 information = e.information;
    information.diagonal().array() += 1e-9 * information.trace() + 1e-12;
    edge.sqrt_information = information.llt().matrixU();
    problem.edges.push_back(edge);
  }
  const auto result = vo::optimize_pose_graph(problem, config_.optimizer_iterations);
  report.initial_cost = result.initial_cost;
  report.final_cost = result.final_cost;
  for (const auto& [id, index] : node) keyframes[id]->pose = problem.nodes[index];
  report.optimize_ms = ms_since(start);
}

}  // namespace slam_native::lsd
