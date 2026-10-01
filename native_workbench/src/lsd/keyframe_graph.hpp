#pragma once
// The keyframe pose graph (LSD-SLAM, ECCV 2014, Secs. 3.5-3.6): Sim(3)
// constraints from direct keyframe alignment, accepted by the reciprocal
// check, and Sim(3) pose-graph optimisation with the alignments' covariances.
#include "lsd/keyframe.hpp"

#include <string>
#include <vector>

namespace slam_native::lsd {

struct GraphEdge {
  int i{}, j{};  // measures S_ji = pose_j * pose_i^-1 (i's units -> j's units)
  Sim3 S_ji;
  Mat7 information{Mat7::Identity()};
  double reciprocal{};   // eq. 20 error of the accepted pair (0 for a fallback edge)
  bool fallback{};       // parent edge from frame tracking (the Sim(3) alignment failed)
  bool loop{};           // from an appearance candidate rather than proximity
};

struct ConstraintConfig {
  ConstraintConfig() { tracker.last_level = 1; }
  Sim3TrackerConfig tracker;      // last_level 1: constraint search at half resolution
  int max_neighbours{10};         // nearby keyframes tried, nearest first (the paper's "closest ten")
  double max_distance{1.5};       // centre distance, in units of the new keyframe's mean depth
  double max_angle_deg{50};       // between optical axes
  // Eq. 20 threshold. The alignment Hessian treats ~50k correlated pixels as
  // independent, so its covariance is ~700x too small: on fr1_desk (2026-10-01)
  // physically consistent pairs (<1.5 deg, <2 % depth, <2 % scale) score up to
  // ~5000 and failed alignments from ~5700.
  double max_reciprocal{5000};
  int min_keyframe_gap{1};        // neighbours at least this many keyframes apart (1: any but itself)
  int optimizer_iterations{20};
};

struct ConstraintReport {
  int tried{}, accepted{}, rejected_tracking{}, rejected_reciprocal{};
  bool parent_fallback{};
  double ms{}, optimize_ms{};
  double initial_cost{}, final_cost{};
  std::vector<double> reciprocal_errors;  // every tried pair's eq. 20 error (diagnostics)
  // Per tried pair: S_ji * S_ij's rotation (deg), translation (in i's mean depths), |log s| (%).
  std::vector<std::array<double, 3>> reciprocal_motion;
  // Per tried pair: i, j, forward and backward change from the initial guess (rotation deg, translation in i's mean depths).
  std::vector<std::array<double, 6>> pair_details;
};

class KeyframeGraph {
 public:
  explicit KeyframeGraph(ConstraintConfig config = {}) : config_(config) {}
  [[nodiscard]] ConstraintConfig& config() { return config_; }
  [[nodiscard]] const std::vector<GraphEdge>& edges() const { return edges_; }

  // Constraints for a keyframe that was just finalised: its parent (always,
  // with a tracking fallback) and nearby finalised keyframes, plus `extra`
  // candidates (loop candidates; initial S_ji estimates given) - then
  // optimises all keyframe poses (keyframe 0 fixed). `keyframes` is indexed by id.
  struct Candidate {
    int id{};
    Sim3 initial;  // S_candidate,new
  };
  ConstraintReport add_keyframe(std::vector<std::unique_ptr<Keyframe>>& keyframes, int id,
                                const std::vector<Candidate>& extra = {});
  // Pose-graph optimisation over finalised keyframes; returns the report's costs.
  void optimize(std::vector<std::unique_ptr<Keyframe>>& keyframes, ConstraintReport& report);

 private:
  // Aligns i and j both ways; on success appends the edge.
  enum class Outcome { accepted, tracking_failed, reciprocal_failed };
  Outcome try_pair(const Keyframe& i, const Keyframe& j, const Sim3& initial_ji, bool loop, ConstraintReport& report);
  [[nodiscard]] bool connected(int a, int b) const;

  ConstraintConfig config_;
  std::vector<GraphEdge> edges_;
  Mat7 last_parent_information_{Mat7::Identity() * 1e4};  // for fallback edges
};

}  // namespace slam_native::lsd
