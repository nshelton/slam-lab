#pragma once
// Loop candidates from appearance (the paper's FAB-MAP role, LSD_SLAM.md):
// keyframes carry SuperPoint features; a pooled global descriptor shortlists
// old keyframes, mutual nearest-neighbour matching (cosine >= 0.7, ratio 0.9)
// pairs features, and the pairs' 3D points from both keyframes' semi-dense
// depth give an initial S_ji by similarity RANSAC (Horn / Umeyama; the paper
// suggests this to extend the convergence radius of large loop closures).
// The candidate is then verified by Sim(3) alignment and the reciprocal check.
#include "lsd/keyframe.hpp"

#include <memory>
#include <vector>

namespace slam_native::lsd {

struct LoopConfig {
  bool enabled{true};
  int min_keyframe_gap{15};     // candidates at least this many keyframes older
  int shortlist{5};             // by global-descriptor similarity
  double min_cosine{0.7};
  double ratio{0.9};            // best distance < ratio * second-best distance
  int min_matches{20};
  int min_inliers{12};          // similarity RANSAC
  double inlier_fraction{0.08};  // 3D threshold: this fraction of the point's depth
  int ransac_iterations{300};
  int max_candidates{1};
};

struct LoopCandidate {
  int id{};          // the old keyframe
  Sim3 initial;      // S_old,new
  int matches{}, inliers{};
  double similarity{};  // global descriptor cosine
};

// Normalised mean of the descriptors (sets features.global).
void finish_features(KeyframeFeatures& features);

// Candidates for keyframe `id` (finalised, with features) among older finalised keyframes.
std::vector<LoopCandidate> find_loop_candidates(const std::vector<std::unique_ptr<Keyframe>>& keyframes, int id,
                                                const LoopConfig& config);

}  // namespace slam_native::lsd
