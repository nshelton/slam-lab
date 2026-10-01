#pragma once
// A keyframe of the LSD pipeline: image pyramid, semi-dense inverse depth in
// its own units, and its Sim(3) pose in the keyframe graph.
#include "lsd/semi_dense_depth.hpp"
#include "lsd/sim3_tracker.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace slam_native::lsd {

// Appearance features of a keyframe for loop candidates: positions on the
// keyframe's (pinhole, working-resolution) grid and L2-normalised descriptors.
struct KeyframeFeatures {
  std::vector<Vec2> pixels;
  std::vector<float> descriptors;  // pixels.size() x dimension
  int dimension{256};
  std::vector<float> global;       // normalised mean descriptor (retrieval)
  [[nodiscard]] bool empty() const { return pixels.empty(); }
};

struct Keyframe {
  int id{};
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  int parent{-1};
  Sim3 pose;          // camera (this keyframe's depth units) <- world
  SE3 from_parent;    // this camera <- parent camera as tracked (same depth units)
  double scale_correction{1};  // factor applied to the depth prior (scale lock)
  double mean_idepth{};
  int points{};
  int propagated{};  // hypotheses inherited from the parent
  int seeded{};      // hypotheses from the depth prior
  Pyramid pyramid;
  InverseDepthMap depth;  // smoothed inverse depth and variance (what tracking uses)
  TrackingReference reference;
  // Live depth filter while this is the current keyframe; released when finalised.
  std::unique_ptr<SemiDenseDepth> filter;
  int observations{};  // stereo updates received
  // After finalisation: depth of every pyramid level (Sim(3) alignment target).
  bool finalized{};
  std::vector<InverseDepthMap> depth_levels;
  KeyframeFeatures features;  // may be empty (no appearance model)
  // Semi-dense points in the keyframe camera (for the 3D view), their
  // intensity and relative inverse-depth sigma; version changes with them.
  std::vector<std::array<float, 3>> cloud;
  std::vector<std::array<std::uint8_t, 3>> cloud_colors;
  std::vector<float> cloud_sigma;
  std::uint32_t version{};
};

}  // namespace slam_native::lsd
