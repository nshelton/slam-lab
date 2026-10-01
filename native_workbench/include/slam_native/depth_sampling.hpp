#pragma once
// Network depth at track positions: the "sampling policy" of
// DEPTH_INTEGRATION.md. Host-only; the DepthMap must be of the same frame as
// the tracks (DepthMap::frame_index == TrackedFrame::frame_index).
#include "slam_native/depth_estimator.hpp"
#include "slam_native/visual_odometry.hpp"

namespace slam_native {

struct DepthSamplingConfig {
  float relative_sigma{0.15F};    // default 1-sigma / depth (Phase 0 measures it)
  // A 3x3 neighbourhood on the output grid with max/min above 1 + edge_ratio
  // is a depth discontinuity (SuperPoint corners sit on them; networks smear
  // depth across them). Edges report the minimum (the foreground) with
  // edge_sigma_factor times the sigma, or are dropped.
  float edge_ratio{0.10F};
  float edge_sigma_factor{3.0F};
  bool drop_edges{false};
  // Samples at or beyond this (the model's output clamp, DepthModelSpec::
  // max_depth_m) are rejected: the true depth is unknown. <= 0: no limit.
  float max_depth_m{0};
};

struct DepthSample {
  float metres{};  // z-depth; 0 = no valid sample
  float sigma{};
  bool edge{};
  [[nodiscard]] bool valid() const { return metres > 0; }
};

// x, y: native source pixel (as tracks and detections).
DepthSample sample_depth(const DepthMap& depth, float x, float y, const DepthSamplingConfig& config);

// Sets depth_m / depth_sigma_m of every observation (0 where there is no valid
// sample). Throws if the map is of another frame. Returns the valid count;
// `edges` (optional) receives the number of edge samples.
int attach_depth(const DepthMap& depth, TrackedFrame& frame, const DepthSamplingConfig& config,
                 int* edges = nullptr);

}  // namespace slam_native
