#pragma once
// Dense depth clouds of keyframes (DEPTH_INTEGRATION.md Phase 2). App/bench
// side: the VO never sees depth maps. A cloud is the keyframe's network depth
// back-projected on a subsampled grid of the network's output, in the
// keyframe's camera frame and in metres, coloured from the frame. Viewers draw
// it with the keyframe's current pose (it follows bundle adjustment) and the
// cloud's local scale (metres per VO unit at that keyframe, since the VO's
// scale drifts within a segment).
#include "slam_native/depth_estimator.hpp"
#include "slam_native/visual_odometry.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace slam_native {

struct KeyframeCloud {
  std::uint64_t frame_index{};  // the keyframe's frame (TrajectorySample::frame_index)
  int segment{};
  double metres_per_unit{};  // local scale at this keyframe; 0 = unknown (use the segment's)
  std::vector<std::array<float, 3>> points;  // keyframe camera frame (x right, y down, z forward), metres
  std::vector<std::array<std::uint8_t, 3>> colors;
};

struct KeyframeDepthConfig {
  int stride{4};            // every stride-th output pixel in each direction
  float edge_ratio{0.10F};  // drop pixels whose 3x3 max/min exceeds 1 + this ("flying pixels")
  float max_depth_m{15.0F};
  float max_median_ratio{3.0F};  // and beyond this many times the map's median depth
  std::size_t max_clouds{2000};  // oldest dropped beyond this
};

class KeyframeDepthStore {
 public:
  explicit KeyframeDepthStore(KeyframeDepthConfig config = {}) : config_(config) {}

  // Native source pixels of the grid for maps with `like`'s geometry (cached).
  // Sample the frame's colour there while the frame is alive and pass the
  // colours to add() (the keyframe decision comes later).
  const std::vector<Keypoint>& grid(const DepthMap& like);
  // Builds and stores the cloud of keyframe `frame_index`. `colors` are those
  // sampled at grid(depth) (grey if the sizes differ). K and k1: the VO's lens
  // in source pixels. Returns the number of points kept.
  std::size_t add(std::uint64_t frame_index, int segment, double metres_per_unit, const DepthMap& depth,
                  const std::vector<std::array<std::uint8_t, 3>>& colors, const CameraIntrinsics& K, double k1);

  [[nodiscard]] const std::vector<KeyframeCloud>& clouds() const { return clouds_; }
  void clear();
  // Changes whenever existing clouds are removed (clear, or dropping the oldest),
  // so renderers know to drop their uploaded copies.
  [[nodiscard]] std::uint64_t generation() const { return generation_; }

 private:
  struct Cell {
    int u, v;  // output pixel
  };
  bool same_geometry(const DepthMap& map) const;
  KeyframeDepthConfig config_;
  std::vector<KeyframeCloud> clouds_;
  std::uint64_t generation_{};
  DepthMap geometry_;  // metres empty: only the transform fields are used
  std::vector<Keypoint> grid_;
  std::vector<Cell> cells_;
};

}  // namespace slam_native
