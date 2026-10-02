#pragma once
// Measuring what every-frame observations are worth (SPLINE_BA.md, milestone
// 0). Nothing here changes the visual odometry. The bench keeps every frame's
// observation of each landmark (the fused track position and the raw
// supporting detection), and at the end of the run, with every frame pose
// fixed at its final value, re-solves each landmark from K of them. The export
// has one row per (landmark, keyframe) sighting of the VO's own map with the
// landmark's depth in that keyframe under each variant;
// tools/landmark_depth_vs_gt.py scores it against TUM depth images.
#include "slam_native/visual_odometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace slam_native {

// A frame's observation of a track. Pixels are source pixels as tracked
// (lens-distorted); `u` is the undistorted pixel the VO works with.
struct LandmarkDepthObservation {
  std::uint64_t frame_index{};
  std::uint64_t track_id{};
  float x{}, y{};          // fused track position, as tracked
  float ux{}, uy{};        // the same, undistorted (VisualOdometryConfig::distortion_k1)
  float raw_x{}, raw_y{};  // the supporting detection, as detected (NaN when unknown)
  float raw_ux{}, raw_uy{};
  bool judged{};  // the track observed a landmark on this frame, so the pose solve judged it
  bool inlier{};  // ... and kept it (OdometryFrameResult::pose_inliers)
};

// Every supported observation of every track, with the landmark the VO had
// the track on at the time. A landmark's observations are those of every
// track the VO ever associated with it, followed through merges
// (VisualOdometry::landmark_merges()), including the frames before the
// landmark was triangulated and the frames the pose solve rejected (`inlier`
// tells them apart). A track whose last landmark was deleted contributes
// nothing.
class LandmarkObservationLog {
 public:
  // After odometry.process(frame): `frame` is the frame handed to the VO (as
  // tracked), `raw` the supporting detection per observation of it (NaN when
  // none, e.g. the track was coasted). Records the frame's observations with
  // the VO's current track -> landmark association and inlier split.
  void record(const TrackedFrame& frame, const std::vector<std::array<float, 2>>& raw, const VisualOdometry& odometry);
  // The same from plain data (tests): `track_landmark` and `inliers` sorted by track id.
  void record(const TrackedFrame& frame, const std::vector<std::array<float, 2>>& raw, double k1,
              const std::vector<std::pair<std::uint64_t, std::uint64_t>>& track_landmark,
              const std::vector<std::uint64_t>& inliers);

  // The observations of each landmark of `map`, by frame, after merges.
  [[nodiscard]] std::unordered_map<std::uint64_t, std::vector<LandmarkDepthObservation>> attribute(
      const std::vector<MapPoint>& map, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& merges) const;

  [[nodiscard]] std::size_t size() const { return observations_.size(); }
  [[nodiscard]] int width() const { return width_; }
  [[nodiscard]] int height() const { return height_; }

 private:
  struct Record {
    LandmarkDepthObservation observation;
    std::optional<std::uint64_t> landmark;  // the VO's association on that frame
  };
  std::vector<Record> observations_;  // in frame order
  int width_{}, height_{};
};

// Camera depth of a world point (world -> camera pose).
double depth_in(const Pose& pose, const std::array<double, 3>& X);

// One observation for the fixed-pose re-solve: a frame's pose and the
// undistorted pixel.
struct FixedPoseObservation {
  Pose pose;
  double ux{}, uy{};
};
// Re-solve a point from observations whose poses are fixed: Huber-robust
// (huber_px, in pixels) Levenberg-Marquardt over the three coordinates from
// `initial`. Empty when there are fewer than two observations, the solve does
// not converge, or the point ends up behind a camera.
std::optional<std::array<double, 3>> refit_landmark(const CameraIntrinsics& K,
                                                    const std::vector<FixedPoseObservation>& observations,
                                                    const std::array<double, 3>& initial, double huber_px = 3.0,
                                                    int iterations = 30);
// Indices of k of n observations spread evenly in time: the first and last
// for k = 2. k = 0 means all. Empty when n < k.
std::vector<std::size_t> spread_indices(std::size_t n, std::size_t k);

// The export. One row per (landmark, keyframe) sighting of the VO's map that
// the log also saw (frame_index, landmark_id, segment, track_id, the pixels of
// LandmarkDepthObservation, inlier, observations usable for the re-solve) with
// depth_keyframes, the VO's own landmark depth in that keyframe, and when
// `ks` is not empty, depth_fused_K and depth_raw_K per K in `ks` (K = 0
// written as "all"): the depth of the landmark re-solved from K of its
// observations, fused positions and raw detections respectively, with the
// frame poses fixed at their final values. Usable observations are the frames
// with a tracked (not predicted) pose in the landmark's segment that the pose
// solve did not reject; a variant that cannot be solved writes nan. The
// intrinsics are the VO's.
void write_landmark_depth_csv(std::ostream& out, const VisualOdometry& odometry, const LandmarkObservationLog& log,
                              const std::vector<std::size_t>& ks = {}, double huber_px = 3.0);

}  // namespace slam_native
