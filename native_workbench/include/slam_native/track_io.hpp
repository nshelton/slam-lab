#pragma once
// Plain-text interchange for tracks and camera trajectories, so tracks from
// any source can drive VisualOdometry and results can be compared offline.
//
// Tracks CSV:   # size WIDTH HEIGHT            (optional, first line)
//               frame_index,timestamp_ns,track_id,x,y
//               one row per observation; rows grouped by frame in order.
// Trajectory CSV: frame_index,timestamp_ns,segment,keyframe,cx,cy,cz,qw,qx,qy,qz
//               camera centre and camera->world rotation quaternion
//               (Hamilton, w first); arbitrary monocular scale per segment.
#include "slam_native/visual_odometry.hpp"

#include <filesystem>
#include <ostream>
#include <vector>

namespace slam_native {
void write_tracks_header(std::ostream& out, int width, int height);
void write_tracks(std::ostream& out, const TrackedFrame& frame);
// width/height override (or supply) the "# size" line when positive.
std::vector<TrackedFrame> read_tracks_csv(const std::filesystem::path& path, int width = 0, int height = 0);
void write_trajectory_csv(std::ostream& out, const std::vector<TrajectorySample>& trajectory);
}  // namespace slam_native
