#pragma once
// Plain-text interchange for tracks and camera trajectories, so tracks from
// any source can drive VisualOdometry and results can be compared offline.
//
// Tracks CSV:   # size WIDTH HEIGHT            (optional, first line)
//               frame_index,timestamp_ns,track_id,x,y[,r,g,b]
//               one row per observation; rows grouped by frame in order.
//               The optional r,g,b columns (0-255) are the image colour at
//               the observation. The reader finds them by name in the header
//               row and ignores other extra columns.
// Trajectory CSV: frame_index,timestamp_ns,segment,keyframe,cx,cy,cz,qw,qx,qy,qz
//               camera centre and camera->world rotation quaternion
//               (Hamilton, w first); arbitrary monocular scale per segment.
// Map CSV:      track_id,segment,x,y,z,r,g,b,has_color,keyframe_observations,
//               first_frame,last_frame,retired,landmark_id
//               (world coordinates, as the trajectory; r,g,b are 0 when
//               has_color is 0)
#include "slam_native/visual_odometry.hpp"

#include <filesystem>
#include <ostream>
#include <vector>

namespace slam_native {
void write_tracks_header(std::ostream& out, int width, int height, bool colors = false);
// With `colors`, every row has r,g,b (0 when unknown).
void write_tracks(std::ostream& out, const TrackedFrame& frame, bool colors = false);
// width/height override (or supply) the "# size" line when positive.
std::vector<TrackedFrame> read_tracks_csv(const std::filesystem::path& path, int width = 0, int height = 0);
void write_trajectory_csv(std::ostream& out, const std::vector<TrajectorySample>& trajectory);
// Retired points first, then the active map.
void write_map_csv(std::ostream& out, const VisualOdometry& odometry);
}  // namespace slam_native
