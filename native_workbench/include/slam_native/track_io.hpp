#pragma once
// Plain-text interchange for tracks and camera trajectories, so tracks from
// any source can drive VisualOdometry and results can be compared offline.
//
// Tracks CSV:   # size WIDTH HEIGHT            (optional, first line)
//               frame_index,timestamp_ns,track_id,x,y[,r,g,b][,sigma][,depth,depth_sigma]
//               one row per observation; rows grouped by frame in order.
//               The optional r,g,b columns (0-255) are the image colour at
//               the observation, sigma its 1-sigma position noise in pixels
//               (TrackObservation::sigma_px), depth/depth_sigma the network
//               z-depth in metres (0 = unknown). The reader takes the
//               optional columns from the header row by name; rows whose
//               value count differs from the header (older files) fall back
//               to counting (1: sigma, 3: rgb, 4: rgb and sigma).
// Trajectory CSV: frame_index,timestamp_ns,segment,keyframe,cx,cy,cz,qw,qx,qy,qz
//               camera centre and camera->world rotation quaternion
//               (Hamilton, w first); arbitrary monocular scale per segment.
// Map CSV:      track_id,segment,x,y,z,r,g,b,has_color,keyframe_observations,
//               first_frame,last_frame,retired,depth_sigma_ratio,
//               max_parallax_deg,reprojection_rms,confidence,landmark_id
//               (track_id: latest track; landmark_id: unique; world
//               coordinates, as the trajectory; r,g,b are 0 when has_color is
//               0; uncertainty columns as MapPoint, nan when unknown)
#include "slam_native/visual_odometry.hpp"

#include <filesystem>
#include <ostream>
#include <vector>

namespace slam_native {
// Optional columns, in this order after x,y: r,g,b | sigma | depth,depth_sigma.
struct TrackColumns {
  bool colors{};
  bool sigmas{};
  bool depths{};
};
void write_tracks_header(std::ostream& out, int width, int height, TrackColumns columns);
void write_tracks_header(std::ostream& out, int width, int height, bool colors = false, bool sigmas = false);
// Exactly the declared columns on every row (unknown values as 0).
void write_tracks(std::ostream& out, const TrackedFrame& frame, TrackColumns columns);
// Legacy: per-row optional columns (colour when known, sigma when > 0).
void write_tracks(std::ostream& out, const TrackedFrame& frame);
// width/height override (or supply) the "# size" line when positive.
std::vector<TrackedFrame> read_tracks_csv(const std::filesystem::path& path, int width = 0, int height = 0);
void write_trajectory_csv(std::ostream& out, const std::vector<TrajectorySample>& trajectory);
// Retired points first, then the active map.
void write_map_csv(std::ostream& out, const VisualOdometry& odometry);
}  // namespace slam_native
