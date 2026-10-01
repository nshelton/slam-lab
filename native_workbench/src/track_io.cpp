#include "slam_native/track_io.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace slam_native {
void write_tracks_header(std::ostream& out, int width, int height, TrackColumns columns) {
  out << "# size " << width << ' ' << height << "\nframe_index,timestamp_ns,track_id,x,y"
      << (columns.colors ? ",r,g,b" : "") << (columns.sigmas ? ",sigma" : "")
      << (columns.depths ? ",depth,depth_sigma" : "") << '\n';
}

void write_tracks_header(std::ostream& out, int width, int height, bool colors, bool sigmas) {
  write_tracks_header(out, width, height, TrackColumns{colors, sigmas, false});
}

void write_tracks(std::ostream& out, const TrackedFrame& frame, TrackColumns columns) {
  out << std::setprecision(9);
  for (const auto& o : frame.observations) {
    out << frame.frame_index << ',' << frame.timestamp_ns << ',' << o.track_id << ',' << o.x << ',' << o.y;
    if (columns.colors) out << ',' << int(o.color[0]) << ',' << int(o.color[1]) << ',' << int(o.color[2]);
    if (columns.sigmas) out << ',' << std::max(o.sigma_px, 0.0F);
    if (columns.depths) out << ',' << std::max(o.depth_m, 0.0F) << ',' << std::max(o.depth_sigma_m, 0.0F);
    out << '\n';
  }
}

void write_tracks(std::ostream& out, const TrackedFrame& frame) {
  out << std::setprecision(9);
  for (const auto& o : frame.observations) {
    out << frame.frame_index << ',' << frame.timestamp_ns << ',' << o.track_id << ',' << o.x << ',' << o.y;
    if (o.has_color) out << ',' << int(o.color[0]) << ',' << int(o.color[1]) << ',' << int(o.color[2]);
    if (o.sigma_px > 0) out << ',' << o.sigma_px;
    out << '\n';
  }
}

std::vector<TrackedFrame> read_tracks_csv(const std::filesystem::path& path, int width, int height) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open tracks file: " + path.string());
  std::vector<TrackedFrame> frames;
  std::string line;
  int file_width = 0, file_height = 0;
  std::vector<std::string> header_extras;
  std::size_t number = 0;
  while (std::getline(in, line)) {
    ++number;
    if (line.empty()) continue;
    if (line[0] == '#') {
      std::istringstream header(line.substr(1));
      std::string word;
      if (header >> word && word == "size") header >> file_width >> file_height;
      continue;
    }
    if (line.rfind("frame_index", 0) == 0) {  // header row: names of the optional columns
      header_extras.clear();
      std::istringstream names(line);
      std::string name;
      int column = 0;
      while (std::getline(names, name, ',')) {
        if (column++ >= 5) header_extras.push_back(name);
      }
      continue;
    }
    std::istringstream row(line);
    std::uint64_t frame_index{}, track_id{};
    std::int64_t timestamp{};
    float x{}, y{};
    char c1{}, c2{}, c3{}, c4{};
    if (!(row >> frame_index >> c1 >> timestamp >> c2 >> track_id >> c3 >> x >> c4 >> y) ||
        c1 != ',' || c2 != ',' || c3 != ',' || c4 != ',') {
      throw std::runtime_error("Malformed tracks row " + std::to_string(number) + ": " + line);
    }
    if (frames.empty() || frames.back().frame_index != frame_index) {
      if (!frames.empty() && frame_index < frames.back().frame_index)
        throw std::runtime_error("Tracks rows must be ordered by frame (row " + std::to_string(number) + ")");
      frames.push_back({frame_index, timestamp, 0, 0, {}, 0, {}});
    }
    TrackObservation observation{track_id, x, y};
    // Optional extras, told apart by count: sigma | r,g,b | r,g,b,sigma.
    std::vector<double> extra;
    char comma{};
    double value{};
    while (row >> comma >> value && comma == ',') extra.push_back(value);
    const auto set_color = [&](std::size_t first) {
      observation.has_color = true;
      for (int c = 0; c < 3; ++c)
        observation.color[c] = static_cast<std::uint8_t>(std::clamp(static_cast<int>(extra[first + c]), 0, 255));
    };
    if (!header_extras.empty() && extra.size() == header_extras.size()) {
      for (std::size_t k = 0; k < extra.size(); ++k) {
        const auto& name = header_extras[k];
        if (name == "r" && k + 2 < extra.size()) set_color(k);
        else if (name == "sigma") observation.sigma_px = static_cast<float>(extra[k]);
        else if (name == "depth") observation.depth_m = static_cast<float>(extra[k]);
        else if (name == "depth_sigma") observation.depth_sigma_m = static_cast<float>(extra[k]);
      }
    } else {
      if (extra.size() == 3 || extra.size() == 4) set_color(0);
      if (extra.size() == 1 || extra.size() == 4) observation.sigma_px = static_cast<float>(extra.back());
    }
    frames.back().observations.push_back(observation);
  }
  const int w = width > 0 ? width : file_width, h = height > 0 ? height : file_height;
  if (w <= 0 || h <= 0) throw std::runtime_error("Image size unknown: pass width/height or a '# size' line");
  for (auto& frame : frames) {
    frame.width = w;
    frame.height = h;
  }
  return frames;
}

void write_trajectory_csv(std::ostream& out, const std::vector<TrajectorySample>& trajectory) {
  out << "frame_index,timestamp_ns,segment,keyframe,cx,cy,cz,qw,qx,qy,qz\n" << std::setprecision(10);
  for (const auto& sample : trajectory) {
    Eigen::Matrix3d R;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) R(r, c) = sample.pose.rotation[r * 3 + c];
    const Eigen::Quaterniond q(R.transpose());  // camera -> world
    const auto center = sample.pose.center();
    out << sample.frame_index << ',' << sample.timestamp_ns << ',' << sample.segment << ','
        << (sample.keyframe ? 1 : 0) << ',' << center[0] << ',' << center[1] << ',' << center[2] << ','
        << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << '\n';
  }
}

void write_map_csv(std::ostream& out, const VisualOdometry& odometry) {
  out << "track_id,segment,x,y,z,r,g,b,has_color,keyframe_observations,first_frame,last_frame,retired,"
         "depth_sigma_ratio,max_parallax_deg,reprojection_rms,confidence,landmark_id\n"
      << std::setprecision(9);
  const auto write = [&](const std::vector<MapPoint>& points, int retired) {
    for (const auto& p : points)
      out << p.track_id << ',' << p.segment << ',' << p.position[0] << ',' << p.position[1] << ',' << p.position[2]
          << ',' << int(p.color[0]) << ',' << int(p.color[1]) << ',' << int(p.color[2]) << ',' << int(p.has_color)
          << ',' << p.keyframe_observations << ',' << p.first_frame << ',' << p.last_frame << ',' << retired << ',' << p.depth_sigma_ratio << ','
          << p.max_parallax_degrees << ',' << p.reprojection_rms << ',' << p.confidence << ',' << p.landmark_id << '\n';
  };
  write(odometry.retired_map(), 1);
  write(odometry.active_map(), 0);
}
}  // namespace slam_native
