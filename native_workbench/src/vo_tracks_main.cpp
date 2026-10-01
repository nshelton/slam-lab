// Run the visual odometry on tracks from any source (CSV), headless.
#include "slam_native/track_io.hpp"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  using namespace slam_native;
  std::string tracks, output, map_output;
  int width = 0, height = 0;
  bool verbose = false;
  VisualOdometryConfig config;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      const auto value = [&]() -> std::string {
        if (++i >= argc) throw std::invalid_argument("Missing value for " + option);
        return argv[i];
      };
      if (option == "--tracks") tracks = value();
      else if (option == "--output") output = value();
      else if (option == "--map") map_output = value();
      else if (option == "--width") width = std::stoi(value());
      else if (option == "--height") height = std::stoi(value());
      else if (option == "--verbose") verbose = true;
      else if (option == "--set") {
        const std::string pair = value();
        const auto eq = pair.find('=');
        if (eq == std::string::npos) throw std::invalid_argument("--set expects key=value");
        const std::string key = pair.substr(0, eq);
        const double v = std::stod(pair.substr(eq + 1));
        if (key == "min_triangulation_parallax_degrees") config.min_triangulation_parallax_degrees = v;
        else if (key == "init_min_parallax_degrees") config.init_min_parallax_degrees = v;
        else if (key == "reprojection_threshold_px") config.reprojection_threshold_px = v;
        else if (key == "window_keyframes") config.window_keyframes = static_cast<int>(v);
        else if (key == "keyframe_max_interval") config.keyframe_max_interval = static_cast<int>(v);
        else if (key == "keyframe_min_interval") config.keyframe_min_interval = static_cast<int>(v);
        else if (key == "keyframe_track_ratio") config.keyframe_track_ratio = v;
        else if (key == "bundle_iterations") config.bundle_iterations = static_cast<int>(v);
        else if (key == "min_tracked_points") config.min_tracked_points = static_cast<int>(v);
        else if (key == "use_observation_sigma") config.use_observation_sigma = v != 0;
        else if (key == "pose_landmark_uncertainty") config.pose_landmark_uncertainty = v != 0;
        else if (key == "pose_landmark_pixel_gate") config.pose_landmark_pixel_gate = v != 0;
        else if (key == "cull_stale_observations") config.cull_stale_observations = v != 0;
        else if (key == "pose_landmark_covariance_scale") config.pose_landmark_covariance_scale = v;
        else if (key == "observation_sigma_px") config.observation_sigma_px = v;
        else if (key == "min_observation_sigma_px") config.min_observation_sigma_px = v;
        else throw std::invalid_argument("Unknown --set key: " + key);
      }
      else if (option == "--hfov") config.horizontal_fov_degrees = std::stod(value());
      else if (option == "--k1") config.distortion_k1 = std::stod(value());
      else if (option == "--intrinsics") {
        CameraIntrinsics k;
        k.fx = std::stod(value()); k.fy = std::stod(value()); k.cx = std::stod(value()); k.cy = std::stod(value());
        config.intrinsics = k;
      } else {
        std::cerr << "usage: slam-native-vo-tracks --tracks TRACKS.csv [--output TRAJECTORY.csv] [--map MAP.csv]\n"
                     "       [--width W --height H] [--hfov DEG | --intrinsics FX FY CX CY] [--k1 K1] [--verbose]\n"
                     "       [--set KEY=VALUE]... (VisualOdometryConfig field, e.g. window_keyframes=10)\n"
                     "Tracks CSV: frame_index,timestamp_ns,track_id,x,y (see include/slam_native/track_io.hpp)\n";
        return option == "--help" ? 0 : 2;
      }
    }
    if (tracks.empty()) throw std::invalid_argument("--tracks is required");
    config.validate();
    const auto frames = read_tracks_csv(tracks, width, height);
    VisualOdometry odometry(config);
    int posed = 0, keyframes = 0, losses = 0;
    double ms = 0;
    std::string last_event;
    for (const auto& frame : frames) {
      const auto result = odometry.process(frame);
      if (verbose && (result.keyframe || result.event != last_event)) {
        std::cout << "frame " << result.frame_index << ' ' << to_string(result.state) << " seg " << result.segment
                  << " inliers " << result.inliers << '/' << result.correspondences << " map " << result.map_points
                  << (result.keyframe ? " KF" : "") << ' ' << result.event << '\n';
        last_event = result.event;
      }
      posed += result.has_pose;
      keyframes += result.keyframe;
      losses += result.state == OdometryState::lost;
      ms += result.ms;
    }
    std::cout << frames.size() << " frames, " << posed << " posed, " << keyframes << " keyframes, " << losses
              << " losses, " << odometry.last().segment + 1 << " segments, "
              << (frames.empty() ? 0 : ms / frames.size()) << " ms/frame\n";
    if (!output.empty()) {
      std::ofstream out(output);
      write_trajectory_csv(out, odometry.trajectory());
      std::cout << "trajectory: " << output << '\n';
    }
    if (!map_output.empty()) {
      std::ofstream out(map_output);
      write_map_csv(out, odometry);
      std::cout << "map: " << map_output << " (" << odometry.retired_count() << " retired + "
                << odometry.active_map().size() << " active points)\n";
    }
  } catch (const std::exception& error) {
    std::cerr << "slam-native-vo-tracks: " << error.what() << '\n';
    return 1;
  }
}
