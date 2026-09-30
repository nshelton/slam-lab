// ORB-SLAM3 monocular baseline driver.
// Reads raw 8-bit grayscale frames (e.g. from `ffmpeg -f rawvideo -pix_fmt gray`)
// from stdin and writes trajectories in the native workbench CSV format
// (see native_workbench/include/slam_native/track_io.hpp):
//   frame_index,timestamp_ns,segment,keyframe,cx,cy,cz,qw,qx,qy,qz
// OUT_PREFIX-live.csv  : pose reported by TrackMonocular for every tracked frame
//                        (what a live mode would display; segment = map id)
// OUT_PREFIX-final.csv : poses after local/global BA and loop closure, from the
//                        largest map (ORB-SLAM3's SaveTrajectoryEuRoC)
// usage: orbslam3_run VOCAB SETTINGS WIDTH HEIGHT FPS FIRST_FRAME OUT_PREFIX
// Environment: ORBSLAM3_VIEWER=1 opens ORB-SLAM3's Pangolin viewer and paces
// frames to real time; ORBSLAM3_HOLD=SECONDS keeps the final map on screen
// before shutdown (default 30 with the viewer).
#include <System.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: orbslam3_run VOCAB SETTINGS WIDTH HEIGHT FPS FIRST_FRAME OUT_PREFIX < frames.gray\n";
    return 2;
  }
  const int width = std::stoi(argv[3]), height = std::stoi(argv[4]);
  const double fps = std::stod(argv[5]);
  const long first = std::stol(argv[6]);
  const std::string out = argv[7];
  const char* viewer_env = std::getenv("ORBSLAM3_VIEWER");
  const bool viewer = viewer_env != nullptr && std::string(viewer_env) != "0";
  const char* hold_env = std::getenv("ORBSLAM3_HOLD");
  const double hold_s = hold_env ? std::stod(hold_env) : (viewer ? 30.0 : 0.0);
  ORB_SLAM3::System slam(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, viewer);
  const auto wall_start = std::chrono::steady_clock::now();
  std::ofstream live(out + "-live.csv");
  live << "frame_index,timestamp_ns,segment,keyframe,cx,cy,cz,qw,qx,qy,qz\n" << std::setprecision(10);
  std::vector<unsigned char> buffer(static_cast<std::size_t>(width) * height);
  long index = first, tracked = 0, frames = 0, losses = 0;
  int segment = 0;
  bool was_ok = false;
  double total_ms = 0, max_ms = 0;
  while (std::fread(buffer.data(), 1, buffer.size(), stdin) == buffer.size()) {
    cv::Mat image(height, width, CV_8UC1, buffer.data());
    const double t = index / fps;
    const auto start = std::chrono::steady_clock::now();
    const Sophus::SE3f Tcw = slam.TrackMonocular(image, t);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    total_ms += ms;
    max_ms = std::max(max_ms, ms);
    ++frames;
    const int state = slam.GetTrackingState();  // 2 = OK, 3 = RECENTLY_LOST, 4 = LOST
    const bool ok = state == 2;
    if (was_ok && !ok) ++losses;
    if (ok && !was_ok && tracked > 0) ++segment;  // relocalized or new map
    was_ok = ok;
    if (ok) {
      ++tracked;
      const Sophus::SE3f Twc = Tcw.inverse();
      const Eigen::Quaternionf q = Twc.unit_quaternion();
      const Eigen::Vector3f c = Twc.translation();
      live << index << ',' << static_cast<long long>(std::llround(t * 1e9)) << ',' << segment << ",0,"
           << c.x() << ',' << c.y() << ',' << c.z() << ',' << q.w() << ',' << q.x() << ',' << q.y() << ','
           << q.z() << '\n';
    }
    ++index;
    if (viewer) {  // pace to real time so the viewer is watchable
      std::this_thread::sleep_until(wall_start + std::chrono::duration<double>(frames / fps));
    }
  }
  live.close();
  if (hold_s > 0) {
    std::fprintf(stderr, "clip finished; holding viewer for %.0f s\n", hold_s);
    std::this_thread::sleep_for(std::chrono::duration<double>(hold_s));
  }
  slam.Shutdown();
  slam.SaveTrajectoryEuRoC(out + "-final-euroc.txt");
  slam.SaveKeyFrameTrajectoryEuRoC(out + "-keyframes-euroc.txt");
  // Convert EuRoC (timestamp_ns tx ty tz qx qy qz qw, camera -> world) to our CSV.
  std::ifstream euroc(out + "-final-euroc.txt");
  std::ofstream final_csv(out + "-final.csv");
  final_csv << "frame_index,timestamp_ns,segment,keyframe,cx,cy,cz,qw,qx,qy,qz\n" << std::setprecision(10);
  std::string line;
  long final_count = 0;
  while (std::getline(euroc, line)) {
    std::istringstream row(line);
    double stamp, tx, ty, tz, qx, qy, qz, qw;
    if (!(row >> stamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;
    const long frame = std::lround(stamp / 1e9 * fps);
    final_csv << frame << ',' << static_cast<long long>(std::llround(stamp)) << ",0,0," << tx << ',' << ty << ','
              << tz << ',' << qw << ',' << qx << ',' << qy << ',' << qz << '\n';
    ++final_count;
  }
  std::printf("ORB-SLAM3: %ld frames, %ld tracked (live), %ld losses, %d relocalizations/new maps, "
              "%ld frames in final largest-map trajectory, %.2f ms/frame (max %.1f)\n",
              frames, tracked, losses, segment, final_count, total_ms / std::max(frames, 1L), max_ms);
  // Outputs are written; skip ORB-SLAM3/Pangolin static teardown, which crashes
  // (X11 BadWindow) after the viewer thread has closed its window.
  std::fflush(nullptr);
  std::_Exit(0);
}
