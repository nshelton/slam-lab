#define GL_GLEXT_PROTOTYPES
#include "slam_native/app.hpp"

#include "slam_native/calibration.hpp"
#include "slam_native/cuda_frame_presenter.hpp"
#include "slam_native/display_environment.hpp"
#include "slam_native/descriptor_projection.hpp"
#include "slam_native/flow_tracker.hpp"
#include "slam_native/launcher.hpp"
#include "slam_native/optical_flow.hpp"
#include "slam_native/color_sampler.hpp"
#include "slam_native/depth_estimator.hpp"
#include "slam_native/depth_filter.hpp"
#include "slam_native/depth_sampling.hpp"
#include "slam_native/keyframe_depth.hpp"
#include "slam_native/focal_estimation.hpp"
#include "slam_native/fov_view.hpp"
#include "slam_native/track_io.hpp"
#include "slam_native/trajectory_view.hpp"
#include "slam_native/visual_odometry.hpp"
#include "slam_native/video_decoder.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <sqlite3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace slam_native {
namespace {
using Clock = std::chrono::steady_clock;

class GlfwLifetime {
 public:
  GlfwLifetime() {
    configure_display_environment();
    glfwSetErrorCallback([](int, const char* description) {
      std::fprintf(stderr, "GLFW: %s\n", description);
    });
    if (!glfwInit()) throw std::runtime_error("Cannot initialize GLFW");
  }
  ~GlfwLifetime() { glfwTerminate(); }
};

class Window {
 public:
  Window() {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    value_ = glfwCreateWindow(1440, 900, "SLAM Native Workbench", nullptr, nullptr);
    if (!value_) throw std::runtime_error("Cannot create OpenGL window");
    glfwMakeContextCurrent(value_);
    glfwSwapInterval(1);
  }
  ~Window() { glfwDestroyWindow(value_); }
  [[nodiscard]] GLFWwindow* get() const { return value_; }

 private:
  GLFWwindow* value_{};
};

class ImGuiLifetime {
 public:
  explicit ImGuiLifetime(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
  }
  ~ImGuiLifetime() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
  }
};

struct Runtime {
  bool playing{true};
  bool step_requested{};
  bool realtime_pacing{true};
  bool eof{};
  bool stopped{};
  bool stop_requested{};
  bool close_requested{};
  bool restart_requested{};
  bool show_features{true};
  int point_display_mode{2}; // 0: tracked points, 1: all raw SuperPoint detections, 2: pose inliers (default)
  float flow_sigma{};       // tracker position fusion; <= 0 snaps to detections
  int display_rotation{};   // clockwise degrees; display only (processing uses native frames)
  int metadata_rotation{};
  float detection_sigma{1.0F};
  float association_radius{6.0F};
  bool show_trails{true};
  bool show_untracked_landmarks{true};
  bool show_flow_vectors{};
  bool show_diagnostics{true};
  bool show_trajectory{true};
  bool show_fov{true};
  bool show_depth{true};
  int depth_model{-1};          // index into depth_model_presets(); -1: off
  std::string depth_error;      // last engine load/run failure
  DepthMap depth;               // latest finished depth (lags the video by a frame or so)
  bool depth_dirty{};           // depth changed since the texture was uploaded
  unsigned int depth_texture{};
  int depth_texture_width{};
  int depth_texture_height{};
  float depth_near{}, depth_far{};  // colour range (2nd / 98th percentile)
  // Depth fusion (DEPTH_FUSION.md): the filter runs on frames that have a
  // depth map and a pose; the window shows its state rendered into the latest
  // such frame.
  struct Fusion {
    bool show{true};
    bool enabled{true};      // also the source of the keyframe depth clouds (fused instead of raw)
    int view{};              // index into kFusionViews
    bool overlay{};
    float opacity{0.6F};
    std::string status;      // why it is not running
    DepthFilterStats stats;
    double ms{};
    DepthFilterView fused;   // rendered into the frame of `raw`
    DepthMap raw;            // that frame's network map
    bool dirty{};
    int uploaded_view{-1};
    unsigned int texture{};
    int texture_width{}, texture_height{};
    float near{}, far{};     // depth colour range (2nd / 98th percentile of the raw map)
    int keyframes_fused{}, keyframes_raw{};  // keyframe clouds built from each
  } fusion;
  OdometryFrameResult odometry;
  int relocalizations{};             // since the odometry was (re)created
  int loops_closed{};
  std::string last_loop;             // its event text
  // Global (appearance) loop search, same span: keyframes searched, their
  // total matching time, verified poses, and the latest search's result.
  int global_searches{}, global_verified{};
  double global_total_ms{};
  OdometryFrameResult last_global;
  std::string intrinsics_source;     // "calibration.json", "--intrinsics" or empty (hfov guess)
  std::size_t reassociations{};      // landmarks re-found by new tracks, same span
  std::string last_relocalization;   // its event text
  bool export_requested{};
  std::string export_status;         // last export's folder, or its error
  bool scrub_active{};
  bool seek_requested{};
  double scrub_seconds{};
  double seek_seconds{};
  Clock::time_point last_seek_request{};
  std::string seek_error;
  unsigned int texture{};
  int frame_width{};
  int frame_height{};
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  std::vector<Keypoint> points;
  std::vector<bool> supported;  // false: carried by flow this frame (no detection)
  std::vector<Keypoint> detections;
  std::vector<std::uint64_t> landmark_ids;
  std::vector<float> similarities;
  std::vector<Keypoint> predictions;
  std::vector<float> corrections;
  bool cache_hit{};
  double flow_ms{};
  std::vector<ProjectedDescriptor> projected;
  std::optional<FlowField> flow_field;
  std::uint64_t selected_landmark{std::numeric_limits<std::uint64_t>::max()};
  struct Trail {
    std::deque<ImVec2> positions;
    std::deque<ProjectedDescriptor> descriptor_positions;
    std::uint64_t last_frame{};
  };
  std::unordered_map<std::uint64_t, Trail> trails;
  std::uint32_t new_landmarks{};
  std::uint32_t matched_landmarks{};
  std::uint32_t coasted_landmarks{};
  double tracking_ms{};
  PipelineStats stats;
  Clock::time_point started{Clock::now()};
  Clock::time_point next_frame{Clock::now()};
};

ImU32 landmark_color(std::uint64_t id, int alpha = 220) {
  const float hue = static_cast<float>((id * 2654435761ULL) % 360) / 360.0F;
  ImVec4 rgb;
  ImGui::ColorConvertHSVtoRGB(hue, 0.8F, 1.0F, rgb.x, rgb.y, rgb.z);
  return ImGui::ColorConvertFloat4ToU32({rgb.x, rgb.y, rgb.z, alpha / 255.0F});
}

std::string format_video_time(double seconds) {
  const auto milliseconds = static_cast<std::uint64_t>(std::max(0.0, seconds) * 1000.0);
  const auto hours = milliseconds / 3600000;
  const auto minutes = (milliseconds / 60000) % 60;
  const auto whole_seconds = (milliseconds / 1000) % 60;
  const auto fraction = milliseconds % 1000;
  char buffer[40];
  std::snprintf(buffer, sizeof(buffer), "%02llu:%02llu:%02llu.%03llu",
                static_cast<unsigned long long>(hours),
                static_cast<unsigned long long>(minutes),
                static_cast<unsigned long long>(whole_seconds),
                static_cast<unsigned long long>(fraction));
  return buffer;
}

void update_trails(Runtime& runtime) {
  for (std::size_t index = 0; index < runtime.points.size(); ++index) {
    auto& trail = runtime.trails[runtime.landmark_ids[index]];
    trail.positions.emplace_back(runtime.points[index].x, runtime.points[index].y);
    if (trail.positions.size() > 8) trail.positions.pop_front();
    if (index < runtime.projected.size()) {
      trail.descriptor_positions.push_back(runtime.projected[index]);
      if (trail.descriptor_positions.size() > 32) {
        trail.descriptor_positions.pop_front();
      }
    }
    trail.last_frame = runtime.frame_index;
  }
  for (auto it = runtime.trails.begin(); it != runtime.trails.end();) {
    if (it->second.last_frame != runtime.frame_index) {
      it = runtime.trails.erase(it);
    } else {
      ++it;
    }
  }
}

void draw_sidebar(Runtime& runtime,
                  const VideoDecoder& decoder,
                  const FeatureStore& store,
                  const OnlineTracker* tracker,
                  const FlowTracker* flow_tracker,
                  const DatabaseStats& database_stats,
                  const std::filesystem::path& database_path) {
  const bool optical_flow = flow_tracker != nullptr;
  ImGui::SetNextWindowPos({8, 8}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({310, 675}, ImGuiCond_FirstUseEver);
  ImGui::Begin("Pipeline");
  ImGui::BeginDisabled(runtime.stopped || runtime.eof);
  const char* run_label = runtime.playing ? "Pause" : "Play";
  if (ImGui::Button(run_label)) runtime.playing = !runtime.playing;
  ImGui::SameLine();
  if (ImGui::Button("Step")) {
    runtime.playing = false;
    runtime.step_requested = true;
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Stop")) {
    runtime.playing = false;
    runtime.stopped = true;
    runtime.stop_requested = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Restart")) runtime.restart_requested = true;
  if (ImGui::Button("Choose another video")) runtime.close_requested = true;
  ImGui::SameLine();
  if (ImGui::Button("Rotate")) runtime.display_rotation = (runtime.display_rotation + 90) % 360;
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Rotate the view 90 deg clockwise (now %d deg; video metadata says %d deg).\n"
                      "Display only: tracking and pose run on the native frames, whose\n"
                      "landscape layout matches the SuperPoint engine's input.",
                      runtime.display_rotation, runtime.metadata_rotation);
  if (ImGui::Button("Export trajectory")) runtime.export_requested = true;
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Write the camera trajectory, map and camera settings (CSV) to out/<video>/.");
  if (!runtime.export_status.empty()) ImGui::TextWrapped("%s", runtime.export_status.c_str());
  ImGui::Checkbox("Realtime pacing", &runtime.realtime_pacing);
  ImGui::SetNextItemWidth(110);
  ImGui::DragFloat("Association radius", &runtime.association_radius, 0.5F,
                   0.5F, 100.0F, "%.1f px", ImGuiSliderFlags_AlwaysClamp);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Maximum distance from a flow prediction to a SuperPoint, in source-image pixels.\n"
                      "Applies on the next frame. Larger values allow more distant matches.\n"
                      "Each previous track still accepts only one detection.");
  }
  ImGui::Combo("Point view", &runtime.point_display_mode,
               "Tracked points\0SuperPoint detections (red)\0Pose inliers (green) / outliers (red)\0");
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Pose inliers: tracks with a 3D landmark whose reprojection under the\n"
                      "current camera pose is within the threshold (green), beyond it (red).\n"
                      "Grey: tracked, no landmark yet (hollow: flow-only this frame).");
  ImGui::SetNextItemWidth(110);
  ImGui::DragFloat("Flow sigma", &runtime.flow_sigma, 0.02F, 0.0F, 10.0F,
                   runtime.flow_sigma > 0 ? "%.2f px" : "off", ImGuiSliderFlags_AlwaysClamp);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Track position fusion (Kalman filter). Flow predicts, the matched SuperPoint\n"
                      "detection corrects. Per-frame flow error: smaller trusts flow more (smoother).\n"
                      "0 = off: tracks snap exactly onto SuperPoint detections. Next frame.");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(70);
  ImGui::DragFloat("Det. sigma", &runtime.detection_sigma, 0.02F, 0.1F, 10.0F, "%.2f px",
                   ImGuiSliderFlags_AlwaysClamp);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("SuperPoint localisation noise used by the fusion (source pixels).");
  ImGui::Checkbox("Show points", &runtime.show_features);
  ImGui::BeginDisabled(runtime.point_display_mode == 1);
  ImGui::Checkbox("Show tracks", &runtime.show_trails);
  if (optical_flow) ImGui::Checkbox("Show flow vectors", &runtime.show_flow_vectors);
  ImGui::Checkbox("Untracked landmarks", &runtime.show_untracked_landmarks);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("3D landmarks that project into this frame under the current pose but have no\n"
                      "track observing them (magenta ring; faint: dormant, its track ended earlier).\n"
                      "Filled magenta with a white ring: re-found this frame by a new track.");
  ImGui::EndDisabled();
  ImGui::Checkbox("Diagnostics", &runtime.show_diagnostics);
  ImGui::SameLine();
  ImGui::Checkbox("Trajectory", &runtime.show_trajectory);
  ImGui::SameLine();
  ImGui::Checkbox("Lens", &runtime.show_fov);
  ImGui::SameLine();
  ImGui::Checkbox("Depth fusion", &runtime.fusion.show);
  ImGui::Separator();
  ImGui::Text("Codec: %s", decoder.codec_name().c_str());
  ImGui::Text("Nominal FPS: %.2f", decoder.nominal_fps());
  ImGui::Text("Frame: %llu", static_cast<unsigned long long>(runtime.frame_index));
  ImGui::Text("Time: %.3f s", std::max(0.0,
              static_cast<double>(runtime.timestamp_ns - decoder.start_time_ns()) / 1e9));
  if (!runtime.seek_error.empty()) {
    ImGui::TextColored({1, 0.5F, 0.4F, 1}, "%s", runtime.seek_error.c_str());
  }
  ImGui::Text("Features: %zu", runtime.point_display_mode == 1 ?
              runtime.detections.size() : runtime.points.size());
  ImGui::Text("Matched / coasted / new: %u / %u / %u", runtime.matched_landmarks,
              runtime.coasted_landmarks, runtime.new_landmarks);
  ImGui::Text("Method: %s", optical_flow ? "SuperPoint + optical flow" : "SuperPoint descriptors");
  ImGui::Text("Active / total tracks: %zu / %llu",
              optical_flow ? flow_tracker->active_count() : tracker->active_count(),
              static_cast<unsigned long long>(optical_flow ? flow_tracker->landmark_count() :
                                                               tracker->landmark_count()));
  ImGui::Separator();
  ImGui::Text("Decode: %.2f ms", runtime.stats.decode_ms);
  ImGui::Text("Preprocess: %.2f ms", runtime.stats.preprocess_ms);
  ImGui::Text("SuperPoint: %.2f ms%s", runtime.stats.inference_ms,
              runtime.cache_hit ? " (cached)" : "");
  ImGui::Text("Optical flow: %.2f ms", runtime.flow_ms);
  ImGui::Text("Readback: %.2f ms", runtime.stats.readback_ms);
  ImGui::Text("Tracking: %.2f ms", runtime.tracking_ms);
  ImGui::Text("Camera pose: %s (%.1f ms)", to_string(runtime.odometry.state), runtime.odometry.ms);
  if (runtime.odometry.has_pose) {
    ImGui::Text("Pose inliers: %d / %d", runtime.odometry.inliers, runtime.odometry.correspondences);
  }
  if (!runtime.intrinsics_source.empty()) ImGui::TextDisabled("Intrinsics: %s", runtime.intrinsics_source.c_str());
  ImGui::Text("Relocalized: %d   Loops closed: %d", runtime.relocalizations, runtime.loops_closed);
  if (ImGui::IsItemHovered() && !(runtime.last_relocalization.empty() && runtime.last_loop.empty()))
    ImGui::SetTooltip("Last relocalization: %s\nLast loop closure: %s",
                      runtime.last_relocalization.empty() ? "-" : runtime.last_relocalization.c_str(),
                      runtime.last_loop.empty() ? "-" : runtime.last_loop.c_str());
  if (runtime.global_searches > 0) {
    const auto& g = runtime.last_global;
    ImGui::Text("Global search: %d, %d verified, %.0f ms avg", runtime.global_searches, runtime.global_verified,
                runtime.global_total_ms / runtime.global_searches);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Appearance search at each keyframe: its descriptors against every older landmark\n"
                        "(brute force), then P3P RANSAC. Last, frame %llu:\n"
                        "  %d queries x %d landmarks, %d matches, %d P3P inliers, %.1f ms%s",
                        static_cast<unsigned long long>(g.frame_index), g.global_queries, g.global_database,
                        g.global_matches, g.global_pose_inliers, g.global_match_ms,
                        g.global_found ? ", verified" : "");
  }
  ImGui::Text("Untracked in view: %zu, re-found %d (%zu)",
              runtime.odometry.untracked_landmarks.size(), runtime.odometry.reassociated,
              runtime.reassociations);
  const double elapsed = std::chrono::duration<double>(Clock::now() - runtime.started).count();
  ImGui::Text("Throughput: %.1f frames/s", elapsed > 0 ? runtime.stats.inferred_frames / elapsed : 0);
  ImGui::Separator();
  ImGui::Text("Point cache: %llu frames (%s)", static_cast<unsigned long long>(database_stats.frame_rows),
              format_bytes(database_stats.main_bytes + database_stats.wal_bytes).c_str());
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("SuperPoint detections, reused when this video is replayed.\n%s\nWaiting to be written: %zu frames",
                      database_path.c_str(), store.queued());
  if (runtime.stopped) ImGui::TextColored({1.0F, 0.65F, 0.4F, 1.0F}, "Stopped and flushed");
  if (runtime.eof) ImGui::TextColored({0.45F, 0.8F, 1.0F, 1.0F}, "End of video");
  ImGui::Separator();
  ImGui::TextDisabled("Click a colored point to inspect its track.");
  if (runtime.selected_landmark != std::numeric_limits<std::uint64_t>::max()) {
    ImGui::Separator();
    ImGui::Text("Landmark %llu",
                static_cast<unsigned long long>(runtime.selected_landmark));
    const LandmarkState* selected = optical_flow ?
        flow_tracker->find(runtime.selected_landmark) : tracker->find(runtime.selected_landmark);
    if (selected) {
      ImGui::Text("Observations: %llu",
                  static_cast<unsigned long long>(selected->observation_count));

      ImGui::Text("First / last frame: %llu / %llu",
                  static_cast<unsigned long long>(selected->first_frame),
                  static_cast<unsigned long long>(selected->last_frame));
      for (std::size_t index = 0; !optical_flow && index < runtime.landmark_ids.size(); ++index) {
        if (runtime.landmark_ids[index] == runtime.selected_landmark &&
            std::isfinite(runtime.similarities[index])) {
          ImGui::Text("Current cosine: %.3f", runtime.similarities[index]);
          break;
        }
      }
    } else {
      ImGui::TextDisabled("Inactive landmark");
    }
  }
  ImGui::End();
}

void draw_video(Runtime& runtime, bool optical_flow, std::int64_t duration_ns,
                std::int64_t start_time_ns) {
  ImGui::SetNextWindowPos({326, 8}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({755, 845}, ImGuiCond_FirstUseEver);
  ImGui::Begin(optical_flow ? "Video + optical flow" : "Video + SuperPoint");
  if (!runtime.texture || runtime.frame_width == 0 || runtime.frame_height == 0) {
    ImGui::TextDisabled("Waiting for the first decoded frame...");
    ImGui::End();
    return;
  }
  const ImVec2 available = ImGui::GetContentRegionAvail();
  const int rotation = runtime.display_rotation;
  const bool sideways = rotation == 90 || rotation == 270;
  const float width = static_cast<float>(runtime.frame_width), height = static_cast<float>(runtime.frame_height);
  const float shown_width = sideways ? height : width, shown_height = sideways ? width : height;
  const float scale = std::min(available.x / shown_width, std::max(1.0F, available.y - 85.0F) / shown_height);
  const ImVec2 size(shown_width * scale, shown_height * scale);
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  // Source pixel -> screen, through the clockwise display rotation.
  const auto screen = [&](float x, float y) {
    float u = x, v = y;
    if (rotation == 90) { u = height - y; v = x; }
    else if (rotation == 180) { u = width - x; v = height - y; }
    else if (rotation == 270) { u = y; v = width - x; }
    return ImVec2(origin.x + u * scale, origin.y + v * scale);
  };
  ImGui::InvisibleButton("##video-image", size);
  {
    // Displayed corners (TL, TR, BR, BL) sample these source corners.
    const ImVec2 corners[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    const int first = (4 - rotation / 90) % 4;
    ImGui::GetWindowDrawList()->AddImageQuad(static_cast<ImTextureID>(runtime.texture), origin,
        {origin.x + size.x, origin.y}, {origin.x + size.x, origin.y + size.y}, {origin.x, origin.y + size.y},
        corners[first], corners[(first + 1) % 4], corners[(first + 2) % 4], corners[(first + 3) % 4]);
  }
  if (runtime.point_display_mode != 1 &&
      ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    const ImVec2 mouse = ImGui::GetMousePos();
    float nearest = 100.0F;
    for (std::size_t index = 0; index < runtime.points.size(); ++index) {
      const ImVec2 at = screen(runtime.points[index].x, runtime.points[index].y);
      const float dx = mouse.x - at.x;
      const float dy = mouse.y - at.y;
      const float squared = dx * dx + dy * dy;
      if (squared < nearest) {
        nearest = squared;
        runtime.selected_landmark = runtime.landmark_ids[index];
      }
    }
  }
  ImDrawList* draw = ImGui::GetWindowDrawList();
  if (runtime.point_display_mode != 1 && runtime.show_trails) {
    for (const auto& [id, trail] : runtime.trails) {
      if (trail.positions.size() < 2) continue;
      if (runtime.selected_landmark != std::numeric_limits<std::uint64_t>::max() &&
          id != runtime.selected_landmark) continue;
      for (std::size_t index = 1; index < trail.positions.size(); ++index) {
        const auto& previous = trail.positions[index - 1];
        const auto& current = trail.positions[index];
        draw->AddLine(screen(previous.x, previous.y), screen(current.x, current.y),
                      landmark_color(id, 130), 1.5F);
      }
    }
  }
  if (runtime.show_features && runtime.point_display_mode == 1) {
    for (const auto& point : runtime.detections) {
      draw->AddCircleFilled(screen(point.x, point.y), 2.3F, IM_COL32(255, 0, 0, 255));
    }
  } else if (runtime.show_features && runtime.point_display_mode == 2) {
    const auto& inliers = runtime.odometry.pose_inliers;
    const auto& outliers = runtime.odometry.pose_outliers;
    for (std::size_t index = 0; index < runtime.points.size(); ++index) {
      const auto& point = runtime.points[index];
      const ImVec2 center = screen(point.x, point.y);
      const std::uint64_t id = runtime.landmark_ids[index];
      const bool supported = index >= runtime.supported.size() || runtime.supported[index];
      if (std::binary_search(inliers.begin(), inliers.end(), id)) {
        draw->AddCircleFilled(center, 3.0F, IM_COL32(60, 235, 90, 255));
      } else if (std::binary_search(outliers.begin(), outliers.end(), id)) {
        draw->AddCircleFilled(center, 3.0F, IM_COL32(255, 60, 50, 255));
      } else if (supported) {
        draw->AddCircleFilled(center, 1.8F, IM_COL32(170, 170, 170, 170));
      } else {
        draw->AddCircle(center, 2.5F, IM_COL32(170, 170, 170, 170), 0, 1.0F);
      }
      if (id == runtime.selected_landmark) draw->AddCircle(center, 7.0F, IM_COL32_WHITE, 0, 2.0F);
    }
  } else if (runtime.show_features) {
    for (std::size_t index = 0; index < runtime.points.size(); ++index) {
      const auto& point = runtime.points[index];
      const ImVec2 center = screen(point.x, point.y);
      const std::uint64_t id = runtime.landmark_ids[index];
      const bool selected = id == runtime.selected_landmark;
      const auto color = landmark_color(id);
      const bool supported = index >= runtime.supported.size() || runtime.supported[index];
      if (supported) draw->AddCircleFilled(center, selected ? 5.0F : 2.3F, color);
      else draw->AddCircle(center, selected ? 5.0F : 3.0F, color, 0, 1.2F);  // flow-only frame
      if (selected) draw->AddCircle(center, 7.0F, IM_COL32_WHITE, 0, 2.0F);
    }
  }
  if (runtime.show_untracked_landmarks && runtime.point_display_mode != 1) {
    for (const auto& landmark : runtime.odometry.untracked_landmarks) {
      const ImVec2 center = screen(landmark.x, landmark.y);
      if (landmark.reassociated) {
        draw->AddCircleFilled(center, 4.0F, IM_COL32(255, 0, 255, 255));
        draw->AddCircle(center, 6.5F, IM_COL32_WHITE, 0, 1.5F);
      } else {
        // Small filled dot (2 px): there are many, rings cluttered the frame.
        draw->AddCircleFilled(center, 1.0F, landmark.dormant ? IM_COL32(255, 0, 255, 110) : IM_COL32(255, 0, 255, 230));
      }
    }
  }
  if (runtime.point_display_mode == 0 &&
      optical_flow && runtime.show_flow_vectors && runtime.flow_field) {
    constexpr int arrow_spacing = 48;
    for (int y = arrow_spacing / 2; y < runtime.frame_height; y += arrow_spacing) {
      for (int x = arrow_spacing / 2; x < runtime.frame_width; x += arrow_spacing) {
        const FlowVector vector = runtime.flow_field->sample(static_cast<float>(x),
                                                              static_cast<float>(y));
        if (!std::isfinite(vector.dx) || !std::isfinite(vector.dy) ||
            vector.dx * vector.dx + vector.dy * vector.dy < 0.25F) continue;
        const ImVec2 from = screen(static_cast<float>(x), static_cast<float>(y));
        const ImVec2 to = screen(x + vector.dx, y + vector.dy);
        draw->AddLine(from, to, IM_COL32(80, 230, 255, 210), 1.5F);
        draw->AddCircleFilled(to, 2.0F, IM_COL32(80, 230, 255, 220));
      }
    }
  }
  ImGui::SetCursorPosY(std::max(ImGui::GetCursorPosY() + 8.0F,
                                 ImGui::GetWindowHeight() - 70.0F));
  if (duration_ns > 0) {
    const double duration = static_cast<double>(duration_ns) / 1e9;
    const double minimum = 0.0;
    double position = runtime.scrub_active ? runtime.scrub_seconds :
        std::clamp(static_cast<double>(runtime.timestamp_ns - start_time_ns) / 1e9,
                   0.0, duration);
    ImGui::SetNextItemWidth(-1);
    const bool changed = ImGui::SliderScalar("##video-scrubber", ImGuiDataType_Double,
                                            &position, &minimum, &duration, "%.2f s",
                                            ImGuiSliderFlags_AlwaysClamp);
    const bool active = ImGui::IsItemActive();
    const bool released = ImGui::IsItemDeactivatedAfterEdit();
    if (changed) {
      runtime.scrub_seconds = position;
      runtime.scrub_active = active;
      runtime.playing = false;
    }
    const auto now = Clock::now();
    if ((changed && now - runtime.last_seek_request > std::chrono::milliseconds(120)) ||
        released) {
      runtime.seek_seconds = position;
      runtime.seek_requested = true;
      runtime.last_seek_request = now;
    }
    if (!active) runtime.scrub_active = false;
    ImGui::Text("%s / %s", format_video_time(position).c_str(),
                format_video_time(duration).c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("Drag to seek");
  } else {
    ImGui::TextDisabled("Video duration unavailable; seeking is disabled.");
  }
  ImGui::End();
}

std::string length_bin_label(std::size_t index) {
  if (index < 16) return std::to_string(index + 1);
  if (index + 1 == kTrackLengthBins) return "2049+";
  const std::uint64_t upper = 32ULL << (index - 16);
  return std::to_string(upper / 2 + 1) + "-" + std::to_string(upper);
}

void draw_diagnostics(Runtime& runtime, const OnlineTracker& tracker,
                      const DescriptorProjection& projection) {
  if (!runtime.show_diagnostics) return;
  ImGui::SetNextWindowPos({1090, 8}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({340, 845}, ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Landmark diagnostics", &runtime.show_diagnostics)) {
    ImGui::End();
    return;
  }
  ImGui::Text("Tracks: %llu total / %zu active",
              static_cast<unsigned long long>(tracker.landmark_count()),
              tracker.active_count());
  const auto& histogram = tracker.length_histogram();
  std::uint64_t repeated = 0;
  for (std::size_t index = 2; index < histogram.size(); ++index) repeated += histogram[index];
  ImGui::Text("1 observation: %llu | 3+: %llu",
              static_cast<unsigned long long>(histogram[0]),
              static_cast<unsigned long long>(repeated));
  ImGui::Separator();
  ImGui::TextUnformatted("Descriptor PCA (256D to 2D)");
  if (!projection.ready()) {
    ImGui::TextDisabled("Collecting descriptors: %zu / 2048", projection.sample_count());
  } else {
    ImGui::TextDisabled("Fixed basis from first %zu descriptors", projection.sample_count());
  }
  const float width = std::max(120.0F, std::min(ImGui::GetContentRegionAvail().x, 306.0F));
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("Descriptor scatter", {width, width});
  ImDrawList* draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(origin, {origin.x + width, origin.y + width}, IM_COL32(19, 23, 32, 255));
  draw->AddRect(origin, {origin.x + width, origin.y + width}, IM_COL32(75, 85, 102, 255));
  draw->AddLine({origin.x + width / 2, origin.y},
                {origin.x + width / 2, origin.y + width}, IM_COL32(65, 72, 85, 255));
  draw->AddLine({origin.x, origin.y + width / 2},
                {origin.x + width, origin.y + width / 2}, IM_COL32(65, 72, 85, 255));
  const auto position = [&](ProjectedDescriptor value) {
    return ImVec2(origin.x + width * (0.5F + 0.46F * std::clamp(value.x, -1.0F, 1.0F)),
                  origin.y + width * (0.5F - 0.46F * std::clamp(value.y, -1.0F, 1.0F)));
  };
  if (projection.ready()) {
    for (std::size_t index = 0; index < runtime.projected.size(); ++index) {
      const auto center = position(runtime.projected[index]);
      const bool selected = runtime.landmark_ids[index] == runtime.selected_landmark;
      draw->AddCircleFilled(center, selected ? 4.0F : 1.8F,
                            landmark_color(runtime.landmark_ids[index], selected ? 255 : 145));
    }
    const auto selected = runtime.trails.find(runtime.selected_landmark);
    if (selected != runtime.trails.end()) {
      const auto& history = selected->second.descriptor_positions;
      for (std::size_t index = 1; index < history.size(); ++index) {
        draw->AddLine(position(history[index - 1]), position(history[index]),
                      IM_COL32(255, 255, 255, 185), 1.5F);
      }
    }
    if (ImGui::IsItemHovered()) {
      const ImVec2 mouse = ImGui::GetMousePos();
      float nearest = 100.0F;
      std::size_t nearest_index = runtime.projected.size();
      for (std::size_t index = 0; index < runtime.projected.size(); ++index) {
        const auto point = position(runtime.projected[index]);
        const float dx = mouse.x - point.x;
        const float dy = mouse.y - point.y;
        if (dx * dx + dy * dy < nearest) {
          nearest = dx * dx + dy * dy;
          nearest_index = index;
        }
      }
      if (nearest_index < runtime.projected.size()) {
        ImGui::SetTooltip("Landmark %llu\nPC1 %.3f, PC2 %.3f",
            static_cast<unsigned long long>(runtime.landmark_ids[nearest_index]),
            runtime.projected[nearest_index].x,
            runtime.projected[nearest_index].y);
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
          runtime.selected_landmark = runtime.landmark_ids[nearest_index];
        }
      }
    }
  }
  ImGui::TextDisabled("PC1 horizontal; PC2 vertical. Color = track ID.");
  ImGui::Separator();
  ImGui::TextUnformatted("Track length histogram");
  std::array<float, kTrackLengthBins> bars{};
  float maximum = 1.0F;
  for (std::size_t index = 0; index < bars.size(); ++index) {
    bars[index] = std::log1p(static_cast<float>(histogram[index]));
    maximum = std::max(maximum, bars[index]);
  }
  ImGui::PlotHistogram("##track-lengths", bars.data(), static_cast<int>(bars.size()),
                       0, nullptr, 0.0F, maximum, {width, 170});
  if (ImGui::IsItemHovered()) {
    const float fraction = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) /
                           (ImGui::GetItemRectMax().x - ImGui::GetItemRectMin().x);
    const std::size_t bin = std::min(histogram.size() - 1,
        static_cast<std::size_t>(std::max(0.0F, fraction) * histogram.size()));
    ImGui::SetTooltip("%s observations: %llu tracks", length_bin_label(bin).c_str(),
                      static_cast<unsigned long long>(histogram[bin]));
  }
  ImGui::TextDisabled("Lengths 1-16, then doubling ranges; log count.");
  ImGui::End();
}

void draw_flow_diagnostics(Runtime& runtime, const FlowTracker& tracker) {
  if (!runtime.show_diagnostics) return;
  ImGui::SetNextWindowPos({1090, 8}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({340, 420}, ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Flow track diagnostics", &runtime.show_diagnostics)) {
    ImGui::End();
    return;
  }
  ImGui::Text("Tracks: %llu total / %zu active",
              static_cast<unsigned long long>(tracker.landmark_count()),
              tracker.active_count());
  ImGui::TextDisabled("SuperPoint seeds; flow prediction; SuperPoint correction.");
  ImGui::TextDisabled("GPU association; only current SuperPoints survive.");
  std::vector<float> corrections;
  for (float value : runtime.corrections) if (std::isfinite(value)) corrections.push_back(value);
  if (!corrections.empty()) {
    ImGui::PlotHistogram("Snap distance (px)", corrections.data(),
                         static_cast<int>(corrections.size()), 0, nullptr, 0,
                         *std::max_element(corrections.begin(), corrections.end()) + 0.1F, {0, 65});
  }
  ImGui::TextUnformatted("Track length histogram");
  const auto& histogram = tracker.length_histogram();
  std::array<float, kTrackLengthBins> bars{};
  float maximum = 1.0F;
  for (std::size_t index = 0; index < bars.size(); ++index) {
    bars[index] = std::log1p(static_cast<float>(histogram[index]));
    maximum = std::max(maximum, bars[index]);
  }
  const float width = std::max(120.0F, std::min(ImGui::GetContentRegionAvail().x, 306.0F));
  ImGui::PlotHistogram("##flow-track-lengths", bars.data(), static_cast<int>(bars.size()),
                       0, nullptr, 0.0F, maximum, {width, 170});
  if (ImGui::IsItemHovered()) {
    const float fraction = (ImGui::GetMousePos().x - ImGui::GetItemRectMin().x) /
                           (ImGui::GetItemRectMax().x - ImGui::GetItemRectMin().x);
    const std::size_t bin = std::min(histogram.size() - 1,
        static_cast<std::size_t>(std::max(0.0F, fraction) * histogram.size()));
    ImGui::SetTooltip("%s observations: %llu tracks", length_bin_label(bin).c_str(),
                      static_cast<unsigned long long>(histogram[bin]));
  }
  ImGui::TextDisabled("Lengths 1-16, then doubling ranges; log count.");
  ImGui::End();
}

// Turbo colormap (Mikhailov, 2019), polynomial approximation; t in [0, 1].
std::array<std::uint8_t, 3> turbo(float t) {
  t = std::clamp(t, 0.0F, 1.0F);
  const float t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
  const float r = 0.13572138F + 4.61539260F * t - 42.66032258F * t2 + 132.13108234F * t3 -
                  152.94239396F * t4 + 59.28637943F * t5;
  const float g = 0.09140261F + 2.19418839F * t + 4.84296658F * t2 - 14.18503333F * t3 +
                  4.27729857F * t4 + 2.82956604F * t5;
  const float b = 0.10667330F + 12.64194608F * t - 60.58204836F * t2 + 110.36276771F * t3 -
                  89.90310912F * t4 + 27.34824973F * t5;
  const auto byte = [](float v) { return static_cast<std::uint8_t>(std::clamp(v, 0.0F, 1.0F) * 255.0F + 0.5F); };
  return {byte(r), byte(g), byte(b)};
}

// Colours the latest depth map (log depth, near = red) into runtime.depth_texture.
void upload_depth_texture(Runtime& runtime) {
  const DepthMap& depth = runtime.depth;
  std::vector<float> valid;
  valid.reserve(depth.metres.size() / 7 + 1);
  for (std::size_t index = 0; index < depth.metres.size(); index += 7)
    if (depth.metres[index] > 0) valid.push_back(depth.metres[index]);
  if (valid.empty()) return;
  const auto percentile = [&](double q) {
    auto at = valid.begin() + static_cast<std::ptrdiff_t>(q * (valid.size() - 1));
    std::nth_element(valid.begin(), at, valid.end());
    return *at;
  };
  runtime.depth_near = std::max(0.05F, percentile(0.02));
  runtime.depth_far = std::max(runtime.depth_near * 1.01F, percentile(0.98));
  const float log_near = std::log(runtime.depth_near);
  const float inv_range = 1.0F / (std::log(runtime.depth_far) - log_near);
  std::vector<std::uint8_t> rgba(depth.metres.size() * 4);
  for (std::size_t index = 0; index < depth.metres.size(); ++index) {
    const float metres = depth.metres[index];
    const auto rgb = metres > 0 ? turbo(1.0F - (std::log(metres) - log_near) * inv_range)
                                : std::array<std::uint8_t, 3>{0, 0, 0};
    std::copy(rgb.begin(), rgb.end(), rgba.begin() + static_cast<std::ptrdiff_t>(index * 4));
    rgba[index * 4 + 3] = 255;
  }
  if (!runtime.depth_texture) {
    glGenTextures(1, &runtime.depth_texture);
    glBindTexture(GL_TEXTURE_2D, runtime.depth_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }
  glBindTexture(GL_TEXTURE_2D, runtime.depth_texture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  if (runtime.depth_texture_width != depth.width || runtime.depth_texture_height != depth.height) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, depth.width, depth.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    runtime.depth_texture_width = depth.width;
    runtime.depth_texture_height = depth.height;
  } else {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, depth.width, depth.height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
  }
}

constexpr const char* kFusionViews[] = {"Fused depth", "Raw network depth", "Consistency (sigma of z+b)",
                                        "Uncertainty (sigma of z)", "Fused - raw", "Observations"};

// Colours the selected fusion view into runtime.fusion.texture (transparent where empty).
void upload_fusion_texture(Runtime::Fusion& f) {
  const DepthFilterView& fused = f.fused;
  const DepthMap& raw = f.raw;
  const std::size_t n = static_cast<std::size_t>(fused.width) * static_cast<std::size_t>(fused.height);
  if (n == 0 || raw.metres.size() != n) return;
  std::vector<float> valid;
  for (std::size_t i = 0; i < n; i += 7)
    if (raw.metres[i] > 0) valid.push_back(raw.metres[i]);
  if (!valid.empty()) {
    const auto percentile = [&](double q) {
      auto at = valid.begin() + static_cast<std::ptrdiff_t>(q * (valid.size() - 1));
      std::nth_element(valid.begin(), at, valid.end());
      return *at;
    };
    f.near = std::max(0.05F, percentile(0.02));
    f.far = std::max(f.near * 1.01F, percentile(0.98));
  }
  const float log_near = std::log(std::max(0.05F, f.near));
  const float inv_range = 1.0F / std::max(1e-3F, std::log(std::max(f.far, f.near * 1.01F)) - log_near);
  // Sigma views: log scale from 0.5 % (blue, consistent) to 30 % (red).
  const float sigma_lo = std::log(0.005F), sigma_inv = 1.0F / (std::log(0.3F) - sigma_lo);
  std::vector<std::uint8_t> rgba(n * 4, 0);
  for (std::size_t i = 0; i < n; ++i) {
    float t = -1;
    switch (f.view) {
      case 0: if (fused.metres[i] > 0) t = 1.0F - (std::log(fused.metres[i]) - log_near) * inv_range; break;
      case 1: if (raw.metres[i] > 0) t = 1.0F - (std::log(raw.metres[i]) - log_near) * inv_range; break;
      case 2: if (fused.consistency[i] > 0) t = (std::log(fused.consistency[i]) - sigma_lo) * sigma_inv; break;
      case 3: if (fused.sigma[i] > 0) t = (std::log(fused.sigma[i]) - sigma_lo) * sigma_inv; break;
      case 4:  // +-15 % around green
        if (fused.metres[i] > 0 && raw.metres[i] > 0) t = 0.5F + std::log(fused.metres[i] / raw.metres[i]) / 0.3F;
        break;
      case 5: if (fused.observations[i] > 0) t = std::log1p(static_cast<float>(fused.observations[i])) / std::log1p(100.0F); break;
      default: break;
    }
    // Fused depth: pixels the anchor keyframe never saw show the raw map, dimmed.
    bool dim = false;
    if (f.view == 0 && t < 0 && raw.metres[i] > 0) {
      t = 1.0F - (std::log(raw.metres[i]) - log_near) * inv_range;
      dim = true;
    }
    if (t < 0 && !(f.view == 4 && fused.metres[i] > 0 && raw.metres[i] > 0)) continue;
    auto rgb = turbo(t);
    if (dim)
      for (auto& c : rgb) c = static_cast<std::uint8_t>(c / 3);
    std::copy(rgb.begin(), rgb.end(), rgba.begin() + static_cast<std::ptrdiff_t>(i * 4));
    rgba[i * 4 + 3] = 255;
  }
  if (!f.texture) {
    glGenTextures(1, &f.texture);
    glBindTexture(GL_TEXTURE_2D, f.texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  }
  glBindTexture(GL_TEXTURE_2D, f.texture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  if (f.texture_width != fused.width || f.texture_height != fused.height) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, fused.width, fused.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    f.texture_width = fused.width;
    f.texture_height = fused.height;
  } else {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fused.width, fused.height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
  }
}

void draw_fusion(Runtime& runtime) {
  auto& f = runtime.fusion;
  if (!f.show) return;
  ImGui::SetNextWindowPos({700, 300}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({720, 440}, ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Depth fusion", &f.show)) {
    ImGui::End();
    return;
  }
  ImGui::Checkbox("Run fusion", &f.enabled);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Per-pixel depth filter (DEPTH_FUSION.md): each network depth map is fused into a state\n"
                      "anchored to the current keyframe, using the camera pose. CPU, ~40 ms per map.\n"
                      "While on, keyframe depth clouds are built from the fused map instead of the raw one.");
  ImGui::SameLine();
  ImGui::TextDisabled("keyframe clouds: %d fused, %d raw", f.keyframes_fused, f.keyframes_raw);
  if (runtime.depth_model < 0) {
    ImGui::TextDisabled("Pick a depth model in the Depth window first.");
  } else if (!f.enabled) {
    ImGui::TextDisabled("Off.");
  } else if (!f.status.empty()) {
    ImGui::TextDisabled("%s", f.status.c_str());
  }
  const DepthFilterStats& st = f.stats;
  if (f.fused.width > 0) {
    ImGui::Text("Frame %llu (video at %llu), anchor keyframe %llu, %.1f ms",
                static_cast<unsigned long long>(f.raw.frame_index), static_cast<unsigned long long>(runtime.frame_index),
                static_cast<unsigned long long>(f.fused.frame_index), f.ms);
    const double updates = std::max(1, st.updated + st.gated);
    ImGui::Text("Valid %d px, gated %.1f%%, resets %d, occluded %d%s", st.valid, 100.0 * st.gated / updates, st.resets,
                st.occluded, st.reanchored ? ", re-anchored" : "");
    if (st.flicker_pixels > 0)
      ImGui::Text("Flicker (median |d log z|): raw %.4f, fused %.4f", st.flicker_raw, st.flicker_fused);
  }
  ImGui::SetNextItemWidth(230);
  if (ImGui::BeginCombo("View", kFusionViews[f.view])) {
    for (int i = 0; i < static_cast<int>(std::size(kFusionViews)); ++i)
      if (ImGui::Selectable(kFusionViews[i], f.view == i)) f.view = i;
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  ImGui::Checkbox("Overlay on video", &f.overlay);
  if (f.overlay) {
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::SliderFloat("##opacity", &f.opacity, 0.0F, 1.0F, "%.2f");
  }
  switch (f.view) {
    case 0: ImGui::TextDisabled("Log depth: red %.2f m ... blue %.2f m (dimmed: raw, outside the keyframe's view)", f.near, f.far); break;
    case 1: ImGui::TextDisabled("Log depth: red %.2f m ... blue %.2f m", f.near, f.far); break;
    case 2: ImGui::TextDisabled("How well successive maps agree: blue 0.5 %% ... red 30 %%"); break;
    case 3: ImGui::TextDisabled("Includes the network's shared bias (only geometry reduces it): blue 0.5 %% ... red 30 %%"); break;
    case 4: ImGui::TextDisabled("log(fused / raw): blue -15 %% ... green 0 ... red +15 %%"); break;
    case 5: ImGui::TextDisabled("Maps fused per pixel: blue 1 ... red 100"); break;
    default: break;
  }
  if (f.dirty || f.uploaded_view != f.view) {
    upload_fusion_texture(f);
    f.dirty = false;
    f.uploaded_view = f.view;
  }
  if (!f.texture || f.fused.width == 0 || f.raw.metres.empty()) {
    ImGui::End();
    return;
  }
  // Depth content: the upright frame inside the letterboxed output grid.
  const DepthMap& geometry = f.raw;
  const bool sideways = geometry.rotation == 90 || geometry.rotation == 270;
  const float upright_width = static_cast<float>(sideways ? geometry.source_height : geometry.source_width);
  const float upright_height = static_cast<float>(sideways ? geometry.source_width : geometry.source_height);
  const ImVec2 uv0(geometry.offset_x / geometry.width, geometry.offset_y / geometry.height);
  const ImVec2 uv1((geometry.offset_x + upright_width * geometry.scale_x) / geometry.width,
                   (geometry.offset_y + upright_height * geometry.scale_y) / geometry.height);
  const ImVec2 available = ImGui::GetContentRegionAvail();
  const float panels = f.overlay ? 1.0F : 2.0F;
  const float gap = f.overlay ? 0.0F : 6.0F;
  const float scale = std::max(0.01F, std::min((available.x - gap) / (panels * upright_width), available.y / upright_height));
  const ImVec2 size(upright_width * scale, upright_height * scale);
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImDrawList* draw = ImGui::GetWindowDrawList();
  // The video frame (current, so it may lead the fused frame by a frame or two).
  if (runtime.texture && runtime.frame_width > 0) {
    const ImVec2 corners[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    const int first = (4 - runtime.display_rotation / 90) % 4;
    draw->AddImageQuad(static_cast<ImTextureID>(runtime.texture), origin, {origin.x + size.x, origin.y},
                       {origin.x + size.x, origin.y + size.y}, {origin.x, origin.y + size.y}, corners[first],
                       corners[(first + 1) % 4], corners[(first + 2) % 4], corners[(first + 3) % 4]);
  }
  const ImVec2 depth_origin = f.overlay ? origin : ImVec2(origin.x + size.x + gap, origin.y);
  const auto alpha = static_cast<int>(std::clamp(f.overlay ? f.opacity : 1.0F, 0.0F, 1.0F) * 255.0F);
  draw->AddImage(static_cast<ImTextureID>(f.texture), depth_origin, {depth_origin.x + size.x, depth_origin.y + size.y},
                 uv0, uv1, IM_COL32(255, 255, 255, alpha));
  ImGui::InvisibleButton("##fusion-image", {f.overlay ? size.x : 2 * size.x + gap, size.y});
  if (ImGui::IsItemHovered()) {
    ImVec2 mouse = ImGui::GetMousePos();
    if (!f.overlay && mouse.x >= depth_origin.x) mouse.x -= size.x + gap;  // either panel
    const float u = uv0.x + (uv1.x - uv0.x) * (mouse.x - origin.x) / size.x;
    const float v = uv0.y + (uv1.y - uv0.y) * (mouse.y - origin.y) / size.y;
    const int x = std::clamp(static_cast<int>(u * f.fused.width), 0, f.fused.width - 1);
    const int y = std::clamp(static_cast<int>(v * f.fused.height), 0, f.fused.height - 1);
    const std::size_t i = static_cast<std::size_t>(y) * f.fused.width + x;
    const float fused = f.fused.metres[i], raw = f.raw.metres[i];
    if (fused > 0)
      ImGui::SetTooltip("fused %.3f m   raw %.3f m (%+.1f%%)\nconsistency %.1f%%   uncertainty %.1f%%\nbias %+.1f%%   maps %d",
                        fused, raw, raw > 0 ? 100.0 * (fused / raw - 1) : 0.0, 100.0 * f.fused.consistency[i],
                        100.0 * f.fused.sigma[i], 100.0 * f.fused.bias[i], f.fused.observations[i]);
    else
      ImGui::SetTooltip("no fused depth   raw %.3f m", raw);
  }
  ImGui::End();
}

// The current (bundle-adjusted) pose of a recent frame; nullopt if it has none.
std::optional<Pose> current_pose(const VisualOdometry& odometry, std::uint64_t frame_index) {
  const std::size_t n = odometry.trajectory_size();
  for (const std::size_t back : {std::size_t{64}, n}) {
    const auto samples = odometry.trajectory(n > back ? n - back : 0);
    const auto it = std::lower_bound(samples.begin(), samples.end(), frame_index,
                                     [](const TrajectorySample& s, std::uint64_t f) { return s.frame_index < f; });
    if (it != samples.end() && it->frame_index == frame_index) return it->pose;
    if (back >= n || (!samples.empty() && samples.front().frame_index <= frame_index)) break;
  }
  return std::nullopt;
}

void draw_depth(Runtime& runtime) {
  if (!runtime.show_depth) return;
  ImGui::SetNextWindowPos({1090, 8}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({360, 420}, ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Depth", &runtime.show_depth)) {
    ImGui::End();
    return;
  }
  const auto& presets = depth_model_presets();
  const char* current = runtime.depth_model >= 0 ? presets[runtime.depth_model].name.c_str() : "Off";
  ImGui::SetNextItemWidth(-1);
  if (ImGui::BeginCombo("##depth-model", current)) {
    if (ImGui::Selectable("Off", runtime.depth_model < 0)) runtime.depth_model = -1;
    for (int index = 0; index < static_cast<int>(presets.size()); ++index) {
      if (ImGui::Selectable(presets[index].name.c_str(), runtime.depth_model == index)) {
        runtime.depth_model = index;
        runtime.depth_error.clear();
      }
    }
    ImGui::EndCombo();
  }
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Monocular metric depth (TensorRT), on its own CUDA stream; a frame is\n"
                      "submitted whenever the previous one has finished. Next frame.");
  if (!runtime.depth_error.empty()) ImGui::TextColored({1, 0.5F, 0.4F, 1}, "%s", runtime.depth_error.c_str());
  const DepthMap& depth = runtime.depth;
  if (runtime.depth_model < 0 || depth.empty()) {
    if (runtime.depth_model >= 0) ImGui::TextDisabled("Waiting for the first depth map...");
    ImGui::End();
    return;
  }
  if (runtime.depth_dirty) {
    upload_depth_texture(runtime);
    runtime.depth_dirty = false;
  }
  ImGui::Text("GPU: %.2f ms   frame %llu (%lld behind)", depth.gpu_ms,
              static_cast<unsigned long long>(depth.frame_index),
              static_cast<long long>(runtime.frame_index) - static_cast<long long>(depth.frame_index));
  ImGui::Text("Range: %.2f - %.2f m (2nd-98th pct, log)", runtime.depth_near, runtime.depth_far);
  if (!runtime.depth_texture) {
    ImGui::End();
    return;
  }
  // Crop the letterbox: the content is the upright frame, scaled.
  const bool sideways = depth.rotation == 90 || depth.rotation == 270;
  const float upright_width = static_cast<float>(sideways ? depth.source_height : depth.source_width);
  const float upright_height = static_cast<float>(sideways ? depth.source_width : depth.source_height);
  const ImVec2 uv0(depth.offset_x / depth.width, depth.offset_y / depth.height);
  const ImVec2 uv1((depth.offset_x + upright_width * depth.scale_x) / depth.width,
                   (depth.offset_y + upright_height * depth.scale_y) / depth.height);
  const ImVec2 available = ImGui::GetContentRegionAvail();
  const float scale = std::max(0.01F, std::min(available.x / upright_width, available.y / upright_height));
  const ImVec2 size(upright_width * scale, upright_height * scale);
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImGui::Image(static_cast<ImTextureID>(runtime.depth_texture), size, uv0, uv1);
  if (ImGui::IsItemHovered()) {
    const ImVec2 mouse = ImGui::GetMousePos();
    const float u = uv0.x + (uv1.x - uv0.x) * (mouse.x - origin.x) / size.x;
    const float v = uv0.y + (uv1.y - uv0.y) * (mouse.y - origin.y) / size.y;
    const int x = std::clamp(static_cast<int>(u * depth.width), 0, depth.width - 1);
    const int y = std::clamp(static_cast<int>(v * depth.height), 0, depth.height - 1);
    ImGui::SetTooltip("%.2f m", depth.metres[static_cast<std::size_t>(y) * depth.width + x]);
  }
  ImGui::End();
}

struct Session {
  explicit Session(AppConfig settings) : config(std::move(settings)) {
    decoder = make_ffmpeg_cuda_decoder();
    decoder->open(config.video);
    presenter = make_cuda_gl_presenter();
    optical_flow = std::make_unique<OpticalFlow>(config.optical_flow);
    superpoint = make_tensorrt_superpoint(config.engine, config.superpoint);
    store = std::make_unique<FeatureStore>(config.database, config.store);
    store->set_session(config.video, config.engine, config.superpoint.input_width,
                       config.superpoint.input_height, config.superpoint.max_keypoints,
                       config.superpoint.detection_threshold);
    flow_tracker = std::make_unique<FlowTracker>(config.flow_tracker);
    if (config.odometry.intrinsics) {
      runtime.intrinsics_source = "--intrinsics";
    } else if (auto calibration = read_calibration(config.video)) {
      config.odometry.intrinsics = calibration;
      runtime.intrinsics_source = "calibration.json";
    }
    odometry = std::make_unique<VisualOdometry>(config.odometry);
    trajectory_view.set_depth_store(&keyframe_depth);
    next_cache_index = static_cast<std::uint64_t>(store->last_frame_index() + 1);
    runtime.association_radius = config.flow_tracker.association_radius;
    runtime.metadata_rotation = decoder->display_rotation();
    runtime.display_rotation = config.display_rotation.value_or(runtime.metadata_rotation);
    runtime.flow_sigma = config.flow_tracker.flow_sigma_px;
    runtime.detection_sigma = config.flow_tracker.detection_sigma_px;
    if (!config.depth_model.empty()) {
      const auto& presets = depth_model_presets();
      const auto* spec = find_depth_model(config.depth_model);
      if (spec) runtime.depth_model = static_cast<int>(spec - presets.data());
      else runtime.depth_error = "Unknown depth model: " + config.depth_model;
    }
  }

  void seek(double seconds) {
    if (!seek_decoder) {
      seek_decoder = make_ffmpeg_cuda_decoder();
      seek_decoder->open(config.video);
    }
    const auto duration = decoder->duration_ns();
    const auto frame_interval = decoder->nominal_fps() > 0 ?
        static_cast<std::int64_t>(1e9 / decoder->nominal_fps()) : 33333333LL;
    const auto relative = static_cast<std::int64_t>(std::max(0.0, seconds) * 1e9);
    const auto clamped = duration > 0 ?
        std::min(relative, std::max<std::int64_t>(0, duration - frame_interval)) : relative;
    const auto target = decoder->start_time_ns() + clamped;
    seek_decoder->seek_ns(target);
    GpuFrame frame;
    bool found = false;
    bool had_previous = false;
    std::int64_t previous_timestamp{};
    while (seek_decoder->next(frame)) {
      if (frame.timestamp_ns >= target) {
        found = true;
        break;
      }
      had_previous = true;
      previous_timestamp = frame.timestamp_ns;
    }
    if (!found && had_previous) {
      seek_decoder->seek_ns(previous_timestamp);
      while (seek_decoder->next(frame)) {
        if (frame.timestamp_ns >= previous_timestamp) {
          found = true;
          break;
        }
      }
    }
    if (!found) throw std::runtime_error("No frame found near the selected time");
    // Finish pending feature writes before revisiting frames, avoiding duplicate cache inserts.
    store->flush();
    decoder.swap(seek_decoder); // The decoder now sits at the displayed frame.
    optical_flow = std::make_unique<OpticalFlow>(config.optical_flow);
    flow_tracker = std::make_unique<FlowTracker>(config.flow_tracker);
    // Track IDs restart with the tracker, so the camera trajectory does too.
    odometry = std::make_unique<VisualOdometry>(config.odometry);
    trajectory_view.reset();
    reset_depth_pipeline();
    focal.restart_tracks();
    runtime.odometry = {};
    runtime.relocalizations = 0;
    runtime.loops_closed = 0;
    runtime.last_loop.clear();
    runtime.global_searches = runtime.global_verified = 0;
    runtime.global_total_ms = 0;
    runtime.reassociations = 0;
    runtime.last_relocalization.clear();
    runtime.trails.clear();
    runtime.projected.clear();
    runtime.selected_landmark = std::numeric_limits<std::uint64_t>::max();
    runtime.eof = false;
    runtime.stopped = false;
    runtime.stop_requested = false;
    runtime.step_requested = false;
    runtime.playing = false;
    runtime.stats = {};
    runtime.started = Clock::now();
    process_frame(frame, 0);
    runtime.scrub_seconds = static_cast<double>(frame.timestamp_ns - decoder->start_time_ns()) / 1e9;
    runtime.seek_error.clear();
  }

  void advance() {
    poll_depth();
    drain_pending();
    if (runtime.seek_requested || runtime.restart_requested) {
      const bool restart = runtime.restart_requested;
      runtime.restart_requested = false;
      runtime.seek_requested = false;
      try {
        seek(restart ? 0.0 : runtime.seek_seconds);
        runtime.playing = restart;
      } catch (const std::exception& error) {
        runtime.seek_error = error.what();
        runtime.playing = false;
      }
      return;
    }
    const auto now = Clock::now();
    const bool pacing_ready = !runtime.realtime_pacing || now >= runtime.next_frame;
    if (runtime.eof || runtime.stopped ||
        !(runtime.step_requested || (runtime.playing && pacing_ready))) return;
    GpuFrame frame;
    const auto decode_start = Clock::now();
    if (!decoder->next(frame)) {
      runtime.eof = true;
      runtime.playing = false;
    } else {
      const double decode_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - decode_start).count();
      process_frame(frame, decode_ms);
    }
    runtime.step_requested = false;
  }

  void process_frame(const GpuFrame& frame, double decode_ms) {
    runtime.texture = presenter->present(frame);
    runtime.frame_width = frame.width;
    runtime.frame_height = frame.height;
    runtime.frame_index = frame.frame_index;
    runtime.timestamp_ns = frame.timestamp_ns;
    runtime.seek_error.clear();
    runtime.stats.decoded_frames++;
    runtime.stats.decode_ms = decode_ms;
    auto cached = store->load_frame(frame);
    FrameFeatures features;
    if (cached) {
      features = std::move(*cached);
      features.frame_index = frame.frame_index;
    } else {
      features = superpoint->infer(frame);
      features.decode_ms = decode_ms;
      auto cache_features = features;
      // Cache row IDs must not use the decoder's seek-local frame ordinal.
      cache_features.frame_index = next_cache_index++;
      store->enqueue(std::move(cache_features));
    }
    const bool depth_submitted = submit_depth(frame);
    runtime.detections = features.keypoints; // Preserve all detections before the tracking cap.
    // GPU hot path: flow fields, detections and track state stay on the device.
    std::optional<DeviceFlowField> field;
    if (optical_flow->has_reference()) field = optical_flow->compute(frame);
    else optical_flow->remember(frame);
    const DeviceDetections device = cached ? DeviceDetections{} : superpoint->device_detections();
    flow_tracker->set_association_radius(runtime.association_radius);
    flow_tracker->set_flow_sigma(runtime.flow_sigma);
    flow_tracker->set_detection_sigma(runtime.detection_sigma);
    flow_tracker->associate(features, field ? &*field : nullptr, cached ? nullptr : &device);
    runtime.flow_ms = field ? optical_flow->gpu_ms() : 0.0;
    // Host flow copy only for the optional overlay.
    runtime.flow_field.reset();
    if (field && runtime.show_flow_vectors) runtime.flow_field = optical_flow->download();
    // Camera pose from the observed (non-coasted) tracks; any tracker's output
    // converted to a TrackedFrame could drive this instead.
    TrackedFrame tracked = tracked_frame_from(features, color_sampler.sample(frame, features.keypoints));
    if (depth) {
      // The VO runs one frame behind, so each frame meets its own depth map
      // (DEPTH_INTEGRATION.md "Timing"). Colours for the keyframe cloud grid
      // are sampled now, while the frame is alive.
      std::vector<std::array<std::uint8_t, 3>> grid_colors;
      if (!runtime.depth.empty()) grid_colors = color_sampler.sample(frame, keyframe_depth.grid(runtime.depth));
      pending.push_back({std::move(tracked), std::move(grid_colors), depth_submitted});
      drain_pending();
    } else {
      drain_pending(true);
      run_odometry(tracked, nullptr, {});
    }
    runtime.points = features.keypoints;
    runtime.supported = features.superpoint_supported;
    runtime.landmark_ids = features.landmark_ids;
    runtime.similarities = features.landmark_similarities;
    runtime.predictions = features.flow_predictions;
    runtime.corrections = features.correction_distances;
    runtime.cache_hit = cached.has_value();
    runtime.new_landmarks = features.new_landmarks;
    runtime.matched_landmarks = features.matched_landmarks;
    runtime.coasted_landmarks = features.coasted_landmarks;
    runtime.tracking_ms = features.tracking_ms;
    update_trails(runtime);
    runtime.stats.inferred_frames++;
    runtime.stats.preprocess_ms = features.preprocess_ms;
    runtime.stats.inference_ms = features.inference_ms;
    runtime.stats.readback_ms = features.readback_ms;
    const double fps = decoder->nominal_fps();
    runtime.next_frame = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                   std::chrono::duration<double>(fps > 0 ? 1.0 / fps : 0.0));
  }

  // VO on one tracked frame (with its depth map, when it has one), and the
  // keyframe's dense cloud.
  void run_odometry(TrackedFrame& tracked, const DepthMap* map,
                    const std::vector<std::array<std::uint8_t, 3>>& grid_colors) {
    if (map) {
      DepthSamplingConfig sampling;
      sampling.max_depth_m = depth ? depth->spec().max_depth_m : 0.0F;
      attach_depth(*map, tracked, sampling);
    }
    runtime.odometry = odometry->process(tracked);
    runtime.reassociations += static_cast<std::size_t>(runtime.odometry.reassociated);
    if (runtime.odometry.relocalized) {
      ++runtime.relocalizations;
      runtime.last_relocalization = "frame " + std::to_string(runtime.odometry.frame_index) + ": " +
                                    runtime.odometry.event;
    }
    const auto& r = runtime.odometry;
    for (const auto& [frame, factor] : r.keyframe_scale_changes) keyframe_depth.rescale(frame, factor);
    if (r.global_search) {
      ++runtime.global_searches;
      runtime.global_verified += r.global_found;
      runtime.global_total_ms += r.global_match_ms;
      runtime.last_global = r;
    }
    if (r.loop_closed) {
      ++runtime.loops_closed;
      runtime.last_loop = "frame " + std::to_string(r.frame_index) + ": " + r.event;
    }
    if (r.settled_keyframe >= 0)  // that keyframe's scale from its final geometry
      keyframe_depth.settle(static_cast<std::uint64_t>(r.settled_keyframe), r.settled_log_scale,
                            r.settled_scale_grid_valid ? &r.settled_scale_grid : nullptr);
    // Depth fusion first: at a keyframe the filter re-anchors there, and its
    // state becomes the keyframe's (fused) depth map for the cloud.
    if (map) run_fusion(tracked, *map, r);
    if (map && r.keyframe && r.has_pose) {
      // Local scale: this keyframe's own grid/estimate (the VO's scale drifts
      // within a segment), else the segment's filtered one, else unknown.
      KeyframeDepthInput input;
      input.frame_index = r.frame_index;
      input.segment = r.segment;
      input.log_scale = std::isfinite(r.keyframe_log_scale) ? r.keyframe_log_scale :
                        r.metric_scale > 0 ? std::log(r.metric_scale) : std::numeric_limits<double>::quiet_NaN();
      input.grid = r.keyframe_scale_grid_valid ? &r.keyframe_scale_grid : nullptr;
      input.pose = r.pose;
      // References for the depth confidence model (keyframe_depth.hpp).
      input.references = depth_references(*odometry, tracked, r.pose);
      const auto K = config.odometry.intrinsics ? *config.odometry.intrinsics :
          CameraIntrinsics::from_horizontal_fov(tracked.width, tracked.height, config.odometry.horizontal_fov_degrees);
      const auto samples = odometry->trajectory();  // current (bundle-adjusted) keyframe poses
      const auto pose_of = [&](std::uint64_t frame) -> std::optional<Pose> {
        const auto it = std::lower_bound(samples.begin(), samples.end(), frame,
                                         [](const TrajectorySample& s, std::uint64_t f) { return s.frame_index < f; });
        if (it == samples.end() || it->frame_index != frame) return std::nullopt;
        return it->pose;
      };
      // Fused when the filter is anchored at this keyframe; the grid shape it
      // puts back is the one the store removes again.
      DepthMap fused;
      const bool use_fused = runtime.fusion.enabled && !depth_filter.empty() && depth_filter.frame_index() == r.frame_index;
      if (use_fused) fused = depth_filter.to_depth_map(input.log_scale, input.grid);
      keyframe_depth.add(input, use_fused ? fused : *map, grid_colors, K, config.odometry.distortion_k1, pose_of);
      runtime.fusion.keyframes_fused += use_fused;
      runtime.fusion.keyframes_raw += !use_fused;
    }
    trajectory_view.update(*odometry);
    focal.add(tracked);  // background thread
  }

  // Depth fusion on a frame with its depth map (DEPTH_FUSION.md). Network
  // maps are converted with the latest keyframe's scale; the filter's own
  // sigma model learns the raw network's error at keyframes.
  void run_fusion(const TrackedFrame& tracked, const DepthMap& map, const OdometryFrameResult& r) {
    auto& f = runtime.fusion;
    if (!f.enabled) {
      if (!depth_filter.empty()) depth_filter.clear();
      return;
    }
    const float max_metres = depth ? depth->spec().max_depth_m : 0.0F;
    if (depth_filter.config().max_metres != max_metres) {
      DepthFilterConfig filter_config;
      filter_config.max_metres = max_metres;
      depth_filter = DepthFilter(filter_config);
    }
    if (r.segment != fusion_segment) {
      fusion_log_scale = std::numeric_limits<double>::quiet_NaN();
      fusion_has_grid = false;
      fusion_segment = r.segment;
    }
    if (r.keyframe && r.has_pose && std::isfinite(r.keyframe_log_scale)) {
      fusion_log_scale = r.keyframe_log_scale;
      fusion_has_grid = r.keyframe_scale_grid_valid;
      if (fusion_has_grid) fusion_grid = r.keyframe_scale_grid;
      add_depth_references(fusion_sigma_model, map, depth_references(*odometry, tracked, r.pose), r.keyframe_log_scale,
                           fusion_has_grid ? &fusion_grid : nullptr);
      fusion_sigma_model.fit(4);
    }
    // Loop corrections move past poses and units: start over.
    if (!r.has_pose || r.relocalized || r.loop_closed || !std::isfinite(fusion_log_scale)) {
      depth_filter.clear();
      f.status = !r.has_pose ? "Waiting for a camera pose." :
                 !std::isfinite(fusion_log_scale) ? "Waiting for the first keyframe's depth scale." :
                 "Restarted after a relocalisation or loop closure.";
      return;
    }
    f.status.clear();
    const auto t0 = Clock::now();
    std::vector<float> sigma_map;
    if (fusion_sigma_model.fitted()) {
      std::vector<Keypoint> landmarks;  // the nearest_landmark cue
      for (const auto& o : tracked.observations)
        if (odometry->has_landmark(o.track_id)) landmarks.push_back({o.x, o.y, 0});
      sigma_map = network_sigma_map(map, fusion_sigma_model, landmarks);
    }
    DepthFilterFrame input;
    input.depth = &map;
    input.pose = r.pose;
    input.segment = r.segment;
    input.keyframe = r.keyframe;
    input.K = config.odometry.intrinsics ? *config.odometry.intrinsics :
        CameraIntrinsics::from_horizontal_fov(tracked.width, tracked.height, config.odometry.horizontal_fov_degrees);
    input.k1 = config.odometry.distortion_k1;
    input.log_scale = fusion_log_scale;
    input.grid = fusion_has_grid ? &fusion_grid : nullptr;
    input.sigma = sigma_map.empty() ? nullptr : &sigma_map;
    input.rotation_sigma_degrees = r.rotation_sigma_degrees;
    input.translation_sigma_ratio = r.translation_sigma_ratio;
    // The anchor keyframe's pose as the VO has refined it since (bundle
    // adjustment): the frame's pose is relative to the refined map.
    if (!depth_filter.empty())
      if (const auto anchor = current_pose(*odometry, depth_filter.frame_index())) depth_filter.set_anchor_pose(*anchor);
    f.stats = depth_filter.process(input);
    depth_filter.render(r.pose, f.fused);
    f.raw = map;
    f.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    f.dirty = true;
  }

  // Runs the VO on queued frames whose depth has arrived (or was never
  // submitted). `force`, or more than two queued frames, runs them anyway.
  void drain_pending(bool force = false) {
    while (!pending.empty()) {
      auto& next = pending.front();
      const auto arrived = depth_maps.find(next.tracked.frame_index);
      const bool ready = arrived != depth_maps.end() || !next.depth_submitted;
      if (!ready && !force && pending.size() <= 2) break;
      run_odometry(next.tracked, arrived != depth_maps.end() ? &arrived->second : nullptr, next.grid_colors);
      pending.pop_front();
    }
  }

  void reset_depth_pipeline() {
    pending.clear();
    depth_maps.clear();
    keyframe_depth.clear();
    depth_filter.clear();
    runtime.fusion.fused = {};
    runtime.fusion.raw = {};
    runtime.fusion.keyframes_fused = runtime.fusion.keyframes_raw = 0;
    runtime.fusion.dirty = true;
  }

  // (Re)creates the depth engine for the selected model and display
  // orientation, then queues this frame unless the last one is still running.
  // Returns whether this frame was submitted.
  bool submit_depth(const GpuFrame& frame) {
    poll_depth();
    if (runtime.depth_model < 0) {
      depth.reset();
      runtime.depth = {};
      return false;
    }
    const auto& spec = depth_model_presets()[runtime.depth_model];
    const int rotation = runtime.display_rotation;
    if (!depth || depth->spec().name != spec.name || depth->rotation() != rotation) {
      depth.reset();
      runtime.depth = {};
      const bool sideways = rotation == 90 || rotation == 270;
      const bool portrait = (sideways ? frame.width : frame.height) > (sideways ? frame.height : frame.width);
      try {
        depth = std::make_unique<DepthEstimator>(spec, depth_engine_path(config.models_dir, spec, portrait), rotation);
      } catch (const std::exception& error) {
        runtime.depth_error = error.what();
        runtime.depth_model = -1;
        return false;
      }
    }
    // The pose's lens, so canonical-camera models agree with the VO.
    const float focal = static_cast<float>(config.odometry.intrinsics ? config.odometry.intrinsics->fx :
        CameraIntrinsics::from_horizontal_fov(frame.width, frame.height, config.odometry.horizontal_fov_degrees).fx);
    try {
      return depth->submit(frame, focal);
    } catch (const std::exception& error) {
      runtime.depth_error = error.what();
      runtime.depth_model = -1;
      depth.reset();
      return false;
    }
  }

  void poll_depth() {
    if (depth && depth->poll(runtime.depth)) {
      runtime.depth_dirty = true;
      depth_maps[runtime.depth.frame_index] = runtime.depth;  // for the delayed VO
      while (depth_maps.size() > 4) depth_maps.erase(depth_maps.begin());
    }
  }

  // Restart the camera pose from the current frame with a new lens model.
  void apply_lens(const LensChoice& lens) {
    config.odometry.intrinsics.reset();
    runtime.intrinsics_source.clear();
    config.odometry.horizontal_fov_degrees = lens.hfov_degrees;
    config.odometry.distortion_k1 = lens.k1;
    odometry = std::make_unique<VisualOdometry>(config.odometry);
    trajectory_view.reset();
    reset_depth_pipeline();
    runtime.odometry = {};
    runtime.relocalizations = 0;
    runtime.loops_closed = 0;
    runtime.last_loop.clear();
    runtime.global_searches = runtime.global_verified = 0;
    runtime.global_total_ms = 0;
    runtime.reassociations = 0;
    runtime.last_relocalization.clear();
  }

  // trajectory.csv, map.csv and camera.txt (the intrinsics they were solved
  // with) in <out_root>/<video stem>/<local time>/. Formats: track_io.hpp.
  void export_results(const std::filesystem::path& out_root) {
    try {
      if (!odometry) throw std::runtime_error("no camera pose yet");
      char stamp[32];
      const std::time_t now = std::time(nullptr);
      std::strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", std::localtime(&now));
      const auto folder = out_root / config.video.stem() / stamp;
      std::filesystem::create_directories(folder);
      // Global bundle adjustment first (all keyframes, every sighting): the
      // live map is updated too (map_generation changes, viewers refetch).
      odometry->global_bundle_adjust();
      const auto samples = odometry->trajectory();
      std::ofstream trajectory(folder / "trajectory.csv");
      write_trajectory_csv(trajectory, samples);
      std::ofstream map(folder / "map.csv");
      write_map_csv(map, *odometry);
      std::ofstream camera(folder / "camera.txt");
      camera << "video " << config.video.string() << "\nsize " << runtime.frame_width << ' '
             << runtime.frame_height << '\n';
      if (const auto& K = config.odometry.intrinsics)
        camera << "intrinsics " << K->fx << ' ' << K->fy << ' ' << K->cx << ' ' << K->cy << '\n';
      else
        camera << "hfov_degrees " << config.odometry.horizontal_fov_degrees << '\n';
      camera << "k1 " << config.odometry.distortion_k1 << '\n';
      if (!trajectory || !map || !camera) throw std::runtime_error("write failed in " + folder.string());
      runtime.export_status = "Exported " + std::to_string(samples.size()) + " poses to " +
                              folder.string();
    } catch (const std::exception& error) {
      runtime.export_status = std::string("Export failed: ") + error.what();
    }
  }

  std::uint64_t next_cache_index{};
  AppConfig config;
  std::unique_ptr<VideoDecoder> decoder;
  std::unique_ptr<VideoDecoder> seek_decoder;
  std::unique_ptr<CudaFramePresenter> presenter;
  std::unique_ptr<SuperPoint> superpoint;
  std::unique_ptr<OpticalFlow> optical_flow;
  std::unique_ptr<FeatureStore> store;
  std::unique_ptr<OnlineTracker> tracker;
  std::unique_ptr<FlowTracker> flow_tracker;
  std::unique_ptr<VisualOdometry> odometry;
  std::unique_ptr<DepthEstimator> depth;
  // Tracked frames waiting for their depth map (the VO runs one frame behind).
  struct PendingFrame {
    TrackedFrame tracked;
    std::vector<std::array<std::uint8_t, 3>> grid_colors;  // at keyframe_depth.grid()
    bool depth_submitted{};
  };
  std::deque<PendingFrame> pending;
  std::map<std::uint64_t, DepthMap> depth_maps;  // recent finished maps by frame index
  KeyframeDepthStore keyframe_depth;
  DepthFilter depth_filter;                  // depth fusion (Runtime::fusion)
  DepthConfidenceModel fusion_sigma_model;   // the raw network's error, for the filter
  double fusion_log_scale{std::numeric_limits<double>::quiet_NaN()};
  bool fusion_has_grid{};
  ScaleGrid fusion_grid{};
  int fusion_segment{-1};
  TrajectoryView trajectory_view;
  ColorSampler color_sampler;
  BackgroundFocalEstimator focal;
  FocalEstimate focal_estimate;  // latest copy from the worker
  std::uint64_t focal_version{};
  FovView fov_view;
  DescriptorProjection projection;
  Runtime runtime;
};

}  // namespace

int run_app(const AppConfig& config) {
  GlfwLifetime glfw;
  Window window;
  validate_cuda_gl_display();
  ImGuiLifetime imgui(window.get());
  std::error_code path_error;
  const std::filesystem::path executable =
      std::filesystem::read_symlink("/proc/self/exe", path_error);
  const auto native_root = path_error ? std::filesystem::current_path() :
      executable.parent_path().parent_path().parent_path();
  AppConfig initial = config;
  if (initial.engine.empty()) {
    initial.engine = native_root / "models/superpoint-1024x576-k2048.engine";
  }
  if (initial.models_dir.empty()) initial.models_dir = native_root / "models";
  if (initial.start_immediately && initial.database.empty()) {
    initial.video = std::filesystem::absolute(initial.video);
    initial.engine = std::filesystem::absolute(initial.engine);
    initial.database = default_point_cache(native_root.parent_path(), initial);
  }
  Launcher launcher(initial, native_root.parent_path());
  DatabaseSummary database_summary;
  std::unique_ptr<Session> session;
  if (config.start_immediately) {
    try {
      session = std::make_unique<Session>(initial);
      launcher.record_recent(session->config);
    } catch (const std::exception& error) {
      launcher.set_error(error.what());
    }
  }

  while (!glfwWindowShouldClose(window.get())) {
    glfwPollEvents();
    if (session) session->advance();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (session) {
      DatabaseStats stats = database_summary.get(session->config.database, false);
      stats.frame_rows = session->store->persisted();
      stats.landmark_rows = session->store->landmark_rows();
      stats.observations = session->store->observation_count();
      draw_sidebar(session->runtime, *session->decoder, *session->store,
                   session->tracker.get(), session->flow_tracker.get(), stats,
                   session->config.database);
      if (session->runtime.stop_requested) {
        session->store->flush();
        session->runtime.stop_requested = false;
      }
      if (session->runtime.export_requested) {
        session->export_results(native_root.parent_path() / "out");
        session->runtime.export_requested = false;
      }
      draw_video(session->runtime, session->optical_flow != nullptr,
                 session->decoder->duration_ns(),
                 session->decoder->start_time_ns());
      if (session->flow_tracker) draw_flow_diagnostics(session->runtime, *session->flow_tracker);
      draw_depth(session->runtime);
      draw_fusion(session->runtime);
      if (session->odometry) {
        session->trajectory_view.set_display_rotation(session->runtime.display_rotation);
        session->trajectory_view.draw(*session->odometry, &session->runtime.show_trajectory,
                                      session->runtime.frame_width, session->runtime.frame_height);
        const auto& pose_config = session->config.odometry;
        session->focal.latest(session->focal_estimate, session->focal_version);
        bool reset_estimate = false;
        const LensChoice applied{pose_config.intrinsics ? 0.0 : pose_config.horizontal_fov_degrees,
                                 pose_config.distortion_k1};
        const auto lens = session->fov_view.draw(session->focal_estimate, applied, &session->runtime.show_fov,
                                                 reset_estimate);
        if (lens) session->apply_lens(*lens);
        if (reset_estimate) session->focal.reset();
      }
      else draw_diagnostics(session->runtime, *session->tracker, session->projection);
    } else {
      AppConfig selected;
      if (launcher.draw(selected, database_summary)) {
        try {
          session = std::make_unique<Session>(std::move(selected));
          launcher.record_recent(session->config);
        } catch (const std::exception& error) {
          launcher.set_error(error.what());
        }
      }
    }
    ImGui::Render();
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window.get(), &width, &height);
    glViewport(0, 0, width, height);
    glClearColor(0.025F, 0.03F, 0.04F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window.get());
    if (session && session->runtime.close_requested) {
      session->store->flush();
      session.reset();
    }
  }
  if (session) session->store->flush();
  return 0;
}

}  // namespace slam_native
