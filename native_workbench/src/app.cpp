#define GL_GLEXT_PROTOTYPES
#include "slam_native/app.hpp"

#include "slam_native/calibration.hpp"
#include "slam_native/cuda_frame_presenter.hpp"
#include "slam_native/depth_overlay.hpp"
#include "slam_native/display_environment.hpp"
#include "slam_native/flow_tracker.hpp"
#include "slam_native/launcher.hpp"
#include "slam_native/live_depth.hpp"
#include "slam_native/optical_flow.hpp"
#include "slam_native/color_sampler.hpp"
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
  int point_display_mode{2}; // 0: tracked points, 1: all raw SuperPoint detections, 2: pose inliers (default), 3: landmark age
  float age_span{300.0F};   // landmark age view: frames from new to the ramp's end
  float flow_sigma{};       // tracker position fusion; <= 0 snaps to detections
  int display_rotation{};   // clockwise degrees; display only (processing uses native frames)
  int metadata_rotation{};
  float detection_sigma{1.0F};
  float association_radius{6.0F};
  bool show_trails{true};
  bool show_flow_vectors{};
  bool show_diagnostics{true};
  bool show_trajectory{true};
  bool show_fov{true};
  // Live depth image over the video (LIVE_DEPTH.md). Nothing runs while it is off.
  bool show_depth{};
  int depth_mode{3};             // LiveDepthMode: 0 keypoints only, 1 one measurement, 2 filter, 3 fused
  int depth_color{};             // DepthOverlayColor: 0 depth, 1 uncertainty
  float depth_opacity{0.7F};
  float depth_max_sigma{0.3F};   // relative sigma at which the overlay has faded out
  unsigned int depth_texture{};  // 0: no image for the shown frame
  float depth_near{}, depth_far{};  // inverse depths at the ends of the colour ramp (from the landmarks)
  LiveDepthStats depth_stats;
  std::string depth_error;
  OdometryFrameResult odometry;
  std::string intrinsics_source;     // "calibration.json", "--intrinsics" or empty (hfov guess)
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
  // Per point: the first keyframe of its 3D landmark (kNoLandmark: none).
  static constexpr std::uint64_t kNoLandmark = std::numeric_limits<std::uint64_t>::max();
  std::vector<std::uint64_t> landmark_first_frames;
  std::vector<float> corrections;
  bool cache_hit{};
  double flow_ms{};
  std::optional<FlowField> flow_field;
  std::uint64_t selected_landmark{std::numeric_limits<std::uint64_t>::max()};
  struct Trail {
    std::deque<ImVec2> positions;
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

// Same ramp as the trajectory view's map points (PointColor::age): yellow
// (new) -> magenta -> blue (t = 1: the age span or older).
ImU32 age_color(float t) {
  const ImVec4 young(1.0F, 0.90F, 0.30F, 1), mid(0.92F, 0.30F, 0.50F, 1), old(0.25F, 0.50F, 1.0F, 1);
  t = std::clamp(t, 0.0F, 1.0F);
  const auto mix = [](const ImVec4& a, const ImVec4& b, float s) {
    return ImVec4(a.x + s * (b.x - a.x), a.y + s * (b.y - a.y), a.z + s * (b.z - a.z), 1);
  };
  return ImGui::ColorConvertFloat4ToU32(t < 0.5F ? mix(young, mid, 2 * t) : mix(mid, old, 2 * t - 1));
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
                  const FlowTracker& flow_tracker,
                  const DatabaseStats& database_stats,
                  const std::filesystem::path& database_path) {
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
               "Tracked points\0SuperPoint detections (red)\0Pose inliers (green) / outliers (red)\0"
               "Landmark age\0");
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Pose inliers: tracks with a 3D landmark whose reprojection under the\n"
                      "current camera pose is within the threshold (green), beyond it (red).\n"
                      "Landmark age: frames since the 3D landmark's first keyframe (yellow: new,\n"
                      "magenta: half the age span, blue: the age span or older). A landmark\n"
                      "taken over by a new track keeps its age.\n"
                      "Grey: tracked, no landmark yet (hollow: flow-only this frame).");
  if (runtime.point_display_mode == 3) {
    ImGui::SetNextItemWidth(110);
    ImGui::SliderFloat("Age span", &runtime.age_span, 10.0F, 10000.0F, "%.0f frames", ImGuiSliderFlags_Logarithmic);
  }
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
  ImGui::Checkbox("Show flow vectors", &runtime.show_flow_vectors);
  ImGui::EndDisabled();
  ImGui::Checkbox("Diagnostics", &runtime.show_diagnostics);
  ImGui::SameLine();
  ImGui::Checkbox("Trajectory", &runtime.show_trajectory);
  ImGui::SameLine();
  ImGui::Checkbox("Lens", &runtime.show_fov);
  ImGui::Checkbox("Depth", &runtime.show_depth);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("A depth estimate for every pixel, drawn over the video: matched against the\n"
                      "frame ten frames back along the camera motion, fused over time, and combined\n"
                      "with the landmarks' depths. Display only. Starts with the next frame.");
  if (runtime.show_depth) {
    ImGui::SameLine();
    const char* modes[] = {"Keypoints only", "One measurement", "Filter", "Fused"};
    ImGui::SetNextItemWidth(130);
    ImGui::Combo("##depth-mode", &runtime.depth_mode, modes, IM_ARRAYSIZE(modes));
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Keypoints only: the landmarks' depths, interpolated.\n"
                        "One measurement: this frame against the one ten frames back, no memory.\n"
                        "Filter: the measurements of all frames so far.\n"
                        "Fused: the filter combined with the keypoints.");
    ImGui::SetNextItemWidth(70);
    ImGui::SliderFloat("Opacity", &runtime.depth_opacity, 0.0F, 1.0F, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(70);
    ImGui::SliderFloat("Max sigma", &runtime.depth_max_sigma, 0.03F, 1.0F, "%.2f", ImGuiSliderFlags_Logarithmic);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Pixels fade out as their relative depth uncertainty approaches this.");
    const char* colors[] = {"Depth", "Uncertainty"};
    ImGui::SetNextItemWidth(110);
    ImGui::Combo("Colour##depth", &runtime.depth_color, colors, IM_ARRAYSIZE(colors));
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Depth: red near, blue far (between the nearest and farthest landmarks in view).\n"
                        "Uncertainty: blue 1%% or better, red 100%%.");
    if (!runtime.depth_error.empty()) ImGui::TextColored({1, 0.4F, 0.3F, 1}, "Depth: %s", runtime.depth_error.c_str());
    else if (!runtime.depth_texture) ImGui::TextDisabled("Depth: no image (needs a tracked pose)");
    else
      ImGui::TextDisabled("Depth: %.0f%% of pixels, %.0f%% measured now, %.1f ms",
                          100.0 * double(runtime.depth_stats.estimated) / double(std::max<std::size_t>(1, runtime.depth_stats.pixels)),
                          100.0 * double(runtime.depth_stats.measured) / double(std::max<std::size_t>(1, runtime.depth_stats.pixels)),
                          runtime.depth_stats.ms);
  }
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
  ImGui::Text("Active / total tracks: %zu / %llu", flow_tracker.active_count(),
              static_cast<unsigned long long>(flow_tracker.landmark_count()));
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
    if (const LandmarkState* selected = flow_tracker.find(runtime.selected_landmark)) {
      ImGui::Text("Observations: %llu",
                  static_cast<unsigned long long>(selected->observation_count));

      ImGui::Text("First / last frame: %llu / %llu",
                  static_cast<unsigned long long>(selected->first_frame),
                  static_cast<unsigned long long>(selected->last_frame));
    } else {
      ImGui::TextDisabled("Inactive landmark");
    }
  }
  ImGui::End();
}

void draw_video(Runtime& runtime, bool optical_flow, std::int64_t duration_ns,
                std::int64_t start_time_ns, const LiveDepth* depth = nullptr) {
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
    if (runtime.show_depth && runtime.depth_texture) {
      // The depth image is in source orientation, like the video texture.
      ImGui::GetWindowDrawList()->AddImageQuad(static_cast<ImTextureID>(runtime.depth_texture), origin,
          {origin.x + size.x, origin.y}, {origin.x + size.x, origin.y + size.y}, {origin.x, origin.y + size.y},
          corners[first], corners[(first + 1) % 4], corners[(first + 2) % 4], corners[(first + 3) % 4],
          IM_COL32(255, 255, 255, static_cast<int>(255.0F * runtime.depth_opacity)));
      if (depth && ImGui::IsItemHovered()) {
        // Screen -> source pixel: the inverse of screen() above.
        const ImVec2 mouse = ImGui::GetMousePos();
        const float u = (mouse.x - origin.x) / scale, v = (mouse.y - origin.y) / scale;
        float x = u, y = v;
        if (rotation == 90) { x = v; y = height - u; }
        else if (rotation == 180) { x = width - u; y = height - v; }
        else if (rotation == 270) { x = width - v; y = u; }
        float inverse_depth = 0, variance = 0;
        if (depth->at(static_cast<int>(x), static_cast<int>(y), &inverse_depth, &variance) && inverse_depth > 0)
          ImGui::SetTooltip("depth %.3f  (+/- %.0f%%)\nin map units: 1 = the first keyframe's median scene depth",
                            1.0 / inverse_depth, 100.0 * std::sqrt(variance) / inverse_depth);
      }
    }
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
  } else if (runtime.show_features && runtime.point_display_mode == 3) {
    for (std::size_t index = 0; index < runtime.points.size(); ++index) {
      const auto& point = runtime.points[index];
      const ImVec2 center = screen(point.x, point.y);
      const std::uint64_t first = index < runtime.landmark_first_frames.size() ?
          runtime.landmark_first_frames[index] : Runtime::kNoLandmark;
      const bool supported = index >= runtime.supported.size() || runtime.supported[index];
      if (first != Runtime::kNoLandmark) {
        const float age = runtime.frame_index > first ? static_cast<float>(runtime.frame_index - first) : 0.0F;
        draw->AddCircleFilled(center, 3.0F, age_color(age / std::max(runtime.age_span, 1.0F)));
      } else if (supported) {
        draw->AddCircleFilled(center, 1.8F, IM_COL32(170, 170, 170, 170));
      } else {
        draw->AddCircle(center, 2.5F, IM_COL32(170, 170, 170, 170), 0, 1.0F);
      }
      if (runtime.landmark_ids[index] == runtime.selected_landmark)
        draw->AddCircle(center, 7.0F, IM_COL32_WHITE, 0, 2.0F);
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
    next_cache_index = static_cast<std::uint64_t>(store->last_frame_index() + 1);
    runtime.association_radius = config.flow_tracker.association_radius;
    runtime.metadata_rotation = decoder->display_rotation();
    runtime.display_rotation = config.display_rotation.value_or(runtime.metadata_rotation);
    runtime.flow_sigma = config.flow_tracker.flow_sigma_px;
    runtime.detection_sigma = config.flow_tracker.detection_sigma_px;
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
    if (live_depth) live_depth->reset();
    focal.restart_tracks();
    runtime.odometry = {};
    runtime.trails.clear();
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
    runtime.odometry = odometry->process(tracked);
    trajectory_view.update(*odometry);
    update_depth(frame, field ? &*field : nullptr, tracked);
    focal.add(std::move(tracked));  // background thread
    runtime.points = features.keypoints;
    runtime.supported = features.superpoint_supported;
    runtime.landmark_ids = features.landmark_ids;
    {
      std::unordered_map<std::uint64_t, std::uint64_t> first_frames;  // by track
      for (const auto& t : odometry->tracked_landmarks()) first_frames.emplace(t.track_id, t.first_frame);
      runtime.landmark_first_frames.assign(runtime.landmark_ids.size(), Runtime::kNoLandmark);
      for (std::size_t i = 0; i < runtime.landmark_ids.size(); ++i)
        if (auto it = first_frames.find(runtime.landmark_ids[i]); it != first_frames.end())
          runtime.landmark_first_frames[i] = it->second;
    }
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

  // The live depth image of this frame, while its overlay is on. It only reads
  // from the odometry.
  void update_depth(const GpuFrame& frame, const DeviceFlowField* flow, const TrackedFrame& tracked) {
    runtime.depth_texture = 0;
    if (!runtime.show_depth) {
      live_depth.reset();  // and its GPU memory
      return;
    }
    try {
      const auto mode = static_cast<LiveDepthMode>(runtime.depth_mode);
      if (!live_depth || live_depth->config().mode != mode) {
        LiveDepthConfig depth_config;
        depth_config.mode = mode;
        live_depth = std::make_unique<LiveDepth>(depth_config);
      }
      const auto& result = runtime.odometry;
      // A loop correction or a merge moved the whole segment, scale included.
      if (result.loop_closed || result.segments_merged) live_depth->reset();
      const bool tracked_pose = result.has_pose && !result.predicted;  // a coasted pose is a prediction
      const auto anchors = depth_anchors(tracked, *odometry, result);
      live_depth->process(frame, flow, live_depth_camera(config.odometry, frame.width, frame.height),
                          tracked_pose ? &result.pose : nullptr, result.segment, anchors);
      runtime.depth_stats = live_depth->stats();
      if (live_depth->valid() && anchors.size() >= 8) {
        // Colour ramp between the nearest and farthest landmarks in view, eased so it does not flicker.
        std::vector<float> depths;
        for (const auto& a : anchors) depths.push_back(a.inverse_depth);
        std::sort(depths.begin(), depths.end());
        const float far = depths[depths.size() / 20], near = depths[depths.size() - 1 - depths.size() / 20];
        const float ease = runtime.depth_near > 0 ? 0.1F : 1.0F;
        runtime.depth_near += ease * (near - runtime.depth_near);
        runtime.depth_far += ease * (far - runtime.depth_far);
      }
      present_depth(true);
      runtime.depth_error.clear();
    } catch (const std::exception& error) {
      runtime.depth_error = error.what();
      runtime.show_depth = false;
      live_depth.reset();
    }
  }

  // Colours the current depth image; without `always`, only when the display settings changed (paused).
  void present_depth(bool always) {
    if (!runtime.show_depth || !live_depth || !live_depth->valid() || !(runtime.depth_near > 0)) return;
    DepthOverlayStyle style;
    style.near_inverse_depth = runtime.depth_near;
    style.far_inverse_depth = std::max(runtime.depth_far, 1e-6F);
    style.color = static_cast<DepthOverlayColor>(runtime.depth_color);
    style.max_sigma = std::max(runtime.depth_max_sigma, style.full_sigma + 1e-3F);
    if (!always && style.color == depth_style.color && style.max_sigma == depth_style.max_sigma) return;
    depth_style = style;
    try {
      runtime.depth_texture = depth_overlay.present(*live_depth, style);
    } catch (const std::exception& error) {
      runtime.depth_error = error.what();
      runtime.depth_texture = 0;
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
    runtime.odometry = {};
    if (live_depth) live_depth->reset();  // segment ids restart with the odometry
    runtime.depth_texture = 0;
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
  std::unique_ptr<FlowTracker> flow_tracker;
  std::unique_ptr<VisualOdometry> odometry;
  TrajectoryView trajectory_view;
  std::unique_ptr<LiveDepth> live_depth;  // while the depth overlay is on
  DepthOverlay depth_overlay;
  DepthOverlayStyle depth_style;          // of the texture shown
  ColorSampler color_sampler;
  BackgroundFocalEstimator focal;
  FocalEstimate focal_estimate;  // latest copy from the worker
  std::uint64_t focal_version{};
  FovView fov_view;
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
      draw_sidebar(session->runtime, *session->decoder, *session->store, *session->flow_tracker, stats,
                   session->config.database);
      if (session->runtime.stop_requested) {
        session->store->flush();
        session->runtime.stop_requested = false;
      }
      if (session->runtime.export_requested) {
        session->export_results(native_root.parent_path() / "out");
        session->runtime.export_requested = false;
      }
      session->present_depth(false);
      draw_video(session->runtime, session->optical_flow != nullptr,
                 session->decoder->duration_ns(),
                 session->decoder->start_time_ns(), session->live_depth.get());
      if (session->flow_tracker) draw_flow_diagnostics(session->runtime, *session->flow_tracker);
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
