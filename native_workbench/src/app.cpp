#define GL_GLEXT_PROTOTYPES
#include "slam_native/app.hpp"

#include "slam_native/cuda_frame_presenter.hpp"
#include "slam_native/descriptor_projection.hpp"
#include "slam_native/flow_tracker.hpp"
#include "slam_native/launcher.hpp"
#include "slam_native/optical_flow.hpp"
#include "slam_native/video_decoder.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
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
  bool show_features{true};
  bool show_trails{true};
  bool show_flow_vectors{};
  bool show_diagnostics{true};
  bool preview_mode{};
  bool preview_has_cache{};
  bool scrub_active{};
  bool seek_requested{};
  double scrub_seconds{};
  double seek_seconds{};
  Clock::time_point last_seek_request{};
  std::string seek_error;
  std::int64_t processing_timestamp_ns{};
  std::uint64_t processing_frame_index{};
  unsigned int texture{};
  int frame_width{};
  int frame_height{};
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  std::vector<Keypoint> points;
  std::vector<std::uint64_t> landmark_ids;
  std::vector<float> similarities;
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
  double tracking_ms{};
  PipelineStats stats;
  Clock::time_point started{Clock::now()};
  Clock::time_point next_frame{Clock::now()};
};

struct CachedOverlay {
  std::uint64_t frame_index{};
  std::vector<Keypoint> points;
  std::vector<std::uint64_t> landmark_ids;
  std::vector<float> similarities;
};

class FramePreviewReader {
 public:
  explicit FramePreviewReader(const std::filesystem::path& path) {
    if (sqlite3_open_v2(path.c_str(), &db_, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
      const std::string error = db_ ? sqlite3_errmsg(db_) : "no SQLite connection";
      if (db_) sqlite3_close(db_);
      db_ = nullptr;
      throw std::runtime_error("Open preview database: " + error);
    }
    sqlite3_busy_timeout(db_, 250);
    constexpr const char* sql =
        "SELECT frame_index,keypoint_count,keypoints_xy_f32,scores_f32,"
        "landmark_ids_u64,landmark_similarities_f32 FROM frames "
        "WHERE timestamp_ns=? AND pts=? LIMIT 1";
    if (sqlite3_prepare_v2(db_, sql, -1, &statement_, nullptr) != SQLITE_OK) {
      const std::string error = sqlite3_errmsg(db_);
      sqlite3_close(db_);
      db_ = nullptr;
      throw std::runtime_error("Prepare preview lookup: " + error);
    }
  }
  ~FramePreviewReader() {
    if (statement_) sqlite3_finalize(statement_);
    if (db_) sqlite3_close(db_);
  }
  FramePreviewReader(const FramePreviewReader&) = delete;
  FramePreviewReader& operator=(const FramePreviewReader&) = delete;

  [[nodiscard]] std::optional<CachedOverlay> find(const GpuFrame& frame) {
    sqlite3_reset(statement_);
    sqlite3_clear_bindings(statement_);
    sqlite3_bind_int64(statement_, 1, frame.timestamp_ns);
    sqlite3_bind_int64(statement_, 2, frame.pts);
    const int result = sqlite3_step(statement_);
    if (result == SQLITE_DONE) return std::nullopt;
    if (result != SQLITE_ROW) {
      throw std::runtime_error("Read preview features: " + std::string(sqlite3_errmsg(db_)));
    }
    const int count = sqlite3_column_int(statement_, 1);
    if (count < 0 || sqlite3_column_bytes(statement_, 2) != count * 2 * static_cast<int>(sizeof(float)) ||
        sqlite3_column_bytes(statement_, 3) != count * static_cast<int>(sizeof(float)) ||
        sqlite3_column_bytes(statement_, 4) != count * static_cast<int>(sizeof(std::uint64_t)) ||
        sqlite3_column_bytes(statement_, 5) != count * static_cast<int>(sizeof(float))) {
      throw std::runtime_error("Invalid cached preview feature blobs");
    }
    CachedOverlay overlay;
    overlay.frame_index = static_cast<std::uint64_t>(sqlite3_column_int64(statement_, 0));
    overlay.points.resize(count);
    overlay.landmark_ids.resize(count);
    overlay.similarities.resize(count);
    const auto* xy = static_cast<const float*>(sqlite3_column_blob(statement_, 2));
    const auto* scores = static_cast<const float*>(sqlite3_column_blob(statement_, 3));
    for (int index = 0; index < count; ++index) {
      overlay.points[index] = {xy[index * 2], xy[index * 2 + 1], scores[index]};
    }
    if (count) {
      std::memcpy(overlay.landmark_ids.data(), sqlite3_column_blob(statement_, 4),
                  count * sizeof(std::uint64_t));
      std::memcpy(overlay.similarities.data(), sqlite3_column_blob(statement_, 5),
                  count * sizeof(float));
    }
    return overlay;
  }

 private:
  sqlite3* db_{};
  sqlite3_stmt* statement_{};
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

void update_trails(Runtime& runtime, std::uint32_t max_inactive_frames) {
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
    if (runtime.frame_index - it->second.last_frame > max_inactive_frames) {
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
  const char* run_label = runtime.playing ? "Pause" :
                          runtime.preview_mode ? "Resume processing" : "Run";
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
  if (ImGui::Button("Choose another video")) runtime.close_requested = true;
  ImGui::Checkbox("Realtime pacing", &runtime.realtime_pacing);
  ImGui::Checkbox(optical_flow ? "Show flow tracks" : "Show SuperPoints",
                  &runtime.show_features);
  ImGui::Checkbox("Show tracks", &runtime.show_trails);
  if (optical_flow) ImGui::Checkbox("Show flow vectors", &runtime.show_flow_vectors);
  ImGui::Checkbox("Diagnostics", &runtime.show_diagnostics);
  ImGui::Separator();
  ImGui::Text("Codec: %s", decoder.codec_name().c_str());
  ImGui::Text("Nominal FPS: %.2f", decoder.nominal_fps());
  if (runtime.preview_mode && !runtime.preview_has_cache) {
    ImGui::Text("Frame: uncached preview");
  } else {
    ImGui::Text("Frame: %llu%s", static_cast<unsigned long long>(runtime.frame_index),
                runtime.preview_mode ? " (preview)" : "");
  }
  ImGui::Text("Time: %.3f s", std::max(0.0,
              static_cast<double>(runtime.timestamp_ns - decoder.start_time_ns()) / 1e9));
  if (runtime.preview_mode) {
    ImGui::TextWrapped("Preview paused. Run continues processing from frame %llu (%.2f s).",
                      static_cast<unsigned long long>(runtime.processing_frame_index),
                      static_cast<double>(runtime.processing_timestamp_ns -
                                          decoder.start_time_ns()) / 1e9);
  }
  if (!runtime.seek_error.empty()) {
    ImGui::TextColored({1, 0.5F, 0.4F, 1}, "%s", runtime.seek_error.c_str());
  }
  ImGui::Text("Features: %zu", runtime.points.size());
  ImGui::Text("Matched / new: %u / %u", runtime.matched_landmarks,
              runtime.new_landmarks);
  ImGui::Text("Method: %s", optical_flow ? "NVIDIA optical flow" : "SuperPoint descriptors");
  ImGui::Text("Active / total tracks: %zu / %llu",
              optical_flow ? flow_tracker->active_count() : tracker->active_count(),
              static_cast<unsigned long long>(optical_flow ? flow_tracker->landmark_count() :
                                                               tracker->landmark_count()));
  ImGui::Separator();
  ImGui::Text("Decode: %.2f ms", runtime.stats.decode_ms);
  ImGui::Text("Preprocess: %.2f ms", runtime.stats.preprocess_ms);
  ImGui::Text("%s: %.2f ms", optical_flow ? "Optical flow" : "SuperPoint",
              runtime.stats.inference_ms);
  if (!optical_flow) ImGui::Text("Readback: %.2f ms", runtime.stats.readback_ms);
  ImGui::Text("Tracking: %.2f ms", runtime.tracking_ms);
  const double elapsed = std::chrono::duration<double>(Clock::now() - runtime.started).count();
  ImGui::Text("Throughput: %.1f frames/s", elapsed > 0 ? runtime.stats.inferred_frames / elapsed : 0);
  ImGui::Separator();
  ImGui::Text("DB queue: %zu", store.queued());
  ImGui::Text("DB committed: %llu", static_cast<unsigned long long>(store.persisted()));
  ImGui::Text("DB size: %s",
              format_bytes(database_stats.main_bytes + database_stats.wal_bytes).c_str());
  ImGui::Text("Rows: %llu frames / %llu landmarks",
              static_cast<unsigned long long>(database_stats.frame_rows),
              static_cast<unsigned long long>(database_stats.landmark_rows));
  ImGui::Text("Saved observations: %llu",
              static_cast<unsigned long long>(database_stats.observations));
  ImGui::TextWrapped("%s", database_path.string().c_str());
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
      if (!optical_flow) ImGui::Text("Concentration: %.3f", selected->concentration);
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
  const float scale = std::min(available.x / runtime.frame_width,
                               std::max(1.0F, available.y - 85.0F) / runtime.frame_height);
  const ImVec2 size(runtime.frame_width * scale, runtime.frame_height * scale);
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImGui::Image(static_cast<ImTextureID>(runtime.texture), size, {0, 0}, {1, 1});
  if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
    const ImVec2 mouse = ImGui::GetMousePos();
    float nearest = 100.0F;
    for (std::size_t index = 0; index < runtime.points.size(); ++index) {
      const float dx = mouse.x - (origin.x + runtime.points[index].x * scale);
      const float dy = mouse.y - (origin.y + runtime.points[index].y * scale);
      const float squared = dx * dx + dy * dy;
      if (squared < nearest) {
        nearest = squared;
        runtime.selected_landmark = runtime.landmark_ids[index];
      }
    }
  }
  ImDrawList* draw = ImGui::GetWindowDrawList();
  if (runtime.show_trails) {
    for (const auto& [id, trail] : runtime.trails) {
      if (trail.positions.size() < 2) continue;
      if (runtime.selected_landmark != std::numeric_limits<std::uint64_t>::max() &&
          id != runtime.selected_landmark) continue;
      for (std::size_t index = 1; index < trail.positions.size(); ++index) {
        const auto& previous = trail.positions[index - 1];
        const auto& current = trail.positions[index];
        draw->AddLine({origin.x + previous.x * scale, origin.y + previous.y * scale},
                      {origin.x + current.x * scale, origin.y + current.y * scale},
                      landmark_color(id, 130), 1.5F);
      }
    }
  }
  if (runtime.show_features) {
    for (std::size_t index = 0; index < runtime.points.size(); ++index) {
      const auto& point = runtime.points[index];
      const ImVec2 center(origin.x + point.x * scale, origin.y + point.y * scale);
      const std::uint64_t id = runtime.landmark_ids[index];
      const bool selected = id == runtime.selected_landmark;
      draw->AddCircleFilled(center, selected ? 5.0F : 2.3F, landmark_color(id));
      if (selected) draw->AddCircle(center, 7.0F, IM_COL32_WHITE, 0, 2.0F);
    }
  }
  if (optical_flow && runtime.show_flow_vectors && runtime.flow_field) {
    constexpr int arrow_spacing = 48;
    for (int y = arrow_spacing / 2; y < runtime.frame_height; y += arrow_spacing) {
      for (int x = arrow_spacing / 2; x < runtime.frame_width; x += arrow_spacing) {
        const FlowVector vector = runtime.flow_field->sample(static_cast<float>(x),
                                                              static_cast<float>(y));
        if (!std::isfinite(vector.dx) || !std::isfinite(vector.dy) ||
            vector.dx * vector.dx + vector.dy * vector.dy < 0.25F) continue;
        const ImVec2 from(origin.x + x * scale, origin.y + y * scale);
        const ImVec2 to(origin.x + (x + vector.dx) * scale,
                        origin.y + (y + vector.dy) * scale);
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
    const ImVec2 slider_min = ImGui::GetItemRectMin();
    const ImVec2 slider_max = ImGui::GetItemRectMax();
    if (runtime.preview_mode) {
      const double head_seconds = static_cast<double>(runtime.processing_timestamp_ns -
                                                      start_time_ns) / 1e9;
      const float fraction = static_cast<float>(std::clamp(head_seconds / duration, 0.0, 1.0));
      const float marker_x = slider_min.x + (slider_max.x - slider_min.x) * fraction;
      ImGui::GetWindowDrawList()->AddLine({marker_x, slider_min.y - 2.0F},
                                           {marker_x, slider_max.y + 2.0F},
                                           IM_COL32(74, 214, 231, 255), 2.0F);
    }
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
    ImGui::TextDisabled("Drag to preview");
    if (runtime.preview_mode) ImGui::TextDisabled("Cyan marker = processing position");
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
  ImGui::TextDisabled("Grid seeds advected by NVIDIA optical flow.");
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
    const bool use_flow = config.tracking_method == TrackingMethod::optical_flow;
    if (use_flow) {
      config.engine.clear();
      config.store.descriptor_encoding = DescriptorEncoding::none;
      optical_flow = std::make_unique<OpticalFlow>();
    } else {
      superpoint = make_tensorrt_superpoint(config.engine, config.superpoint);
    }
    store = std::make_unique<FeatureStore>(config.database, config.store);
    store->set_session(config.video, config.engine, config.superpoint.input_width,
                       config.superpoint.input_height, config.superpoint.max_keypoints,
                       config.superpoint.detection_threshold, config.tracker, use_flow);
    if (use_flow) {
      flow_tracker = std::make_unique<FlowTracker>(
          store->load_active_landmarks(1), store->load_latest_positions(),
          store->next_landmark_id(), store->load_length_histogram());
    } else {
      tracker = std::make_unique<OnlineTracker>(
          config.tracker, store->load_active_landmarks(config.tracker.max_inactive_frames),
          store->next_landmark_id(), store->load_length_histogram());
    }
    cached_through = store->last_frame_index();
  }

  void preview_seek(double seconds) {
    if (!preview_decoder) {
      preview_decoder = make_ffmpeg_cuda_decoder();
      preview_decoder->open(config.video);
      preview_reader = std::make_unique<FramePreviewReader>(config.database);
    }
    const auto duration = decoder->duration_ns();
    const auto frame_interval = decoder->nominal_fps() > 0 ?
        static_cast<std::int64_t>(1e9 / decoder->nominal_fps()) : 33333333LL;
    const auto relative = static_cast<std::int64_t>(std::max(0.0, seconds) * 1e9);
    const auto clamped = duration > 0 ?
        std::min(relative, std::max<std::int64_t>(0, duration - frame_interval)) : relative;
    const auto target = decoder->start_time_ns() + clamped;
    preview_decoder->seek_ns(target);
    GpuFrame frame;
    bool found = false;
    bool had_previous = false;
    std::int64_t previous_timestamp{};
    while (preview_decoder->next(frame)) {
      if (frame.timestamp_ns >= target) {
        found = true;
        break;
      }
      had_previous = true;
      previous_timestamp = frame.timestamp_ns;
    }
    if (!found && had_previous) {
      preview_decoder->seek_ns(previous_timestamp);
      while (preview_decoder->next(frame)) {
        if (frame.timestamp_ns >= previous_timestamp) {
          found = true;
          break;
        }
      }
    }
    if (!found) throw std::runtime_error("No frame found near the selected time");
    runtime.texture = presenter->present(frame);
    runtime.frame_width = frame.width;
    runtime.frame_height = frame.height;
    runtime.timestamp_ns = frame.timestamp_ns;
    runtime.frame_index = 0;
    runtime.preview_mode = true;
    runtime.preview_has_cache = false;
    runtime.playing = false;
    runtime.points.clear();
    runtime.landmark_ids.clear();
    runtime.similarities.clear();
    runtime.projected.clear();
    runtime.flow_field.reset();
    runtime.trails.clear();
    runtime.selected_landmark = std::numeric_limits<std::uint64_t>::max();
    runtime.new_landmarks = 0;
    runtime.matched_landmarks = 0;
    runtime.tracking_ms = 0;
    if (const auto cached = preview_reader->find(frame)) {
      runtime.frame_index = cached->frame_index;
      runtime.points = cached->points;
      runtime.landmark_ids = cached->landmark_ids;
      runtime.similarities = cached->similarities;
      runtime.preview_has_cache = true;
    }
    runtime.seek_error.clear();
  }

  void advance() {
    if (runtime.seek_requested) {
      runtime.seek_requested = false;
      try {
        preview_seek(runtime.seek_seconds);
      } catch (const std::exception& error) {
        runtime.seek_error = error.what();
      }
      return;
    }
    if (runtime.preview_mode && (runtime.playing || runtime.step_requested) &&
        !runtime.eof && !runtime.stopped) {
      runtime.preview_mode = false;
      runtime.preview_has_cache = false;
    }
    const auto now = Clock::now();
    const bool replaying_cached = cached_through >= 0 &&
        static_cast<std::int64_t>(runtime.processing_frame_index) <= cached_through;
    const bool pacing_ready = replaying_cached || !runtime.realtime_pacing ||
                              now >= runtime.next_frame;
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
      runtime.texture = presenter->present(frame);
      runtime.frame_width = frame.width;
      runtime.frame_height = frame.height;
      runtime.frame_index = frame.frame_index;
      runtime.timestamp_ns = frame.timestamp_ns;
      runtime.seek_error.clear();
      runtime.processing_frame_index = frame.frame_index;
      runtime.processing_timestamp_ns = frame.timestamp_ns;
      runtime.stats.decoded_frames++;
      runtime.stats.decode_ms = decode_ms;
      if (static_cast<std::int64_t>(frame.frame_index) > cached_through) {
        FrameFeatures features;
        if (optical_flow) {
          features.frame_index = frame.frame_index;
          features.timestamp_ns = frame.timestamp_ns;
          features.pts = frame.pts;
          features.width = frame.width;
          features.height = frame.height;
          std::optional<FlowField> field;
          const auto flow_start = Clock::now();
          if (optical_flow->has_reference()) field = optical_flow->compute(frame);
          else optical_flow->remember(frame);
          features.inference_ms = std::chrono::duration<double, std::milli>(
              Clock::now() - flow_start).count();
          flow_tracker->associate(features, field);
          runtime.flow_field = std::move(field);
        } else {
          features = superpoint->infer(frame);
          tracker->associate(features);
          projection.observe(features);
          runtime.projected = projection.project(features);
        }
        features.decode_ms = decode_ms;
        runtime.points = features.keypoints;
        runtime.landmark_ids = features.landmark_ids;
        runtime.similarities = features.landmark_similarities;
        runtime.new_landmarks = features.new_landmarks;
        runtime.matched_landmarks = features.matched_landmarks;
        runtime.tracking_ms = features.tracking_ms;
        update_trails(runtime, config.tracker.max_inactive_frames);
        runtime.stats.inferred_frames++;
        runtime.stats.preprocess_ms = features.preprocess_ms;
        runtime.stats.inference_ms = features.inference_ms;
        runtime.stats.readback_ms = features.readback_ms;
        store->enqueue(std::move(features));
      } else {
        if (optical_flow && static_cast<std::int64_t>(frame.frame_index) == cached_through) {
          optical_flow->remember(frame);
        }
        runtime.points.clear();
        runtime.landmark_ids.clear();
        runtime.projected.clear();
        runtime.flow_field.reset();
      }
      const double fps = decoder->nominal_fps();
      runtime.next_frame = now + std::chrono::duration_cast<Clock::duration>(
                                     std::chrono::duration<double>(fps > 0 ? 1.0 / fps : 0.0));
    }
    runtime.step_requested = false;
  }

  AppConfig config;
  std::unique_ptr<VideoDecoder> decoder;
  std::unique_ptr<VideoDecoder> preview_decoder;
  std::unique_ptr<FramePreviewReader> preview_reader;
  std::unique_ptr<CudaFramePresenter> presenter;
  std::unique_ptr<SuperPoint> superpoint;
  std::unique_ptr<OpticalFlow> optical_flow;
  std::unique_ptr<FeatureStore> store;
  std::unique_ptr<OnlineTracker> tracker;
  std::unique_ptr<FlowTracker> flow_tracker;
  DescriptorProjection projection;
  Runtime runtime;
  std::int64_t cached_through{-1};
};

}  // namespace

int run_app(const AppConfig& config) {
  GlfwLifetime glfw;
  Window window;
  ImGuiLifetime imgui(window.get());
  std::error_code path_error;
  const std::filesystem::path executable =
      std::filesystem::read_symlink("/proc/self/exe", path_error);
  const auto native_root = path_error ? std::filesystem::current_path() :
      executable.parent_path().parent_path().parent_path();
  AppConfig initial = config;
  if (initial.tracking_method == TrackingMethod::superpoint && initial.engine.empty()) {
    initial.engine = native_root / "models/superpoint-1024x576-k2048.engine";
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
      draw_video(session->runtime, session->optical_flow != nullptr,
                 session->decoder->duration_ns(),
                 session->decoder->start_time_ns());
      if (session->flow_tracker) draw_flow_diagnostics(session->runtime, *session->flow_tracker);
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
