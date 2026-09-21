#define GL_GLEXT_PROTOTYPES
#include "slam_native/app.hpp"

#include "slam_native/cuda_frame_presenter.hpp"
#include "slam_native/video_decoder.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
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
  bool show_features{true};
  unsigned int texture{};
  int frame_width{};
  int frame_height{};
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  std::vector<Keypoint> points;
  PipelineStats stats;
  Clock::time_point started{Clock::now()};
  Clock::time_point next_frame{Clock::now()};
};

void draw_sidebar(Runtime& runtime,
                  const VideoDecoder& decoder,
                  const FeatureStore& store) {
  ImGui::SetNextWindowPos({8, 8}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({310, 675}, ImGuiCond_FirstUseEver);
  ImGui::Begin("Pipeline");
  ImGui::BeginDisabled(runtime.stopped);
  if (ImGui::Button(runtime.playing ? "Pause" : "Run")) runtime.playing = !runtime.playing;
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
  ImGui::Checkbox("Realtime pacing", &runtime.realtime_pacing);
  ImGui::Checkbox("Show SuperPoints", &runtime.show_features);
  ImGui::Separator();
  ImGui::Text("Codec: %s", decoder.codec_name().c_str());
  ImGui::Text("Nominal FPS: %.2f", decoder.nominal_fps());
  ImGui::Text("Frame: %llu", static_cast<unsigned long long>(runtime.frame_index));
  ImGui::Text("Time: %.3f s", static_cast<double>(runtime.timestamp_ns) / 1e9);
  ImGui::Text("Features: %zu", runtime.points.size());
  ImGui::Separator();
  ImGui::Text("Decode: %.2f ms", runtime.stats.decode_ms);
  ImGui::Text("Preprocess: %.2f ms", runtime.stats.preprocess_ms);
  ImGui::Text("SuperPoint: %.2f ms", runtime.stats.inference_ms);
  ImGui::Text("Readback: %.2f ms", runtime.stats.readback_ms);
  const double elapsed = std::chrono::duration<double>(Clock::now() - runtime.started).count();
  ImGui::Text("Throughput: %.1f frames/s", elapsed > 0 ? runtime.stats.inferred_frames / elapsed : 0);
  ImGui::Separator();
  ImGui::Text("DB queue: %zu", store.queued());
  ImGui::Text("DB committed: %llu", static_cast<unsigned long long>(store.persisted()));
  if (runtime.stopped) ImGui::TextColored({1.0F, 0.65F, 0.4F, 1.0F}, "Stopped and flushed");
  if (runtime.eof) ImGui::TextColored({0.45F, 0.8F, 1.0F, 1.0F}, "End of video");
  ImGui::Separator();
  ImGui::TextDisabled("Online landmarks attach between inference and storage.");
  ImGui::End();
}

void draw_video(const Runtime& runtime) {
  ImGui::SetNextWindowPos({326, 8}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({1095, 845}, ImGuiCond_FirstUseEver);
  ImGui::Begin("Video + SuperPoint");
  if (!runtime.texture || runtime.frame_width == 0 || runtime.frame_height == 0) {
    ImGui::TextDisabled("Waiting for the first decoded frame...");
    ImGui::End();
    return;
  }
  const ImVec2 available = ImGui::GetContentRegionAvail();
  const float scale = std::min(available.x / runtime.frame_width,
                               available.y / runtime.frame_height);
  const ImVec2 size(runtime.frame_width * scale, runtime.frame_height * scale);
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImGui::Image(static_cast<ImTextureID>(runtime.texture), size, {0, 0}, {1, 1});
  if (runtime.show_features) {
    ImDrawList* draw = ImGui::GetWindowDrawList();
    for (const auto& point : runtime.points) {
      const ImVec2 center(origin.x + point.x * scale, origin.y + point.y * scale);
      draw->AddCircleFilled(center, 2.0F, IM_COL32(255, 205, 55, 230));
    }
  }
  ImGui::End();
}

}  // namespace

int run_app(const AppConfig& config) {
  GlfwLifetime glfw;
  Window window;
  ImGuiLifetime imgui(window.get());

  auto decoder = make_ffmpeg_cuda_decoder();
  decoder->open(config.video);
  auto presenter = make_cuda_gl_presenter();
  auto superpoint = make_tensorrt_superpoint(config.engine, config.superpoint);
  FeatureStore store(config.database, config.store);
  store.set_session(config.video, config.engine, config.superpoint.input_width,
                    config.superpoint.input_height, config.superpoint.max_keypoints,
                    config.superpoint.detection_threshold);
  Runtime runtime;
  const auto cached_through = store.last_frame_index();

  while (!glfwWindowShouldClose(window.get())) {
    glfwPollEvents();
    const auto now = Clock::now();
    const bool pacing_ready = !runtime.realtime_pacing || now >= runtime.next_frame;
    if (!runtime.eof && !runtime.stopped &&
        (runtime.step_requested || (runtime.playing && pacing_ready))) {
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
        runtime.stats.decoded_frames++;
        runtime.stats.decode_ms = decode_ms;
        if (static_cast<std::int64_t>(frame.frame_index) > cached_through) {
          FrameFeatures features = superpoint->infer(frame);
          features.decode_ms = decode_ms;
          runtime.points = features.keypoints;
          runtime.stats.inferred_frames++;
          runtime.stats.preprocess_ms = features.preprocess_ms;
          runtime.stats.inference_ms = features.inference_ms;
          runtime.stats.readback_ms = features.readback_ms;

          // The online landmark tracker plugs in immediately before this move.
          // It sees the current GPU-derived result without reading from SQLite.
          store.enqueue(std::move(features));
        } else {
          runtime.points.clear();
        }
        const double fps = decoder->nominal_fps();
        runtime.next_frame = now + std::chrono::duration_cast<Clock::duration>(
                                       std::chrono::duration<double>(fps > 0 ? 1.0 / fps : 0.0));
      }
      runtime.step_requested = false;
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    draw_sidebar(runtime, *decoder, store);
    if (runtime.stop_requested) {
      store.flush();
      runtime.stop_requested = false;
    }
    draw_video(runtime);
    ImGui::Render();
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window.get(), &width, &height);
    glViewport(0, 0, width, height);
    glClearColor(0.025F, 0.03F, 0.04F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window.get());
  }
  store.flush();
  return 0;
}

}  // namespace slam_native
