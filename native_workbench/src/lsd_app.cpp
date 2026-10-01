#define GL_GLEXT_PROTOTYPES
#include "lsd_app.hpp"

#include "lsd/undistort.hpp"
#include "lsd_graph_view.hpp"
#include "slam_native/calibration.hpp"
#include "slam_native/cuda_frame_presenter.hpp"
#include "slam_native/depth_estimator.hpp"
#include "slam_native/display_environment.hpp"
#include "slam_native/launcher.hpp"
#include "slam_native/superpoint.hpp"
#include "slam_native/video_decoder.hpp"

#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace slam_native {
namespace {
using Clock = std::chrono::steady_clock;
using lsd::ImageF;

class GlfwLifetime {
 public:
  GlfwLifetime() {
    configure_display_environment();
    glfwSetErrorCallback([](int, const char* description) { std::fprintf(stderr, "GLFW: %s\n", description); });
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
    value_ = glfwCreateWindow(1600, 940, "LSD Workbench", nullptr, nullptr);
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
    ImGui::GetIO().IniFilename = "imgui-lsd.ini";  // separate layout from the SuperPoint workbench
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
  }
  ~ImGuiLifetime() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
  }
};

class Texture {
 public:
  Texture() = default;
  Texture(const Texture&) = delete;
  Texture& operator=(const Texture&) = delete;
  ~Texture() {
    if (id_) glDeleteTextures(1, &id_);
  }
  void upload(int width, int height, const std::vector<std::uint8_t>& rgba) {
    if (!id_) {
      glGenTextures(1, &id_);
      glBindTexture(GL_TEXTURE_2D, id_);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glBindTexture(GL_TEXTURE_2D, id_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (width != width_ || height != height_) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
      width_ = width;
      height_ = height;
    } else {
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    }
  }
  [[nodiscard]] unsigned int id() const { return id_; }

 private:
  unsigned int id_{};
  int width_{}, height_{};
};

// Turbo colormap (Mikhailov, 2019), polynomial approximation; t in [0, 1].
std::array<std::uint8_t, 3> turbo(float t) {
  t = std::clamp(t, 0.0F, 1.0F);
  const float t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
  const float r = 0.13572138F + 4.61539260F * t - 42.66032258F * t2 + 132.13108234F * t3 - 152.94239396F * t4 +
                  59.28637943F * t5;
  const float g = 0.09140261F + 2.19418839F * t + 4.84296658F * t2 - 14.18503333F * t3 + 4.27729857F * t4 +
                  2.82956604F * t5;
  const float b = 0.10667330F + 12.64194608F * t - 60.58204836F * t2 + 110.36276771F * t3 - 89.90310912F * t4 +
                  27.34824973F * t5;
  const auto byte = [](float v) { return static_cast<std::uint8_t>(std::clamp(v, 0.0F, 1.0F) * 255.0F + 0.5F); };
  return {byte(r), byte(g), byte(b)};
}

// An image drawn into the window, fitted and rotated clockwise by `rotation`.
struct Placement {
  ImVec2 origin;
  float scale{1};
  float width{}, height{};  // unrotated image size
  int rotation{};
  // Image pixel (pixel centres at integers) -> screen.
  [[nodiscard]] ImVec2 screen(float x, float y) const {
    x += 0.5F;
    y += 0.5F;
    float u = x, v = y;
    if (rotation == 90) { u = height - y; v = x; }
    else if (rotation == 180) { u = width - x; v = height - y; }
    else if (rotation == 270) { u = y; v = width - x; }
    return {origin.x + u * scale, origin.y + v * scale};
  }
};

Placement draw_image(unsigned int texture, int width, int height, int rotation, float reserve_bottom) {
  const ImVec2 available = ImGui::GetContentRegionAvail();
  const bool sideways = rotation == 90 || rotation == 270;
  const float w = static_cast<float>(width), h = static_cast<float>(height);
  const float shown_w = sideways ? h : w, shown_h = sideways ? w : h;
  const float scale = std::max(0.01F, std::min(available.x / shown_w, std::max(1.0F, available.y - reserve_bottom) / shown_h));
  const ImVec2 size(shown_w * scale, shown_h * scale);
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImGui::InvisibleButton("##image", size);
  const ImVec2 corners[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  const int first = (4 - rotation / 90) % 4;
  ImGui::GetWindowDrawList()->AddImageQuad(static_cast<ImTextureID>(texture), origin, {origin.x + size.x, origin.y},
                                           {origin.x + size.x, origin.y + size.y}, {origin.x, origin.y + size.y},
                                           corners[first], corners[(first + 1) % 4], corners[(first + 2) % 4],
                                           corners[(first + 3) % 4]);
  return {origin, scale, w, h, rotation};
}

ImageF download_luma(const GpuFrame& frame) {
  if (frame.ready_event) cudaEventSynchronize(static_cast<cudaEvent_t>(frame.ready_event));
  const int bytes = frame.format == PixelFormat::p010 ? 2 : 1;
  const std::size_t row = static_cast<std::size_t>(frame.width) * bytes;
  std::vector<std::uint8_t> host(row * frame.height);
  const cudaError_t status = cudaMemcpy2D(host.data(), row, reinterpret_cast<const void*>(frame.luma),
                                          frame.luma_pitch, row, frame.height, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) throw std::runtime_error(std::string("luma download: ") + cudaGetErrorString(status));
  ImageF image(frame.width, frame.height);
  if (bytes == 1) {
    for (std::size_t i = 0; i < image.data.size(); ++i) image.data[i] = host[i];
  } else {
    const auto* words = reinterpret_cast<const std::uint16_t*>(host.data());
    for (std::size_t i = 0; i < image.data.size(); ++i) image.data[i] = static_cast<float>(words[i]) / 256.0F;
  }
  return image;
}

constexpr const char* kDepthViews[] = {"Inverse depth", "Depth uncertainty (sigma / depth)", "Tracking residual",
                                       "Huber weight", "Network depth (keyframe prior)", "Keyframe image",
                                       "Validity (filter)"};

struct Session {
  explicit Session(LsdAppConfig settings) : config(std::move(settings)) {
    open_decoder();
    presenter = make_cuda_gl_presenter();
    rotation = config.base.display_rotation.value_or(decoder->display_rotation());
    graph.set_display_rotation(rotation);
    const auto& presets = depth_model_presets();
    if (const auto* spec = find_depth_model(config.depth_model)) depth_model = static_cast<int>(spec - presets.data());
    // SuperPoint on keyframes for loop candidates (optional).
    std::error_code missing;
    if (!config.base.engine.empty() && std::filesystem::is_regular_file(config.base.engine, missing)) {
      try {
        superpoint = make_tensorrt_superpoint(config.base.engine, config.base.superpoint);
      } catch (const std::exception& e) {
        loop_note = std::string("loop candidates off: ") + e.what();
      }
    } else {
      loop_note = "loop candidates off: no SuperPoint engine";
    }
  }

  // SuperPoint features of a keyframe frame on the working pinhole grid.
  lsd::KeyframeFeatures keyframe_features(const GpuFrame& frame) {
    lsd::KeyframeFeatures out;
    if (!superpoint) return out;
    const auto start = Clock::now();
    const FrameFeatures f = superpoint->infer(frame);
    const auto& cam = odometry->camera();
    const float factor = static_cast<float>(1 << levels_down);
    const std::size_t dim = f.keypoints.empty() ? 256 : f.descriptors.size() / f.keypoints.size();
    out.dimension = static_cast<int>(dim);
    for (std::size_t i = 0; i < f.keypoints.size(); ++i) {
      const lsd::Vec2 distorted{(f.keypoints[i].x + 0.5) / factor - 0.5, (f.keypoints[i].y + 0.5) / factor - 0.5};
      const lsd::Vec2 p = undistorter.undistort_pixel(distorted);
      if (p.x() < 0 || p.y() < 0 || p.x() > cam.width - 1 || p.y() > cam.height - 1) continue;
      out.pixels.push_back(p);
      out.descriptors.insert(out.descriptors.end(), f.descriptors.begin() + static_cast<long>(i * dim),
                             f.descriptors.begin() + static_cast<long>((i + 1) * dim));
    }
    superpoint_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return out;
  }

  void open_decoder() {
    decoder = make_ffmpeg_cuda_decoder();
    decoder->open(config.base.video);
  }

  void restart() {
    open_decoder();
    odometry.reset();
    graph.reset();
    eof = false;
    lost_frames = rejected_keyframes = frames = 0;
    track_ms_total = mapping_ms_total = graph_ms_total = 0;
    reciprocal_errors.clear();
    reciprocal_motion.clear();
    debug = {};
    debug_keyframe = -1;
    last_prior = {};
    depth_dirty = true;
  }

  // Working camera: source intrinsics, halved until at most max_width wide.
  void create_odometry(int width, int height) {
    CameraIntrinsics k;
    if (config.base.odometry.intrinsics) {
      k = *config.base.odometry.intrinsics;
      intrinsics_source = "--intrinsics";
    } else if (const auto calibration = read_calibration(config.base.video)) {
      k = *calibration;
      intrinsics_source = "calibration.json";
    } else {
      k = CameraIntrinsics::from_horizontal_fov(width, height, config.base.odometry.horizontal_fov_degrees);
      intrinsics_source = "hfov " + std::to_string(static_cast<int>(config.base.odometry.horizontal_fov_degrees)) + " deg";
    }
    source_focal = k.fx;
    lsd::Camera camera{k.fx, k.fy, k.cx, k.cy, width, height};
    levels_down = 0;
    while (camera.width > config.max_width) {
      camera = camera.half();
      ++levels_down;
    }
    odometry = std::make_unique<lsd::Odometry>(camera, config.odometry);
    undistorter = {};
    distortion_note = "lens distortion: none in calibration";
    if (intrinsics_source == "calibration.json") {
      if (const auto d = read_distortion(config.base.video)) {
        if (config.undistort) {
          undistorter = lsd::Undistorter(camera, *d);
          odometry->set_mask(undistorter.mask());
          char text[96];
          std::snprintf(text, sizeof text, "undistorted (k1 %.3f, k2 %.3f)", (*d)[0], (*d)[1]);
          distortion_note = text;
        } else {
          distortion_note = "lens distortion ignored (--no-undistort)";
        }
      }
    } else {
      distortion_note = "lens distortion unknown (no calibration.json)";
    }
  }

  void advance() {
    if (restart_requested) {
      restart_requested = false;
      restart();
      playing = true;
    }
    if (eof || !(step_requested || playing)) return;
    if (realtime && Clock::now() < next_frame) return;
    GpuFrame frame;
    if (!decoder->next(frame)) {
      eof = true;
      playing = false;
      return;
    }
    step_requested = false;
    const double fps = decoder->nominal_fps() > 0 ? decoder->nominal_fps() : 30;
    next_frame = Clock::now() + std::chrono::microseconds(static_cast<long>(1e6 / fps));
    try {
      process_frame(frame);
    } catch (const std::exception& e) {
      error = e.what();
      playing = false;
    }
  }

  void process_frame(const GpuFrame& frame) {
    video_texture = presenter->present(frame);
    frame_width = frame.width;
    frame_height = frame.height;
    frame_index = frame.frame_index;
    timestamp_ns = frame.timestamp_ns;
    if (!odometry) create_odometry(frame.width, frame.height);
    ImageF image = download_luma(frame);
    for (int l = 0; l < levels_down; ++l) image = lsd::downsample(image);
    image = undistorter.apply(image);

    const auto& r = odometry->track(frame.frame_index, frame.timestamp_ns, image, &debug);
    debug_keyframe = r.keyframe;
    ++frames;
    track_ms_total += r.ms;
    mapping_ms_total += r.mapping_ms;
    if (r.state == lsd::TrackingState::lost) ++lost_frames;
    if (r.keyframe_wanted) {
      ImageF prior;
      if (run_depth(frame, prior)) {
        last_prior = prior;
        if (keep_priors) priors.push_back(prior);
        if (!odometry->add_keyframe(prior, keyframe_features(frame))) {
          ++rejected_keyframes;
        } else {
          const auto& g = odometry->last_constraints();
          graph_ms_total += g.ms + g.optimize_ms;
          reciprocal_errors.insert(reciprocal_errors.end(), g.reciprocal_errors.begin(), g.reciprocal_errors.end());
          reciprocal_motion.insert(reciprocal_motion.end(), g.reciprocal_motion.begin(), g.reciprocal_motion.end());
          pair_details.insert(pair_details.end(), g.pair_details.begin(), g.pair_details.end());
        }
      }
    }
    depth_dirty = true;
  }

  // Runs the depth network on this frame and waits for it (keyframes only).
  // Output: metres on the working grid, 0 where unknown.
  bool run_depth(const GpuFrame& frame, ImageF& out) {
    if (depth_model < 0) {
      depth_error = "No depth model: keyframes need a depth prior until the stereo filter exists.";
      return false;
    }
    const auto& spec = depth_model_presets()[depth_model];
    if (!depth || depth->spec().name != spec.name || depth->rotation() != rotation) {
      depth.reset();
      const bool sideways = rotation == 90 || rotation == 270;
      const bool portrait = (sideways ? frame.width : frame.height) > (sideways ? frame.height : frame.width);
      try {
        depth = std::make_unique<DepthEstimator>(spec, depth_engine_path(config.models_dir, spec, portrait),
                                                 rotation);
      } catch (const std::exception& e) {
        depth_error = e.what();
        depth_model = -1;
        return false;
      }
    }
    const auto start = Clock::now();
    DepthMap map;
    while (depth->busy()) {
      depth->poll(map);
      std::this_thread::yield();
    }
    if (!depth->submit(frame, static_cast<float>(source_focal))) return false;
    while (!depth->poll(map)) std::this_thread::yield();
    depth_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    depth_error.clear();
    const lsd::Camera& cam = odometry->camera();
    const float factor = static_cast<float>(1 << levels_down);
    out = ImageF(cam.width, cam.height);
    for (int y = 0; y < cam.height; ++y) {
      for (int x = 0; x < cam.width; ++x) {
        // The network saw the distorted frame: sample it where this pinhole pixel came from.
        const lsd::Vec2 d = undistorter.source(x, y);
        const float z = map.at_source((static_cast<float>(d.x()) + 0.5F) * factor - 0.5F,
                                      (static_cast<float>(d.y()) + 0.5F) * factor - 0.5F);
        out.at(x, y) = std::isfinite(z) && z > 0 ? z : 0.0F;
      }
    }
    return true;
  }

  LsdAppConfig config;
  std::unique_ptr<VideoDecoder> decoder;
  std::unique_ptr<CudaFramePresenter> presenter;
  std::unique_ptr<DepthEstimator> depth;
  std::unique_ptr<lsd::Odometry> odometry;
  std::unique_ptr<SuperPoint> superpoint;
  std::string loop_note;
  double superpoint_ms{};
  lsd::Undistorter undistorter;
  std::string distortion_note;
  LsdGraphView graph;

  unsigned int video_texture{};
  int frame_width{}, frame_height{};
  std::uint64_t frame_index{};
  std::int64_t timestamp_ns{};
  int rotation{};
  int levels_down{};
  double source_focal{};
  std::string intrinsics_source;
  bool playing{true}, step_requested{}, eof{}, restart_requested{}, close_requested{};
  bool realtime{false};
  Clock::time_point next_frame{};
  std::string error;

  int depth_model{-1};
  std::string depth_error;
  double depth_ms{};
  ImageF last_prior;
  bool keep_priors{};          // keep every keyframe's prior (depth audits)
  std::vector<ImageF> priors;  // in keyframe-attempt order (index = keyframe id while none is rejected)

  lsd::Se3TrackingDebug debug;  // aligned with keyframe debug_keyframe's level-0 points
  int debug_keyframe{-1};
  std::uint64_t frames{}, lost_frames{}, rejected_keyframes{};
  double track_ms_total{}, mapping_ms_total{}, graph_ms_total{};
  std::vector<double> reciprocal_errors;  // every constraint pair tried (diagnostics)
  std::vector<std::array<double, 3>> reciprocal_motion;
  std::vector<std::array<double, 6>> pair_details;

  bool show_depth{true}, show_graph{true};
  bool video_overlay{true};
  int overlay_stride{2};
  int depth_view{0};
  bool depth_on_image{true};
  bool depth_dirty{true};
  Texture depth_texture;
  float idepth_low{0}, idepth_high{1};  // colour range (current keyframe, 2nd-98th percentile)
};

// Inverse-depth colour: near (large inverse depth) = red.
std::array<std::uint8_t, 3> idepth_color(const Session& s, float d) {
  return turbo((d - s.idepth_low) / std::max(1e-6F, s.idepth_high - s.idepth_low));
}

void update_idepth_range(Session& s, const lsd::Keyframe& kf) {
  std::vector<float> values;
  for (std::size_t i = 0; i < kf.depth.idepth.data.size(); i += 3)
    if (kf.depth.variance.data[i] > 0) values.push_back(kf.depth.idepth.data[i]);
  if (values.size() < 10) return;
  std::sort(values.begin(), values.end());
  s.idepth_low = values[values.size() * 2 / 100];
  s.idepth_high = values[values.size() * 98 / 100];
}

void upload_depth_view(Session& s) {
  const lsd::Keyframe* current = s.odometry->current_keyframe();
  const bool tracking_view = s.depth_view == 2 || s.depth_view == 3;
  const lsd::Keyframe* kf = current;
  if (tracking_view && s.debug_keyframe >= 0 &&
      s.debug_keyframe < static_cast<int>(s.odometry->keyframes().size()))
    kf = s.odometry->keyframes()[s.debug_keyframe].get();
  if (!kf) return;
  update_idepth_range(s, *current);
  const lsd::ImageF& image = kf->pyramid.levels[0].image;
  const int w = image.width, h = image.height;
  std::vector<std::uint8_t> rgba(static_cast<std::size_t>(w) * h * 4, 255);
  const float dim = s.depth_on_image && s.depth_view != 5 ? 0.45F : (s.depth_view == 5 ? 1.0F : 0.0F);
  const auto put = [&](int x, int y, std::array<std::uint8_t, 3> c) {
    auto* p = &rgba[(static_cast<std::size_t>(y) * w + x) * 4];
    p[0] = c[0];
    p[1] = c[1];
    p[2] = c[2];
  };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const auto g = static_cast<std::uint8_t>(std::clamp(image.at(x, y) * dim, 0.0F, 255.0F));
      put(x, y, {g, g, g});
    }
  if (s.depth_view == 0 || s.depth_view == 1) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        if (!kf->depth.valid(x, y)) continue;
        const float d = kf->depth.idepth.at(x, y);
        if (s.depth_view == 0) {
          put(x, y, idepth_color(s, d));
        } else {
          // Relative sigma, log scale 1% (blue) .. 100% (red).
          const float rel = std::sqrt(kf->depth.variance.at(x, y)) / d;
          put(x, y, turbo((std::log10(std::max(rel, 1e-3F)) + 2.0F) / 2.0F));
        }
      }
  } else if (tracking_view) {
    const auto& points = kf->reference.points(0);
    const auto& values = s.depth_view == 2 ? s.debug.residual : s.debug.weight;
    if (values.size() == points.size()) {
      for (std::size_t i = 0; i < points.size(); ++i) {
        const float v = values[i];
        const int x = static_cast<int>(points[i].u), y = static_cast<int>(points[i].v);
        if (!std::isfinite(v)) {
          put(x, y, {70, 70, 70});  // not used (projected outside)
        } else if (s.depth_view == 2) {
          // r / sigma: blue negative, red positive, saturating at +-3.
          const float t = std::clamp(v / 3.0F, -1.0F, 1.0F);
          const auto a = static_cast<std::uint8_t>(255 * std::abs(t));
          put(x, y, t < 0 ? std::array<std::uint8_t, 3>{40, 40, static_cast<std::uint8_t>(60 + a * 195 / 255)}
                          : std::array<std::uint8_t, 3>{static_cast<std::uint8_t>(60 + a * 195 / 255), 40, 40});
        } else {
          const auto good = static_cast<std::uint8_t>(255 * v);
          put(x, y, {static_cast<std::uint8_t>(255 - good), good, 40});
        }
      }
    }
  } else if (s.depth_view == 6) {
    if (kf->filter) {
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          const auto& p = kf->filter->at(x, y);
          if (p.valid) put(x, y, turbo(p.validity / 40.0F));
        }
    }
  } else if (s.depth_view == 4 && s.last_prior.width == w && s.last_prior.height == h) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const float z = s.last_prior.at(x, y);
        if (z > 0) put(x, y, idepth_color(s, 1.0F / z));
      }
  }
  s.depth_texture.upload(w, h, rgba);
}

void draw_sidebar(Session& s) {
  ImGui::SetNextWindowPos({8, 8}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({330, 900}, ImGuiCond_FirstUseEver);
  ImGui::Begin("LSD pipeline");
  if (ImGui::Button(s.playing ? "Pause" : "Play")) s.playing = !s.playing;
  ImGui::SameLine();
  if (ImGui::Button("Step")) {
    s.playing = false;
    s.step_requested = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Restart")) s.restart_requested = true;
  ImGui::SameLine();
  if (ImGui::Button("Close")) s.close_requested = true;
  ImGui::Checkbox("Realtime pacing", &s.realtime);
  ImGui::TextWrapped("%s", s.config.base.video.filename().c_str());
  ImGui::Text("Frame %llu  %.2f s%s", static_cast<unsigned long long>(s.frame_index),
              static_cast<double>(s.timestamp_ns - s.decoder->start_time_ns()) / 1e9, s.eof ? "  (end)" : "");
  if (!s.error.empty()) ImGui::TextColored({1, 0.45F, 0.35F, 1}, "%s", s.error.c_str());
  ImGui::Separator();
  if (s.odometry) {
    const auto& cam = s.odometry->camera();
    ImGui::Text("Working image %dx%d (1/%d), f %.1f px", cam.width, cam.height, 1 << s.levels_down, cam.fx);
    ImGui::TextDisabled("Intrinsics: %s; %s", s.intrinsics_source.c_str(), s.distortion_note.c_str());
    const auto& r = s.odometry->last();
    const char* state = r.state == lsd::TrackingState::tracking ? "tracking" :
                        r.state == lsd::TrackingState::lost ? "LOST" : "waiting for a keyframe";
    const ImVec4 color = r.state == lsd::TrackingState::tracking ? ImVec4(0.4F, 0.9F, 0.4F, 1) :
                         r.state == lsd::TrackingState::lost ? ImVec4(1, 0.4F, 0.3F, 1) : ImVec4(1, 0.8F, 0.3F, 1);
    ImGui::TextColored(color, "%s", state);
    ImGui::Text("Keyframes %zu (current #%d)", s.odometry->keyframes().size(), r.keyframe);
    if (const auto* kf = s.odometry->current_keyframe()) {
      ImGui::Text("  %d depth pixels, %d stereo updates", kf->points, kf->observations);
      ImGui::Text("  start: %d propagated, %d from prior (x%.3f)", kf->propagated, kf->seeded, kf->scale_correction);
    }
    ImGui::Text("Usage %.2f  energy %.3f  points %d", r.tracking.usage, r.tracking.mean_energy, r.tracking.used_points);
    ImGui::Text("Iterations %d  track %.1f ms (mean %.1f)", r.tracking.iterations, r.ms,
                s.frames ? s.track_ms_total / static_cast<double>(s.frames) : 0.0);
    ImGui::Text("Keyframe score %.2f (new at 1)", r.keyframe_score);
    if (r.stereo.candidates > 0) {
      ImGui::Text("Stereo: %d matched, %d new, %d inconsistent", r.stereo.matched, r.stereo.created, r.stereo.inconsistent);
      ImGui::Text("  %d failed, %d skipped of %d; %d dropped", r.stereo.failed, r.stereo.skipped, r.stereo.candidates,
                  r.stereo.removed);
    }
    ImGui::Text("Mapping %.1f ms (mean %.1f)", r.mapping_ms,
                s.frames ? s.mapping_ms_total / static_cast<double>(s.frames) : 0.0);
    if (!s.loop_note.empty()) ImGui::TextDisabled("%s", s.loop_note.c_str());
    else ImGui::Text("Loop candidates: %d (SuperPoint %.1f ms per keyframe)", s.odometry->loop_candidates_total(),
                     s.superpoint_ms);
    const auto& g = s.odometry->last_constraints();
    if (g.tried > 0 || g.parent_fallback) {
      ImGui::Text("Graph: %d/%d constraints accepted%s", g.accepted, g.tried, g.parent_fallback ? ", parent fallback" : "");
      ImGui::Text("  %d alignment failed, %d reciprocal; %.0f + %.0f ms", g.rejected_tracking, g.rejected_reciprocal,
                  g.ms, g.optimize_ms);
    }
    ImGui::Text("Lost frames %llu, rejected keyframes %llu", static_cast<unsigned long long>(s.lost_frames),
                static_cast<unsigned long long>(s.rejected_keyframes));
    if (!s.odometry->event().empty()) ImGui::TextColored({1, 0.8F, 0.3F, 1}, "%s", s.odometry->event().c_str());
  }
  ImGui::Separator();
  ImGui::TextUnformatted("Keyframe depth prior");
  const auto& presets = depth_model_presets();
  const char* current = s.depth_model >= 0 ? presets[s.depth_model].name.c_str() : "Off";
  ImGui::SetNextItemWidth(-1);
  if (ImGui::BeginCombo("##depth-model", current)) {
    for (int i = 0; i < static_cast<int>(presets.size()); ++i)
      if (ImGui::Selectable(presets[i].name.c_str(), i == s.depth_model)) s.depth_model = i;
    ImGui::EndCombo();
  }
  if (s.depth_ms > 0) ImGui::Text("Network %.1f ms per keyframe", s.depth_ms);
  if (!s.depth_error.empty()) {
    ImGui::PushTextWrapPos(0);
    ImGui::TextColored({1, 0.45F, 0.35F, 1}, "%s", s.depth_error.c_str());
    ImGui::PopTextWrapPos();
  }
  ImGui::Separator();
  ImGui::TextUnformatted("Settings");
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Live. Gradient threshold and prior sigma apply to new keyframes.");
  auto& c = s.odometry ? s.odometry->config() : s.config.odometry;
  float grad = static_cast<float>(c.filter.gradient_threshold), prior = static_cast<float>(c.prior_relative_sigma);
  float sigma_l = static_cast<float>(c.filter.sigma_epipolar_px), max_ssd = static_cast<float>(c.filter.max_ssd);
  float kfd = static_cast<float>(c.keyframe_distance_weight), kfu = static_cast<float>(c.keyframe_usage_weight);
  float huber = static_cast<float>(c.tracker.huber), sigma_i = static_cast<float>(c.tracker.sigma_intensity);
  ImGui::SetNextItemWidth(140);
  if (ImGui::SliderFloat("Gradient threshold", &grad, 1, 30, "%.1f")) c.filter.gradient_threshold = grad;
  ImGui::SetNextItemWidth(140);
  if (ImGui::SliderFloat("Prior sigma (rel.)", &prior, 0.02F, 1.0F, "%.2f", ImGuiSliderFlags_Logarithmic))
    c.prior_relative_sigma = prior;
  ImGui::SetNextItemWidth(140);
  if (ImGui::SliderFloat("KF distance weight", &kfd, 0.5F, 15, "%.1f")) c.keyframe_distance_weight = kfd;
  ImGui::SetNextItemWidth(140);
  if (ImGui::SliderFloat("KF usage weight", &kfu, 0.5F, 15, "%.1f")) c.keyframe_usage_weight = kfu;
  ImGui::SetNextItemWidth(140);
  if (ImGui::SliderFloat("Huber (sigmas)", &huber, 0.5F, 10, "%.1f")) c.tracker.huber = huber;
  ImGui::SetNextItemWidth(140);
  if (ImGui::SliderFloat("Image noise sigma", &sigma_i, 0.5F, 20, "%.1f")) c.tracker.sigma_intensity = sigma_i;
  ImGui::Checkbox("Stereo depth filter", &c.stereo);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("On: every tracked frame refines the keyframe's depth (epipolar stereo, Kalman\n"
                      "fusion, regularisation) and new keyframes inherit it. Off: network depth only.");
  ImGui::SetNextItemWidth(140);
  if (ImGui::SliderFloat("Epipolar sigma (px)", &sigma_l, 0.05F, 3.0F, "%.2f", ImGuiSliderFlags_Logarithmic))
    c.filter.sigma_epipolar_px = sigma_l;
  ImGui::SetNextItemWidth(140);
  if (ImGui::SliderFloat("Max stereo SSD", &max_ssd, 100, 10000, "%.0f", ImGuiSliderFlags_Logarithmic))
    c.filter.max_ssd = max_ssd;
  ImGui::Checkbox("Pose graph (Sim(3) constraints)", &c.graph);
  float max_reciprocal = static_cast<float>(c.constraints.max_reciprocal);
  ImGui::SetNextItemWidth(140);
  if (ImGui::SliderFloat("Reciprocal limit", &max_reciprocal, 1, 10000, "%.0f", ImGuiSliderFlags_Logarithmic))
    c.constraints.max_reciprocal = max_reciprocal;
  ImGui::Checkbox("Constant velocity", &c.constant_velocity);
  ImGui::SameLine();
  ImGui::Checkbox("Scale lock", &c.scale_lock);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Rescale each new keyframe's network depth to the previous keyframe's depth\n"
                      "(median ratio over overlapping pixels), so the trajectory keeps one scale.");
  if (!s.odometry) s.config.odometry = c;
  ImGui::Separator();
  ImGui::Checkbox("Keyframe depth", &s.show_depth);
  ImGui::SameLine();
  ImGui::Checkbox("Pose graph", &s.show_graph);
  ImGui::End();
}

void draw_video(Session& s) {
  ImGui::SetNextWindowPos({346, 8}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({600, 470}, ImGuiCond_FirstUseEver);
  ImGui::Begin("Video");
  if (!s.video_texture || !s.odometry) {
    ImGui::TextDisabled("Waiting for the first decoded frame...");
    ImGui::End();
    return;
  }
  ImGui::Checkbox("Keyframe depth reprojected", &s.video_overlay);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("The current keyframe's semi-dense points warped into this frame with the\n"
                      "tracked pose, coloured by inverse depth (red = near).");
  if (s.video_overlay) {
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::SliderInt("stride", &s.overlay_stride, 1, 8);
  }
  const Placement place = draw_image(s.video_texture, s.frame_width, s.frame_height, s.rotation, 0);
  const auto& r = s.odometry->last();
  const auto& keyframes = s.odometry->keyframes();
  if (s.video_overlay && r.state == lsd::TrackingState::tracking && r.keyframe >= 0 &&
      r.keyframe < static_cast<int>(keyframes.size())) {
    const auto& kf = *keyframes[r.keyframe];
    // Pose of this frame relative to the keyframe (identity on the keyframe's own frame).
    const lsd::SE3 relative = r.tracking.pose;
    const auto& cam = s.odometry->camera();
    const float factor = static_cast<float>(1 << s.levels_down);
    const float half = std::max(1.0F, 0.5F * factor * place.scale);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->PushClipRect(place.origin, ImGui::GetItemRectMax(), true);
    const auto& points = kf.reference.points(0);
    for (std::size_t i = 0; i < points.size(); i += static_cast<std::size_t>(s.overlay_stride)) {
      const lsd::Vec3 X = relative * points[i].x;
      if (X.z() <= 0) continue;
      const lsd::Vec2 pinhole = cam.project(X);
      if (pinhole.x() < 0 || pinhole.y() < 0 || pinhole.x() > cam.width - 1 || pinhole.y() > cam.height - 1) continue;
      const lsd::Vec2 uv = s.undistorter.distort_pixel(pinhole);  // the video shows the distorted frame
      const ImVec2 p = place.screen((static_cast<float>(uv.x()) + 0.5F) * factor - 0.5F,
                                    (static_cast<float>(uv.y()) + 0.5F) * factor - 0.5F);
      const auto c = idepth_color(s, static_cast<float>(1.0 / X.z()));
      draw->AddRectFilled({p.x - half, p.y - half}, {p.x + half, p.y + half}, IM_COL32(c[0], c[1], c[2], 200));
    }
    draw->PopClipRect();
  }
  ImGui::End();
}

void draw_depth(Session& s) {
  if (!s.show_depth) return;
  ImGui::SetNextWindowPos({346, 486}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({600, 446}, ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Keyframe depth", &s.show_depth)) {
    ImGui::End();
    return;
  }
  ImGui::SetNextItemWidth(260);
  if (ImGui::Combo("##depth-view", &s.depth_view, kDepthViews, IM_ARRAYSIZE(kDepthViews))) s.depth_dirty = true;
  ImGui::SameLine();
  if (ImGui::Checkbox("Over image", &s.depth_on_image)) s.depth_dirty = true;
  const lsd::Keyframe* kf = s.odometry ? s.odometry->current_keyframe() : nullptr;
  if (!kf) {
    ImGui::TextDisabled("No keyframe yet.");
    ImGui::End();
    return;
  }
  switch (s.depth_view) {
    case 0: ImGui::TextDisabled("Keyframe #%d: inverse depth %.3f (blue, far) .. %.3f (red, near)", kf->id, s.idepth_low, s.idepth_high); break;
    case 1: ImGui::TextDisabled("Keyframe #%d: inverse-depth sigma / inverse depth, 1%% (blue) .. 100%% (red)", kf->id); break;
    case 2: ImGui::TextDisabled("Keyframe #%d, last frame: r / sigma, blue < 0 < red (saturates at 3), grey unused", s.debug_keyframe); break;
    case 3: ImGui::TextDisabled("Keyframe #%d, last frame: Huber weight, red 0 .. green 1, grey unused", s.debug_keyframe); break;
    case 4: ImGui::TextDisabled("Dense network depth of the latest keyframe, before semi-dense selection"); break;
    case 6: ImGui::TextDisabled("Keyframe #%d: validity count, 0 (blue) .. 40 (red); drives hole filling and removal", kf->id); break;
    default: ImGui::TextDisabled("Keyframe #%d (frame %llu)", kf->id, static_cast<unsigned long long>(kf->frame_index)); break;
  }
  if (s.depth_dirty) {
    upload_depth_view(s);
    s.depth_dirty = false;
  }
  const auto& image = kf->pyramid.levels[0].image;
  if (s.depth_texture.id()) draw_image(s.depth_texture.id(), image.width, image.height, s.rotation, 0);
  ImGui::End();
}
}  // namespace

int run_lsd_app(const LsdAppConfig& config) {
  GlfwLifetime glfw;
  Window window;
  validate_cuda_gl_display();
  ImGuiLifetime imgui(window.get());
  std::error_code path_error;
  const std::filesystem::path executable = std::filesystem::read_symlink("/proc/self/exe", path_error);
  const auto native_root =
      path_error ? std::filesystem::current_path() : executable.parent_path().parent_path().parent_path();
  LsdAppConfig initial = config;
  if (initial.models_dir.empty()) initial.models_dir = native_root / "models";
  if (initial.base.engine.empty()) initial.base.engine = native_root / "models/superpoint-1024x576-k2048.engine";
  Launcher launcher(initial.base, native_root.parent_path());
  launcher.set_video_only(
      "Direct monocular SLAM after LSD-SLAM (Engel et al., ECCV 2014): whole-image alignment on a semi-dense "
      "inverse-depth map. No features or caches; keyframe depth is seeded from the depth network.");
  DatabaseSummary database_summary;
  std::unique_ptr<Session> session;
  const auto start = [&](const AppConfig& selected) {
    LsdAppConfig chosen = initial;
    chosen.base.video = selected.video;
    try {
      session = std::make_unique<Session>(chosen);
      launcher.record_recent(selected);
    } catch (const std::exception& error) {
      launcher.set_error(error.what());
    }
  };
  if (config.base.start_immediately) {
    AppConfig selected = initial.base;
    selected.video = std::filesystem::absolute(selected.video);
    start(selected);
  }

  while (!glfwWindowShouldClose(window.get())) {
    glfwPollEvents();
    if (session) session->advance();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (session) {
      draw_sidebar(*session);
      draw_video(*session);
      draw_depth(*session);
      if (session->odometry) session->graph.draw(*session->odometry, &session->show_graph);
    } else {
      AppConfig selected;
      if (launcher.draw(selected, database_summary)) start(selected);
    }
    ImGui::Render();
    int width = 0, height = 0;
    glfwGetFramebufferSize(window.get(), &width, &height);
    glViewport(0, 0, width, height);
    glClearColor(0.025F, 0.03F, 0.04F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window.get());
    if (session && session->close_requested) session.reset();
  }
  return 0;
}

}  // namespace slam_native
