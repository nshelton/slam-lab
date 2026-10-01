// Live depth checks on a synthetic scene (built with the CUDA app; needs a GPU):
// a slanted textured plane seen by a camera that translates and turns, rendered
// analytically, with the exact optical flow on the 4x4 grid.
#include "slam_native/live_depth.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace slam_native;
namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

constexpr int kWidth = 320, kHeight = 240;

struct Scene {
  LiveDepthCamera camera{kWidth, kHeight, 250.0F, 250.0F, 159.5F, 119.5F, 0.0F};
  double noise{};  // luma sigma added to every image (another draw per frame)

  // Camera k: world -> camera, x_c = R (X - c).
  static Pose pose(int k) {
    const double yaw = 0.004 * k, c[3] = {0.02 * k, 0.006 * k, 0.004 * k};
    Pose p;
    p.rotation = {std::cos(yaw), 0, -std::sin(yaw), 0, 1, 0, std::sin(yaw), 0, std::cos(yaw)};
    for (int i = 0; i < 3; ++i) {
      p.translation[i] = 0;
      for (int j = 0; j < 3; ++j) p.translation[i] -= p.rotation[i * 3 + j] * c[j];
    }
    return p;
  }
  // The plane Z = 2 + 0.3 X under source pixel (u, v) of camera k: the world
  // point, and its depth in that camera.
  bool hit(int k, double u, double v, double* X, double* depth) const {
    const Pose p = pose(k);
    float fu = static_cast<float>(u), fv = static_cast<float>(v);
    undistort_point(fu, fv, kWidth, kHeight, camera.k1);
    const double r[3] = {(fu - camera.cx) / camera.fx, (fv - camera.cy) / camera.fy, 1.0};
    const auto c = p.center();
    double d[3];
    for (int i = 0; i < 3; ++i) d[i] = p.rotation[i] * r[0] + p.rotation[3 + i] * r[1] + p.rotation[6 + i] * r[2];
    const double denominator = d[2] - 0.3 * d[0];
    if (std::abs(denominator) < 1e-9) return false;
    const double s = (2.0 + 0.3 * c[0] - c[2]) / denominator;
    if (s <= 0) return false;
    for (int i = 0; i < 3; ++i) X[i] = c[i] + s * d[i];
    *depth = s;  // r has z = 1, so s is the depth along the optical axis
    return true;
  }
  bool project(int k, const double* X, double* u, double* v) const {
    const Pose p = pose(k);
    double x[3];
    for (int i = 0; i < 3; ++i)
      x[i] = p.rotation[i * 3] * X[0] + p.rotation[i * 3 + 1] * X[1] + p.rotation[i * 3 + 2] * X[2] + p.translation[i];
    if (x[2] <= 0.1) return false;
    float fu = static_cast<float>(camera.fx * x[0] / x[2] + camera.cx);
    float fv = static_cast<float>(camera.fy * x[1] / x[2] + camera.cy);
    distort_point(fu, fv, kWidth, kHeight, camera.k1);
    *u = fu;
    *v = fv;
    return true;
  }
  static double texture(const double* X) {
    return 128 + 38 * std::sin(61 * X[0] + 1) * std::sin(47 * X[1]) + 30 * std::sin(83 * X[0] + 29 * X[1]) +
           22 * std::sin(71 * X[1] - 37 * X[0] + 2);
  }
  std::vector<std::uint8_t> image(int k) const {
    std::vector<std::uint8_t> out(kWidth * kHeight, 0);
    std::mt19937 rng(1000 + k);
    std::normal_distribution<double> random(0.0, 1.0);
    for (int v = 0; v < kHeight; ++v)
      for (int u = 0; u < kWidth; ++u) {
        double X[3], depth;
        if (hit(k, u, v, X, &depth))
          out[v * kWidth + u] =
              static_cast<std::uint8_t>(std::clamp(texture(X) + noise * random(rng) + 0.5, 0.0, 255.0));
      }
    return out;
  }
  // Frame k - 1 -> k, at the centres of the 4x4 cells.
  std::vector<FlowVector> flow(int k) const {
    std::vector<FlowVector> out((kWidth / 4) * (kHeight / 4));
    for (int r = 0; r < kHeight / 4; ++r)
      for (int c = 0; c < kWidth / 4; ++c) {
        const double u = 4 * c + 1.5, v = 4 * r + 1.5;
        double X[3], depth, qu, qv;
        if (hit(k - 1, u, v, X, &depth) && project(k, X, &qu, &qv))
          out[r * (kWidth / 4) + c] = {static_cast<float>(qu - u), static_cast<float>(qv - v)};
      }
    return out;
  }
};

template <class T> T* upload(const std::vector<T>& host) {
  T* device = nullptr;
  if (cudaMalloc(&device, host.size() * sizeof(T)) != cudaSuccess ||
      cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
    throw std::runtime_error("upload failed");
  return device;
}

// Feeds frames first .. count - 1 (frame 0 has no flow and starts a history);
// `posed` says which have a pose and `segment` their segment.
template <class Posed, class Segment>
void run(LiveDepth& depth, const Scene& scene, int count, Posed posed, Segment segment,
         const std::vector<DepthAnchor>& anchors = {}, int first = 0) {
  for (int k = first; k < count; ++k) {
    auto* luma = upload(scene.image(k));
    FlowVector* field = k > 0 ? upload(scene.flow(k)) : nullptr;
    GpuFrame frame;
    frame.frame_index = static_cast<std::uint64_t>(k);
    frame.width = kWidth;
    frame.height = kHeight;
    frame.luma = reinterpret_cast<std::uintptr_t>(luma);
    frame.luma_pitch = kWidth;
    DeviceFlowField flow;
    flow.forward = field;
    flow.width = kWidth / 4;
    flow.height = kHeight / 4;
    const Pose pose = Scene::pose(k);
    depth.process(frame, field ? &flow : nullptr, scene.camera, posed(k) ? &pose : nullptr, segment(k), anchors);
    cudaFree(luma);
    cudaFree(field);
  }
}

struct Errors {
  double coverage, median, p95, calibration;
};
Errors errors(const LiveDepthImage& image, const Scene& scene, int frame) {
  std::vector<double> relative, normalized;
  for (int v = 0; v < kHeight; ++v)
    for (int u = 0; u < kWidth; ++u) {
      const float rho = image.inverse_depth[v * kWidth + u], variance = image.variance[v * kWidth + u];
      if (!(rho > 0) || !std::isfinite(variance)) continue;
      double X[3], depth;
      if (!scene.hit(frame, u, v, X, &depth)) continue;
      const double error = std::abs(rho * depth - 1.0);
      relative.push_back(error);
      normalized.push_back(std::abs(rho - 1.0 / depth) / std::sqrt(variance));
    }
  require(relative.size() > 100, "The measurement must cover part of the image");
  std::sort(relative.begin(), relative.end());
  std::sort(normalized.begin(), normalized.end());
  return {double(relative.size()) / (kWidth * kHeight), relative[relative.size() / 2],
          relative[relative.size() * 95 / 100], normalized[normalized.size() / 2] / 0.6745};
}

void measurement(float k1, int gap, double median_limit) {
  Scene scene;
  scene.camera.k1 = k1;
  LiveDepthConfig config;
  config.reference_gap = gap;
  LiveDepth depth(config);
  run(depth, scene, gap + 3, [](int) { return true; }, [](int) { return 0; });
  require(depth.valid() && depth.reference_gap() == gap, "A frame with enough history is measured at the asked gap");
  const auto e = errors(depth.download(), scene, gap + 2);
  const std::string label = " (k1 " + std::to_string(k1) + ", gap " + std::to_string(gap) + "): ";
  if (std::getenv("LIVE_DEPTH_TEST_LOG"))
    std::cout << label << "coverage " << e.coverage << ", median " << e.median << ", p95 " << e.p95
              << ", err/sigma " << e.calibration << '\n';
  require(e.coverage > 0.6, "Most of a textured plane is measured" + label + std::to_string(e.coverage));
  require(e.median < median_limit, "The inverse depth matches the plane" + label + std::to_string(e.median));
  require(e.p95 < 6 * median_limit, "Few measurements are far off" + label + std::to_string(e.p95));
  require(e.calibration < 2.0, "The variance is not badly optimistic" + label + std::to_string(e.calibration));
}

void history() {
  Scene scene;
  LiveDepthConfig config;
  config.reference_gap = 4;
  {
    LiveDepth depth(config);
    run(depth, scene, 1, [](int) { return true; }, [](int) { return 0; });
    require(!depth.valid(), "The first frame has nothing to be matched against");
    run(depth, scene, 3, [](int) { return true; }, [](int) { return 0; });  // frame 0 again: the history starts over
    require(depth.valid() && depth.reference_gap() == 2, "A short history is used as far as it goes");
  }
  {
    LiveDepth depth(config);
    run(depth, scene, 6, [](int k) { return k != 5; }, [](int) { return 0; });
    require(!depth.valid(), "A frame without a pose is not measured");
  }
  {
    LiveDepth depth(config);
    run(depth, scene, 6, [](int) { return true; }, [](int k) { return k < 5 ? 0 : 1; });
    require(!depth.valid(), "A frame is not matched against another segment's frames");
    run(depth, scene, 0, [](int) { return true; }, [](int) { return 0; });
  }
  {
    // Frames 2 and 3 have no pose: frame 5 reaches back over them to frame 1.
    LiveDepth depth(config);
    run(depth, scene, 6, [](int k) { return k != 2 && k != 3; }, [](int) { return 0; });
    require(depth.valid() && depth.reference_gap() == 4, "Frames without a pose are skipped over");
    require(errors(depth.download(), scene, 5).median < 0.01, "and the measurement is as good");
  }
}

std::vector<DepthAnchor> grid_anchors(const Scene& scene, int frame) {
  std::vector<DepthAnchor> points;
  for (int v = 20; v < kHeight; v += 40)
    for (int u = 20; u < kWidth; u += 40) {
      double X[3], z;
      if (scene.hit(frame, u, v, X, &z)) points.push_back({float(u), float(v), float(1.0 / z)});
    }
  return points;
}

// The filter carries the estimate from frame to frame and fuses each frame's
// measurement into it.
void filter() {
  Scene scene;
  scene.noise = 6.0;  // so that a single measurement is limited by noise, which fusing averages out
  const auto all = [](int) { return true; };
  const auto one = [](int) { return 0; };
  LiveDepthConfig config;
  config.reference_gap = 5;
  config.mode = LiveDepthMode::measurement;
  double single = 0;
  {
    LiveDepth depth(config);
    run(depth, scene, 21, all, one);
    single = errors(depth.download(), scene, 20).median;
  }
  config.mode = LiveDepthMode::filter;
  {
    LiveDepth depth(config);
    run(depth, scene, 21, all, one);
    require(depth.valid(), "The filter gives an image on a posed frame");
    const auto e = errors(depth.download(), scene, 20);
    require(e.coverage > 0.75 && double(depth.stats().estimated) / (kWidth * kHeight) > 0.75,
            "The filter covers at least what the measurements did: " + std::to_string(e.coverage));
    if (std::getenv("LIVE_DEPTH_TEST_LOG"))
      std::cout << " filter: coverage " << e.coverage << ", median " << e.median << " (one measurement " << single
                << "), p95 " << e.p95 << ", err/sigma " << e.calibration << '\n';
    require(e.median < 0.6 * single, "Fusing frames beats one measurement: " + std::to_string(e.median) + " against " +
                                     std::to_string(single));
    require(e.p95 < 0.02 && e.calibration < 2.0, "The fused estimate is not badly optimistic");

    // Two frames without a pose: no image (it would be in no known scale),
    // but the state rides the flow and is there when the pose returns.
    run(depth, scene, 23, [](int) { return false; }, one, {}, 21);
    require(!depth.valid(), "No image without a pose");
    run(depth, scene, 24, all, one, {}, 23);
    const auto after = errors(depth.download(), scene, 23);
    require(depth.valid() && after.coverage > 0.6 && after.median < 0.01,
            "The state survives frames without a pose: " + std::to_string(after.coverage) + ", " +
            std::to_string(after.median));

    // Another segment: other frame, other scale. Nothing is carried over.
    run(depth, scene, 25, all, [](int) { return 1; }, {}, 24);
    require(depth.valid() && depth.stats().estimated == 0, "A new segment starts with an empty state");
  }
  // A state in the wrong scale is brought back to the landmarks under it.
  {
    LiveDepth depth(config);
    run(depth, scene, 12, all, one);
    // Scaling the scene would need new poses; scaling the anchors asks for the same correction.
    auto anchors = grid_anchors(scene, 12);
    for (auto& a : anchors) a.inverse_depth *= 1.25F;
    run(depth, scene, 13, all, one, anchors, 12);
    require(std::abs(depth.stats().scale - 1.25F) < 0.02F, "The scale between the landmarks and the state is found: " +
                                                           std::to_string(depth.stats().scale));
  }
  // Fused with the anchors, every pixel has an estimate.
  config.mode = LiveDepthMode::fused;
  {
    LiveDepth depth(config);
    for (int k = 0; k < 21; ++k) run(depth, scene, k + 1, all, one, grid_anchors(scene, k), k);
    const auto e = errors(depth.download(), scene, 20);
    require(depth.stats().estimated == std::size_t(kWidth) * kHeight && e.coverage > 0.99,
            "The fused image covers every pixel");
    require(e.median < 0.004, "and keeps the filter's accuracy: " + std::to_string(e.median));
  }
}

void anchors() {
  Scene scene;
  LiveDepthConfig config;
  config.mode = LiveDepthMode::anchors;
  LiveDepth depth(config);
  std::vector<DepthAnchor> points;
  for (int v = 20; v < kHeight; v += 40)
    for (int u = 20; u < kWidth; u += 40) {
      double X[3], z;
      scene.hit(0, u, v, X, &z);
      points.push_back({float(u), float(v), float(1.0 / z)});
    }
  run(depth, scene, 1, [](int) { return true; }, [](int) { return 0; }, points);
  require(depth.valid(), "Anchors alone give an image");
  const auto image = depth.download();
  for (const auto& p : points)
    require(std::abs(image.inverse_depth[int(p.y) * kWidth + int(p.x)] / p.inverse_depth - 1) < 0.01,
            "A pixel at an anchor takes its inverse depth");
  require(errors(image, scene, 0).median < 0.02, "Between anchors a plane is interpolated closely");
}
}  // namespace

int main() {
  try {
    // The error is a few hundredths of a pixel of disparity, so it falls with the baseline.
    measurement(0.0F, 1, 0.02);
    measurement(0.0F, 5, 0.005);
    measurement(-0.08F, 5, 0.005);  // barrel distortion
    measurement(0.0F, 12, 0.003);
    history();
    filter();
    anchors();
  } catch (const std::exception& error) {
    std::cerr << "live depth test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "live depth tests passed\n";
  return 0;
}
