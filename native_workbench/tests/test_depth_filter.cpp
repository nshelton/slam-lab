// DepthFilter on a synthetic scene: a background wall and a nearer box, a
// camera moving sideways and forward (optionally turning), network maps with a
// per-surface bias shared between frames plus independent per-frame noise.
#include "slam_native/depth_filter.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using namespace slam_native;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

constexpr int kSourceW = 320, kSourceH = 240, kW = 160, kH = 120;
const CameraIntrinsics kK{300, 300, 159.5, 119.5};

// Camera centre (step_x k, 0, 0.03 k), turning by yaw_per_frame about y.
double g_step_x = 0.02;
double g_yaw_per_frame = 0;

std::array<double, 9> rotation_at(int frame) {
  const double a = g_yaw_per_frame * frame, c = std::cos(a), s = std::sin(a);
  return {c, 0, -s, 0, 1, 0, s, 0, c};
}

Pose pose_at(int frame) {
  Pose pose;  // x = R (X - C)
  pose.rotation = rotation_at(frame);
  const double C[3] = {g_step_x * frame, 0, 0.03 * frame};
  for (int r = 0; r < 3; ++r)
    pose.translation[r] = -(pose.rotation[r * 3] * C[0] + pose.rotation[r * 3 + 1] * C[1] + pose.rotation[r * 3 + 2] * C[2]);
  return pose;
}

// Standard normal from a hash of an integer cell (bias fixed to the surface).
double cell_normal(int a, int b, int c) {
  std::uint64_t h = static_cast<std::uint64_t>(a) * 73856093ULL ^ static_cast<std::uint64_t>(b) * 19349663ULL ^
                    static_cast<std::uint64_t>(c) * 83492791ULL;
  std::mt19937_64 rng(h);
  return std::normal_distribution<double>(0, 1)(rng);
}

struct Truth {
  double depth;  // camera z
  double bias;   // the network's log bias on that surface
};

// moving_patch: a region where the scene changed (an object appeared), 0.6x nearer.
Truth trace(int frame, int u, int v, double bias_sigma, bool moving_patch = false) {
  const double x = (u + 0.5) / 0.5 - 0.5, y = (v + 0.5) / 0.5 - 0.5;
  const double ray[3] = {(x - kK.cx) / kK.fx, (y - kK.cy) / kK.fy, 1};
  const auto R = rotation_at(frame);
  double dir[3];  // world direction R^T ray (camera z component 1, so the ray parameter is the depth)
  for (int c = 0; c < 3; ++c) dir[c] = R[c] * ray[0] + R[3 + c] * ray[1] + R[6 + c] * ray[2];
  const double C[3] = {g_step_x * frame, 0, 0.03 * frame};
  double s = (5.0 - C[2]) / dir[2];  // wall at Z = 5
  int surface = 0;
  const double box = (3.0 - C[2]) / dir[2];  // box face at Z = 3, |X| < 0.4, |Y| < 0.3
  if (std::abs(C[0] + box * dir[0]) < 0.4 && std::abs(box * dir[1]) < 0.3) { s = box; surface = 1; }
  const double X = C[0] + s * dir[0], Y = s * dir[1];
  if (moving_patch && u >= 20 && u < 40 && v >= 20 && v < 40) s *= 0.6;
  return {s, bias_sigma * cell_normal(static_cast<int>(std::floor(X / 0.25)), static_cast<int>(std::floor(Y / 0.25)),
                                      surface)};
}

DepthMap network(int frame, std::mt19937& rng, double bias_sigma, double noise_sigma, bool moving_patch = false) {
  DepthMap map;
  map.frame_index = static_cast<std::uint64_t>(frame);
  map.width = kW;
  map.height = kH;
  map.source_width = kSourceW;
  map.source_height = kSourceH;
  map.scale_x = map.scale_y = 0.5F;
  map.metres.resize(static_cast<std::size_t>(kW) * kH);
  std::normal_distribution<double> noise(0, 1);
  for (int v = 0; v < kH; ++v)
    for (int u = 0; u < kW; ++u) {
      const Truth t = trace(frame, u, v, bias_sigma, moving_patch);
      map.metres[static_cast<std::size_t>(v) * kW + u] =
          static_cast<float>(t.depth * std::exp(t.bias + noise_sigma * noise(rng)));
    }
  return map;
}

DepthFilterFrame input(const DepthMap& map, int frame, bool keyframe) {
  DepthFilterFrame f;
  f.depth = &map;
  f.pose = pose_at(frame);
  f.keyframe = keyframe;
  f.K = kK;
  f.log_scale = 0;  // VO units = metres
  return f;
}

bool interior(int u, int v) { return u > 7 && v > 7 && u < kW - 8 && v < kH - 8; }

// Pixels within 6 px of a depth edge may legitimately carry the other surface for a
// few frames (re-anchoring onto a surface that is sliding over another).
bool near_edge(int frame, int u, int v) {
  const double truth = std::log(trace(frame, u, v, 0).depth);
  for (int dv = -6; dv <= 6; ++dv)
    for (int du = -6; du <= 6; ++du)
      if (std::abs(std::log(trace(frame, u + du, v + dv, 0).depth) - truth) > 0.01) return true;
  return false;
}

// Exact maps: the state (in its anchor's pixels) keeps the true depth.
void exact(const char* label, int keyframe_every, std::mt19937& rng) {
  DepthFilter filter;
  int checked = 0;
  double worst = 0;
  for (int frame = 0; frame < 12; ++frame) {
    const DepthMap map = network(frame, rng, 0, 0);
    const auto stats = filter.process(input(map, frame, frame % keyframe_every == 0));
    // Re-anchoring may put the wrong surface on a few edge pixels (gated, then reset).
    const int allowed = keyframe_every < 1000 ? stats.updated / 100 : 0;
    require(stats.gated <= allowed, std::string(label) + ": exact measurements gated");
    if (frame == 0) continue;
    const int anchor = static_cast<int>(filter.frame_index());
    for (int v = 0; v < kH; ++v)
      for (int u = 0; u < kW; ++u) {
        const std::size_t i = static_cast<std::size_t>(v) * kW + u;
        if (!interior(u, v) || filter.observations(i) < 2 || near_edge(anchor, u, v)) continue;
        worst = std::max(worst, std::abs(filter.log_depth(i) - std::log(trace(anchor, u, v, 0).depth)));
        ++checked;
      }
  }
  std::cout << "depth filter: exact, " << label << ", " << checked << " pixels, worst |d log z| " << worst << "\n";
  require(checked > 10000 && worst < 2e-3, std::string(label) + ": the state keeps the true depth");
  // Rendered into the last frame (not the anchor), the state matches that frame's truth.
  DepthFilterView view;
  filter.render(pose_at(11), view);
  int rendered = 0;
  double rendered_worst = 0;
  for (int v = 0; v < kH; ++v)
    for (int u = 0; u < kW; ++u) {
      const float metres = view.metres[static_cast<std::size_t>(v) * kW + u];
      if (!interior(u, v) || !(metres > 0) || near_edge(11, u, v)) continue;
      rendered_worst = std::max(rendered_worst, std::abs(std::log(metres / trace(11, u, v, 0).depth)));
      ++rendered;
    }
  std::cout << "  rendered into frame 11: " << rendered << " pixels, worst |d log z| " << rendered_worst << "\n";
  require(rendered > 10000 && rendered_worst < 2e-3, std::string(label) + ": render matches the frame's truth");
}
}  // namespace

int main() {
  try {
    std::mt19937 rng(11);
    exact("anchored", 1000, rng);
    exact("re-anchored every frame", 1, rng);
    g_yaw_per_frame = 0.01;  // 0.57 deg per frame
    exact("anchored, turning", 1000, rng);
    exact("re-anchored every 4th frame, turning", 4, rng);
    g_yaw_per_frame = 0;
    {  // Shared bias + independent noise: flicker drops; sigma stays calibrated.
      const double bias_sigma = 0.15, noise_sigma = 0.03;
      DepthFilterConfig config;
      config.prior_sigma = std::hypot(bias_sigma, noise_sigma);
      config.independent_fraction = noise_sigma / config.prior_sigma;
      config.bias_sigma = bias_sigma;
      DepthFilter filter(config);
      double raw = 0, fused = 0, z2 = 0, scalar2 = 0;
      int n = 0, frames = 0;
      for (int frame = 0; frame < 30; ++frame) {
        const DepthMap map = network(frame, rng, bias_sigma, noise_sigma);
        const auto stats = filter.process(input(map, frame, frame % 8 == 0));
        if (frame < 10) continue;
        if (stats.flicker_pixels > 100) {
          raw += stats.flicker_raw;
          fused += stats.flicker_fused;
          ++frames;
        }
        const int anchor = static_cast<int>(filter.frame_index());
        for (int v = 0; v < kH; v += 3)
          for (int u = 0; u < kW; u += 3) {
            const std::size_t i = static_cast<std::size_t>(v) * kW + u;
            if (!interior(u, v) || filter.observations(i) < 5) continue;
            const double error = filter.log_depth(i) - std::log(trace(anchor, u, v, 0).depth);
            const double s = filter.sigma(i);
            z2 += error * error / (s * s);
            // A scalar filter after k independent-looking updates: prior_sigma / sqrt(k).
            const double scalar = config.prior_sigma / std::sqrt(filter.observations(i));
            scalar2 += error * error / (scalar * scalar);
            ++n;
          }
      }
      require(frames > 5, "flicker measured");
      raw /= frames;
      fused /= frames;
      const double nees = std::sqrt(z2 / n), scalar_nees = std::sqrt(scalar2 / n);
      std::cout << "depth filter: flicker raw " << raw << " fused " << fused << "; normalised error rms " << nees
                << " (scalar filter would be " << scalar_nees << ")\n";
      require(fused < 0.5 * raw, "fusion halves flicker");
      require(nees > 0.6 && nees < 1.5, "fused sigma calibrated");
      require(scalar_nees > 2.0, "a scalar filter would be overconfident");
    }
    {  // A fast sideways move uncovers wall behind the box: filled from the measurement at re-anchoring.
      g_step_x = 0.15;
      DepthFilter filter;
      const DepthMap first = network(0, rng, 0, 0);
      filter.process(input(first, 0, true));
      const DepthMap second = network(1, rng, 0, 0);
      const auto stats = filter.process(input(second, 1, true));
      int wrong = 0;
      for (int v = 0; v < kH; ++v)
        for (int u = 0; u < kW; ++u) {
          const std::size_t i = static_cast<std::size_t>(v) * kW + u;
          if (filter.observations(i) == 1 && std::abs(filter.log_depth(i) - std::log(trace(1, u, v, 0).depth)) > 1e-4)
            ++wrong;
        }
      std::cout << "depth filter: fast move, " << stats.initialised << " pixels initialised, " << wrong << " wrong\n";
      require(stats.initialised > 100, "disocclusion left empty by the splat");
      require(wrong == 0, "empty pixels take the measurement");
      g_step_x = 0.02;
    }
    {  // Anchored fast move: anchor wall pixels now behind the box are occluded, not compared.
      g_step_x = 0.15;
      DepthFilter filter;
      const DepthMap first = network(0, rng, 0, 0);
      filter.process(input(first, 0, true));
      const DepthMap second = network(1, rng, 0, 0);
      const auto stats = filter.process(input(second, 1, false));
      std::cout << "depth filter: anchored fast move, " << stats.occluded << " occluded, " << stats.gated << " gated\n";
      require(stats.occluded > 100, "wall pixels behind the box are occluded");
      require(stats.gated == 0, "occluded pixels are not compared");
      g_step_x = 0.02;
    }
    {  // A scene change is gated, then reset after reset_after frames.
      DepthFilter filter;
      for (int frame = 0; frame < 5; ++frame) {
        const DepthMap map = network(frame, rng, 0, 0.01);
        filter.process(input(map, frame, true));
      }
      int resets = 0;
      for (int frame = 5; frame < 9; ++frame) {
        const DepthMap map = network(frame, rng, 0, 0.01, true);
        const auto stats = filter.process(input(map, frame, true));
        resets += stats.resets;
        if (frame < 5 + filter.config().reset_after - 1) require(stats.gated >= 300, "change gated");
      }
      const std::size_t i = static_cast<std::size_t>(30) * kW + 30;
      const double truth = std::log(trace(8, 30, 30, 0, true).depth);
      std::cout << "depth filter: resets " << resets << ", patch error " << filter.log_depth(i) - truth << "\n";
      require(resets >= 300, "patch reset");
      require(std::abs(filter.log_depth(i) - truth) < 0.05, "patch follows the new surface");
    }
    {  // Rescale and the metric round trip.
      DepthFilter filter;
      const DepthMap map = network(0, rng, 0, 0);
      filter.process(input(map, 0, true));
      const std::size_t i = static_cast<std::size_t>(60) * kW + 80;
      const float before = filter.log_depth(i);
      const double tz = filter.anchor_pose().translation[2];
      filter.rescale(2.0);
      require(filter.log_depth(i) == before, "rescale keeps the metric state");
      require(std::abs(filter.anchor_pose().translation[2] - tz / 2) < 1e-12, "rescale converts the anchor pose");
      const DepthMap metric = filter.to_depth_map(std::log(2.0), nullptr);
      require(std::abs(metric.metres[i] - map.metres[i]) < 1e-4 * map.metres[i], "metric round trip");
    }
    std::cout << "depth filter: ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << "\n";
    return 1;
  }
}
