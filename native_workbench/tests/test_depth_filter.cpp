// DepthFilter on a synthetic scene: a background wall and a nearer box, a
// camera moving sideways and forward, network maps with a per-surface bias
// shared between frames plus independent per-frame noise.
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

double g_step_x = 0.02;  // camera centre (step_x k, 0, 0.03 k)

Pose pose_at(int frame) {
  Pose pose;  // R = I: x = X - C
  pose.translation = {-g_step_x * frame, 0, -0.03 * frame};
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
  const double rx = (x - kK.cx) / kK.fx, ry = (y - kK.cy) / kK.fy;
  const double cx = g_step_x * frame, cz = 0.03 * frame;
  double s = 5.0 - cz;  // wall at Z = 5
  int surface = 0;
  const double box = 3.0 - cz;  // box face at Z = 3, |X| < 0.4, |Y| < 0.3
  const double bx = cx + box * rx, by = box * ry;
  if (std::abs(bx) < 0.4 && std::abs(by) < 0.3) { s = box; surface = 1; }
  const double X = cx + s * rx, Y = s * ry;
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

DepthFilterFrame input(const DepthMap& map, int frame) {
  DepthFilterFrame f;
  f.depth = &map;
  f.pose = pose_at(frame);
  f.K = kK;
  f.log_scale = 0;  // VO units = metres
  return f;
}

bool interior(int u, int v) { return u > 4 && v > 4 && u < kW - 5 && v < kH - 5; }
}  // namespace

int main() {
  try {
    std::mt19937 rng(11);
    {  // Exact maps: the warp carries the true depth; the update changes nothing.
      DepthFilter filter;
      int checked = 0;
      double worst = 0;
      for (int frame = 0; frame < 6; ++frame) {
        const DepthMap map = network(frame, rng, 0, 0);
        const auto stats = filter.process(input(map, frame));
        if (frame == 0) continue;
        require(stats.predicted && stats.carried > kW * kH / 2, "state carried forward");
        require(stats.gated == 0, "no exact measurement gated");
        for (int v = 0; v < kH; ++v)
          for (int u = 0; u < kW; ++u) {
            const std::size_t i = static_cast<std::size_t>(v) * kW + u;
            if (!interior(u, v) || filter.observations(i) < 2) continue;
            const double truth = std::log(trace(frame, u, v, 0).depth);
            // Pixels near the box edge may legitimately carry the other surface.
            bool near_edge = false;
            for (int dv = -3; dv <= 3; ++dv)
              for (int du = -3; du <= 3; ++du)
                near_edge |= std::abs(std::log(trace(frame, u + du, v + dv, 0).depth) - truth) > 0.01;
            if (near_edge) continue;
            worst = std::max(worst, std::abs(filter.log_depth(i) - truth));
            ++checked;
          }
      }
      std::cout << "depth filter: exact warp, " << checked << " pixels, worst |d log z| " << worst << "\n";
      require(checked > 10000 && worst < 1e-3, "warp preserves depth");
    }
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
        const auto stats = filter.process(input(map, frame));
        if (frame < 10) continue;
        raw += stats.flicker_raw;
        fused += stats.flicker_fused;
        ++frames;
        for (int v = 0; v < kH; v += 3)
          for (int u = 0; u < kW; u += 3) {
            const std::size_t i = static_cast<std::size_t>(v) * kW + u;
            if (!interior(u, v) || filter.observations(i) < 5) continue;
            const double error = filter.log_depth(i) - std::log(trace(frame, u, v, 0).depth);
            const double s = filter.sigma(i);
            z2 += error * error / (s * s);
            // A scalar filter after k independent-looking updates: prior_sigma / sqrt(k).
            const double scalar = config.prior_sigma / std::sqrt(filter.observations(i));
            scalar2 += error * error / (scalar * scalar);
            ++n;
          }
      }
      raw /= frames;
      fused /= frames;
      const double nees = std::sqrt(z2 / n), scalar_nees = std::sqrt(scalar2 / n);
      std::cout << "depth filter: flicker raw " << raw << " fused " << fused << "; normalised error rms " << nees
                << " (scalar filter would be " << scalar_nees << ")\n";
      require(fused < 0.4 * raw, "fusion reduces flicker");
      require(nees > 0.6 && nees < 1.5, "fused sigma calibrated");
      require(scalar_nees > 2.0, "a scalar filter would be overconfident");
    }
    {  // A fast sideways move uncovers wall behind the box: filled from the measurement.
      g_step_x = 0.15;
      DepthFilter filter;
      const DepthMap first = network(0, rng, 0, 0);
      filter.process(input(first, 0));
      const DepthMap second = network(1, rng, 0, 0);
      const auto stats = filter.process(input(second, 1));
      int wrong = 0;
      for (int v = 0; v < kH; ++v)
        for (int u = 0; u < kW; ++u) {
          const std::size_t i = static_cast<std::size_t>(v) * kW + u;
          if (filter.observations(i) == 1 && std::abs(filter.log_depth(i) - std::log(trace(1, u, v, 0).depth)) > 1e-4)
            ++wrong;
        }
      std::cout << "depth filter: fast move, " << stats.initialised << " pixels initialised, " << wrong << " wrong\n";
      require(stats.initialised > 100, "disocclusion left empty by the warp");
      require(wrong == 0, "empty pixels take the measurement");
      g_step_x = 0.02;
    }
    {  // A scene change is gated, then reset after reset_after frames.
      DepthFilter filter;
      for (int frame = 0; frame < 5; ++frame) {
        const DepthMap map = network(frame, rng, 0, 0.01);
        filter.process(input(map, frame));
      }
      int resets = 0;
      for (int frame = 5; frame < 9; ++frame) {
        const DepthMap map = network(frame, rng, 0, 0.01, true);
        const auto stats = filter.process(input(map, frame));
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
      filter.process(input(map, 0));
      const std::size_t i = static_cast<std::size_t>(60) * kW + 80;
      const float before = filter.log_depth(i);
      filter.rescale(2.0);
      require(std::abs(filter.log_depth(i) - (before - std::log(2.0F))) < 1e-5, "rescale shifts log depth");
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
