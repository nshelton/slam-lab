// Depth sampling at tracks and keyframe depth clouds, on synthetic depth maps
// (DEPTH_INTEGRATION.md). No engine, video or GPU work.
#include "slam_native/depth_sampling.hpp"
#include "slam_native/keyframe_depth.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

using namespace slam_native;
namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

// A 1920x1080 source shown with `rotation`, letterboxed into a 518x294 or
// 294x518 output, every pixel `metres` deep.
DepthMap make_map(int rotation, float metres) {
  DepthMap map;
  map.source_width = 1920;
  map.source_height = 1080;
  map.rotation = rotation;
  const bool sideways = rotation == 90 || rotation == 270;
  map.width = sideways ? 294 : 518;
  map.height = sideways ? 518 : 294;
  const float upright_w = sideways ? 1080.0F : 1920.0F, upright_h = sideways ? 1920.0F : 1080.0F;
  const float scale = std::min(map.width / upright_w, map.height / upright_h);
  map.scale_x = map.scale_y = scale;
  map.offset_x = 0.5F * (map.width - upright_w * scale);
  map.offset_y = 0.5F * (map.height - upright_h * scale);
  map.metres.assign(static_cast<std::size_t>(map.width) * map.height, 0.0F);
  for (int v = 0; v < map.height; ++v)
    for (int u = 0; u < map.width; ++u) {
      const bool inside = u + 0.5F > map.offset_x && u + 0.5F < map.offset_x + upright_w * scale &&
                          v + 0.5F > map.offset_y && v + 0.5F < map.offset_y + upright_h * scale;
      if (inside) map.metres[static_cast<std::size_t>(v) * map.width + u] = metres;
    }
  return map;
}

void grid_inverts_the_map_transform() {
  for (int rotation : {0, 90, 180, 270}) {
    KeyframeDepthStore store;
    const DepthMap map = make_map(rotation, 5);
    const auto& grid = store.grid(map);
    require(grid.size() > 5000, "grid too small for rotation " + std::to_string(rotation));
    for (const auto& p : grid) {
      require(p.x >= 0 && p.y >= 0 && p.x <= 1919 && p.y <= 1079, "grid point outside the source image");
      float u = 0, v = 0;
      map.to_output(p.x, p.y, u, v);
      // KeyframeDepthConfig::stride 4: cells at 2, 6, 10, ...
      require(std::abs(u - std::round(u)) < 1e-3F && std::abs(v - std::round(v)) < 1e-3F &&
              (static_cast<int>(std::lround(u)) - 2) % 4 == 0 && (static_cast<int>(std::lround(v)) - 2) % 4 == 0,
              "grid does not invert DepthMap::to_output (rotation " + std::to_string(rotation) + ")");
    }
  }
  std::cout << "keyframe grid inverts the depth map transform for all rotations\n";
}

void samples_follow_the_policy() {
  DepthMap map = make_map(0, 5);
  map.frame_index = 7;
  DepthSamplingConfig config;
  config.max_depth_m = 20;
  const DepthSample flat = sample_depth(map, 960, 540, config);
  require(flat.valid() && !flat.edge && std::abs(flat.metres - 5) < 1e-4F &&
          std::abs(flat.sigma - 0.75F) < 1e-4F, "flat sample");
  // A depth step (foreground 2 m left of a column of output pixels): the edge
  // reports the foreground with an inflated sigma, or is dropped.
  float u = 0, v = 0;
  map.to_output(960, 540, u, v);
  const int column = static_cast<int>(std::lround(u));
  for (int row = 0; row < map.height; ++row)
    for (int c = 0; c < column; ++c)
      if (map.metres[static_cast<std::size_t>(row) * map.width + c] > 0)
        map.metres[static_cast<std::size_t>(row) * map.width + c] = 2;
  const DepthSample edge = sample_depth(map, 960, 540, config);
  require(edge.valid() && edge.edge && edge.metres == 2 && std::abs(edge.sigma - 0.9F) < 1e-4F, "edge sample");
  config.drop_edges = true;
  require(!sample_depth(map, 960, 540, config).valid(), "dropped edge sample");
  config.drop_edges = false;
  // Clamp: depth at the model's maximum is unknown.
  DepthMap far = make_map(0, 20);
  require(!sample_depth(far, 960, 540, config).valid(), "clamped sample");
  // Letterbox: a 1920x1080 source in a 518x294 output has no padding rows,
  // but a portrait display does.
  const DepthMap portrait = make_map(90, 5);
  require(sample_depth(portrait, 960, 540, config).valid(), "portrait centre");

  TrackedFrame frame{7, 0, 1920, 1080, {}, 0, {}};
  frame.observations = {{1, 960, 540}, {2, 100, 100}};
  int edges = 0;
  DepthMap right = make_map(0, 5);
  right.frame_index = 7;
  require(attach_depth(right, frame, config, &edges) == 2 && edges == 0 && frame.observations[0].depth_m == 5 &&
          frame.observations[1].depth_sigma_m > 0, "attach_depth");
  DepthMap wrong = make_map(0, 5);
  wrong.frame_index = 8;
  bool threw = false;
  try {
    attach_depth(wrong, frame, config);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "attach_depth must refuse a map of another frame");
  std::cout << "depth sampling: flat, edge (foreground, 3x sigma), dropped edge, clamp, frame check\n";
}

void clouds_back_project_and_drop_edges() {
  KeyframeDepthStore store;
  const auto K = CameraIntrinsics::from_horizontal_fov(1920, 1080, 70);
  DepthMap plane = make_map(0, 4);
  const std::vector<std::array<std::uint8_t, 3>> colors(store.grid(plane).size(), {10, 20, 30});
  KeyframeDepthInput input;
  input.frame_index = 11;
  input.log_scale = std::log(2.5);
  const std::size_t kept = store.add(input, plane, colors, K, 0);
  require(kept == store.grid(plane).size(), "flat plane: every grid point kept");
  const auto& cloud = store.clouds().back();
  bool centre = false;
  for (const auto& p : cloud.points) {
    require(std::abs(p[2] - 4) < 1e-5F, "plane points at z = 4");
    centre |= std::abs(p[0]) < 0.05F && std::abs(p[1]) < 0.05F;
  }
  require(centre && cloud.colors.front()[2] == 30 && std::abs(cloud.metres_per_unit - 2.5) < 1e-9,
          "cloud contents");
  // A step: pixels next to it are flying pixels and are dropped.
  DepthMap step = plane;
  for (int v = 0; v < step.height; ++v)
    for (int u = 0; u < step.width / 2; ++u)
      if (step.metres[static_cast<std::size_t>(v) * step.width + u] > 0)
        step.metres[static_cast<std::size_t>(v) * step.width + u] = 2;
  input.frame_index = 12;
  input.log_scale = std::numeric_limits<double>::quiet_NaN();
  const std::size_t with_step = store.add(input, step, colors, K, 0);
  require(with_step < kept && with_step > kept * 9 / 10, "edge pixels dropped, the rest kept");
  const auto generation = store.generation();
  store.clear();
  require(store.clouds().empty() && store.generation() != generation, "clear bumps the generation");
  std::cout << "keyframe clouds: plane back-projected at its depth, step edge dropped (" << kept - with_step
            << " of " << kept << ")\n";
}
}  // namespace

// The scale grid corrects per pixel; settle() re-applies a new one.
void grid_scales_points() {
  KeyframeDepthStore store;
  const auto K = CameraIntrinsics::from_horizontal_fov(1920, 1080, 70);
  const DepthMap plane = make_map(0, 4);
  ScaleGrid grid{};
  for (int y = 0; y < kScaleGridY; ++y)
    for (int x = 0; x < kScaleGridX; ++x) grid[static_cast<std::size_t>(y * kScaleGridX + x)] = x == 0 ? 0.2 : 0.0;
  KeyframeDepthInput input;
  input.frame_index = 5;
  input.log_scale = 0;
  input.grid = &grid;
  store.add(input, plane, {}, K, 0);
  const auto& cloud = store.clouds().back();
  // In VO units (points / metres_per_unit): depth / exp(L). Left edge L = 0.2, right L = 0.
  float lowest = 10, highest = 0;
  for (std::size_t i = 0; i < cloud.points.size(); ++i) {
    const double L = scale_grid_at(grid, 1920, 1080, cloud.pixels[i][0], cloud.pixels[i][1]);
    require(std::abs(cloud.points[i][2] - 4 * std::exp(-L)) < 1e-4, "grid applied per pixel");
    lowest = std::min(lowest, cloud.points[i][2]);
    highest = std::max(highest, cloud.points[i][2]);
  }
  require(lowest < 3.4F && std::abs(highest - 4) < 1e-4F, "grid varies across the image");
  const auto version = cloud.version;
  require(store.settle(5, std::log(2.0), nullptr) && store.clouds().back().version != version &&
          std::abs(store.clouds().back().points[0][2] - 4) < 1e-4F &&
          std::abs(store.clouds().back().metres_per_unit - 2) < 1e-9, "settle re-applies the scale");
  std::cout << "keyframe clouds: per-pixel scale grid, settle\n";
}

// Two views of a plane confirm each other; a floating patch that the other
// view sees through is hidden; the plane behind an occluder is not penalised.
void multi_view_check() {
  KeyframeDepthStore store;
  const auto K = CameraIntrinsics::from_horizontal_fov(1920, 1080, 70);
  std::map<std::uint64_t, Pose> poses;
  const auto pose_of = [&](std::uint64_t f) -> std::optional<Pose> {
    auto it = poses.find(f);
    return it == poses.end() ? std::nullopt : std::optional<Pose>(it->second);
  };
  KeyframeDepthInput a;
  a.frame_index = 1;
  a.log_scale = 0;  // metres = units
  poses[1] = a.pose;
  store.add(a, make_map(0, 4), {}, K, 0, pose_of);
  KeyframeDepthInput b = a;
  b.frame_index = 2;
  b.pose.translation = {-0.2, 0, 0};  // camera 0.2 m to the right
  poses[2] = b.pose;
  DepthMap floating = make_map(0, 4);
  int patch = 0;  // a 3 m patch in front of the 4 m plane, centre of the view
  for (int v = 120; v < 170; ++v)
    for (int u = 230; u < 290; ++u) {
      floating.metres[static_cast<std::size_t>(v) * floating.width + u] = 3;
      ++patch;
    }
  store.add(b, floating, {}, K, 0, pose_of);
  const auto& first = store.clouds()[0];
  const auto& second = store.clouds()[1];
  std::size_t first_hidden = 0, second_hidden = 0, second_agree = 0, patch_points = 0, patch_hidden = 0;
  for (std::size_t i = 0; i < first.shown.size(); ++i) first_hidden += !first.shown[i];
  for (std::size_t i = 0; i < second.shown.size(); ++i) {
    second_hidden += !second.shown[i];
    second_agree += second.agree[i] > 0;
    if (std::abs(second.raw[i][2] - 3) < 1e-4F) {
      ++patch_points;
      patch_hidden += !second.shown[i];
    }
  }
  std::cout << "multi-view: first " << first_hidden << " hidden of " << first.shown.size() << "; second "
            << second_hidden << " hidden, " << second_agree << " confirmed; floating patch " << patch_hidden << "/"
            << patch_points << " hidden\n";
  require(second_agree > second.shown.size() / 2, "the plane must be confirmed by the other view");
  require(patch_points > 50 && patch_hidden * 10 >= patch_points * 9, "the floating patch must be hidden");
  require(first_hidden * 100 <= first.shown.size(), "occluded plane points must not be hidden");
  require(second_hidden <= patch_points + second.shown.size() / 100, "only the patch is hidden");
}

int main() {
  try {
    grid_inverts_the_map_transform();
    samples_follow_the_policy();
    clouds_back_project_and_drop_edges();
    grid_scales_points();
    multi_view_check();
    std::cout << "depth checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
