#include "lsd_graph_view.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace slam_native {
namespace {
struct V3 {
  float x, y, z;
};

// World (camera convention: x right, y down, z forward) -> display with y up,
// after the video's clockwise display rotation; as in trajectory_view.cpp.
V3 display(const lsd::Vec3& p, int rotation) {
  double u = p.x(), v = p.y();
  if (rotation == 90) { u = -p.y(); v = p.x(); }
  else if (rotation == 180) { u = -p.x(); v = -p.y(); }
  else if (rotation == 270) { u = p.y(); v = -p.x(); }
  return {float(u), float(-v), float(p.z())};
}
}  // namespace

void LsdGraphView::draw(const lsd::Odometry& odometry, bool* open) {
  if (open && !*open) return;
  ImGui::SetNextWindowPos({960, 30}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({620, 640}, ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Pose graph", open)) {
    ImGui::End();
    return;
  }
  const auto& keyframes = odometry.keyframes();
  const auto& trajectory = odometry.trajectory();
  const auto& edges = odometry.graph().edges();
  int loops = 0, fallbacks = 0;
  for (const auto& e : edges) {
    loops += e.loop;
    fallbacks += e.fallback;
  }
  ImGui::Text("%zu keyframes, %zu edges (%d loop, %d fallback), %zu frames", keyframes.size(), edges.size(), loops,
              fallbacks, trajectory.size());
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Edges: Sim(3) constraints from direct keyframe alignment that passed the reciprocal\n"
                      "check (orange; magenta = loop candidate). Red: parent edge from frame tracking only\n"
                      "(alignment failed). Dashed grey: the current keyframe's tracked link to its parent.");
  const char* modes[] = {"Orbit", "Top (X-Z)", "Side (Z-Y)", "Front (X-Y)"};
  ImGui::SetNextItemWidth(120);
  ImGui::Combo("##view", &mode_, modes, IM_ARRAYSIZE(modes));
  ImGui::SameLine();
  if (ImGui::SmallButton("Fit")) {
    zoom_ = dolly_ = 1;
    pan_ = {};
  }
  ImGui::SameLine();
  ImGui::Checkbox("Follow", &follow_);
  if (mode_ == 0) {
    ImGui::SameLine();
    ImGui::Checkbox("Perspective", &perspective_);
  }
  ImGui::Checkbox("Semi-dense clouds", &show_clouds_);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(80);
  ImGui::SliderFloat("##cloud-size", &cloud_point_pixels_, 1.0F, 5.0F, "%.1f px");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(110);
  ImGui::SliderFloat("max sigma", &max_sigma_, 0.005F, 1.0F, "%.3f", ImGuiSliderFlags_Logarithmic);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Hide points whose inverse-depth sigma / inverse depth exceeds this (>= 1 shows\n"
                      "all). Like the paper's Fig. 2: lower = sparser but cleaner.");
  ImGui::SameLine();
  ImGui::Checkbox("colour by sigma", &color_by_sigma_);
  ImGui::Checkbox("Trajectory", &show_trajectory_);
  ImGui::SameLine();
  ImGui::Checkbox("Frustums", &show_frustums_);

  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImVec2 size = ImGui::GetContentRegionAvail();
  size.x = std::max(size.x, 50.0F);
  size.y = std::max(size.y, 50.0F);
  ImGui::InvisibleButton("##graph-canvas", size, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
  const bool hovered = ImGui::IsItemHovered();
  const ImGuiIO& io = ImGui::GetIO();
  const float wheel = hovered ? io.MouseWheel : 0.0F;
  const bool orbiting = ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left) && mode_ == 0;
  const bool panning = ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Right);
  const ImVec2 pan_pixels = panning ? io.MouseDelta : ImVec2(0, 0);
  if (orbiting) {
    yaw_ += io.MouseDelta.x * 0.01F;
    pitch_ = std::clamp(pitch_ + io.MouseDelta.y * 0.01F, -1.55F, 1.55F);
  }
  if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
    zoom_ = dolly_ = 1;
    pan_ = {};
  }
  ImDrawList* draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(origin, {origin.x + size.x, origin.y + size.y}, IM_COL32(88, 90, 95, 255));
  draw->PushClipRect(origin, {origin.x + size.x, origin.y + size.y}, true);
  if (keyframes.empty()) {
    draw->AddText({origin.x + 12, origin.y + 12}, IM_COL32(220, 220, 220, 255), "No keyframes yet.");
    draw->PopClipRect();
    ImGui::End();
    return;
  }

  const auto centre = [&](const lsd::Sim3& pose) { return display(pose.inverse().t, rotation_); };
  // Bounds: keyframe centres and the trajectory (fixed under orbiting).
  V3 low{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
  V3 high{-low.x, -low.y, -low.z};
  const auto extend = [&](V3 p) {
    low = {std::min(low.x, p.x), std::min(low.y, p.y), std::min(low.z, p.z)};
    high = {std::max(high.x, p.x), std::max(high.y, p.y), std::max(high.z, p.z)};
  };
  for (const auto& kf : keyframes) extend(centre(kf->pose));
  for (const auto& t : trajectory) extend(centre(t.pose));
  V3 pivot{0.5F * (low.x + high.x), 0.5F * (low.y + high.y), 0.5F * (low.z + high.z)};
  // Radius: at least the first keyframe's mean depth, so a still camera still frames its cloud.
  float radius = 0.5F / static_cast<float>(std::max(keyframes.front()->mean_idepth, 1e-6));
  const auto grow = [&](V3 p) {
    radius = std::max(radius, std::sqrt((p.x - pivot.x) * (p.x - pivot.x) + (p.y - pivot.y) * (p.y - pivot.y) +
                                        (p.z - pivot.z) * (p.z - pivot.z)));
  };
  for (const auto& kf : keyframes) grow(centre(kf->pose));
  for (const auto& t : trajectory) grow(centre(t.pose));
  const lsd::Sim3 current_pose = odometry.last().pose;
  if (follow_) pivot = centre(current_pose);

  float yaw = yaw_, pitch = pitch_;
  if (mode_ == 1) { yaw = 0; pitch = 1.5707963F; }
  if (mode_ == 2) { yaw = 1.5707963F; pitch = 0; }
  if (mode_ == 3) { yaw = 0; pitch = 0; }
  const float cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(pitch), sp = std::sin(pitch);
  const V3 right{cy, 0, -sy};
  const V3 up{sp * sy, cp, sp * cy};
  const V3 toward{-sy * cp, sp, -cy * cp};
  const bool perspective = mode_ == 0 && perspective_;
  constexpr float kHalfFov = 0.5F * 50.0F * 3.14159265F / 180.0F;
  const float focal = 0.5F * std::min(size.x, size.y) / std::tan(kHalfFov);
  if (wheel != 0) {
    if (perspective) dolly_ = std::clamp(dolly_ * std::pow(0.85F, wheel), 0.02F, 20.0F);
    else zoom_ = std::clamp(zoom_ * std::pow(1.15F, wheel), 0.05F, 200.0F);
  }
  const float eye_distance = dolly_ * 1.1F * radius / std::sin(kHalfFov);
  const float scale = zoom_ * 0.85F * std::min(size.x, size.y) / (2.0F * radius);
  const float world_per_pixel = perspective ? eye_distance / focal : 1.0F / scale;
  pan_[0] += (-pan_pixels.x * right.x + pan_pixels.y * up.x) * world_per_pixel;
  pan_[1] += (-pan_pixels.x * right.y + pan_pixels.y * up.y) * world_per_pixel;
  pan_[2] += (-pan_pixels.x * right.z + pan_pixels.y * up.z) * world_per_pixel;
  pivot = {pivot.x + pan_[0], pivot.y + pan_[1], pivot.z + pan_[2]};
  const ImVec2 middle(origin.x + size.x * 0.5F, origin.y + size.y * 0.5F);
  const auto dot = [](V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; };
  struct Eye {
    float r, u, depth;
  };
  const auto eye = [&](V3 v) {
    const V3 d{v.x - pivot.x, v.y - pivot.y, v.z - pivot.z};
    return Eye{dot(right, d), dot(up, d), eye_distance - dot(toward, d)};
  };
  const float near_plane = 0.01F * eye_distance;
  const auto to_screen = [&](const Eye& e) {
    const float s = perspective ? focal / e.depth : scale;
    return ImVec2(middle.x + e.r * s, middle.y - e.u * s);
  };
  const auto line = [&](V3 a, V3 b, ImU32 color, float thickness) {
    Eye ea = eye(a), eb = eye(b);
    if (perspective) {
      if (ea.depth < near_plane && eb.depth < near_plane) return;
      const auto clip = [&](Eye& behind, const Eye& front) {
        const float t = (near_plane - behind.depth) / (front.depth - behind.depth);
        behind = {behind.r + t * (front.r - behind.r), behind.u + t * (front.u - behind.u), near_plane};
      };
      if (ea.depth < near_plane) clip(ea, eb);
      else if (eb.depth < near_plane) clip(eb, ea);
    }
    draw->AddLine(to_screen(ea), to_screen(eb), color, thickness);
  };

  // Floor grid through the first keyframe (y = 0) and its axes.
  {
    const float raw = radius / 3.0F;
    const float decade = std::pow(10.0F, std::floor(std::log10(raw)));
    const float spacing = raw / decade < 2 ? decade : raw / decade < 5 ? 2 * decade : 5 * decade;
    const int half = static_cast<int>(std::ceil(2.0F * radius / spacing)) + 1;
    const float gx = std::round(pivot.x / spacing) * spacing, gz = std::round(pivot.z / spacing) * spacing;
    for (int k = -half; k <= half; ++k) {
      const float x = gx + k * spacing, z = gz + k * spacing;
      line({x, 0, gz - half * spacing}, {x, 0, gz + half * spacing}, IM_COL32(0, 0, 0, 110), 1.0F);
      line({gx - half * spacing, 0, z}, {gx + half * spacing, 0, z}, IM_COL32(0, 0, 0, 110), 1.0F);
    }
    const float axis = 0.3F * radius;
    line({0, 0, 0}, {axis, 0, 0}, IM_COL32(220, 60, 60, 255), 2.0F);
    line({0, 0, 0}, {0, axis, 0}, IM_COL32(60, 200, 60, 255), 2.0F);
    line({0, 0, 0}, {0, 0, axis}, IM_COL32(60, 100, 235, 255), 2.0F);
  }

  if (show_clouds_) {
    std::vector<DenseCloudDraw> clouds;
    for (const auto& kf : keyframes) {
      DenseCloudDraw d;
      d.key = static_cast<std::uint64_t>(kf->id);
      d.version = kf->version;
      d.points = &kf->cloud;
      d.colors = &kf->cloud_colors;
      d.sigma = &kf->cloud_sigma;
      d.max_sigma = max_sigma_ >= 1.0F ? std::numeric_limits<float>::infinity() : max_sigma_;
      d.color_by_confidence = color_by_sigma_;
      d.sigma_reference = 0.05F;
      const lsd::Sim3 world_from_camera = kf->pose.inverse();
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) d.rotation[r * 3 + c] = static_cast<float>(world_from_camera.R(r, c));
      d.origin = {static_cast<float>(world_from_camera.t.x()), static_cast<float>(world_from_camera.t.y()),
                  static_cast<float>(world_from_camera.t.z())};
      d.units_per_point_unit = static_cast<float>(world_from_camera.s);  // keyframe depth units -> world
      clouds.push_back(d);
    }
    PointCloudCamera camera;
    camera.pivot = {pivot.x, pivot.y, pivot.z};
    camera.right = {right.x, right.y, right.z};
    camera.up = {up.x, up.y, up.z};
    camera.toward = {toward.x, toward.y, toward.z};
    camera.eye_distance = eye_distance;
    camera.perspective = perspective;
    camera.focal = focal;
    camera.scale = scale;
    camera.near_plane = near_plane;
    camera.rotation = rotation_;
    renderer_.render(draw, {origin.x, origin.y}, {origin.x + size.x, origin.y + size.y},
                     io.DisplayFramebufferScale.x, camera, nullptr, {}, nullptr, {}, &clouds, cloud_point_pixels_,
                     generation_);
  }

  if (show_trajectory_) {
    for (std::size_t i = 1; i < trajectory.size(); ++i)
      line(centre(trajectory[i - 1].pose), centre(trajectory[i].pose), IM_COL32(80, 200, 255, 200), 1.5F);
  }
  // Graph edges between keyframe centres.
  for (const auto& e : edges) {
    const ImU32 color = e.fallback ? IM_COL32(230, 60, 50, 255) : e.loop ? IM_COL32(255, 60, 230, 255) :
                        IM_COL32(255, 170, 60, 255);
    line(centre(keyframes[e.i]->pose), centre(keyframes[e.j]->pose), color, e.loop ? 2.5F : 1.5F);
  }
  if (const auto* current = odometry.current_keyframe(); current && current->parent >= 0) {
    // Not in the graph yet: dashed link to its parent.
    const V3 a = centre(keyframes[current->parent]->pose), b = centre(current->pose);
    for (int k = 0; k < 10; k += 2) {
      const float t0 = k / 10.0F, t1 = (k + 1) / 10.0F;
      line({a.x + t0 * (b.x - a.x), a.y + t0 * (b.y - a.y), a.z + t0 * (b.z - a.z)},
           {a.x + t1 * (b.x - a.x), a.y + t1 * (b.y - a.y), a.z + t1 * (b.z - a.z)}, IM_COL32(200, 200, 200, 255), 1.5F);
    }
  }
  // Frustums: image corners at a depth proportional to the scene size.
  const lsd::Camera& cam = odometry.camera();
  const auto frustum = [&](const lsd::Sim3& pose, float depth, ImU32 color, float thickness) {
    const lsd::Sim3 inv = pose.inverse();
    const V3 apex = display(inv.t, rotation_);
    std::array<V3, 4> c;
    const double corners[4][2] = {{0, 0}, {cam.width - 1.0, 0}, {cam.width - 1.0, cam.height - 1.0}, {0, cam.height - 1.0}};
    // Depth in world units at the camera's own scale.
    for (int k = 0; k < 4; ++k)
      c[k] = display(inv * (cam.ray(corners[k][0], corners[k][1]) * (depth / inv.s)), rotation_);
    for (int k = 0; k < 4; ++k) {
      line(apex, c[k], color, thickness);
      line(c[k], c[(k + 1) % 4], color, thickness);
    }
  };
  const float frustum_depth = 0.06F * radius;
  if (show_frustums_) {
    const int current = odometry.last().keyframe;
    for (const auto& kf : keyframes)
      frustum(kf->pose, frustum_depth, kf->id == current ? IM_COL32(255, 230, 80, 255) : IM_COL32(255, 170, 60, 200),
              kf->id == current ? 2.0F : 1.0F);
  }
  frustum(current_pose, 1.4F * frustum_depth, IM_COL32_WHITE, 2.0F);
  draw->PopClipRect();
  ImGui::End();
}

}  // namespace slam_native
