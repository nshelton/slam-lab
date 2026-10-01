#include "slam_native/trajectory_view.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>

namespace slam_native {
namespace {
struct V3 {
  float x, y, z;
};

// World (camera convention: x right, y down, z forward) -> display with y up,
// after the video's clockwise display rotation (image right/down become
// rotated sensor axes).
V3 display(double x, double y, double z, int rotation) {
  double u = x, v = y;
  if (rotation == 90) { u = -y; v = x; }
  else if (rotation == 180) { u = -x; v = -y; }
  else if (rotation == 270) { u = y; v = -x; }
  return {float(u), float(-v), float(z)};
}

// Camera -> world direction of a camera-frame vector.
std::array<double, 3> to_world(const Pose& pose, double x, double y, double z) {
  const auto& R = pose.rotation;  // world -> camera, row-major; R^T maps camera -> world
  return {R[0] * x + R[3] * y + R[6] * z, R[1] * x + R[4] * y + R[7] * z, R[2] * x + R[5] * y + R[8] * z};
}

ImU32 segment_color(int segment, int alpha = 255) {
  static constexpr std::array<ImU32, 6> palette{IM_COL32(80, 200, 255, 255), IM_COL32(255, 170, 60, 255),
      IM_COL32(120, 230, 120, 255), IM_COL32(240, 110, 200, 255), IM_COL32(250, 230, 90, 255),
      IM_COL32(170, 150, 255, 255)};
  const ImU32 color = palette[static_cast<std::size_t>(segment) % palette.size()];
  return (color & 0x00FFFFFFU) | (static_cast<ImU32>(alpha) << 24U);
}
}  // namespace

void TrajectoryView::reset() {
  samples_.clear();
  local_.clear();
  outside_.clear();
}

void TrajectoryView::update(const VisualOdometry& odometry) {
  const std::size_t stable = std::min(odometry.stable_prefix(), samples_.size());
  if (odometry.trajectory_size() < samples_.size()) samples_.clear();  // odometry was reset
  samples_.resize(std::min(samples_.size(), stable));
  auto tail = odometry.trajectory(samples_.size());
  samples_.insert(samples_.end(), tail.begin(), tail.end());
  local_.clear();
  outside_.clear();
  for (auto& point : odometry.map()) (point.local ? local_ : outside_).push_back(point);
}

void TrajectoryView::draw(const VisualOdometry& odometry, bool* open, int frame_width, int frame_height) {
  if (open && !*open) return;
  ImGui::SetNextWindowPos({1090, 436}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({340, 455}, ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Camera trajectory", open)) {
    ImGui::End();
    return;
  }
  const auto& last = odometry.last();
  const ImVec4 state_color = last.state == OdometryState::tracking ? ImVec4(0.4F, 0.9F, 0.4F, 1) :
      last.state == OdometryState::coasting ? ImVec4(1, 0.6F, 0.2F, 1) :
      last.state == OdometryState::lost ? ImVec4(1, 0.4F, 0.3F, 1) : ImVec4(1, 0.8F, 0.3F, 1);
  ImGui::TextColored(state_color, "%s", to_string(last.state));
  ImGui::SameLine();
  ImGui::Text("segment %d  |  %.1f ms", last.segment, last.ms);
  ImGui::Text("Inliers %d / %d  |  keyframes %zu", last.inliers, last.correspondences, last.keyframes);
  // Local: the landmarks the odometry tracks and refines now; the rest are
  // kept and can be found again by descriptor.
  ImGui::Text("Map: %zu landmarks, %zu in the local map", local_.size() + outside_.size(), local_.size());
  if (last.has_pose && !last.predicted) ImGui::Text("Median reprojection %.2f px", last.median_reprojection_px);
  if (last.has_pose) ImGui::Text("Pose confidence %.2f  (sigma %.1f px)", last.confidence, last.pose_sigma_px);
  ImGui::PushTextWrapPos(0);
  if (!last.event.empty()) ImGui::TextDisabled("%s", last.event.c_str());
  else ImGui::TextDisabled("Monocular: scale is arbitrary and differs between segments.");
  ImGui::PopTextWrapPos();

  const char* modes[] = {"Orbit", "Top (X-Z)", "Side (Z-Y)", "Front (X-Y)"};
  ImGui::SetNextItemWidth(120);
  ImGui::Combo("##view", &mode_, modes, IM_ARRAYSIZE(modes));
  ImGui::SameLine();
  if (ImGui::SmallButton("Fit")) {
    zoom_ = dolly_ = 1;
    pan_ = {};
  }
  ImGui::Checkbox("Map points", &show_points_);
  ImGui::SameLine();
  ImGui::Checkbox("Not local", &show_outside_);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Landmarks outside the local map: no live track and not seen by the recent keyframes.\n"
                      "They stay in the map and are found again by descriptor. Local points are brighter.");
  ImGui::SameLine();
  ImGui::Checkbox("Follow", &follow_);
  ImGui::SameLine();
  ImGui::Checkbox("All segments", &all_segments_);
  if (mode_ == 0) {
    ImGui::SameLine();
    ImGui::Checkbox("Perspective", &perspective_);
  }
  if (show_points_) {
    const char* colors[] = {"Plain", "Image colour", "Age"};
    ImGui::SetNextItemWidth(110);
    ImGui::Combo("##colour", &color_mode_, colors, IM_ARRAYSIZE(colors));
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("Age: frames since the landmark's first keyframe.\n"
                        "Yellow: new; magenta: half the age span; blue: the age span or older.");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    ImGui::SliderFloat("Point size", &point_size_, 0.001F, 0.2F, "%.3f", ImGuiSliderFlags_Logarithmic);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip("World-space diameter; 1 = the first keyframe's median scene depth.");
    if (show_outside_) {
      ImGui::SameLine();
      ImGui::Checkbox("Dim not local", &dim_outside_);
    }
    if (color_mode_ == 2) {
      ImGui::SetNextItemWidth(110);
      ImGui::SliderFloat("Age span", &age_span_, 10.0F, 10000.0F, "%.0f frames", ImGuiSliderFlags_Logarithmic);
    }
  }

  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImVec2 size = ImGui::GetContentRegionAvail();
  size.x = std::max(size.x, 50.0F);
  size.y = std::max(size.y, 50.0F);
  ImGui::InvisibleButton("##trajectory-canvas", size,
                         ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
  const bool hovered = ImGui::IsItemHovered();
  const ImGuiIO& io = ImGui::GetIO();
  // Input is gathered here and applied once the eye basis is known below.
  const float wheel = hovered ? io.MouseWheel : 0.0F;
  const bool orbiting = ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left) && mode_ == 0;
  const bool panning = ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Right);
  const ImVec2 pan_pixels = panning ? io.MouseDelta : ImVec2(0, 0);
  if (orbiting) {
    yaw_ += io.MouseDelta.x * 0.01F;
    pitch_ = std::clamp(pitch_ + io.MouseDelta.y * 0.01F, -1.55F, 1.55F);
  }
  if (wheel != 0 || orbiting || panning) last_interaction_ = ImGui::GetTime();
  if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
    zoom_ = dolly_ = 1;
    pan_ = {};
  }
  const auto display = [&](const auto& p) { return slam_native::display(p[0], p[1], p[2], rotation_); };
  ImDrawList* draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(origin, {origin.x + size.x, origin.y + size.y}, IM_COL32(88, 90, 95, 255));
  draw->PushClipRect(origin, {origin.x + size.x, origin.y + size.y}, true);

  const int segment = samples_.empty() ? last.segment : samples_.back().segment;
  const auto visible = [&](const TrajectorySample& s) { return all_segments_ || s.segment == segment; };

  // Everything below is in display space: x right, y up, z the first
  // keyframe's forward (see display()). Note display space is a reflection of
  // the right-handed camera frame (y flipped).
  // Pivot and size come from the visible trajectory in 3D, so they do not
  // change with the viewing angle: orbiting moves the eye on a sphere.
  V3 low{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
  V3 high{-low.x, -low.y, -low.z};
  const TrajectorySample* current = nullptr;
  for (const auto& s : samples_) {
    if (!visible(s)) continue;
    const V3 p = display(s.pose.center());
    low = {std::min(low.x, p.x), std::min(low.y, p.y), std::min(low.z, p.z)};
    high = {std::max(high.x, p.x), std::max(high.y, p.y), std::max(high.z, p.z)};
    current = &s;
  }
  if (!current) {
    draw->AddText({origin.x + 12, origin.y + 12}, IM_COL32(200, 200, 200, 255),
                  "No camera poses yet: move the camera sideways/forward to initialize.");
    draw->PopClipRect();
    ImGui::End();
    return;
  }
  V3 pivot{0.5F * (low.x + high.x), 0.5F * (low.y + high.y), 0.5F * (low.z + high.z)};
  float radius = 1e-3F;
  for (const auto& s : samples_) {
    if (!visible(s)) continue;
    const V3 p = display(s.pose.center());
    radius = std::max(radius, std::sqrt((p.x - pivot.x) * (p.x - pivot.x) + (p.y - pivot.y) * (p.y - pivot.y) +
                                        (p.z - pivot.z) * (p.z - pivot.z)));
  }
  if (follow_) pivot = display(current->pose.center());

  float yaw = yaw_, pitch = pitch_;
  if (mode_ == 1) { yaw = 0; pitch = 1.5707963F; }
  if (mode_ == 2) { yaw = 1.5707963F; pitch = 0; }
  if (mode_ == 3) { yaw = 0; pitch = 0; }
  const float cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(pitch), sp = std::sin(pitch);
  // Orthonormal eye basis (display space): screen right, screen up, and the
  // direction from the pivot toward the eye. In the right-handed camera frame
  // toward = right x up; display space flips y, which negates the cross product.
  const V3 right{cy, 0, -sy};
  const V3 up{sp * sy, cp, sp * cy};
  const V3 toward{-sy * cp, sp, -cy * cp};
  const bool perspective = mode_ == 0 && perspective_;
  // Perspective: a fixed lens (vertical FOV over the canvas's smaller side);
  // the wheel dollies the eye toward the orbit point instead of zooming the
  // lens. Orthographic: the wheel magnifies.
  constexpr float kHalfFov = 0.5F * 50.0F * 3.14159265F / 180.0F;
  const float focal = 0.5F * std::min(size.x, size.y) / std::tan(kHalfFov);
  if (wheel != 0) {
    if (perspective) dolly_ = std::clamp(dolly_ * std::pow(0.85F, wheel), 0.02F, 20.0F);
    else zoom_ = std::clamp(zoom_ * std::pow(1.15F, wheel), 0.05F, 200.0F);
  }
  const float fitted_distance = 1.1F * radius / std::sin(kHalfFov);  // the trajectory's sphere fills the view
  const float eye_distance = dolly_ * fitted_distance;
  const float scale = zoom_ * 0.85F * std::min(size.x, size.y) / (2.0F * radius);
  // World length of one pixel at the orbit point (pan and trackball sizing).
  const float world_per_pixel = perspective ? eye_distance / focal : 1.0F / scale;
  pan_[0] += (-pan_pixels.x * right.x + pan_pixels.y * up.x) * world_per_pixel;
  pan_[1] += (-pan_pixels.x * right.y + pan_pixels.y * up.y) * world_per_pixel;
  pan_[2] += (-pan_pixels.x * right.z + pan_pixels.y * up.z) * world_per_pixel;
  pivot = {pivot.x + pan_[0], pivot.y + pan_[1], pivot.z + pan_[2]};
  const ImVec2 middle(origin.x + size.x * 0.5F, origin.y + size.y * 0.5F);
  const auto dot = [](V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; };
  // Eye coordinates: (right, up, distance in front of the eye).
  struct Eye { float r, u, depth; };
  const auto eye = [&](V3 v) {
    const V3 d{v.x - pivot.x, v.y - pivot.y, v.z - pivot.z};
    return Eye{dot(right, d), dot(up, d), eye_distance - dot(toward, d)};
  };
  const float near_plane = 0.01F * eye_distance;
  const auto to_screen = [&](const Eye& e, float* magnification = nullptr) {
    if (magnification) *magnification = perspective ? eye_distance / e.depth : 1.0F;
    const float s = perspective ? focal / e.depth : scale;  // pixels per world unit at this depth
    return ImVec2(middle.x + e.r * s, middle.y - e.u * s);
  };
  // Point projection; nullopt behind the near plane (perspective only).
  const auto project = [&](V3 v, float* magnification = nullptr) -> std::optional<ImVec2> {
    const Eye e = eye(v);
    if (perspective && e.depth < near_plane) return std::nullopt;
    return to_screen(e, magnification);
  };
  // Line with near-plane clipping.
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

  // Floor: grid on the XZ plane (y = 0: the first keyframe camera's height;
  // monocular VO knows no ground plane), centred under the pivot.
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
    // Axes at the segment origin (first keyframe): x red, y (up) green, z blue.
    const float axis = 0.3F * radius;
    line({0, 0, 0}, {axis, 0, 0}, IM_COL32(220, 60, 60, 255), 2.0F);
    line({0, 0, 0}, {0, axis, 0}, IM_COL32(60, 200, 60, 255), 2.0F);
    line({0, 0, 0}, {0, 0, axis}, IM_COL32(60, 100, 235, 255), 2.0F);
  }

  if (show_points_) {
    // Solid icosahedrons on the GPU, depth-tested among themselves and drawn
    // over the grid, under the trajectory and frustum overlays.
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
    const int only = all_segments_ ? -1 : segment;
    const auto color = static_cast<PointColor>(color_mode_);
    const auto now = static_cast<float>(current->frame_index);
    const PointCloudStyle outside{point_size_, dim_outside_ ? 0.6F : 1.0F, color, only, now, age_span_};
    const PointCloudStyle local{point_size_, 1.0F, color, only, now, age_span_};
    points_renderer_.render(draw, {origin.x, origin.y}, {origin.x + size.x, origin.y + size.y},
                            io.DisplayFramebufferScale.x, camera, show_outside_ ? &outside_ : nullptr, outside,
                            &local_, local);
  }
  const TrajectorySample* previous = nullptr;
  for (const auto& s : samples_) {
    if (!visible(s)) { previous = nullptr; continue; }
    const V3 p = display(s.pose.center());
    if (previous && previous->segment == s.segment)
      line(display(previous->pose.center()), p, segment_color(s.segment, s.predicted ? 110 : 255), 2.0F);  // coasted: dim
    if (s.keyframe)
      if (const auto q = project(p))
        draw->AddRect({q->x - 2.5F, q->y - 2.5F}, {q->x + 2.5F, q->y + 2.5F}, segment_color(s.segment, 220));
    previous = &s;
  }
  {
    // Frustum of the current camera, sized relative to the trajectory.
    const auto& cfg = odometry.config();
    const double half_width = cfg.intrinsics ? 0.5 * frame_width / cfg.intrinsics->fx :
        std::tan(0.5 * cfg.horizontal_fov_degrees * 3.14159265358979 / 180.0);
    const double half_height = cfg.intrinsics ? 0.5 * frame_height / cfg.intrinsics->fy :
        half_width * frame_height / std::max(frame_width, 1);
    const double depth = 0.16 * radius;
    const auto c = current->pose.center();
    const V3 apex = display(c);
    std::array<V3, 4> corners;
    const double sx[4] = {-1, 1, 1, -1}, syy[4] = {-1, -1, 1, 1};
    for (int k = 0; k < 4; ++k) {
      const auto d = to_world(current->pose, sx[k] * half_width * depth, syy[k] * half_height * depth, depth);
      corners[k] = display(std::array<double, 3>{c[0] + d[0], c[1] + d[1], c[2] + d[2]});
      line(apex, corners[k], IM_COL32_WHITE, 1.2F);
    }
    for (int k = 0; k < 4; ++k) line(corners[k], corners[(k + 1) % 4], IM_COL32_WHITE, 1.2F);
    if (const auto q = project(apex)) draw->AddCircleFilled(*q, 3.5F, IM_COL32_WHITE);
  }
  if (mode_ == 0) {
    // Trackball around the orbit point while the view is being controlled
    // (fades out after scrolling). Rings: around x red, y green, z blue.
    const bool held = ImGui::IsItemActive() && (ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
                                                ImGui::IsMouseDown(ImGuiMouseButton_Right));
    const float fade = held ? 1.0F :
        std::clamp(1.0F - static_cast<float>(ImGui::GetTime() - last_interaction_ - 0.3) / 0.4F, 0.0F, 1.0F);
    if (fade > 0) {
      const float ring = 0.175F * std::min(size.x, size.y) * world_per_pixel;
      const int alpha = static_cast<int>(26 * fade);  // 10% opacity
      const ImU32 colors[3] = {IM_COL32(230, 70, 70, alpha), IM_COL32(80, 220, 80, alpha), IM_COL32(80, 130, 255, alpha)};
      constexpr int kSegments = 72;
      for (int axis = 0; axis < 3; ++axis) {
        const auto point_at = [&](int k) {
          const float t = 6.2831853F * k / kSegments, a = ring * std::cos(t), b = ring * std::sin(t);
          if (axis == 0) return V3{pivot.x, pivot.y + a, pivot.z + b};
          if (axis == 1) return V3{pivot.x + a, pivot.y, pivot.z + b};
          return V3{pivot.x + a, pivot.y + b, pivot.z};
        };
        for (int k = 0; k < kSegments; ++k) line(point_at(k), point_at(k + 1), colors[axis], 2.0F);
      }
      if (const auto q = project(pivot)) draw->AddCircleFilled(*q, 4.0F, IM_COL32(255, 255, 255, alpha));
    }
  }
  draw->PopClipRect();
  if (hovered && !ImGui::IsItemActive()) {
    ImGui::SetTooltip("%s\nRight-drag: pan | Double-click: fit",
                      mode_ != 0 ? "Fixed view | Wheel: zoom" :
                      perspective_ ? "Left-drag: orbit | Wheel: move toward the orbit point" :
                                     "Left-drag: orbit | Wheel: zoom");
  }
  ImGui::End();
}

}  // namespace slam_native
