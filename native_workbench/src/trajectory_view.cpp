#include "slam_native/trajectory_view.hpp"

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace slam_native {
namespace {
struct V3 {
  float x, y, z;
};

// World (camera convention: x right, y down, z forward) -> display with y up.
V3 display(const std::array<double, 3>& p) { return {float(p[0]), float(-p[1]), float(p[2])}; }
V3 display(const std::array<float, 3>& p) { return {p[0], -p[1], p[2]}; }

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
  points_.clear();
}

void TrajectoryView::update(const VisualOdometry& odometry) {
  const std::size_t stable = std::min(odometry.stable_prefix(), samples_.size());
  if (odometry.trajectory_size() < samples_.size()) samples_.clear();  // odometry was reset
  samples_.resize(std::min(samples_.size(), stable));
  auto tail = odometry.trajectory(samples_.size());
  samples_.insert(samples_.end(), tail.begin(), tail.end());
  points_ = odometry.map_points();
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
      last.state == OdometryState::lost ? ImVec4(1, 0.4F, 0.3F, 1) : ImVec4(1, 0.8F, 0.3F, 1);
  ImGui::TextColored(state_color, "%s", to_string(last.state));
  ImGui::SameLine();
  ImGui::Text("segment %d  |  %.1f ms", last.segment, last.ms);
  ImGui::Text("Inliers %d / %d  |  map %zu  |  keyframes %zu", last.inliers, last.correspondences,
              last.map_points, last.keyframes);
  if (last.has_pose) ImGui::Text("Median reprojection %.2f px", last.median_reprojection_px);
  ImGui::PushTextWrapPos(0);
  if (!last.event.empty()) ImGui::TextDisabled("%s", last.event.c_str());
  else ImGui::TextDisabled("Monocular: scale is arbitrary and differs between segments.");
  ImGui::PopTextWrapPos();

  const char* modes[] = {"Orbit", "Top (X-Z)", "Side (Z-Y)", "Front (X-Y)"};
  ImGui::SetNextItemWidth(120);
  ImGui::Combo("##view", &mode_, modes, IM_ARRAYSIZE(modes));
  ImGui::SameLine();
  if (ImGui::SmallButton("Fit")) {
    zoom_ = 1;
    pan_x_ = pan_y_ = 0;
  }
  ImGui::Checkbox("Map points", &show_points_);
  ImGui::SameLine();
  ImGui::Checkbox("Follow", &follow_);
  ImGui::SameLine();
  ImGui::Checkbox("All segments", &all_segments_);

  const ImVec2 origin = ImGui::GetCursorScreenPos();
  ImVec2 size = ImGui::GetContentRegionAvail();
  size.x = std::max(size.x, 50.0F);
  size.y = std::max(size.y, 50.0F);
  ImGui::InvisibleButton("##trajectory-canvas", size,
                         ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
  const bool hovered = ImGui::IsItemHovered();
  const ImGuiIO& io = ImGui::GetIO();
  if (hovered && io.MouseWheel != 0) zoom_ = std::clamp(zoom_ * std::pow(1.15F, io.MouseWheel), 0.05F, 200.0F);
  if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left) && mode_ == 0) {
    yaw_ += io.MouseDelta.x * 0.01F;
    pitch_ = std::clamp(pitch_ + io.MouseDelta.y * 0.01F, -1.55F, 1.55F);
  }
  if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
    pan_x_ += io.MouseDelta.x;
    pan_y_ += io.MouseDelta.y;
  }
  if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
    zoom_ = 1;
    pan_x_ = pan_y_ = 0;
  }
  ImDrawList* draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(origin, {origin.x + size.x, origin.y + size.y}, IM_COL32(12, 14, 18, 255));
  draw->PushClipRect(origin, {origin.x + size.x, origin.y + size.y}, true);

  const int segment = samples_.empty() ? last.segment : samples_.back().segment;
  const auto visible = [&](const TrajectorySample& s) { return all_segments_ || s.segment == segment; };
  float yaw = yaw_, pitch = pitch_;
  if (mode_ == 1) { yaw = 0; pitch = 1.5707963F; }
  if (mode_ == 2) { yaw = 1.5707963F; pitch = 0; }
  if (mode_ == 3) { yaw = 0; pitch = 0; }
  const float cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(pitch), sp = std::sin(pitch);
  // Orthographic view: returns (right, up).
  const auto view = [&](V3 v) {
    const float x1 = cy * v.x - sy * v.z, z1 = sy * v.x + cy * v.z;
    return ImVec2(x1, cp * v.y + sp * z1);
  };

  // Fit the visible trajectory (the map would let outliers dominate).
  float min_x = std::numeric_limits<float>::max(), min_y = min_x, max_x = -min_x, max_y = -min_x;
  for (const auto& s : samples_) {
    if (!visible(s)) continue;
    const ImVec2 p = view(display(s.pose.center()));
    min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
  }
  if (min_x > max_x) {
    draw->AddText({origin.x + 12, origin.y + 12}, IM_COL32(170, 170, 170, 255),
                  "No camera poses yet: move the camera sideways/forward to initialize.");
    draw->PopClipRect();
    ImGui::End();
    return;
  }
  const float extent = std::max({max_x - min_x, max_y - min_y, 1e-3F});
  const float scale = zoom_ * 0.85F * std::min(size.x, size.y) / extent;
  ImVec2 center((min_x + max_x) * 0.5F, (min_y + max_y) * 0.5F);
  const TrajectorySample* current = nullptr;
  for (auto it = samples_.rbegin(); it != samples_.rend(); ++it)
    if (visible(*it)) { current = &*it; break; }
  if (follow_ && current) center = view(display(current->pose.center()));
  const auto screen = [&](ImVec2 p) {
    return ImVec2(origin.x + size.x * 0.5F + pan_x_ + (p.x - center.x) * scale,
                  origin.y + size.y * 0.5F + pan_y_ - (p.y - center.y) * scale);
  };

  // Ground-plane axes at the segment origin (first keyframe).
  const float axis = 0.15F * extent;
  const ImVec2 o = screen(view({0, 0, 0}));
  draw->AddLine(o, screen(view({axis, 0, 0})), IM_COL32(200, 70, 70, 200), 1.5F);
  draw->AddLine(o, screen(view({0, axis, 0})), IM_COL32(70, 200, 70, 200), 1.5F);
  draw->AddLine(o, screen(view({0, 0, axis})), IM_COL32(70, 110, 230, 200), 1.5F);

  if (show_points_ && !points_.empty()) {
    const std::size_t stride = std::max<std::size_t>(1, points_.size() / 6000);
    for (std::size_t i = 0; i < points_.size(); i += stride) {
      const ImVec2 p = screen(view(display(points_[i])));
      draw->AddRectFilled(p, {p.x + 1.5F, p.y + 1.5F}, IM_COL32(150, 150, 160, 110));
    }
  }
  const TrajectorySample* previous = nullptr;
  for (const auto& s : samples_) {
    if (!visible(s)) { previous = nullptr; continue; }
    const ImVec2 p = screen(view(display(s.pose.center())));
    if (previous && previous->segment == s.segment)
      draw->AddLine(screen(view(display(previous->pose.center()))), p, segment_color(s.segment), 2.0F);
    if (s.keyframe) draw->AddRect({p.x - 2.5F, p.y - 2.5F}, {p.x + 2.5F, p.y + 2.5F}, segment_color(s.segment, 200));
    previous = &s;
  }
  if (current) {
    // Frustum of the current camera, sized relative to the trajectory.
    const auto& cfg = odometry.config();
    const double half_width = cfg.intrinsics ? 0.5 * frame_width / cfg.intrinsics->fx :
        std::tan(0.5 * cfg.horizontal_fov_degrees * 3.14159265358979 / 180.0);
    const double half_height = cfg.intrinsics ? 0.5 * frame_height / cfg.intrinsics->fy :
        half_width * frame_height / std::max(frame_width, 1);
    const double depth = 0.08 * extent;
    const auto c = current->pose.center();
    const ImVec2 apex = screen(view(display(c)));
    std::array<ImVec2, 4> corners;
    const double sx[4] = {-1, 1, 1, -1}, syy[4] = {-1, -1, 1, 1};
    for (int k = 0; k < 4; ++k) {
      const auto d = to_world(current->pose, sx[k] * half_width * depth, syy[k] * half_height * depth, depth);
      corners[k] = screen(view(display(std::array<double, 3>{c[0] + d[0], c[1] + d[1], c[2] + d[2]})));
      draw->AddLine(apex, corners[k], IM_COL32_WHITE, 1.2F);
    }
    for (int k = 0; k < 4; ++k) draw->AddLine(corners[k], corners[(k + 1) % 4], IM_COL32_WHITE, 1.2F);
    draw->AddCircleFilled(apex, 3.5F, IM_COL32_WHITE);
  }
  draw->PopClipRect();
  if (hovered && !ImGui::IsItemActive()) {
    ImGui::SetTooltip("%s\nWheel: zoom | Right-drag: pan | Double-click: fit",
                      mode_ == 0 ? "Left-drag: orbit" : "Fixed view");
  }
  ImGui::End();
}

}  // namespace slam_native
