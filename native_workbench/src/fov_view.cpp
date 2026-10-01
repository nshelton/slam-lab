#include "slam_native/fov_view.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace slam_native {
namespace {
constexpr ImU32 kCurve = IM_COL32(80, 200, 255, 255);
constexpr ImU32 kEstimate = IM_COL32(120, 230, 120, 255);
constexpr ImU32 kSlider = IM_COL32(250, 220, 80, 255);
constexpr ImU32 kApplied = IM_COL32(150, 150, 150, 200);

struct Marker {
  double x;
  ImU32 color;
  float thickness;
};

// Curve over a grid with vertical markers; `ticks` are drawn along the bottom.
// Square-root vertical scale keeps the shallow region near the minimum readable.
void plot(const char* id, const std::vector<double>& x, const std::vector<double>& y, double label_step,
          const char* label_format, const std::vector<Marker>& markers, const std::vector<double>& ticks = {}) {
  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const ImVec2 size(std::max(ImGui::GetContentRegionAvail().x, 100.0F), 110.0F);
  ImGui::InvisibleButton(id, size);
  ImDrawList* draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(origin, {origin.x + size.x, origin.y + size.y}, IM_COL32(12, 14, 18, 255));
  if (x.size() < 2) return;
  const double x0 = x.front(), x1 = x.back();
  const auto sx = [&](double v) { return origin.x + static_cast<float>((v - x0) / (x1 - x0)) * size.x; };
  for (double v = std::ceil(x0 / label_step) * label_step; v <= x1 + 1e-9; v += label_step) {
    draw->AddLine({sx(v), origin.y}, {sx(v), origin.y + size.y}, IM_COL32(40, 44, 52, 255));
    char label[16];
    std::snprintf(label, sizeof label, label_format, std::abs(v) < 1e-9 ? 0.0 : v);
    draw->AddText({sx(v) + 2, origin.y + 2}, IM_COL32(110, 115, 125, 255), label);
  }
  if (y.size() == x.size()) {
    const auto [low, high] = std::minmax_element(y.begin(), y.end());
    const double span = std::max(*high - *low, 1e-12);
    const auto sy = [&](double v) {
      return origin.y + size.y - 12 - static_cast<float>(std::sqrt((v - *low) / span)) * (size.y - 28);
    };
    for (std::size_t k = 1; k < x.size(); ++k)
      draw->AddLine({sx(x[k - 1]), sy(y[k - 1])}, {sx(x[k]), sy(y[k])}, kCurve, 1.5F);
  }
  for (double t : ticks) draw->AddLine({sx(t), origin.y + size.y - 7}, {sx(t), origin.y + size.y}, IM_COL32(200, 200, 200, 90));
  for (const auto& m : markers)
    if (m.x >= x0 && m.x <= x1) draw->AddLine({sx(m.x), origin.y}, {sx(m.x), origin.y + size.y}, m.color, m.thickness);
}
}  // namespace

std::optional<LensChoice> FovView::draw(const FocalEstimate& estimate, LensChoice applied, bool* open,
                                        bool& reset_estimate) {
  reset_estimate = false;
  if (open && !*open) return std::nullopt;
  if (!initialized_ && applied.hfov_degrees > 0) {
    hfov_ = static_cast<float>(applied.hfov_degrees);
    k1_ = static_cast<float>(applied.k1);
    initialized_ = true;
  }
  ImGui::SetNextWindowPos({1090, 8}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({360, 520}, ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Camera lens", open)) {
    ImGui::End();
    return std::nullopt;
  }
  const bool have_k1 = estimate.pairs_distortion > 0;
  const bool have_fov = estimate.pairs_used > 0;
  if (follow_) {
    if (have_k1) k1_ = static_cast<float>(estimate.best_k1);
    if (have_fov) hfov_ = static_cast<float>(estimate.best_hfov_degrees);
  }

  if (have_fov) {
    ImGui::Text("Estimate: %.1f deg horizontal (fx %.0f px), k1 %.3f", estimate.best_hfov_degrees,
                estimate.best_focal_px, estimate.best_k1);
    ImGui::Text("Pair FOV minima: %.0f / %.0f / %.0f deg (25/50/75%%)", estimate.pair_p25_degrees,
                estimate.pair_median_degrees, estimate.pair_p75_degrees);
  } else if (have_k1) {
    ImGui::Text("Estimate: k1 %.3f; FOV needs a pair with rotation", estimate.best_k1);
  } else {
    ImGui::TextDisabled("Estimate: waiting for frame pairs with motion");
  }
  ImGui::TextDisabled("%d pairs for k1, %d for FOV, %d no rotation, %d homography, %d weak",
                      estimate.pairs_distortion, estimate.pairs_used, estimate.pairs_flat, estimate.pairs_homography,
                      estimate.pairs_weak);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Each pair of frames with enough track motion gives a fundamental matrix F.\n"
                      "k1: lens distortion bends epipolar lines, so the right k1 lets F fit the\n"
                      "undistorted tracks best (lowest epipolar residual, bottom plot).\n"
                      "FOV: with the right focal length, K^T F K is a valid essential matrix\n"
                      "(two equal singular values). Top plot: mean violation at the slider's k1.\n"
                      "No rotation: pure translation informs k1 but not the focal length.\n"
                      "Homography: pure rotation or one plane; F is undetermined.\n"
                      "Assumes square pixels and a centred principal point.");
  }

  ImGui::SeparatorText("Field of view (at the k1 below)");
  const auto curve = estimate.cost_at(k1_);
  std::vector<Marker> fov_markers;
  if (applied.hfov_degrees > 0) fov_markers.push_back({applied.hfov_degrees, kApplied, 1.0F});
  if (have_fov) fov_markers.push_back({estimate.best_hfov_degrees, kEstimate, 1.5F});
  fov_markers.push_back({hfov_, kSlider, 1.0F});
  plot("##fov-curve", estimate.hfov_degrees, curve, 30, "%.0f", fov_markers,
       std::abs(k1_ - estimate.best_k1) < 0.011 ? estimate.pair_best_degrees : std::vector<double>{});
  const auto& grid = estimate.hfov_degrees;
  ImGui::SetNextItemWidth(-1);
  if (ImGui::SliderFloat("##fov", &hfov_, grid.empty() ? 30.0F : float(grid.front()),
                         grid.empty() ? 150.0F : float(grid.back()), "%.1f deg horizontal"))
    follow_ = false;
  if (!follow_ && have_fov) {
    const double best_here = estimate.best_hfov_at(k1_);
    ImGui::TextDisabled("Best FOV at this k1: %.1f deg", best_here);
    ImGui::SameLine();
    if (ImGui::SmallButton("Use")) hfov_ = static_cast<float>(best_here);
  }

  ImGui::SeparatorText("Radial distortion k1");
  std::vector<Marker> k1_markers{{applied.k1, kApplied, 1.0F}};
  if (have_k1) k1_markers.push_back({estimate.best_k1, kEstimate, 1.5F});
  k1_markers.push_back({k1_, kSlider, 1.0F});
  plot("##k1-curve", estimate.k1_values, estimate.k1_residual_px, 0.2, "%.1f", k1_markers);
  if (have_k1 && !estimate.k1_residual_px.empty()) {
    const auto [low, high] = std::minmax_element(estimate.k1_residual_px.begin(), estimate.k1_residual_px.end());
    ImGui::TextDisabled("Epipolar RMS %.3f px at best, %.3f px worst", *low, *high);
  }
  const auto& k1s = estimate.k1_values;
  ImGui::SetNextItemWidth(-1);
  if (ImGui::SliderFloat("##k1", &k1_, k1s.empty() ? -0.6F : float(k1s.front()), k1s.empty() ? 0.3F : float(k1s.back()),
                         "k1 %.3f (< 0 barrel)"))
    follow_ = false;

  ImGui::Separator();
  ImGui::Checkbox("Follow estimate", &follow_);
  ImGui::SameLine();
  if (ImGui::SmallButton("Reset estimate")) reset_estimate = true;
  if (applied.hfov_degrees > 0) ImGui::Text("Pose uses %.1f deg, k1 %.3f", applied.hfov_degrees, applied.k1);
  else ImGui::TextColored({1, 0.8F, 0.3F, 1}, "Pose uses explicit --intrinsics, k1 %.3f", applied.k1);
  std::optional<LensChoice> apply;
  const bool unchanged = applied.hfov_degrees > 0 && std::abs(hfov_ - applied.hfov_degrees) < 0.05 &&
                         std::abs(k1_ - applied.k1) < 0.0005;
  ImGui::BeginDisabled(unchanged);
  if (ImGui::Button("Apply to pose")) apply = LensChoice{hfov_, k1_};
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("Restart the camera pose from the current frame with this lens.\n"
                      "Use Restart in the Pipeline window to replay the video from the start.");
  ImGui::End();
  return apply;
}

}  // namespace slam_native
