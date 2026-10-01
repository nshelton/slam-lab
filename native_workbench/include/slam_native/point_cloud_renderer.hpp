#pragma once
// OpenGL renderer for the trajectory view's map points: every point is a
// solid, unlit (constant-colour) icosahedron, drawn with one instanced call per point set
// into an offscreen (multisampled, depth-tested) target that is composited
// into an ImGui draw list. All calls need the app's OpenGL 3.3 core context.
#include "slam_native/visual_odometry.hpp"

#include <array>
#include <cstddef>
#include <vector>

struct ImDrawList;

namespace slam_native {

// Camera in the trajectory view's display space (see trajectory_view.cpp),
// projected exactly like the view's 2D overlays so the two line up.
struct PointCloudCamera {
  std::array<float, 3> pivot{};
  std::array<float, 3> right{1, 0, 0};
  std::array<float, 3> up{0, 1, 0};
  std::array<float, 3> toward{0, 0, 1};  // unit vector from the pivot toward the eye
  float eye_distance{1};
  bool perspective{true};
  float focal{1};  // perspective: pixels per world unit at depth 1
  float scale{1};  // orthographic: pixels per world unit
  float near_plane{0.01F};
  int rotation{};  // clockwise display rotation of the video (degrees)
};

enum class PointColor { plain, image };

struct PointCloudStyle {
  float size{0.01F};   // world-space diameter (segment units)
  float brightness{1};  // multiplies the colour (retired points are dimmer)
  PointColor color{PointColor::image};  // image: falls back to plain without a colour
  int segment{-1};      // only this segment's points; < 0 draws all
};

class PointCloudRenderer {
 public:
  PointCloudRenderer() = default;
  ~PointCloudRenderer();
  PointCloudRenderer(const PointCloudRenderer&) = delete;
  PointCloudRenderer& operator=(const PointCloudRenderer&) = delete;

  // Forget uploaded points (the owner's retired map was reset). No GL calls.
  void clear() { retired_.uploaded = 0; }

  // Renders both point sets over the screen rectangle [min, max] (logical
  // pixels; the target has pixel_scale x as many) and adds the image to
  // `draw`: anything added before it is behind the points, anything after in
  // front. `retired` is append-only: only its new tail is uploaded. `active`
  // is re-uploaded every call. Null sets are skipped.
  void render(ImDrawList* draw, std::array<float, 2> min, std::array<float, 2> max, float pixel_scale,
              const PointCloudCamera& camera, const std::vector<MapPoint>* retired,
              const PointCloudStyle& retired_style, const std::vector<MapPoint>* active,
              const PointCloudStyle& active_style);

 private:
  struct InstanceSet {
    unsigned int buffer{}, vao{};
    std::size_t capacity{}, uploaded{};
  };
  bool ensure_gl();
  void ensure_target(int width, int height);
  void upload(InstanceSet& set, const std::vector<MapPoint>& points, bool incremental);
  void draw_set(const InstanceSet& set, const PointCloudStyle& style);

  InstanceSet retired_, active_;
  bool gl_ready_{}, gl_failed_{};
  unsigned int program_{}, mesh_{};
  int u_pivot_{-1}, u_right_{-1}, u_up_{-1}, u_toward_{-1}, u_eye_distance_{-1}, u_perspective_{-1},
      u_focal_{-1}, u_scale_{-1}, u_near_{-1}, u_far_{-1}, u_viewport_{-1}, u_rotation_{-1}, u_size_{-1},
      u_brightness_{-1}, u_color_mode_{-1}, u_segment_{-1};
  unsigned int msaa_fbo_{}, msaa_color_{}, msaa_depth_{}, resolve_fbo_{}, texture_{};
  int width_{}, height_{};
};

}  // namespace slam_native
