#pragma once
// OpenGL renderer for the trajectory view's map points: every point is a
// solid, unlit (constant-colour) icosahedron, drawn with one instanced call per point set
// into an offscreen (multisampled, depth-tested) target that is composited
// into an ImGui draw list. All calls need the app's OpenGL 3.3 core context.
#include "slam_native/visual_odometry.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <cstdint>
#include <unordered_map>
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

enum class PointColor { plain, image, confidence };

struct PointCloudStyle {
  float size{0.01F};   // world-space diameter (segment units)
  float brightness{1};  // multiplies the colour (retired points are dimmer)
  PointColor color{PointColor::image};  // image: falls back to plain without a colour
  int segment{-1};      // only this segment's points; < 0 draws all
  float min_confidence{0};  // > 0 hides points below it, and those with unknown confidence
  // Draw each point as its uncertainty ellipsoid (MapPoint::covariance), axes
  // x sigma_scale, each clamped to max_axis world units; points without a
  // covariance keep the plain size.
  bool uncertainty_shape{false};
  float sigma_scale{1};
  float max_axis{1};
};

// A dense cloud (e.g. a keyframe's back-projected depth): points in its own
// frame, drawn as screen-space point sprites with world = R * (p * units) + o.
// Buffers are uploaded once per key; the transform is applied every frame, so
// the cloud follows its keyframe's pose without re-uploading.
struct DenseCloudDraw {
  std::uint64_t key{};  // stable id (keyframe frame index)
  std::uint32_t version{};  // re-uploaded when it changes (points, colours or `shown` changed)
  const std::vector<std::array<float, 3>>* points{};
  const std::vector<std::array<std::uint8_t, 3>>* colors{};
  const std::vector<std::uint8_t>* shown{};  // optional per-point mask: 0 = not drawn
  std::array<float, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};  // cloud -> world, row-major
  std::array<float, 3> origin{};                              // cloud origin in world
  float units_per_point_unit{1};                              // e.g. 1 / (metres per VO unit)
  // Optional per-point network depth sigma (log depth, KeyframeCloud::sigma):
  // points above max_sigma are not drawn; color_by_confidence replaces the
  // colour by 1 / (1 + (sigma / sigma_reference)^2) on the map points' ramp.
  const std::vector<float>* sigma{};
  float max_sigma{std::numeric_limits<float>::infinity()};
  bool color_by_confidence{};
  float sigma_reference{0.2F};
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
  // `clouds` (optional): dense clouds, depth-tested with the map points;
  // cloud_point_pixels is their sprite size. A changed cloud_generation drops
  // all uploaded clouds first (the owner removed some).
  void render(ImDrawList* draw, std::array<float, 2> min, std::array<float, 2> max, float pixel_scale,
              const PointCloudCamera& camera, const std::vector<MapPoint>* retired,
              const PointCloudStyle& retired_style, const std::vector<MapPoint>* active,
              const PointCloudStyle& active_style, const std::vector<DenseCloudDraw>* clouds = nullptr,
              float cloud_point_pixels = 2.0F, std::uint64_t cloud_generation = 0);

 private:
  struct InstanceSet {
    unsigned int buffer{}, vao{};
    std::size_t capacity{}, uploaded{};
  };
  struct CloudBuffer {
    unsigned int buffer{}, vao{};
    int count{};
    std::uint32_t version{};
    bool masked{};
  };
  bool ensure_gl();
  bool ensure_cloud_gl();
  void draw_clouds(const std::vector<DenseCloudDraw>& clouds, const PointCloudCamera& camera, float logical_width,
                   float logical_height, float point_pixels, std::uint64_t generation);
  void ensure_target(int width, int height);
  void upload(InstanceSet& set, const std::vector<MapPoint>& points, bool incremental);
  void draw_set(const InstanceSet& set, const PointCloudStyle& style);

  InstanceSet retired_, active_;
  bool gl_ready_{}, gl_failed_{};
  unsigned int program_{}, mesh_{};
  int u_pivot_{-1}, u_right_{-1}, u_up_{-1}, u_toward_{-1}, u_eye_distance_{-1}, u_perspective_{-1},
      u_focal_{-1}, u_scale_{-1}, u_near_{-1}, u_far_{-1}, u_viewport_{-1}, u_rotation_{-1}, u_size_{-1},
      u_brightness_{-1}, u_color_mode_{-1}, u_segment_{-1}, u_min_confidence_{-1}, u_shape_mode_{-1},
      u_sigma_scale_{-1}, u_max_axis_{-1};
  unsigned int msaa_fbo_{}, msaa_color_{}, msaa_depth_{}, resolve_fbo_{}, texture_{};
  bool cloud_gl_ready_{}, cloud_gl_failed_{};
  unsigned int cloud_program_{};
  int c_pivot_{-1}, c_right_{-1}, c_up_{-1}, c_toward_{-1}, c_eye_distance_{-1}, c_perspective_{-1}, c_focal_{-1},
      c_scale_{-1}, c_near_{-1}, c_far_{-1}, c_viewport_{-1}, c_rotation_{-1}, c_model_{-1}, c_origin_{-1},
      c_units_{-1}, c_point_size_{-1}, c_max_sigma_{-1}, c_sigma_reference_{-1}, c_color_by_confidence_{-1};
  std::unordered_map<std::uint64_t, CloudBuffer> clouds_;
  std::uint64_t cloud_generation_{};
  int width_{}, height_{};
};

}  // namespace slam_native
