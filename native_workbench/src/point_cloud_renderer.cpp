#define GL_GLEXT_PROTOTYPES
#include "slam_native/point_cloud_renderer.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace slam_native {
namespace {

// Per-point instance data (20 bytes). Positions stay in the camera-frame
// world: the shader applies the display rotation, so rotating the video never
// re-uploads the map.
struct Instance {
  float position[3];
  std::uint8_t color[4];  // rgb, a = 255 when the point has an image colour
  std::int32_t segment;
};
static_assert(sizeof(Instance) == 20);

constexpr const char* kVertexShader = R"(#version 330 core
layout(location = 0) in vec3 a_vertex;    // unit icosahedron corner
layout(location = 2) in vec3 i_position;  // camera-frame world position
layout(location = 3) in vec4 i_color;     // rgb, a > 0.5: has an image colour
layout(location = 4) in int i_segment;

uniform vec3 u_pivot, u_right, u_up, u_toward;
uniform float u_eye_distance, u_focal, u_scale, u_near, u_far;
uniform int u_perspective, u_rotation, u_color_mode, u_segment;  // colour: 0 plain, 1 image
uniform vec2 u_viewport;  // logical pixels
uniform float u_size, u_brightness;  // u_size: world-space diameter

flat out vec3 v_color;

// Camera frame (x right, y down) -> display space (y up) after the video's
// clockwise rotation; mirrors display() in trajectory_view.cpp.
vec3 display(vec3 p) {
  float u = p.x, v = p.y;
  if (u_rotation == 90) { u = -p.y; v = p.x; }
  else if (u_rotation == 180) { u = -p.x; v = -p.y; }
  else if (u_rotation == 270) { u = p.y; v = -p.x; }
  return vec3(u, -v, p.z);
}

void main() {
  vec3 d = display(i_position) - u_pivot;
  // Eye coordinates: (screen right, screen up, depth in front of the eye).
  vec3 center = vec3(dot(u_right, d), dot(u_up, d), u_eye_distance - dot(u_toward, d));
  bool perspective = u_perspective != 0;
  if ((u_segment >= 0 && i_segment != u_segment) || (perspective && center.z < u_near)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);  // whole instance outside the clip volume
    v_color = vec3(0.0);
    return;
  }
  // World-space size (shrinks with distance like real geometry), floored at
  // ~1 px so distant points thin out instead of vanishing.
  float pixels_per_unit = perspective ? u_focal / center.z : u_scale;
  float radius = max(0.5 * u_size, 0.5 / pixels_per_unit);
  vec3 offset = radius * a_vertex;  // display space
  vec3 e = center + vec3(dot(u_right, offset), dot(u_up, offset), -dot(u_toward, offset));
  vec2 ndc = 2.0 / u_viewport;
  if (perspective) {
    float a = (u_far + u_near) / (u_far - u_near), b = -2.0 * u_far * u_near / (u_far - u_near);
    gl_Position = vec4(e.x * u_focal * ndc.x, e.y * u_focal * ndc.y, a * e.z + b, e.z);
  } else {
    gl_Position = vec4(e.x * u_scale * ndc.x, e.y * u_scale * ndc.y, (e.z - u_eye_distance) / u_far, 1.0);
  }
  vec3 base = vec3(235.0 / 255.0);
  if (u_color_mode == 1 && i_color.a > 0.5) base = i_color.rgb;
  v_color = base * u_brightness;  // unlit: constant colour
}
)";

constexpr const char* kFragmentShader = R"(#version 330 core
flat in vec3 v_color;
out vec4 frag;
void main() { frag = vec4(v_color, 1.0); }
)";

GLuint compile(GLenum type, const char* source) {
  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint ok = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    std::fprintf(stderr, "PointCloudRenderer: shader compile failed:\n%s\n", log);
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}

// Unit icosahedron as a list of 20 triangles.
std::vector<float> icosahedron() {
  const float t = 0.5F * (1.0F + std::sqrt(5.0F));
  const float corners[12][3] = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                                {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  const int faces[20][3] = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
                            {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8},
                            {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
  const float norm = std::sqrt(1 + t * t);
  std::vector<float> out;
  out.reserve(20 * 3 * 3);
  for (const auto& face : faces)
    for (int k : face) out.insert(out.end(), {corners[k][0] / norm, corners[k][1] / norm, corners[k][2] / norm});
  return out;
}

constexpr GLsizei kMeshVertices = 60;

void composite_premultiplied(const ImDrawList*, const ImDrawCmd*) {
  // The resolved target is premultiplied (clear alpha 0, MSAA edge coverage).
  glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
}
}  // namespace

PointCloudRenderer::~PointCloudRenderer() {
  if (!gl_ready_) return;
  for (auto* set : {&retired_, &active_}) {
    glDeleteVertexArrays(1, &set->vao);
    glDeleteBuffers(1, &set->buffer);
  }
  glDeleteBuffers(1, &mesh_);
  glDeleteProgram(program_);
  glDeleteFramebuffers(1, &msaa_fbo_);
  glDeleteFramebuffers(1, &resolve_fbo_);
  glDeleteRenderbuffers(1, &msaa_color_);
  glDeleteRenderbuffers(1, &msaa_depth_);
  glDeleteTextures(1, &texture_);
}

bool PointCloudRenderer::ensure_gl() {
  if (gl_ready_ || gl_failed_) return gl_ready_;
  const GLuint vertex = compile(GL_VERTEX_SHADER, kVertexShader);
  const GLuint fragment = compile(GL_FRAGMENT_SHADER, kFragmentShader);
  if (!vertex || !fragment) {
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    gl_failed_ = true;
    return false;
  }
  program_ = glCreateProgram();
  glAttachShader(program_, vertex);
  glAttachShader(program_, fragment);
  glLinkProgram(program_);
  glDeleteShader(vertex);
  glDeleteShader(fragment);
  GLint ok = 0;
  glGetProgramiv(program_, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[2048];
    glGetProgramInfoLog(program_, sizeof(log), nullptr, log);
    std::fprintf(stderr, "PointCloudRenderer: program link failed:\n%s\n", log);
    glDeleteProgram(program_);
    program_ = 0;
    gl_failed_ = true;
    return false;
  }
  const auto uniform = [&](const char* name) { return glGetUniformLocation(program_, name); };
  u_pivot_ = uniform("u_pivot");
  u_right_ = uniform("u_right");
  u_up_ = uniform("u_up");
  u_toward_ = uniform("u_toward");
  u_eye_distance_ = uniform("u_eye_distance");
  u_perspective_ = uniform("u_perspective");
  u_focal_ = uniform("u_focal");
  u_scale_ = uniform("u_scale");
  u_near_ = uniform("u_near");
  u_far_ = uniform("u_far");
  u_viewport_ = uniform("u_viewport");
  u_rotation_ = uniform("u_rotation");
  u_size_ = uniform("u_size");
  u_brightness_ = uniform("u_brightness");
  u_color_mode_ = uniform("u_color_mode");
  u_segment_ = uniform("u_segment");

  GLint previous_vao = 0, previous_buffer = 0;
  glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &previous_vao);
  glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &previous_buffer);
  const auto mesh = icosahedron();
  glGenBuffers(1, &mesh_);
  glBindBuffer(GL_ARRAY_BUFFER, mesh_);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(mesh.size() * sizeof(float)), mesh.data(), GL_STATIC_DRAW);
  for (auto* set : {&retired_, &active_}) {
    glGenVertexArrays(1, &set->vao);
    glGenBuffers(1, &set->buffer);
    glBindVertexArray(set->vao);
    glBindBuffer(GL_ARRAY_BUFFER, mesh_);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    // Instance attributes; the buffer is (re)allocated in upload(), which
    // keeps its name and therefore these bindings.
    glBindBuffer(GL_ARRAY_BUFFER, set->buffer);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Instance),
                          reinterpret_cast<const void*>(offsetof(Instance, position)));
    glVertexAttribDivisor(2, 1);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Instance),
                          reinterpret_cast<const void*>(offsetof(Instance, color)));
    glVertexAttribDivisor(3, 1);
    glEnableVertexAttribArray(4);
    glVertexAttribIPointer(4, 1, GL_INT, sizeof(Instance), reinterpret_cast<const void*>(offsetof(Instance, segment)));
    glVertexAttribDivisor(4, 1);
  }
  glBindVertexArray(static_cast<GLuint>(previous_vao));
  glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(previous_buffer));

  glGenFramebuffers(1, &msaa_fbo_);
  glGenFramebuffers(1, &resolve_fbo_);
  glGenRenderbuffers(1, &msaa_color_);
  glGenRenderbuffers(1, &msaa_depth_);
  glGenTextures(1, &texture_);
  gl_ready_ = true;
  return true;
}

void PointCloudRenderer::ensure_target(int width, int height) {
  if (width == width_ && height == height_) return;
  width_ = width;
  height_ = height;
  GLint max_samples = 0;
  glGetIntegerv(GL_MAX_SAMPLES, &max_samples);
  const GLsizei samples = std::clamp(max_samples, 0, 4);
  glBindRenderbuffer(GL_RENDERBUFFER, msaa_color_);
  glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8, width, height);
  glBindRenderbuffer(GL_RENDERBUFFER, msaa_depth_);
  glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT24, width, height);
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, msaa_fbo_);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, msaa_color_);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, msaa_depth_);

  GLint previous_texture = 0;
  glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous_texture);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(previous_texture));
  glBindFramebuffer(GL_FRAMEBUFFER, resolve_fbo_);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_, 0);
}

void PointCloudRenderer::upload(InstanceSet& set, const std::vector<MapPoint>& points, bool incremental) {
  if (!incremental || points.size() < set.uploaded) set.uploaded = 0;
  glBindBuffer(GL_ARRAY_BUFFER, set.buffer);
  if (points.size() > set.capacity) {
    set.capacity = std::max({points.size(), 2 * set.capacity, std::size_t{1024}});
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(set.capacity * sizeof(Instance)), nullptr,
                 GL_DYNAMIC_DRAW);
    set.uploaded = 0;
  }
  if (points.size() == set.uploaded) return;
  std::vector<Instance> staging;
  staging.reserve(points.size() - set.uploaded);
  for (std::size_t i = set.uploaded; i < points.size(); ++i) {
    const auto& p = points[i];
    staging.push_back({{p.position[0], p.position[1], p.position[2]},
                       {p.color[0], p.color[1], p.color[2], static_cast<std::uint8_t>(p.has_color ? 255 : 0)},
                       p.segment});
  }
  glBufferSubData(GL_ARRAY_BUFFER, static_cast<GLintptr>(set.uploaded * sizeof(Instance)),
                  static_cast<GLsizeiptr>(staging.size() * sizeof(Instance)), staging.data());
  set.uploaded = points.size();
}

void PointCloudRenderer::draw_set(const InstanceSet& set, const PointCloudStyle& style) {
  if (set.uploaded == 0) return;
  glUniform1f(u_size_, style.size);
  glUniform1f(u_brightness_, style.brightness);
  glUniform1i(u_color_mode_, static_cast<int>(style.color));
  glUniform1i(u_segment_, style.segment);
  glBindVertexArray(set.vao);
  glDrawArraysInstanced(GL_TRIANGLES, 0, kMeshVertices, static_cast<GLsizei>(set.uploaded));
}

void PointCloudRenderer::render(ImDrawList* draw, std::array<float, 2> min, std::array<float, 2> max,
                                float pixel_scale, const PointCloudCamera& camera,
                                const std::vector<MapPoint>* retired, const PointCloudStyle& retired_style,
                                const std::vector<MapPoint>* active, const PointCloudStyle& active_style) {
  if (!retired && !active) return;
  const float logical_width = max[0] - min[0], logical_height = max[1] - min[1];
  const int width = std::max(1, static_cast<int>(std::lround(logical_width * pixel_scale)));
  const int height = std::max(1, static_cast<int>(std::lround(logical_height * pixel_scale)));
  if (!ensure_gl()) return;

  // Save the state this pass touches (ImGui's backend resets its own state
  // at render time, but other GL users in the frame may not).
  GLint framebuffer = 0, viewport[4]{}, program = 0, vao = 0, array_buffer = 0;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
  glGetIntegerv(GL_VIEWPORT, viewport);
  glGetIntegerv(GL_CURRENT_PROGRAM, &program);
  glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
  glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &array_buffer);
  const GLboolean blend = glIsEnabled(GL_BLEND), depth_test = glIsEnabled(GL_DEPTH_TEST),
                  scissor = glIsEnabled(GL_SCISSOR_TEST), cull = glIsEnabled(GL_CULL_FACE),
                  depth_clamp = glIsEnabled(GL_DEPTH_CLAMP);

  if (retired) upload(retired_, *retired, true);
  if (active) upload(active_, *active, false);
  ensure_target(width, height);

  glBindFramebuffer(GL_FRAMEBUFFER, msaa_fbo_);
  glViewport(0, 0, width, height);
  glDisable(GL_BLEND);
  glDisable(GL_SCISSOR_TEST);
  glDisable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glDepthMask(GL_TRUE);
  glEnable(GL_DEPTH_CLAMP);  // keep far points instead of clipping them at the far plane
  glClearColor(0, 0, 0, 0);
  glClearDepth(1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glUseProgram(program_);
  glUniform3fv(u_pivot_, 1, camera.pivot.data());
  glUniform3fv(u_right_, 1, camera.right.data());
  glUniform3fv(u_up_, 1, camera.up.data());
  glUniform3fv(u_toward_, 1, camera.toward.data());
  glUniform1f(u_eye_distance_, camera.eye_distance);
  glUniform1i(u_perspective_, camera.perspective ? 1 : 0);
  glUniform1f(u_focal_, camera.focal);
  glUniform1f(u_scale_, camera.scale);
  glUniform1f(u_near_, camera.near_plane);
  glUniform1f(u_far_, 1000.0F * camera.eye_distance);
  glUniform2f(u_viewport_, logical_width, logical_height);
  glUniform1i(u_rotation_, camera.rotation);
  if (retired) draw_set(retired_, retired_style);
  if (active) draw_set(active_, active_style);

  glBindFramebuffer(GL_READ_FRAMEBUFFER, msaa_fbo_);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, resolve_fbo_);
  glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

  glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(framebuffer));
  glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
  glUseProgram(static_cast<GLuint>(program));
  glBindVertexArray(static_cast<GLuint>(vao));
  glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(array_buffer));
  const auto restore = [](GLenum cap, GLboolean on) { on ? glEnable(cap) : glDisable(cap); };
  restore(GL_BLEND, blend);
  restore(GL_DEPTH_TEST, depth_test);
  restore(GL_SCISSOR_TEST, scissor);
  restore(GL_CULL_FACE, cull);
  restore(GL_DEPTH_CLAMP, depth_clamp);

  draw->AddCallback(composite_premultiplied, nullptr);
  // GL textures are bottom-up: flip v.
  draw->AddImage(static_cast<ImTextureID>(texture_), {min[0], min[1]}, {max[0], max[1]}, {0, 1}, {1, 0});
  draw->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
}

}  // namespace slam_native
