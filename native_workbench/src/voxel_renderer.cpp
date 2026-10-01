#define GL_GLEXT_PROTOTYPES
#include "slam_native/voxel_renderer.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>

namespace slam_native {
namespace {

// Same display mapping and projection as the map points (point_cloud_renderer.cpp).
constexpr const char* kVertexShader = R"(#version 330 core
layout(location = 0) in vec3 a_vertex;  // cube corner, -1 .. 1
layout(location = 1) in vec3 a_normal;
layout(location = 2) in ivec3 i_coord;  // voxel index, camera-frame world
layout(location = 3) in vec4 i_color;   // rgb, a > 0.5: has an image colour
layout(location = 4) in int i_segment;
layout(location = 5) in float i_weight;
layout(location = 6) in float i_distance;  // mean signed distance, voxels
layout(location = 7) in vec3 i_visibility_negative;  // ambient visibility of the -x, -y, -z faces
layout(location = 8) in vec3 i_visibility_positive;  // and of the +x, +y, +z faces

uniform vec3 u_pivot, u_right, u_up, u_toward;
uniform float u_eye_distance, u_focal, u_scale, u_near, u_far;
uniform int u_perspective, u_rotation, u_color_mode, u_segment;  // colour: 0 plain, 1 image, 2 weight, 3 distance
uniform vec2 u_viewport;  // logical pixels
uniform float u_voxel_size, u_fill, u_min_weight, u_band, u_weight_span, u_ambient;

flat out vec3 v_color;

vec3 display(vec3 p) {
  float u = p.x, v = p.y;
  if (u_rotation == 90) { u = -p.y; v = p.x; }
  else if (u_rotation == 180) { u = -p.x; v = -p.y; }
  else if (u_rotation == 270) { u = p.y; v = -p.x; }
  return vec3(u, -v, p.z);
}

void main() {
  vec3 d = display((vec3(i_coord) + 0.5) * u_voxel_size) - u_pivot;
  vec3 center = vec3(dot(u_right, d), dot(u_up, d), u_eye_distance - dot(u_toward, d));
  bool perspective = u_perspective != 0;
  // Depth clamping is on, so a cube that reaches behind the eye is dropped whole.
  if ((u_segment >= 0 && i_segment != u_segment) || i_weight < u_min_weight || abs(i_distance) > u_band ||
      (perspective && center.z < u_near + u_voxel_size)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    v_color = vec3(0.0);
    return;
  }
  vec3 offset = display(a_vertex) * (0.5 * u_voxel_size * u_fill);
  vec3 e = center + vec3(dot(u_right, offset), dot(u_up, offset), -dot(u_toward, offset));
  vec2 ndc = 2.0 / u_viewport;
  if (perspective) {
    float a = (u_far + u_near) / (u_far - u_near), b = -2.0 * u_far * u_near / (u_far - u_near);
    gl_Position = vec4(e.x * u_focal * ndc.x, e.y * u_focal * ndc.y, a * e.z + b, e.z);
  } else {
    gl_Position = vec4(e.x * u_scale * ndc.x, e.y * u_scale * ndc.y, (e.z - u_eye_distance) / u_far, 1.0);
  }
  vec3 base = vec3(0.82);
  if (u_color_mode == 1 && i_color.a > 0.5) base = i_color.rgb;
  if (u_color_mode == 2) {  // dark blue (light) -> yellow (u_weight_span or more)
    float t = clamp(i_weight / max(u_weight_span, 1e-6), 0.0, 1.0);
    base = mix(vec3(0.16, 0.22, 0.50), vec3(1.0, 0.88, 0.30), t);
  }
  if (u_color_mode == 3) {  // blue (behind the points) - white (at them) - red (in front: free)
    float t = clamp(i_distance / max(u_band, 1e-6), -1.0, 1.0);
    base = t < 0.0 ? mix(vec3(0.95), vec3(0.20, 0.40, 0.90), -t) : mix(vec3(0.95), vec3(0.90, 0.25, 0.20), t);
  }
  // One fixed light in eye space (from the upper left, toward the viewer), so
  // a cube's three visible faces differ.
  vec3 n = display(a_normal);
  vec3 eye_normal = vec3(dot(u_right, n), dot(u_up, n), dot(u_toward, n));
  float diffuse = max(dot(eye_normal, normalize(vec3(-0.35, 0.60, 0.72))), 0.0);
  // Ambient visibility of this face (a_normal has one non-zero component). It
  // is a fraction of light; the colours are display values, hence the gamma.
  vec3 faces = a_normal.x + a_normal.y + a_normal.z > 0.0 ? i_visibility_positive : i_visibility_negative;
  float open = pow(dot(abs(a_normal), faces), 1.0 / 2.2);
  v_color = base * (0.45 + 0.55 * diffuse) * mix(1.0, open, u_ambient);
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
    std::fprintf(stderr, "VoxelRenderer: shader compile failed:\n%s\n", log);
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}

// Unit cube (-1 .. 1) as 12 triangles: position and face normal per vertex.
std::vector<float> cube() {
  std::vector<float> out;
  out.reserve(36 * 6);
  for (int axis = 0; axis < 3; ++axis) {
    for (int side = -1; side <= 1; side += 2) {
      const int u = (axis + 1) % 3, v = (axis + 2) % 3;
      const float corners[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
      for (int k : {0, 1, 2, 0, 2, 3}) {
        float position[3]{}, normal[3]{};
        position[axis] = static_cast<float>(side);
        position[u] = corners[k][0];
        position[v] = corners[k][1];
        normal[axis] = static_cast<float>(side);
        out.insert(out.end(), {position[0], position[1], position[2], normal[0], normal[1], normal[2]});
      }
    }
  }
  return out;
}

constexpr GLsizei kMeshVertices = 36;

void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

}  // namespace

VoxelRenderer::VoxelRenderer() = default;

VoxelRenderer::~VoxelRenderer() {
  if (resource_) cudaGraphicsUnregisterResource(resource_);
  if (!gl_ready_) return;
  glDeleteVertexArrays(1, &vao_);
  glDeleteBuffers(1, &buffer_);
  glDeleteBuffers(1, &mesh_);
  glDeleteProgram(program_);
}

VoxelHashStats VoxelRenderer::stats() const { return hash_ ? hash_->stats() : VoxelHashStats{}; }

void VoxelRenderer::set_samples(std::vector<VoxelSample> samples) {
  samples_ = std::move(samples);
  dirty_ = true;
}

void VoxelRenderer::set_config(const VoxelHashConfig& config) {
  if (config.voxel_size == config_.voxel_size && config.truncation_voxels == config_.truncation_voxels &&
      config.carve_weight == config_.carve_weight && config.max_ray_steps == config_.max_ray_steps)
    return;
  config_ = config;
  dirty_ = true;
}

bool VoxelRenderer::ensure_gl() {
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
    std::fprintf(stderr, "VoxelRenderer: program link failed:\n%s\n", log);
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
  u_voxel_size_ = uniform("u_voxel_size");
  u_fill_ = uniform("u_fill");
  u_min_weight_ = uniform("u_min_weight");
  u_band_ = uniform("u_band");
  u_color_mode_ = uniform("u_color_mode");
  u_segment_ = uniform("u_segment");
  u_weight_span_ = uniform("u_weight_span");
  u_ambient_ = uniform("u_ambient");

  const auto mesh = cube();
  glGenBuffers(1, &mesh_);
  glGenBuffers(1, &buffer_);
  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);
  glBindBuffer(GL_ARRAY_BUFFER, mesh_);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(mesh.size() * sizeof(float)), mesh.data(), GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<const void*>(3 * sizeof(float)));
  // Instance attributes; the buffer is (re)allocated in ensure_buffer(), which
  // keeps its name and therefore these bindings.
  glBindBuffer(GL_ARRAY_BUFFER, buffer_);
  const auto at = [](std::size_t offset) { return reinterpret_cast<const void*>(offset); };
  glEnableVertexAttribArray(2);
  glVertexAttribIPointer(2, 3, GL_INT, sizeof(VoxelInstance), at(offsetof(VoxelInstance, coord)));
  glVertexAttribDivisor(2, 1);
  glEnableVertexAttribArray(3);
  glVertexAttribPointer(3, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(VoxelInstance), at(offsetof(VoxelInstance, color)));
  glVertexAttribDivisor(3, 1);
  glEnableVertexAttribArray(4);
  glVertexAttribIPointer(4, 1, GL_INT, sizeof(VoxelInstance), at(offsetof(VoxelInstance, segment)));
  glVertexAttribDivisor(4, 1);
  glEnableVertexAttribArray(5);
  glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, sizeof(VoxelInstance), at(offsetof(VoxelInstance, weight)));
  glVertexAttribDivisor(5, 1);
  glEnableVertexAttribArray(6);
  glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, sizeof(VoxelInstance), at(offsetof(VoxelInstance, distance)));
  glVertexAttribDivisor(6, 1);
  glEnableVertexAttribArray(7);
  glVertexAttribPointer(7, 3, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(VoxelInstance),
                        at(offsetof(VoxelInstance, visibility_negative)));
  glVertexAttribDivisor(7, 1);
  glEnableVertexAttribArray(8);
  glVertexAttribPointer(8, 3, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(VoxelInstance),
                        at(offsetof(VoxelInstance, visibility_positive)));
  glVertexAttribDivisor(8, 1);
  gl_ready_ = true;
  return true;
}

void VoxelRenderer::ensure_buffer(std::size_t instances) {
  if (instances <= capacity_) return;
  if (resource_) {
    cuda_check(cudaGraphicsUnregisterResource(resource_), "unregister voxel instance buffer");
    resource_ = nullptr;
  }
  const std::size_t wanted = std::max({instances, 2 * capacity_, std::size_t{4096}});
  capacity_ = 0;
  glBindBuffer(GL_ARRAY_BUFFER, buffer_);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(wanted * sizeof(VoxelInstance)), nullptr, GL_DYNAMIC_DRAW);
  cuda_check(cudaGraphicsGLRegisterBuffer(&resource_, buffer_, cudaGraphicsRegisterFlagsWriteDiscard),
             "register voxel instance buffer with CUDA");
  capacity_ = wanted;
}

void VoxelRenderer::rebuild() {
  count_ = 0;
  if (samples_.empty()) {
    if (hash_) hash_->reset(config_);
    return;
  }
  if (!hash_) hash_ = std::make_unique<VoxelHash>(config_);
  hash_->reset(config_);
  hash_->integrate(samples_.data(), samples_.size());
}

// One more Monte Carlo pass over the faces, unless the averages have all the
// samples they keep. True when the visibilities changed.
bool VoxelRenderer::shade(const VoxelStyle& style) {
  if (!(style.ambient > 0) || !hash_ || hash_->stats().voxels == 0) return false;
  VoxelShadeConfig config;
  config.min_weight = style.min_weight;
  config.band = style.band;
  config.opaque_weight = style.opaque_weight;
  config.max_distance = style.ambient_distance;
  // Other solid voxels or other rays: the averages are of something else.
  const bool restart = config.min_weight != shade_.min_weight || config.band != shade_.band ||
                       config.opaque_weight != shade_.opaque_weight || config.max_distance != shade_.max_distance;
  if (restart) shade_samples_ = 0;
  if (shade_samples_ >= config.max_history) return false;
  shade_ = config;
  hash_->shade(config, restart);
  shade_samples_ += config.rays_per_face;
  return true;
}

void VoxelRenderer::upload() {
  count_ = 0;
  const std::size_t voxels = hash_ ? hash_->stats().voxels : 0;
  if (voxels == 0) return;
  ensure_buffer(voxels);
  const auto stream = static_cast<cudaStream_t>(hash_->stream());
  cuda_check(cudaGraphicsMapResources(1, &resource_, stream), "map voxel instance buffer");
  std::exception_ptr failure;
  try {
    void* instances = nullptr;
    std::size_t bytes = 0;
    cuda_check(cudaGraphicsResourceGetMappedPointer(&instances, &bytes, resource_), "get voxel instance pointer");
    count_ = std::min(hash_->compact(instances, capacity_), capacity_);
  } catch (...) {
    failure = std::current_exception();
  }
  cuda_check(cudaGraphicsUnmapResources(1, &resource_, stream), "unmap voxel instance buffer");
  if (failure) std::rethrow_exception(failure);
}

void VoxelRenderer::draw(const PointCloudCamera& camera, float logical_width, float logical_height,
                         const VoxelStyle& style) {
  if (!ensure_gl()) {
    error_ = "voxel shader failed to build";
    return;
  }
  // A failed rebuild is retried when the samples or the configuration change.
  const bool rebuilt = std::exchange(dirty_, false);
  if (rebuilt || error_.empty()) {
    try {
      const auto begin = std::chrono::steady_clock::now();
      if (rebuilt) {
        rebuild();
        shade_samples_ = 0;  // the averages carried over are short, see VoxelShadeConfig::history
      }
      const bool shaded = shade(style);
      if (rebuilt || shaded) upload();
      if (rebuilt)
        build_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
      error_.clear();
    } catch (const std::exception& e) {
      count_ = 0;
      error_ = e.what();
    }
  }
  if (count_ == 0) return;
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
  glUniform1f(u_voxel_size_, config_.voxel_size);
  glUniform1f(u_fill_, style.fill);
  glUniform1f(u_min_weight_, style.min_weight);
  glUniform1f(u_band_, style.band);
  glUniform1i(u_color_mode_, static_cast<int>(style.color));
  glUniform1i(u_segment_, style.segment);
  glUniform1f(u_weight_span_, style.weight_span);
  glUniform1f(u_ambient_, style.ambient);
  glBindVertexArray(vao_);
  glDrawArraysInstanced(GL_TRIANGLES, 0, kMeshVertices, static_cast<GLsizei>(count_));
}

}  // namespace slam_native
