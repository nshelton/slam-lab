#include "slam_native/cuda_frame_presenter.hpp"
#include "slam_native/display_environment.hpp"
#include "slam_native/video_decoder.hpp"
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>

// Optional desktop/GPU regression check: uses the production renderer and
// startup defaults, with no shell-provided NVIDIA offload settings required.
int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: slam-native-display-test VIDEO\n";
    return 2;
  }
  GLFWwindow* window = nullptr;
  int result = 0;
  try {
    slam_native::configure_display_environment();
    if (!glfwInit()) throw std::runtime_error("Cannot initialize GLFW");
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(32, 32, "Display test", nullptr, nullptr);
    if (!window) throw std::runtime_error("Cannot create OpenGL context");
    glfwMakeContextCurrent(window);
    slam_native::validate_cuda_gl_display();
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << '\n';
    auto decoder = slam_native::make_ffmpeg_cuda_decoder();
    decoder->open(argv[1]);
    auto presenter = slam_native::make_cuda_gl_presenter();
    for (int index = 0; index < 3; ++index) {
      slam_native::GpuFrame frame;
      if (!decoder->next(frame)) throw std::runtime_error("Need at least three video frames");
      const auto texture = presenter->present(frame);
      if (!glIsTexture(texture)) throw std::runtime_error("Missing display texture");
      glFinish();
      if (glGetError() != GL_NO_ERROR) throw std::runtime_error("OpenGL display error");
    }
    std::cout << "Decoded and displayed three frames with CUDA/OpenGL sharing.\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    result = 1;
  }
  if (window) glfwDestroyWindow(window);
  glfwTerminate();
  return result;
}
