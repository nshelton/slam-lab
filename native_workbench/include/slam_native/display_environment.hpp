#pragma once

#include <cstdlib>
#include <stdexcept>

namespace slam_native {

// Set before glfwInit/GLX initialization. These affect this process only.
// Explicit user overrides are preserved and checked by the interop preflight.
inline void configure_display_environment() {
#if defined(__linux__)
  if (setenv("__NV_PRIME_RENDER_OFFLOAD", "1", 0) != 0 ||
      setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 0) != 0) {
    throw std::runtime_error("Cannot configure NVIDIA OpenGL rendering");
  }
#endif
}

}  // namespace slam_native
