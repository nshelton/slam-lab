#pragma once
// Turns a LiveDepth image into an RGBA texture for drawing over the video:
// colour from the inverse depth (or from its uncertainty), opacity from the
// confidence. The image is coloured on the GPU straight into an OpenGL pixel
// buffer (CUDA-GL interop). Requires the app's OpenGL context.
#include "slam_native/live_depth.hpp"

#include <memory>

namespace slam_native {

enum class DepthOverlayColor { depth, sigma };

struct DepthOverlayStyle {
  // The colour ramp runs from `far` (blue) to `near` (red), in inverse depth.
  float near_inverse_depth{2.0F};
  float far_inverse_depth{0.2F};
  DepthOverlayColor color{DepthOverlayColor::depth};
  // Opacity by relative sigma (sigma / inverse depth): opaque up to
  // full_sigma, fading to nothing at max_sigma.
  float full_sigma{0.03F};
  float max_sigma{0.3F};
};

class DepthOverlay {
 public:
  DepthOverlay();
  ~DepthOverlay();
  DepthOverlay(const DepthOverlay&) = delete;
  DepthOverlay& operator=(const DepthOverlay&) = delete;

  // The texture of the depth image's last frame (straight alpha, source
  // orientation), or 0 when the image is not valid.
  unsigned int present(const LiveDepth& depth, const DepthOverlayStyle& style);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace slam_native
