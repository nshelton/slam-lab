#include "lsd/image.hpp"

#include <algorithm>

namespace slam_native::lsd {

ImageF downsample(const ImageF& image) {
  ImageF out(image.width / 2, image.height / 2);
  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < out.width; ++x) {
      out.at(x, y) = 0.25F * (image.at(2 * x, 2 * y) + image.at(2 * x + 1, 2 * y) + image.at(2 * x, 2 * y + 1) +
                              image.at(2 * x + 1, 2 * y + 1));
    }
  }
  return out;
}

void gradients(const ImageF& image, ImageF& gx, ImageF& gy) {
  const int w = image.width, h = image.height;
  gx = ImageF(w, h);
  gy = ImageF(w, h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const int xl = x > 0 ? x - 1 : x, xr = x + 1 < w ? x + 1 : x;
      const int yu = y > 0 ? y - 1 : y, yd = y + 1 < h ? y + 1 : y;
      gx.at(x, y) = (image.at(xr, y) - image.at(xl, y)) / static_cast<float>(xr - xl);
      gy.at(x, y) = (image.at(x, yd) - image.at(x, yu)) / static_cast<float>(yd - yu);
    }
  }
}

namespace {
ImageF downsample_mask(const ImageF& mask) {
  ImageF out(mask.width / 2, mask.height / 2);
  for (int y = 0; y < out.height; ++y)
    for (int x = 0; x < out.width; ++x)
      out.at(x, y) = std::min({mask.at(2 * x, 2 * y), mask.at(2 * x + 1, 2 * y), mask.at(2 * x, 2 * y + 1),
                               mask.at(2 * x + 1, 2 * y + 1)});
  return out;
}
}  // namespace

Pyramid build_pyramid(const ImageF& image, const Camera& camera, int levels, const ImageF* mask) {
  Pyramid pyramid;
  Level level;
  level.camera = camera;
  level.camera.width = image.width;
  level.camera.height = image.height;
  level.image = image;
  if (mask && mask->width == image.width && mask->height == image.height) level.mask = *mask;
  for (int l = 0; l < levels; ++l) {
    gradients(level.image, level.gx, level.gy);
    pyramid.levels.push_back(level);
    if (level.image.width / 2 < 4 || level.image.height / 2 < 4) break;
    Level next;
    next.camera = level.camera.half();
    next.image = downsample(level.image);
    if (!level.mask.empty()) next.mask = downsample_mask(level.mask);
    level = std::move(next);
  }
  return pyramid;
}

}  // namespace slam_native::lsd
