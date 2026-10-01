#include "lsd/place_recognition.hpp"

#include "pose_graph.hpp"

#include <algorithm>
#include <cmath>

namespace slam_native::lsd {

void finish_features(KeyframeFeatures& f) {
  f.global.assign(static_cast<std::size_t>(f.dimension), 0.0F);
  for (std::size_t i = 0; i < f.pixels.size(); ++i)
    for (int k = 0; k < f.dimension; ++k) f.global[k] += f.descriptors[i * f.dimension + k];
  double norm = 0;
  for (const float v : f.global) norm += v * v;
  norm = std::sqrt(norm);
  if (norm > 0)
    for (float& v : f.global) v = static_cast<float>(v / norm);
}

namespace {
// Inverse depth at a keyframe pixel: nearest valid within 2 px.
bool depth_at(const Keyframe& kf, const Vec2& p, Vec3& X) {
  const int u0 = static_cast<int>(std::lround(p.x())), v0 = static_cast<int>(std::lround(p.y()));
  double best = 1e9;
  float d = 0;
  for (int dv = -2; dv <= 2; ++dv)
    for (int du = -2; du <= 2; ++du) {
      const int u = u0 + du, v = v0 + dv;
      if (u < 0 || v < 0 || u >= kf.depth.idepth.width || v >= kf.depth.idepth.height || !kf.depth.valid(u, v)) continue;
      if (du * du + dv * dv < best) {
        best = du * du + dv * dv;
        d = kf.depth.idepth.at(u, v);
      }
    }
  if (!(d > 0)) return false;
  X = kf.pyramid.levels[0].camera.ray(p.x(), p.y()) / d;
  return true;
}

// Mutual nearest neighbours with cosine and ratio tests: pairs (index in a, index in b).
std::vector<std::pair<int, int>> match(const KeyframeFeatures& a, const KeyframeFeatures& b, const LoopConfig& c) {
  const int na = static_cast<int>(a.pixels.size()), nb = static_cast<int>(b.pixels.size()), dim = a.dimension;
  std::vector<int> best_a(na, -1), best_b(nb, -1);
  std::vector<float> sim_a(na, -2), second_a(na, -2), sim_b(nb, -2);
  for (int i = 0; i < na; ++i) {
    const float* x = &a.descriptors[static_cast<std::size_t>(i) * dim];
    for (int j = 0; j < nb; ++j) {
      const float* y = &b.descriptors[static_cast<std::size_t>(j) * dim];
      float s = 0;
      for (int k = 0; k < dim; ++k) s += x[k] * y[k];
      if (s > sim_a[i]) {
        second_a[i] = sim_a[i];
        sim_a[i] = s;
        best_a[i] = j;
      } else if (s > second_a[i]) {
        second_a[i] = s;
      }
      if (s > sim_b[j]) {
        sim_b[j] = s;
        best_b[j] = i;
      }
    }
  }
  std::vector<std::pair<int, int>> pairs;
  for (int i = 0; i < na; ++i) {
    const int j = best_a[i];
    if (j < 0 || best_b[j] != i || sim_a[i] < c.min_cosine) continue;
    // Distances of unit vectors: sqrt(2 - 2 cos).
    const double d1 = std::sqrt(std::max(0.0, 2.0 - 2.0 * sim_a[i]));
    const double d2 = std::sqrt(std::max(0.0, 2.0 - 2.0 * second_a[i]));
    if (d1 >= c.ratio * d2) continue;
    pairs.emplace_back(i, j);
  }
  return pairs;
}
}  // namespace

std::vector<LoopCandidate> find_loop_candidates(const std::vector<std::unique_ptr<Keyframe>>& keyframes, int id,
                                                const LoopConfig& config) {
  std::vector<LoopCandidate> out;
  const Keyframe& k = *keyframes[id];
  if (!config.enabled || k.features.empty() || k.features.global.empty()) return out;
  // Shortlist by global similarity.
  std::vector<std::pair<double, int>> ranked;
  for (const auto& other : keyframes) {
    if (!other || !other->finalized || other->id > id - config.min_keyframe_gap || other->features.global.empty())
      continue;
    double s = 0;
    for (std::size_t i = 0; i < k.features.global.size(); ++i) s += k.features.global[i] * other->features.global[i];
    ranked.emplace_back(-s, other->id);
  }
  std::sort(ranked.begin(), ranked.end());
  if (static_cast<int>(ranked.size()) > config.shortlist) ranked.resize(config.shortlist);

  for (const auto& [negative_similarity, other_id] : ranked) {
    const Keyframe& old = *keyframes[other_id];
    const auto pairs = match(k.features, old.features, config);
    if (static_cast<int>(pairs.size()) < config.min_matches) continue;
    std::vector<vo::Vec3> X, Y;
    std::vector<double> threshold;
    for (const auto& [a, b] : pairs) {
      Vec3 x, y;
      if (!depth_at(k, k.features.pixels[a], x) || !depth_at(old, old.features.pixels[b], y)) continue;
      X.push_back(x);
      Y.push_back(y);
      threshold.push_back(config.inlier_fraction * y.norm());
    }
    if (static_cast<int>(X.size()) < config.min_inliers) continue;
    Sim3 model;
    std::vector<char> inliers;
    const int count = vo::ransac_similarity(X, Y, threshold, config.ransac_iterations,
                                            static_cast<std::uint32_t>(id * 7919 + other_id), model, inliers);
    if (count < config.min_inliers) continue;
    out.push_back({other_id, model, static_cast<int>(pairs.size()), count, -negative_similarity});
  }
  std::sort(out.begin(), out.end(), [](const LoopCandidate& a, const LoopCandidate& b) { return a.inliers > b.inliers; });
  if (static_cast<int>(out.size()) > config.max_candidates) out.resize(config.max_candidates);
  return out;
}

}  // namespace slam_native::lsd
