#pragma once
// Self-calibrated uncertainty of network depth (DEPTH_INTEGRATION.md, Phase
// 0b). The VO's well-triangulated landmarks are references: at a keyframe,
// each gives a residual in log depth after per-keyframe scale alignment,
//     r = log z_net - log z_vo - median_keyframe(...),
// with known geometric variance (MapPoint::depth_sigma_ratio^2). The
// network's own log-depth variance is modelled from per-pixel cues,
//     sigma_net^2 = exp(w . (1, cues)),   r ~ N(0, sigma_net^2 + sigma_geo^2),
// and w is refitted by maximum likelihood (Fisher scoring) over a rolling
// window of references. Measured offline (tools/depth_confidence.py): this beats
// a constant sigma clearly on dreamworks and ties on castle.
// Header-only dependencies; no VO internals, no GPU.
#include <array>
#include <cstddef>
#include <vector>

namespace slam_native {

// Per-pixel cues (all cheap to compute from the depth map and the landmarks).
struct DepthCues {
  float edge{};             // log(max / min) of depth over the 3x3 output neighbourhood
  float log_depth{};        // log(metres)
  float radius{};           // distance from the image centre / half-diagonal (0..1)
  float nearest_landmark{}; // pixel distance to the nearest reference landmark / image width
};

struct DepthConfidenceConfig {
  std::size_t max_references{20000};  // rolling window
  std::size_t min_references{200};    // below this, sigma() returns the prior
  double prior_sigma{0.2};            // log depth; the constant fit was 0.20-0.23
  double gross_log{0.6931471805599453};  // |r| > log 2: outlier, not used in the fit
  double ridge{1.0};                  // pull of w toward the constant prior (stability)
};

class DepthConfidenceModel {
 public:
  static constexpr std::size_t kTerms = 5;  // bias + 4 cues

  explicit DepthConfidenceModel(DepthConfidenceConfig config = {});

  // One reference: cues at the landmark's pixel, its log-depth residual after
  // per-keyframe alignment, and its geometric log-depth sigma.
  void add_reference(const DepthCues& cues, double residual, double geometric_sigma);
  // Refit w on the current window (a few Fisher-scoring steps from the last w).
  void fit(int iterations = 8);
  // Predicted network sigma in log depth (≈ relative depth error).
  [[nodiscard]] double sigma(const DepthCues& cues) const;
  // 0..1 for display: 1 / (1 + (sigma / reference)^2), reference = prior_sigma.
  [[nodiscard]] double confidence(const DepthCues& cues) const;

  [[nodiscard]] std::size_t references() const { return count_; }
  [[nodiscard]] const std::array<double, kTerms>& weights() const { return w_; }
  [[nodiscard]] bool fitted() const { return fitted_; }

 private:
  struct Reference {
    std::array<double, kTerms> x;
    double r2, geo2;
  };
  [[nodiscard]] std::array<double, kTerms> features(const DepthCues& cues) const;
  DepthConfidenceConfig config_;
  std::vector<Reference> references_;  // ring buffer
  std::size_t next_{}, count_{};
  std::array<double, kTerms> w_{};
  bool fitted_{};
};

}  // namespace slam_native
