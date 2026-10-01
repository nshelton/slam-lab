// Estimate the camera's horizontal field of view from a tracks CSV (headless).
#include "slam_native/focal_estimation.hpp"
#include "slam_native/track_io.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  using namespace slam_native;
  if (argc < 2 || std::string(argv[1]) == "--help") {
    std::cerr << "usage: slam-native-focal TRACKS.csv [FIRST_FRAME] [LAST_FRAME]\n";
    return argc < 2 ? 2 : 0;
  }
  try {
    const auto frames = read_tracks_csv(argv[1]);
    const std::uint64_t first = argc > 2 ? std::stoull(argv[2]) : 0;
    const std::uint64_t last = argc > 3 ? std::stoull(argv[3]) : ~0ULL;
    FocalEstimator estimator;
    int reported = 0;
    for (const auto& frame : frames) {
      if (frame.frame_index < first || frame.frame_index > last) continue;
      if (estimator.add(frame)) {
        const auto& e = estimator.estimate();
        if (e.pairs_used % 10 == 1 && e.pairs_used != reported) {
          reported = e.pairs_used;
          std::printf("frame %5llu: %3d pairs  estimate %.1f deg, k1 %.3f\n",
                      static_cast<unsigned long long>(frame.frame_index), e.pairs_used, e.best_hfov_degrees,
                      e.best_k1);
        }
      }
    }
    const auto& e = estimator.estimate();
    std::printf("\npairs: %d for distortion, %d for FOV, %d flat (no rotation), %d homography "
                "(rotation-only/planar), %d weak\n",
                e.pairs_distortion, e.pairs_used, e.pairs_flat, e.pairs_homography, e.pairs_weak);
    if (e.pairs_distortion == 0) return 1;
    std::printf("distortion k1 estimate: %.3f  (division model, radius / half-diagonal; < 0 = barrel)\n",
                e.best_k1);
    std::printf("\n    k1  epipolar RMS px\n");
    for (std::size_t j = 0; j < e.k1_values.size(); j += 3)
      std::printf("%6.2f  %.3f\n", e.k1_values[j], e.k1_residual_px[j]);
    if (e.pairs_used == 0) return 1;
    std::printf("\nFOV at k1 = 0 would be %.1f deg\n", e.best_hfov_at(0.0));
    std::printf("horizontal FOV estimate at k1 %.3f: %.1f deg  (fx = %.0f px at width %d)\n", e.best_k1, e.best_hfov_degrees,
                e.best_focal_px, e.width);
    std::printf("individual pair minima: p25 %.1f  median %.1f  p75 %.1f deg\n", e.pair_p25_degrees,
                e.pair_median_degrees, e.pair_p75_degrees);
    std::printf("\n  hfov  mean cost\n");
    const double low = *std::min_element(e.cost.begin(), e.cost.end());
    for (std::size_t k = 0; k < e.cost.size(); k += 10) {
      const int bar = static_cast<int>(200 * (e.cost[k] - low));
      std::printf("%6.0f  %.4f %s\n", e.hfov_degrees[k], e.cost[k], std::string(std::min(bar, 60), '#').c_str());
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
