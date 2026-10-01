// GPU flow-tracker checks (built with the CUDA app; needs a GPU).
#include "slam_native/flow_tracker.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

using namespace slam_native;
namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
template <class F> void rejects(F action, const char* message) {
  bool rejected = false;
  try { action(); } catch (const std::exception&) { rejected = true; }
  require(rejected, message);
}
std::vector<float> one_hot(int c) {
  std::vector<float> d(256, 0.0F);
  d[static_cast<std::size_t>(c) % 256] = 1.0F;
  return d;
}
// Points with descriptor classes (default: class = index, so all distinct).
FrameFeatures frame(std::uint64_t index, std::vector<Keypoint> points, std::vector<int> classes = {},
                    int size = 96) {
  FrameFeatures result;
  result.frame_index = index;
  result.timestamp_ns = static_cast<std::int64_t>(index + 1) * 16'666'667;
  result.pts = static_cast<std::int64_t>(index);
  result.width = result.height = size;
  result.keypoints = std::move(points);
  for (std::size_t k = 0; k < result.keypoints.size(); ++k) {
    const auto d = one_hot(classes.empty() ? static_cast<int>(k) : classes[k]);
    result.descriptors.insert(result.descriptors.end(), d.begin(), d.end());
  }
  return result;
}
FlowField uniform(int size, float dx, float dy) {
  FlowField flow;
  flow.width = flow.height = size / 4;
  flow.vectors.assign(static_cast<std::size_t>(flow.width) * flow.height, {dx, dy});
  flow.backward.assign(flow.vectors.size(), {-dx, -dy});
  return flow;
}
FlowTrackerConfig settings(int coast = 3, int rounds = 4) {
  FlowTrackerConfig config;
  config.max_coast_frames = coast;
  config.assignment_rounds = rounds;
  config.flow_sigma_px = 0;  // snap to detections: exact positions in the rule checks
  return config;
}
int index_of(const FrameFeatures& f, std::uint64_t id) {
  for (std::size_t k = 0; k < f.landmark_ids.size(); ++k) if (f.landmark_ids[k] == id) return int(k);
  return -1;
}

void basic_rules() {
  const auto flow = uniform(96, 2, 1);
  FlowTracker tracker(settings(0));
  auto first = frame(0, {{10, 10, 0.9F}, {70, 70, 0.8F}});
  tracker.associate(first, std::nullopt);
  require(first.keypoints.size() == 2 && first.new_landmarks == 2, "Must seed SuperPoints");
  auto second = frame(1, {{13, 12, 0.9F}, {30, 30, 0.8F}}, {0, 1});
  tracker.associate(second, flow);
  require(second.landmark_ids[0] == first.landmark_ids[0] && second.keypoints[0].x == 13 &&
          second.keypoints[0].y == 12, "Matched track must snap to SuperPoint");
  require(second.flow_predictions[0].x == 12 && second.flow_predictions[0].y == 11 &&
          std::abs(second.correction_distances[0] - std::sqrt(2.0F)) < 1e-5F &&
          std::abs(second.landmark_similarities[0] - 1.0F) < 1e-3F,
          "Must retain prediction, correction and descriptor similarity");
  require(second.track_descriptors.size() == 2 * 256 && second.track_descriptors[0] == 1.0F &&
          second.track_descriptors[256 + 1] == 1.0F && second.track_descriptors[1] == 0.0F,
          "Each track must carry the descriptor of the detection it matched this frame");
  require(second.keypoints.size() == 2 && second.new_landmarks == 1 && second.matched_landmarks == 1 &&
          !tracker.find(first.landmark_ids[1]), "With coasting disabled, unmatched tracks die");
  auto empty = frame(2, {});
  tracker.associate(empty, flow);
  require(tracker.active_count() == 0, "No detections and no coasting must mean no tracks");

  FlowTracker adjustable(settings(0));
  auto seed = frame(0, {{10, 10, 1}});
  adjustable.associate(seed, std::nullopt);
  auto too_far = frame(1, {{20, 11, 1}}, {0});
  adjustable.associate(too_far, flow);
  require(too_far.matched_landmarks == 0, "Default radius accepted an 8-pixel correction");
  adjustable.set_association_radius(10);
  auto wider = frame(2, {{30, 12, 1}}, {0});
  adjustable.associate(wider, flow);
  require(wider.matched_landmarks == 1 && wider.landmark_ids[0] == too_far.landmark_ids[0],
          "Live radius update must apply without resetting tracks");
  rejects([&] { adjustable.set_association_radius(0); }, "Invalid radius accepted");

  FlowTracker breaks(settings(3));
  auto a = frame(0, {{10, 10, 1}});
  breaks.associate(a, std::nullopt);
  auto missing = frame(1, {{12, 11, 1}}, {0});
  breaks.associate(missing, std::nullopt);
  require(missing.new_landmarks == 1 && missing.coasted_landmarks == 0, "Missing flow must break identity");
  auto jump = frame(9, {{14, 12, 1}}, {0});
  breaks.associate(jump, flow);
  require(jump.new_landmarks == 1, "Nonconsecutive frames must not associate");
  FlowField malformed = flow;
  malformed.vectors.pop_back();
  auto bad = frame(10, {{16, 13, 1}}, {0});
  breaks.associate(bad, malformed);
  require(bad.new_landmarks == 1, "Malformed flow must not imply zero motion");
  FlowTracker border(settings(3));
  auto edge = frame(0, {{95, 13, 1}});
  border.associate(edge, std::nullopt);
  auto outside = frame(1, {{95, 13, 1}}, {0});
  border.associate(outside, uniform(96, 3, 0));
  require(outside.new_landmarks == 1 && outside.coasted_landmarks == 0 && outside.keypoints.size() == 1,
          "Out-of-image prediction must neither associate nor coast");
  rejects([] { FlowTracker invalid({-1, 1000}); }, "Invalid settings accepted");
}

void coasting() {
  const auto flow = uniform(96, 2, 1);
  FlowTracker tracker(settings(2));
  auto seed = frame(0, {{10, 10, 1}, {60, 60, 1}}, {0, 1});
  tracker.associate(seed, std::nullopt);
  const auto id = seed.landmark_ids[0];
  auto gap1 = frame(1, {{62, 61, 1}}, {1});
  tracker.associate(gap1, flow);
  int k = index_of(gap1, id);
  require(k >= 0 && !gap1.superpoint_supported[k] && gap1.keypoints[k].x == 12 &&
          gap1.keypoints[k].y == 11 && gap1.coasted_landmarks == 1,
          "A missed track must coast to its flow prediction");
  auto gap2 = frame(2, {{64, 62, 1}}, {1});
  tracker.associate(gap2, flow);
  k = index_of(gap2, id);
  require(k >= 0 && gap2.keypoints[k].x == 14, "Second coast frame");
  auto back = frame(3, {{16.5F, 12.5F, 1}, {66, 63, 1}}, {0, 1});
  tracker.associate(back, flow);
  k = index_of(back, id);
  require(k >= 0 && back.superpoint_supported[k] && back.keypoints[k].x == 16.5F &&
          back.matched_landmarks == 2 && tracker.find(id)->observation_count == 2,
          "A coasting track must be reacquired by a detection with its ID");
  for (int f = 4; f <= 6; ++f) {
    auto lonely = frame(f, {{60.0F + 2 * f, 60.0F + f, 1}}, {1});
    tracker.associate(lonely, flow);
    if (f == 6) require(index_of(lonely, id) < 0, "Coasting must stop after max_coast_frames");
  }

  // Forward-backward inconsistency prevents coasting but not a supported match.
  FlowTracker checked(settings(3));
  auto s = frame(0, {{10, 10, 1}, {50, 50, 1}}, {0, 1});
  checked.associate(s, std::nullopt);
  auto inconsistent = uniform(96, 2, 1);
  inconsistent.backward.assign(inconsistent.vectors.size(), {3, 3});
  auto f1 = frame(1, {{52, 51, 1}}, {1});
  checked.associate(f1, inconsistent);
  require(f1.matched_landmarks == 1 && f1.coasted_landmarks == 0 && f1.keypoints.size() == 1,
          "Failed forward-backward check must not coast but may still match");
}

void descriptor_gate_and_rounds() {
  const auto flow = uniform(96, 2, 1);
  FlowTracker gated(settings(0));
  auto seed = frame(0, {{10, 10, 1}}, {0});
  gated.associate(seed, std::nullopt);
  auto impostor = frame(1, {{12, 11, 1}}, {5});
  gated.associate(impostor, flow);
  require(impostor.matched_landmarks == 0 && impostor.new_landmarks == 1,
          "Dissimilar descriptor at the predicted location must not associate");

  // Tracks A (10,10) and B (16,10) predict (12,11) and (18,11). Both prefer
  // d0 (15.5,11); B is closer. A's second choice d1 (12,15) is in its radius only. One round leaves A
  // unmatched; more rounds let A take its second choice.
  for (int rounds : {1, 4}) {
    FlowTracker tracker(settings(0, rounds));
    auto s = frame(0, {{10, 10, 1}, {16, 10, 0.9F}}, {0, 0});
    tracker.associate(s, std::nullopt);
    auto next = frame(1, {{15.5F, 11, 1}, {12, 15, 0.5F}}, {0, 0});
    tracker.associate(next, flow);
    require(next.matched_landmarks == (rounds == 1 ? 1U : 2U), "Assignment rounds mismatch");
    if (rounds == 4) {
      require(next.landmark_ids[0] == s.landmark_ids[1] && next.landmark_ids[1] == s.landmark_ids[0],
              "Second-choice assignment must keep the closest pair");
    }
  }
}

void cap_priority() {
  const auto flow = uniform(96, 2, 1);
  FlowTracker capped({6, 2, 0.7F, 1, 0});
  auto many = frame(0, {{10, 10, 0.1F}, {20, 20, 0.9F}, {30, 30, 0.7F}});
  capped.associate(many, std::nullopt);
  require(many.keypoints.size() == 2 && many.keypoints[0].x == 20 && many.keypoints[1].x == 30,
          "Cap must retain strongest detections");
  auto next_many = frame(1, {{12, 11, 0.1F}, {22, 21, 0.9F}, {32, 31, 0.7F}}, {0, 1, 2});
  capped.associate(next_many, flow);
  require(next_many.matched_landmarks == 2 && capped.active_count() == 2, "Cap changed identities");
  auto weak = frame(2, {{70, 70, 0.99F}, {80, 80, 0.98F}, {24, 22, 0.1F}, {34, 32, 0.05F}},
                    {7, 8, 1, 2});
  capped.associate(weak, flow);
  require(weak.matched_landmarks == 2 && weak.new_landmarks == 0 &&
          weak.landmark_ids == next_many.landmark_ids && weak.keypoints[0].x == 24,
          "Low-score continuations must survive ahead of stronger new detections");
  auto refill = frame(3, {{50, 50, 0.5F}, {70, 70, 0.9F}, {26, 23, 0.01F}, {80, 80, 0.7F}},
                      {9, 10, 1, 11});
  capped.associate(refill, flow);
  require(refill.matched_landmarks == 1 && refill.new_landmarks == 1 &&
          refill.landmark_ids[0] == weak.landmark_ids[0] && refill.keypoints[1].x == 70 &&
          !capped.find(weak.landmark_ids[1]),
          "Vacant slots must use strongest unmatched detections");
}

// CPU reference of gating + propose/accept rounds, on exactly representable
// inputs (integer positions and flow, one-hot descriptors).
std::vector<int> reference(const std::vector<Keypoint>& previous, const std::vector<int>& previous_class,
                           const std::vector<Keypoint>& current, const std::vector<int>& current_class,
                           float fx, float fy, float radius, float min_similarity, int rounds) {
  const float r2 = radius * radius;
  struct Candidate { float cost; int j; };
  std::vector<std::vector<Candidate>> candidates(previous.size());
  for (std::size_t i = 0; i < previous.size(); ++i) {
    const float px = previous[i].x + fx, py = previous[i].y + fy;
    for (std::size_t j = 0; j < current.size(); ++j) {
      const float dx = current[j].x - px, dy = current[j].y - py, d2 = dx * dx + dy * dy;
      const float similarity = previous_class[i] == current_class[j] ? 1.0F : 0.0F;
      if (d2 <= r2 && similarity >= min_similarity)
        candidates[i].push_back({d2 / r2 + (1.0F - similarity), int(j)});
    }
    std::stable_sort(candidates[i].begin(), candidates[i].end(),
                     [](auto a, auto b) { return a.cost < b.cost; });
    if (candidates[i].size() > 8) candidates[i].resize(8);
  }
  std::vector<int> track_match(previous.size(), -1), detection_match(current.size(), -1);
  for (int round = 0; round < rounds; ++round) {
    std::vector<std::pair<float, int>> best(current.size(), {INFINITY, -1});
    for (std::size_t i = 0; i < previous.size(); ++i) {
      if (track_match[i] >= 0) continue;
      for (auto c : candidates[i]) {
        if (detection_match[c.j] >= 0) continue;
        if (std::make_pair(c.cost, int(i)) < best[c.j]) best[c.j] = {c.cost, int(i)};
        break;
      }
    }
    for (std::size_t j = 0; j < current.size(); ++j) {
      if (best[j].second < 0) continue;
      detection_match[j] = best[j].second;
      track_match[best[j].second] = int(j);
    }
  }
  return detection_match;
}

void matches_reference() {
  std::mt19937 rng(3);
  const int size = 256;  // dense: several gated candidates per track, many conflicts
  std::vector<Keypoint> previous, current;
  std::vector<int> previous_class, current_class;
  std::set<std::pair<int, int>> used;
  auto unique_point = [&](float score) {
    for (;;) {
      const int x = 8 + int(rng() % (size - 40)), y = 8 + int(rng() % (size - 40));
      if (used.insert({x, y}).second) return Keypoint{float(x), float(y), score};
    }
  };
  for (int i = 0; i < 1000; ++i) {
    previous.push_back(unique_point(1.0F - i * 1e-4F));
    previous_class.push_back(int(rng() % 3));
  }
  used.clear();
  for (int j = 0; j < 3000; ++j) {
    current.push_back(unique_point(1.0F - j * 1e-4F));
    current_class.push_back(int(rng() % 3));
  }
  for (int rounds : {1, 4}) {
    FlowTrackerConfig config = settings(0, rounds);
    config.max_tracks = 4000;
    config.min_descriptor_similarity = 0.5F;
    FlowTracker tracker(config);
    auto seed = frame(0, previous, previous_class, size);
    tracker.associate(seed, std::nullopt);  // IDs follow score order == index order
    auto next = frame(1, current, current_class, size);
    tracker.associate(next, uniform(size, 2, 1));
    std::map<std::pair<float, float>, int> position;
    for (std::size_t j = 0; j < current.size(); ++j) position[{current[j].x, current[j].y}] = int(j);
    std::vector<int> actual(current.size(), -1);
    for (std::size_t k = 0; k < next.keypoints.size(); ++k) {
      if (next.landmark_ids[k] >= previous.size()) continue;  // new track
      actual[position.at({next.keypoints[k].x, next.keypoints[k].y})] = int(next.landmark_ids[k]);
    }
    const auto expected = reference(previous, previous_class, current, current_class, 2, 1,
                                    config.association_radius, 0.5F, rounds);
    require(actual == expected, "GPU assignment disagrees with CPU reference (rounds=" +
                                    std::to_string(rounds) + ")");
    std::size_t matched = 0;
    for (int value : actual) matched += value >= 0;
    std::cout << "reference agreement, rounds=" << rounds << ": " << matched << " matches\n";
    static std::size_t single_round = 0;
    if (rounds == 1) single_round = matched;
    else require(matched > single_round, "Extra rounds must recover second-choice matches");
  }
}

// Synthetic translation with detector dropouts and distractors. Ground truth
// identities let us count ID switches and measure track length.
struct SyntheticResult { double mean_length; std::size_t switches; };
SyntheticResult synthetic(int coast) {
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> u(0, 1);
  const int size = 640, frames = 120, points = 400;
  const float vx = 3.25F, vy = -1.5F;
  std::vector<float> x(points), y(points);
  for (int p = 0; p < points; ++p) { x[p] = 40 + u(rng) * 560; y[p] = 40 + u(rng) * 560; }
  FlowTrackerConfig config = settings(coast);
  config.max_tracks = 2000;
  FlowTracker tracker(config);
  std::map<std::uint64_t, int> truth;            // track id -> ground-truth point
  std::map<std::uint64_t, std::size_t> observed; // track id -> supported observations
  std::size_t switches = 0;
  for (int f = 0; f < frames; ++f) {
    std::vector<Keypoint> detections;
    std::vector<int> classes, owner;
    for (int p = 0; p < points; ++p) {
      const float px = x[p] + vx * f, py = y[p] + vy * f;
      if (px < 0 || py < 0 || px >= size || py >= size) continue;
      if (u(rng) < 0.12F) continue;  // detector dropout
      // SuperPoint-like localization noise (quantization at 1.875 px).
      const float qx = std::round(px / 1.875F) * 1.875F, qy = std::round(py / 1.875F) * 1.875F;
      detections.push_back({std::clamp(qx, 0.0F, size - 1.0F), std::clamp(qy, 0.0F, size - 1.0F), 0.5F + 0.4F * u(rng)});
      classes.push_back(p % 200);
      owner.push_back(p);
    }
    for (int d = 0; d < 40; ++d) {  // distractors with random appearance
      detections.push_back({u(rng) * (size - 1), u(rng) * (size - 1), 0.3F * u(rng)});
      classes.push_back(200 + int(rng() % 56));
      owner.push_back(-1);
    }
    auto features = frame(std::uint64_t(f), detections, classes, size);
    const auto raw = features.keypoints;
    tracker.associate(features, f == 0 ? std::nullopt : std::optional<FlowField>(uniform(size, vx, vy)));
    for (std::size_t k = 0; k < features.keypoints.size(); ++k) {
      if (!features.superpoint_supported[k]) continue;
      const auto point = features.keypoints[k];
      int source = -1;
      for (std::size_t r = 0; r < raw.size(); ++r)
        if (raw[r].x == point.x && raw[r].y == point.y) { source = owner[r]; break; }
      const auto id = features.landmark_ids[k];
      auto it = truth.find(id);
      if (it == truth.end()) truth[id] = source;
      else if (it->second != source) { ++switches; it->second = source; }
      ++observed[id];
    }
  }
  double total = 0;
  std::size_t count = 0;
  for (const auto& [id, n] : observed) if (truth[id] >= 0) { total += n; ++count; }
  return {total / std::max<std::size_t>(count, 1), switches};
}
}  // namespace

// Kalman position fusion: one point moving with exact flow, noisy detections.
// Fusion must follow the motion without lag and average out detection noise.
double fusion_rms(float flow_sigma, double* bias = nullptr) {
  std::mt19937 rng(11);
  std::normal_distribution<float> noise(0, 1);
  const float vx = 0.25F, vy = 0.1F;
  const auto flow = uniform(96, vx, vy);
  auto config = settings(0);
  config.flow_sigma_px = flow_sigma;
  config.detection_sigma_px = 1.0F;
  FlowTracker tracker(config);
  double squared = 0, sum_x = 0;
  int count = 0;
  std::uint64_t id = 0;
  for (int f = 0; f < 280; ++f) {
    const float tx = 10 + vx * f, ty = 30 + vy * f;
    auto current = frame(static_cast<std::uint64_t>(f), {{tx + noise(rng), ty + noise(rng), 1}}, {0});
    tracker.associate(current, f == 0 ? std::optional<FlowField>{} : std::optional<FlowField>{flow});
    require(current.keypoints.size() == 1, "fusion test lost its track");
    if (f == 0) id = current.landmark_ids[0];
    require(current.landmark_ids[0] == id, "fusion test switched track");
    if (f >= 40) {  // after the filter settles
      const double ex = current.keypoints[0].x - tx, ey = current.keypoints[0].y - ty;
      squared += ex * ex + ey * ey;
      sum_x += ex;
      ++count;
    }
  }
  if (bias) *bias = sum_x / count;
  return std::sqrt(squared / (2 * count));  // per axis
}

void position_fusion() {
  // Exact first update: seed P = det^2 = 1; predict P = 1 + q; K = P / (P + 1).
  {
    auto config = settings(0);
    config.flow_sigma_px = 0.5F;
    FlowTracker tracker(config);
    auto seed = frame(0, {{10, 10, 1}});
    tracker.associate(seed, std::nullopt);
    auto next = frame(1, {{14, 11, 1}}, {0});
    tracker.associate(next, uniform(96, 2, 1));  // prediction (12, 11)
    const float gain = 1.25F / 2.25F;
    require(std::abs(next.keypoints[0].x - (12 + gain * 2)) < 1e-4F && std::abs(next.keypoints[0].y - 11) < 1e-4F,
            "Kalman update must blend prediction and detection");
    require(std::abs(next.correction_distances[0] - 2.0F) < 1e-5F, "Correction stays |detection - prediction|");
  }
  double bias = 0;
  const double off = fusion_rms(0.0F), fused = fusion_rms(0.1F, &bias);
  std::cout << "position fusion: per-axis error " << off << " px snapped -> " << fused << " px fused (bias "
            << bias << " px)\n";
  require(off > 0.8 && off < 1.2, "Snapped tracks must carry the detection noise");
  require(fused < 0.5 * off, "Fusion must reduce position noise");
  require(std::abs(bias) < 0.15, "Fusion with exact flow must not lag");
}

int main() {
  try {
    basic_rules();
    coasting();
    descriptor_gate_and_rounds();
    cap_priority();
    matches_reference();
    position_fusion();
    const auto without = synthetic(0), with = synthetic(3);
    std::cout << "synthetic translation: mean true-track length " << without.mean_length
              << " (no coasting, " << without.switches << " ID switches) -> " << with.mean_length
              << " (coast 3, " << with.switches << " ID switches)\n";
    require(with.switches == 0 && without.switches == 0, "Synthetic tracking produced ID switches");
    require(with.mean_length > 3 * without.mean_length, "Coasting must bridge detector dropouts");
    std::cout << "flow tracker checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
