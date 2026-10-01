#include "lsd/odometry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace slam_native::lsd {
namespace {
using Clock = std::chrono::steady_clock;

double median(std::vector<double>& values) {
  const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  return *middle;
}
}  // namespace

Odometry::Odometry(Camera camera, OdometryConfig config) : camera_(camera), config_(config) {}

const FrameResult& Odometry::track(std::uint64_t frame_index, std::int64_t timestamp_ns, const ImageF& image,
                                   Se3TrackingDebug* debug) {
  const auto start = Clock::now();
  const FrameResult previous = last_;
  last_ = {};
  last_.frame_index = frame_index;
  last_.timestamp_ns = timestamp_ns;
  last_image_ = image;
  event_.clear();

  const Keyframe* kf = current_keyframe();
  if (!kf) {
    last_.state = TrackingState::waiting;
    last_.keyframe_wanted = true;
    last_.ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return last_;
  }

  // Initial guess: the previous frame's pose relative to the keyframe,
  // extrapolated at constant velocity.
  SE3 initial = last_relative_;
  if (config_.constant_velocity && has_previous_ && previous.state == TrackingState::tracking)
    initial = (last_relative_ * previous_relative_.inverse()) * last_relative_;

  const Pyramid pyramid = build_pyramid(image, camera_, config_.pyramid_levels, mask_.empty() ? nullptr : &mask_);
  last_.tracking = track_se3(kf->reference, pyramid, initial, config_.tracker, debug);
  last_.keyframe = kf->id;
  if (last_.tracking.success) {
    last_.state = TrackingState::tracking;
    last_.pose = Sim3::from(last_.tracking.pose) * kf->pose;
    const double d = last_.tracking.pose.t.norm() * kf->mean_idepth * config_.keyframe_distance_weight;
    const double u = (1 - last_.tracking.usage) * config_.keyframe_usage_weight;
    last_.keyframe_score = d * d + u * u;
    last_.keyframe_wanted = last_.keyframe_score >= 1;
    trajectory_.push_back({frame_index, timestamp_ns, kf->id, last_.tracking.pose, last_.pose, false});
    has_previous_ = previous.state == TrackingState::tracking;
    previous_relative_ = last_relative_;
    last_relative_ = last_.tracking.pose;
    // Refine the keyframe's depth with this frame (a frame about to become the
    // next keyframe would only add a nearly zero-baseline observation).
    Keyframe& current = *keyframes_.back();
    if (config_.stereo && current.filter && !last_.keyframe_wanted) {
      const auto mapping_start = Clock::now();
      current.filter->config() = config_.filter;
      last_.stereo = current.filter->observe(pyramid.levels[0], last_.tracking.pose);
      current.filter->regularize();
      ++current.observations;
      refresh(current);
      last_.mapping_ms = std::chrono::duration<double, std::milli>(Clock::now() - mapping_start).count();
    }
  } else {
    // Lost: hold the last good pose and restart from a fresh keyframe there.
    last_.state = TrackingState::lost;
    last_.pose = previous.pose;
    last_.keyframe_wanted = true;
    has_previous_ = false;
    event_ = "tracking lost (usage " + std::to_string(last_.tracking.usage).substr(0, 4) + ")";
  }
  last_.ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  return last_;
}

void Odometry::refresh(Keyframe& kf) const {
  kf.depth = kf.filter->tracking_map();
  kf.reference = TrackingReference(kf.pyramid, kf.depth);
  const Level& level = kf.pyramid.levels[0];
  kf.cloud.clear();
  kf.cloud_colors.clear();
  kf.cloud_sigma.clear();
  double sum = 0;
  int points = 0;
  for (int y = 0; y < level.image.height; ++y) {
    for (int x = 0; x < level.image.width; ++x) {
      if (!kf.depth.valid(x, y)) continue;
      const double d = kf.depth.idepth.at(x, y);
      sum += d;
      ++points;
      const Vec3 X = level.camera.ray(x, y) / d;
      kf.cloud.push_back({static_cast<float>(X.x()), static_cast<float>(X.y()), static_cast<float>(X.z())});
      const auto g = static_cast<std::uint8_t>(std::clamp(level.image.at(x, y), 0.0F, 255.0F));
      kf.cloud_colors.push_back({g, g, g});
      kf.cloud_sigma.push_back(static_cast<float>(std::sqrt(kf.depth.variance.at(x, y)) / d));
    }
  }
  kf.points = points;
  kf.mean_idepth = points > 0 ? sum / points : 0;
  ++kf.version;
}

void Odometry::finalize(Keyframe& kf) {
  kf.filter.reset();
  kf.depth_levels = depth_pyramid(kf.depth, kf.pyramid.size());
  kf.finalized = true;
  if (!config_.graph) return;
  graph_.config() = config_.constraints;
  last_loop_candidates_ = find_loop_candidates(keyframes_, kf.id, config_.loops);
  loop_candidates_total_ += static_cast<int>(last_loop_candidates_.size());
  std::vector<KeyframeGraph::Candidate> extra;
  for (const auto& c : last_loop_candidates_) extra.push_back({c.id, c.initial});
  last_constraints_ = graph_.add_keyframe(keyframes_, kf.id, extra);
  ++graph_generation_;
}

void Odometry::update_poses() {
  Keyframe& current = *keyframes_.back();
  if (current.parent >= 0) current.pose = Sim3::from(current.from_parent) * keyframes_[current.parent]->pose;
  for (auto& entry : trajectory_) entry.pose = Sim3::from(entry.relative) * keyframes_[entry.keyframe]->pose;
  last_.pose = trajectory_.back().pose;
}

bool Odometry::add_keyframe(const ImageF& depth_m, KeyframeFeatures features) {
  const bool has_prior = depth_m.width == last_image_.width && depth_m.height == last_image_.height;
  auto kf = std::make_unique<Keyframe>();
  kf->id = static_cast<int>(keyframes_.size());
  kf->frame_index = last_.frame_index;
  kf->timestamp_ns = last_.timestamp_ns;
  kf->pose = last_.pose;
  kf->features = std::move(features);
  if (!kf->features.empty()) finish_features(kf->features);
  kf->pyramid = build_pyramid(last_image_, camera_, config_.pyramid_levels, mask_.empty() ? nullptr : &mask_);
  const Level& level = kf->pyramid.levels[0];
  const int w = level.image.width, h = level.image.height;
  kf->filter = std::make_unique<SemiDenseDepth>(level, config_.filter, mask_.empty() ? nullptr : &mask_);

  Keyframe* parent = keyframes_.empty() ? nullptr : keyframes_.back().get();
  const bool relocalizing = last_.state == TrackingState::lost;
  if (parent) {
    kf->parent = parent->id;
    if (relocalizing) {
      const Sim3 relative = kf->pose * parent->pose.inverse();  // pose held, no tracked edge
      kf->from_parent = {relative.R, relative.t / relative.s};
      event_ = "new keyframe after tracking loss";
    } else {
      kf->from_parent = last_.tracking.pose;
      if (config_.stereo && parent->filter) kf->propagated = kf->filter->propagate(*parent->filter, kf->from_parent);
    }
  }

  if (has_prior) {
    ImageF prior(w, h);
    for (std::size_t i = 0; i < prior.data.size(); ++i) {
      const float z = depth_m.data[i];
      prior.data[i] = z > 0 && std::isfinite(z) ? 1.0F / z : 0.0F;
    }
    // Scale lock: the previous keyframe's depth, seen from here, sets the prior's scale.
    if (parent && !relocalizing && config_.scale_lock) {
      std::vector<double> ratios;
      for (const auto& p : parent->reference.points(0)) {
        const Vec3 X = kf->from_parent * p.x;
        if (X.z() <= 0) continue;
        const Vec2 uv = camera_.project(X);
        const int u = static_cast<int>(std::lround(uv.x())), v = static_cast<int>(std::lround(uv.y()));
        if (u < 0 || v < 0 || u >= w || v >= h || !(prior.at(u, v) > 0)) continue;
        ratios.push_back((1.0 / X.z()) / prior.at(u, v));
      }
      if (ratios.size() >= 100) {
        kf->scale_correction = median(ratios);
        for (auto& d : prior.data) d *= static_cast<float>(kf->scale_correction);
      }
    }
    kf->seeded = kf->filter->seed(prior, config_.prior_relative_sigma);
  }
  kf->filter->regularize();
  refresh(*kf);
  if (kf->points < 100) {
    event_ = "keyframe rejected: " + std::to_string(kf->points) + " depth pixels";
    return false;
  }

  // The frame now is the keyframe: its trajectory entry becomes relative to it.
  if (!trajectory_.empty() && trajectory_.back().frame_index == kf->frame_index) trajectory_.pop_back();
  trajectory_.push_back({kf->frame_index, kf->timestamp_ns, kf->id, SE3{}, kf->pose, true});
  // Velocity relative to the new keyframe.
  previous_relative_ = relocalizing ? SE3{} : previous_relative_ * kf->from_parent.inverse();
  last_relative_ = SE3{};
  if (relocalizing) has_previous_ = false;

  keyframes_.push_back(std::move(kf));
  last_.state = TrackingState::tracking;
  last_.keyframe = keyframes_.back()->id;
  last_.keyframe_wanted = false;
  if (parent) {
    finalize(*parent);  // its depth is final: into the graph
    update_poses();
  }
  return true;
}

}  // namespace slam_native::lsd
