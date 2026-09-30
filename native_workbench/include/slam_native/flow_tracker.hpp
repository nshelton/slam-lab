#pragma once
#include "slam_native/online_tracker.hpp"
#include "slam_native/optical_flow.hpp"
#include <memory>
#include <optional>

namespace slam_native {
struct FlowTrackerConfig {
  float association_radius{6.0F};  // source-image pixels around the flow prediction
  std::size_t max_tracks{1000};
  // Cosine gate between a track's last observed descriptor and a candidate
  // detection. Values <= -1 disable the gate.
  float min_descriptor_similarity{0.7F};
  // Association cost = d^2 / radius^2 + descriptor_weight * (1 - cosine).
  float descriptor_weight{1.0F};
  // Frames a track may be carried by flow alone when no detection supports
  // it. 0 restores "unmatched tracks die immediately".
  int max_coast_frames{5};
  // Forward-backward flow consistency (pixels). A track whose flow fails the
  // check cannot coast. Applies only when the flow field includes backward
  // vectors (OpticalFlowConfig::backward); <= 0 disables it.
  float forward_backward_threshold{2.0F};
  // Propose/accept rounds of the one-to-one assignment. 1 reproduces the old
  // single-pass behaviour (no second choices).
  int assignment_rounds{4};

  void validate() const;
};

// Tracks SuperPoint detections with NVIDIA optical-flow predictions entirely on
// the GPU: prediction, gating, assignment, coasting, seeding, ID allocation and
// track state (positions, descriptors, lengths) stay in device memory. The
// host receives one compact record per live track per frame for display and
// statistics.
class FlowTracker {
 public:
  explicit FlowTracker(FlowTrackerConfig config);
  ~FlowTracker();
  FlowTracker(const FlowTracker&) = delete;
  FlowTracker& operator=(const FlowTracker&) = delete;

  // GPU path. `detections` must be in the same order as frame.keypoints; when
  // null, frame.keypoints/descriptors are uploaded (cached frames, tests).
  // A null `flow` breaks temporal identity (first frame, seek, missing flow).
  void associate(FrameFeatures& frame, const DeviceFlowField* flow,
                 const DeviceDetections* detections = nullptr);
  // Convenience for tests: uploads a host flow field first.
  void associate(FrameFeatures& frame, const std::optional<FlowField>& flow);

  void set_association_radius(float radius);
  [[nodiscard]] const FlowTrackerConfig& config() const { return config_; }
  [[nodiscard]] std::size_t active_count() const { return active_.size(); }
  [[nodiscard]] std::uint64_t landmark_count() const { return next_id_; }
  [[nodiscard]] const LandmarkState* find(std::uint64_t id) const;
  [[nodiscard]] const TrackLengthHistogram& length_histogram() const { return histogram_; }

 private:
  struct HostTrack {
    std::uint64_t id{}, observation_count{}, first_frame{}, last_frame{};
  };
  struct Device;
  FlowTrackerConfig config_;
  std::unique_ptr<Device> device_;
  std::vector<HostTrack> active_;
  std::uint64_t next_id_{};
  TrackLengthHistogram histogram_{};
  std::optional<std::int64_t> timestamp_ns_;
  std::uint64_t frame_index_{};
  int width_{}, height_{};
  mutable LandmarkState inspection_;
};
}  // namespace slam_native
