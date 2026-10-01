#pragma once
// Descriptor index over the map's landmarks, for search without a pose prior
// (internal to the visual odometry). Brute force: every query against every
// entry, exactly; the baseline an approximate index would be compared against.
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace slam_native::vo {

class PlaceIndex {
 public:
  struct Match {
    int query;         // row of `queries`
    std::uint64_t id;  // entry
    float similarity;  // cosine
  };
  // Insert or replace an entry's descriptor (unit length). The first call
  // fixes the dimension; other dimensions are ignored.
  void set(std::uint64_t id, const float* descriptor, int dimension);
  void erase(std::uint64_t id);
  [[nodiscard]] std::size_t size() const { return ids_.size(); }
  [[nodiscard]] int dimension() const { return dimension_; }

  // Mutual best matches of `queries` (count x dimension(), unit rows) among
  // the entries `accept` allows: cosine >= min_similarity, and cosine distance
  // <= ratio x the distance to the query's runner-up. Ordered by entry id.
  [[nodiscard]] std::vector<Match> match(const float* queries, int count, double min_similarity, double ratio,
                                         const std::function<bool(std::uint64_t)>& accept) const;

 private:
  int dimension_{};
  std::vector<float> rows_;  // size() x dimension_
  std::vector<std::uint64_t> ids_;
  std::unordered_map<std::uint64_t, std::size_t> row_of_;
};

}  // namespace slam_native::vo
