#pragma once
// Descriptor index over the map's landmarks, for search without a pose prior
// (internal to the visual odometry). Brute force: every query against every
// entry, exactly; the baseline an approximate index would be compared against.
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace slam_native::vo {

class PlaceIndex {
 public:
  struct Match {
    int query;         // row of `queries`
    std::size_t row;   // entry, as its position in ids()
    std::uint64_t id;  // entry
    float similarity;  // cosine
  };
  // Insert or replace an entry's descriptor (unit length). The first call
  // fixes the dimension; other dimensions are ignored.
  void set(std::uint64_t id, const float* descriptor, int dimension);
  void erase(std::uint64_t id);
  [[nodiscard]] std::size_t size() const { return ids_.size(); }
  [[nodiscard]] int dimension() const { return dimension_; }
  // The entries, in row order (changes with set() and erase()).
  [[nodiscard]] const std::vector<std::uint64_t>& ids() const { return ids_; }

  // Mutual best matches of `queries` (count x dimension(), unit rows) among
  // the entries with accepted[row] != 0: cosine >= min_similarity, and cosine
  // distance <= ratio x the distance to the query's runner-up. Ordered by
  // entry id. Const and self-contained: it may run on another thread while
  // the index is not modified.
  [[nodiscard]] std::vector<Match> match(const float* queries, int count, double min_similarity, double ratio,
                                         const std::vector<char>& accepted) const;

 private:
  int dimension_{};
  std::vector<float> rows_;  // size() x dimension_
  std::vector<std::uint64_t> ids_;
  std::unordered_map<std::uint64_t, std::size_t> row_of_;
};

}  // namespace slam_native::vo
