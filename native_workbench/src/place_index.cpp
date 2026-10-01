#include "place_index.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <thread>

namespace slam_native::vo {
namespace {
using Rows = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
constexpr std::size_t kBlock = 2048;  // entries per similarity block
}  // namespace

void PlaceIndex::set(std::uint64_t id, const float* descriptor, int dimension) {
  if (dimension <= 0) return;
  if (ids_.empty()) dimension_ = dimension;
  if (dimension != dimension_) return;
  auto [it, fresh] = row_of_.try_emplace(id, ids_.size());
  if (fresh) {
    ids_.push_back(id);
    rows_.resize(ids_.size() * static_cast<std::size_t>(dimension_));
  }
  std::copy(descriptor, descriptor + dimension_, rows_.begin() + static_cast<std::ptrdiff_t>(it->second * dimension_));
}

void PlaceIndex::erase(std::uint64_t id) {
  auto it = row_of_.find(id);
  if (it == row_of_.end()) return;
  const std::size_t row = it->second, last = ids_.size() - 1;
  if (row != last) {  // the last entry takes the freed row
    std::copy(rows_.begin() + static_cast<std::ptrdiff_t>(last * dimension_),
              rows_.begin() + static_cast<std::ptrdiff_t>((last + 1) * dimension_),
              rows_.begin() + static_cast<std::ptrdiff_t>(row * dimension_));
    ids_[row] = ids_[last];
    row_of_[ids_[row]] = row;
  }
  row_of_.erase(it);
  ids_.pop_back();
  rows_.resize(ids_.size() * static_cast<std::size_t>(dimension_));
}

std::vector<PlaceIndex::Match> PlaceIndex::match(const float* queries, int count, double min_similarity, double ratio,
                                                 const std::function<bool(std::uint64_t)>& accept) const {
  std::vector<Match> matches;
  const std::size_t entries = ids_.size();
  if (count <= 0 || entries == 0) return matches;
  std::vector<char> accepted(entries);
  for (std::size_t e = 0; e < entries; ++e) accepted[e] = accept(ids_[e]);
  const Eigen::Map<const Rows> Q(queries, count, dimension_);

  // Per block of entries: each query's best two entries and each entry's best
  // query. Blocks are independent, so they run on several threads; merging
  // them in entry order keeps the result the same for any thread count.
  constexpr float kNone = -2.0F;
  struct Best {
    std::vector<float> best, second;       // per query
    std::vector<std::size_t> best_entry;   // per query
  };
  const std::size_t blocks = (entries + kBlock - 1) / kBlock;
  std::vector<Best> per_block(blocks);
  std::vector<float> entry_best(entries, kNone);
  std::vector<int> entry_query(entries, -1);
  const auto work = [&](std::size_t first_block, std::size_t step) {
    Rows S;
    for (std::size_t b = first_block; b < blocks; b += step) {
      const std::size_t begin = b * kBlock, size = std::min(kBlock, entries - begin);
      const Eigen::Map<const Rows> D(rows_.data() + begin * dimension_, static_cast<Eigen::Index>(size), dimension_);
      S.noalias() = Q * D.transpose();  // count x size
      Best& out = per_block[b];
      out.best.assign(count, kNone);
      out.second.assign(count, kNone);
      out.best_entry.assign(count, 0);
      for (int q = 0; q < count; ++q) {
        const float* row = S.data() + static_cast<std::size_t>(q) * size;
        for (std::size_t e = 0; e < size; ++e) {
          if (!accepted[begin + e]) continue;
          const float s = row[e];
          if (s > out.best[q]) {
            out.second[q] = out.best[q];
            out.best[q] = s;
            out.best_entry[q] = begin + e;
          } else if (s > out.second[q]) {
            out.second[q] = s;
          }
          if (s > entry_best[begin + e]) {  // queries in order: the first of equal ones wins
            entry_best[begin + e] = s;
            entry_query[begin + e] = q;
          }
        }
      }
    }
  };
  const std::size_t flops = entries * static_cast<std::size_t>(count) * dimension_;
  const std::size_t threads = flops < 50'000'000 ? 1 :
      std::min<std::size_t>({blocks, 8, std::max(1U, std::thread::hardware_concurrency())});
  if (threads <= 1) {
    work(0, 1);
  } else {
    std::vector<std::thread> pool;
    for (std::size_t t = 1; t < threads; ++t) pool.emplace_back(work, t, threads);
    work(0, threads);
    for (auto& thread : pool) thread.join();
  }

  for (int q = 0; q < count; ++q) {
    float best = kNone, second = kNone;
    std::size_t entry = 0;
    for (const Best& block : per_block) {
      if (block.best[q] > best) {
        second = std::max(best, block.second[q]);
        best = block.best[q];
        entry = block.best_entry[q];
      } else {
        second = std::max(second, block.best[q]);
      }
    }
    if (best < min_similarity || entry_query[entry] != q) continue;
    if (second > kNone && 1 - best > ratio * (1 - second)) continue;
    matches.push_back({q, ids_[entry], best});
  }
  std::sort(matches.begin(), matches.end(), [](const Match& l, const Match& r) { return l.id < r.id; });
  return matches;
}

}  // namespace slam_native::vo
