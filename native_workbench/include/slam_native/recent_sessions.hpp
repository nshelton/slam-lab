#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace slam_native {

struct RecentSession {
  std::filesystem::path video;
  std::filesystem::path engine;
  std::filesystem::path database;
};

class RecentSessions {
 public:
  static constexpr std::size_t kCapacity = 16;  // the launcher shows a 4x4 grid

  explicit RecentSessions(std::filesystem::path storage_path);

  [[nodiscard]] static std::filesystem::path default_storage_path();
  [[nodiscard]] const std::vector<RecentSession>& entries() const { return entries_; }
  void record(RecentSession session);

 private:
  void save() const;

  std::filesystem::path storage_path_;
  std::vector<RecentSession> entries_;
};

}  // namespace slam_native
