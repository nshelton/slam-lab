#pragma once

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
