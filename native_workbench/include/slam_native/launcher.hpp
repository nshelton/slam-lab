#pragma once

#include "slam_native/app.hpp"
#include "slam_native/recent_sessions.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace slam_native {

struct DatabaseStats {
  bool exists{};
  std::uintmax_t main_bytes{};
  std::uintmax_t wal_bytes{};
  std::uint64_t frame_rows{};
  std::uint64_t landmark_rows{};
  std::uint64_t observations{};
  std::string schema_version;
  std::string source_video;
  std::string engine_path;
  std::string tracker_algorithm;
  std::string error;
};

class DatabaseSummary {
 public:
  [[nodiscard]] const DatabaseStats& get(const std::filesystem::path& path,
                                         bool inspect_rows = true);

 private:
  std::filesystem::path path_;
  DatabaseStats stats_;
  bool inspected_rows_{};
  std::chrono::steady_clock::time_point refreshed_{};
};

std::string format_bytes(std::uintmax_t bytes);

// Where a session's SuperPoint detections are cached:
// <repository>/pointcache/<video stem>-<hash>.sqlite3. The hash covers everything
// that changes the cached detections (video path and size, engine, SuperPoint
// settings, schema), so a cache always matches the session that opens it.
std::filesystem::path default_point_cache(const std::filesystem::path& repository_root, const AppConfig& config);

class Launcher {
 public:
  Launcher(AppConfig initial, std::filesystem::path repository_root);
  bool draw(AppConfig& selected, DatabaseSummary& database_summary);
  void set_error(std::string error);
  void record_recent(const AppConfig& config);

 private:
  enum class Target { video, engine };
  void open_browser(Target target);
  void draw_browser();
  void refresh_entries();
  void change_directory(const std::filesystem::path& directory);
  [[nodiscard]] std::string& target_path();

  AppConfig initial_;
  RecentSessions recents_;
  std::filesystem::path repository_root_;
  std::string video_;
  std::string engine_;
  std::string error_;
  bool browser_open_{};
  bool browser_popup_pending_{};
  bool show_all_files_{};
  Target browser_target_{Target::video};
  std::filesystem::path browser_directory_;
  std::string browser_folder_;
  std::string browser_filename_;
  std::string browser_error_;
  std::vector<std::filesystem::directory_entry> entries_;
};

}  // namespace slam_native
