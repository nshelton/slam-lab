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

class Launcher {
 public:
  Launcher(AppConfig initial, std::filesystem::path repository_root);
  bool draw(AppConfig& selected, DatabaseSummary& database_summary);
  void set_error(std::string error);
  void record_recent(const AppConfig& config);

 private:
  enum class Target { video, engine, database };
  void open_browser(Target target);
  void draw_browser();
  void set_video(std::string value);
  void refresh_entries();
  void change_directory(const std::filesystem::path& directory);
  [[nodiscard]] std::string& target_path();

  AppConfig initial_;
  TrackingMethod tracking_method_{TrackingMethod::superpoint};
  RecentSessions recents_;
  std::filesystem::path repository_root_;
  std::string video_;
  std::string engine_;
  std::string database_;
  std::string error_;
  bool database_custom_{};
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
