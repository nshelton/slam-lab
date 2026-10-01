#pragma once

#include "slam_native/app.hpp"
#include "slam_native/recent_sessions.hpp"
#include "slam_native/video_thumbnail.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
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

// Where a video's launcher thumbnail is cached:
// <repository>/pointcache/thumbnails/<video stem>-<hash>.ppm, keyed by the
// video's path, size and modification time.
std::filesystem::path thumbnail_path(const std::filesystem::path& repository_root, const std::filesystem::path& video);

// The session picker: a 4x4 grid of recent videos with thumbnails (click to
// select, double-click to start), "Load new sequence..." (file browser) and
// the engine. Needs the OpenGL context (thumbnail textures); thumbnails are
// decoded on a worker thread and cached on disk.
class Launcher {
 public:
  Launcher(AppConfig initial, std::filesystem::path repository_root);
  ~Launcher();
  Launcher(const Launcher&) = delete;
  Launcher& operator=(const Launcher&) = delete;
  bool draw(AppConfig& selected, DatabaseSummary& database_summary);
  void set_error(std::string error);
  void record_recent(const AppConfig& config);
  // Pipelines without SuperPoint: hide the engine and point cache; only the
  // video is chosen. `description` replaces the SuperPoint tracking text.
  void set_video_only(std::string description) { video_only_ = std::move(description); }

 private:
  enum class Target { video, engine };
  void open_browser(Target target);
  void draw_browser();
  void refresh_entries();
  void change_directory(const std::filesystem::path& directory);
  [[nodiscard]] std::string& target_path();
  // Recent-video grid; returns true on a double-click (start that session).
  bool draw_recent_grid();
  struct ThumbnailSlot {
    enum class State { pending, ready, failed } state{State::pending};
    unsigned int texture{};
    int width{}, height{};
    std::string error;
  };
  ThumbnailSlot& thumbnail(const std::filesystem::path& video);
  void collect_thumbnails();
  void thumbnail_worker();

  AppConfig initial_;
  std::optional<std::string> video_only_;
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

  std::map<std::filesystem::path, ThumbnailSlot> thumbnails_;
  struct ThumbnailResult {
    std::filesystem::path video;
    std::optional<Thumbnail> image;
    std::string error;
  };
  std::mutex thumbnail_mutex_;
  std::condition_variable thumbnail_wake_;
  std::deque<std::filesystem::path> thumbnail_queue_;   // videos to decode (worker)
  std::vector<ThumbnailResult> thumbnail_results_;      // decoded, to upload (main thread)
  bool thumbnail_stop_{};
  std::thread thumbnail_thread_;
};

}  // namespace slam_native
