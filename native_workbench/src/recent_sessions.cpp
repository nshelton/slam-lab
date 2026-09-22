#include "slam_native/recent_sessions.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace slam_native {
namespace {

namespace fs = std::filesystem;
constexpr std::size_t kMaxRecentVideos = 10;

fs::path normalized(const fs::path& path) {
  if (path.empty()) return {};
  std::error_code error;
  const fs::path canonical = fs::weakly_canonical(path, error);
  if (!error) return canonical;
  return fs::absolute(path).lexically_normal();
}

}  // namespace

fs::path RecentSessions::default_storage_path() {
  if (const char* config_home = std::getenv("XDG_CONFIG_HOME"); config_home && *config_home) {
    return fs::path(config_home) / "slam-native-workbench/recent_videos.txt";
  }
  if (const char* home = std::getenv("HOME"); home && *home) {
    return fs::path(home) / ".config/slam-native-workbench/recent_videos.txt";
  }
  return fs::current_path() / ".slam-native-workbench/recent_videos.txt";
}

RecentSessions::RecentSessions(fs::path storage_path)
    : storage_path_(std::move(storage_path)) {
  std::ifstream input(storage_path_);
  std::string line;
  while (std::getline(input, line) && entries_.size() < kMaxRecentVideos) {
    std::istringstream row(line);
    std::string video, engine, database;
    if (!(row >> std::quoted(video) >> std::quoted(engine) >> std::quoted(database)) ||
        video.empty()) continue;
    RecentSession entry{normalized(video), normalized(engine), normalized(database)};
    if (std::none_of(entries_.begin(), entries_.end(), [&](const RecentSession& previous) {
          return previous.video == entry.video;
        })) entries_.push_back(std::move(entry));
  }
}

void RecentSessions::record(RecentSession session) {
  session.video = normalized(session.video);
  session.engine = normalized(session.engine);
  session.database = normalized(session.database);
  if (session.video.empty()) return;
  entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const RecentSession& previous) {
                   return previous.video == session.video;
                 }), entries_.end());
  entries_.insert(entries_.begin(), std::move(session));
  if (entries_.size() > kMaxRecentVideos) entries_.resize(kMaxRecentVideos);
  save();
}

void RecentSessions::save() const {
  fs::create_directories(storage_path_.parent_path());
  const fs::path temporary = storage_path_.string() + ".tmp";
  {
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot open recent videos file for writing");
    for (const RecentSession& entry : entries_) {
      output << std::quoted(entry.video.string()) << ' '
             << std::quoted(entry.engine.string()) << ' '
             << std::quoted(entry.database.string()) << '\n';
    }
    output.close();
    if (!output) throw std::runtime_error("Cannot write recent videos file");
  }
  fs::rename(temporary, storage_path_);
}

}  // namespace slam_native
