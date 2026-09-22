#include "slam_native/recent_sessions.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int main() {
  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path root = fs::temp_directory_path() /
                        ("slam-native-recents-test-" + std::to_string(unique));
  const fs::path storage = root / "config/recent_videos.txt";
  try {
    slam_native::RecentSessions recents(storage);
    for (int index = 0; index < 12; ++index) {
      recents.record({root / ("video " + std::to_string(index) + ".mp4"),
                      root / "model with spaces.engine",
                      root / ("database " + std::to_string(index) + ".sqlite3")});
    }
    if (recents.entries().size() != 10 ||
        recents.entries().front().video.filename() != "video 11.mp4" ||
        recents.entries().back().video.filename() != "video 2.mp4") {
      std::cerr << "Recent videos did not retain the last ten entries\n";
      return 1;
    }
    recents.record({root / "video 5.mp4", root / "new engine.engine",
                    root / "new database.sqlite3"});
    slam_native::RecentSessions loaded(storage);
    if (loaded.entries().size() != 10 ||
        loaded.entries().front().video.filename() != "video 5.mp4" ||
        loaded.entries().front().engine.filename() != "new engine.engine" ||
        loaded.entries().front().database.filename() != "new database.sqlite3" ||
        loaded.entries().back().video.filename() != "video 2.mp4") {
      std::cerr << "Recent video order, deduplication, or persistence failed\n";
      return 1;
    }
    fs::remove_all(root);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
