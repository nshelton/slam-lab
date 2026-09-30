// CPU-only feature cache checks (core-debug and app builds).
#include "slam_native/feature_store.hpp"
#include <sqlite3.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace slam_native;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class F> void rejects(F action, const char* message) {
  bool rejected = false;
  try { action(); } catch (const std::exception&) { rejected = true; }
  require(rejected, message);
}
FrameFeatures frame(std::uint64_t index, double seconds, std::vector<Keypoint> points = {}) {
  FrameFeatures result;
  result.frame_index = index;
  result.timestamp_ns = static_cast<std::int64_t>(seconds * 1e9);
  result.pts = index;
  result.width = result.height = 96;
  result.keypoints = std::move(points);
  result.descriptors.assign(result.keypoints.size() * 256, 0.125F);
  return result;
}
GpuFrame identity(const FrameFeatures& features) {
  GpuFrame result;
  result.frame_index = features.frame_index;
  result.timestamp_ns = features.timestamp_ns;
  result.pts = features.pts;
  result.width = features.width;
  result.height = features.height;
  return result;
}

int main() {
  const fs::path root = fs::temp_directory_path() / ("slam-store-test-" +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  try {
    fs::create_directories(root);
    const auto source = root / "video";
    const auto engine = root / "engine";
    std::ofstream(source) << "test video";
    std::ofstream(engine) << "test engine";
    const auto raw = frame(0, 0, {{10, 10, 0.8F}});
    for (const auto encoding : {DescriptorEncoding::float16, DescriptorEncoding::float32}) {
      StoreConfig settings;
      settings.descriptor_encoding = encoding;
      const auto database = root / (encoding == DescriptorEncoding::float16 ? "half.db" : "float.db");
      {
        FeatureStore store(database, settings);
        store.set_session(source, engine, 1024, 576, 2048, 0.0005F);
        store.enqueue(raw);
        store.enqueue(frame(1, 0.1));
        auto tracked = frame(2, 0.2, {{12, 11, 0.7F}});
        tracked.landmark_ids = {7};
        rejects([&] { store.enqueue(tracked); }, "Derived tracking state must not persist");
        auto coasted = frame(3, 0.3);
        coasted.coasted_landmarks = 1;
        rejects([&] { store.enqueue(coasted); }, "Coasting diagnostics must not persist");
        store.flush();
        require(store.persisted() == 2, "Raw frames were not cached");
        auto loaded = store.load_frame(identity(raw));
        require(loaded && loaded->keypoints[0].x == 10 &&
                loaded->descriptors == raw.descriptors && loaded->landmark_ids.empty(),
                "Cache round trip lost observations or included track identities");
        require(store.load_frame(identity(frame(1, 0.1)))->keypoints.empty(),
                "Empty detection frames must cache");
        require(!store.load_frame(identity(frame(2, 0.2))), "Cache miss was not detected");
      }
      {
        FeatureStore reopened(database, settings);
        reopened.set_session(source, engine, 1024, 576, 2048, 0.0005F);
        auto loaded = *reopened.load_frame(identity(raw));
        require(loaded.landmark_ids.empty() && loaded.keypoints.size() == 1,
                "Replay must load raw observations only");
        rejects([&] { reopened.set_session(source, engine, 1024, 576, 2048, 0.001F); },
                "Changed extraction settings must invalidate cache");
      }
      sqlite3* db{};
      require(sqlite3_open(database.c_str(), &db) == SQLITE_OK, "Open cache inspection");
      sqlite3_stmt* stmt{};
      sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM sqlite_master WHERE name='landmarks'", -1, &stmt, nullptr);
      sqlite3_step(stmt);
      require(sqlite3_column_int(stmt, 0) == 0, "Cache must not contain landmarks table");
      sqlite3_finalize(stmt);
      sqlite3_prepare_v2(db, "SELECT landmark_ids_u64 FROM frames", -1, &stmt, nullptr);
      require(stmt == nullptr, "Cache must not have track columns");
      sqlite3_close(db);
    }
    std::ofstream(engine) << "changed engine";
    {
      FeatureStore store(root / "half.db", {});
      rejects([&] { store.set_session(source, engine, 1024, 576, 2048, 0.0005F); },
              "Changed engine must invalidate cache");
    }
    sqlite3* legacy{};
    const auto legacy_path = root / "legacy.db";
    sqlite3_open(legacy_path.c_str(), &legacy);
    sqlite3_exec(legacy, "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);"
                         "INSERT INTO metadata VALUES('schema_version','2');", nullptr, nullptr, nullptr);
    sqlite3_close(legacy);
    const auto old_size = fs::file_size(legacy_path);
    const auto old_time = fs::last_write_time(legacy_path);
    rejects([&] { FeatureStore store(legacy_path, {}); }, "Legacy database must be rejected");
    require(fs::file_size(legacy_path) == old_size && fs::last_write_time(legacy_path) == old_time,
            "Legacy database was modified");
    fs::remove_all(root);
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\nTemporary test directory: " << root << '\n';
    return 1;
  }
}
