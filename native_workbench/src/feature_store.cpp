#include "slam_native/feature_store.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace slam_native {
namespace {

constexpr std::string_view kSchema = R"sql(
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS frames (
  frame_index INTEGER PRIMARY KEY,
  timestamp_ns INTEGER NOT NULL,
  pts INTEGER,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  keypoint_count INTEGER NOT NULL,
  keypoints_xy_f32 BLOB NOT NULL,
  scores_f32 BLOB NOT NULL,
  descriptors BLOB NOT NULL,
  descriptor_encoding TEXT NOT NULL CHECK(descriptor_encoding IN ('f16', 'f32', 'none')),
  decode_ms REAL NOT NULL,
  preprocess_ms REAL NOT NULL,
  inference_ms REAL NOT NULL,
  readback_ms REAL NOT NULL,
  landmark_ids_u64 BLOB NOT NULL,
  landmark_similarities_f32 BLOB NOT NULL,
  new_landmarks INTEGER NOT NULL,
  matched_landmarks INTEGER NOT NULL,
  tracking_ms REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS landmarks (
  landmark_id INTEGER PRIMARY KEY,
  observation_count INTEGER NOT NULL,
  resultant_f32 BLOB NOT NULL,
  concentration REAL NOT NULL,
  first_frame INTEGER NOT NULL,
  last_frame INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS frames_timestamp ON frames(timestamp_ns);
)sql";

void check(int result, sqlite3* db, std::string_view operation) {
  if (result != SQLITE_OK && result != SQLITE_DONE && result != SQLITE_ROW) {
    throw std::runtime_error(std::string(operation) + ": " + sqlite3_errmsg(db));
  }
}

void bind_blob(sqlite3_stmt* statement, int index, const void* data,
               std::size_t size, sqlite3* db) {
  // SQLite treats a null pointer as SQL NULL even when the byte count is zero.
  // Zero-keypoint frames still need valid, empty blobs in the NOT NULL columns.
  static constexpr unsigned char empty = 0;
  check(sqlite3_bind_blob64(statement, index, size ? data : &empty, size,
                            SQLITE_TRANSIENT), db, "bind feature blob");
}

std::uint16_t float_to_half(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t exponent = (bits >> 23U) & 0xffU;
  const std::uint32_t mantissa = bits & 0x7fffffU;

  if (exponent == 0xffU) {
    return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa ? 0x0200U : 0U));
  }
  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t normalized = mantissa | 0x800000U;
    const int shift = 14 - half_exponent;
    const std::uint32_t rounded = normalized + (1U << (shift - 1));
    return static_cast<std::uint16_t>(sign | (rounded >> shift));
  }
  const std::uint32_t rounded = mantissa + 0x1000U;
  if (rounded & 0x800000U) {
    const int incremented_exponent = half_exponent + 1;
    return static_cast<std::uint16_t>(
        sign | (incremented_exponent >= 31 ? 0x7c00U : incremented_exponent << 10U));
  }
  return static_cast<std::uint16_t>(sign | (half_exponent << 10U) | (rounded >> 13U));
}

void set_metadata(sqlite3* db, const char* key, const std::string& value) {
  sqlite3_stmt* statement = nullptr;
  check(sqlite3_prepare_v2(db,
                           "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
                           -1, &statement, nullptr),
        db, "prepare metadata");
  check(sqlite3_bind_text(statement, 1, key, -1, SQLITE_STATIC), db, "bind metadata key");
  check(sqlite3_bind_text(statement, 2, value.c_str(), -1, SQLITE_TRANSIENT), db,
        "bind metadata value");
  const int result = sqlite3_step(statement);
  sqlite3_finalize(statement);
  check(result, db, "write metadata");
}

void set_immutable_metadata(sqlite3* db, const char* key, const std::string& value) {
  sqlite3_stmt* statement = nullptr;
  check(sqlite3_prepare_v2(db, "SELECT value FROM metadata WHERE key=?", -1,
                           &statement, nullptr), db, "read session metadata");
  sqlite3_bind_text(statement, 1, key, -1, SQLITE_STATIC);
  const int found = sqlite3_step(statement);
  const std::string existing = found == SQLITE_ROW
                                   ? reinterpret_cast<const char*>(sqlite3_column_text(statement, 0))
                                   : "";
  sqlite3_finalize(statement);
  if (found != SQLITE_ROW && found != SQLITE_DONE) check(found, db, "read session metadata");
  bool matches = existing.empty() || existing == value;
  if (!matches && (std::strcmp(key, "source_video") == 0 ||
                   std::strcmp(key, "engine_path") == 0)) {
    std::error_code old_error, new_error;
    const auto old_path = std::filesystem::weakly_canonical(existing, old_error);
    const auto new_path = std::filesystem::weakly_canonical(value, new_error);
    matches = !old_error && !new_error && old_path == new_path;
  }
  if (!matches) {
    throw std::runtime_error(std::string("Feature database setting differs: ") + key);
  }
  set_metadata(db, key, value);
}

}  // namespace

FeatureStore::FeatureStore(const std::filesystem::path& path, StoreConfig config)
    : config_(config) {
  static_assert(std::endian::native == std::endian::little,
                "Feature blobs require a little-endian machine");
  if (config_.transaction_frames == 0 || config_.max_queued_frames == 0) {
    throw std::invalid_argument("FeatureStore queue and transaction sizes must be positive");
  }
  std::filesystem::create_directories(path.parent_path().empty() ? "." : path.parent_path());
  const int opened = sqlite3_open_v2(path.c_str(), &db_,
                                      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
  if (opened != SQLITE_OK) {
    const std::string message = db_ ? sqlite3_errmsg(db_) : "no SQLite connection";
    if (db_) sqlite3_close(db_);
    db_ = nullptr;
    throw std::runtime_error("open feature database: " + message);
  }
  sqlite3_busy_timeout(db_, 5000);
  char* error = nullptr;
  const int result = sqlite3_exec(db_, kSchema.data(), nullptr, nullptr, &error);
  if (result != SQLITE_OK) {
    const std::string message = error ? error : sqlite3_errmsg(db_);
    sqlite3_free(error);
    sqlite3_close(db_);
    db_ = nullptr;
    throw std::runtime_error("create feature database: " + message);
  }
  sqlite3_stmt* version = nullptr;
  check(sqlite3_prepare_v2(db_, "SELECT value FROM metadata WHERE key='schema_version'",
                           -1, &version, nullptr), db_, "read schema version");
  const int found = sqlite3_step(version);
  const std::string existing = found == SQLITE_ROW
                                   ? reinterpret_cast<const char*>(sqlite3_column_text(version, 0))
                                   : "";
  sqlite3_finalize(version);
  if (found != SQLITE_ROW && found != SQLITE_DONE) check(found, db_, "read schema version");
  if (!existing.empty() && existing != "2") {
    sqlite3_close(db_);
    db_ = nullptr;
    throw std::runtime_error("Unsupported native feature database schema: " + existing);
  }
  set_metadata(db_, "schema_version", "2");
  set_metadata(db_, "image_storage", "source-video-only");
  sqlite3_stmt* last = nullptr;
  check(sqlite3_prepare_v2(db_,
                           "SELECT MAX(frame_index),COUNT(*),"
                           "COALESCE(SUM(keypoint_count),0) FROM frames",
                           -1, &last, nullptr), db_, "read stored frame count");
  const int loaded = sqlite3_step(last);
  if (loaded != SQLITE_ROW) {
    sqlite3_finalize(last);
    check(loaded, db_, "read stored frame count");
  }
  last_frame_index_ = sqlite3_column_type(last, 0) == SQLITE_NULL
                          ? -1 : sqlite3_column_int64(last, 0);
  persisted_ = static_cast<std::uint64_t>(sqlite3_column_int64(last, 1));
  observation_count_ = static_cast<std::uint64_t>(sqlite3_column_int64(last, 2));
  sqlite3_finalize(last);
  sqlite3_stmt* landmarks = nullptr;
  check(sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM landmarks", -1,
                           &landmarks, nullptr), db_, "read landmark count");
  check(sqlite3_step(landmarks), db_, "read landmark count");
  landmark_rows_ = static_cast<std::uint64_t>(sqlite3_column_int64(landmarks, 0));
  sqlite3_finalize(landmarks);
  writer_ = std::thread(&FeatureStore::writer_loop, this);
}

FeatureStore::~FeatureStore() {
  try {
    flush();
  } catch (...) {
  }
  {
    std::lock_guard lock(mutex_);
    stopping_ = true;
  }
  readable_.notify_all();
  if (writer_.joinable()) {
    writer_.join();
  }
  if (db_) {
    sqlite3_close(db_);
  }
}

void FeatureStore::set_session(const std::filesystem::path& source_video,
                               const std::filesystem::path& engine_path,
                               int input_width,
                               int input_height,
                               int max_keypoints,
                               float threshold,
                               const TrackerConfig& tracker, bool optical_flow) {
  // Call before frames are enqueued. This connection is only otherwise used by
  // the writer thread after construction.
  sqlite3_stmt* method = nullptr;
  check(sqlite3_prepare_v2(db_,
      "SELECT value FROM metadata WHERE key='tracker_algorithm'", -1, &method, nullptr),
      db_, "read tracker algorithm");
  const int method_result = sqlite3_step(method);
  const std::string existing_method = method_result == SQLITE_ROW ?
      reinterpret_cast<const char*>(sqlite3_column_text(method, 0)) : "";
  sqlite3_finalize(method);
  if (method_result != SQLITE_ROW && method_result != SQLITE_DONE) {
    check(method_result, db_, "read tracker algorithm");
  }
  const char* algorithm = optical_flow ? "nvof-grid-tracker-v1" :
                                         "online-spherical-mean-greedy-v1";
  if (!existing_method.empty() && existing_method != algorithm) {
    throw std::runtime_error("Feature database belongs to another tracking method");
  }
  const std::string source = std::filesystem::weakly_canonical(source_video).string();
  set_immutable_metadata(db_, "source_video", source);
  set_immutable_metadata(db_, "engine_path", optical_flow ? "" :
                         std::filesystem::weakly_canonical(engine_path).string());
  set_immutable_metadata(db_, "descriptor_dimension", optical_flow ? "0" : "256");
  set_immutable_metadata(db_, "input_width", optical_flow ? "0" : std::to_string(input_width));
  set_immutable_metadata(db_, "input_height", optical_flow ? "0" : std::to_string(input_height));
  set_immutable_metadata(db_, "max_keypoints", optical_flow ? "0" : std::to_string(max_keypoints));
  set_immutable_metadata(db_, "detection_threshold", optical_flow ? "0" : std::to_string(threshold));
  set_immutable_metadata(db_, "descriptor_encoding",
                         config_.descriptor_encoding == DescriptorEncoding::none ? "none" :
                         config_.descriptor_encoding == DescriptorEncoding::float16 ? "f16" : "f32");
  set_immutable_metadata(db_, "tracker_algorithm", algorithm);
  if (!optical_flow) {
    set_immutable_metadata(db_, "tracker_min_similarity", std::to_string(tracker.min_similarity));
    set_immutable_metadata(db_, "tracker_min_margin", std::to_string(tracker.min_margin));
  }
  set_immutable_metadata(db_, "tracker_max_inactive_frames",
                         std::to_string(tracker.max_inactive_frames));
}

std::vector<LandmarkState> FeatureStore::load_active_landmarks(
    std::uint32_t max_inactive_frames) const {
  std::vector<LandmarkState> active;
  if (last_frame_index_ < 0) return active;
  sqlite3_stmt* statement = nullptr;
  check(sqlite3_prepare_v2(db_,
      "SELECT landmark_id,observation_count,resultant_f32,concentration,"
      "first_frame,last_frame FROM landmarks WHERE last_frame>=? ORDER BY landmark_id",
      -1, &statement, nullptr), db_, "prepare active landmarks");
  sqlite3_bind_int64(statement, 1,
      std::max<std::int64_t>(0, last_frame_index_ - max_inactive_frames));
  int result;
  while ((result = sqlite3_step(statement)) == SQLITE_ROW) {
    const int resultant_bytes = sqlite3_column_bytes(statement, 2);
    const bool valid_resultant = config_.descriptor_encoding == DescriptorEncoding::none ?
        (resultant_bytes == 0 || resultant_bytes == 256 * static_cast<int>(sizeof(float))) :
        resultant_bytes == 256 * static_cast<int>(sizeof(float));
    if (!valid_resultant) {
      sqlite3_finalize(statement);
      throw std::runtime_error("Invalid landmark resultant in feature database");
    }
    LandmarkState state;
    state.id = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 0));
    state.observation_count = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 1));
    if (resultant_bytes) {
      std::memcpy(state.resultant.data(), sqlite3_column_blob(statement, 2),
                  state.resultant.size() * sizeof(float));
    }
    state.concentration = static_cast<float>(sqlite3_column_double(statement, 3));
    state.first_frame = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 4));
    state.last_frame = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 5));
    active.push_back(state);
  }
  sqlite3_finalize(statement);
  check(result, db_, "read active landmarks");
  return active;
}

std::uint64_t FeatureStore::next_landmark_id() const {
  sqlite3_stmt* statement = nullptr;
  check(sqlite3_prepare_v2(db_, "SELECT COALESCE(MAX(landmark_id),-1)+1 FROM landmarks",
                           -1, &statement, nullptr), db_, "prepare next landmark ID");
  const int result = sqlite3_step(statement);
  if (result != SQLITE_ROW) {
    sqlite3_finalize(statement);
    check(result, db_, "read next landmark ID");
  }
  const auto next = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 0));
  sqlite3_finalize(statement);
  return next;
}

TrackLengthHistogram FeatureStore::load_length_histogram() const {
  TrackLengthHistogram histogram{};
  sqlite3_stmt* statement = nullptr;
  check(sqlite3_prepare_v2(db_,
      "SELECT observation_count,COUNT(*) FROM landmarks GROUP BY observation_count",
      -1, &statement, nullptr), db_, "prepare track length histogram");
  int result;
  while ((result = sqlite3_step(statement)) == SQLITE_ROW) {
    const auto length = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 0));
    histogram[track_length_bin(length)] +=
        static_cast<std::uint64_t>(sqlite3_column_int64(statement, 1));
  }
  sqlite3_finalize(statement);
  check(result, db_, "read track length histogram");
  return histogram;
}

std::vector<FlowTrackPosition> FeatureStore::load_latest_positions() const {
  std::vector<FlowTrackPosition> positions;
  if (last_frame_index_ < 0) return positions;
  sqlite3_stmt* statement = nullptr;
  check(sqlite3_prepare_v2(db_,
      "SELECT keypoint_count,keypoints_xy_f32,landmark_ids_u64 FROM frames "
      "WHERE frame_index=?", -1, &statement, nullptr), db_, "prepare latest flow positions");
  sqlite3_bind_int64(statement, 1, last_frame_index_);
  const int result = sqlite3_step(statement);
  if (result == SQLITE_ROW) {
    const int count = sqlite3_column_int(statement, 0);
    if (count < 0 ||
        sqlite3_column_bytes(statement, 1) != count * 2 * static_cast<int>(sizeof(float)) ||
        sqlite3_column_bytes(statement, 2) != count * static_cast<int>(sizeof(std::uint64_t))) {
      sqlite3_finalize(statement);
      throw std::runtime_error("Invalid saved flow positions");
    }
    const auto* xy = static_cast<const float*>(sqlite3_column_blob(statement, 1));
    const auto* ids = static_cast<const std::uint64_t*>(sqlite3_column_blob(statement, 2));
    positions.reserve(count);
    for (int index = 0; index < count; ++index) {
      positions.push_back({ids[index], xy[index * 2], xy[index * 2 + 1]});
    }
  }
  sqlite3_finalize(statement);
  if (result != SQLITE_ROW && result != SQLITE_DONE) check(result, db_, "read latest flow positions");
  return positions;
}

void FeatureStore::enqueue(FrameFeatures features) {
  if (config_.descriptor_encoding == DescriptorEncoding::none ?
      !features.descriptors.empty() :
      features.descriptors.size() != features.keypoints.size() * 256) {
    throw std::invalid_argument("Invalid descriptor array for selected tracking mode");
  }
  if (features.landmark_ids.size() != features.keypoints.size() ||
      features.landmark_similarities.size() != features.keypoints.size() ||
      features.landmark_updates.size() != features.keypoints.size()) {
    throw std::invalid_argument("Tracker must assign every detected feature");
  }
  std::unique_lock lock(mutex_);
  writable_.wait(lock, [this] {
    return queue_.size() < config_.max_queued_frames || writer_error_ || stopping_;
  });
  throw_writer_error();
  if (stopping_) {
    throw std::runtime_error("FeatureStore is stopping");
  }
  queue_.push(std::move(features));
  lock.unlock();
  readable_.notify_one();
}

void FeatureStore::flush() {
  std::unique_lock lock(mutex_);
  drained_.wait(lock, [this] { return (queue_.empty() && !writing_) || writer_error_; });
  throw_writer_error();
}

std::size_t FeatureStore::queued() const {
  std::lock_guard lock(mutex_);
  return queue_.size();
}

std::uint64_t FeatureStore::persisted() const {
  std::lock_guard lock(mutex_);
  return persisted_;
}

std::uint64_t FeatureStore::landmark_rows() const {
  std::lock_guard lock(mutex_);
  return landmark_rows_;
}

std::uint64_t FeatureStore::observation_count() const {
  std::lock_guard lock(mutex_);
  return observation_count_;
}

std::int64_t FeatureStore::last_frame_index() const { return last_frame_index_; }

void FeatureStore::throw_writer_error() {
  if (writer_error_) {
    std::rethrow_exception(writer_error_);
  }
}

void FeatureStore::writer_loop() {
  try {
    for (;;) {
      std::vector<FrameFeatures> batch;
      {
        std::unique_lock lock(mutex_);
        readable_.wait(lock, [this] { return !queue_.empty() || stopping_; });
        if (queue_.empty() && stopping_) {
          break;
        }
        const std::size_t count = std::min(config_.transaction_frames, queue_.size());
        batch.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
          batch.push_back(std::move(queue_.front()));
          queue_.pop();
        }
        writing_ = true;
      }
      writable_.notify_all();
      write_batch(batch);
      {
        std::lock_guard lock(mutex_);
        persisted_ += batch.size();
        for (const auto& frame : batch) {
          landmark_rows_ += frame.new_landmarks;
          observation_count_ += frame.keypoints.size();
        }
        writing_ = false;
      }
      drained_.notify_all();
    }
  } catch (...) {
    {
      std::lock_guard lock(mutex_);
      writer_error_ = std::current_exception();
      writing_ = false;
    }
    writable_.notify_all();
    drained_.notify_all();
  }
}

void FeatureStore::write_batch(std::vector<FrameFeatures>& batch) {
  std::uint64_t committed_frames;
  std::uint64_t committed_landmarks;
  std::uint64_t committed_observations;
  {
    std::lock_guard lock(mutex_);
    committed_frames = persisted_;
    committed_landmarks = landmark_rows_;
    committed_observations = observation_count_;
  }
  check(sqlite3_exec(db_, "BEGIN IMMEDIATE", nullptr, nullptr, nullptr), db_, "begin frame batch");
  sqlite3_stmt* statement = nullptr;
  sqlite3_stmt* landmark_statement = nullptr;
  try {
    constexpr auto sql =
        "INSERT OR REPLACE INTO frames(frame_index,timestamp_ns,pts,width,height,"
        "keypoint_count,keypoints_xy_f32,scores_f32,descriptors,descriptor_encoding,"
        "decode_ms,preprocess_ms,inference_ms,readback_ms,landmark_ids_u64,"
        "landmark_similarities_f32,new_landmarks,matched_landmarks,tracking_ms) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";
    check(sqlite3_prepare_v2(db_, sql, -1, &statement, nullptr), db_, "prepare frame insert");
    constexpr auto landmark_sql =
        "INSERT INTO landmarks(landmark_id,observation_count,resultant_f32,"
        "concentration,first_frame,last_frame) VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(landmark_id) DO UPDATE SET observation_count=excluded.observation_count,"
        "resultant_f32=excluded.resultant_f32,concentration=excluded.concentration,"
        "last_frame=excluded.last_frame";
    check(sqlite3_prepare_v2(db_, landmark_sql, -1, &landmark_statement, nullptr),
          db_, "prepare landmark update");
    for (const auto& frame : batch) {
      committed_frames += 1;
      committed_landmarks += frame.new_landmarks;
      committed_observations += frame.keypoints.size();
      std::vector<float> xy;
      std::vector<float> scores;
      xy.reserve(frame.keypoints.size() * 2);
      scores.reserve(frame.keypoints.size());
      for (const auto& point : frame.keypoints) {
        xy.push_back(point.x);
        xy.push_back(point.y);
        scores.push_back(point.score);
      }
      std::vector<std::uint16_t> half_descriptors;
      const void* descriptor_data = frame.descriptors.data();
      std::size_t descriptor_bytes = frame.descriptors.size() * sizeof(float);
      const char* encoding = "f32";
      if (config_.descriptor_encoding == DescriptorEncoding::none) {
        encoding = "none";
      } else if (config_.descriptor_encoding == DescriptorEncoding::float16) {
        half_descriptors.resize(frame.descriptors.size());
        std::transform(frame.descriptors.begin(), frame.descriptors.end(),
                       half_descriptors.begin(), float_to_half);
        descriptor_data = half_descriptors.data();
        descriptor_bytes = half_descriptors.size() * sizeof(std::uint16_t);
        encoding = "f16";
      }

      sqlite3_bind_int64(statement, 1, static_cast<sqlite3_int64>(frame.frame_index));
      sqlite3_bind_int64(statement, 2, frame.timestamp_ns);
      sqlite3_bind_int64(statement, 3, frame.pts);
      sqlite3_bind_int(statement, 4, frame.width);
      sqlite3_bind_int(statement, 5, frame.height);
      sqlite3_bind_int(statement, 6, static_cast<int>(frame.keypoints.size()));
      bind_blob(statement, 7, xy.data(), xy.size() * sizeof(float), db_);
      bind_blob(statement, 8, scores.data(), scores.size() * sizeof(float), db_);
      bind_blob(statement, 9, descriptor_data, descriptor_bytes, db_);
      sqlite3_bind_text(statement, 10, encoding, -1, SQLITE_STATIC);
      sqlite3_bind_double(statement, 11, frame.decode_ms);
      sqlite3_bind_double(statement, 12, frame.preprocess_ms);
      sqlite3_bind_double(statement, 13, frame.inference_ms);
      sqlite3_bind_double(statement, 14, frame.readback_ms);
      bind_blob(statement, 15, frame.landmark_ids.data(),
                frame.landmark_ids.size() * sizeof(std::uint64_t), db_);
      bind_blob(statement, 16, frame.landmark_similarities.data(),
                frame.landmark_similarities.size() * sizeof(float), db_);
      sqlite3_bind_int(statement, 17, static_cast<int>(frame.new_landmarks));
      sqlite3_bind_int(statement, 18, static_cast<int>(frame.matched_landmarks));
      sqlite3_bind_double(statement, 19, frame.tracking_ms);
      check(sqlite3_step(statement), db_, "insert frame");
      check(sqlite3_reset(statement), db_, "reset frame insert");
      sqlite3_clear_bindings(statement);
      for (const auto& landmark : frame.landmark_updates) {
        sqlite3_bind_int64(landmark_statement, 1, static_cast<sqlite3_int64>(landmark.id));
        sqlite3_bind_int64(landmark_statement, 2,
                           static_cast<sqlite3_int64>(landmark.observation_count));
        bind_blob(landmark_statement, 3, landmark.resultant.data(),
                  config_.descriptor_encoding == DescriptorEncoding::none ? 0 :
                  landmark.resultant.size() * sizeof(float), db_);
        sqlite3_bind_double(landmark_statement, 4, landmark.concentration);
        sqlite3_bind_int64(landmark_statement, 5,
                           static_cast<sqlite3_int64>(landmark.first_frame));
        sqlite3_bind_int64(landmark_statement, 6,
                           static_cast<sqlite3_int64>(landmark.last_frame));
        check(sqlite3_step(landmark_statement), db_, "update landmark");
        check(sqlite3_reset(landmark_statement), db_, "reset landmark update");
        sqlite3_clear_bindings(landmark_statement);
      }
    }
    sqlite3_finalize(statement);
    statement = nullptr;
    sqlite3_finalize(landmark_statement);
    landmark_statement = nullptr;
    set_metadata(db_, "frame_rows", std::to_string(committed_frames));
    set_metadata(db_, "landmark_rows", std::to_string(committed_landmarks));
    set_metadata(db_, "observation_count", std::to_string(committed_observations));
    check(sqlite3_exec(db_, "COMMIT", nullptr, nullptr, nullptr), db_, "commit frame batch");
  } catch (...) {
    if (statement) {
      sqlite3_finalize(statement);
    }
    if (landmark_statement) sqlite3_finalize(landmark_statement);
    sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
    throw;
  }
}

}  // namespace slam_native
