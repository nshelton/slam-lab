#include "slam_native/feature_store.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <cmath>
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
  descriptor_encoding TEXT NOT NULL CHECK(descriptor_encoding IN ('f16', 'f32')),
  decode_ms REAL NOT NULL,
  preprocess_ms REAL NOT NULL,
  inference_ms REAL NOT NULL,
  readback_ms REAL NOT NULL
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


std::string file_identity(const std::filesystem::path& path, bool hash_contents) {
  std::ostringstream result;
  result << std::filesystem::file_size(path) << ':' <<
      std::filesystem::last_write_time(path).time_since_epoch().count();
  if (hash_contents) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot fingerprint engine: " + path.string());
    std::uint64_t hash = 14695981039346656037ULL;
    char buffer[65536];
    while (input.read(buffer, sizeof(buffer)) || input.gcount()) {
      for (std::streamsize i = 0; i < input.gcount(); ++i) {
        hash ^= static_cast<unsigned char>(buffer[i]);
        hash *= 1099511628211ULL;
      }
    }
    if (!input.eof()) throw std::runtime_error("Cannot read engine fingerprint");
    result << ':' << std::hex << hash;
  }
  return result.str();
}

float half_to_float(std::uint16_t bits) {
  const int exponent = (bits >> 10) & 31;
  const int mantissa = bits & 1023;
  const float value = exponent == 0 ? std::ldexp(float(mantissa), -24) :
      exponent == 31 ? (mantissa ? std::numeric_limits<float>::quiet_NaN() :
                                  std::numeric_limits<float>::infinity()) :
      std::ldexp(float(1024 + mantissa), exponent - 25);
  return bits & 0x8000 ? -value : value;
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
  // Reject legacy databases before any schema, metadata, or journal writes.
  sqlite3_stmt* probe = nullptr;
  if (sqlite3_prepare_v2(db_, "SELECT value FROM metadata WHERE key='schema_version'",
                       -1, &probe, nullptr) == SQLITE_OK) {
    const int result = sqlite3_step(probe);
    const std::string version = result == SQLITE_ROW ?
        reinterpret_cast<const char*>(sqlite3_column_text(probe, 0)) : "";
    sqlite3_finalize(probe);
    if (version != "3") {
      sqlite3_close(db_);
      db_ = nullptr;
      throw std::runtime_error("Legacy database is unchanged; choose a new SuperPoint cache path");
    }
  } else if (probe) sqlite3_finalize(probe);
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
  if (!existing.empty() && existing != "3") {
    sqlite3_close(db_);
    db_ = nullptr;
    throw std::runtime_error("Unsupported native feature database schema: " + existing);
  }
  set_metadata(db_, "schema_version", "3");
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
  check(sqlite3_open_v2(path.c_str(), &reader_, SQLITE_OPEN_READONLY, nullptr), db_,
        "open cache reader");
  sqlite3_busy_timeout(reader_, 5000);
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
  if (reader_) sqlite3_close(reader_);
  if (db_) {
    sqlite3_close(db_);
  }
}

void FeatureStore::set_session(const std::filesystem::path& source_video,
                               const std::filesystem::path& engine_path,
                               int input_width, int input_height, int max_keypoints,
                               float threshold) {
  set_immutable_metadata(db_, "source_video", std::filesystem::canonical(source_video).string());
  set_immutable_metadata(db_, "source_identity", file_identity(source_video, false));
  set_immutable_metadata(db_, "engine_path", std::filesystem::canonical(engine_path).string());
  set_immutable_metadata(db_, "engine_identity", file_identity(engine_path, true));
  set_immutable_metadata(db_, "extractor", "tensorrt-superpoint-letterbox-v1");
  set_immutable_metadata(db_, "descriptor_dimension", "256");
  set_immutable_metadata(db_, "input_width", std::to_string(input_width));
  set_immutable_metadata(db_, "input_height", std::to_string(input_height));
  set_immutable_metadata(db_, "max_keypoints", std::to_string(max_keypoints));
  std::ostringstream precise;
  precise << std::hexfloat << threshold;
  set_immutable_metadata(db_, "detection_threshold", precise.str());
  set_immutable_metadata(db_, "descriptor_encoding",
      config_.descriptor_encoding == DescriptorEncoding::float16 ? "f16" : "f32");
}

std::optional<FrameFeatures> FeatureStore::load_frame(const GpuFrame& frame) const {
  sqlite3_stmt* statement = nullptr;
  check(sqlite3_prepare_v2(reader_,
      "SELECT frame_index,timestamp_ns,pts,width,height,keypoint_count,keypoints_xy_f32,"
      "scores_f32,descriptors,descriptor_encoding FROM frames WHERE timestamp_ns=? AND pts=?",
      -1, &statement, nullptr), reader_, "prepare cached frame");
  const auto finalize = [](sqlite3_stmt* value) { sqlite3_finalize(value); };
  std::unique_ptr<sqlite3_stmt, decltype(finalize)> guard(statement, finalize);
  sqlite3_bind_int64(statement, 1, frame.timestamp_ns);
  sqlite3_bind_int64(statement, 2, frame.pts);
  const int status = sqlite3_step(statement);
  if (status == SQLITE_DONE) return std::nullopt;
  check(status, reader_, "read cached frame");
  FrameFeatures result;
  result.frame_index = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 0));
  result.timestamp_ns = sqlite3_column_int64(statement, 1);
  result.pts = sqlite3_column_int64(statement, 2);
  result.width = sqlite3_column_int(statement, 3);
  result.height = sqlite3_column_int(statement, 4);
  const int count = sqlite3_column_int(statement, 5);
  const std::string encoding = reinterpret_cast<const char*>(sqlite3_column_text(statement, 9));
  const std::int64_t descriptor_bytes = encoding == "f16" ? 2 : 4;
  if (count < 0 || result.width != frame.width || result.height != frame.height ||
      (encoding != "f16" && encoding != "f32") ||
      sqlite3_column_bytes(statement, 6) != std::int64_t(count) * 8 ||
      sqlite3_column_bytes(statement, 7) != std::int64_t(count) * 4 ||
      sqlite3_column_bytes(statement, 8) != std::int64_t(count) * 256 * descriptor_bytes) {
    throw std::runtime_error("Invalid cached SuperPoint frame");
  }
  const auto* xy = static_cast<const float*>(sqlite3_column_blob(statement, 6));
  const auto* scores = static_cast<const float*>(sqlite3_column_blob(statement, 7));
  for (int i = 0; i < count; ++i) result.keypoints.push_back({xy[2*i], xy[2*i+1], scores[i]});
  result.descriptors.resize(static_cast<std::size_t>(count) * 256);
  if (count && encoding == "f32") {
    std::memcpy(result.descriptors.data(), sqlite3_column_blob(statement, 8),
                result.descriptors.size() * sizeof(float));
  } else if (count) {
    const auto* values = static_cast<const std::uint16_t*>(sqlite3_column_blob(statement, 8));
    std::transform(values, values + result.descriptors.size(), result.descriptors.begin(), half_to_float);
  }
  return result;
}

void FeatureStore::enqueue(FrameFeatures features) {
  if (config_.descriptor_encoding == DescriptorEncoding::none ||
      features.descriptors.size() != features.keypoints.size() * 256) {
    throw std::invalid_argument("Cache requires SuperPoint descriptors");
  }
  if (!features.landmark_ids.empty() || !features.landmark_updates.empty() ||
      !features.track_sigmas_px.empty() || !features.landmark_similarities.empty() ||
      !features.superpoint_supported.empty() || !features.flow_predictions.empty() ||
      !features.correction_distances.empty() || features.new_landmarks || features.matched_landmarks ||
      features.coasted_landmarks) {
    throw std::invalid_argument("Only raw SuperPoint observations may be persisted");
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
  std::uint64_t committed_observations;
  {
    std::lock_guard lock(mutex_);
    committed_frames = persisted_;
    committed_observations = observation_count_;
  }
  check(sqlite3_exec(db_, "BEGIN IMMEDIATE", nullptr, nullptr, nullptr), db_, "begin frame batch");
  sqlite3_stmt* statement = nullptr;
  try {
    constexpr auto sql =
        "INSERT INTO frames(frame_index,timestamp_ns,pts,width,height,"
        "keypoint_count,keypoints_xy_f32,scores_f32,descriptors,descriptor_encoding,"
        "decode_ms,preprocess_ms,inference_ms,readback_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)";
    check(sqlite3_prepare_v2(db_, sql, -1, &statement, nullptr), db_, "prepare frame insert");
    for (const auto& frame : batch) {
      committed_frames += 1;
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
      check(sqlite3_step(statement), db_, "insert frame");
      check(sqlite3_reset(statement), db_, "reset frame insert");
      sqlite3_clear_bindings(statement);
    }
    sqlite3_finalize(statement);
    statement = nullptr;
    set_metadata(db_, "frame_rows", std::to_string(committed_frames));
    set_metadata(db_, "observation_count", std::to_string(committed_observations));
    check(sqlite3_exec(db_, "COMMIT", nullptr, nullptr, nullptr), db_, "commit frame batch");
  } catch (...) {
    if (statement) {
      sqlite3_finalize(statement);
    }
    sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
    throw;
  }
}

}  // namespace slam_native
