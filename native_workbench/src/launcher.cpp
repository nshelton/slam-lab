#define GL_GLEXT_PROTOTYPES
#include "slam_native/launcher.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h>
#include <sqlite3.h>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <system_error>
#include <utility>

namespace slam_native {
namespace {

namespace fs = std::filesystem;

fs::path expand_home(const std::string& value) {
  if (value == "~" || value.rfind("~/", 0) == 0) {
    if (const char* home = std::getenv("HOME")) {
      return fs::path(home) / value.substr(value == "~" ? 1 : 2);
    }
  }
  return value;
}

std::uintmax_t file_size_if_present(const fs::path& path) {
  std::error_code error;
  return fs::is_regular_file(path, error) ? fs::file_size(path, error) : 0;
}

std::string sqlite_string(sqlite3* db, const char* sql) {
  sqlite3_stmt* statement = nullptr;
  if (sqlite3_prepare_v2(db, sql, -1, &statement, nullptr) != SQLITE_OK) {
    const std::string error = sqlite3_errmsg(db);
    if (statement) sqlite3_finalize(statement);
    return "error: " + error;
  }
  std::string value;
  if (sqlite3_step(statement) == SQLITE_ROW && sqlite3_column_type(statement, 0) != SQLITE_NULL) {
    value = reinterpret_cast<const char*>(sqlite3_column_text(statement, 0));
  }
  sqlite3_finalize(statement);
  return value;
}

std::uint64_t sqlite_count(sqlite3* db, const char* sql) {
  sqlite3_stmt* statement = nullptr;
  if (sqlite3_prepare_v2(db, sql, -1, &statement, nullptr) != SQLITE_OK) {
    if (statement) sqlite3_finalize(statement);
    return 0;
  }
  std::uint64_t count = 0;
  if (sqlite3_step(statement) == SQLITE_ROW) {
    count = static_cast<std::uint64_t>(sqlite3_column_int64(statement, 0));
  }
  sqlite3_finalize(statement);
  return count;
}

std::uint64_t sqlite_count_or_metadata(sqlite3* db, const char* key,
                                       const char* fallback_sql) {
  const std::string query = "SELECT value FROM metadata WHERE key='" +
                            std::string(key) + "'";
  const std::string value = sqlite_string(db, query.c_str());
  std::uint64_t count = 0;
  if (!value.empty()) {
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);
    if (parsed.ec == std::errc() && parsed.ptr == value.data() + value.size()) return count;
  }
  return sqlite_count(db, fallback_sql);
}

// Absolute path of a text field; empty stays empty (fs::absolute throws on it).
fs::path absolute_path(const std::string& value) {
  if (value.empty()) return {};
  std::error_code error;
  const fs::path path = fs::absolute(expand_home(value), error);
  return error ? expand_home(value) : path;
}

DatabaseStats inspect_database(const fs::path& path, bool inspect_rows) {
  DatabaseStats stats;
  if (path.empty()) return stats;
  std::error_code error;
  stats.exists = fs::is_regular_file(path, error);
  if (!stats.exists) return stats;
  stats.main_bytes = file_size_if_present(path);
  stats.wal_bytes = file_size_if_present(path.string() + "-wal");
  if (!inspect_rows) return stats;
  sqlite3* db = nullptr;
  if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    stats.error = db ? sqlite3_errmsg(db) : "Cannot open database";
    if (db) sqlite3_close(db);
    return stats;
  }
  sqlite3_busy_timeout(db, 100);
  stats.schema_version = sqlite_string(
      db, "SELECT value FROM metadata WHERE key='schema_version'");
  if (stats.schema_version == "3") {
    stats.source_video = sqlite_string(
        db, "SELECT value FROM metadata WHERE key='source_video'");
    stats.engine_path = sqlite_string(
        db, "SELECT value FROM metadata WHERE key='engine_path'");
    stats.tracker_algorithm = sqlite_string(
        db, "SELECT value FROM metadata WHERE key='tracker_algorithm'");
    stats.frame_rows = sqlite_count_or_metadata(db, "frame_rows", "SELECT COUNT(*) FROM frames");
    stats.landmark_rows = sqlite_count_or_metadata(db, "landmark_rows",
                                                   "SELECT COUNT(*) FROM landmarks");
    stats.observations = sqlite_count_or_metadata(
        db, "observation_count", "SELECT COALESCE(SUM(keypoint_count),0) FROM frames");
  } else if (stats.schema_version.empty()) {
    stats.error = "Not a native workbench database";
  } else if (stats.schema_version.rfind("error:", 0) == 0) {
    stats.error = stats.schema_version;
  } else {
    stats.error = "Database schema " + stats.schema_version +
                  " is incompatible; choose a new database path";
  }
  sqlite3_close(db);
  return stats;
}

// FNV-1a: stable across runs and builds, unlike std::hash.
std::uint64_t fnv1a(const std::string& text) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (unsigned char character : text) {
    hash ^= character;
    hash *= 1099511628211ULL;
  }
  return hash;
}

bool is_expected_file(const fs::path& path, int target) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (target == 0) {
    return extension == ".mp4" || extension == ".mov" || extension == ".mkv" ||
           extension == ".avi" || extension == ".m4v" || extension == ".webm" ||
           extension == ".mpg" || extension == ".mpeg" || extension == ".mts" ||
           extension == ".m2ts" || extension == ".ts";
  }
  return extension == ".engine" || extension == ".plan";
}

}  // namespace

const DatabaseStats& DatabaseSummary::get(const fs::path& path, bool inspect_rows) {
  const auto now = std::chrono::steady_clock::now();
  if (path != path_ || inspect_rows != inspected_rows_ ||
      now - refreshed_ >= std::chrono::milliseconds(500)) {
    path_ = path;
    inspected_rows_ = inspect_rows;
    stats_ = inspect_database(path, inspect_rows);
    refreshed_ = now;
  }
  return stats_;
}

// Size and modification time, as FeatureStore's source/engine identity (minus
// the engine content hash, which is too slow to recompute every UI frame).
static std::string file_stamp(const fs::path& path) {
  std::error_code size_error, time_error;
  const auto size = fs::file_size(path, size_error);
  const auto time = fs::last_write_time(path, time_error);
  if (size_error || time_error) return "missing";
  return std::to_string(size) + ':' + std::to_string(time.time_since_epoch().count());
}

fs::path default_point_cache(const fs::path& repository_root, const AppConfig& config) {
  std::error_code video_error, engine_error;
  const fs::path video = fs::weakly_canonical(config.video, video_error);
  const fs::path engine = fs::weakly_canonical(config.engine, engine_error);
  // Everything FeatureStore::set_session refuses to change in an existing cache.
  std::ostringstream key;
  key << video.string() << '\n' << file_stamp(video) << '\n' << engine.string() << '\n' << file_stamp(engine)
      << "\ntensorrt-superpoint-letterbox-v1 schema 3 " << config.superpoint.input_width << 'x'
      << config.superpoint.input_height << ' ' << config.superpoint.max_keypoints << ' ' << std::hexfloat
      << config.superpoint.detection_threshold << ' ' << static_cast<int>(config.store.descriptor_encoding);
  std::string stem = config.video.stem().string();
  if (stem.empty()) stem = "video";
  for (char& character : stem) {
    if (!std::isalnum(static_cast<unsigned char>(character)) &&
        character != '-' && character != '_') character = '-';
  }
  std::ostringstream name;
  name << stem << '-' << std::hex << std::setw(12) << std::setfill('0') << (fnv1a(key.str()) >> 16) << ".sqlite3";
  return repository_root / "pointcache" / name.str();
}

fs::path thumbnail_path(const fs::path& repository_root, const fs::path& video) {
  std::error_code error;
  const fs::path canonical = fs::weakly_canonical(video, error);
  std::string stem = video.stem().string();
  if (stem.empty()) stem = "video";
  for (char& character : stem)
    if (!std::isalnum(static_cast<unsigned char>(character)) && character != '-' && character != '_') character = '-';
  std::ostringstream name;
  name << stem << '-' << std::hex << std::setw(12) << std::setfill('0')
       << (fnv1a(canonical.string() + '\n' + file_stamp(canonical)) >> 16) << ".ppm";
  return repository_root / "pointcache" / "thumbnails" / name.str();
}

std::string format_bytes(std::uintmax_t bytes) {
  constexpr const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  std::size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < std::size(units)) {
    value /= 1024.0;
    ++unit;
  }
  std::ostringstream text;
  text << std::fixed << std::setprecision(unit == 0 ? 0 : 1) << value << ' ' << units[unit];
  return text.str();
}

Launcher::Launcher(AppConfig initial, fs::path repository_root)
    : initial_(std::move(initial)),
      recents_(RecentSessions::default_storage_path()),
      repository_root_(std::move(repository_root)),
      video_(initial_.video.string()), engine_(initial_.engine.string()) {}

Launcher::~Launcher() {
  {
    std::lock_guard lock(thumbnail_mutex_);
    thumbnail_stop_ = true;
  }
  thumbnail_wake_.notify_all();
  if (thumbnail_thread_.joinable()) thumbnail_thread_.join();
  for (auto& [video, slot] : thumbnails_)
    if (slot.texture) glDeleteTextures(1, &slot.texture);
}

// Worker: decode queued videos one at a time (disk cache first).
void Launcher::thumbnail_worker() {
  for (;;) {
    fs::path video;
    {
      std::unique_lock lock(thumbnail_mutex_);
      thumbnail_wake_.wait(lock, [&] { return thumbnail_stop_ || !thumbnail_queue_.empty(); });
      if (thumbnail_stop_) return;
      video = thumbnail_queue_.front();
      thumbnail_queue_.pop_front();
    }
    ThumbnailResult result{video, std::nullopt, {}};
    try {
      const fs::path cached = thumbnail_path(repository_root_, video);
      result.image = load_ppm(cached);
      if (!result.image) {
        std::error_code error;
        if (!fs::is_regular_file(video, error)) throw std::runtime_error("file missing");
        result.image = make_thumbnail(video);
        save_ppm(cached, *result.image);
      }
    } catch (const std::exception& error) {
      result.error = error.what();
    }
    std::lock_guard lock(thumbnail_mutex_);
    thumbnail_results_.push_back(std::move(result));
  }
}

Launcher::ThumbnailSlot& Launcher::thumbnail(const fs::path& video) {
  auto [it, inserted] = thumbnails_.try_emplace(video);
  if (inserted) {
    {
      std::lock_guard lock(thumbnail_mutex_);
      thumbnail_queue_.push_back(video);
    }
    if (!thumbnail_thread_.joinable()) thumbnail_thread_ = std::thread([this] { thumbnail_worker(); });
    thumbnail_wake_.notify_one();
  }
  return it->second;
}

// Main thread: upload finished thumbnails as textures.
void Launcher::collect_thumbnails() {
  std::vector<ThumbnailResult> results;
  {
    std::lock_guard lock(thumbnail_mutex_);
    results.swap(thumbnail_results_);
  }
  for (auto& result : results) {
    auto& slot = thumbnails_[result.video];
    if (!result.image) {
      slot.state = ThumbnailSlot::State::failed;
      slot.error = result.error;
      continue;
    }
    GLint previous = 0, alignment = 4;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &previous);
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &alignment);
    if (!slot.texture) glGenTextures(1, &slot.texture);
    glBindTexture(GL_TEXTURE_2D, slot.texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, result.image->width, result.image->height, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 result.image->rgb.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(previous));
    slot.state = ThumbnailSlot::State::ready;
    slot.width = result.image->width;
    slot.height = result.image->height;
  }
}

bool Launcher::draw_recent_grid() {
  constexpr int kColumns = 4;
  const auto& recents = recents_.entries();
  if (recents.empty()) {
    ImGui::TextDisabled("No recent videos yet: load a sequence below.");
    return false;
  }
  collect_thumbnails();
  const ImGuiStyle& style = ImGui::GetStyle();
  const float cell_width = (ImGui::GetContentRegionAvail().x - (kColumns - 1) * style.ItemSpacing.x) / kColumns;
  const float image_height = cell_width * 9.0F / 16.0F;
  const float text_height = ImGui::GetTextLineHeight();
  ImDrawList* draw = ImGui::GetWindowDrawList();
  bool start = false;
  for (std::size_t i = 0; i < recents.size() && i < RecentSessions::kCapacity; ++i) {
    const RecentSession& recent = recents[i];
    if (i % kColumns) ImGui::SameLine();
    ImGui::PushID(static_cast<int>(i));
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    const ImVec2 size{cell_width, image_height + text_height + 6};
    const bool clicked = ImGui::InvisibleButton("recent", size);
    const bool hovered = ImGui::IsItemHovered();
    const bool selected = absolute_path(video_) == recent.video;
    std::error_code error;
    const bool exists = fs::is_regular_file(recent.video, error);
    if (clicked) {
      video_ = recent.video.string();
      if (!recent.engine.empty()) engine_ = recent.engine.string();
      error_.clear();
      start = exists && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
    }
    // Image area: thumbnail letterboxed into 16:9, or a placeholder.
    const ImVec2 i0 = p0, i1{p0.x + cell_width, p0.y + image_height};
    draw->AddRectFilled(i0, i1, IM_COL32(24, 26, 30, 255), 4.0F);
    const ThumbnailSlot& slot = thumbnail(recent.video);
    if (slot.state == ThumbnailSlot::State::ready) {
      const float scale = std::min(cell_width / slot.width, image_height / slot.height);
      const float w = slot.width * scale, h = slot.height * scale;
      const ImVec2 a{i0.x + 0.5F * (cell_width - w), i0.y + 0.5F * (image_height - h)};
      draw->AddImage(static_cast<ImTextureID>(slot.texture), a, {a.x + w, a.y + h});
    } else {
      const char* label = slot.state == ThumbnailSlot::State::pending ? "loading..." :
                          exists ? "no preview" : "file missing";
      const ImVec2 text = ImGui::CalcTextSize(label);
      draw->AddText({i0.x + 0.5F * (cell_width - text.x), i0.y + 0.5F * (image_height - text.y)},
                    IM_COL32(150, 150, 150, 255), label);
    }
    if (!exists) draw->AddRectFilled(i0, i1, IM_COL32(0, 0, 0, 140), 4.0F);
    const ImU32 border = selected ? ImGui::GetColorU32(ImGuiCol_ButtonActive) :
                         hovered ? ImGui::GetColorU32(ImGuiCol_ButtonHovered) : IM_COL32(60, 64, 72, 255);
    draw->AddRect(i0, i1, border, 4.0F, 0, selected ? 3.0F : 1.0F);
    // Name, clipped to the cell; with the parent folder when another recent
    // video has the same file name (e.g. the TUM sequences' rgb.mp4).
    std::string name = recent.video.filename().string();
    if (std::count_if(recents.begin(), recents.end(),
                      [&](const RecentSession& other) { return other.video.filename() == recent.video.filename(); }) > 1)
      name = recent.video.parent_path().filename().string() + "/" + name;
    draw->PushClipRect({p0.x, i1.y}, {p0.x + cell_width, p0.y + size.y}, true);
    draw->AddText({p0.x + 2, i1.y + 3}, exists ? ImGui::GetColorU32(ImGuiCol_Text) :
                                          ImGui::GetColorU32(ImGuiCol_TextDisabled), name.c_str());
    draw->PopClipRect();
    if (hovered) {
      ImGui::SetTooltip("%s%s%s\nClick to select, double-click to open", recent.video.c_str(),
                        exists ? "" : "\n(file missing)",
                        slot.state == ThumbnailSlot::State::failed && exists ? ("\npreview: " + slot.error).c_str() : "");
    }
    ImGui::PopID();
  }
  return start;
}

void Launcher::set_error(std::string error) { error_ = std::move(error); }

void Launcher::record_recent(const AppConfig& config) {
  try {
    recents_.record({config.video,
                     config.engine,
                     config.database});
  } catch (const std::exception& error) {
    error_ = std::string("Could not save recent videos: ") + error.what();
  }
}

void Launcher::change_directory(const fs::path& directory) {
  std::error_code error;
  const fs::path resolved = fs::absolute(directory, error);
  if (error || !fs::is_directory(resolved, error)) {
    browser_error_ = "Cannot open folder: " + directory.string();
    return;
  }
  browser_directory_ = resolved;
  browser_folder_ = resolved.string();
  browser_error_.clear();
  refresh_entries();
}

void Launcher::refresh_entries() {
  entries_.clear();
  std::error_code error;
  for (fs::directory_iterator it(browser_directory_,
                                  fs::directory_options::skip_permission_denied, error), end;
       it != end && !error; it.increment(error)) {
    entries_.push_back(*it);
  }
  if (error) browser_error_ = error.message();
  std::sort(entries_.begin(), entries_.end(), [](const auto& left, const auto& right) {
    std::error_code left_error, right_error;
    const bool left_directory = left.is_directory(left_error);
    const bool right_directory = right.is_directory(right_error);
    if (left_directory != right_directory) return left_directory;
    return left.path().filename().string() < right.path().filename().string();
  });
}

std::string& Launcher::target_path() {
  return browser_target_ == Target::video ? video_ : engine_;
}

void Launcher::open_browser(Target target) {
  browser_target_ = target;
  browser_open_ = true;
  browser_popup_pending_ = true;
  browser_filename_.clear();
  const fs::path current = expand_home(target_path());
  std::error_code error;
  fs::path directory = fs::is_directory(current, error) ? current : current.parent_path();
  while (!directory.empty() && !fs::is_directory(directory, error)) {
    directory = directory.parent_path();
  }
  if (directory.empty() || !fs::is_directory(directory, error)) {
    if (const char* home = std::getenv("HOME")) directory = home;
    else directory = fs::current_path();
  }
  if (!fs::is_directory(current, error)) browser_filename_ = current.filename().string();
  change_directory(directory);
}

void Launcher::draw_browser() {
  if (browser_popup_pending_) {
    ImGui::OpenPopup("Browse files");
    browser_popup_pending_ = false;
  }
  if (!ImGui::BeginPopupModal("Browse files", &browser_open_,
                             ImGuiWindowFlags_AlwaysAutoResize)) return;
  const char* label = browser_target_ == Target::video ? "Video" : "TensorRT engine";
  ImGui::Text("Choose %s", label);
  ImGui::SetNextItemWidth(620);
  ImGui::InputText("Folder", &browser_folder_);
  ImGui::SameLine();
  if (ImGui::Button("Go")) change_directory(expand_home(browser_folder_));
  if (ImGui::Button("Up")) change_directory(browser_directory_.parent_path());
  ImGui::SameLine();
  if (ImGui::Button("Home")) {
    if (const char* home = std::getenv("HOME")) change_directory(home);
  }
  ImGui::SameLine();
  if (ImGui::Button("Refresh")) refresh_entries();
  ImGui::SameLine();
  ImGui::Checkbox("Show all files", &show_all_files_);
  if (!browser_error_.empty()) ImGui::TextColored({1, 0.5F, 0.4F, 1}, "%s", browser_error_.c_str());
  ImGui::BeginChild("File list", {800, 360}, ImGuiChildFlags_Borders);
  fs::path next_directory;
  for (const auto& entry : entries_) {
    std::error_code error;
    const bool directory = entry.is_directory(error);
    if (!directory && !show_all_files_ &&
        !is_expected_file(entry.path(), static_cast<int>(browser_target_))) continue;
    const std::string name = (directory ? "[folder] " : "          ") +
                             entry.path().filename().string();
    if (ImGui::Selectable(name.c_str(), !directory && browser_filename_ == entry.path().filename())) {
      if (directory) next_directory = entry.path();
      else browser_filename_ = entry.path().filename().string();
      if (!directory && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        target_path() = (browser_directory_ / browser_filename_).string();
        browser_open_ = false;
        ImGui::CloseCurrentPopup();
      }
    }
  }
  ImGui::EndChild();
  if (!next_directory.empty()) change_directory(next_directory);
  ImGui::SetNextItemWidth(620);
  ImGui::InputText("File name", &browser_filename_);
  if (ImGui::Button("Open file")) {
    const fs::path picked = browser_directory_ / browser_filename_;
    std::error_code error;
    if (browser_filename_.empty() || !fs::is_regular_file(picked, error)) {
      browser_error_ = "Select an existing file";
    } else {
      target_path() = picked.string();
      browser_open_ = false;
      ImGui::CloseCurrentPopup();
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel")) {
    browser_open_ = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

bool Launcher::draw(AppConfig& selected, DatabaseSummary& database_summary) {
  ImGui::SetNextWindowPos({30, 30}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize({900, 840}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSizeConstraints({700, 780}, {100000, 100000});  // room for the 4x4 grid
  ImGui::Begin("Open SLAM session");
  if (video_only_) {
    ImGui::TextWrapped("%s", video_only_->c_str());
    ImGui::Separator();
  } else {
    ImGui::TextWrapped("SuperPoint detections are cached. Tracks are rebuilt in memory on each run using optical flow.");
    ImGui::Separator();
    ImGui::TextUnformatted("Tracking: SuperPoint + NVIDIA optical flow");
  }
  ImGui::TextUnformatted("Recent videos");
  const bool open_recent = draw_recent_grid();
  ImGui::Spacing();
  ImGui::TextUnformatted("Video");
  ImGui::SetNextItemWidth(-170);
  ImGui::InputText("##video", &video_);
  ImGui::SameLine();
  if (ImGui::Button("Load new sequence...")) open_browser(Target::video);
  if (video_only_) {
    AppConfig session = initial_;
    session.video = absolute_path(video_);
    ImGui::Separator();
    if (!error_.empty()) ImGui::TextColored({1, 0.5F, 0.4F, 1}, "%s", error_.c_str());
    ImGui::Spacing();
    std::error_code video_error;
    const bool valid = fs::is_regular_file(session.video, video_error);
    ImGui::BeginDisabled(!valid);
    const bool start = ImGui::Button("Start session", {160, 34}) || (open_recent && valid);
    ImGui::EndDisabled();
    if (!valid) ImGui::TextDisabled("Select an existing video.");
    if (start) {
      selected = std::move(session);
      error_.clear();
    }
    ImGui::End();
    draw_browser();
    return start;
  }
  ImGui::TextUnformatted("TensorRT engine");
  ImGui::SetNextItemWidth(-95);
  ImGui::InputText("##engine", &engine_);
  ImGui::SameLine();
  if (ImGui::Button("Browse##engine")) open_browser(Target::engine);
  AppConfig session = initial_;
  session.video = absolute_path(video_);
  session.engine = absolute_path(engine_);
  session.database = video_.empty() ? fs::path{} : default_point_cache(repository_root_, session);
  ImGui::Separator();
  const auto& stats = database_summary.get(session.database);
  if (video_.empty()) {
    ImGui::TextDisabled("Point cache: chosen automatically for the video");
  } else if (!stats.exists) {
    ImGui::TextUnformatted("Point cache: none yet (SuperPoint runs on every frame the first time)");
  } else {
    ImGui::Text("Point cache: %llu frames (%s), reused on replay",
                static_cast<unsigned long long>(stats.frame_rows),
                format_bytes(stats.main_bytes + stats.wal_bytes).c_str());
  }
  if (!video_.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", session.database.c_str());
  if (!stats.error.empty()) ImGui::TextColored({1, 0.5F, 0.4F, 1}, "%s", stats.error.c_str());
  if (!error_.empty()) ImGui::TextColored({1, 0.5F, 0.4F, 1}, "%s", error_.c_str());
  ImGui::Spacing();
  std::error_code video_error, engine_error;
  const bool valid = fs::is_regular_file(session.video, video_error) &&
                     fs::is_regular_file(session.engine, engine_error) && stats.error.empty();
  ImGui::BeginDisabled(!valid);
  const bool start = ImGui::Button(stats.exists ? "Replay with cache" : "Start session", {160, 34}) ||
                     (open_recent && valid);
  ImGui::EndDisabled();
  if (!valid) ImGui::TextDisabled("Select an existing video and engine.");
  if (start) {
    selected = std::move(session);
    error_.clear();
  }
  ImGui::End();
  draw_browser();
  return start;
}

}  // namespace slam_native
