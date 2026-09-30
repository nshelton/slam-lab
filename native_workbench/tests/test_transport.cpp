// Exercise the actual Session transport, including its decoder, tracker and cache.
#include "../src/app.cpp"
#include <iostream>

int main(int argc, char** argv) {
  using namespace slam_native;
  if (argc != 3) return 2;
  const auto root = std::filesystem::temp_directory_path() /
      ("slam-transport-" + std::to_string(Clock::now().time_since_epoch().count()));
  try {
    GlfwLifetime glfw;
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    Window window;
    validate_cuda_gl_display();
    AppConfig config;
    config.video = argv[1];
    config.engine = argv[2];
    config.database = root / "features.db";
    const auto require = [](bool value, const char* message) {
      if (!value) throw std::runtime_error(message);
    };
    {
      Session session(config);
      session.runtime.realtime_pacing = false;
      session.runtime.show_flow_vectors = true;  // host flow copy exists only for the overlay
      session.advance();
      const auto beginning = session.runtime.timestamp_ns;
      session.runtime.seek_seconds = 2;
      session.runtime.seek_requested = true;
      session.advance();
      const auto sought = session.runtime.timestamp_ns;
      require(sought >= beginning + 1900000000LL && !session.runtime.playing,
              "Seek did not pause at requested position");
      require(session.runtime.matched_landmarks == 0 &&
              session.flow_tracker->landmark_count() == session.runtime.points.size(),
              "Seek did not reset tracking");
      require(session.odometry->trajectory_size() == 0 &&
              session.runtime.odometry.state == OdometryState::initializing,
              "Seek did not reset the camera trajectory");
      session.runtime.playing = true;
      session.advance();
      require(session.runtime.timestamp_ns > sought &&
              session.runtime.timestamp_ns < sought + 100000000LL &&
              session.runtime.flow_field.has_value(),
              "Play did not continue with flow from sought frame");
      session.runtime.stopped = session.runtime.eof = true;
      session.runtime.seek_seconds = 1;
      session.runtime.seek_requested = true;
      session.advance();
      require(!session.runtime.stopped && !session.runtime.eof &&
              session.runtime.timestamp_ns < sought, "Seek failed after stop/EOF");
      session.runtime.step_requested = true;
      const auto before_step = session.runtime.timestamp_ns;
      session.advance();
      require(session.runtime.timestamp_ns > before_step && !session.runtime.playing,
              "Step after seek failed");
      session.runtime.restart_requested = true;
      session.advance();
      require(session.runtime.timestamp_ns == beginning && session.runtime.playing &&
              session.runtime.matched_landmarks == 0 && session.runtime.cache_hit,
              "Restart did not reset to cached beginning");
      session.store->flush();
      const auto rows = session.store->persisted();
      session.runtime.restart_requested = true;
      session.advance();
      session.store->flush();
      require(session.store->persisted() == rows, "Restart duplicated cached detections");
    }
    std::filesystem::remove_all(root);
    std::cout << "Seek/play, backward seek, step, restart, stop/EOF recovery and cache reuse passed.\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\nTest cache: " << root << '\n';
    return 1;
  }
}
