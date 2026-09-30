// Headless screenshot of the real workbench UI after processing some frames
// (visual check of the panels). Not part of CTest.
// usage: slam-native-snapshot VIDEO ENGINE START_SECONDS FRAMES OUT.ppm
#include "../src/app.cpp"
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
  using namespace slam_native;
  if (argc != 6) {
    std::cerr << "usage: slam-native-snapshot VIDEO ENGINE START_SECONDS FRAMES OUT.ppm\n";
    return 2;
  }
  const auto root = std::filesystem::temp_directory_path() /
      ("slam-snapshot-" + std::to_string(Clock::now().time_since_epoch().count()));
  try {
    GlfwLifetime glfw;
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    Window window;
    validate_cuda_gl_display();
    ImGuiLifetime imgui(window.get());
    ImGui::GetIO().IniFilename = nullptr;  // never touch the user's layout
    AppConfig config;
    config.video = argv[1];
    config.engine = argv[2];
    config.database = root / "features.db";
    Session session(config);
    session.runtime.realtime_pacing = false;
    if (std::stod(argv[3]) > 0) {
      session.runtime.seek_seconds = std::stod(argv[3]);
      session.runtime.seek_requested = true;
      session.advance();
    }
    session.runtime.playing = true;
    const int frames = std::stoi(argv[4]);
    for (int i = 0; i < frames; ++i) session.advance();
    DatabaseSummary database_summary;
    int width = 0, height = 0;
    for (int pass = 0; pass < 3; ++pass) {  // first passes settle window sizes
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      DatabaseStats stats = database_summary.get(session.config.database, false);
      draw_sidebar(session.runtime, *session.decoder, *session.store, session.tracker.get(),
                   session.flow_tracker.get(), stats, session.config.database);
      draw_video(session.runtime, true, session.decoder->duration_ns(), session.decoder->start_time_ns());
      draw_flow_diagnostics(session.runtime, *session.flow_tracker);
      session.trajectory_view.draw(*session.odometry, &session.runtime.show_trajectory,
                                   session.runtime.frame_width, session.runtime.frame_height);
      ImGui::Render();
      glfwGetFramebufferSize(window.get(), &width, &height);
      glViewport(0, 0, width, height);
      glClearColor(0.025F, 0.03F, 0.04F, 1.0F);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
    std::vector<unsigned char> pixels(static_cast<std::size_t>(width) * height * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    std::ofstream out(argv[5], std::ios::binary);
    out << "P6\n" << width << ' ' << height << "\n255\n";
    for (int y = height - 1; y >= 0; --y)
      out.write(reinterpret_cast<const char*>(pixels.data() + static_cast<std::size_t>(y) * width * 3), width * 3);
    session.store->flush();
    std::cout << "odometry: " << to_string(session.odometry->last().state) << ", "
              << session.odometry->trajectory_size() << " posed frames\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::filesystem::remove_all(root);
}
