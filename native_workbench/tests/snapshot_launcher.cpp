// Headless screenshot of the launcher (recent-video grid with thumbnails), for
// a visual check. Uses the user's real recent-video list and thumbnail cache
// (thumbnails it generates are kept). Not part of CTest.
// usage: slam-native-snapshot-launcher REPOSITORY_ROOT OUT.ppm [SECONDS]
// SECONDS (default 20): how long to render while thumbnails decode.
#include "../src/app.cpp"
#include <fstream>
#include <iostream>
#include <thread>

int main(int argc, char** argv) {
  using namespace slam_native;
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: slam-native-snapshot-launcher REPOSITORY_ROOT OUT.ppm [SECONDS]\n";
    return 2;
  }
  try {
    GlfwLifetime glfw;
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    Window window;
    ImGuiLifetime imgui(window.get());
    ImGui::GetIO().IniFilename = nullptr;  // never touch the user's layout
    AppConfig initial;
    Launcher launcher(initial, argv[1]);
    DatabaseSummary database_summary;
    int width = 0, height = 0;
    // Render while the thumbnails decode (worker thread).
    const auto deadline = Clock::now() + std::chrono::seconds(argc == 4 ? std::stoi(argv[3]) : 20);
    while (Clock::now() < deadline) {
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      AppConfig selected;
      launcher.draw(selected, database_summary);
      ImGui::Render();
      glfwGetFramebufferSize(window.get(), &width, &height);
      glViewport(0, 0, width, height);
      glClearColor(0.025F, 0.03F, 0.04F, 1.0F);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::vector<unsigned char> pixels(static_cast<std::size_t>(width) * height * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    std::ofstream out(argv[2], std::ios::binary);
    out << "P6\n" << width << ' ' << height << "\n255\n";
    for (int y = height - 1; y >= 0; --y)
      out.write(reinterpret_cast<const char*>(pixels.data() + static_cast<std::size_t>(y) * width * 3), width * 3);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
