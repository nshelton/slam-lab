// Headless screenshot of the LSD workbench after processing some frames, plus
// its trajectory in the track_io CSV format (for tools/tum_eval.py). Not part of CTest.
// usage: slam-native-lsd-snapshot VIDEO MODELS_DIR FRAMES OUT.ppm [TRAJECTORY.csv] [DEPTH_VIEW]
// Environment: LSD_STEREO=0 (network depth only), LSD_UNDISTORT=0,
// LSD_GRAPH=0, LSD_RECIPROCAL=X, LSD_SIGMA_L / LSD_SIGMA_I (stereo noise model), LSD_EXPORT_DEPTH=FILE.csv: every keyframe's depth at distorted source pixels
// with the scale-locked network prior there (tools/lsd_depth_vs_gt.py).
#include "../src/lsd_app.cpp"

#include <Eigen/Geometry>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  using namespace slam_native;
  if (argc < 5) {
    std::cerr << "usage: slam-native-lsd-snapshot VIDEO MODELS_DIR FRAMES OUT.ppm [TRAJECTORY.csv] [DEPTH_VIEW]\n";
    return 2;
  }
  try {
    GlfwLifetime glfw;
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    Window window;
    validate_cuda_gl_display();
    ImGuiLifetime imgui(window.get());
    ImGui::GetIO().IniFilename = nullptr;  // never touch the user's layout
    LsdAppConfig config;
    config.base.video = argv[1];
    config.base.models_dir = argv[2];
    config.base.engine = std::filesystem::path(argv[2]) / "superpoint-1024x576-k2048.engine";
    if (const char* v = std::getenv("LSD_LOOPS")) config.odometry.loops.enabled = std::string(v) != "0";
    if (const char* v = std::getenv("LSD_STEREO")) config.odometry.stereo = std::string(v) != "0";
    if (const char* v = std::getenv("LSD_UNDISTORT")) config.undistort = std::string(v) != "0";
    if (const char* v = std::getenv("LSD_SIGMA_L")) config.odometry.filter.sigma_epipolar_px = std::stod(v);
    if (const char* v = std::getenv("LSD_SIGMA_I")) config.odometry.filter.sigma_intensity = std::stod(v);
    if (const char* v = std::getenv("LSD_PRIOR_SIGMA")) config.odometry.prior_relative_sigma = std::stod(v);
    if (const char* v = std::getenv("LSD_GRAPH")) config.odometry.graph = std::string(v) != "0";
    if (const char* v = std::getenv("LSD_RECIPROCAL")) config.odometry.constraints.max_reciprocal = std::stod(v);
    if (const char* v = std::getenv("LSD_CONSTRAINT_LEVEL")) config.odometry.constraints.tracker.first_level = std::stoi(v);
    Session session(config);
    if (argc > 6) session.depth_view = std::atoi(argv[6]);
    const char* export_depth = std::getenv("LSD_EXPORT_DEPTH");
    session.keep_priors = export_depth != nullptr;
    const int frames = std::stoi(argv[3]);
    const auto start = Clock::now();
    for (int i = 0; i < frames && !session.eof; ++i) session.advance();
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    int width = 0, height = 0;
    for (int pass = 0; pass < 3; ++pass) {  // first passes settle window sizes
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      draw_sidebar(session);
      draw_video(session);
      draw_depth(session);
      if (session.odometry) session.graph.draw(*session.odometry, &session.show_graph);
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
    std::ofstream out(argv[4], std::ios::binary);
    out << "P6\n" << width << ' ' << height << "\n255\n";
    for (int y = height - 1; y >= 0; --y)
      out.write(reinterpret_cast<const char*>(pixels.data() + static_cast<std::size_t>(y) * width * 3), width * 3);
    if (!session.error.empty()) std::cout << "error: " << session.error << '\n';
    if (!session.depth_error.empty()) std::cout << "depth: " << session.depth_error << '\n';
    if (!session.odometry) return 1;
    std::cout << session.frames << " frames in " << seconds << " s, track " << session.track_ms_total / session.frames
              << " ms/frame, map " << session.mapping_ms_total / session.frames << " ms/frame, "
              << session.odometry->keyframes().size() << " keyframes, " << session.lost_frames
              << " lost, " << session.rejected_keyframes << " rejected, network " << session.depth_ms << " ms\n";
    {
      int accepted = 0, fallback = 0;
      for (const auto& e : session.odometry->graph().edges()) (e.fallback ? fallback : accepted)++;
      std::cout << "graph: " << accepted << " alignment edges, " << fallback << " fallback, " << session.graph_ms_total
                << " ms total; reciprocal errors";
      std::vector<double> errors = session.reciprocal_errors;
      std::sort(errors.begin(), errors.end());
      for (const double q : {0.1, 0.5, 0.9})
        if (!errors.empty()) std::cout << ' ' << errors[static_cast<std::size_t>(q * (errors.size() - 1))];
      std::cout << " (10/50/90 %, " << errors.size() << " pairs)\n";
      int loops = 0;
      for (const auto& e : session.odometry->graph().edges()) loops += e.loop;
      std::cout << "loops: " << session.odometry->loop_candidates_total() << " candidates, " << loops
                << " accepted edges" << (session.loop_note.empty() ? "" : " (" + session.loop_note + ")") << '\n';
      if (const char* path = std::getenv("LSD_PAIR_DETAILS")) {
        std::ofstream pairs(path);
        pairs << "i,j,fwd_rot_deg,fwd_trans_rel,bwd_rot_deg,bwd_trans_rel\n";
        for (const auto& p : session.pair_details)
          pairs << p[0] << ',' << p[1] << ',' << p[2] << ',' << p[3] << ',' << p[4] << ',' << p[5] << '\n';
      }
      if (const char* path = std::getenv("LSD_EDGES")) {
        // Accepted edges with both keyframes' frames: S_ji as (s, quaternion, t).
        std::ofstream out(path);
        out << "i,j,frame_i,frame_j,fallback,loop,reciprocal,s,qw,qx,qy,qz,tx,ty,tz,info_trace\n" << std::setprecision(10);
        const auto& kfs = session.odometry->keyframes();
        for (const auto& e : session.odometry->graph().edges()) {
          const Eigen::Quaterniond q(e.S_ji.R);
          out << e.i << ',' << e.j << ',' << kfs[e.i]->frame_index << ',' << kfs[e.j]->frame_index << ','
              << e.fallback << ',' << e.loop << ',' << e.reciprocal << ',' << e.S_ji.s << ',' << q.w() << ',' << q.x()
              << ',' << q.y() << ',' << q.z() << ',' << e.S_ji.t.x() << ',' << e.S_ji.t.y() << ',' << e.S_ji.t.z()
              << ',' << e.information.trace() << '\n';
        }
      }
      if (const char* path = std::getenv("LSD_PAIRS")) {
        std::ofstream pairs(path);
        pairs << "error,rot_deg,trans_rel,scale_pct\n";
        for (std::size_t k = 0; k < session.reciprocal_errors.size(); ++k)
          pairs << session.reciprocal_errors[k] << ',' << session.reciprocal_motion[k][0] << ','
                << session.reciprocal_motion[k][1] << ',' << session.reciprocal_motion[k][2] << '\n';
      }
    }
    if (export_depth && session.rejected_keyframes == 0) {
      std::ofstream csv(export_depth);
      csv << "keyframe,frame_index,x,y,idepth,sigma,prior_idepth\n";
      const float factor = static_cast<float>(1 << session.levels_down);
      for (const auto& kf : session.odometry->keyframes()) {
        const auto* prior = kf->id < static_cast<int>(session.priors.size()) ? &session.priors[kf->id] : nullptr;
        for (int y = 0; y < kf->depth.idepth.height; ++y)
          for (int x = 0; x < kf->depth.idepth.width; ++x) {
            if (!kf->depth.valid(x, y)) continue;
            const lsd::Vec2 d = session.undistorter.source(x, y);
            const float p = prior && prior->at(x, y) > 0 ? static_cast<float>(kf->scale_correction) / prior->at(x, y) : 0;
            csv << kf->id << ',' << kf->frame_index << ',' << (d.x() + 0.5) * factor - 0.5 << ','
                << (d.y() + 0.5) * factor - 0.5 << ',' << kf->depth.idepth.at(x, y) << ','
                << std::sqrt(kf->depth.variance.at(x, y)) << ',' << p << '\n';
          }
      }
    }
    if (argc > 5) {
      std::ofstream csv(argv[5]);
      csv << "frame_index,timestamp_ns,segment,keyframe,cx,cy,cz,qw,qx,qy,qz\n" << std::setprecision(10);
      for (const auto& t : session.odometry->trajectory()) {
        const lsd::Sim3 world_from_camera = t.pose.inverse();
        const Eigen::Quaterniond q(world_from_camera.R);
        csv << t.frame_index << ',' << t.timestamp_ns << ",0," << (t.is_keyframe ? 1 : 0) << ','
            << world_from_camera.t.x() << ',' << world_from_camera.t.y() << ',' << world_from_camera.t.z() << ','
            << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << '\n';
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
