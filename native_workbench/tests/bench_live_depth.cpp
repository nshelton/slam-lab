// Headless live-depth run on real video (GPU): the pipeline of the tracking
// bench plus LiveDepth, with the depth image dumped every few frames for
// tools/depth_vs_gt.py. Not part of CTest.
// usage: slam-native-live-depth-bench VIDEO ENGINE FRAMES OUT_DIR [KEY VALUE]...
//   mode anchors|measurement   gap N (reference frames back)   dump-every N   skip N
//   hfov, k1, fx fy cx cy      (camera, as the tracking bench)
//   patch-length, patch-width, search-radius, search-step, max-epipolar,
//   min-gradient, max-residual, disparity-floor, image-noise
// Dump: OUT_DIR/depth_<frame>.bin = int32 width, height; uint64 frame index;
// float32 inverse depth [w*h]; float32 variance [w*h] (inf: no estimate).
#include "slam_native/color_sampler.hpp"
#include "slam_native/flow_tracker.hpp"
#include "slam_native/live_depth.hpp"
#include "slam_native/optical_flow.hpp"
#include "slam_native/superpoint.hpp"
#include "slam_native/video_decoder.hpp"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

using namespace slam_native;

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: slam-native-live-depth-bench VIDEO ENGINE FRAMES OUT_DIR [KEY VALUE]...\n";
    return 2;
  }
  try {
    const int frames = std::stoi(argv[3]);
    const std::filesystem::path out = argv[4];
    std::filesystem::create_directories(out);
    VisualOdometryConfig vo_config;
    LiveDepthConfig config;
    int skip = 0, dump_every = 20;
    for (int a = 5; a + 1 < argc; a += 2) {
      const std::string key = argv[a], text = argv[a + 1];
      if (key == "mode") {
        if (text == "anchors") config.mode = LiveDepthMode::anchors;
        else if (text == "measurement") config.mode = LiveDepthMode::measurement;
        else throw std::invalid_argument("unknown mode " + text);
        continue;
      }
      const float value = std::stof(text);
      if (key == "skip") skip = static_cast<int>(value);
      else if (key == "dump-every") dump_every = static_cast<int>(value);
      else if (key == "hfov") vo_config.horizontal_fov_degrees = value;
      else if (key == "k1") vo_config.distortion_k1 = value;
      else if (key == "fx" || key == "fy" || key == "cx" || key == "cy") {
        if (!vo_config.intrinsics) vo_config.intrinsics = CameraIntrinsics{};
        (key == "fx" ? vo_config.intrinsics->fx : key == "fy" ? vo_config.intrinsics->fy :
         key == "cx" ? vo_config.intrinsics->cx : vo_config.intrinsics->cy) = value;
      }
      else if (key == "gap") config.reference_gap = static_cast<int>(value);
      else if (key == "patch-length") config.patch_half_length = static_cast<int>(value);
      else if (key == "patch-width") config.patch_half_width = static_cast<int>(value);
      else if (key == "search-radius") config.search_radius_px = value;
      else if (key == "search-step") config.search_step_px = value;
      else if (key == "max-epipolar") config.max_epipolar_distance_px = value;
      else if (key == "min-gradient") config.min_gradient = value;
      else if (key == "max-residual") config.max_residual = value;
      else if (key == "disparity-floor") config.disparity_floor_px = value;
      else if (key == "image-noise") config.image_noise = value;
      else throw std::invalid_argument("unknown option " + key);
    }
    auto decoder = make_ffmpeg_cuda_decoder();
    decoder->open(argv[1]);
    auto superpoint = make_tensorrt_superpoint(argv[2], SuperPointConfig{});
    OpticalFlow optical_flow{OpticalFlowConfig{}};
    FlowTracker tracker{FlowTrackerConfig{}};
    VisualOdometry odometry(vo_config);
    ColorSampler color_sampler;
    LiveDepth depth(config);

    LiveDepthStats total;
    int images = 0, dumps = 0, posed = 0, processed = 0;
    double ms = 0;
    for (int index = 0; index < skip; ++index) {
      GpuFrame image;
      if (!decoder->next(image)) break;
    }
    for (int index = 0; index < frames; ++index) {
      GpuFrame image;
      if (!decoder->next(image)) break;
      auto features = superpoint->infer(image);
      std::optional<DeviceFlowField> field;
      if (optical_flow.has_reference()) field = optical_flow.compute(image);
      else optical_flow.remember(image);
      const DeviceDetections device = superpoint->device_detections();
      tracker.associate(features, field ? &*field : nullptr, &device);
      const auto tracked = tracked_frame_from(features, color_sampler.sample(image, features.keypoints));
      const auto pose = odometry.process(tracked);
      ++processed;
      posed += pose.has_pose && !pose.predicted;
      // Only tracked poses: a coasted one is a prediction.
      const bool usable = pose.has_pose && !pose.predicted;
      depth.process(image, field ? &*field : nullptr, live_depth_camera(vo_config, image.width, image.height),
                    usable ? &pose.pose : nullptr, pose.segment, depth_anchors(tracked, odometry, pose));
      if (!depth.valid()) continue;
      const auto& s = depth.stats();
      ++images;
      ms += s.ms;
      total.pixels += s.pixels;
      total.measured += s.measured;
      total.outside += s.outside;
      total.off_line += s.off_line;
      total.weak_gradient += s.weak_gradient;
      total.at_search_edge += s.at_search_edge;
      total.poor_match += s.poor_match;
      total.behind += s.behind;
      if (dump_every > 0 && index % dump_every == 0) {  // by frame, so runs in different modes dump the same ones
        const auto host = depth.download();
        std::ofstream file(out / ("depth_" + std::to_string(host.frame_index) + ".bin"), std::ios::binary);
        const std::int32_t size[2] = {host.width, host.height};
        file.write(reinterpret_cast<const char*>(size), sizeof(size));
        file.write(reinterpret_cast<const char*>(&host.frame_index), sizeof(host.frame_index));
        file.write(reinterpret_cast<const char*>(host.inverse_depth.data()),
                   static_cast<std::streamsize>(host.inverse_depth.size() * sizeof(float)));
        file.write(reinterpret_cast<const char*>(host.variance.data()),
                   static_cast<std::streamsize>(host.variance.size() * sizeof(float)));
        ++dumps;
      }
    }
    const auto share = [&](std::size_t n) { return 100.0 * double(n) / double(std::max<std::size_t>(1, total.pixels)); };
    std::printf("frames %d, tracked poses %d, depth images %d (%d dumped), %.2f ms per image\n", processed, posed,
                images, dumps, ms / std::max(1, images));
    std::printf("pixels: measured %.1f%% | outside %.1f%% | off the epipolar line %.1f%% | weak gradient %.1f%% | "
                "search edge %.1f%% | poor match %.1f%% | behind %.1f%%\n",
                share(total.measured), share(total.outside), share(total.off_line), share(total.weak_gradient),
                share(total.at_search_edge), share(total.poor_match), share(total.behind));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
