cmake_minimum_required(VERSION 3.24)

project(slam_native_workbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(SQLite3 REQUIRED)
include(CTest)
option(SLAM_NATIVE_BUILD_APP "Build the CUDA/ImGui workbench" ON)

add_library(slam-native-store STATIC
  src/feature_store.cpp
  src/flow_tracker.cpp
)
target_include_directories(slam-native-store PUBLIC include)
target_link_libraries(slam-native-store PUBLIC SQLite3::SQLite3)
target_compile_options(slam-native-store PRIVATE -Wall -Wextra -Wpedantic)

# Visual odometry (CPU, Eigen). Eigen is header-only and pinned like ImGui;
# SOURCE_SUBDIR points at a missing directory so only the headers are fetched.
include(FetchContent)
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR headers-only
)
FetchContent_MakeAvailable(eigen)
add_library(slam-native-vo STATIC
  src/visual_odometry.cpp
  src/vo_geometry.cpp
  src/place_index.cpp
  src/track_io.cpp
  src/focal_estimation.cpp
)
find_package(Threads REQUIRED)
target_include_directories(slam-native-vo PUBLIC include PRIVATE src)
target_link_libraries(slam-native-vo PUBLIC Threads::Threads)
target_include_directories(slam-native-vo SYSTEM PRIVATE ${eigen_SOURCE_DIR})
target_compile_options(slam-native-vo PRIVATE -Wall -Wextra -Wpedantic)
# Eigen at -O0 is ~20x slower; keep the solver usable in Debug builds.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(slam-native-vo PRIVATE -O2)
endif()
add_executable(slam-native-focal src/focal_main.cpp)
target_link_libraries(slam-native-focal PRIVATE slam-native-vo)
target_compile_options(slam-native-focal PRIVATE -Wall -Wextra -Wpedantic)

# Direct monocular SLAM after LSD-SLAM (LSD_SLAM.md): CPU core, shares vo_geometry.
add_library(slam-native-lsd STATIC
  src/lsd/image.cpp
  src/lsd/se3_tracker.cpp
  src/lsd/odometry.cpp
  src/lsd/semi_dense_depth.cpp
  src/lsd/undistort.cpp
  src/lsd/sim3_tracker.cpp
  src/lsd/keyframe_graph.cpp
  src/lsd/place_recognition.cpp
  src/pose_graph.cpp
)
target_include_directories(slam-native-lsd PUBLIC include src)
target_include_directories(slam-native-lsd SYSTEM PUBLIC ${eigen_SOURCE_DIR})
target_link_libraries(slam-native-lsd PUBLIC slam-native-vo)
target_compile_options(slam-native-lsd PRIVATE -Wall -Wextra -Wpedantic -O2)

add_library(slam-native-recents STATIC src/recent_sessions.cpp)
target_include_directories(slam-native-recents PUBLIC include)
target_compile_options(slam-native-recents PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_TESTING)
  add_executable(slam-native-feature-store-test tests/test_feature_store.cpp)
  target_link_libraries(slam-native-feature-store-test PRIVATE slam-native-store)
  target_compile_options(slam-native-feature-store-test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native_feature_store COMMAND slam-native-feature-store-test)
  add_executable(slam-native-vo-test tests/test_visual_odometry.cpp)
  target_link_libraries(slam-native-vo-test PRIVATE slam-native-vo)
  target_include_directories(slam-native-vo-test PRIVATE src)
  target_include_directories(slam-native-vo-test SYSTEM PRIVATE ${eigen_SOURCE_DIR})
  target_compile_options(slam-native-vo-test PRIVATE -Wall -Wextra -Wpedantic)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(slam-native-vo-test PRIVATE -O2)
  endif()
  add_test(NAME native_visual_odometry COMMAND slam-native-vo-test)
  add_executable(slam-native-focal-test tests/test_focal_estimation.cpp)
  target_link_libraries(slam-native-focal-test PRIVATE slam-native-vo)
  target_include_directories(slam-native-focal-test SYSTEM PRIVATE ${eigen_SOURCE_DIR})
  target_compile_options(slam-native-focal-test PRIVATE -Wall -Wextra -Wpedantic -O2)
  add_test(NAME native_focal_estimation COMMAND slam-native-focal-test)
  add_executable(slam-native-recents-test tests/test_recent_sessions.cpp)
  target_link_libraries(slam-native-recents-test PRIVATE slam-native-recents)
  target_compile_options(slam-native-recents-test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native_recent_sessions COMMAND slam-native-recents-test)
  add_executable(slam-native-lsd-tracker-test tests/test_lsd_tracker.cpp)
  target_link_libraries(slam-native-lsd-tracker-test PRIVATE slam-native-lsd)
  target_compile_options(slam-native-lsd-tracker-test PRIVATE -Wall -Wextra -Wpedantic -O2)
  add_test(NAME native_lsd_tracker COMMAND slam-native-lsd-tracker-test)
  add_executable(slam-native-lsd-depth-filter-test tests/test_lsd_depth_filter.cpp)
  target_link_libraries(slam-native-lsd-depth-filter-test PRIVATE slam-native-lsd)
  target_compile_options(slam-native-lsd-depth-filter-test PRIVATE -Wall -Wextra -Wpedantic -O2)
  add_test(NAME native_lsd_depth_filter COMMAND slam-native-lsd-depth-filter-test)
  add_executable(slam-native-lsd-sim3-test tests/test_lsd_sim3.cpp)
  target_link_libraries(slam-native-lsd-sim3-test PRIVATE slam-native-lsd)
  target_compile_options(slam-native-lsd-sim3-test PRIVATE -Wall -Wextra -Wpedantic -O2)
  add_test(NAME native_lsd_sim3 COMMAND slam-native-lsd-sim3-test)
endif()

if(SLAM_NATIVE_BUILD_APP)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()

  find_package(CUDAToolkit REQUIRED)
  find_package(OpenGL REQUIRED)
  find_package(PkgConfig REQUIRED)
  find_package(glfw3 3.3 REQUIRED)

  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavcodec
    libavformat
    libavutil
  )

  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu /usr/local/TensorRT/include)
  find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/TensorRT/lib)
  if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT development files were not found (NvInfer.h, libnvinfer)")
  endif()

  # Dear ImGui has no required runtime and is compiled directly into this binary.
  FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9b
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(imgui)

  add_executable(slam-native-workbench
    src/main.cpp
    src/app.cpp
    src/launcher.cpp
    src/video_thumbnail.cpp
    src/ffmpeg_cuda_decoder.cpp
    src/tensorrt_superpoint.cu
    src/online_tracker.cu
    src/optical_flow.cu
    src/flow_association.cu
    src/cuda_frame_presenter.cu
    src/trajectory_view.cpp
    src/point_cloud_renderer.cpp
    src/voxel_hash.cu
    src/voxel_renderer.cpp
    src/fov_view.cpp
    src/color_sampler.cu
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  )

  target_include_directories(slam-native-workbench PRIVATE
    include
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${TENSORRT_INCLUDE_DIR}
    third_party/nvof
  )

  target_link_libraries(slam-native-workbench PRIVATE
    slam-native-store
    slam-native-vo
    slam-native-recents
    CUDA::cudart
    CUDA::cublas
    CUDA::cuda_driver
    OpenGL::GL
    PkgConfig::FFMPEG
    glfw
    ${CMAKE_DL_LIBS}
    ${TENSORRT_LIBRARY}
  )

  target_compile_options(slam-native-workbench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  )
  set_target_properties(slam-native-workbench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  # The LSD workbench (LSD_SLAM.md): same launcher, decoder and 3D renderer, direct pipeline.
  add_executable(slam-native-lsd-workbench
    src/lsd_main.cpp
    src/lsd_app.cpp
    src/lsd_graph_view.cpp
    src/launcher.cpp
    src/video_thumbnail.cpp
    src/ffmpeg_cuda_decoder.cpp
    src/cuda_frame_presenter.cu
    src/point_cloud_renderer.cpp
    src/depth_estimator.cu
    src/tensorrt_superpoint.cu
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  )
  target_include_directories(slam-native-lsd-workbench PRIVATE
    include
    src
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${TENSORRT_INCLUDE_DIR}
  )
  target_link_libraries(slam-native-lsd-workbench PRIVATE
    slam-native-lsd
    slam-native-store
    slam-native-recents
    CUDA::cudart
    CUDA::cublas
    CUDA::cuda_driver
    OpenGL::GL
    PkgConfig::FFMPEG
    glfw
    ${CMAKE_DL_LIBS}
    ${TENSORRT_LIBRARY}
  )
  target_compile_options(slam-native-lsd-workbench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
  # The tracker runs per pixel on the UI thread: optimised even in Debug builds.
  set_source_files_properties(src/lsd_app.cpp src/lsd_graph_view.cpp TARGET_DIRECTORY slam-native-lsd-workbench
    PROPERTIES COMPILE_OPTIONS "-O2")

  if(BUILD_TESTING)
    # Optional desktop/GPU transport integration test, using the real Session.
    get_target_property(transport_sources slam-native-workbench SOURCES)
    list(REMOVE_ITEM transport_sources src/main.cpp src/app.cpp)
    add_executable(slam-native-transport-test tests/test_transport.cpp ${transport_sources})
    get_target_property(transport_includes slam-native-workbench INCLUDE_DIRECTORIES)
    get_target_property(transport_libraries slam-native-workbench LINK_LIBRARIES)
    target_include_directories(slam-native-transport-test PRIVATE ${transport_includes})
    target_link_libraries(slam-native-transport-test PRIVATE ${transport_libraries})
    # GPU tests (the app build already requires a CUDA device).
    add_executable(slam-native-snapshot tests/snapshot_app.cpp ${transport_sources})
    target_include_directories(slam-native-snapshot PRIVATE ${transport_includes})
    target_link_libraries(slam-native-snapshot PRIVATE ${transport_libraries})
    get_target_property(lsd_sources slam-native-lsd-workbench SOURCES)
    list(REMOVE_ITEM lsd_sources src/lsd_main.cpp src/lsd_app.cpp)
    add_executable(slam-native-lsd-snapshot tests/snapshot_lsd.cpp ${lsd_sources})
    get_target_property(lsd_includes slam-native-lsd-workbench INCLUDE_DIRECTORIES)
    get_target_property(lsd_libraries slam-native-lsd-workbench LINK_LIBRARIES)
    target_include_directories(slam-native-lsd-snapshot PRIVATE ${lsd_includes})
    target_link_libraries(slam-native-lsd-snapshot PRIVATE ${lsd_libraries})
    target_compile_options(slam-native-lsd-snapshot PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)
    add_executable(slam-native-snapshot-launcher tests/snapshot_launcher.cpp ${transport_sources})
    target_include_directories(slam-native-snapshot-launcher PRIVATE ${transport_includes})
    target_link_libraries(slam-native-snapshot-launcher PRIVATE ${transport_libraries})
    add_executable(slam-native-gpu-tracker-test tests/test_flow_tracker.cpp src/flow_association.cu)
    target_link_libraries(slam-native-gpu-tracker-test PRIVATE slam-native-store CUDA::cudart)
    add_test(NAME native_gpu_flow_tracker COMMAND slam-native-gpu-tracker-test)
    add_executable(slam-native-voxel-hash-test tests/test_voxel_hash.cpp src/voxel_hash.cu)
    target_include_directories(slam-native-voxel-hash-test PRIVATE include)
    target_link_libraries(slam-native-voxel-hash-test PRIVATE CUDA::cudart)
    add_test(NAME native_voxel_hash COMMAND slam-native-voxel-hash-test)
    add_executable(slam-native-live-depth-test tests/test_live_depth.cpp src/live_depth.cu)
    target_include_directories(slam-native-live-depth-test PRIVATE include)
    target_link_libraries(slam-native-live-depth-test PRIVATE slam-native-store slam-native-vo CUDA::cudart)
    add_test(NAME native_live_depth COMMAND slam-native-live-depth-test)
    add_executable(slam-native-live-depth-bench
      tests/bench_live_depth.cpp
      src/live_depth.cu
      src/ffmpeg_cuda_decoder.cpp
      src/tensorrt_superpoint.cu
      src/optical_flow.cu
      src/flow_association.cu
      src/color_sampler.cu
    )
    target_include_directories(slam-native-live-depth-bench PRIVATE include third_party/nvof ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(slam-native-live-depth-bench PRIVATE slam-native-store slam-native-vo PkgConfig::FFMPEG
      CUDA::cudart CUDA::cuda_driver ${TENSORRT_LIBRARY} ${CMAKE_DL_LIBS})
    add_executable(slam-native-display-test
      tests/test_display.cpp
      src/ffmpeg_cuda_decoder.cpp
      src/cuda_frame_presenter.cu
    )
    target_include_directories(slam-native-display-test PRIVATE include)
    target_link_libraries(slam-native-display-test PRIVATE PkgConfig::FFMPEG
      CUDA::cudart CUDA::cuda_driver OpenGL::GL glfw)
    add_executable(slam-native-hybrid-test
      tests/test_hybrid_pipeline.cpp
      src/ffmpeg_cuda_decoder.cpp
      src/tensorrt_superpoint.cu
      src/optical_flow.cu
      src/flow_association.cu
    )
    target_include_directories(slam-native-hybrid-test PRIVATE include third_party/nvof ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(slam-native-hybrid-test PRIVATE slam-native-store PkgConfig::FFMPEG
      CUDA::cudart CUDA::cuda_driver ${TENSORRT_LIBRARY} ${CMAKE_DL_LIBS})
    add_executable(slam-native-tracking-bench
      tests/bench_tracking.cpp
      src/ffmpeg_cuda_decoder.cpp
      src/tensorrt_superpoint.cu
      src/optical_flow.cu
      src/flow_association.cu
      src/color_sampler.cu
              )
    target_include_directories(slam-native-tracking-bench PRIVATE include third_party/nvof ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(slam-native-tracking-bench PRIVATE slam-native-store slam-native-vo PkgConfig::FFMPEG
      CUDA::cudart CUDA::cuda_driver ${TENSORRT_LIBRARY} ${CMAKE_DL_LIBS})
    add_executable(slam-native-optical-flow-test
      tests/test_optical_flow.cpp
      src/ffmpeg_cuda_decoder.cpp
      src/optical_flow.cu
    )
    target_include_directories(slam-native-optical-flow-test PRIVATE include third_party/nvof)
    target_link_libraries(slam-native-optical-flow-test PRIVATE slam-native-store PkgConfig::FFMPEG
      CUDA::cudart CUDA::cuda_driver ${CMAKE_DL_LIBS})
    add_test(NAME native_optical_flow_synthetic COMMAND slam-native-optical-flow-test)
    add_executable(slam-native-video-seek-test
      tests/test_video_seek.cpp
      src/ffmpeg_cuda_decoder.cpp
    )
    target_include_directories(slam-native-video-seek-test PRIVATE include)
    target_link_libraries(slam-native-video-seek-test PRIVATE PkgConfig::FFMPEG CUDA::cuda_driver)
    target_compile_options(slam-native-video-seek-test PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()
