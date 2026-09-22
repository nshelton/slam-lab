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

add_library(slam-native-recents STATIC src/recent_sessions.cpp)
target_include_directories(slam-native-recents PUBLIC include)
target_compile_options(slam-native-recents PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_TESTING)
  add_executable(slam-native-flow-tracker-test tests/test_flow_tracker.cpp)
  target_link_libraries(slam-native-flow-tracker-test PRIVATE slam-native-store)
  target_compile_options(slam-native-flow-tracker-test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native_flow_tracker COMMAND slam-native-flow-tracker-test)
  add_executable(slam-native-recents-test tests/test_recent_sessions.cpp)
  target_link_libraries(slam-native-recents-test PRIVATE slam-native-recents)
  target_compile_options(slam-native-recents-test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native_recent_sessions COMMAND slam-native-recents-test)
  add_executable(slam-native-diagnostics-test
    tests/test_descriptor_projection.cpp
    src/descriptor_projection.cpp
  )
  target_include_directories(slam-native-diagnostics-test PRIVATE include)
  target_compile_options(slam-native-diagnostics-test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME native_descriptor_projection COMMAND slam-native-diagnostics-test)
endif()

if(SLAM_NATIVE_BUILD_APP)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()

  include(FetchContent)
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
    src/descriptor_projection.cpp
    src/ffmpeg_cuda_decoder.cpp
    src/tensorrt_superpoint.cu
    src/online_tracker.cu
    src/optical_flow.cu
    src/cuda_frame_presenter.cu
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

  if(BUILD_TESTING)
    add_executable(slam-native-video-seek-test
      tests/test_video_seek.cpp
      src/ffmpeg_cuda_decoder.cpp
    )
    target_include_directories(slam-native-video-seek-test PRIVATE include)
    target_link_libraries(slam-native-video-seek-test PRIVATE PkgConfig::FFMPEG CUDA::cuda_driver)
    target_compile_options(slam-native-video-seek-test PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()
