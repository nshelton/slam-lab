cmake_minimum_required(VERSION 3.24)

project(slam_native_workbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(SQLite3 REQUIRED)
option(SLAM_NATIVE_BUILD_APP "Build the CUDA/ImGui workbench" ON)

add_library(slam-native-store STATIC
  src/feature_store.cpp
)
target_include_directories(slam-native-store PUBLIC include)
target_link_libraries(slam-native-store PUBLIC SQLite3::SQLite3)
target_compile_options(slam-native-store PRIVATE -Wall -Wextra -Wpedantic)

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
    src/ffmpeg_cuda_decoder.cpp
    src/tensorrt_superpoint.cu
    src/cuda_frame_presenter.cu
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  )

  target_include_directories(slam-native-workbench PRIVATE
    include
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${TENSORRT_INCLUDE_DIR}
  )

  target_link_libraries(slam-native-workbench PRIVATE
    slam-native-store
    CUDA::cudart
    OpenGL::GL
    PkgConfig::FFMPEG
    glfw
    ${TENSORRT_LIBRARY}
  )

  target_compile_options(slam-native-workbench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  )
  set_target_properties(slam-native-workbench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()
