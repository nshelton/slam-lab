#!/bin/bash
# User-local (no sudo) build of the ORB-SLAM3 monocular baseline, as a separate
# program (nothing is linked into the native workbench). Layout under
# <repo>/third_party/ (git-ignored):
#   ORB_SLAM3/          ORB_SLAM3_U24 fork (ORB-SLAM3 v1.0 patched for Ubuntu 24.04+),
#                       github.com/vijaysaini-ra/ORB_SLAM3_U24 @ c500153c, GPLv3
#   orbslam3-deps/sysroot/  headers + dev symlinks extracted from Ubuntu .debs
#                       (Boost serialization, OpenSSL, libepoxy, EGL/GLVND); the
#                       runtime libraries are the system's
#   orbslam3-deps/prefix/   OpenCV 4.10.0 (minimal modules), Eigen 3.4.0, Pangolin v0.9.4
#   orbslam3-deps/bin/orbslam3_run  baseline driver built from orbslam3_run.cc
# Safe to rerun: finished steps are skipped.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ROOT=${ORBSLAM3_DEPS:-$REPO/third_party/orbslam3-deps}
O=${ORBSLAM3_SOURCE:-$REPO/third_party/ORB_SLAM3}
N=${NINJA:-$REPO/.venv-cuda/bin/ninja}
JOBS=$(nproc)
mkdir -p "$ROOT"/{debs,sysroot,prefix,bin}
cd "$ROOT"
P=$ROOT/prefix; SR=$ROOT/sysroot/usr; LIB=$SR/lib/x86_64-linux-gnu

if [ ! -f sysroot/.done ]; then
  (cd debs && apt-get download libboost1.90-dev libboost-serialization1.90-dev libboost-serialization1.90.0 \
     libssl-dev libepoxy-dev libegl-dev libopengl-dev libglx-dev libgl-dev)
  for d in debs/*.deb; do dpkg -x "$d" sysroot; done
  # Point the dev symlinks at the system runtime libraries.
  for pair in libepoxy.so:libepoxy.so.0 libssl.so:libssl.so.3 libcrypto.so:libcrypto.so.3 \
              libEGL.so:libEGL.so.1 libOpenGL.so:libOpenGL.so.0 libGLX.so:libGLX.so.0 libGL.so:libGL.so.1; do
    ln -sf "/usr/lib/x86_64-linux-gnu/${pair#*:}" "$LIB/${pair%%:*}"
  done
  touch sysroot/.done
fi

[ -d opencv ] || git clone -q --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git opencv
[ -d pangolin ] || git clone -q --depth 1 --branch v0.9.4 https://github.com/stevenlovegrove/Pangolin.git pangolin
if [ ! -d "$O" ]; then
  git clone -q https://github.com/vijaysaini-ra/ORB_SLAM3_U24.git "$O"
  git -C "$O" checkout -q c500153cec1825150e7abf07a8291da33499ecd0
fi
EIGEN_SRC=${EIGEN_SRC:-$REPO/native_workbench/build/app-debug/_deps/eigen-src}
[ -d "$EIGEN_SRC" ] || { git clone -q --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git "$ROOT/eigen"; EIGEN_SRC=$ROOT/eigen; }

COMMON=(-G Ninja -DCMAKE_MAKE_PROGRAM="$N" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P"
        -DCMAKE_PREFIX_PATH="$P;$SR" -DCMAKE_INCLUDE_PATH="$SR/include" -DCMAKE_LIBRARY_PATH="$LIB")
if [ ! -f "$P/share/eigen3/cmake/Eigen3Config.cmake" ]; then
  cmake -S "$EIGEN_SRC" -B build-eigen "${COMMON[@]}" -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF
  "$N" -C build-eigen install
fi
if [ ! -f "$P/lib/libopencv_core.so" ]; then
  cmake -S opencv -B build-opencv "${COMMON[@]}" -DBUILD_LIST=core,imgproc,imgcodecs,features2d,calib3d,flann,highgui \
    -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_OPENGL=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_V4L=OFF \
    -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_TBB=OFF -DWITH_OPENEXR=OFF -DWITH_TIFF=OFF \
    -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DBUILD_PNG=ON -DBUILD_JPEG=ON -DBUILD_ZLIB=ON \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF
  "$N" -C build-opencv -j "$JOBS" install
fi
if [ ! -d "$P/lib/cmake/Pangolin" ]; then
  cmake -S pangolin -B build-pangolin "${COMMON[@]}" -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
    -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_TESTS=OFF -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_REALSENSE2=OFF \
    -DBUILD_PANGOLIN_OPENNI2=OFF -DBUILD_PANGOLIN_LIBDC1394=OFF -DBUILD_PANGOLIN_V4L=OFF \
    -DBUILD_PANGOLIN_LIBPNG=OFF -DBUILD_PANGOLIN_LIBJPEG=OFF -DBUILD_PANGOLIN_LIBTIFF=OFF \
    -DBUILD_PANGOLIN_LIBOPENEXR=OFF -DBUILD_PANGOLIN_ZSTD=OFF -DBUILD_PANGOLIN_LZ4=OFF
  "$N" -C build-pangolin -j "$JOBS" install
fi

FLAGS=(-DOpenCV_DIR="$P/lib/cmake/opencv4" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
       -DCMAKE_CXX_FLAGS="-I$SR/include -I$SR/include/x86_64-linux-gnu -Wno-deprecated-declarations"
       -DCMAKE_SHARED_LINKER_FLAGS="-L$LIB -Wl,-rpath,$P/lib:$LIB"
       -DCMAKE_EXE_LINKER_FLAGS="-L$LIB -Wl,-rpath,$P/lib:$LIB")
cd "$O"
# Draw the current frame inside the Pangolin viewer (our OpenCV has no GUI backend).
if git apply --check "$HERE/viewer-frame-in-pangolin.patch" 2>/dev/null; then
  git apply "$HERE/viewer-frame-in-pangolin.patch"
  rm -f lib/libORB_SLAM3.so
fi
if [ ! -f lib/libORB_SLAM3.so ]; then
  for lib in DBoW2 g2o; do
    cmake -S Thirdparty/$lib -B Thirdparty/$lib/build "${COMMON[@]}" "${FLAGS[@]}"
    "$N" -C Thirdparty/$lib/build -j "$JOBS"
  done
  cmake -S Thirdparty/Sophus -B Thirdparty/Sophus/build "${COMMON[@]}" "${FLAGS[@]}" -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
  cmake -S . -B build "${COMMON[@]}" "${FLAGS[@]}" -DPangolin_DIR="$P/lib/cmake/Pangolin"
  "$N" -C build -j "$JOBS" ORB_SLAM3
fi
# The fork stores ORBvoc.txt as a Git LFS pointer; the real file is in the tarball.
if [ ! -f Vocabulary/ORBvoc.txt ] || [ "$(stat -c %s Vocabulary/ORBvoc.txt)" -lt 1000000 ]; then
  tar -xf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary
fi
cd "$ROOT"

# Driver (GPLv3 like ORB-SLAM3, which it links).
if [ "$ROOT/bin/orbslam3_run" -ot "$HERE/orbslam3_run.cc" ] || [ ! -x "$ROOT/bin/orbslam3_run" ]; then
c++ -std=c++14 -O2 -march=native -DCOMPILEDWITHC11 -DHAVE_EPOXY -DHAVE_EIGEN "$HERE/orbslam3_run.cc" -o bin/orbslam3_run \
  -I"$O" -I"$O/include" -I"$O/include/CameraModels" -I"$O/Thirdparty/Sophus" -I"$P/include/eigen3" \
  -I"$P/include/opencv4" -I"$P/include" -I"$SR/include" -I"$SR/include/x86_64-linux-gnu" \
  -Wno-deprecated-declarations -Wno-c++17-extensions \
  -L"$O/lib" -L"$O/Thirdparty/DBoW2/lib" -L"$O/Thirdparty/g2o/lib" -L"$P/lib" -L"$LIB" \
  -lORB_SLAM3 -lDBoW2 -lg2o -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_calib3d \
  -lpango_display -lpango_opengl -lpango_windowing -lpango_image -lpango_core \
  -lepoxy -lboost_serialization -lcrypto \
  -Wl,--disable-new-dtags -Wl,-rpath,"$O/lib:$O/Thirdparty/DBoW2/lib:$O/Thirdparty/g2o/lib:$P/lib:$LIB"
fi
echo "ORB-SLAM3 baseline ready: $ROOT/bin/orbslam3_run"
