#!/usr/bin/env bash
set -euo pipefail

# Native workbench build dependencies for this Ubuntu 26.04 x86_64 host.
# CUDA 13.4 and TensorRT 11.3 are a matched pair from NVIDIA's network repo.
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "26.04" || "$(uname -m)" != "x86_64" ]]; then
  printf 'This installer is for Ubuntu 26.04 x86_64 only.\n' >&2
  exit 2
fi

# Authenticate in the user's terminal once. The toolkit package does not
# install or upgrade the NVIDIA driver.
sudo -v

install_dir="$(mktemp -d)"
trap 'rm -rf "$install_dir"' EXIT
curl -fL \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb \
  -o "$install_dir/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i "$install_dir/cuda-keyring_1.1-1_all.deb"
sudo apt-get update

# Install the C++ compiler, FFmpeg/graphics/database headers, the complete
# CUDA development toolkit, and the TensorRT C++ runtime, headers, and CLI.
# Avoid the tensorrt-dev metapackage: it includes a 1.5 GiB Windows builder
# resource which this Linux-only app does not need.
trt_version='11.3.0.99-1+cuda13.4'
sudo apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build pkg-config gdb \
  libavcodec-dev libavformat-dev libavutil-dev \
  libsqlite3-dev libglfw3-dev libgl-dev \
  cuda-toolkit-13-4 \
  "libnvinfer-dev=$trt_version" \
  "libnvinfer-bin=$trt_version" \
  "libnvonnxparsers-dev=$trt_version"

printf '\nInstalled native build tools:\n'
/usr/local/cuda-13.4/bin/nvcc --version | tail -1
g++ --version | head -1
cmake --version | head -1
dpkg-query -W -f='${Package} ${Version}\n' \
  libnvinfer-dev libnvinfer-bin libnvonnxparsers-dev
printf '\nNext: run the VS Code task "Native: build app (Debug)".\n'
