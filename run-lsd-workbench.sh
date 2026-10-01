#!/bin/bash
# Rebuild (incremental) and launch the LSD workbench (native_workbench/LSD_SLAM.md).
#
# usage: ./run-lsd-workbench.sh [args...]
#   no args        opens the launcher to pick a video
#   --video V      skips the launcher
#
# Set NO_BUILD=1 to skip the rebuild step.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cmake="$root/.venv-cuda/bin/cmake"
cd "$root"

if [[ "${NO_BUILD:-0}" != 1 ]]; then
  if [[ ! -f native_workbench/build/app-debug/build.ninja ]]; then
    (cd native_workbench && "$cmake" --preset app-debug)
  fi
  (cd native_workbench && "$cmake" --build --preset app-debug --target slam-native-lsd-workbench)
fi

exec native_workbench/build/app-debug/slam-native-lsd-workbench "$@"
