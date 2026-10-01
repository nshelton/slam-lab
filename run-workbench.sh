#!/bin/bash
# Rebuild (incremental) and launch the native workbench from the repo root.
#
# usage: ./run-workbench.sh [workbench args...]
#   no args        opens the launcher to pick a video
#   --video V      skips the launcher (point cache: pointcache/, automatic)
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
  (cd native_workbench && "$cmake" --build --preset app-debug)
fi

exec native_workbench/build/app-debug/slam-native-workbench "$@"
