#!/bin/bash
# Run the ORB-SLAM3 monocular baseline on a video clip (viewer off; set
# ORBSLAM3_VIEWER=1 to watch it live, ORBSLAM3_HOLD=SECONDS to keep the final map up).
# Build first: scripts/orbslam3/setup.sh
#
# usage: scripts/run_orbslam3.sh VIDEO START_SECONDS DURATION_SECONDS OUT_DIR [WIDTH=640] [HFOV_DEG=60] [FEATURES=1000]
#
# Frames are decoded by ffmpeg, scaled to WIDTH (aspect preserved), converted to
# 8-bit grayscale and streamed to the driver; frame i gets timestamp
# (first_frame + i) / fps, so frame indices match the source video.
# The camera is a pinhole guess: fx = fy = (W/2)/tan(HFOV/2), centred principal
# point, zero distortion (same assumption as the native VO and Python solver).
# Outputs in OUT_DIR:
#   camera.yaml                 settings used
#   orbslam3-live.csv           per-frame poses as tracked (segment = relocalization/new map count)
#   orbslam3-final.csv          largest map after BA/loop closure (all tracked frames)
#   orbslam3-final-euroc.txt, orbslam3-keyframes-euroc.txt  native ORB-SLAM3 outputs
#   run.log                     ORB-SLAM3 console output and summary line
# CSV columns: frame_index,timestamp_ns,segment,keyframe,cx,cy,cz,qw,qx,qy,qz
# (camera centre and camera->world quaternion; see native_workbench/include/slam_native/track_io.hpp).
set -euo pipefail
[ $# -ge 4 ] || { sed -n '5,7p' "$0"; exit 2; }
VIDEO=$1; START=$2; DURATION=$3; OUT=$4
WIDTH=${5:-640}; HFOV=${6:-60}; FEATURES=${7:-1000}
REPO=$(cd "$(dirname "$0")/.." && pwd)
BIN=$REPO/third_party/orbslam3-deps/bin/orbslam3_run
VOCAB=$REPO/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt
[ -x "$BIN" ] && [ -f "$VOCAB" ] || { echo "Build first: scripts/orbslam3/setup.sh" >&2; exit 1; }
mkdir -p "$OUT"
read -r SRC_W SRC_H RATE < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate \
  -of csv=p=0 "$VIDEO" | tr ',' ' ')
FPS=$(python3 -c "print(eval('$RATE'))")
HEIGHT=$(( (WIDTH * SRC_H / SRC_W) / 2 * 2 ))
FIRST=$(python3 -c "print(round($START * $FPS))")
FX=$(python3 -c "import math; print(0.5 * $WIDTH / math.tan(math.radians($HFOV) / 2))")
cat > "$OUT/camera.yaml" <<YAML
%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: $FX
Camera1.fy: $FX
Camera1.cx: $(python3 -c "print(($WIDTH - 1) / 2)")
Camera1.cy: $(python3 -c "print(($HEIGHT - 1) / 2)")
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera.width: $WIDTH
Camera.height: $HEIGHT
Camera.fps: $(python3 -c "print(round($FPS))")
Camera.RGB: 1
ORBextractor.nFeatures: $FEATURES
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
YAML
echo "clip: $VIDEO from ${START}s for ${DURATION}s -> ${WIDTH}x${HEIGHT} gray @ $FPS fps, first frame $FIRST, fx=$FX"
ffmpeg -v error -ss "$START" -i "$VIDEO" -t "$DURATION" -an \
  -vf "scale=${WIDTH}:${HEIGHT}:flags=area,format=gray" -f rawvideo -pix_fmt gray - |
  "$BIN" "$VOCAB" "$OUT/camera.yaml" "$WIDTH" "$HEIGHT" "$FPS" "$FIRST" "$OUT/orbslam3" > "$OUT/run.log" 2>&1
tail -1 "$OUT/run.log"
