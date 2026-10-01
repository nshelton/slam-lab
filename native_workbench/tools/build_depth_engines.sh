#!/bin/bash
# Fetch, export and build the TensorRT engines for the live depth presets
# (src/depth_estimator.cu, depth_model_presets()). Each model gets a landscape
# and a portrait engine, <stem>-<W>x<H>.engine, so the network always sees the
# upright frame. Existing engines are kept; delete one to rebuild it.
#
# usage: native_workbench/tools/build_depth_engines.sh   (from the repo root)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
python=.venv-cuda/bin/python
models=native_workbench/models
cache=.slam-cache
mkdir -p "$models" "$cache/models/depth_anything_v2" "$cache/models/metric3d"

fetch() {  # url destination
  [[ -s "$2" ]] || { echo "download $2"; curl -fSL -o "$2.part" "$1" && mv "$2.part" "$2"; }
}

engine() {  # onnx engine [trtexec args...]
  local onnx=$1 out=$2
  shift 2
  [[ -s "$out" ]] && { echo "keep $out"; return; }
  echo "build $out"
  trtexec --onnx="$onnx" --saveEngine="$out.part" "$@" > "${out%.engine}.log" 2>&1 ||
    { echo "trtexec failed, see ${out%.engine}.log"; rm -f "$out.part"; exit 1; }
  mv "$out.part" "$out"
  grep -E "GPU Compute Time: min" "${out%.engine}.log" || true
}

# Depth Anything V2 Small, metric (Hypersim indoor 20 m, VKITTI outdoor 80 m).
[[ -d $cache/Depth-Anything-V2 ]] ||
  git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git "$cache/Depth-Anything-V2"
for variant in hypersim:Hypersim:20 vkitti:VKITTI:80; do
  IFS=: read -r name hub max_depth <<< "$variant"
  checkpoint=$cache/models/depth_anything_v2/depth_anything_v2_metric_${name}_vits.pth
  fetch "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-$hub-Small/resolve/main/depth_anything_v2_metric_${name}_vits.pth" \
    "$checkpoint"
  for size in 518x294 294x518; do
    stem=$models/depth-anything-v2-s-$name-$size
    [[ -s $stem.onnx ]] || "$python" native_workbench/tools/export_depth_anything_onnx.py "$stem.onnx" \
      --checkpoint "$checkpoint" --max-depth "$max_depth" --width "${size%x*}" --height "${size#*x}"
    engine "$stem.onnx" "$stem.engine"
  done
done

# Metric3D v2 ViT-S (community fp16 ONNX, dynamic shape; fixed per engine).
onnx=$cache/models/metric3d/model_fp16.onnx
fetch https://huggingface.co/onnx-community/metric3d-vit-small/resolve/main/onnx/model_fp16.onnx "$onnx"
engine "$onnx" "$models/metric3d-vit-s-1064x616.engine" --shapes=pixel_values:1x3x616x1064
engine "$onnx" "$models/metric3d-vit-s-616x1064.engine" --shapes=pixel_values:1x3x1064x616
