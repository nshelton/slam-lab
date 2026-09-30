# Frame-to-frame tracking survey (2026-09-24)

Scope: GPU-resident frame-to-frame 2D tracking for the native workbench, targeting
long, accurate tracks for downstream SfM/SLAM. Hardware target: one RTX 2080 Ti
(Turing, sm_75, 11 GB, NVOF gen 2, TensorRT), ~1080p60 input processed at
~1024x576. Real-time budget at 60 fps is **16.7 ms/frame for everything**
(decode, SuperPoint, flow, association, display).

The current tracker is described in [ARCHITECTURE.md](ARCHITECTURE.md):
SuperPoint (TensorRT) every frame, NVOF 4x4 flow as a predictor, brute-force
CUDA spatial association, no descriptors, no geometric check, tracks die on the
first missed detection.

Conventions in this document:

- **Measured** = number published by the source on the stated hardware.
- **Extrapolated** = my estimate for the 2080 Ti; treat as a hypothesis to benchmark.
- **Unverified** = claim I could not confirm from a primary source in this session.

---

## 1. Comparison table

| Approach | How it works | Fully GPU? | Published speed (source HW) | 2080 Ti @ 1024x576 (extrapolated) | Track length / accuracy | License | Integration effort (C++/CUDA/TRT) |
|---|---|---|---|---|---|---|---|
| **Current: SuperPoint + NVOF + spatial association** | Detect every frame; flow predicts; nearest detection snaps | Yes | 1.13 ms association (Debug, this repo) | as measured | Fragments on any missed detection; pixel-grid SuperPoint locations; no outlier rejection | SuperPoint weights: Magic Leap **non-commercial** | — |
| **GPU pyramidal KLT (cuVSLAM `sof`)** | GFTT seeds selected per grid cell, pyramidal LK with **NCC check at every pyramid level**; pose-predicted initial guesses via `IFeaturePredictor` | Yes (CUDA kernels in `libs/cuda_modules`) | Mono tracking ~2.7 ms on Jetson Orin AGX at 768x480; stereo ~0.4 ms on RTX 4090 ([paper](https://arxiv.org/abs/2506.04359)) | ~1–3 ms for 500–1000 tracks | Long, sub-pixel tracks; drifts slowly on long tracks; fails on fast motion / low texture | NVIDIA Community License: commercial OK, **NVIDIA hardware only** ([LICENSE](https://github.com/nvidia-isaac/cuVSLAM/blob/main/LICENSE)) | Medium: source is open ([repo](https://github.com/nvidia-isaac/cuVSLAM), `libs/sof`, `libs/cuda_modules/lk_tracker.*`) but embedded in cuVSLAM's image/pyramid types |
| GPU KLT (OpenCV CUDA / VPI / vilib) | Same algorithm, library form | Yes | vilib KLT: sub-ms per 100–250 features on GTX 960M/TX2 ([Nagy et al. IROS'20](https://arxiv.org/abs/2003.13493)); VPI/OpenCV: no published desktop numbers found | ~1–2 ms | Same as above; OpenCV CUDA LK has no NCC/fwd-bwd built in | OpenCV Apache-2.0; VPI proprietary (Jetson-first, x86 availability not verified); vilib MIT-style (unverified) | Low (OpenCV `cuda::SparsePyrLKOpticalFlow`) to Medium |
| **SuperPoint + LightGlue (TensorRT)** | Detect every frame, learned GNN matching prev→cur (or keyframe→cur) | Yes | LightGlue README: 150 FPS @1024 kp, 50 FPS @4096 kp on RTX 3080 (PyTorch, adaptive) ([repo](https://github.com/cvg/LightGlue)); TRT: SP 0.95 ms + LG 2.54 ms for 256 kp on RTX 3080 ([yuefanhao](https://github.com/yuefanhao/SuperPoint-LightGlue-TensorRT)) | LG FP16 TRT ~5–15 ms at 1024–2048 kp (no adaptive depth/width in TRT export, no FlashAttention-2 or FP8 on Turing) | Very robust to large motion/blur, re-finds points; positions still SuperPoint-pixel quality; pairwise chaining still fragments | LightGlue Apache-2.0; SP weights non-commercial; ALIKED BSD-3, DISK Apache-2.0 | Low–Medium: [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) (Apache-2.0, TRT, dynamic shapes, max 3840 kp in TRT per [blog](https://fabio-sim.github.io/blog/accelerating-lightglue-inference-onnx-runtime-tensorrt/)) |
| XFeat / XFeat* + LighterGlue | Tiny CNN (64-d desc); XFeat* semi-dense refinement; LighterGlue ≈3x faster than LightGlue | Yes | >150 FPS single-image VGA GPU; ~1400 FPS batched on RTX 4090 ([repo](https://github.com/verlab/accelerated_features)) | Extractor ~2–4 ms; LighterGlue ~2–5 ms | Less repeatable/precise than SuperPoint/ALIKED; good for speed | **Apache-2.0 code and weights** | Medium: community TRT ports ([XFeatTensorRT](https://github.com/PranavNedunghat/XFeatTensorRT)); no official export |
| EfficientLoFTR (semi-dense) | Detector-free transformer matching, coarse-to-fine | Yes | 35.6 ms FP32 / 27.0 ms mixed, 640x480, RTX 3090 ([paper](https://arxiv.org/abs/2403.04765)) | ~50–70 ms at 1024x576 | Many matches, but not repeatable keypoints → matches don't chain into tracks without quantization/merging | Repo license "Other" (not verified) | High (TRT export of aggregated attention not established) |
| RoMa v2 / DKM (dense) | Dense warp + certainty (DINO features) | Yes | RoMa v2: 30.9 pairs/s, 640x640, batch 8, **H200** ([paper](https://arxiv.org/abs/2511.15706)) | ≫100 ms/pair | Best robustness for wide baselines; offline verification tool, not a tracker | MIT ([romav2](https://github.com/Parskatt/romav2)) | High; offline Python only |
| MASt3R-SLAM matching | 3D pointmap prior + iterative projective matching kernels | Yes | Whole system 15 FPS on RTX 4090; matching kernel ~2 ms ([paper](https://arxiv.org/abs/2412.12392)) | <5 FPS system | Dense, geometry-aware; tied to MASt3R network | CC BY-NC-SA (MASt3R, unverified) | Very high |
| NVOF (hardware flow, current) | Fixed-function block matching, 4x4 grid on Turing (1x1/2x2 need Ampere+), quarter-pel | Yes, no SM usage | 1080p FAST 768 fps / SLOW 225 fps on Turing ([app note](https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-application-note)) | ~1–4 ms | Good predictor; blocky at motion boundaries; supports external hints and fwd/bwd flow; EPE only given as chart | NVIDIA SDK | Already integrated |
| **NeuFlow v2** | Lightweight learned flow (1/16 + 1/8 refinement) | Yes | **15 ms at 1024x436 on RTX 2080**; 20+ FPS at 512x384 on Orin Nano ([paper](https://arxiv.org/abs/2408.10161)) | ~15–20 ms at 1024x576 FP32; less with TRT FP16 | KITTI-15 EPE 4.33 vs RAFT 5.04; better than NVOF at boundaries (not measured head-to-head) | Apache-2.0 ([repo](https://github.com/neufieldrobotics/NeuFlow_v2)) | Medium: PyTorch→ONNX→TRT (no official export) |
| SEA-RAFT | RAFT with fewer iterations, better init | Yes | S: 47.5 ms, M: 70.9 ms at 540x960 on RTX 3090 ([paper](https://arxiv.org/abs/2405.14793)) | ~60–100 ms | High accuracy; too slow for 60 fps | BSD-3 ([repo](https://github.com/princeton-vl/SEA-RAFT)) | Medium–High |
| RAFT / FlowFormer / MemFlow | Iterative all-pairs correlation / transformers | Yes | RAFT 125 ms at 1024x436 on RTX 2080 ([NeuFlow v2 table](https://arxiv.org/abs/2408.10161)) | ≥100 ms | Accurate but offline | BSD/Apache (varies) | High |
| CoTracker3 (online) | Transformer over 16-frame sliding window, joint tracks | Yes | 405 µs per frame per point (online), 209 µs (offline) — hardware not stated ([paper, Table 2](https://arxiv.org/abs/2410.11831)); runs at 256x256-ish internal res | ~0.4 s/frame for 1000 points | Best-in-class occlusion handling and long tracks; low internal resolution → not sub-pixel at 1024x576 | **CC-BY-NC** ([repo](https://github.com/facebookresearch/co-tracker)) | High |
| Track-On / Track-On2 / Track-On-R | Causal transformer with spatial + context memory | Yes | Track-On: 16.8 FPS, ~400 pts, A100, 256x256; Track-On2: ~35 FPS @256 pts, ~20 FPS @1024 pts, A100 FP32 ([Track-On](https://arxiv.org/abs/2501.18487), [Track-On2](https://arxiv.org/abs/2509.19115)) | ~5–10 FPS at 1000 pts | Long-term, handles occlusion; low-res → coarse positions | MIT; DINOv3 backbone has its own gated license | High (no TRT export) |
| TAPNext | Causal ViT+SSM, tracking as masked-token prediction | Yes | 1024 pts: 23 ms on V100, 5.3 ms on H100 ([project page](https://tap-next.github.io/)) | ~25–35 ms for 1024 pts (V100 ≈ 2080 Ti class) | SOTA online accuracy on TAP-Vid; 256x256 input (per paper, unverified here) | Code Apache-2.0 ([tapnet](https://github.com/google-deepmind/tapnet)); weights license unverified | High (JAX primary; PyTorch port in tapnet) |
| LocoTrack / TAPIR / BootsTAPIR | Local 4D correlation / per-frame matching + refinement | Yes | LocoTrack >7000 points/s on RTX 3090 ([paper](https://arxiv.org/abs/2407.15420)) | ~0.15–0.3 s/frame at 1000 pts | Good offline long tracks | Apache-2.0 | High |
| DPVO / DPV-SLAM | Random 3x3-feature patches (96/frame in DPV-SLAM), recurrent update + differentiable BA | Yes (PyTorch + CUDA ext) | DPVO 60 FPS avg on RTX 3090, 4.9 GB; DPV-SLAM 39–50 FPS ([DPVO](https://arxiv.org/abs/2208.04726), [DPV-SLAM](https://arxiv.org/abs/2408.01654)) | ~30–40 FPS | Excellent pose; patches live only inside the optimization window → **not long tracks** | MIT ([repo](https://github.com/princeton-vl/DPVO)) | Very high as a tracker; use as a pose reference instead |
| DROID-SLAM | Dense RAFT-style flow + DBA | Yes | ~20 FPS on EuRoC, ~20 GB ([DPV-SLAM comparison](https://arxiv.org/abs/2408.01654)) | <10 FPS | Dense; no sparse tracks | BSD-3 | Very high |
| MAC-VO | Learned matching + metric covariance, stereo | Yes | 7.5 FPS (12.5 fast) at 640x480 ([repo](https://github.com/MAC-VO/MAC-VO)) | slower | Stereo-only | MIT | N/A for monocular |
| KLTNet (Aug 2026) | Coarse dense flow init + **triplet-patch refinement against fixed reference patch** + anisotropic confidence | Yes | Real-time on Jetson AGX Orin with TRT; exact numbers only in a figure ([paper](https://arxiv.org/abs/2608.24544)) | likely <5 ms for ≤150 tracks | Beats KLT in VINS-Mono/OpenVINS; shows that **chaining RAFT flow (RAFT-Sparse) does not beat KLT** because errors accumulate | No code link found | N/A until code released; the design idea is directly usable |

Notes on the table:

- 2080 Ti extrapolations assume FP16 TensorRT where the model exports cleanly.
  Turing has FP16 tensor cores but no FP8 (Ada+) and no FlashAttention-2 (Ampere+),
  so published RTX 30/40-series LightGlue gains from those features do not transfer.
- Point-tracker timings (CoTracker3/Track-On/TAPNext) are at ~256x256 internal
  resolution. Their accuracy metrics (δ_avg at 256x256) are not sub-pixel accuracy
  at 1024x576; they need a local refinement stage to feed BA.
- VINS-Mono details below come from its well-known source (GFTT + pyramidal LK
  21x21/3 levels + F-matrix RANSAC + minimum-distance mask); not re-fetched in
  this session.

---

## 2. How production systems do it

| System | Detection | Frame-to-frame association | Outlier rejection | New-feature policy |
|---|---|---|---|---|
| cuVSLAM | GFTT, top-k per grid cell | GPU pyramidal LK, **NCC gate at each level**, pose-predicted init | NCC; reprojection in BA | Refill cells when tracked count drops; keyframe on overlap loss |
| Basalt | FAST in 50 px grid | Pyramidal inverse-compositional **SE(2) patch** LK with locally-scaled SSD (gain-invariant) | **Forward-backward** track must return to start | One per empty grid cell |
| VINS-Mono/Fusion | GFTT | Pyramidal LK | F-matrix RANSAC, optional reverse LK | Minimum-distance mask around live tracks |
| ORB-SLAM3 | ORB every frame (grid-bucketed) | Project map points with constant-velocity model; **descriptor search in a window**, Hamming threshold + ratio test ([paper](https://arxiv.org/abs/2007.11898)) | Motion-only BA; orientation-consistency histogram | Keyframe insertion |
| OKVIS2 | BRISK every frame | Match to landmarks using **both descriptor distance and reprojection distance** ([paper](https://arxiv.org/abs/2202.09199)) | Estimator robust costs | Triangulate from live frame vs keyframes |
| AirSLAM | PLNet points+lines (TRT) | Learned matching (TRT, C++); 73 Hz on PC ([repo](https://github.com/sair-lab/AirSLAM)) | Geometric | — |
| DL-VINS-Factory (2026) | SuperPoint/ALIKED/XFeat/RaCo | **LK flow or LightGlue** ([paper](https://arxiv.org/abs/2607.01757)) | — | — |

DL-VINS-Factory's finding is the most relevant head-to-head: optical-flow
tracking keeps **longer tracks**, LightGlue **re-establishes correspondences**
and wins under large viewpoint change; neither dominates everywhere. The 2026
"Why does deep learning improve visual SLAM?" study
([arXiv 2607.06023](https://arxiv.org/abs/2607.06023)) attributes learned-SLAM
gains to **learned 2D data association plus per-observation uncertainty**, not
recurrent architecture.

Common patterns the current tracker lacks: every production system (1) has an
explicit **outlier test** (NCC, forward-backward, F-RANSAC, or BA), (2) uses
**appearance** (patch or descriptor) to confirm a prediction, not position only,
(3) enforces **spatial distribution** when seeding, and (4) keeps a track
alive by **searching near the prediction** instead of requiring an independent
detector to fire.

---

## 3. Ranked recommendation (new tracking modes to prototype)

### 1. Detector-seeded GPU KLT with flow initialization ("KLT mode")

SuperPoint (or GFTT) only **seeds** tracks into empty grid cells; existing tracks
are propagated by pyramidal LK on the GPU, initialized from NVOF (or pose
prediction), gated by NCC against the track's **reference patch** and by a
forward-backward check, then filtered by a GPU F/H RANSAC.

- Why first: the cheapest (~1–3 ms), sub-pixel, and the standard way to get
  long tracks (cuVSLAM, Basalt, VINS). Tracks survive frames where SuperPoint
  does not re-detect, which is the main cause of fragmentation today.
- Code to borrow: cuVSLAM `libs/cuda_modules/lk_tracker.*`, `gftt.*`,
  `selection_v2.*` (NVIDIA Community License, fine on NVIDIA hardware); or
  OpenCV `cuda::SparsePyrLKOpticalFlow` for a quick baseline.
- Risk: drift on long tracks. Mitigate KLTNet-style: refine against the
  **first-observation patch** (reference-anchored), not only the previous frame.

### 2. SuperPoint + LightGlue (TensorRT) with flow/geometry gating ("LightGlue mode")

Run LightGlue between the previous frame (or last keyframe) and the current
frame on the SuperPoint features already computed. Accept a match only if it
also agrees with the NVOF prediction (radius) and passes F-RANSAC.

- Why: learned association is where the measurable gains are (2607.06023,
  DL-VINS-Factory); robust to fast motion, blur and repeated texture; reuses the
  exact SuperPoint features and LightGlue weights of the offline
  `match-all` pipeline ([MATCHING.md](../MATCHING.md)), so online and offline tracks
  are comparable.
- Cost: likely 5–15 ms on the 2080 Ti at 1024–2048 keypoints (extrapolated).
  Consider capping to 1024 keypoints for matching, or running LightGlue only
  against keyframes while KLT/flow handles the in-between frames.
- Integration: export via LightGlue-ONNX (fixed K, no adaptive depth/width in TRT).
  Consider ALIKED (BSD-3) or XFeat (Apache-2.0) if SuperPoint's non-commercial
  license matters.

### 3. Upgraded hybrid: SuperPoint + flow + descriptor gate ("hybrid v2")

Keep the current structure but adopt the pieces listed in section 4. This is the
lowest-risk change and a baseline for comparing modes 1 and 2.

### Not recommended for the real-time path

- **CoTracker3 / Track-On2 / TAPNext**: excellent long-term trackers but
  256x256-class internal resolution and 25 ms to 400 ms per frame at 1000 points on
  this GPU (extrapolated), no TRT exports, CoTracker non-commercial. Useful as
  an **offline oracle**: run on short clips to measure how often the online
  tracker fragments a point that is actually continuous. TAPNext is the only one
  plausibly close to real time and is worth one offline experiment.
- **EfficientLoFTR / RoMa v2 / MASt3R**: too slow for 60 fps on Turing and not
  repeatable keypoints; better used for offline pair verification or loop edges.
- **DPVO / DROID / MAC-VO**: full VO systems; patches don't form long exportable
  tracks. DPVO is useful as a **pose reference** for evaluating tracks.
- **SEA-RAFT / RAFT** as predictor: too slow. If NVOF quality is limiting,
  **NeuFlow v2** (15 ms at 1024x436 on an RTX 2080, Apache-2.0) is the learned
  flow to try, but only as a predictor.

---

## 4. What the SuperPoint + motion-vector design should borrow

Ordered by expected impact per effort:

1. **Never chain flow into positions.** KLTNet shows chained RAFT flow drifts
   and does not beat KLT. The current design already snaps to detections; keep
   flow strictly as the predictor.
2. **Guided detection instead of independent detection.** Export the SuperPoint
   heatmap (pre-NMS score map) from the TRT engine and, for each live track,
   take the local maximum within the gate radius of the flow prediction at a
   **lower threshold** than global detection. This is the ORB-SLAM
   "search by projection" idea applied to the detector, and removes most
   missed-detection fragmentation without flow-only carryover.
3. **Descriptor gate + ratio test.** Sample the 256-d descriptor at candidate
   locations (the descriptor map is already computed) and require cosine
   similarity to the track's **reference descriptor** (first observation) and last
   descriptor above a threshold, plus a best/second-best ratio inside the
   radius. OKVIS2 uses descriptor + reprojection distance; ORB-SLAM uses
   window + Hamming + ratio. This fixes the "nearest wins" ambiguity documented in
   ARCHITECTURE.md.
4. **Global assignment inside the gate.** Replace nearest-proposal-then-accept
   with a cost `α·dist² + β·(1 − cos)` and mutual-best or a GPU auction/Hungarian on
   the sparse gated bipartite graph, so losers can take their second choice.
5. **Forward-backward consistency.** Compute NVOF backward flow (the SDK lists
   forward-backward support; confirm on Turing via capability query) and drop
   associations whose round trip exceeds ~1 px (Basalt, VINS reverse check). Also
   use the NVOF **cost buffer** as a per-vector confidence (availability on Turing
   not verified here).
6. **Geometric verification per frame.** GPU RANSAC for F (or H for
   low-parallax frames) over all associations; mark outliers and don't extend
   those tracks. Every production system has this step; the tracker has none.
7. **Sub-pixel refinement.** SuperPoint keypoints are pixel-grid; add a soft-argmax
   on the heatmap or a small LK/NCC refinement against the reference patch
   (cuVSLAM NCC, KLTNet reference anchoring). BA benefits directly.
8. **Short coasting with appearance verification.** Allow a track to survive
   k≤3 frames without a detection if an NCC/LK check against its reference
   patch passes at the predicted position. Mark those observations
   as refined, not detected, so they can be weighted separately. This differs from
   the removed confidence-decay experiment because each coasted frame is
   verified by appearance.
9. **Spatially uniform seeding.** Seed new tracks per grid cell (cuVSLAM,
   Basalt 50 px grid, VINS minimum distance) instead of by strongest global score.
   Score-ranked filling clusters on high-texture regions such as captions.
10. **Re-acquisition of recently dead tracks.** Keep descriptors of tracks that
    died in the last N frames; match new unmatched detections to them (descriptor
    + flow-propagated position) to merge fragments online. The offline
    `match-all` global association already does this across all pairs.
11. **Per-observation uncertainty.** Save snap distance, descriptor similarity,
    NCC and fwd-bwd error per observation; use them as BA weights
    (KLTNet anisotropic confidence, MAC-VO covariance, 2607.06023 findings).
12. **Pipeline overlap.** NVOF runs on the fixed-function block, so it can overlap
    SuperPoint inference on a separate CUDA stream, and the host flow round-trip
    noted in ARCHITECTURE.md can be removed. That frees budget for items 2–6.

---

## 5. Suggested evaluation

- Metrics per mode: track-length histogram (median, p90), fraction of
  observations surviving F-RANSAC, reprojection error after a short-window BA,
  ms per stage.
- Ground truth: TAP-Vid-DAVIS/Kinetics (2D tracks), EuRoC/TUM (poses, via the
  existing solver). For Osaka, use offline LightGlue + geometry tracks from
  `match-all → verify-matches` as a reference and CoTracker3/TAPNext on short clips
  as a continuity oracle.

## Sources

- cuVSLAM: [paper](https://arxiv.org/abs/2506.04359), [repo](https://github.com/nvidia-isaac/cuVSLAM), [LICENSE](https://github.com/nvidia-isaac/cuVSLAM/blob/main/LICENSE)
- KLTNet: [arXiv 2608.24544](https://arxiv.org/abs/2608.24544)
- Faster than FAST / vilib: [arXiv 2003.13493](https://arxiv.org/abs/2003.13493), [vilib](https://github.com/uzh-rpg/vilib)
- OpenCV CUDA LK: [docs](https://docs.opencv.org/4.13.0/d7/d05/classcv_1_1cuda_1_1SparsePyrLKOpticalFlow.html); VPI LK: [docs](https://archive.docs.nvidia.com/vpi/algo_optflow_lk.html)
- LightGlue: [repo](https://github.com/cvg/LightGlue); LightGlue-ONNX: [repo](https://github.com/fabio-sim/LightGlue-ONNX), [TRT blog](https://fabio-sim.github.io/blog/accelerating-lightglue-inference-onnx-runtime-tensorrt/), [FP8 blog](https://fabio-sim.github.io/blog/fp8-quantized-lightglue-tensorrt-nvidia-model-optimizer/); [SuperPoint-LightGlue-TensorRT](https://github.com/yuefanhao/SuperPoint-LightGlue-TensorRT)
- SuperPoint license: [Magic Leap LICENSE](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE); MIT reimplementation: [rpautrat/SuperPoint](https://github.com/rpautrat/SuperPoint)
- XFeat: [repo](https://github.com/verlab/accelerated_features), [XFeatTensorRT](https://github.com/PranavNedunghat/XFeatTensorRT); ALIKED TRT: [aliked-tensorrt](https://github.com/ajuric/aliked-tensorrt)
- EfficientLoFTR: [paper](https://arxiv.org/abs/2403.04765), [repo](https://github.com/zju3dv/EfficientLoFTR); RoMa v2: [paper](https://arxiv.org/abs/2511.15706); MASt3R-SLAM: [paper](https://arxiv.org/abs/2412.12392)
- NVOF: [application note](https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-application-note), [intro blog](https://developer.nvidia.com/blog/an-introduction-to-the-nvidia-optical-flow-sdk/)
- NeuFlow v2: [paper](https://arxiv.org/abs/2408.10161), [repo](https://github.com/neufieldrobotics/NeuFlow_v2); SEA-RAFT: [paper](https://arxiv.org/abs/2405.14793), [repo](https://github.com/princeton-vl/SEA-RAFT)
- CoTracker3: [paper](https://arxiv.org/abs/2410.11831), [repo](https://github.com/facebookresearch/co-tracker); Track-On: [paper](https://arxiv.org/abs/2501.18487), [Track-On2](https://arxiv.org/abs/2509.19115), [repo](https://github.com/gorkaydemir/track_on); TAPNext: [paper](https://arxiv.org/abs/2504.05579), [page](https://tap-next.github.io/); LocoTrack: [paper](https://arxiv.org/abs/2407.15420)
- DPVO: [paper](https://arxiv.org/abs/2208.04726), [repo](https://github.com/princeton-vl/DPVO); DPV-SLAM: [paper](https://arxiv.org/abs/2408.01654); MAC-VO: [repo](https://github.com/MAC-VO/MAC-VO)
- Basalt: [paper](https://arxiv.org/abs/1904.06504); ORB-SLAM3: [paper](https://arxiv.org/abs/2007.11898); OKVIS2: [paper](https://arxiv.org/abs/2202.09199); AirSLAM: [repo](https://github.com/sair-lab/AirSLAM)
- DL-VINS-Factory: [arXiv 2607.01757](https://arxiv.org/abs/2607.01757); Why does deep learning improve visual SLAM?: [arXiv 2607.06023](https://arxiv.org/abs/2607.06023)
