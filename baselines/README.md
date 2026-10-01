# Pipeline baselines

Each folder is one snapshot of what the full pipeline (decode → SuperPoint → flow
tracker → VO, `slam-native-tracking-bench`) produces, made with
`native_workbench/tools/baseline.py --skips 0,150,300`: every sequence from
three start frames, because monocular VO is chaotic (one keyframe decision can
change a whole run). `manifest.json` has the commit, binary hash and exact
command per run; runs are deterministic.

- **v1**: commit `fdbdc89` (before the cleanup). VO with loop closure (spatial +
  global appearance search), relocalization, dormant landmarks, re-association
  and merging, uncertainty weighting, final global BA. Loops solved
  synchronously (`loop-sync 1`); the bench was patched only to add `skip`.
- **v2**: the minimal VO (two-view init, PnP tracking with P3P recovery,
  keyframes, triangulation, 8-keyframe local BA, new segment on loss). Run with
  the v3 binary and `reassoc-radius 0 local-map 8`, which reproduces it
  byte for byte.
- **v3**: commit `4bb589f`: v2 plus the local map (30 keyframes) with
  re-association and merging.
- **v4**: commit `4cf0c42`: v3 plus coasting on the motion model,
  relocalization in the local map, and a covariance and confidence per pose
  (the trajectory CSV gains `predicted` and `confidence` columns). The run
  was interrupted and resumed: the first 14 runs come from a build of the
  same source before a formatting-only edit (their manifest commands show the
  old `/home/nick` path), the last 4 from the binary in the manifest.
  fr1_desk re-run with that binary is byte-identical.

## ATE (cm, largest segment, sim3-aligned): mean ± range over the three starts

| sequence        | v1            | v2            | v3            | v4            |
|-----------------|--------------:|--------------:|--------------:|--------------:|
| fr1_desk        |  3.4 ±   2.4  |  8.9 ±   9.6  |  9.3 ±  12.5  |  9.3 ±  12.5  |
| fr1_room        | 33.6 ±  17.8  | 51.7 ±  54.1  | 43.2 ±  13.8  | 43.2 ±  13.8  |
| fr1_xyz         |  1.2 ±   0.1  |  6.7 ±  11.3  |  1.7 ±   0.2  |  1.7 ±   0.2  |
| fr2_desk        |  3.9 ±   6.9  | 14.6 ±   6.0  | 13.4 ±   5.1  | 13.4 ±   5.1  |
| fr3_long_office | 58.8 ± 170.1  | 63.4 ± 164.0  |  5.7 ±   1.5  |  5.3 ±   2.4  |

dreamworks (no ground truth), full run: v1 posed 831/882 frames in 4 segments
(2 relocalized, largest 822), v2 847/882 in 6 (largest 557), v3 828/882 in 4
(largest 557), v4 878/882 in 1 (2 relocalized, 56 frames coasted). From frame
150: v3 678/732 in 4, v4 728/732 in 1 (3 relocalized). From frame 300: v3
528/582 in 4, v4 576/582 in 2 (1 relocalized; lost at frame 675 after 7
coasted frames).

## Reading

- Re-association (v3) removes v2's blow-ups (fr1_xyz, fr3) and narrows the
  spread; on fr3 it beats v1, which also blew up from one start.
- The remaining gap to v1 (fr1_desk, fr1_room, fr2_desk) is loop closure and
  global BA. Single-start attribution on fr1 (v1 with features switched off):
  without loops fr1_desk 3.0 → 13.7, fr1_room 24.7 → 61.1; also without the
  descriptor features 11.0 / 74.2; GBA + uncertainty weighting are the last
  1–5 cm.
- Coasting and relocalization (v4) leave TUM unchanged except one run: 14 of
  the 15 summary rows are identical to v3 (fr1_room is still 2 segments), and
  fr3 from frame 300 moves from 5.5 to 4.2 cm, still in 2 segments. On
  dreamworks they carry the pose behind the columns: 4 segments become 1 from
  two of the three starts.
- Motion-model calibration (v4, dreamworks): at the six relocalizations the
  prediction was off by 13–99 px (median) against a sigma of 36–143 px, so
  sigma was 1.1–11× the error (median 2.5×): conservative.
- v2/v3 run 3–4× faster than v1 (which spent most of its time in the
  brute-force global loop search).
- v1 crashed once on fr2_desk (`double free or corruption`) and passed on the
  rerun: an intermittent memory bug in code that was deleted.
