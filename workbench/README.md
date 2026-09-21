# SLAM Workbench

This is the browser-based SLAM workbench. The Python adapter reads persisted match,
geometry, and reconstruction artifacts; the Three.js frontend renders image matches,
match matrices, and sparse reconstructions. Its pipeline supervisor launches the
existing CLI commands in worker processes and treats their files as authoritative.
It also streams the original video and renders live SuperPoint detections or causal
online landmark tracks over frames selected by timestamp.

## Build and run

From the repository root:

```bash
cd workbench
npm install
npm run build
cd ..
.venv/bin/python -m slam_lab.workbench_server \
  recordings/osaka-allpairs-lightglue recordings/osaka-allpairs-cosine
```

Then open <http://127.0.0.1:8765>. The server binds to loopback by default and does
does not modify existing artifacts.

For frontend development, run the Python server in one terminal and Vite in another:

```bash
cd workbench
npm run dev
```

Open <http://127.0.0.1:5173>. Vite proxies `/api` to the Python server on port 8765.

## Current interaction

* Choose any completed pair of matching rows.
* Toggle the same pair between LightGlue and cosine matching runs.
* Switch to a Turbo-colored, log-scaled match-count matrix and click a cell to open it.
* Explore sparse reconstructed points, camera frustums and the estimated trajectory in 3D.
* Pan and zoom the image planes.
* Toggle correspondence links.
* Toggle all cached SuperPoint features behind the accepted matches.
* Follow a feature cache while detection is running; scrub committed rows and inspect
  per-frame SuperPoint inference time without storing duplicate images.
* Play or scrub online landmark tracks with stable ID colors and adjustable trails.
* Click an online feature to inspect landmark lifetime, observation count,
  descriptor concentration, and current/mean cosine similarity.
* Click a matched feature to inspect feature IDs, score, descriptor distance and
  both pixel coordinates.
* See whether the run uses LightGlue, cosine matching or the NN ratio matcher.
* For mixed LightGlue runs, see whether the selected pair was produced on CPU or
  CUDA.
* Launch the lightweight video → SuperPoint → online tracks workflow from the
  Pipeline view, or select the extended matching → geometry → reconstruction path.
  Follow stage progress and logs or stop the active worker.
* Select newly published artifacts without restarting the server; the catalog scans
  `recordings/` every five seconds.

The UI polls run status while matching continues. A pair becomes viewable as soon
as its SQLite transaction commits.
