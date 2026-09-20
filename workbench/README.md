# SLAM Workbench

This is the browser-based, read-only workbench for matching artifacts. The Python
adapter reads `matches.sqlite3` through `MatchStore`; the Three.js frontend renders
the two source images, matched keypoints and correspondence links.

## Build and run

From the repository root:

```bash
cd workbench
npm install
npm run build
cd ..
.venv/bin/python -m slam_lab.workbench_server recordings/osaka-allpairs-lightglue
```

Then open <http://127.0.0.1:8765>. The server binds to loopback by default and does
not modify the matching run.

For frontend development, run the Python server in one terminal and Vite in another:

```bash
cd workbench
npm run dev
```

Open <http://127.0.0.1:5173>. Vite proxies `/api` to the Python server on port 8765.

## Current interaction

* Choose any completed pair of matching rows.
* Pan and zoom the image planes.
* Toggle correspondence links.
* Click a matched feature to inspect feature IDs, score, descriptor distance and
  both pixel coordinates.
* See whether the run uses LightGlue, cosine matching or the NN ratio matcher.
* For mixed LightGlue runs, see whether the selected pair was produced on CPU or
  CUDA.

The UI polls run status while matching continues. A pair becomes viewable as soon
as its SQLite transaction commits.
