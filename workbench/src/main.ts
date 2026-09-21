import * as THREE from "three";
import { MapControls } from "three/addons/controls/MapControls.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import "./style.css";

type Matcher = Record<string, unknown> & { name: string; score?: string; device?: string };

type FrameInfo = {
  row: number;
  frame_index: number;
  timestamp_ns: number;
  width: number;
  height: number;
  keypoint_count: number;
  image_url?: string;
  video_url?: string;
};

type PairData = {
  first: FrameInfo;
  second: FrameInfo;
  matcher: Matcher;
  pair_runtime: Matcher | null;
  matches: {
    count: number;
    indices: [number, number][];
    scores: number[];
    distances: number[];
    first_xy: [number, number][];
    second_xy: [number, number][];
  };
  features: {
    first_xy: [number, number][];
    second_xy: [number, number][];
  };
};

type RunOverview = {
  run: string;
  source: string | null;
  status: {
    running: boolean;
    phase: string;
    frames: number;
    completed_pairs: number;
    total_pairs: number;
    matcher: Matcher;
  };
  suggested_pair: { first: number; second: number; matches: number } | null;
};

type RunCatalog = {
  default: string;
  runs: {
    id: string;
    run: string;
    status: RunOverview["status"];
    reconstructions: string[];
  }[];
  reconstructions: ReconstructionOption[];
  geometry: GeometryOption[];
  online_tracks: OnlineTrackOption[];
  detections: DetectionOption[];
};

type DetectionOption = {
  id: string;
  name: string;
  path: string;
  source: string | null;
  complete: boolean;
  selection_complete: boolean;
  selection_max_frames: number | null;
  frames: number;
  latest_row: number | null;
  average_detection_ms: number | null;
  extractor: Record<string, unknown>;
  video: { total_frames?: number; duration_ns?: number; average_rate?: number };
  updated_at: string | null;
};

type DetectionFrame = {
  overview: DetectionOption;
  frame: FrameInfo & { extraction_ms: number };
  keypoints: [number, number][];
  scores: number[];
};

type OnlineTrackOption = {
  id: string;
  name: string;
  path: string;
  phase: string;
  running: boolean;
  processed_frames: number;
  total_frames: number;
  observations: number;
  landmarks: number;
  matched_observations: number;
  new_observations: number;
  source_path: string | null;
  source_cache: string;
  config: {
    algorithm?: string;
    settings?: {
      min_similarity?: number;
      min_margin?: number;
      max_inactive_frames?: number;
      device?: string;
    };
  };
  updated_at: string | null;
};

type OnlineObservation = {
  feature_id: number;
  x: number;
  y: number;
  score: number;
  landmark_id: number;
  state: "new" | "matched";
  similarity: number | null;
  second_similarity: number | null;
  observation_count: number;
  concentration: number;
  first_row: number;
  last_row: number;
  trail: [number, number, number][];
};

type OnlineTrackFrame = {
  overview: OnlineTrackOption;
  frame: FrameInfo & {
    new_count: number;
    matched_count: number;
    mean_similarity: number | null;
  };
  observations: OnlineObservation[];
};

type OnlineLandmark = {
  landmark_id: number;
  observation_count: number;
  concentration: number;
  first_row: number;
  last_row: number;
  observations: {
    row: number;
    frame_index: number;
    timestamp_ns: number;
    feature_id: number;
    x: number;
    y: number;
    score: number;
    state: string;
    similarity: number | null;
    second_similarity: number | null;
  }[];
};

type ReconstructionOption = ReconstructionData["overview"] & {
  id: string;
  name: string;
  match_run_id: string | null;
  parent_artifact_id: string | null;
  updated_at: string | null;
};

type GeometryOption = {
  name: string;
  path: string;
  artifact_id: string | null;
  parent_artifact_id: string | null;
  state: string | null;
  updated_at: string | null;
  frames: number | null;
  pairs: number | null;
  verified_edges: number | null;
  tracks: number | null;
};

type PipelineStage = {
  id: "features" | "online_tracks" | "matches" | "geometry" | "reconstruction";
  label: string;
  state: "queued" | "running" | "complete" | "failed" | "canceled";
  current: number;
  total: number | null;
  detail: string | null;
};

type PipelineJob = {
  job_id: string;
  name: string;
  state: "starting" | "running" | "complete" | "failed" | "canceled";
  phase: string;
  revision: number;
  updated_at: string;
  last_error: { code: string; message: string } | null;
  config: {
    source: string;
    frames: number;
    stride: number;
    matcher: string;
    device: string;
    workflow: "tracks" | "full";
  };
  paths: Record<string, string>;
  stages: PipelineStage[];
  log_tail: string;
  worker_alive?: boolean;
  child_alive?: boolean;
};

type PipelineOverview = { sources: string[]; jobs: PipelineJob[] };

type MatrixData = {
  size: number;
  counts: number[];
  max_count: number;
  matcher: Matcher;
};

type ReconstructionData = {
  overview: {
    name: string;
    path: string;
    artifact_id: string | null;
    parent_artifact_id: string | null;
    updated_at: string | null;
    backend: string;
    registered_frames: number;
    frames: number;
    points: number;
    observations: number;
    median_reprojection_error_px: number | null;
    p95_reprojection_error_px: number | null;
  };
  points: [number, number, number][];
  colors: [number, number, number][];
  track_lengths: number[];
  poses: number[][][];
  frame_indices: number[];
  timestamps_ns: number[];
  K: number[][];
  pose_convention: string;
  scale: string | null;
};

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing element #${id}`);
  return found as T;
}

function matcherLabel(matcher: Matcher): string {
  if (matcher.name === "lightglue-superpoint") return "LightGlue";
  if (matcher.name === "cosine-mutual-nearest-neighbor") return "Cosine";
  if (matcher.name === "mutual-nearest-neighbor") return "NN ratio";
  return matcher.name;
}

function matcherDescription(matcher: Matcher): string {
  if (matcher.name === "lightglue-superpoint") {
    return "Learned SuperPoint feature matching. Color identifies a correspondence; score is LightGlue confidence.";
  }
  if (matcher.name === "cosine-mutual-nearest-neighbor") {
    return "Mutual best descriptor matches filtered by cosine similarity. Score is cosine similarity.";
  }
  return String(matcher.score ?? "Appearance matcher confidence");
}

function detailRow(term: string, value: string): HTMLDivElement {
  const row = document.createElement("div");
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = term;
  dd.textContent = value;
  row.append(dt, dd);
  return row;
}

function turbo(value: number): [number, number, number] {
  const x = THREE.MathUtils.clamp(value, 0, 1);
  const evaluate = (coefficients: number[]) =>
    coefficients.reduceRight((result, coefficient) => result * x + coefficient, 0);
  return [
    evaluate([0.13572138, 4.6153926, -42.66032258, 132.13108234, -152.94239396, 59.28637943]),
    evaluate([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604]),
    evaluate([0.1066733, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973]),
  ].map((channel) => Math.round(255 * THREE.MathUtils.clamp(channel, 0, 1))) as [
    number,
    number,
    number,
  ];
}

class PairScene {
  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.OrthographicCamera;
  private perspective: THREE.PerspectiveCamera;
  private activeCamera: THREE.Camera;
  private controls: OrbitControls;
  private content = new THREE.Group();
  private points: THREE.Points | null = null;
  private links: THREE.LineSegments | null = null;
  private features: THREE.Points | null = null;
  private matrixMesh: THREE.Mesh | null = null;
  private matrix: MatrixData | null = null;
  private highlight = new THREE.Group();
  private extents: { width: number; height: number; centerX: number } | null = null;
  private reconstructionBounds: THREE.Box3 | null = null;
  private reconstructionPoses: THREE.Matrix4[] = [];
  private reconstructionPoints: THREE.Points | null = null;
  private cameraFrustums: THREE.LineSegments | null = null;
  private cameraTrajectory: THREE.Line | null = null;
  private reconstructionAxes: THREE.AxesHelper | null = null;
  private reconstructionGrid: THREE.GridHelper | null = null;
  private frustumBaseSize = 1;
  private frustumScale = 1;
  private reconstructionPointSize = 3;
  private reconstructionPointOpacity = 1;
  private cameraOpacity = 0.8;
  private pointsVisible = true;
  private camerasVisible = true;
  private trajectoryVisible = true;
  private axesVisible = true;
  private gridVisible = false;
  private pair: PairData | null = null;
  private onlineFrame: OnlineTrackFrame | null = null;
  private video: HTMLVideoElement | null = null;
  private videoTexture: THREE.VideoTexture | null = null;
  private videoUrl = "";
  private pairVideos: HTMLVideoElement[] = [];
  private selectionHandler: (index: number) => void;
  private matrixHandler: (row: number, column: number, count: number) => void;
  private onlineSelectionHandler: (index: number) => void;

  constructor(
    container: HTMLElement,
    selectionHandler: (index: number) => void,
    matrixHandler: (row: number, column: number, count: number) => void,
    onlineSelectionHandler: (index: number) => void,
  ) {
    this.container = container;
    this.selectionHandler = selectionHandler;
    this.matrixHandler = matrixHandler;
    this.onlineSelectionHandler = onlineSelectionHandler;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.prepend(this.renderer.domElement);

    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 3000);
    this.camera.position.set(0, 0, 1000);
    this.perspective = new THREE.PerspectiveCamera(48, 1, 0.001, 10000);
    this.activeCamera = this.camera;
    this.controls = new MapControls(this.camera, this.renderer.domElement);
    this.controls.enableRotate = false;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.minZoom = 0.08;
    this.controls.maxZoom = 12;

    this.scene.add(this.content, this.highlight);
    new ResizeObserver(() => this.resize()).observe(container);
    this.renderer.domElement.addEventListener("pointerdown", (event) => this.selectAt(event));
    this.resize();
    this.animate();
  }

  private resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.left = -width / 2;
    this.camera.right = width / 2;
    this.camera.top = height / 2;
    this.camera.bottom = -height / 2;
    this.camera.updateProjectionMatrix();
    this.perspective.aspect = width / height;
    this.perspective.updateProjectionMatrix();
  }

  private animate = () => {
    requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.activeCamera);
  };

  private use2DControls() {
    if (this.activeCamera === this.camera) return;
    this.controls.dispose();
    this.activeCamera = this.camera;
    this.controls = new MapControls(this.camera, this.renderer.domElement);
    this.controls.enableRotate = false;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.minZoom = 0.08;
    this.controls.maxZoom = 12;
  }

  private use3DControls() {
    if (this.activeCamera === this.perspective) return;
    this.controls.dispose();
    this.activeCamera = this.perspective;
    this.controls = new OrbitControls(this.perspective, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.zoomToCursor = true;
  }

  private texture(url: string): Promise<THREE.Texture> {
    return new Promise((resolve, reject) => {
      new THREE.TextureLoader().load(
        url,
        (texture) => {
          texture.colorSpace = THREE.SRGBColorSpace;
          texture.minFilter = THREE.LinearFilter;
          resolve(texture);
        },
        undefined,
        reject,
      );
    });
  }

  private async videoAt(url: string, timestampSeconds: number): Promise<THREE.VideoTexture> {
    if (!this.video || this.videoUrl !== url) {
      this.video?.pause();
      this.video?.removeAttribute("src");
      this.video?.load();
      this.videoTexture?.dispose();
      const video = document.createElement("video");
      video.preload = "auto";
      video.muted = true;
      video.playsInline = true;
      video.src = url;
      this.video = video;
      this.videoUrl = url;
      this.videoTexture = new THREE.VideoTexture(video);
      this.videoTexture.colorSpace = THREE.SRGBColorSpace;
      this.videoTexture.minFilter = THREE.LinearFilter;
      await new Promise<void>((resolve, reject) => {
        video.addEventListener("loadedmetadata", () => resolve(), { once: true });
        video.addEventListener("error", () => reject(new Error("Could not load source video")), {
          once: true,
        });
        video.load();
      });
    }
    const video = this.video;
    if (!video || !this.videoTexture) throw new Error("Source video is unavailable");
    const target = Math.max(0, Math.min(timestampSeconds, video.duration || timestampSeconds));
    if (target <= 0.0005 && video.readyState < 2) {
      await new Promise<void>((resolve, reject) => {
        video.addEventListener("loadeddata", () => resolve(), { once: true });
        video.addEventListener("error", () => reject(new Error("Could not decode source video")), {
          once: true,
        });
      });
    } else if (Math.abs(video.currentTime - target) > 0.0005) {
      await new Promise<void>((resolve, reject) => {
        const done = () => resolve();
        video.addEventListener("seeked", done, { once: true });
        video.addEventListener("error", () => reject(new Error("Could not seek source video")), {
          once: true,
        });
        video.currentTime = target;
      });
    }
    this.videoTexture.needsUpdate = true;
    return this.videoTexture;
  }

  private async pairVideoAt(url: string, timestampSeconds: number): Promise<THREE.VideoTexture> {
    const video = document.createElement("video");
    video.preload = "auto";
    video.muted = true;
    video.playsInline = true;
    video.src = url;
    this.pairVideos.push(video);
    await new Promise<void>((resolve, reject) => {
      video.addEventListener("loadedmetadata", () => resolve(), { once: true });
      video.addEventListener("error", () => reject(new Error("Could not load source video")), {
        once: true,
      });
      video.load();
    });
    const target = Math.max(0, Math.min(timestampSeconds, video.duration));
    if (target <= 0.0005) {
      if (video.readyState < 2) {
        await new Promise<void>((resolve, reject) => {
          video.addEventListener("loadeddata", () => resolve(), { once: true });
          video.addEventListener("error", () => reject(new Error("Could not decode source video")), {
            once: true,
          });
        });
      }
    } else {
      await new Promise<void>((resolve, reject) => {
        video.addEventListener("seeked", () => resolve(), { once: true });
        video.addEventListener("error", () => reject(new Error("Could not seek source video")), {
          once: true,
        });
        video.currentTime = target;
      });
    }
    const texture = new THREE.VideoTexture(video);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    return texture;
  }

  private clear() {
    for (const video of this.pairVideos) {
      video.pause();
      video.removeAttribute("src");
      video.load();
    }
    this.pairVideos = [];
    for (const object of [...this.content.children, ...this.highlight.children]) {
      object.removeFromParent();
      const renderable = object as THREE.Mesh;
      renderable.geometry?.dispose();
      const materials = Array.isArray(renderable.material) ? renderable.material : [renderable.material];
      for (const material of materials) {
        if (!material) continue;
        const map = (material as THREE.MeshBasicMaterial).map;
        if (map !== this.videoTexture) map?.dispose();
        material.dispose();
      }
    }
    this.points = null;
    this.links = null;
    this.features = null;
    this.matrixMesh = null;
    this.matrix = null;
    this.onlineFrame = null;
    this.extents = null;
    this.reconstructionBounds = null;
    this.reconstructionPoses = [];
    this.reconstructionPoints = null;
    this.cameraFrustums = null;
    this.cameraTrajectory = null;
    this.reconstructionAxes = null;
    this.reconstructionGrid = null;
  }

  private frustumGeometry(size: number): THREE.BufferGeometry {
    const vertices: THREE.Vector3[] = [];
    for (const matrix of this.reconstructionPoses) {
      const origin = new THREE.Vector3(0, 0, 0).applyMatrix4(matrix);
      origin.y *= -1;
      const corners = [
        new THREE.Vector3(-size, -size * 0.65, size * 1.5),
        new THREE.Vector3(size, -size * 0.65, size * 1.5),
        new THREE.Vector3(size, size * 0.65, size * 1.5),
        new THREE.Vector3(-size, size * 0.65, size * 1.5),
      ].map((corner) => {
        corner.applyMatrix4(matrix);
        corner.y *= -1;
        return corner;
      });
      for (const corner of corners) vertices.push(origin, corner);
      for (let index = 0; index < 4; index += 1) {
        vertices.push(corners[index], corners[(index + 1) % 4]);
      }
    }
    return new THREE.BufferGeometry().setFromPoints(vertices);
  }

  setFrustumScale(scale: number) {
    this.frustumScale = THREE.MathUtils.clamp(scale, 0.1, 5);
    if (!this.cameraFrustums || !this.reconstructionPoses.length) return;
    const previous = this.cameraFrustums.geometry;
    this.cameraFrustums.geometry = this.frustumGeometry(
      this.frustumBaseSize * this.frustumScale,
    );
    previous.dispose();
  }

  setReconstructionPointSize(size: number) {
    this.reconstructionPointSize = THREE.MathUtils.clamp(size, 1, 12);
    if (this.reconstructionPoints) {
      (this.reconstructionPoints.material as THREE.PointsMaterial).size =
        this.reconstructionPointSize;
    }
  }

  setReconstructionPointOpacity(opacity: number) {
    this.reconstructionPointOpacity = THREE.MathUtils.clamp(opacity, 0.05, 1);
    if (this.reconstructionPoints) {
      const material = this.reconstructionPoints.material as THREE.PointsMaterial;
      material.opacity = this.reconstructionPointOpacity;
      material.transparent = this.reconstructionPointOpacity < 1;
      material.needsUpdate = true;
    }
  }

  setCameraOpacity(opacity: number) {
    this.cameraOpacity = THREE.MathUtils.clamp(opacity, 0.05, 1);
    if (this.cameraFrustums) {
      const material = this.cameraFrustums.material as THREE.LineBasicMaterial;
      material.opacity = this.cameraOpacity;
      material.transparent = this.cameraOpacity < 1;
      material.needsUpdate = true;
    }
  }

  setReconstructionVisibility(
    target: "points" | "cameras" | "trajectory" | "axes" | "grid",
    visible: boolean,
  ) {
    if (target === "points") {
      this.pointsVisible = visible;
      if (this.reconstructionPoints) this.reconstructionPoints.visible = visible;
    } else if (target === "cameras") {
      this.camerasVisible = visible;
      if (this.cameraFrustums) this.cameraFrustums.visible = visible;
    } else if (target === "trajectory") {
      this.trajectoryVisible = visible;
      if (this.cameraTrajectory) this.cameraTrajectory.visible = visible;
    } else if (target === "axes") {
      this.axesVisible = visible;
      if (this.reconstructionAxes) this.reconstructionAxes.visible = visible;
    } else {
      this.gridVisible = visible;
      if (this.reconstructionGrid) this.reconstructionGrid.visible = visible;
    }
  }

  async show(pair: PairData) {
    this.clear();
    this.use2DControls();
    this.pair = pair;
    const [firstTexture, secondTexture] = await Promise.all([
      pair.first.image_url
        ? this.texture(pair.first.image_url)
        : this.pairVideoAt(pair.first.video_url!, pair.first.timestamp_ns / 1e9),
      pair.second.image_url
        ? this.texture(pair.second.image_url)
        : this.pairVideoAt(pair.second.video_url!, pair.second.timestamp_ns / 1e9),
    ]);
    const gap = Math.max(48, Math.round(Math.max(pair.first.width, pair.second.width) * 0.05));
    const firstX = pair.first.width / 2;
    const secondOrigin = pair.first.width + gap;
    const secondX = secondOrigin + pair.second.width / 2;
    const totalWidth = pair.first.width + gap + pair.second.width;
    const totalHeight = Math.max(pair.first.height, pair.second.height);
    this.extents = { width: totalWidth, height: totalHeight, centerX: totalWidth / 2 };

    const image = (frame: FrameInfo, texture: THREE.Texture, x: number) => {
      const mesh = new THREE.Mesh(
        new THREE.PlaneGeometry(frame.width, frame.height),
        new THREE.MeshBasicMaterial({ map: texture, toneMapped: false }),
      );
      mesh.position.set(x, 0, 0);
      const border = new THREE.LineSegments(
        new THREE.EdgesGeometry(mesh.geometry),
        new THREE.LineBasicMaterial({ color: 0x40505d, transparent: true, opacity: 0.8 }),
      );
      border.position.copy(mesh.position).setZ(0.5);
      this.content.add(mesh, border);
    };
    image(pair.first, firstTexture, firstX);
    image(pair.second, secondTexture, secondX);

    const featurePositions: number[] = [];
    for (const [x, y] of pair.features.first_xy) {
      featurePositions.push(x, pair.first.height / 2 - y, 2);
    }
    for (const [x, y] of pair.features.second_xy) {
      featurePositions.push(secondOrigin + x, pair.second.height / 2 - y, 2);
    }
    const featureGeometry = new THREE.BufferGeometry();
    featureGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(featurePositions, 3),
    );
    this.features = new THREE.Points(
      featureGeometry,
      new THREE.PointsMaterial({
        color: 0xaab4c0,
        size: 3,
        sizeAttenuation: false,
        transparent: true,
        opacity: 0.55,
      }),
    );
    this.features.visible = element<HTMLInputElement>("show-features").checked;

    const pointPositions: number[] = [];
    const linkPositions: number[] = [];
    const colors: number[] = [];
    const linkColors: number[] = [];
    pair.matches.first_xy.forEach(([ax, ay], index) => {
      const [bx, by] = pair.matches.second_xy[index];
      const a = [ax, pair.first.height / 2 - ay, 3];
      a[0] += 0;
      const b = [secondOrigin + bx, pair.second.height / 2 - by, 3];
      pointPositions.push(...a, ...b);
      linkPositions.push(...a.slice(0, 2), 1.5, ...b.slice(0, 2), 1.5);
      const color = new THREE.Color().setHSL((index * 0.61803398875) % 1, 0.78, 0.62);
      colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
      linkColors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    });

    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute("position", new THREE.Float32BufferAttribute(pointPositions, 3));
    pointGeometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    this.points = new THREE.Points(
      pointGeometry,
      new THREE.PointsMaterial({ size: 5, sizeAttenuation: false, vertexColors: true }),
    );
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute("position", new THREE.Float32BufferAttribute(linkPositions, 3));
    lineGeometry.setAttribute("color", new THREE.Float32BufferAttribute(linkColors, 3));
    this.links = new THREE.LineSegments(
      lineGeometry,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.24 }),
    );
    this.content.add(this.features, this.links, this.points);
    this.fit();
  }

  async showOnlineFrame(frame: OnlineTrackFrame) {
    this.clear();
    this.use2DControls();
    this.pair = null;
    this.onlineFrame = frame;
    if (!frame.frame.video_url) throw new Error("Online track has no source video");
    const texture = await this.videoAt(frame.frame.video_url, frame.frame.timestamp_ns / 1e9);
    const image = new THREE.Mesh(
      new THREE.PlaneGeometry(frame.frame.width, frame.frame.height),
      new THREE.MeshBasicMaterial({ map: texture, toneMapped: false }),
    );
    image.position.set(frame.frame.width / 2, 0, 0);
    const border = new THREE.LineSegments(
      new THREE.EdgesGeometry(image.geometry),
      new THREE.LineBasicMaterial({ color: 0x40505d, transparent: true, opacity: 0.8 }),
    );
    border.position.copy(image.position).setZ(0.5);
    this.extents = {
      width: frame.frame.width,
      height: frame.frame.height,
      centerX: frame.frame.width / 2,
    };

    const positions: number[] = [];
    const colors: number[] = [];
    const trailPositions: number[] = [];
    const trailColors: number[] = [];
    for (const observation of frame.observations) {
      const color = new THREE.Color().setHSL(
        (observation.landmark_id * 0.61803398875) % 1,
        observation.state === "new" ? 0.35 : 0.78,
        observation.state === "new" ? 0.78 : 0.62,
      );
      positions.push(observation.x, frame.frame.height / 2 - observation.y, 3);
      colors.push(color.r, color.g, color.b);
      for (let index = 1; index < observation.trail.length; index += 1) {
        const previous = observation.trail[index - 1];
        const current = observation.trail[index];
        trailPositions.push(
          previous[1], frame.frame.height / 2 - previous[2], 2,
          current[1], frame.frame.height / 2 - current[2], 2,
        );
        trailColors.push(color.r, color.g, color.b, color.r, color.g, color.b);
      }
    }
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    pointGeometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    this.points = new THREE.Points(
      pointGeometry,
      new THREE.PointsMaterial({ size: 6, sizeAttenuation: false, vertexColors: true }),
    );
    const trailGeometry = new THREE.BufferGeometry();
    trailGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(trailPositions, 3),
    );
    trailGeometry.setAttribute("color", new THREE.Float32BufferAttribute(trailColors, 3));
    this.links = new THREE.LineSegments(
      trailGeometry,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.72 }),
    );
    this.content.add(image, border, this.links, this.points);
    this.fit();
  }

  async showDetectionFrame(frame: DetectionFrame) {
    this.clear();
    this.use2DControls();
    this.pair = null;
    if (!frame.frame.video_url) throw new Error("Feature cache has no source video");
    const texture = await this.videoAt(frame.frame.video_url, frame.frame.timestamp_ns / 1e9);
    const image = new THREE.Mesh(
      new THREE.PlaneGeometry(frame.frame.width, frame.frame.height),
      new THREE.MeshBasicMaterial({ map: texture, toneMapped: false }),
    );
    image.position.set(frame.frame.width / 2, 0, 0);
    const positions: number[] = [];
    const colors: number[] = [];
    const maximum = Math.max(1e-12, ...frame.scores);
    frame.keypoints.forEach(([x, y], index) => {
      positions.push(x, frame.frame.height / 2 - y, 3);
      const [r, g, b] = turbo(frame.scores[index] / maximum);
      colors.push(r / 255, g / 255, b / 255);
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    this.points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({ size: 5, sizeAttenuation: false, vertexColors: true }),
    );
    this.extents = {
      width: frame.frame.width,
      height: frame.frame.height,
      centerX: frame.frame.width / 2,
    };
    this.content.add(image, this.points);
    this.fit();
  }

  showReconstruction(reconstruction: ReconstructionData) {
    this.clear();
    this.use3DControls();
    this.pair = null;

    // Saved SLAM coordinates use Y down. Reflect world-space geometry for Three.js Y up.
    const positions = reconstruction.points.flatMap(([x, y, z]) => [x, -y, z]);
    const colors = reconstruction.colors.flatMap((color) => color.map((value) => value / 255));
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    pointGeometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    const cloud = new THREE.Points(
      pointGeometry,
      new THREE.PointsMaterial({
        size: this.reconstructionPointSize,
        sizeAttenuation: false,
        vertexColors: true,
        transparent: this.reconstructionPointOpacity < 1,
        opacity: this.reconstructionPointOpacity,
      }),
    );
    cloud.visible = this.pointsVisible;
    this.reconstructionPoints = cloud;
    this.content.add(cloud);

    const bounds = new THREE.Box3();
    if (reconstruction.points.length) {
      bounds.setFromBufferAttribute(pointGeometry.getAttribute("position") as THREE.BufferAttribute);
    }
    const poseMatrices = reconstruction.poses.map((pose) =>
      new THREE.Matrix4().set(
        pose[0][0], pose[0][1], pose[0][2], pose[0][3],
        pose[1][0], pose[1][1], pose[1][2], pose[1][3],
        pose[2][0], pose[2][1], pose[2][2], pose[2][3],
        pose[3][0], pose[3][1], pose[3][2], pose[3][3],
      ),
    );
    const centers = poseMatrices.map((matrix) => {
      const center = new THREE.Vector3().setFromMatrixPosition(matrix);
      center.y *= -1;
      return center;
    });
    for (const center of centers) bounds.expandByPoint(center);
    if (bounds.isEmpty()) bounds.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(2, 2, 2));
    const radius = Math.max(0.01, bounds.getSize(new THREE.Vector3()).length() / 2);
    this.reconstructionPoses = poseMatrices;
    this.frustumBaseSize = radius * 0.035;
    this.cameraFrustums = new THREE.LineSegments(
      this.frustumGeometry(this.frustumBaseSize * this.frustumScale),
      new THREE.LineBasicMaterial({
        color: 0x73e2b5,
        transparent: this.cameraOpacity < 1,
        opacity: this.cameraOpacity,
      }),
    );
    this.cameraFrustums.visible = this.camerasVisible;
    const trajectory = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(centers),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 }),
    );
    trajectory.visible = this.trajectoryVisible;
    this.cameraTrajectory = trajectory;
    const axes = new THREE.AxesHelper(radius * 0.2);
    axes.visible = this.axesVisible;
    this.reconstructionAxes = axes;
    const grid = new THREE.GridHelper(radius * 2, 20, 0x3b4a56, 0x202932);
    grid.visible = this.gridVisible;
    this.reconstructionGrid = grid;
    this.content.add(this.cameraFrustums, trajectory, axes, grid);
    this.reconstructionBounds = bounds;
    this.fit();
  }

  showMatrix(matrix: MatrixData) {
    this.clear();
    this.use2DControls();
    this.pair = null;
    this.matrix = matrix;
    const pixels = new Uint8Array(matrix.size * matrix.size * 4);
    const denominator = Math.log1p(Math.max(1, matrix.max_count));
    for (let row = 0; row < matrix.size; row += 1) {
      for (let column = 0; column < matrix.size; column += 1) {
        const count = matrix.counts[row * matrix.size + column];
        let color: [number, number, number];
        if (count === -2) color = [25, 31, 38];
        else if (count < 0) color = [53, 59, 66];
        else color = turbo(Math.log1p(count) / denominator);
        const target = ((matrix.size - row - 1) * matrix.size + column) * 4;
        pixels.set([...color, 255], target);
      }
    }
    const texture = new THREE.DataTexture(
      pixels,
      matrix.size,
      matrix.size,
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
    );
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.NearestFilter;
    texture.magFilter = THREE.NearestFilter;
    texture.needsUpdate = true;
    this.matrixMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(matrix.size, matrix.size),
      new THREE.MeshBasicMaterial({ map: texture, toneMapped: false }),
    );
    this.matrixMesh.position.set(matrix.size / 2, 0, 0);
    this.content.add(this.matrixMesh);
    this.extents = { width: matrix.size, height: matrix.size, centerX: matrix.size / 2 };
    this.fit();
  }

  setLinks(visible: boolean) {
    if (this.links) this.links.visible = visible;
  }

  setFeatures(visible: boolean) {
    if (this.features) this.features.visible = visible;
  }

  fit() {
    if (this.reconstructionBounds && this.activeCamera === this.perspective) {
      const center = this.reconstructionBounds.getCenter(new THREE.Vector3());
      const radius = Math.max(
        0.01,
        this.reconstructionBounds.getSize(new THREE.Vector3()).length() / 2,
      );
      this.perspective.near = Math.max(0.0001, radius / 10000);
      this.perspective.far = radius * 100;
      this.perspective.position.copy(center).add(new THREE.Vector3(radius * 1.4, radius, radius * 1.4));
      this.perspective.updateProjectionMatrix();
      this.controls.target.copy(center);
      this.controls.update();
      return;
    }
    if (!this.extents) return;
    const padding = 56;
    const zoom = Math.min(
      this.container.clientWidth / (this.extents.width + padding * 2),
      this.container.clientHeight / (this.extents.height + padding * 2),
    );
    this.camera.zoom = Math.max(0.05, zoom);
    this.camera.position.set(this.extents.centerX, 0, 1000);
    this.controls.target.set(this.extents.centerX, 0, 0);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  private selectAt(event: PointerEvent) {
    const bounds = this.renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
      -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, this.camera);
    if (this.matrixMesh && this.matrix) {
      const hit = raycaster.intersectObject(this.matrixMesh, false)[0];
      if (!hit?.uv) return;
      const column = Math.min(this.matrix.size - 1, Math.floor(hit.uv.x * this.matrix.size));
      const row = Math.min(this.matrix.size - 1, Math.floor((1 - hit.uv.y) * this.matrix.size));
      const count = this.matrix.counts[row * this.matrix.size + column];
      if (row !== column && count >= 0) this.matrixHandler(row, column, count);
      return;
    }
    if (this.points && this.onlineFrame) {
      raycaster.params.Points = { threshold: 8 / this.camera.zoom };
      const hit = raycaster.intersectObject(this.points, false)[0];
      if (!hit || hit.index === undefined) return;
      this.highlightOnlineObservation(hit.index);
      return;
    }
    if (!this.points || !this.pair) return;
    raycaster.params.Points = { threshold: 8 / this.camera.zoom };
    const hit = raycaster.intersectObject(this.points, false)[0];
    if (!hit || hit.index === undefined) return;
    this.highlightMatch(Math.floor(hit.index / 2));
  }

  private highlightMatch(index: number) {
    if (!this.points || !this.pair) return;
    this.highlight.clear();
    const position = this.points.geometry.getAttribute("position");
    const a = new THREE.Vector3().fromBufferAttribute(position, index * 2);
    const b = new THREE.Vector3().fromBufferAttribute(position, index * 2 + 1);
    const geometry = new THREE.BufferGeometry().setFromPoints([a, b]);
    const line = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 }),
    );
    const points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({ color: 0xffffff, size: 11, sizeAttenuation: false }),
    );
    line.position.z = points.position.z = 5;
    this.highlight.add(line, points);
    this.selectionHandler(index);
  }

  private highlightOnlineObservation(index: number) {
    if (!this.points || !this.onlineFrame) return;
    this.highlight.clear();
    const position = this.points.geometry.getAttribute("position");
    const selected = new THREE.Vector3().fromBufferAttribute(position, index);
    const geometry = new THREE.BufferGeometry().setFromPoints([selected]);
    const point = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({ color: 0xffffff, size: 12, sizeAttenuation: false }),
    );
    point.position.z = 5;
    this.highlight.add(point);
    this.onlineSelectionHandler(index);
  }
}

const viewport = element<HTMLElement>("viewport");
let currentPair: PairData | null = null;
let currentRun = "";
let currentReconstruction = "";
let currentOnlineTrack = "";
let currentOnlineFrame: OnlineTrackFrame | null = null;
let currentTrackRow = 0;
let currentDetection = "";
let currentDetectionFrame: DetectionFrame | null = null;
let currentDetectionRow = 0;
let trackPlayback: number | null = null;
let artifactCatalog: RunCatalog | null = null;
let artifactCatalogSignature = "";
let currentPipelineJob = "";
type ViewMode = "images" | "matrix" | "detections" | "tracks" | "reconstruction" | "pipeline";
let currentMode: ViewMode = "images";
let pairInitialized = false;
let loadController: AbortController | null = null;

const scene = new PairScene(
  viewport,
  (index) => {
    if (!currentPair) return;
    const indices = currentPair.matches.indices[index];
    const score = currentPair.matches.scores[index];
    const distance = currentPair.matches.distances[index];
    const firstXY = currentPair.matches.first_xy[index];
    const secondXY = currentPair.matches.second_xy[index];
    const fields = element<HTMLDListElement>("selection-fields");
    fields.replaceChildren(
      detailRow("Match", `#${index}`),
      detailRow("Feature IDs", `${indices[0]} → ${indices[1]}`),
      detailRow("Score", score.toFixed(5)),
      detailRow("Descriptor L2", distance.toFixed(5)),
      detailRow("First pixel", `${firstXY[0].toFixed(1)}, ${firstXY[1].toFixed(1)}`),
      detailRow("Second pixel", `${secondXY[0].toFixed(1)}, ${secondXY[1].toFixed(1)}`),
    );
    fields.classList.remove("hidden");
    element("selection-empty").classList.add("hidden");
  },
  (row, column) => {
    element<HTMLInputElement>("first-row").value = String(row);
    element<HTMLInputElement>("second-row").value = String(column);
    setMode("images");
  },
  (index) => {
    const observation = currentOnlineFrame?.observations[index];
    if (observation) void inspectOnlineLandmark(observation);
  },
);

async function inspectOnlineLandmark(observation: OnlineObservation) {
  try {
    const query = new URLSearchParams({ track: currentOnlineTrack });
    const response = await fetch(
      `/api/online-track/landmark/${observation.landmark_id}?${query}`,
      { cache: "no-store" },
    );
    if (!response.ok) throw new Error((await response.json()).error ?? response.statusText);
    const landmark = (await response.json()) as OnlineLandmark;
    const similarities = landmark.observations
      .map((item) => item.similarity)
      .filter((value): value is number => value !== null);
    const meanSimilarity = similarities.length
      ? similarities.reduce((sum, value) => sum + value, 0) / similarities.length
      : null;
    const first = landmark.observations[0];
    const last = landmark.observations.at(-1);
    const lifetime = first && last ? (last.timestamp_ns - first.timestamp_ns) / 1e9 : 0;
    const fields = element<HTMLDListElement>("selection-fields");
    fields.replaceChildren(
      detailRow("Landmark", `#${landmark.landmark_id}`),
      detailRow("Observations", landmark.observation_count.toLocaleString()),
      detailRow("Rows", `${landmark.first_row} → ${landmark.last_row}`),
      detailRow("Lifetime", `${lifetime.toFixed(3)} s`),
      detailRow("Concentration", landmark.concentration.toFixed(5)),
      detailRow("Current cosine", observation.similarity?.toFixed(5) ?? "new"),
      detailRow("Mean cosine", meanSimilarity?.toFixed(5) ?? "—"),
      detailRow("Pixel", `${observation.x.toFixed(1)}, ${observation.y.toFixed(1)}`),
    );
    fields.classList.remove("hidden");
    element("selection-empty").classList.add("hidden");
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  }
}

function showError(message: string) {
  const toast = element("error-toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.setTimeout(() => toast.classList.add("hidden"), 6000);
}

function basename(path: string): string {
  return path.split("/").filter(Boolean).pop() ?? path;
}

function artifactPathLabel(path: string): string {
  const parts = path.split("/").filter(Boolean);
  const leaf = parts.at(-1) ?? path;
  if (["matches", "geometry", "reconstruction", "online-tracks"].includes(leaf) && parts.length > 1) {
    return `${parts.at(-2)}/${leaf}`;
  }
  return leaf;
}

function reconstructionLabel(reconstruction: ReconstructionOption): string {
  return `${artifactPathLabel(reconstruction.path)} · ${reconstruction.registered_frames}/${reconstruction.frames} cams · ${reconstruction.points.toLocaleString()} pts`;
}

function catalogSignature(catalog: RunCatalog): string {
  return JSON.stringify({
    runs: catalog.runs.map((run) => run.id),
    reconstructions: catalog.reconstructions.map((item) => [item.id, item.updated_at]),
    geometry: catalog.geometry.map((item) => [item.artifact_id, item.updated_at]),
    online_tracks: catalog.online_tracks.map((item) => [item.id, item.processed_frames, item.updated_at]),
    detections: catalog.detections.map((item) => [item.id, item.frames, item.updated_at]),
  });
}

function renderDetectionOptions(catalog: RunCatalog, preferred?: string | null) {
  const selector = element<HTMLSelectElement>("detection-selector");
  selector.replaceChildren();
  for (const detection of catalog.detections) {
    const option = document.createElement("option");
    option.value = detection.id;
    option.textContent = `${artifactPathLabel(detection.path)} · ${detection.frames.toLocaleString()} frames`;
    option.title = detection.path;
    selector.append(option);
  }
  currentDetection = catalog.detections.some((item) => item.id === preferred)
    ? preferred!
    : catalog.detections[0]?.id ?? "";
  selector.value = currentDetection;
  selector.disabled = !catalog.detections.length;
}

function renderOnlineTrackOptions(catalog: RunCatalog, preferred?: string | null) {
  const selector = element<HTMLSelectElement>("track-selector");
  selector.replaceChildren();
  for (const track of catalog.online_tracks) {
    const option = document.createElement("option");
    option.value = track.id;
    option.textContent = `${artifactPathLabel(track.path)} · ${track.processed_frames} frames · ${track.landmarks.toLocaleString()} landmarks`;
    option.title = track.path;
    selector.append(option);
  }
  currentOnlineTrack = catalog.online_tracks.some((item) => item.id === preferred)
    ? preferred!
    : catalog.online_tracks[0]?.id ?? "";
  selector.value = currentOnlineTrack;
  selector.disabled = !catalog.online_tracks.length;
}

function renderMatchOptions(catalog: RunCatalog) {
  const selector = element<HTMLSelectElement>("match-selector");
  selector.replaceChildren();
  for (const run of catalog.runs) {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = `${artifactPathLabel(run.run)} · ${matcherLabel(run.status.matcher)} · ${run.status.frames} frames`;
    option.title = run.run;
    selector.append(option);
  }
}

function updateReconstructionOptions(preferred?: string | null) {
  if (!artifactCatalog) return;
  const selector = element<HTMLSelectElement>("reconstruction-selector");
  selector.replaceChildren();
  const compatible = artifactCatalog.reconstructions.filter(
    (reconstruction) => reconstruction.match_run_id === currentRun,
  );
  const other = artifactCatalog.reconstructions.filter(
    (reconstruction) => reconstruction.match_run_id !== currentRun,
  );
  const appendGroup = (label: string, values: ReconstructionOption[]) => {
    if (!values.length) return;
    const group = document.createElement("optgroup");
    group.label = label;
    for (const reconstruction of values) {
      const option = document.createElement("option");
      option.value = reconstruction.id;
      option.textContent = reconstructionLabel(reconstruction);
      option.title = reconstruction.path;
      group.append(option);
    }
    selector.append(group);
  };
  appendGroup("From selected matches", compatible);
  appendGroup("Other reconstructions", other);
  const requested = artifactCatalog.reconstructions.some(
    (reconstruction) => reconstruction.id === preferred,
  )
    ? preferred!
    : compatible[0]?.id ?? artifactCatalog.reconstructions[0]?.id ?? "";
  currentReconstruction = requested;
  selector.value = requested;
  selector.disabled = !artifactCatalog.reconstructions.length;
}

function renderArtifactCatalog(catalog: RunCatalog) {
  element("artifact-counts").textContent =
    `${catalog.detections.length} detection · ${catalog.online_tracks.length} online · ${catalog.runs.length} match · ${catalog.geometry.length} geometry · ${catalog.reconstructions.length} reconstruction`;
  const list = element("geometry-list");
  list.replaceChildren();
  if (!catalog.geometry.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No geometry artifacts found.";
    list.append(empty);
    return;
  }
  for (const geometry of catalog.geometry) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `${artifactPathLabel(geometry.path)} · ${geometry.frames ?? "—"} frames`;
    summary.title = geometry.path;
    const fields = document.createElement("dl");
    fields.append(
      detailRow("Pairs", geometry.pairs?.toLocaleString() ?? "—"),
      detailRow("Verified edges", geometry.verified_edges?.toLocaleString() ?? "—"),
      detailRow("Tracks ≥3", geometry.tracks?.toLocaleString() ?? "—"),
    );
    details.append(summary, fields);
    list.append(details);
  }
}

async function refreshArtifactCatalog() {
  try {
    const response = await fetch("/api/runs", { cache: "no-store" });
    if (!response.ok) return;
    const catalog = (await response.json()) as RunCatalog;
    const signature = catalogSignature(catalog);
    if (signature === artifactCatalogSignature) return;
    const previousRun = currentRun;
    const previousReconstruction = currentReconstruction;
    const previousOnlineTrack = currentOnlineTrack;
    const previousDetection = currentDetection;
    artifactCatalog = catalog;
    artifactCatalogSignature = signature;
    renderMatchOptions(catalog);
    renderArtifactCatalog(catalog);
    renderOnlineTrackOptions(catalog, previousOnlineTrack);
    renderDetectionOptions(catalog, previousDetection);
    currentRun = catalog.runs.some((run) => run.id === previousRun)
      ? previousRun
      : catalog.default;
    element<HTMLSelectElement>("match-selector").value = currentRun;
    updateReconstructionOptions(previousReconstruction);
    if (
      currentMode === "reconstruction" &&
      previousReconstruction &&
      previousReconstruction !== currentReconstruction
    ) {
      void loadReconstruction();
    }
    if (currentMode === "detections" && previousDetection !== currentDetection) {
      currentDetectionRow = 0;
      void loadDetectionFrame(0);
    }
  } catch {
    // The existing catalog stays usable while a file is being finalized.
  }
}

function renderMatcher(matcher: Matcher, runtime?: Matcher | null) {
  const label = matcherLabel(matcher);
  element("matcher-name").textContent = label;
  element("matcher-title").textContent = label;
  element("matcher-description").textContent = matcherDescription(matcher);
  const device = String(runtime?.device ?? matcher.device ?? "CPU").toUpperCase();
  element("matcher-device").textContent = device;
  const fields = element<HTMLDListElement>("matcher-fields");
  const rows: HTMLDivElement[] = [];
  if (matcher.name === "lightglue-superpoint") {
    rows.push(detailRow("Filter threshold", String(matcher.filter_threshold ?? "—")));
    rows.push(detailRow("Layers", String(matcher.n_layers ?? "—")));
    rows.push(detailRow("Weights", String(matcher.weights_sha256 ?? "—").slice(0, 12)));
  } else if (matcher.name === "cosine-mutual-nearest-neighbor") {
    rows.push(detailRow("Min similarity", String(matcher.min_similarity ?? "—")));
    rows.push(detailRow("Selection", "Mutual best"));
  } else if (matcher.ratio !== undefined) {
    rows.push(detailRow("Ratio threshold", String(matcher.ratio)));
  }
  rows.push(detailRow("Score", String(matcher.score ?? "confidence")));
  fields.replaceChildren(...rows);
}

async function refreshOverview() {
  if (currentMode === "tracks" || currentMode === "detections") return;
  if (!currentRun) return;
  try {
    const response = await fetch(`/api/run?run=${encodeURIComponent(currentRun)}`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error((await response.json()).error ?? response.statusText);
    const overview = (await response.json()) as RunOverview;
    const status = overview.status;
    const state = status.running ? "running" : status.phase;
    const statePill = element("run-state");
    statePill.textContent = state;
    statePill.className = `status-pill status-${state}`;
    element("source-name").textContent = overview.source?.split("/").pop() ?? overview.run;
    element("progress-label").textContent = `${status.completed_pairs.toLocaleString()} / ${status.total_pairs.toLocaleString()} pairs`;
    const percent = status.total_pairs ? (100 * status.completed_pairs) / status.total_pairs : 0;
    element("progress-bar").style.width = `${percent}%`;
    renderMatcher(status.matcher, currentPair?.pair_runtime);

    if (!pairInitialized && overview.suggested_pair) {
      const query = new URLSearchParams(location.search);
      const first = Number(query.get("first") ?? overview.suggested_pair.first);
      const second = Number(query.get("second") ?? overview.suggested_pair.second);
      element<HTMLInputElement>("first-row").value = String(first);
      element<HTMLInputElement>("second-row").value = String(second);
      pairInitialized = true;
      if (currentMode === "matrix") void loadMatrix();
      else if (currentMode === "reconstruction") void loadReconstruction();
      else if (currentMode === "pipeline") void loadPipeline();
      else void loadPair(first, second);
    }
  } catch (error) {
    if (!pairInitialized) showError(error instanceof Error ? error.message : String(error));
  }
}

async function loadPair(first: number, second: number) {
  loadController?.abort();
  loadController = new AbortController();
  element("loading").classList.remove("hidden");
  try {
    const response = await fetch(
      `/api/pairs/${first}/${second}?run=${encodeURIComponent(currentRun)}`,
      {
      cache: "no-store",
      signal: loadController.signal,
      },
    );
    if (!response.ok) throw new Error((await response.json()).error ?? response.statusText);
    const pair = (await response.json()) as PairData;
    currentPair = pair;
    await scene.show(pair);
    scene.setLinks(element<HTMLInputElement>("show-lines").checked);
    element("empty-state").classList.add("hidden");
    element("pair-title").textContent = `Rows ${pair.first.row} → ${pair.second.row}`;
    element("match-stat-label").textContent = "Matches";
    element("runtime-stat-label").textContent = "Pair runtime";
    element("gap-stat-label").textContent = "Frame gap";
    element("time-stat-label").textContent = "Time gap";
    element("match-count").textContent = pair.matches.count.toLocaleString();
    element("frame-gap").textContent = String(Math.abs(pair.second.frame_index - pair.first.frame_index));
    element("time-gap").textContent = `${Math.abs(pair.second.timestamp_ns - pair.first.timestamp_ns) / 1e9}s`;
    element("pair-runtime").textContent = String(pair.pair_runtime?.device ?? pair.matcher.device ?? "—").toUpperCase();
    element("first-label").textContent = `Row ${pair.first.row} · source frame ${pair.first.frame_index}`;
    element("second-label").textContent = `Row ${pair.second.row} · source frame ${pair.second.frame_index}`;
    element("selection-empty").classList.remove("hidden");
    element("selection-empty").textContent =
      "Click a colored feature point to inspect its correspondence.";
    element("selection-fields").classList.add("hidden");
    element("matrix-legend").classList.add("hidden");
    element("first-label").classList.remove("hidden");
    element("second-label").classList.remove("hidden");
    renderMatcher(pair.matcher, pair.pair_runtime);
    updateLocation(first, second);
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "AbortError")) {
      showError(error instanceof Error ? error.message : String(error));
    }
  } finally {
    element("loading").classList.add("hidden");
  }
}

async function loadMatrix() {
  loadController?.abort();
  loadController = new AbortController();
  element("loading").textContent = "Loading match matrix…";
  element("loading").classList.remove("hidden");
  try {
    const response = await fetch(`/api/matrix?run=${encodeURIComponent(currentRun)}`, {
      cache: "no-store",
      signal: loadController.signal,
    });
    if (!response.ok) throw new Error((await response.json()).error ?? response.statusText);
    const matrix = (await response.json()) as MatrixData;
    currentPair = null;
    scene.showMatrix(matrix);
    element("empty-state").classList.add("hidden");
    element("pair-title").textContent = `${matrix.size} × ${matrix.size} match matrix`;
    element("match-stat-label").textContent = "Cell maximum";
    element("runtime-stat-label").textContent = "Pairs";
    element("gap-stat-label").textContent = "X axis";
    element("time-stat-label").textContent = "Y axis";
    element("match-count").textContent = `max ${matrix.max_count.toLocaleString()}`;
    element("pair-runtime").textContent = ((matrix.size * (matrix.size - 1)) / 2).toLocaleString();
    element("frame-gap").textContent = "Second row";
    element("time-gap").textContent = "First row";
    element("first-label").classList.add("hidden");
    element("second-label").classList.add("hidden");
    element("matrix-max").textContent = matrix.max_count.toLocaleString();
    element("matrix-legend").classList.remove("hidden");
    element("selection-empty").classList.remove("hidden");
    element("selection-empty").textContent =
      "Click a matrix cell to inspect that image pair. Gray cells are pending.";
    element("selection-fields").classList.add("hidden");
    renderMatcher(matrix.matcher);
    updateLocation();
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "AbortError")) {
      showError(error instanceof Error ? error.message : String(error));
    }
  } finally {
    element("loading").textContent = "Loading pair…";
    element("loading").classList.add("hidden");
  }
}

function renderOnlineContract(overview: OnlineTrackOption) {
  const settings = overview.config.settings ?? {};
  element("matcher-name").textContent = "Online cosine";
  element("matcher-title").textContent = "Spherical landmark mean";
  element("matcher-description").textContent =
    "Each cached SuperPoint descriptor is compared only with landmark means from earlier frames. Assignments are causal and unique within a frame.";
  element("matcher-device").textContent = String(settings.device ?? "cpu").toUpperCase();
  element("matcher-fields").replaceChildren(
    detailRow("Min similarity", String(settings.min_similarity ?? "—")),
    detailRow("Min margin", String(settings.min_margin ?? "—")),
    detailRow("Inactive window", `${settings.max_inactive_frames ?? "—"} frames`),
    detailRow("Association", "Greedy unique landmark"),
  );
}

async function loadDetectionFrame(requestedRow: number) {
  if (!currentDetection) {
    showError("No feature caches are available.");
    return;
  }
  const option = artifactCatalog?.detections.find((item) => item.id === currentDetection);
  if (!option?.frames) return;
  const row = THREE.MathUtils.clamp(Math.round(requestedRow), 0, option.frames - 1);
  loadController?.abort();
  loadController = new AbortController();
  element("loading").textContent = "Seeking video and loading detections…";
  element("loading").classList.remove("hidden");
  try {
    const query = new URLSearchParams({ detection: currentDetection });
    const response = await fetch(`/api/detection/frame/${row}?${query}`, {
      cache: "no-store",
      signal: loadController.signal,
    });
    if (!response.ok) throw new Error((await response.json()).error ?? response.statusText);
    const data = (await response.json()) as DetectionFrame;
    currentPair = null;
    currentOnlineFrame = null;
    currentDetectionFrame = data;
    currentDetectionRow = row;
    await scene.showDetectionFrame(data);
    const frame = data.frame;
    const overview = data.overview;
    const rowInput = element<HTMLInputElement>("track-row");
    rowInput.max = String(Math.max(0, overview.frames - 1));
    rowInput.value = String(row);
    element<HTMLOutputElement>("track-row-value").value = `${row + 1} / ${overview.frames}`;
    element("pair-title").textContent = `Row ${row} · source frame ${frame.frame_index}`;
    element("match-stat-label").textContent = "SuperPoints";
    element("runtime-stat-label").textContent = "Inference";
    element("gap-stat-label").textContent = "Source frame";
    element("time-stat-label").textContent = "Video time";
    element("match-count").textContent = frame.keypoint_count.toLocaleString();
    element("pair-runtime").textContent = `${frame.extraction_ms.toFixed(2)} ms`;
    element("frame-gap").textContent = frame.frame_index.toLocaleString();
    element("time-gap").textContent = `${(frame.timestamp_ns / 1e9).toFixed(3)} s`;
    element("first-label").textContent =
      `Row ${row} · source frame ${frame.frame_index} · ${(frame.timestamp_ns / 1e9).toFixed(3)} s`;
    element("first-label").classList.remove("hidden");
    element("second-label").classList.add("hidden");
    element("matrix-legend").classList.add("hidden");
    element("empty-state").classList.add("hidden");
    element("selection-fields").classList.add("hidden");
    element("selection-empty").classList.remove("hidden");
    element("selection-empty").textContent =
      "Point color shows the SuperPoint score relative to this frame.";
    element("run-state").textContent = overview.selection_complete ? "ready" : "running";
    element("run-state").className =
      `status-pill status-${overview.selection_complete ? "complete" : "running"}`;
    element("source-name").textContent = basename(overview.source ?? overview.path);
    element("progress-label").textContent = `${overview.frames.toLocaleString()} committed frames`;
    element("progress-bar").style.width = overview.selection_complete ? "100%" : "8%";
    element("matcher-name").textContent = "SuperPoint";
    element("matcher-title").textContent = "Live feature detection";
    element("matcher-description").textContent =
      "The browser seeks the compressed source video by timestamp. Only feature coordinates, scores, and descriptors are persisted.";
    element("matcher-device").textContent = String(overview.extractor.device ?? "GPU").toUpperCase();
    element("matcher-fields").replaceChildren(
      detailRow("Frame inference", `${frame.extraction_ms.toFixed(2)} ms`),
      detailRow("Average inference", overview.average_detection_ms === null ? "—" : `${overview.average_detection_ms.toFixed(2)} ms`),
      detailRow("Image storage", "Source video only"),
      detailRow("Feature storage", "Raw float32"),
    );
    updateLocation();
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "AbortError")) {
      showError(error instanceof Error ? error.message : String(error));
    }
  } finally {
    element("loading").textContent = "Loading pair…";
    element("loading").classList.add("hidden");
  }
}

async function loadOnlineFrame(requestedRow: number) {
  if (!currentOnlineTrack) {
    showError("No online track artifacts are available.");
    return;
  }
  const option = artifactCatalog?.online_tracks.find((item) => item.id === currentOnlineTrack);
  if (!option?.processed_frames) {
    showError("This online tracker has not committed a frame yet.");
    return;
  }
  const row = THREE.MathUtils.clamp(Math.round(requestedRow), 0, option.processed_frames - 1);
  loadController?.abort();
  loadController = new AbortController();
  element("loading").textContent = "Loading online tracks…";
  element("loading").classList.remove("hidden");
  try {
    const query = new URLSearchParams({
      track: currentOnlineTrack,
      trail: element<HTMLInputElement>("track-trail").value,
    });
    const response = await fetch(`/api/online-track/frame/${row}?${query}`, {
      cache: "no-store",
      signal: loadController.signal,
    });
    if (!response.ok) throw new Error((await response.json()).error ?? response.statusText);
    const data = (await response.json()) as OnlineTrackFrame;
    currentPair = null;
    currentOnlineFrame = data;
    currentTrackRow = row;
    await scene.showOnlineFrame(data);
    const frame = data.frame;
    const overview = data.overview;
    const rowInput = element<HTMLInputElement>("track-row");
    rowInput.max = String(Math.max(0, overview.processed_frames - 1));
    rowInput.value = String(row);
    element<HTMLOutputElement>("track-row-value").value =
      `${row + 1} / ${overview.processed_frames}`;
    element("pair-title").textContent = `Row ${row} · source frame ${frame.frame_index}`;
    element("match-stat-label").textContent = "Features";
    element("runtime-stat-label").textContent = "Matched";
    element("gap-stat-label").textContent = "New";
    element("time-stat-label").textContent = "Mean cosine";
    element("match-count").textContent = frame.keypoint_count.toLocaleString();
    element("pair-runtime").textContent = frame.matched_count.toLocaleString();
    element("frame-gap").textContent = frame.new_count.toLocaleString();
    element("time-gap").textContent = frame.mean_similarity?.toFixed(4) ?? "—";
    element("first-label").textContent =
      `Row ${row} · source frame ${frame.frame_index} · ${(frame.timestamp_ns / 1e9).toFixed(3)} s`;
    element("first-label").classList.remove("hidden");
    element("second-label").classList.add("hidden");
    element("matrix-legend").classList.add("hidden");
    element("empty-state").classList.add("hidden");
    element("selection-empty").classList.remove("hidden");
    element("selection-empty").textContent =
      "Click a feature to inspect its persistent landmark history.";
    element("selection-fields").classList.add("hidden");
    element("run-state").textContent = overview.running ? "running" : overview.phase;
    element("run-state").className =
      `status-pill status-${overview.running ? "running" : overview.phase}`;
    element("source-name").textContent = basename(overview.source_path ?? overview.path);
    element("progress-label").textContent =
      `${overview.processed_frames.toLocaleString()} / ${overview.total_frames.toLocaleString()} frames`;
    element("progress-bar").style.width =
      `${overview.total_frames ? (100 * overview.processed_frames) / overview.total_frames : 0}%`;
    renderOnlineContract(overview);
    updateLocation();
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "AbortError")) {
      showError(error instanceof Error ? error.message : String(error));
    }
  } finally {
    element("loading").textContent = "Loading pair…";
    element("loading").classList.add("hidden");
  }
}

function stopTrackPlayback() {
  if (trackPlayback !== null) window.clearInterval(trackPlayback);
  trackPlayback = null;
  element("track-play").textContent = "Play";
}

function toggleTrackPlayback() {
  if (trackPlayback !== null) {
    stopTrackPlayback();
    return;
  }
  element("track-play").textContent = "Pause";
  trackPlayback = window.setInterval(() => {
    if (currentMode === "detections") {
      const total = currentDetectionFrame?.overview.frames ?? 0;
      if (currentDetectionRow + 1 >= total) {
        if (currentDetectionFrame?.overview.selection_complete) stopTrackPlayback();
        return;
      }
      void loadDetectionFrame(currentDetectionRow + 1);
      return;
    }
    const total = currentOnlineFrame?.overview.processed_frames ?? 0;
    if (currentTrackRow + 1 >= total) {
      if (!currentOnlineFrame?.overview.running) stopTrackPlayback();
      return;
    }
    void loadOnlineFrame(currentTrackRow + 1);
  }, 250);
}

async function loadReconstruction() {
  if (!currentReconstruction) {
    showError("No reconstruction artifacts are available.");
    return;
  }
  loadController?.abort();
  loadController = new AbortController();
  element("loading").textContent = "Loading reconstruction…";
  element("loading").classList.remove("hidden");
  try {
    const query = new URLSearchParams({
      run: currentRun,
      reconstruction: currentReconstruction,
    });
    const response = await fetch(`/api/reconstruction?${query}`, {
      cache: "no-store",
      signal: loadController.signal,
    });
    if (!response.ok) throw new Error((await response.json()).error ?? response.statusText);
    const reconstruction = (await response.json()) as ReconstructionData;
    currentPair = null;
    scene.showReconstruction(reconstruction);
    const overview = reconstruction.overview;
    element("empty-state").classList.add("hidden");
    element("pair-title").textContent = overview.path.split("/").pop() ?? "Reconstruction";
    element("match-stat-label").textContent = "3D points";
    element("runtime-stat-label").textContent = "Cameras";
    element("gap-stat-label").textContent = "Observations";
    element("time-stat-label").textContent = "Median error";
    element("match-count").textContent = overview.points.toLocaleString();
    element("pair-runtime").textContent = `${overview.registered_frames} / ${overview.frames}`;
    element("frame-gap").textContent = overview.observations.toLocaleString();
    element("time-gap").textContent = overview.median_reprojection_error_px === null
      ? "—"
      : `${overview.median_reprojection_error_px.toFixed(3)} px`;
    element("first-label").classList.add("hidden");
    element("second-label").classList.add("hidden");
    element("matrix-legend").classList.add("hidden");
    element("selection-fields").classList.add("hidden");
    element("selection-empty").classList.remove("hidden");
    element("selection-empty").textContent =
      "Drag to orbit, right-drag to pan, and scroll to move through the reconstruction.";
    updateLocation();
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "AbortError")) {
      showError(error instanceof Error ? error.message : String(error));
    }
  } finally {
    element("loading").textContent = "Loading pair…";
    element("loading").classList.add("hidden");
  }
}

function renderPipelineJob(job: PipelineJob | null) {
  const container = element("pipeline-job");
  if (!job) {
    container.classList.add("hidden");
    element("pipeline-state").textContent = "ready";
    element("pipeline-state").className = "status-pill status-idle";
    element<HTMLButtonElement>("pipeline-start").disabled = false;
    element("pipeline-cancel").classList.add("hidden");
    element("pipeline-open").classList.add("hidden");
    return;
  }
  container.classList.remove("hidden");
  currentPipelineJob = job.job_id;
  const state = element("pipeline-state");
  state.textContent = job.state;
  state.className = `status-pill status-${job.state}`;
  const workflow = job.config.workflow ?? "full";
  element("pipeline-job-name").textContent =
    `${job.name} · ${workflow === "tracks" ? "online tracks" : job.config.matcher} · ${job.config.frames} frames`;
  element("pipeline-updated").textContent = new Date(job.updated_at).toLocaleTimeString();
  const stages = element("pipeline-stages");
  stages.replaceChildren();
  for (const stage of job.stages) {
    const card = document.createElement("article");
    card.className = "pipeline-stage";
    card.dataset.state = stage.state;
    const heading = document.createElement("div");
    heading.className = "pipeline-stage-head";
    const title = document.createElement("strong");
    title.textContent = stage.label;
    const stageState = document.createElement("span");
    stageState.textContent = stage.state;
    heading.append(title, stageState);
    const progress = document.createElement("div");
    progress.className = "pipeline-stage-progress";
    const bar = document.createElement("div");
    const percent = stage.state === "complete"
      ? 100
      : stage.total
        ? Math.min(100, (100 * stage.current) / stage.total)
        : stage.state === "running"
          ? 8
          : 0;
    bar.style.width = `${percent}%`;
    progress.append(bar);
    const detail = document.createElement("p");
    const counts = stage.total === null
      ? ""
      : `${stage.current.toLocaleString()} / ${stage.total.toLocaleString()} · `;
    detail.textContent = `${counts}${stage.detail ?? "Waiting"}`;
    card.append(heading, progress, detail);
    stages.append(card);
  }
  const log = job.last_error
    ? `${job.last_error.code}: ${job.last_error.message}\n\n${job.log_tail ?? ""}`
    : job.log_tail || "Waiting for output…";
  element("pipeline-log").textContent = log;
  const running = job.state === "starting" || job.state === "running"
    || job.worker_alive === true || job.child_alive === true;
  element<HTMLButtonElement>("pipeline-start").disabled = running;
  element("pipeline-cancel").classList.toggle("hidden", !running);
  const open = element<HTMLButtonElement>("pipeline-open");
  const featureStage = job.stages.find((stage) => stage.id === "features");
  const canWatchDetections = workflow === "tracks" && (featureStage?.current ?? 0) > 0;
  open.classList.toggle("hidden", job.state !== "complete" && !canWatchDetections);
  open.textContent = job.state === "complete"
    ? workflow === "tracks" ? "Open tracks" : "Open reconstruction"
    : "Watch detections";
}

async function loadPipeline() {
  try {
    const response = await fetch("/api/pipeline", { cache: "no-store" });
    if (!response.ok) throw new Error((await response.json()).error ?? response.statusText);
    const overview = (await response.json()) as PipelineOverview;
    const source = element<HTMLSelectElement>("pipeline-source");
    const selectedSource = source.value;
    source.replaceChildren();
    for (const path of overview.sources) {
      const option = document.createElement("option");
      option.value = path;
      option.textContent = basename(path);
      option.title = path;
      source.append(option);
    }
    if (overview.sources.includes(selectedSource)) source.value = selectedSource;
    const job = overview.jobs.find((item) => item.job_id === currentPipelineJob)
      ?? overview.jobs[0]
      ?? null;
    renderPipelineJob(job);
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  }
}

function updatePipelineWorkflow() {
  const tracks = element<HTMLSelectElement>("pipeline-workflow").value === "tracks";
  const frames = element<HTMLInputElement>("pipeline-frames");
  if (tracks) frames.removeAttribute("max");
  else frames.max = "1000";
  element<HTMLSelectElement>("pipeline-matcher").disabled = tracks;
  element("pipeline-matcher-field").classList.toggle("disabled-field", tracks);
  element<HTMLButtonElement>("pipeline-start").textContent =
    tracks ? "Run online tracker" : "Run full pipeline";
}

async function startPipeline() {
  const request = {
    source: element<HTMLSelectElement>("pipeline-source").value,
    name: element<HTMLInputElement>("pipeline-name").value,
    frames: Number(element<HTMLInputElement>("pipeline-frames").value),
    stride: Number(element<HTMLInputElement>("pipeline-stride").value),
    matcher: element<HTMLSelectElement>("pipeline-matcher").value,
    device: element<HTMLSelectElement>("pipeline-device").value,
    workflow: element<HTMLSelectElement>("pipeline-workflow").value,
  };
  element<HTMLButtonElement>("pipeline-start").disabled = true;
  try {
    const response = await fetch("/api/pipeline/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error((await response.json()).error ?? response.statusText);
    const job = (await response.json()) as PipelineJob;
    currentPipelineJob = job.job_id;
    renderPipelineJob(job);
    updateLocation();
  } catch (error) {
    element<HTMLButtonElement>("pipeline-start").disabled = false;
    showError(error instanceof Error ? error.message : String(error));
  }
}

async function cancelPipeline() {
  if (!currentPipelineJob) return;
  try {
    const response = await fetch(
      `/api/pipeline/jobs/${encodeURIComponent(currentPipelineJob)}/cancel`,
      { method: "POST" },
    );
    if (!response.ok) throw new Error((await response.json()).error ?? response.statusText);
    await loadPipeline();
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  }
}

async function openPipelineResult() {
  await refreshArtifactCatalog();
  if (!artifactCatalog) return;
  const job = (await (await fetch("/api/pipeline", { cache: "no-store" })).json() as PipelineOverview)
    .jobs.find((item) => item.job_id === currentPipelineJob);
  if (job?.state !== "complete") {
    const detection = artifactCatalog.detections.find(
      (item) => item.path.startsWith(`${job?.paths.run}/cache/`),
    );
    if (!detection) {
      showError("The detector has not committed its first frame yet.");
      return;
    }
    currentDetection = detection.id;
    currentDetectionRow = Math.max(0, detection.frames - 1);
    renderDetectionOptions(artifactCatalog, detection.id);
    setMode("detections");
    return;
  }
  if (job?.config.workflow === "tracks") {
    const tracks = artifactCatalog.online_tracks.find(
      (item) => item.path === job.paths.online_tracks,
    );
    if (!tracks) {
      showError("The online tracks are complete but have not appeared in the artifact catalog yet.");
      return;
    }
    currentOnlineTrack = tracks.id;
    currentTrackRow = 0;
    renderOnlineTrackOptions(artifactCatalog, tracks.id);
    setMode("tracks");
    return;
  }
  const reconstruction = artifactCatalog.reconstructions.find(
    (item) => item.path === job?.paths.reconstruction,
  );
  if (!reconstruction) {
    showError("The reconstruction is complete but has not appeared in the artifact catalog yet.");
    return;
  }
  currentReconstruction = reconstruction.id;
  if (reconstruction.match_run_id) {
    selectRun(reconstruction.match_run_id, reconstruction.id);
  } else {
    updateReconstructionOptions(reconstruction.id);
  }
  setMode("reconstruction");
}

function updateLocation(first?: number, second?: number) {
  const query = new URLSearchParams({ run: currentRun, mode: currentMode });
  if (currentReconstruction) query.set("reconstruction", currentReconstruction);
  if (currentOnlineTrack) query.set("track", currentOnlineTrack);
  if (currentDetection) query.set("detection", currentDetection);
  if (currentMode === "tracks") query.set("trackFrame", String(currentTrackRow));
  if (currentMode === "detections") query.set("detectionFrame", String(currentDetectionRow));
  query.set("frustum", element<HTMLInputElement>("frustum-size").value);
  if (currentPipelineJob) query.set("pipelineJob", currentPipelineJob);
  if (first !== undefined && second !== undefined) {
    query.set("first", String(first));
    query.set("second", String(second));
  }
  history.replaceState(null, "", `?${query}`);
}

function setMode(mode: ViewMode) {
  if ((currentMode === "tracks" || currentMode === "detections") && mode !== currentMode) {
    stopTrackPlayback();
  }
  currentMode = mode;
  element("image-mode").classList.toggle("active", mode === "images");
  element("matrix-mode").classList.toggle("active", mode === "matrix");
  element("detections-mode").classList.toggle("active", mode === "detections");
  element("tracks-mode").classList.toggle("active", mode === "tracks");
  element("reconstruction-mode").classList.toggle("active", mode === "reconstruction");
  element("pipeline-mode").classList.toggle("active", mode === "pipeline");
  element("show-lines").closest("label")?.classList.toggle("hidden", mode !== "images");
  element("show-features").closest("label")?.classList.toggle("hidden", mode !== "images");
  element("match-picker").classList.toggle("hidden", mode === "tracks" || mode === "detections");
  element("track-picker").classList.toggle("hidden", mode !== "tracks");
  element("detection-picker").classList.toggle("hidden", mode !== "detections");
  element("track-controls").classList.toggle("hidden", mode !== "tracks" && mode !== "detections");
  element("track-trail").closest("label")?.classList.toggle("hidden", mode !== "tracks");
  element("pair-form").classList.toggle(
    "hidden",
    mode === "detections" || mode === "tracks" || mode === "reconstruction" || mode === "pipeline",
  );
  element("reconstruction-picker").classList.toggle("hidden", mode !== "reconstruction");
  element("pipeline-panel").classList.toggle("hidden", mode !== "pipeline");
  viewport.classList.toggle("pipeline-view", mode === "pipeline");
  const vizMenu = element<HTMLDetailsElement>("viz-menu");
  vizMenu.classList.toggle("hidden", mode !== "reconstruction");
  if (mode !== "reconstruction") vizMenu.open = false;
  if (mode === "matrix") {
    void loadMatrix();
  } else if (mode === "detections") {
    element("inspector-context").textContent = "Detection frame";
    element("contract-context").textContent = "Feature contract";
    element("selection-context").textContent = "SuperPoint score";
    void loadDetectionFrame(currentDetectionRow);
  } else if (mode === "tracks") {
    element("inspector-context").textContent = "Online frame";
    element("contract-context").textContent = "Association contract";
    element("selection-context").textContent = "Landmark";
    void loadOnlineFrame(currentTrackRow);
  } else if (mode === "reconstruction") {
    void loadReconstruction();
  } else if (mode === "pipeline") {
    element("first-label").classList.add("hidden");
    element("second-label").classList.add("hidden");
    element("matrix-legend").classList.add("hidden");
    element("empty-state").classList.add("hidden");
    void loadPipeline();
    updateLocation();
  } else {
    const first = Number(element<HTMLInputElement>("first-row").value);
    const second = Number(element<HTMLInputElement>("second-row").value);
    void loadPair(first, second);
  }
  if (mode !== "tracks" && mode !== "detections") {
    element("inspector-context").textContent = mode === "reconstruction" ? "Reconstruction" : "Selected pair";
    element("contract-context").textContent = "Matcher contract";
    element("selection-context").textContent = "Feature match";
  }
}

function selectDetection(detectionId: string) {
  currentDetection = detectionId;
  currentDetectionRow = 0;
  currentDetectionFrame = null;
  stopTrackPlayback();
  element<HTMLSelectElement>("detection-selector").value = detectionId;
  if (currentMode === "detections") void loadDetectionFrame(0);
  updateLocation();
}

function selectRun(runId: string, preferredReconstruction?: string | null) {
  currentRun = runId;
  element<HTMLSelectElement>("match-selector").value = runId;
  updateReconstructionOptions(preferredReconstruction);
  if (pairInitialized) {
    if (currentMode === "matrix") void loadMatrix();
    else if (currentMode === "detections") void loadDetectionFrame(currentDetectionRow);
    else if (currentMode === "tracks") void loadOnlineFrame(currentTrackRow);
    else if (currentMode === "reconstruction") void loadReconstruction();
    else if (currentMode === "pipeline") void loadPipeline();
    else {
      const first = Number(element<HTMLInputElement>("first-row").value);
      const second = Number(element<HTMLInputElement>("second-row").value);
      void loadPair(first, second);
    }
  }
  void refreshOverview();
}

function selectOnlineTrack(trackId: string) {
  currentOnlineTrack = trackId;
  currentTrackRow = 0;
  stopTrackPlayback();
  element<HTMLSelectElement>("track-selector").value = trackId;
  if (currentMode === "tracks") void loadOnlineFrame(0);
  updateLocation();
}

async function refreshOnlineLive() {
  if (currentMode !== "tracks" || !currentOnlineTrack) return;
  try {
    const query = new URLSearchParams({ track: currentOnlineTrack });
    const response = await fetch(`/api/online-track?${query}`, { cache: "no-store" });
    if (!response.ok) return;
    const overview = (await response.json()) as OnlineTrackOption;
    const previousTotal = currentOnlineFrame?.overview.processed_frames ?? 0;
    if (currentOnlineFrame) currentOnlineFrame.overview = overview;
    const rowInput = element<HTMLInputElement>("track-row");
    rowInput.max = String(Math.max(0, overview.processed_frames - 1));
    element("run-state").textContent = overview.running ? "running" : overview.phase;
    element("run-state").className =
      `status-pill status-${overview.running ? "running" : overview.phase}`;
    element("progress-label").textContent =
      `${overview.processed_frames.toLocaleString()} / ${overview.total_frames.toLocaleString()} frames`;
    element("progress-bar").style.width =
      `${overview.total_frames ? (100 * overview.processed_frames) / overview.total_frames : 0}%`;
    if (
      trackPlayback !== null
      && currentTrackRow + 1 >= previousTotal
      && overview.processed_frames > previousTotal
    ) {
      void loadOnlineFrame(currentTrackRow + 1);
    }
  } catch {
    // Keep the last committed frame visible during a writer transaction.
  }
}

async function refreshDetectionLive() {
  if (currentMode !== "detections" || !currentDetection) return;
  try {
    const query = new URLSearchParams({ detection: currentDetection });
    const response = await fetch(`/api/detection?${query}`, { cache: "no-store" });
    if (!response.ok) return;
    const overview = (await response.json()) as DetectionOption;
    const previousTotal = currentDetectionFrame?.overview.frames ?? 0;
    const followedLatest = currentDetectionFrame === null || currentDetectionRow + 1 >= previousTotal;
    if (currentDetectionFrame) currentDetectionFrame.overview = overview;
    const rowInput = element<HTMLInputElement>("track-row");
    rowInput.max = String(Math.max(0, overview.frames - 1));
    element("progress-label").textContent = `${overview.frames.toLocaleString()} committed frames`;
    if (overview.frames > previousTotal && followedLatest) {
      void loadDetectionFrame(overview.frames - 1);
    }
  } catch {
    // Keep the last committed detection frame visible during a writer transaction.
  }
}

async function initialize() {
  try {
    const response = await fetch("/api/runs", { cache: "no-store" });
    if (!response.ok) throw new Error((await response.json()).error ?? response.statusText);
    const catalog = (await response.json()) as RunCatalog;
    artifactCatalog = catalog;
    artifactCatalogSignature = catalogSignature(catalog);
    renderMatchOptions(catalog);
    renderArtifactCatalog(catalog);
    const requestedOnlineTrack = new URLSearchParams(location.search).get("track");
    renderOnlineTrackOptions(catalog, requestedOnlineTrack);
    const requestedDetection = new URLSearchParams(location.search).get("detection");
    renderDetectionOptions(catalog, requestedDetection);
    const query = new URLSearchParams(location.search);
    const requestedRun = query.get("run");
    const requestedReconstruction = query.get("reconstruction");
    currentPipelineJob = query.get("pipelineJob") ?? "";
    currentTrackRow = Math.max(0, Number(query.get("trackFrame") ?? "0") || 0);
    currentDetectionRow = Math.max(0, Number(query.get("detectionFrame") ?? "0") || 0);
    const requestedFrustumScale = Number(query.get("frustum") ?? "1");
    const frustumScale = Number.isFinite(requestedFrustumScale)
      ? THREE.MathUtils.clamp(requestedFrustumScale, 0.1, 5)
      : 1;
    element<HTMLInputElement>("frustum-size").value = String(frustumScale);
    element<HTMLOutputElement>("frustum-size-value").value = `${frustumScale.toFixed(1)}×`;
    scene.setFrustumScale(frustumScale);
    const requestedMode = query.get("mode");
    currentMode = requestedMode === "matrix" || requestedMode === "detections" || requestedMode === "tracks" || requestedMode === "reconstruction" || requestedMode === "pipeline"
      ? requestedMode
      : "images";
    if (currentMode === "tracks" && !catalog.online_tracks.length) {
      currentMode = "pipeline";
    }
    if (currentMode === "detections" && !catalog.detections.length) {
      currentMode = "pipeline";
    }
    if (currentMode === "reconstruction" && !catalog.reconstructions.length) {
      currentMode = "pipeline";
    }
    element("image-mode").classList.toggle("active", currentMode === "images");
    element("matrix-mode").classList.toggle("active", currentMode === "matrix");
    element("detections-mode").classList.toggle("active", currentMode === "detections");
    element("tracks-mode").classList.toggle("active", currentMode === "tracks");
    element("reconstruction-mode").classList.toggle("active", currentMode === "reconstruction");
    element("pipeline-mode").classList.toggle("active", currentMode === "pipeline");
    element("show-lines").closest("label")?.classList.toggle("hidden", currentMode !== "images");
    element("show-features").closest("label")?.classList.toggle("hidden", currentMode !== "images");
    element("match-picker").classList.toggle("hidden", currentMode === "tracks" || currentMode === "detections");
    element("track-picker").classList.toggle("hidden", currentMode !== "tracks");
    element("detection-picker").classList.toggle("hidden", currentMode !== "detections");
    element("track-controls").classList.toggle("hidden", currentMode !== "tracks" && currentMode !== "detections");
    element("track-trail").closest("label")?.classList.toggle("hidden", currentMode !== "tracks");
    element("pair-form").classList.toggle(
      "hidden",
      currentMode === "detections" || currentMode === "tracks" || currentMode === "reconstruction" || currentMode === "pipeline",
    );
    element("reconstruction-picker").classList.toggle("hidden", currentMode !== "reconstruction");
    element("viz-menu").classList.toggle("hidden", currentMode !== "reconstruction");
    element("pipeline-panel").classList.toggle("hidden", currentMode !== "pipeline");
    viewport.classList.toggle("pipeline-view", currentMode === "pipeline");
    selectRun(
      catalog.runs.some((run) => run.id === requestedRun) ? requestedRun! : catalog.default,
      requestedReconstruction,
    );
    if (currentMode === "tracks") {
      element("inspector-context").textContent = "Online frame";
      element("contract-context").textContent = "Association contract";
      element("selection-context").textContent = "Landmark";
      void loadOnlineFrame(currentTrackRow);
    } else if (currentMode === "detections") {
      element("inspector-context").textContent = "Detection frame";
      element("contract-context").textContent = "Feature contract";
      element("selection-context").textContent = "SuperPoint score";
      void loadDetectionFrame(currentDetectionRow);
    }
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
  }
}

element<HTMLFormElement>("pair-form").addEventListener("submit", (event) => {
  event.preventDefault();
  setMode("images");
});
element<HTMLInputElement>("show-lines").addEventListener("change", (event) => {
  scene.setLinks((event.target as HTMLInputElement).checked);
});
element<HTMLInputElement>("show-features").addEventListener("change", (event) => {
  scene.setFeatures((event.target as HTMLInputElement).checked);
});
element<HTMLSelectElement>("match-selector").addEventListener("change", (event) => {
  selectRun((event.target as HTMLSelectElement).value);
});
element<HTMLSelectElement>("reconstruction-selector").addEventListener("change", (event) => {
  currentReconstruction = (event.target as HTMLSelectElement).value;
  if (currentMode === "reconstruction") void loadReconstruction();
});
element<HTMLSelectElement>("track-selector").addEventListener("change", (event) => {
  selectOnlineTrack((event.target as HTMLSelectElement).value);
});
element<HTMLSelectElement>("detection-selector").addEventListener("change", (event) => {
  selectDetection((event.target as HTMLSelectElement).value);
});
element<HTMLInputElement>("track-row").addEventListener("input", (event) => {
  stopTrackPlayback();
  const row = Number((event.target as HTMLInputElement).value);
  if (currentMode === "detections") void loadDetectionFrame(row);
  else void loadOnlineFrame(row);
});
element<HTMLInputElement>("track-trail").addEventListener("input", (event) => {
  const value = Number((event.target as HTMLInputElement).value);
  element<HTMLOutputElement>("track-trail-value").value = String(value);
});
element<HTMLInputElement>("track-trail").addEventListener("change", () => {
  if (currentMode === "tracks") void loadOnlineFrame(currentTrackRow);
});
element("track-previous").addEventListener("click", () => {
  stopTrackPlayback();
  if (currentMode === "detections") void loadDetectionFrame(currentDetectionRow - 1);
  else void loadOnlineFrame(currentTrackRow - 1);
});
element("track-next").addEventListener("click", () => {
  stopTrackPlayback();
  if (currentMode === "detections") void loadDetectionFrame(currentDetectionRow + 1);
  else void loadOnlineFrame(currentTrackRow + 1);
});
element("track-play").addEventListener("click", toggleTrackPlayback);
element<HTMLInputElement>("frustum-size").addEventListener("input", (event) => {
  const value = Number((event.target as HTMLInputElement).value);
  scene.setFrustumScale(value);
  element<HTMLOutputElement>("frustum-size-value").value = `${value.toFixed(1)}×`;
  updateLocation();
});
element<HTMLInputElement>("point-size").addEventListener("input", (event) => {
  const value = Number((event.target as HTMLInputElement).value);
  scene.setReconstructionPointSize(value);
  element<HTMLOutputElement>("point-size-value").value = `${value.toFixed(1)} px`;
});
element<HTMLInputElement>("point-opacity").addEventListener("input", (event) => {
  const value = Number((event.target as HTMLInputElement).value);
  scene.setReconstructionPointOpacity(value);
  element<HTMLOutputElement>("point-opacity-value").value = `${Math.round(value * 100)}%`;
});
element<HTMLInputElement>("camera-opacity").addEventListener("input", (event) => {
  const value = Number((event.target as HTMLInputElement).value);
  scene.setCameraOpacity(value);
  element<HTMLOutputElement>("camera-opacity-value").value = `${Math.round(value * 100)}%`;
});
for (const [id, target] of [
  ["viz-points", "points"],
  ["viz-cameras", "cameras"],
  ["viz-trajectory", "trajectory"],
  ["viz-axes", "axes"],
  ["viz-grid", "grid"],
] as const) {
  element<HTMLInputElement>(id).addEventListener("change", (event) => {
    scene.setReconstructionVisibility(target, (event.target as HTMLInputElement).checked);
  });
}
element("viz-reset").addEventListener("click", () => {
  const ranges = [
    ["point-size", 3, "point-size-value", "3.0 px"],
    ["point-opacity", 1, "point-opacity-value", "100%"],
    ["frustum-size", 1, "frustum-size-value", "1.0×"],
    ["camera-opacity", 0.8, "camera-opacity-value", "80%"],
  ] as const;
  for (const [inputId, value, outputId, label] of ranges) {
    element<HTMLInputElement>(inputId).value = String(value);
    element<HTMLOutputElement>(outputId).value = label;
  }
  const toggles = [
    ["viz-points", "points", true],
    ["viz-cameras", "cameras", true],
    ["viz-trajectory", "trajectory", true],
    ["viz-axes", "axes", true],
    ["viz-grid", "grid", false],
  ] as const;
  for (const [inputId, target, visible] of toggles) {
    element<HTMLInputElement>(inputId).checked = visible;
    scene.setReconstructionVisibility(target, visible);
  }
  scene.setReconstructionPointSize(3);
  scene.setReconstructionPointOpacity(1);
  scene.setFrustumScale(1);
  scene.setCameraOpacity(0.8);
  updateLocation();
});
element("fit-view").addEventListener("click", () => scene.fit());
element<HTMLFormElement>("pipeline-form").addEventListener("submit", (event) => {
  event.preventDefault();
  void startPipeline();
});
element<HTMLSelectElement>("pipeline-workflow").addEventListener(
  "change",
  updatePipelineWorkflow,
);
element("pipeline-cancel").addEventListener("click", () => void cancelPipeline());
element("pipeline-open").addEventListener("click", () => void openPipelineResult());
element("image-mode").addEventListener("click", () => setMode("images"));
element("matrix-mode").addEventListener("click", () => setMode("matrix"));
element("detections-mode").addEventListener("click", () => setMode("detections"));
element("tracks-mode").addEventListener("click", () => setMode("tracks"));
element("reconstruction-mode").addEventListener("click", () => setMode("reconstruction"));
element("pipeline-mode").addEventListener("click", () => setMode("pipeline"));

updatePipelineWorkflow();
void initialize();
window.setInterval(refreshOverview, 1000);
window.setInterval(refreshArtifactCatalog, 5000);
window.setInterval(refreshOnlineLive, 1000);
window.setInterval(refreshDetectionLive, 500);
window.setInterval(() => {
  if (currentMode === "pipeline") void loadPipeline();
}, 1000);
