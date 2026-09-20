import * as THREE from "three";
import { MapControls } from "three/addons/controls/MapControls.js";
import "./style.css";

type Matcher = Record<string, unknown> & { name: string; score?: string; device?: string };

type FrameInfo = {
  row: number;
  frame_index: number;
  timestamp_ns: number;
  width: number;
  height: number;
  keypoint_count: number;
  image_url: string;
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

class PairScene {
  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.OrthographicCamera;
  private controls: MapControls;
  private content = new THREE.Group();
  private points: THREE.Points | null = null;
  private links: THREE.LineSegments | null = null;
  private highlight = new THREE.Group();
  private extents: { width: number; height: number; centerX: number } | null = null;
  private pair: PairData | null = null;
  private selectionHandler: (index: number) => void;

  constructor(container: HTMLElement, selectionHandler: (index: number) => void) {
    this.container = container;
    this.selectionHandler = selectionHandler;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.prepend(this.renderer.domElement);

    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 3000);
    this.camera.position.set(0, 0, 1000);
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
  }

  private animate = () => {
    requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

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

  private clear() {
    for (const object of [...this.content.children, ...this.highlight.children]) {
      object.removeFromParent();
      const renderable = object as THREE.Mesh;
      renderable.geometry?.dispose();
      const materials = Array.isArray(renderable.material) ? renderable.material : [renderable.material];
      for (const material of materials) {
        if (!material) continue;
        const map = (material as THREE.MeshBasicMaterial).map;
        map?.dispose();
        material.dispose();
      }
    }
    this.points = null;
    this.links = null;
  }

  async show(pair: PairData) {
    this.clear();
    this.pair = pair;
    const [firstTexture, secondTexture] = await Promise.all([
      this.texture(pair.first.image_url),
      this.texture(pair.second.image_url),
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
    this.content.add(this.links, this.points);
    this.fit();
  }

  setLinks(visible: boolean) {
    if (this.links) this.links.visible = visible;
  }

  fit() {
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
    if (!this.points || !this.pair) return;
    const bounds = this.renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
      -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 8 / this.camera.zoom };
    raycaster.setFromCamera(mouse, this.camera);
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
}

const viewport = element<HTMLElement>("viewport");
let currentPair: PairData | null = null;
let initialized = false;
let loadController: AbortController | null = null;

const scene = new PairScene(viewport, (index) => {
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
});

function showError(message: string) {
  const toast = element("error-toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.setTimeout(() => toast.classList.add("hidden"), 6000);
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
  try {
    const response = await fetch("/api/run", { cache: "no-store" });
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

    if (!initialized && overview.suggested_pair) {
      const query = new URLSearchParams(location.search);
      const first = Number(query.get("first") ?? overview.suggested_pair.first);
      const second = Number(query.get("second") ?? overview.suggested_pair.second);
      element<HTMLInputElement>("first-row").value = String(first);
      element<HTMLInputElement>("second-row").value = String(second);
      initialized = true;
      void loadPair(first, second);
    }
  } catch (error) {
    if (!initialized) showError(error instanceof Error ? error.message : String(error));
  }
}

async function loadPair(first: number, second: number) {
  loadController?.abort();
  loadController = new AbortController();
  element("loading").classList.remove("hidden");
  try {
    const response = await fetch(`/api/pairs/${first}/${second}`, {
      cache: "no-store",
      signal: loadController.signal,
    });
    if (!response.ok) throw new Error((await response.json()).error ?? response.statusText);
    const pair = (await response.json()) as PairData;
    currentPair = pair;
    await scene.show(pair);
    scene.setLinks(element<HTMLInputElement>("show-lines").checked);
    element("empty-state").classList.add("hidden");
    element("pair-title").textContent = `Rows ${pair.first.row} → ${pair.second.row}`;
    element("match-count").textContent = pair.matches.count.toLocaleString();
    element("frame-gap").textContent = String(Math.abs(pair.second.frame_index - pair.first.frame_index));
    element("time-gap").textContent = `${Math.abs(pair.second.timestamp_ns - pair.first.timestamp_ns) / 1e9}s`;
    element("pair-runtime").textContent = String(pair.pair_runtime?.device ?? pair.matcher.device ?? "—").toUpperCase();
    element("first-label").textContent = `Row ${pair.first.row} · source frame ${pair.first.frame_index}`;
    element("second-label").textContent = `Row ${pair.second.row} · source frame ${pair.second.frame_index}`;
    element("selection-empty").classList.remove("hidden");
    element("selection-fields").classList.add("hidden");
    renderMatcher(pair.matcher, pair.pair_runtime);
    history.replaceState(null, "", `?first=${first}&second=${second}`);
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "AbortError")) {
      showError(error instanceof Error ? error.message : String(error));
    }
  } finally {
    element("loading").classList.add("hidden");
  }
}

element<HTMLFormElement>("pair-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const first = Number(element<HTMLInputElement>("first-row").value);
  const second = Number(element<HTMLInputElement>("second-row").value);
  void loadPair(first, second);
});
element<HTMLInputElement>("show-lines").addEventListener("change", (event) => {
  scene.setLinks((event.target as HTMLInputElement).checked);
});
element("fit-view").addEventListener("click", () => scene.fit());

void refreshOverview();
window.setInterval(refreshOverview, 1000);
