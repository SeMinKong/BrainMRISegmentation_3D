import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Expand, Focus, LoaderCircle, RotateCcw, Rotate3D } from "lucide-react";
import { api, number } from "../api";
import type { DiffMeshData, Label, MeshData } from "../api";
import { DIFF } from "../labels";
import { getCamera, holdAutoRotate, isAutoRotating, publishCamera, setAutoRotate, subscribeAutoRotate, subscribeCamera } from "../cameraSync";
import type { CameraState } from "../cameraSync";

export type RegionInfo = { id: number; name: string; volume?: number | null };

type Props = {
  caseId: string;
  segmentation: string;
  labels: Label[];
  visibleLabels: number[];
  isolated: boolean;
  brainOpacity: number;
  exploded: boolean;
  resetKey: number;
  compact?: boolean;
  /** Label drawn inside the viewer (e.g. "판독 마스크" when two viewers sit side by side). */
  title?: string;
  /** Plain names (and volumes when known) for the floating region labels and hover tooltip. */
  regions?: RegionInfo[];
  /**
   * Difference mode: the reference regions turn grey and two extra surfaces show where the prediction
   * `prediction` missed (blue) or added (red) tumour. Labels MISSED/EXTRA identify them in `regions`.
   */
  diff?: { prediction: string; showMissed: boolean; showExtra: boolean } | null;
  /** Bumped by the parent to frame the tumour (the "종양 위치로" button). */
  focusKey?: number;
};

export const MISSED = -1;
export const EXTRA = -2;

type RegionMesh = { mesh: THREE.Mesh; label: number; offset: THREE.Vector3; center: THREE.Vector3 };

// Two viewers of the same case (brain+tumour and tumour-only) share one download.
const meshCache = new Map<string, Promise<MeshData>>();
function loadMesh(caseId: string, segmentation: string): Promise<MeshData> {
  const key = `${caseId}|${segmentation}`;
  let pending = meshCache.get(key);
  if (!pending) {
    pending = api<MeshData>(`/cases/${caseId}/mesh?segmentation=${encodeURIComponent(segmentation)}`).catch((error) => {
      meshCache.delete(key);
      throw error;
    });
    meshCache.set(key, pending);
    while (meshCache.size > 6) meshCache.delete(meshCache.keys().next().value!);
  }
  return pending;
}
const diffCache = new Map<string, Promise<DiffMeshData>>();
function loadDiff(caseId: string, prediction: string): Promise<DiffMeshData> {
  const key = `${caseId}|${prediction}`;
  let pending = diffCache.get(key);
  if (!pending) {
    pending = api<DiffMeshData>(`/cases/${caseId}/diff-mesh?prediction=${encodeURIComponent(prediction)}`).catch((error) => {
      diffCache.delete(key);
      throw error;
    });
    diffCache.set(key, pending);
    while (diffCache.size > 6) diffCache.delete(diffCache.keys().next().value!);
  }
  return pending;
}

// RAS axes for the orientation compass: +x right, +y anterior, +z superior.
const COMPASS = [
  { axis: new THREE.Vector3(1, 0, 0), letter: "R" },
  { axis: new THREE.Vector3(-1, 0, 0), letter: "L" },
  { axis: new THREE.Vector3(0, 1, 0), letter: "A" },
  { axis: new THREE.Vector3(0, -1, 0), letter: "P" },
  { axis: new THREE.Vector3(0, 0, 1), letter: "S" },
  { axis: new THREE.Vector3(0, 0, -1), letter: "I" },
];

export default function MeshViewer(props: Props) {
  const { caseId, segmentation, labels, visibleLabels, isolated, brainOpacity, exploded, resetKey, compact, title, regions, diff, focusKey } = props;
  const host = useRef<HTMLDivElement>(null);
  const root = useRef<HTMLDivElement>(null);
  const labelLayer = useRef<HTMLDivElement>(null);
  const compassLayer = useRef<HTMLDivElement>(null);
  const tip = useRef<HTMLDivElement>(null);
  // Read inside the scene effect without re-creating the scene when toggles change.
  const isolatedRef = useRef(isolated);
  isolatedRef.current = isolated;
  const explodedRef = useRef(exploded);
  explodedRef.current = exploded;
  const regionsRef = useRef(regions);
  regionsRef.current = regions;
  const runtime = useRef<{
    brain?: THREE.Mesh;
    meshes: RegionMesh[];
    reset: () => void;
    focus: (tumorOnly: boolean) => void;
  } | null>(null);
  const [data, setData] = useState<(MeshData & { diff?: DiffMeshData }) | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  // Auto-rotation is a property of the shared camera, not of this viewer, so it follows tab switches.
  const [autoRotate, setAutoRotateState] = useState(() => isAutoRotating(caseId));
  useEffect(() => {
    setAutoRotateState(isAutoRotating(caseId));
    return subscribeAutoRotate(caseId, setAutoRotateState);
  }, [caseId]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    setData(null);
    const prediction = diff?.prediction;
    Promise.all([loadMesh(caseId, segmentation), prediction ? loadDiff(caseId, prediction) : Promise.resolve(undefined)])
      .then(([mesh, difference]) => {
        if (!cancelled) setData(difference ? { ...mesh, diff: difference } : mesh);
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [caseId, segmentation, diff?.prediction]);

  useEffect(() => {
    if (!data || !host.current) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    } catch {
      setError("WebGL을 사용할 수 없습니다. 브라우저의 하드웨어 가속을 확인하세요. 단면 뷰어는 계속 사용할 수 있습니다.");
      return;
    }
    const element = host.current;
    if (tip.current) tip.current.hidden = true; // a rebuilt scene must not keep the previous mask's tooltip
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0e141b, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.setAttribute("aria-label", "회전과 확대가 가능한 3D 뇌 및 종양 모델");
    renderer.domElement.setAttribute("role", "img");
    element.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 10000);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    scene.add(new THREE.HemisphereLight(0xf4efe9, 0x3a4652, 1.6));
    const light = new THREE.DirectionalLight(0xffffff, 2.6);
    light.position.set(120, -160, 280);
    scene.add(light);
    const rim = new THREE.DirectionalLight(0xbfd8ea, 1.4);
    rim.position.set(-160, 120, 40);
    scene.add(rim);
    const fill = new THREE.DirectionalLight(0xffe9dc, 0.8);
    fill.position.set(80, 120, -120);
    scene.add(fill);
    const center = new THREE.Vector3(...(data.center as [number, number, number]));
    const objects: THREE.Mesh[] = [];
    const makeMesh = (vertices: number[], faces: number[], color: string, brain: boolean, muted = false) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
      geometry.setIndex(faces);
      geometry.computeVertexNormals();
      geometry.translate(-center.x, -center.y, -center.z);
      // Brain: matte tissue tone with a faint sheen so cortical folds catch the light; tumour regions stay saturated.
      const material = new THREE.MeshPhysicalMaterial({
        color,
        roughness: brain ? 0.58 : 0.38,
        metalness: 0.0,
        transparent: brain || muted,
        opacity: brain ? 0.32 : muted ? 0.3 : 1,
        depthWrite: !brain && !muted,
        side: brain ? THREE.FrontSide : THREE.DoubleSide,
        clearcoat: brain ? 0.12 : 0.3,
        clearcoatRoughness: 0.6,
        sheen: brain ? 0.4 : 0,
        sheenColor: new THREE.Color("#f3d9d2"),
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.renderOrder = brain ? 2 : 1;
      objects.push(mesh);
      scene.add(mesh);
      return mesh;
    };
    const brain = data.brain?.vertices.length ? makeMesh(data.brain.vertices, data.brain.faces, "#d7b3aa", true) : undefined;
    const present = data.regions.filter((r) => r.vertices.length > 0);
    const tumourBox = new THREE.Box3();
    // In difference mode the reference regions are a quiet grey context; the two difference surfaces carry the colour.
    const differing = data.diff;
    const surfaces: { vertices: number[]; faces: number[]; label: number; color: string; muted: boolean }[] = [
      ...present.map((r) => ({ ...r, color: differing ? DIFF.overlap.color : labels.find((l) => l.id === r.label)?.color || "#5eead4", muted: !!differing })),
      ...(differing ? [
        { ...differing.missed, label: MISSED, color: DIFF.missed.color, muted: false },
        { ...differing.extra, label: EXTRA, color: DIFF.extra.color, muted: false },
      ].filter((s) => s.vertices.length > 0) : []),
    ];
    const meshes: RegionMesh[] = surfaces.map((r) => {
      const mesh = makeMesh(r.vertices, r.faces, r.color, false, r.muted);
      if (r.muted) mesh.renderOrder = 0;
      mesh.geometry.computeBoundingBox();
      const box = mesh.geometry.boundingBox!;
      tumourBox.union(box);
      return { mesh, label: r.label, offset: new THREE.Vector3(), center: box.getCenter(new THREE.Vector3()) };
    });
    // Explode radially away from the tumour's centre so nested regions separate; a fallback axis handles
    // regions that sit exactly on the centre.
    const tumourCenter = tumourBox.isEmpty() ? new THREE.Vector3() : tumourBox.getCenter(new THREE.Vector3());
    const tumourExtent = tumourBox.isEmpty() ? 40 : Math.max(...tumourBox.getSize(new THREE.Vector3()).toArray());
    meshes.forEach((item, index) => {
      if (item.label < 0) return; // difference surfaces stay where the disagreement is
      const direction = item.center.clone().sub(tumourCenter);
      if (direction.length() < 1) direction.set(Math.cos((index / meshes.length) * Math.PI * 2), Math.sin((index / meshes.length) * Math.PI * 2), 0);
      direction.normalize();
      item.offset.copy(direction.multiplyScalar(tumourExtent * 0.45 + 18));
    });
    const bounding = new THREE.Box3();
    objects.forEach((o) => bounding.expandByObject(o));
    const diameter = Math.max(...bounding.getSize(new THREE.Vector3()).toArray(), 90);
    const reset = () => {
      camera.position.set(diameter * 1.1, -diameter * 1.7, diameter * 0.75);
      controls.target.set(0, 0, 0);
      controls.update();
    };
    controls.minDistance = diameter * 0.15;
    controls.maxDistance = diameter * 6;
    // Camera sharing: publish our moves, follow everyone else's, and start from the group's saved view.
    const me = Symbol("mesh-viewer");
    let applying = false;
    const apply = (state: CameraState) => {
      applying = true;
      camera.position.fromArray(state.position);
      camera.up.fromArray(state.up);
      controls.target.fromArray(state.target);
      controls.update();
      applying = false;
    };
    controls.addEventListener("change", () => {
      if (applying) return;
      publishCamera(caseId, {
        position: camera.position.toArray() as [number, number, number],
        target: controls.target.toArray() as [number, number, number],
        up: camera.up.toArray() as [number, number, number],
      }, me);
    });
    const unsubscribe = subscribeCamera(caseId, (state, source) => {
      if (source !== me) apply(state);
    });
    // Dragging pauses the shared auto-rotation for every viewer in the group.
    const onStart = () => holdAutoRotate(caseId, true);
    const onEnd = () => holdAutoRotate(caseId, false);
    controls.addEventListener("start", onStart);
    controls.addEventListener("end", onEnd);
    const focus = (tumorOnly: boolean) => {
      const box = new THREE.Box3();
      if (!tumorOnly) {
        reset();
        return;
      }
      meshes.filter((item) => item.mesh.visible).forEach((item) => box.expandByObject(item.mesh));
      if (box.isEmpty()) return;
      const target = box.getCenter(new THREE.Vector3());
      const extent = Math.max(...box.getSize(new THREE.Vector3()).toArray(), 15) * (explodedRef.current ? 1.25 : 1);
      const direction = camera.position.clone().sub(controls.target).normalize();
      const distance = (extent / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)))) * 1.65;
      camera.position.copy(target.clone().add(direction.multiplyScalar(distance)));
      controls.target.copy(target);
      controls.update();
    };
    runtime.current = { brain, meshes, reset, focus };
    const saved = getCamera(caseId);
    if (saved) apply(saved);
    else focus(isolatedRef.current);

    // Hover: pick the region under the pointer, lift it slightly, and name it in a tooltip.
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let hovered: RegionMesh | null = null;
    const setHover = (next: RegionMesh | null) => {
      if (hovered === next) return;
      if (hovered) (hovered.mesh.material as THREE.MeshPhysicalMaterial).emissive.setHex(0x000000);
      hovered = next;
      if (hovered) (hovered.mesh.material as THREE.MeshPhysicalMaterial).emissive.setHex(0x333333);
      renderer.domElement.style.cursor = hovered ? "pointer" : "";
    };
    const onMove = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(meshes.filter((m) => m.mesh.visible).map((m) => m.mesh), false)[0];
      setHover(hit ? meshes.find((m) => m.mesh === hit.object) ?? null : null);
      if (tip.current) {
        if (hovered) {
          const info = regionsRef.current?.find((r) => r.id === hovered!.label);
          tip.current.hidden = false;
          tip.current.style.left = `${event.clientX - rect.left + 14}px`;
          tip.current.style.top = `${event.clientY - rect.top + 14}px`;
          tip.current.textContent = info ? `${info.name}${info.volume != null ? ` · ${number(info.volume, 1)} mL` : ""}`
            : hovered.label === MISSED ? DIFF.missed.name : hovered.label === EXTRA ? DIFF.extra.name : `라벨 ${hovered.label}`;
        } else tip.current.hidden = true;
      }
    };
    const onLeave = () => {
      setHover(null);
      if (tip.current) tip.current.hidden = true;
    };
    renderer.domElement.addEventListener("pointermove", onMove);
    renderer.domElement.addEventListener("pointerleave", onLeave);

    const resize = new ResizeObserver(() => {
      const width = element.clientWidth, height = element.clientHeight;
      if (!width || !height) return;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    });
    resize.observe(element);

    // Overlay helpers: floating region labels (while exploded) and the RAS compass, projected each frame.
    const project = (point: THREE.Vector3) => {
      const p = point.clone().project(camera);
      const width = element.clientWidth, height = element.clientHeight;
      return { x: (p.x + 1) / 2 * width, y: (1 - p.y) / 2 * height, front: p.z < 1 };
    };
    let explodeT = explodedRef.current ? 1 : 0;
    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      const targetT = explodedRef.current ? 1 : 0;
      explodeT += (targetT - explodeT) * 0.14;
      if (Math.abs(targetT - explodeT) < 0.002) explodeT = targetT;
      meshes.forEach((item) => item.mesh.position.copy(item.offset).multiplyScalar(explodeT));
      controls.update();
      renderer.render(scene, camera);
      if (labelLayer.current) {
        const showLabels = explodedRef.current || explodeT > 0.5; // appear at once, positions follow the animation
        const children = labelLayer.current.children;
        meshes.forEach((item, index) => {
          const node = children[index] as HTMLElement | undefined;
          if (!node) return;
          const visible = showLabels && item.mesh.visible;
          node.hidden = !visible;
          if (!visible) return;
          const p = project(item.center.clone().add(item.mesh.position));
          node.style.transform = `translate(${p.x}px, ${p.y}px) translate(-50%, -140%)`;
          node.style.opacity = p.front ? "1" : "0";
        });
      }
      if (compassLayer.current) {
        const nodes = compassLayer.current.children;
        const forward = camera.getWorldDirection(new THREE.Vector3());
        COMPASS.forEach((entry, index) => {
          const node = nodes[index] as HTMLElement | undefined;
          if (!node) return;
          const dir = entry.axis.clone().applyQuaternion(camera.quaternion.clone().invert());
          const depth = -entry.axis.dot(forward); // > 0 when the axis points toward the viewer
          node.style.transform = `translate(${dir.x * 22}px, ${-dir.y * 22}px)`;
          node.style.opacity = depth > 0 ? "1" : "0.35";
          node.style.zIndex = depth > 0 ? "2" : "1";
        });
      }
    };
    animate();
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      unsubscribe();
      controls.removeEventListener("start", onStart);
      controls.removeEventListener("end", onEnd);
      renderer.domElement.removeEventListener("pointermove", onMove);
      renderer.domElement.removeEventListener("pointerleave", onLeave);
      controls.dispose();
      objects.forEach((o) => {
        o.geometry.dispose();
        (o.material as THREE.Material).dispose();
      });
      renderer.dispose();
      renderer.forceContextLoss();
      element.replaceChildren();
      runtime.current = null;
    };
  }, [data, labels]);

  useEffect(() => {
    const r = runtime.current;
    if (!r) return;
    if (r.brain) {
      r.brain.visible = !isolated;
      (r.brain.material as THREE.MeshPhysicalMaterial).opacity = brainOpacity;
    }
    r.meshes.forEach(({ mesh, label }) => {
      mesh.visible = label === MISSED ? !!diff?.showMissed : label === EXTRA ? !!diff?.showExtra : visibleLabels.includes(label);
    });
  }, [data, visibleLabels, isolated, brainOpacity, labels, diff?.showMissed, diff?.showExtra]);
  useEffect(() => {
    if (focusKey) runtime.current?.focus(true);
  }, [focusKey]);
  // Re-frame only when the user toggles tumour-only or explode; a rebuilt scene keeps the shared camera instead.
  useEffect(() => {
    runtime.current?.focus(isolated);
  }, [isolated, exploded]);
  useEffect(() => {
    runtime.current?.reset();
  }, [resetKey]);
  const fullScreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void root.current?.requestFullscreen().catch(() => setError("이 브라우저에서는 전체 화면을 지원하지 않습니다."));
  };
  const presentIds = [
    ...(data?.regions ?? []).filter((r) => r.vertices.length > 0).map((r) => r.label),
    ...(data?.diff ? [MISSED, EXTRA].filter((id) => (id === MISSED ? data.diff!.missed : data.diff!.extra).vertices.length > 0) : []),
  ];
  const regionLabels = presentIds.map((id) => regions?.find((x) => x.id === id) ?? { id, name: id === MISSED ? DIFF.missed.name : id === EXTRA ? DIFF.extra.name : `라벨 ${id}` });
  const swatch = (id: number) => (id === MISSED ? DIFF.missed.color : id === EXTRA ? DIFF.extra.color : data?.diff ? DIFF.overlap.color : labels.find((l) => l.id === id)?.color);

  return (
    <div ref={root} className={`mesh-viewer ${compact ? "compact" : ""}`}>
      {title && <span className="mesh-title">{title}</span>}
      <div className="viewer-buttons">
        <button className="neu-icon" onClick={() => setAutoRotate(caseId, !isAutoRotating(caseId))} aria-label="자동 회전" aria-pressed={autoRotate} title="자동 회전">
          <Rotate3D size={16} />
        </button>
        <button className="neu-icon" onClick={() => runtime.current?.reset()} aria-label="3D 시점 초기화" title="시점 초기화">
          <RotateCcw size={16} />
        </button>
        <button className="neu-icon" onClick={fullScreen} aria-label="3D 전체 화면" title="전체 화면">
          <Expand size={16} />
        </button>
      </div>
      <div ref={host} className="canvas-host" />
      <div ref={labelLayer} className="mesh-labels" aria-hidden="true">
        {regionLabels.map((r) => (
          <span key={r.id} className="mesh-label" hidden>
            <span className="swatch" style={{ background: swatch(r.id) }} />
            {r.name}
            {r.volume != null && <small> {number(r.volume, 1)} mL</small>}
          </span>
        ))}
      </div>
      <div ref={tip} className="mesh-tip" hidden />
      <div ref={compassLayer} className="compass" aria-label="방향: R 오른쪽, L 왼쪽, A 앞, P 뒤, S 위, I 아래" title="R 오른쪽 · L 왼쪽 · A 앞 · P 뒤 · S 위 · I 아래">
        {COMPASS.map((entry) => (
          <span key={entry.letter}>{entry.letter}</span>
        ))}
      </div>
      {loading && (
        <div className="viewer-message">
          <LoaderCircle className="spin" size={26} />
          <span>3D 표면을 구성하고 있습니다</span>
        </div>
      )}
      {error && (
        <div className="viewer-message error">
          <Focus size={26} />
          <span>{error}</span>
        </div>
      )}
      {!loading && !error && data && !data.diff && data.regions.every((r) => !r.vertices.length) && (
        <div className="viewer-message">
          <span>선택한 결과에 종양 마스크가 없습니다.</span>
        </div>
      )}
      <div className="viewer-help">드래그 회전 · 휠 확대 · 우클릭 이동 · 영역에 마우스를 올리면 이름이 보입니다</div>
    </div>
  );
}
