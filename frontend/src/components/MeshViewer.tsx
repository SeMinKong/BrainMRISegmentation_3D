import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Expand, Focus, LoaderCircle, RotateCcw, Rotate3D } from "lucide-react";
import { api } from "../api";
import type { Label, MeshData } from "../api";

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
};
export default function MeshViewer(props: Props) {
  const {
    caseId,
    segmentation,
    labels,
    visibleLabels,
    isolated,
    brainOpacity,
    exploded,
    resetKey,
    compact,
  } = props;
  const host = useRef<HTMLDivElement>(null);
  const root = useRef<HTMLDivElement>(null);
  const runtime = useRef<{
    brain?: THREE.Mesh;
    meshes: { mesh: THREE.Mesh; label: number; offset: THREE.Vector3 }[];
    reset: () => void;
    focus: (tumorOnly: boolean) => void;
  } | null>(null);
  const [data, setData] = useState<MeshData | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);
  const rotate = useRef(false);
  rotate.current = autoRotate;
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    setData(null);
    api<MeshData>(
      `/cases/${caseId}/mesh?segmentation=${encodeURIComponent(segmentation)}`,
      { signal: controller.signal },
    )
      .then(setData)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [caseId, segmentation]);
  useEffect(() => {
    if (!data || !host.current) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
    } catch {
      setError(
        "WebGL을 사용할 수 없습니다. 브라우저의 하드웨어 가속을 확인하세요. 단면 뷰어는 계속 사용할 수 있습니다.",
      );
      return;
    }
    const element = host.current;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0e141b, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.setAttribute(
      "aria-label",
      "회전과 확대가 가능한 3D 뇌 및 종양 모델",
    );
    renderer.domElement.setAttribute("role", "img");
    element.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 10000);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.autoRotateSpeed = 0.65;
    scene.add(new THREE.AmbientLight(0xc4dbef, 1.7));
    const light = new THREE.DirectionalLight(0xffffff, 3.2);
    light.position.set(100, -150, 300);
    scene.add(light);
    const rim = new THREE.DirectionalLight(0x7dd3fc, 2.4);
    rim.position.set(-150, 100, 30);
    scene.add(rim);
    const fill = new THREE.DirectionalLight(0x5eead4, 1.2);
    fill.position.set(100, 100, -100);
    scene.add(fill);
    const center = new THREE.Vector3(
      ...(data.center as [number, number, number]),
    );
    const objects: THREE.Mesh[] = [];
    const makeMesh = (
      vertices: number[],
      faces: number[],
      color: string,
      brain: boolean,
    ) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(vertices, 3),
      );
      geometry.setIndex(faces);
      geometry.computeVertexNormals();
      geometry.translate(-center.x, -center.y, -center.z);
      const material = new THREE.MeshPhysicalMaterial({
        color,
        roughness: brain ? 0.65 : 0.38,
        metalness: 0.05,
        transparent: brain,
        opacity: brain ? 0.14 : 1,
        depthWrite: !brain,
        side: THREE.DoubleSide,
        clearcoat: brain ? 0 : 0.3,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.renderOrder = brain ? 2 : 1;
      objects.push(mesh);
      scene.add(mesh);
      return mesh;
    };
    const brain = data.brain?.vertices.length
      ? makeMesh(data.brain.vertices, data.brain.faces, "#aec2d2", true)
      : undefined;
    const meshes = data.regions
      .filter((r) => r.vertices.length > 0)
      .map((r, index) => {
        const mesh = makeMesh(
          r.vertices,
          r.faces,
          labels.find((l) => l.id === r.label)?.color || "#5eead4",
          false,
        );
        return {
          mesh,
          label: r.label,
          offset: new THREE.Vector3(
            (index - (data.regions.length - 1) / 2) * 35,
            0,
            index % 2 ? 10 : -10,
          ),
        };
      });
    const bounding = new THREE.Box3();
    objects.forEach((o) => bounding.expandByObject(o));
    const diameter = Math.max(
      ...bounding.getSize(new THREE.Vector3()).toArray(),
      90,
    );
    const reset = () => {
      camera.position.set(diameter * 1.1, -diameter * 1.7, diameter * 0.75);
      controls.target.set(0, 0, 0);
      controls.update();
    };
    reset();
    controls.minDistance = diameter * 0.15;
    controls.maxDistance = diameter * 6;
    const focus = (tumorOnly: boolean) => {
      const box = new THREE.Box3();
      if (!tumorOnly) {
        reset();
        return;
      }
      meshes
        .filter((item) => item.mesh.visible)
        .forEach((item) => box.expandByObject(item.mesh));
      if (box.isEmpty()) return;
      const target = box.getCenter(new THREE.Vector3());
      const extent = Math.max(
        ...box.getSize(new THREE.Vector3()).toArray(),
        15,
      );
      const direction = camera.position
        .clone()
        .sub(controls.target)
        .normalize();
      const distance =
        (extent / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)))) *
        1.65;
      camera.position.copy(
        target.clone().add(direction.multiplyScalar(distance)),
      );
      controls.target.copy(target);
      controls.update();
    };
    runtime.current = { brain, meshes, reset, focus };
    const resize = new ResizeObserver(() => {
      const width = element.clientWidth,
        height = element.clientHeight;
      if (!width || !height) return;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    });
    resize.observe(element);
    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      controls.autoRotate = rotate.current;
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
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
    r.meshes.forEach(({ mesh, label, offset }) => {
      mesh.visible = visibleLabels.includes(label);
      mesh.position.copy(exploded ? offset : new THREE.Vector3());
    });
  }, [data, visibleLabels, isolated, brainOpacity, exploded, labels]);
  useEffect(() => {
    runtime.current?.focus(isolated);
  }, [data, isolated, exploded]);
  useEffect(() => {
    runtime.current?.reset();
  }, [resetKey]);
  const fullScreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else
      void root.current
        ?.requestFullscreen()
        .catch(() =>
          setError("이 브라우저에서는 전체 화면을 지원하지 않습니다."),
        );
  };
  return (
    <div ref={root} className={`mesh-viewer ${compact ? "compact" : ""}`}>
      <div className="viewer-label">
        <span className="live-dot" />
        {isolated ? "ISOLATED TUMOR" : "3D RECONSTRUCTION"}
        {exploded && <span className="tiny-tag">영역 분리</span>}
      </div>
      <div className="viewer-buttons">
        <button
          className={autoRotate ? "icon-btn active" : "icon-btn"}
          onClick={() => setAutoRotate((v) => !v)}
          aria-label="자동 회전"
          aria-pressed={autoRotate}
          title="자동 회전"
        >
          <Rotate3D size={17} />
        </button>
        <button
          className="icon-btn"
          onClick={() => runtime.current?.reset()}
          aria-label="3D 시점 초기화"
          title="시점 초기화"
        >
          <RotateCcw size={17} />
        </button>
        <button
          className="icon-btn"
          onClick={fullScreen}
          aria-label="3D 전체 화면"
          title="전체 화면"
        >
          <Expand size={17} />
        </button>
      </div>
      <div ref={host} className="canvas-host" />
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
      {!loading &&
        !error &&
        data &&
        data.regions.every((r) => !r.vertices.length) && (
          <div className="viewer-message">
            <span>선택한 결과에 종양 마스크가 없습니다.</span>
          </div>
        )}
      <div className="viewer-help">
        드래그하여 회전 <span>·</span> 스크롤하여 확대 <span>·</span> 우클릭하여
        이동
      </div>
      <span className="coordinate-note">RAS · mm</span>
    </div>
  );
}
