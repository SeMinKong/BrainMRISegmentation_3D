import { useEffect, useRef, useState } from "react";
import { LoaderCircle, ScanLine } from "lucide-react";
import type { Case } from "../api";
import { drawSlice, planeAxis, sliceGeometry } from "../volumes";
import type { Plane, Volume } from "../volumes";

type Props = {
  data: Case;
  plane: Plane;
  volume: Volume | null;
  mask: Volume | null;
  loading?: boolean;
  error?: string;
  visibleLabels: number[];
  overlay: boolean;
  opacity: number;
  window: number;
  resetKey: number;
  focusVoxel?: number[] | null;
  focusKey: number;
  /** Controlled slice position (0-based). When given, the parent owns the index. */
  index?: number;
  onIndexChange?: (index: number) => void;
  title?: string;
};
export const planes = {
  axial: { name: "축상면", hint: "위에서 내려다본 단면", top: "A", left: "R", right: "L", bottom: "P" },
  coronal: { name: "관상면", hint: "앞에서 본 단면", top: "S", left: "R", right: "L", bottom: "I" },
  sagittal: { name: "시상면", hint: "옆에서 본 단면", top: "S", left: "A", right: "P", bottom: "I" },
} as const;

/** Width/height of a plane's image in physical units, known from the case header before any voxel arrives. */
export function planeAspect(data: Case, plane: Plane): number {
  const [nx, ny, nz] = data.shape;
  const [sx, sy, sz] = data.spacing;
  if (plane === "axial") return (nx * sx) / (ny * sy);
  if (plane === "coronal") return (nx * sx) / (nz * sz);
  return (ny * sy) / (nz * sz);
}

/** Slices are drawn from the in-browser volume, so wheel and slider changes show on the next frame. */
export default function SliceViewer(props: Props) {
  const { data, plane, volume, mask, loading, error, visibleLabels, overlay, opacity, window: contrast, resetKey,
    focusVoxel, focusKey, index: controlled, onIndexChange, title } = props;
  const meta = planes[plane];
  const axis = planeAxis[plane];
  const count = data.shape[axis];
  const [internal, setInternal] = useState(Math.floor(count / 2));
  const index = Math.max(0, Math.min(count - 1, controlled ?? internal));
  const canvas = useRef<HTMLCanvasElement>(null);
  const frame = useRef<HTMLDivElement>(null);
  const setIndex = (next: number | ((v: number) => number)) => {
    const value = Math.max(0, Math.min(count - 1, typeof next === "function" ? next(index) : next));
    if (onIndexChange) onIndexChange(value);
    else setInternal(value);
  };
  const indexRef = useRef(setIndex);
  indexRef.current = setIndex;
  useEffect(() => {
    setInternal(Math.floor(count / 2));
  }, [count, data.id, resetKey]);
  useEffect(() => {
    if (focusKey && focusVoxel) setIndex(Math.round(focusVoxel[axis]));
  }, [focusKey]);
  useEffect(() => {
    if (!volume || !canvas.current) return;
    drawSlice(canvas.current, volume, mask, plane, index, { window: contrast, overlay, opacity, visibleLabels, labels: data.labels });
  }, [volume, mask, plane, index, contrast, overlay, opacity, visibleLabels, data.labels]);
  useEffect(() => {
    // Native listener so the page does not scroll while the wheel scrubs slices.
    const element = frame.current;
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      indexRef.current((v) => v + (event.deltaY > 0 ? 1 : -1));
    };
    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, []);
  const aspect = volume ? (() => { const g = sliceGeometry(volume, plane); return g.physicalWidth / g.physicalHeight; })() : planeAspect(data, plane);

  return (
    <section className="slice-card" aria-label={`${meta.name} MRI 단면`}>
      <div
        ref={frame}
        className="slice-image"
        style={{ aspectRatio: `${aspect}` }}
        tabIndex={0}
        role="img"
        aria-label={`${meta.name} ${index + 1}번째 단면. 휠 또는 위아래 화살표 키로 이동`}
        onKeyDown={(e) => {
          if (e.key === "ArrowUp" || e.key === "ArrowRight") setIndex((v) => v + 1);
          if (e.key === "ArrowDown" || e.key === "ArrowLeft") setIndex((v) => v - 1);
          if (e.key === "PageUp") setIndex((v) => v + 10);
          if (e.key === "PageDown") setIndex((v) => v - 10);
          if (e.key === "Home") setIndex(0);
          if (e.key === "End") setIndex(count - 1);
        }}
      >
        <canvas ref={canvas} className={volume ? "" : "hidden"} aria-hidden="true" />
        {!volume && (
          <div className="slice-empty">
            {error ? <ScanLine size={22} /> : <LoaderCircle className="spin" size={22} />}
            <span>{error || (loading ? "영상을 불러오는 중" : "영상 없음")}</span>
          </div>
        )}
        <span className="slice-tag" title={meta.hint}>{title ?? meta.name}</span>
        <span className="slice-index mono">
          {index + 1}
          <em> / {count}</em>
        </span>
        <span className="orientation top">{meta.top}</span>
        <span className="orientation left">{meta.left}</span>
        <span className="orientation right">{meta.right}</span>
        <span className="orientation bottom">{meta.bottom}</span>
      </div>
      <input
        aria-label={`${meta.name} 단면 위치`}
        type="range"
        min={0}
        max={count - 1}
        value={index}
        onChange={(e) => setIndex(+e.target.value)}
      />
    </section>
  );
}
