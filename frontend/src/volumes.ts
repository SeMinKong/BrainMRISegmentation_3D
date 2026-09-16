import { useEffect, useState } from "react";
import { ApiError } from "./api";
import type { Label } from "./api";

/** A whole RAS volume held in the browser so slices render locally; matches the server's `_slice` orientation. */
export type Volume = { data: Uint8Array; shape: [number, number, number]; spacing: [number, number, number] };
export type Plane = "axial" | "coronal" | "sagittal";

const MAX_ENTRIES = 8;
const cache = new Map<string, Promise<Volume>>();

function remember(key: string, load: () => Promise<Volume>): Promise<Volume> {
  const hit = cache.get(key);
  if (hit) {
    cache.delete(key);
    cache.set(key, hit);
    return hit;
  }
  const pending = load().catch((error) => {
    cache.delete(key);
    throw error;
  });
  cache.set(key, pending);
  while (cache.size > MAX_ENTRIES) cache.delete(cache.keys().next().value!);
  return pending;
}

async function fetchVolume(path: string): Promise<Volume> {
  const response = await fetch(`/api${path}`);
  if (!response.ok) throw new ApiError(`볼륨을 불러오지 못했습니다 (${response.status})`, response.status);
  const shape = (response.headers.get("x-shape") || "").split(",").map(Number) as [number, number, number];
  const spacing = (response.headers.get("x-spacing") || "1,1,1").split(",").map(Number) as [number, number, number];
  const data = new Uint8Array(await response.arrayBuffer());
  if (shape.length !== 3 || data.length !== shape[0] * shape[1] * shape[2]) throw new ApiError("볼륨 크기가 응답 헤더와 다릅니다", 500);
  return { data, shape, spacing };
}

export const loadModality = (caseId: string, modality: string) =>
  remember(`m:${caseId}:${modality}`, () => fetchVolume(`/cases/${caseId}/volumes/${encodeURIComponent(modality)}`));
export const loadMask = (caseId: string, segmentation: string) =>
  remember(`s:${caseId}:${segmentation}`, () => fetchVolume(`/cases/${caseId}/segmentations/${encodeURIComponent(segmentation)}/volume`));

type Loaded<T> = { value: T | null; error: string; loading: boolean };

function useLoaded(key: string, load: (() => Promise<Volume>) | null): Loaded<Volume> {
  const [state, setState] = useState<Loaded<Volume>>({ value: null, error: "", loading: !!load });
  useEffect(() => {
    if (!load) {
      setState({ value: null, error: "", loading: false });
      return;
    }
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: "" }));
    load().then(
      (value) => !cancelled && setState({ value, error: "", loading: false }),
      (error: Error) => !cancelled && setState({ value: null, error: error.message, loading: false }),
    );
    return () => {
      cancelled = true;
    };
  }, [key]);
  return state;
}

export const useModality = (caseId: string, modality: string) =>
  useLoaded(`m:${caseId}:${modality}`, caseId && modality ? () => loadModality(caseId, modality) : null);
export const useMask = (caseId: string, segmentation: string) =>
  useLoaded(`s:${caseId}:${segmentation}`, caseId && segmentation ? () => loadMask(caseId, segmentation) : null);

export const planeAxis: Record<Plane, 0 | 1 | 2> = { axial: 2, coronal: 1, sagittal: 0 };

/** Pixel size and physical extent of a slice in a plane (radiological display: patient right on screen left). */
export function sliceGeometry(volume: Volume, plane: Plane) {
  const [nx, ny, nz] = volume.shape;
  const [sx, sy, sz] = volume.spacing;
  if (plane === "axial") return { width: nx, height: ny, physicalWidth: nx * sx, physicalHeight: ny * sy };
  if (plane === "coronal") return { width: nx, height: nz, physicalWidth: nx * sx, physicalHeight: nz * sz };
  return { width: ny, height: nz, physicalWidth: ny * sy, physicalHeight: nz * sz };
}

export type RenderOptions = {
  window: number; // percent, 100 = the server's robust window
  overlay: boolean;
  opacity: number;
  visibleLabels: number[];
  labels: Label[];
  /**
   * Difference mode: `mask` is the prediction and `reference` the expert mask. Voxels in both are drawn grey,
   * reference-only (missed) blue and prediction-only (extra) red, so disagreement needs no side-by-side comparison.
   */
  reference?: Volume | null;
  diffColors?: { overlap: string; missed: string; extra: string };
  showMissed?: boolean;
  showExtra?: boolean;
};

/** Screen pixel (column, row) of a slice -> flat voxel index; the inverse mapping lives in `voxelAt`. */
function flatIndex(shape: [number, number, number], plane: Plane, k: number, c: number, r: number): number {
  const [nx, ny, nz] = shape;
  if (plane === "axial") return ((nx - 1 - c) * ny + (ny - 1 - r)) * nz + k;
  if (plane === "coronal") return ((nx - 1 - c) * ny + k) * nz + (nz - 1 - r);
  return (k * ny + (ny - 1 - c)) * nz + (nz - 1 - r);
}

/** Voxel (x, y, z) under a point given as fractions of the slice image, for the linked crosshair. */
export function voxelAt(shape: [number, number, number], plane: Plane, index: number, fx: number, fy: number): [number, number, number] {
  const [nx, ny, nz] = shape;
  const clamp = (v: number, n: number) => Math.max(0, Math.min(n - 1, Math.floor(v)));
  if (plane === "axial") return [clamp((1 - fx) * nx, nx), clamp((1 - fy) * ny, ny), index];
  if (plane === "coronal") return [clamp((1 - fx) * nx, nx), index, clamp((1 - fy) * nz, nz)];
  return [index, clamp((1 - fx) * ny, ny), clamp((1 - fy) * nz, nz)];
}

/** Where a voxel falls on a plane's image, as fractions (0..1) of width and height; the crosshair is drawn there. */
export function pointOf(shape: [number, number, number], plane: Plane, voxel: number[]): { fx: number; fy: number } {
  const [nx, ny, nz] = shape;
  const [x, y, z] = voxel;
  if (plane === "axial") return { fx: (nx - 0.5 - x) / nx, fy: (ny - 0.5 - y) / ny };
  if (plane === "coronal") return { fx: (nx - 0.5 - x) / nx, fy: (nz - 0.5 - z) / nz };
  return { fx: (ny - 0.5 - y) / ny, fy: (nz - 0.5 - z) / nz };
}

const hexToRgb = (hex: string): [number, number, number] => {
  const value = parseInt(hex.replace("#", ""), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
};

/** Draw one slice into `target` (resized to the slice's pixel grid). Pure CPU, ~1 ms for 240x240. */
export function drawSlice(target: HTMLCanvasElement, volume: Volume, mask: Volume | null, plane: Plane, index: number, options: RenderOptions) {
  const { width, height } = sliceGeometry(volume, plane);
  if (target.width !== width || target.height !== height) {
    target.width = width;
    target.height = height;
  }
  const context = target.getContext("2d");
  if (!context) return;
  const image = context.createImageData(width, height);
  const out = image.data;
  const data = volume.data;
  const labels = mask && options.overlay && options.visibleLabels.length ? mask.data : null;
  const reference = labels && options.reference ? options.reference.data : null;
  const colors = new Map<number, [number, number, number]>();
  for (const label of options.labels) if (options.visibleLabels.includes(label.id)) colors.set(label.id, hexToRgb(label.color));
  const diff = options.diffColors ?? { overlap: "#9aa4ae", missed: "#3b82f6", extra: "#ef4444" };
  const overlapColor = hexToRgb(diff.overlap), missedColor = hexToRgb(diff.missed), extraColor = hexToRgb(diff.extra);
  const showMissed = options.showMissed ?? true, showExtra = options.showExtra ?? true;
  const alpha = Math.max(0, Math.min(1, options.opacity));
  const w = Math.max(0.01, options.window / 100);
  const lo = 0.5 - w / 2;
  const scale = 255 / w;
  const k = Math.max(0, Math.min(index, volume.shape[planeAxis[plane]] - 1));
  let o = 0;
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const flat = flatIndex(volume.shape, plane, k, c, r);
      let gray = (data[flat] / 255 - lo) * scale;
      gray = gray < 0 ? 0 : gray > 255 ? 255 : gray;
      let red = gray, green = gray, blue = gray;
      if (reference && labels) {
        // Foreground-level comparison restricted to the labels currently shown.
        const predicted = colors.has(labels[flat]), expected = colors.has(reference[flat]);
        const color = predicted && expected ? overlapColor : expected && showMissed ? missedColor : predicted && showExtra ? extraColor : null;
        if (color) {
          const a = predicted && expected ? alpha * 0.55 : alpha; // agreement stays quiet; disagreement stays loud
          red = gray * (1 - a) + color[0] * a;
          green = gray * (1 - a) + color[1] * a;
          blue = gray * (1 - a) + color[2] * a;
        }
      } else if (labels) {
        const color = colors.get(labels[flat]);
        if (color) {
          red = gray * (1 - alpha) + color[0] * alpha;
          green = gray * (1 - alpha) + color[1] * alpha;
          blue = gray * (1 - alpha) + color[2] * alpha;
        }
      }
      out[o++] = red;
      out[o++] = green;
      out[o++] = blue;
      out[o++] = 255;
    }
  }
  context.putImageData(image, 0, 0);
}
