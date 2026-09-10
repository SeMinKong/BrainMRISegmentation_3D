import { useEffect, useState } from "react";
import { ScanLine } from "lucide-react";
import type { Case } from "../api";

type Props = {
  data: Case;
  plane: "axial" | "coronal" | "sagittal";
  modality: string;
  segmentation: string;
  visibleLabels: number[];
  overlay: boolean;
  opacity: number;
  window: number;
  resetKey: number;
  focusVoxel?: number[] | null;
  focusKey: number;
};
const planes = {
  axial: {
    name: "축상면",
    english: "AXIAL",
    axis: 2,
    top: "A",
    left: "R",
    right: "L",
    bottom: "P",
  },
  coronal: {
    name: "관상면",
    english: "CORONAL",
    axis: 1,
    top: "S",
    left: "R",
    right: "L",
    bottom: "I",
  },
  sagittal: {
    name: "시상면",
    english: "SAGITTAL",
    axis: 0,
    top: "S",
    left: "A",
    right: "P",
    bottom: "I",
  },
};
export default function SliceViewer({
  data,
  plane,
  modality,
  segmentation,
  visibleLabels,
  overlay,
  opacity,
  window: contrast,
  resetKey,
  focusVoxel,
  focusKey,
}: Props) {
  const meta = planes[plane];
  const count = data.shape[meta.axis];
  const [index, setIndex] = useState(Math.floor(count / 2));
  const [url, setUrl] = useState("");
  const [error, setError] = useState(false);
  useEffect(() => {
    setIndex(Math.floor(count / 2));
  }, [count, data.id, resetKey]);
  useEffect(() => {
    if (focusKey && focusVoxel)
      setIndex(
        Math.max(0, Math.min(count - 1, Math.round(focusVoxel[meta.axis]))),
      );
  }, [focusKey]);
  useEffect(() => {
    const controller = new AbortController();
    let objectUrl = "";
    const timeout = setTimeout(() => {
      const query = new URLSearchParams({
        modality,
        segmentation,
        overlay: String(overlay && visibleLabels.length > 0),
        opacity: String(opacity),
        labels: visibleLabels.join(","),
        window: String(contrast),
      });
      fetch(`/api/cases/${data.id}/slices/${plane}/${index}?${query}`, {
        signal: controller.signal,
      })
        .then((r) => {
          if (!r.ok) throw Error();
          return r.blob();
        })
        .then((blob) => {
          objectUrl = URL.createObjectURL(blob);
          setUrl(objectUrl);
          setError(false);
        })
        .catch((e) => {
          if (e.name !== "AbortError") setError(true);
        });
    }, 35);
    return () => {
      clearTimeout(timeout);
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [
    data.id,
    plane,
    index,
    modality,
    segmentation,
    visibleLabels,
    overlay,
    opacity,
    contrast,
  ]);
  return (
    <section className="slice-card" aria-label={`${meta.name} MRI 단면`}>
      <header>
        <span>
          {meta.name} <small>{meta.english}</small>
        </span>
        <span className="mono">
          {index + 1}
          <em> / {count}</em>
        </span>
      </header>
      <div
        className="slice-image"
        onWheel={(e) => {
          setIndex((v) =>
            Math.max(0, Math.min(count - 1, v + (e.deltaY > 0 ? 1 : -1))),
          );
        }}
      >
        {url && !error ? (
          <img
            src={url}
            alt={`${meta.name} ${index + 1}번째 MRI 단면과 선택한 종양 마스크`}
          />
        ) : (
          <div className="slice-empty">
            <ScanLine size={22} />
            <span>{error ? "단면 로드 실패" : "불러오는 중"}</span>
          </div>
        )}
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
