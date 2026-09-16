import { useState } from "react";
import type { ReactNode } from "react";
import { Crosshair, Focus, RotateCcw } from "lucide-react";
import type { Case, Segmentation, Stats } from "../api";
import { friendlyLabel, sequenceHint, sequenceName } from "../labels";
import type { Patient } from "../patients";
import { useMask, useModality } from "../volumes";
import type { Plane } from "../volumes";
import { Help } from "./Help";
import MeshViewer from "./MeshViewer";
import SliceViewer, { planes } from "./SliceViewer";
import Timeline from "./Timeline";

export type ViewState = {
  modality: string;
  visibleLabels: number[];
  opacity: number;
  overlay: boolean;
  isolated: boolean;
  exploded: boolean;
  brainOpacity: number;
  contrast: number;
  resetKey: number;
  focusKey: number;
};

export type Shown = "reference" | "prediction" | "both";

type Props = {
  data: Case;
  patient?: Patient;
  /** Card rendered above the 3D model in the left column (the result table), so the columns stay balanced. */
  lead?: ReactNode;
  prediction: Segmentation | null;
  shown: Shown;
  onShown: (shown: Shown) => void;
  stats: Stats | null;
  predictionStats: Stats | null;
  view: ViewState;
  onView: (patch: Partial<ViewState>) => void;
  onFocus: () => void;
  onReset: () => void;
  onSelectCase: (caseId: string) => void;
};

const PLANES = Object.keys(planes) as Plane[];

type SliceSettings = Pick<ViewState, "visibleLabels" | "overlay" | "opacity" | "resetKey" | "focusKey"> & { window: number };

/** Loads one case's current sequence and one mask once, then feeds any number of slice views. */
function CaseSlices({ data, modality, segmentation, settings, planesToShow, focusVoxel, index, onIndexChange, title }: {
  data: Case; modality: string; segmentation: string; settings: SliceSettings; planesToShow: Plane[];
  focusVoxel?: number[] | null; index?: number; onIndexChange?: (index: number) => void; title?: string;
}) {
  const sequence = useModality(data.id, data.modalities.includes(modality) ? modality : data.modalities[0]);
  const mask = useMask(data.id, segmentation);
  return (
    <>
      {planesToShow.map((plane) => (
        <SliceViewer
          key={plane}
          data={data}
          plane={plane}
          volume={sequence.value}
          mask={mask.value}
          loading={sequence.loading}
          error={sequence.error}
          {...settings}
          focusVoxel={focusVoxel}
          index={index}
          onIndexChange={onIndexChange}
          title={title}
        />
      ))}
    </>
  );
}

/**
 * The viewer: which mask to show (expert vs model vs both side by side), the sequence controls, the 3D model
 * on the left and the three planes on the right in golden-ratio columns.
 */
export default function ExplainView(props: Props) {
  const { data, patient, lead, prediction, shown, onShown, stats, predictionStats, view, onView, onFocus, onReset, onSelectCase } = props;
  // Plain names plus the volume of whichever mask a viewer is showing, for its floating labels and tooltip.
  const regionsFor = (segmentationId: string) => {
    const source = prediction && segmentationId === prediction.id ? predictionStats : stats;
    return data.labels.map((label) => ({ id: label.id, name: friendlyLabel(label, data), volume: source?.regions.find((r) => r.label === label.id)?.volume_ml ?? null }));
  };
  const [comparePlane, setComparePlane] = useState<Plane>("axial");
  const [compareIndex, setCompareIndex] = useState<number | null>(null);
  const hasReference = data.segmentations.some((s) => s.id === "reference");
  const hasMask = data.segmentations.length > 0;
  const segmentation = shown === "prediction" && prediction ? prediction.id : hasReference ? "reference" : data.segmentations[0]?.id ?? "";
  const meshProps = {
    caseId: data.id,
    labels: data.labels,
    visibleLabels: view.visibleLabels,
    brainOpacity: view.brainOpacity,
    resetKey: view.resetKey,
  };
  const sliceSettings: SliceSettings = {
    visibleLabels: view.visibleLabels,
    overlay: view.overlay,
    opacity: view.opacity,
    window: view.contrast,
    resetKey: view.resetKey,
    focusKey: view.focusKey,
  };
  const compareAxis = { axial: 2, coronal: 1, sagittal: 0 }[comparePlane];
  const sharedIndex = compareIndex ?? Math.floor(data.shape[compareAxis] / 2);

  const maskSwitch = (
    <div className="mask-switch neu-card">
      <div className="segmented" role="group" aria-label="표시할 종양 영역">
        <button className={shown === "reference" ? "pressed" : ""} aria-pressed={shown === "reference"} disabled={!hasReference} onClick={() => onShown("reference")}>
          <strong>판독 마스크</strong>
          <small>전문가 표시 (정답)</small>
        </button>
        <button className={shown === "prediction" ? "pressed" : ""} aria-pressed={shown === "prediction"} disabled={!prediction} onClick={() => onShown("prediction")}>
          <strong>내 모델 예측</strong>
          <small>{prediction ? "학습한 모델이 그린 영역" : "먼저 예측을 실행하세요"}</small>
        </button>
        <button className={shown === "both" ? "pressed" : ""} aria-pressed={shown === "both"} disabled={!prediction || !hasReference} onClick={() => onShown("both")}>
          <strong>나란히 비교</strong>
          <small>같은 단면을 함께 넘기기</small>
        </button>
      </div>
      <div className="mask-switch-actions">
        <button className="neu-btn" onClick={onFocus} disabled={!stats?.tumor_center_voxel} title="세 단면을 종양 중심으로 이동">
          <Crosshair size={16} /> 종양 위치로
        </button>
        <button className="neu-icon" onClick={onReset} aria-label="보기 초기화" title="보기 초기화">
          <RotateCcw size={16} />
        </button>
      </div>
    </div>
  );

  const controls = (vertical: boolean) => (
    <div className={`viewer-toolbar neu-card ${vertical ? "vertical" : ""}`}>
      <div className="segmented" role="group" aria-label="MRI 시퀀스">
        {data.modalities.map((m) => (
          <button key={m} className={m === view.modality ? "pressed" : ""} aria-pressed={m === view.modality} onClick={() => onView({ modality: m })} title={sequenceHint[m]}>
            <strong>{sequenceName[m] ?? m}</strong>
            <small>{sequenceHint[m] ?? ""}</small>
          </button>
        ))}
        <Help term="sequence" />
      </div>
      <label className="slider-field">
        <span>
          영역 표시 진하기 <b>{Math.round(view.opacity * 100)}%</b>
        </span>
        <input aria-label="영역 표시 진하기" type="range" min={0} max={1} step={0.05} value={view.overlay ? view.opacity : 0} disabled={!hasMask}
          onChange={(e) => onView({ opacity: +e.target.value, overlay: +e.target.value > 0 })} />
      </label>
      <label className="slider-field">
        <span>
          영상 밝기 범위 <b>{view.contrast}%</b>
        </span>
        <input aria-label="영상 밝기 범위" type="range" min={25} max={180} step={5} value={view.contrast} onChange={(e) => onView({ contrast: +e.target.value })} />
      </label>
    </div>
  );

  const meshFoot = (
      <div className="mesh-foot">
        <div className="mesh-modes" role="group" aria-label="3D 보기 방식">
          <button className={`neu-btn ${!view.isolated ? "pressed" : ""}`} aria-pressed={!view.isolated} onClick={() => onView({ isolated: false })}>
            뇌 + 종양
          </button>
          <button className={`neu-btn ${view.isolated ? "pressed" : ""}`} aria-pressed={view.isolated} disabled={!hasMask} onClick={() => onView({ isolated: true })}>
            <Focus size={15} /> 종양만 분리
          </button>
          <label className="toggle-field">
            <input type="checkbox" checked={view.exploded} disabled={!hasMask} onChange={(e) => onView({ exploded: e.target.checked })} />
            <span>영역 펼쳐 보기</span>
          </label>
        </div>
        <label className="slider-field inline">
          <span>뇌 표면 <b>{Math.round(view.brainOpacity * 100)}%</b></span>
          <input aria-label="뇌 표면 불투명도" type="range" min={0.05} max={0.9} step={0.05} value={view.brainOpacity} disabled={view.isolated} onChange={(e) => onView({ brainOpacity: +e.target.value })} />
        </label>
      </div>
  );

  const mesh = (
    <section className="mesh-well" aria-label="3D 모형">
      <MeshViewer {...meshProps} segmentation={segmentation} isolated={view.isolated} exploded={view.exploded} regions={regionsFor(segmentation)} />
      {meshFoot}
    </section>
  );

  const timeline = patient && patient.timepoints.length > 1 && (
    <Timeline patient={patient} data={data} visibleLabels={view.visibleLabels} onSelect={onSelectCase} />
  );

  if (shown === "both" && prediction && hasReference) {
    const pair = [{ id: "reference", title: "판독 마스크 (정답)" }, { id: prediction.id, title: "내 모델 예측" }];
    return (
      <div className="explain-view">
        {lead}
        {maskSwitch}
        <section className="mesh-pair" aria-label="판독 마스크와 모델 예측 3D 나란히 보기">
          <div className="compare-grid meshes">
            {pair.map((item) => (
              <div className="mesh-well" key={item.id}>
                <MeshViewer {...meshProps} segmentation={item.id} isolated={view.isolated} exploded={view.exploded} title={item.title} regions={regionsFor(item.id)} compact />
              </div>
            ))}
          </div>
          <div className="neu-card mesh-pair-foot">
            <span className="muted">두 모형은 함께 회전·확대됩니다. 한쪽을 돌리면 다른 쪽도 같은 방향을 봅니다.</span>
            {meshFoot}
          </div>
        </section>
        {controls(false)}
        <section className="compare-visits" aria-label="판독 마스크와 모델 예측 단면 나란히 보기">
          <header className="card-head">
            <div>
              <h3>왼쪽 판독 마스크, 오른쪽 내 모델 예측</h3>
              <p>같은 단면을 함께 넘깁니다. 색이 다른 곳이 모델이 틀린 부분입니다. <Help term="planes" /></p>
            </div>
            <div className="segmented small" role="group" aria-label="단면 방향">
              {PLANES.map((p) => (
                <button key={p} className={p === comparePlane ? "pressed" : ""} aria-pressed={p === comparePlane} onClick={() => { setComparePlane(p); setCompareIndex(null); }}>
                  {planes[p].name}
                </button>
              ))}
            </div>
          </header>
          <div className="compare-grid">
            {pair.map((item) => (
              <CaseSlices
                key={item.id}
                data={data}
                modality={view.modality}
                segmentation={item.id}
                settings={sliceSettings}
                planesToShow={[comparePlane]}
                title={item.title}
                index={sharedIndex}
                onIndexChange={setCompareIndex}
                focusVoxel={stats?.tumor_center_voxel}
              />
            ))}
          </div>
        </section>
        {timeline}
      </div>
    );
  }

  return (
    <div className="explain-view">
      {maskSwitch}
      <div className="explain-grid">
        <div className="explain-main">
          {lead}
          {mesh}
          {timeline}
        </div>
        <div className="explain-side">
          {controls(true)}
          <CaseSlices data={data} modality={view.modality} segmentation={segmentation} settings={sliceSettings} planesToShow={PLANES} focusVoxel={stats?.tumor_center_voxel} />
        </div>
      </div>
    </div>
  );
}
