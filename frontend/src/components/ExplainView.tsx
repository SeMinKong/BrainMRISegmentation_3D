import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { Crosshair, Focus, RotateCcw, X } from "lucide-react";
import { number } from "../api";
import type { Case, Segmentation, Stats } from "../api";
import { DIFF, friendlyLabel, sequenceHint, sequenceName } from "../labels";
import type { Patient } from "../patients";
import { useMask, useModality } from "../volumes";
import type { Plane } from "../volumes";
import { Help } from "./Help";
import MeshViewer, { EXTRA, MISSED } from "./MeshViewer";
import type { RegionInfo } from "./MeshViewer";
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
  /** Difference view: show where the model missed tumour (blue) / drew extra tumour (red). */
  showMissed: boolean;
  showExtra: boolean;
  resetKey: number;
  focusKey: number;
};

export type Shown = "reference" | "prediction" | "diff" | "both";

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
  /** Presentation mode hides the timeline and explanatory copy so the images fill the screen. */
  presenting?: boolean;
};

const PLANES = Object.keys(planes) as Plane[];
type Voxel = [number, number, number];

type SliceSettings = Pick<ViewState, "visibleLabels" | "overlay" | "opacity" | "resetKey" | "showMissed" | "showExtra"> & { window: number };

/** Loads one case's current sequence and one or two masks once, then feeds any number of slice views. */
function CaseSlices({ data, modality, segmentation, reference, settings, planesToShow, index, onIndexChange, cursor, onCursor, title }: {
  data: Case; modality: string; segmentation: string; reference?: string; settings: SliceSettings; planesToShow: Plane[];
  index?: number; onIndexChange?: (index: number) => void; cursor: Voxel | null; onCursor: (voxel: Voxel) => void; title?: string;
}) {
  const sequence = useModality(data.id, data.modalities.includes(modality) ? modality : data.modalities[0]);
  const mask = useMask(data.id, segmentation);
  const expected = useMask(data.id, reference ?? "");
  return (
    <>
      {planesToShow.map((plane) => (
        <SliceViewer
          key={plane}
          data={data}
          plane={plane}
          volume={sequence.value}
          mask={mask.value}
          reference={reference ? expected.value : null}
          diffColors={{ overlap: DIFF.overlap.color, missed: DIFF.missed.color, extra: DIFF.extra.color }}
          loading={sequence.loading}
          error={sequence.error}
          {...settings}
          index={index}
          onIndexChange={onIndexChange}
          cursor={cursor}
          onCursor={onCursor}
          title={title}
        />
      ))}
    </>
  );
}

/**
 * The viewer: which mask to show (expert, model, their difference, or both side by side), the sequence controls,
 * the 3D model on the left and the three planes on the right in golden-ratio columns. Clicking a slice moves
 * every plane to that point (linked crosshair).
 */
export default function ExplainView(props: Props) {
  const { data, patient, lead, prediction, shown, onShown, stats, predictionStats, view, onView, onFocus, onReset, onSelectCase, presenting } = props;
  const [comparePlane, setComparePlane] = useState<Plane>("axial");
  const [compareIndex, setCompareIndex] = useState<number | null>(null);
  const [cursor, setCursor] = useState<Voxel | null>(null);
  useEffect(() => setCursor(null), [data.id, view.resetKey]);
  // "종양 위치로": every plane jumps to the tumour centre and the 3D view frames the tumour.
  useEffect(() => {
    if (view.focusKey && stats?.tumor_center_voxel) setCursor(stats.tumor_center_voxel.map(Math.round) as Voxel);
  }, [view.focusKey]);
  const hasReference = data.segmentations.some((s) => s.id === "reference");
  const hasMask = data.segmentations.length > 0;
  const diffMode = shown === "diff" && !!prediction && hasReference;
  const segmentation = shown === "prediction" && prediction ? prediction.id : hasReference ? "reference" : data.segmentations[0]?.id ?? "";
  // Plain names plus the volume of whichever mask a viewer is showing, for its floating labels and tooltip.
  const regionsFor = (segmentationId: string): RegionInfo[] => {
    const source = prediction && segmentationId === prediction.id ? predictionStats : stats;
    const regions: RegionInfo[] = data.labels.map((label) => ({ id: label.id, name: friendlyLabel(label, data), volume: source?.regions.find((r) => r.label === label.id)?.volume_ml ?? null }));
    if (diffMode) regions.push({ id: MISSED, name: DIFF.missed.name, volume: predictionStats?.missed_ml ?? null }, { id: EXTRA, name: DIFF.extra.name, volume: predictionStats?.extra_ml ?? null });
    return regions;
  };
  const meshProps = {
    caseId: data.id,
    labels: data.labels,
    visibleLabels: view.visibleLabels,
    brainOpacity: view.brainOpacity,
    resetKey: view.resetKey,
    focusKey: view.focusKey,
  };
  const sliceSettings: SliceSettings = {
    visibleLabels: view.visibleLabels,
    overlay: view.overlay,
    opacity: view.opacity,
    window: view.contrast,
    resetKey: view.resetKey,
    showMissed: view.showMissed,
    showExtra: view.showExtra,
  };
  const compareAxis = { axial: 2, coronal: 1, sagittal: 0 }[comparePlane];
  const sharedIndex = compareIndex ?? Math.floor(data.shape[compareAxis] / 2);
  const diff = diffMode && prediction ? { prediction: prediction.id, showMissed: view.showMissed, showExtra: view.showExtra } : null;

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
        <button className={shown === "diff" ? "pressed" : ""} aria-pressed={shown === "diff"} disabled={!prediction || !hasReference} onClick={() => onShown("diff")}>
          <strong>차이 보기</strong>
          <small>틀린 곳만 색으로</small>
        </button>
        <button className={shown === "both" ? "pressed" : ""} aria-pressed={shown === "both"} disabled={!prediction || !hasReference} onClick={() => onShown("both")}>
          <strong>나란히 비교</strong>
          <small>같은 단면을 함께 넘기기</small>
        </button>
      </div>
      {diffMode ? (
        <div className="diff-legend" role="group" aria-label="차이 표시 항목">
          <span className="legend-item"><span className="swatch" style={{ background: DIFF.overlap.color }} /> {DIFF.overlap.name}</span>
          <button className={`chip ${view.showMissed ? "pressed" : ""}`} aria-pressed={view.showMissed} onClick={() => onView({ showMissed: !view.showMissed })}>
            <span className="swatch" style={{ background: DIFF.missed.color }} /> {DIFF.missed.name}
            {predictionStats?.missed_ml != null && <b className="mono">{number(predictionStats.missed_ml, 1)} mL</b>}
          </button>
          <button className={`chip ${view.showExtra ? "pressed" : ""}`} aria-pressed={view.showExtra} onClick={() => onView({ showExtra: !view.showExtra })}>
            <span className="swatch" style={{ background: DIFF.extra.color }} /> {DIFF.extra.name}
            {predictionStats?.extra_ml != null && <b className="mono">{number(predictionStats.extra_ml, 1)} mL</b>}
          </button>
          <Help term="diff" />
        </div>
      ) : (
        <span className="mask-hint muted">
          {shown === "reference" ? "전문가가 표시한 종양 영역입니다." : shown === "prediction" ? "내 모델이 MRI만 보고 그린 영역입니다." : "왼쪽 판독, 오른쪽 예측을 같은 단면으로 봅니다."}
        </span>
      )}
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
      <div className="slice-actions">
        <button className="neu-btn" onClick={onFocus} disabled={!stats?.tumor_center_voxel} title="세 단면과 3D를 종양 중심으로 이동">
          <Crosshair size={16} /> 종양 위치로
        </button>
        {cursor ? (
          <button className="neu-btn" onClick={() => setCursor(null)} title="십자선 지우기">
            <X size={16} /> 십자선 지우기
          </button>
        ) : (
          <span className="muted small-print">단면을 클릭하면 세 단면이 그 지점으로 맞춰집니다. <Help term="crosshair" /></span>
        )}
        <button className="neu-icon" onClick={onReset} aria-label="보기 초기화" title="보기 초기화">
          <RotateCcw size={16} />
        </button>
      </div>
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
            <input type="checkbox" checked={view.exploded} disabled={!hasMask || diffMode} onChange={(e) => onView({ exploded: e.target.checked })} />
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
      <MeshViewer {...meshProps} segmentation={diffMode ? "reference" : segmentation} isolated={view.isolated} exploded={view.exploded && !diffMode}
        regions={regionsFor(diffMode ? "reference" : segmentation)} diff={diff} title={diffMode ? "회색 = 일치 · 파랑 = 놓침 · 빨강 = 더 그림" : undefined} />
      {meshFoot}
    </section>
  );

  const timeline = !presenting && patient && patient.timepoints.length > 1 && (
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
                cursor={cursor}
                onCursor={(voxel) => { setCursor(voxel); setCompareIndex(voxel[compareAxis]); }}
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
          <CaseSlices data={data} modality={view.modality} segmentation={diffMode && prediction ? prediction.id : segmentation}
            reference={diffMode ? "reference" : undefined} settings={sliceSettings} planesToShow={PLANES} cursor={cursor} onCursor={setCursor} />
        </div>
      </div>
    </div>
  );
}
