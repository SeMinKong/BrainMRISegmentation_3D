export type Label = { id: number; name: string; color: string };
/** Per-label comparison of one prediction with the reference, stored by the server when the job finishes. */
export type LabelMetrics = { dice?: number | null; volume_ml: number; missed_ml?: number | null; extra_ml?: number | null; reference_volume_ml?: number | null };
export type SegmentationMetrics = {
  dice?: number | null;
  hd95_mm?: number | null;
  missed_ml?: number | null;
  extra_ml?: number | null;
  prediction_volume_ml?: number | null;
  reference_volume_ml?: number | null;
  labels?: Record<string, LabelMetrics>;
};
export type Segmentation = {
  id: string;
  name: string;
  kind: string;
  created_at?: string;
  provenance?: { model_id?: string; [key: string]: unknown };
  metrics?: SegmentationMetrics;
};
/** Reference-mask volumes per label, filled in by the server in the background for list sorting. */
export type ReferenceSummary = { volumes_ml: Record<string, number>; total_volume_ml: number };
export type Case = {
  id: string;
  name: string;
  source: string;
  demo: boolean;
  label_preset: string;
  shape: number[];
  spacing: number[];
  orientation: string;
  modalities: string[];
  segmentations: Segmentation[];
  labels: Label[];
  study?: { patient_id?: string | null; split?: string | null; manifest?: string | null };
  reference_summary?: ReferenceSummary | null;
};
export const sourceLabel = (c: Case) =>
  c.demo ? "합성 데모" : c.source === "linked" ? "MU-Glioma-Post" : "NIfTI 볼륨";
export const sourceDescription = (c: Case) =>
  c.demo
    ? "조작 연습용 가상 데이터"
    : c.source === "linked"
      ? "data/ 원본 폴더에서 직접 연결한 실제 MRI"
      : "로컬에서 가져온 MRI";
export type ModelTraining = {
  validation_mean_dice?: number | null;
  validated_epoch?: number | null;
  epoch?: number | null;
  global_step?: number | null;
  train_cases?: number | null;
  val_cases?: number | null;
};
export type Model = {
  id: string;
  name: string;
  available: boolean;
  reason?: string;
  description: string;
  demo_only?: boolean;
  training?: ModelTraining;
};
export type Job = {
  id: string;
  case_id: string;
  model_id: string;
  status: string;
  progress: number;
  message: string;
  segmentation_id?: string;
  elapsed_seconds?: number;
  created_at?: string;
};
export type Region = {
  label: number;
  name: string;
  color: string;
  voxels: number;
  volume_ml: number;
  components: number;
  dice?: number | null;
  hd95_mm?: number | null;
  missed_ml?: number | null;
  extra_ml?: number | null;
  reference_volume_ml?: number | null;
};
export type Stats = {
  regions: Region[];
  total_volume_ml: number | null;
  dice?: number | null;
  hd95_mm?: number | null;
  missed_ml?: number | null;
  extra_ml?: number | null;
  reference_volume_ml?: number | null;
  tumor_center_voxel?: number[] | null;
  [key: string]: unknown;
};
export type Geometry = { vertices: number[]; faces: number[] };
export type MeshData = {
  brain: Geometry | null;
  regions: (Geometry & { label: number })[];
  center: number[];
};
/** Surfaces of what the model missed (reference only) and added (prediction only). */
export type DiffMeshData = { missed: Geometry; extra: Geometry; center: number[] };
export type OverviewCase = SegmentationMetrics & {
  case_id: string;
  name: string;
  study?: Case["study"];
  segmentation_id: string;
  created_at?: string;
};
export type Overview = {
  model_id: string;
  split: string | null;
  predicted: number;
  candidates: number;
  pending_jobs: number;
  summary: {
    mean_dice: number | null;
    median_dice: number | null;
    /** Mean of the per-label means: comparable with the validation Dice printed during training. */
    mean_label_dice: number | null;
    labels: Record<string, { mean: number | null; count: number }>;
    histogram: { from: number; to: number; count: number }[];
  };
  cases: OverviewCase[];
};
export type BatchResult = { queued: number; skipped: { case_id: string; reason: string }[]; jobs: Job[] };
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(
      typeof body.detail === "string"
        ? body.detail
        : `요청을 처리하지 못했습니다 (${response.status})`,
      response.status,
    );
  }
  return response.json() as Promise<T>;
}
export const number = (value: number | null | undefined, digits = 2) =>
  value == null || !Number.isFinite(value)
    ? "—"
    : value.toLocaleString("en-US", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      });
export const modalityName: Record<string, string> = {
  t1n: "T1",
  t1c: "T1ce",
  t2w: "T2",
  t2f: "FLAIR",
};
