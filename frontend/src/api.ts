export type Label = { id: number; name: string; color: string };
export type Segmentation = { id: string; name: string; kind: string };
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
};
export type Stats = {
  regions: Region[];
  total_volume_ml: number | null;
  dice?: number | null;
  hd95_mm?: number | null;
  tumor_center_voxel?: number[] | null;
  [key: string]: unknown;
};
export type Geometry = { vertices: number[]; faces: number[] };
export type MeshData = {
  brain: Geometry | null;
  regions: (Geometry & { label: number })[];
  center: number[];
};
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
