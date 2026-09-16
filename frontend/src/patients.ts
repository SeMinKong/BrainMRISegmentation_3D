import type { Case, Segmentation } from "./api";

export type Timepoint = { case: Case; index: number | null; label: string };
export type Patient = { id: string; name: string; split: string | null; timepoints: Timepoint[] };
export type ListSort = "id" | "dice" | "volume";
export type PatientFilter = {
  query: string;
  split: "all" | "train" | "val";
  predictedOnly: boolean;
  /** Only visits whose reference mask contains a resection cavity (label 4). */
  withCavity: boolean;
  sort: ListSort;
};

const TIMEPOINT = /_Timepoint_(\d+)$/i;

export const hasPrediction = (data: Case) =>
  data.segmentations.some((s) => s.kind === "prediction" || s.kind === "demo");

/** The newest model output on a case; demo fixtures count on the synthetic case only. */
export const latestPrediction = (data?: Case): Segmentation | null =>
  [...(data?.segmentations ?? [])].reverse().find((s) => s.kind === "prediction" || (data?.demo && s.kind === "demo")) ?? null;

/** Whole-tumour Dice of the newest prediction, when the server has compared it with a reference. */
export const caseDice = (data: Case): number | null => {
  const dice = latestPrediction(data)?.metrics?.dice;
  return dice == null ? null : dice;
};

/** Reference tumour volume in mL: from the background summary, else from stored prediction metrics. */
export const caseVolume = (data: Case): number | null =>
  data.reference_summary?.total_volume_ml ?? latestPrediction(data)?.metrics?.reference_volume_ml ?? null;

export const hasCavity = (data: Case): boolean | null => {
  const summary = data.reference_summary;
  if (summary) return (summary.volumes_ml["4"] ?? 0) > 0;
  const label = latestPrediction(data)?.metrics?.labels?.["4"];
  return label ? (label.reference_volume_ml ?? 0) > 0 : null;
};

export function patientIdOf(data: Case): string {
  return data.study?.patient_id || data.id.replace(TIMEPOINT, "") || data.id;
}

export function patientName(id: string): string {
  const match = /^PatientID_(\d+)$/i.exec(id);
  return match ? `환자 ${match[1]}` : id;
}

export function timepointOf(data: Case): Timepoint {
  const match = TIMEPOINT.exec(data.id);
  const index = match ? Number(match[1]) : null;
  return { case: data, index, label: index === null ? data.name : `${index}차` };
}

/** Group cases by patient, ordered by patient id then timepoint. Uploaded/synthetic cases form single-visit patients. */
export function groupPatients(cases: Case[]): Patient[] {
  const groups = new Map<string, Patient>();
  for (const data of cases) {
    const id = patientIdOf(data);
    const patient = groups.get(id) ?? { id, name: patientName(id), split: data.study?.split ?? null, timepoints: [] };
    patient.timepoints.push(timepointOf(data));
    groups.set(id, patient);
  }
  const collator = new Intl.Collator("en", { numeric: true });
  return [...groups.values()]
    .map((patient) => ({
      ...patient,
      timepoints: [...patient.timepoints].sort((a, b) => (a.index ?? 0) - (b.index ?? 0) || collator.compare(a.case.id, b.case.id)),
    }))
    .sort((a, b) => collator.compare(a.id, b.id));
}

const minOf = (values: (number | null)[]) => {
  const known = values.filter((v): v is number => v != null);
  return known.length ? Math.min(...known) : null;
};
const maxOf = (values: (number | null)[]) => {
  const known = values.filter((v): v is number => v != null);
  return known.length ? Math.max(...known) : null;
};

/** Worst Dice among a patient's visits: the number that decides where the patient sits in a "lowest first" list. */
export const patientDice = (patient: Patient) => minOf(patient.timepoints.map((t) => caseDice(t.case)));
export const patientVolume = (patient: Patient) => maxOf(patient.timepoints.map((t) => caseVolume(t.case)));

/**
 * Filter, then order. "dice" puts the lowest agreement first (unscored visits last) and "volume" the largest
 * tumours first, so the list doubles as a worklist for finding failures.
 */
export function filterPatients(patients: Patient[], filter: PatientFilter): Patient[] {
  const query = filter.query.trim().toLowerCase().replace(/^patientid_/, "").replace(/^환자\s*/, "");
  const visible = patients
    .filter((patient) => filter.split === "all" || patient.split === filter.split)
    .map((patient) => ({
      ...patient,
      timepoints: patient.timepoints
        .filter((t) => !filter.predictedOnly || hasPrediction(t.case))
        .filter((t) => !filter.withCavity || hasCavity(t.case) === true),
    }))
    .filter((patient) => patient.timepoints.length > 0)
    .filter((patient) => {
      if (!query) return true;
      const haystack = [patient.id, patient.name, ...patient.timepoints.map((t) => t.case.id + " " + t.case.name)].join(" ").toLowerCase();
      return haystack.includes(query) || patient.id.toLowerCase().replace(/^patientid_/, "").startsWith(query);
    });
  if (filter.sort === "id") return visible;
  const collator = new Intl.Collator("en", { numeric: true });
  const key = filter.sort === "dice" ? patientDice : patientVolume;
  const ascending = filter.sort === "dice";
  return visible
    .map((patient) => ({
      ...patient,
      timepoints: [...patient.timepoints].sort((a, b) => {
        const x = filter.sort === "dice" ? caseDice(a.case) : caseVolume(a.case);
        const y = filter.sort === "dice" ? caseDice(b.case) : caseVolume(b.case);
        if (x == null && y == null) return 0;
        if (x == null) return 1;
        if (y == null) return -1;
        return ascending ? x - y : y - x;
      }),
    }))
    .sort((a, b) => {
      const x = key(a), y = key(b);
      if (x == null && y == null) return collator.compare(a.id, b.id);
      if (x == null) return 1;
      if (y == null) return -1;
      return (ascending ? x - y : y - x) || collator.compare(a.id, b.id);
    });
}

/** Visits in the order the list shows them: what the previous/next buttons and arrow keys step through. */
export const visibleCaseIds = (patients: Patient[]) => patients.flatMap((patient) => patient.timepoints.map((t) => t.case.id));

export function findPatient(patients: Patient[], caseId: string): Patient | undefined {
  return patients.find((patient) => patient.timepoints.some((t) => t.case.id === caseId));
}
