import type { Case } from "./api";

export type Timepoint = { case: Case; index: number | null; label: string };
export type Patient = { id: string; name: string; split: string | null; timepoints: Timepoint[] };
export type PatientFilter = { query: string; split: "all" | "train" | "val"; predictedOnly: boolean };

const TIMEPOINT = /_Timepoint_(\d+)$/i;

export const hasPrediction = (data: Case) =>
  data.segmentations.some((s) => s.kind === "prediction" || s.kind === "demo");

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

export function filterPatients(patients: Patient[], filter: PatientFilter): Patient[] {
  const query = filter.query.trim().toLowerCase().replace(/^patientid_/, "").replace(/^환자\s*/, "");
  return patients
    .filter((patient) => filter.split === "all" || patient.split === filter.split)
    .map((patient) => ({
      ...patient,
      timepoints: filter.predictedOnly ? patient.timepoints.filter((t) => hasPrediction(t.case)) : patient.timepoints,
    }))
    .filter((patient) => patient.timepoints.length > 0)
    .filter((patient) => {
      if (!query) return true;
      const haystack = [patient.id, patient.name, ...patient.timepoints.map((t) => t.case.id + " " + t.case.name)].join(" ").toLowerCase();
      return haystack.includes(query) || patient.id.toLowerCase().replace(/^patientid_/, "").startsWith(query);
    });
}

export function findPatient(patients: Patient[], caseId: string): Patient | undefined {
  return patients.find((patient) => patient.timepoints.some((t) => t.case.id === caseId));
}
