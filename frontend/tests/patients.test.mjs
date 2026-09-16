import assert from "node:assert/strict";
import test from "node:test";
import { caseDice, caseVolume, filterPatients, findPatient, groupPatients, hasCavity, hasPrediction, patientName, visibleCaseIds } from "../src/patients.ts";

const linked = (patient, timepoint, split = "train", segmentations = [{ id: "reference", name: "Reference mask", kind: "reference" }]) => ({
  id: `PatientID_${patient}_Timepoint_${timepoint}`,
  name: `Patient ${patient} · Timepoint ${timepoint} · ${split}`,
  source: "linked",
  demo: false,
  label_preset: "mu_glioma_post",
  shape: [240, 240, 155],
  spacing: [1, 1, 1],
  orientation: "RAS",
  modalities: ["t1n", "t1c", "t2w", "t2f"],
  segmentations,
  labels: [],
  study: { patient_id: `PatientID_${patient}`, split, manifest: "m.json" },
});
const uploaded = { id: "case-abc123", name: "외부 환자 A", source: "uploaded", demo: false, label_preset: "generic", shape: [10, 10, 10],
  spacing: [1, 1, 1], orientation: "RAS", modalities: ["t1n"], segmentations: [], labels: [] };

test("cases are grouped per patient with timepoints in numeric order", () => {
  const patients = groupPatients([linked("0010", 5), linked("0003", 2, "val"), linked("0010", 1), uploaded, linked("0003", 10, "val")]);
  assert.deepEqual(patients.map((p) => p.id), ["case-abc123", "PatientID_0003", "PatientID_0010"]);
  assert.deepEqual(patients[1].timepoints.map((t) => t.index), [2, 10]);
  assert.deepEqual(patients[2].timepoints.map((t) => t.label), ["1차", "5차"]);
  assert.equal(patients[1].split, "val");
  assert.equal(patients[1].name, "환자 0003");
  assert.equal(patients[0].name, "case-abc123");
  assert.equal(patients[0].timepoints[0].label, "외부 환자 A");
  assert.equal(patientName("PatientID_0275"), "환자 0275");
});

test("filters by split, prediction presence and free text", () => {
  const predicted = linked("0020", 1, "train", [{ id: "reference", kind: "reference", name: "r" }, { id: "pred-1", kind: "prediction", name: "3D U-Net" }]);
  const patients = groupPatients([linked("0003", 1, "val"), linked("0003", 2, "val"), linked("0010", 1), predicted]);
  const base = { query: "", split: "all", predictedOnly: false, withCavity: false, sort: "id" };
  assert.deepEqual(filterPatients(patients, { ...base, split: "val" }).map((p) => p.id), ["PatientID_0003"]);
  assert.deepEqual(filterPatients(patients, { ...base, predictedOnly: true }).map((p) => p.id), ["PatientID_0020"]);
  assert.deepEqual(filterPatients(patients, { ...base, query: "10" }).map((p) => p.id), ["PatientID_0010"]);
  assert.deepEqual(filterPatients(patients, { ...base, query: "환자 0003" }).map((p) => p.id), ["PatientID_0003"]);
  assert.deepEqual(filterPatients(patients, { ...base, query: "PatientID_0020" }).map((p) => p.id), ["PatientID_0020"]);
  assert.equal(filterPatients(patients, { ...base, query: "없는환자" }).length, 0);
  assert.equal(hasPrediction(predicted), true);
  assert.equal(hasPrediction(linked("0003", 1)), false);
  assert.equal(findPatient(patients, "PatientID_0010_Timepoint_1")?.id, "PatientID_0010");
  assert.equal(findPatient(patients, "missing"), undefined);
});

test("list doubles as a worklist: lowest Dice first, largest tumour first, cavity filter", () => {
  const scored = (patient, timepoint, dice, volume, rc = 0) => ({
    ...linked(patient, timepoint, "val", [
      { id: "reference", kind: "reference", name: "r" },
      { id: "pred-1", kind: "prediction", name: "3D U-Net", provenance: { model_id: "unet3d" }, metrics: { dice, reference_volume_ml: volume, labels: { 4: { volume_ml: 0, reference_volume_ml: rc } } } },
    ]),
    reference_summary: { volumes_ml: { 1: 0, 2: volume - rc, 3: 0, 4: rc }, total_volume_ml: volume },
  });
  const unscored = linked("0050", 1, "val");
  const patients = groupPatients([scored("0001", 1, 0.9, 20), scored("0002", 1, 0.4, 80, 5), scored("0002", 2, 0.6, 60, 5), scored("0003", 1, 0.7, 5), unscored]);
  const base = { query: "", split: "all", predictedOnly: false, withCavity: false, sort: "id" };
  assert.deepEqual(filterPatients(patients, { ...base, sort: "dice" }).map((p) => p.id), ["PatientID_0002", "PatientID_0003", "PatientID_0001", "PatientID_0050"], "unscored last");
  assert.deepEqual(filterPatients(patients, { ...base, sort: "dice" })[0].timepoints.map((t) => t.index), [1, 2], "worst visit first inside the patient");
  assert.deepEqual(filterPatients(patients, { ...base, sort: "volume" }).map((p) => p.id), ["PatientID_0002", "PatientID_0001", "PatientID_0003", "PatientID_0050"]);
  assert.deepEqual(filterPatients(patients, { ...base, withCavity: true }).map((p) => p.id), ["PatientID_0002"]);
  assert.deepEqual(visibleCaseIds(filterPatients(patients, { ...base, sort: "dice" })).slice(0, 3),
    ["PatientID_0002_Timepoint_1", "PatientID_0002_Timepoint_2", "PatientID_0003_Timepoint_1"]);
  assert.equal(caseDice(unscored), null);
  assert.equal(caseVolume(scored("0009", 1, 0.5, 33)), 33);
  assert.equal(hasCavity(unscored), null);
});
