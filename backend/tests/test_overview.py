"""Batch prediction, stored comparison metrics, model overview and difference meshes."""
from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient
import nibabel as nib
import numpy as np
import pytest

import backend.app.main as backend_main
from backend.app.main import create_app
from backend.app.store import CaseStore
from backend.tests.test_linked_cases import LPS_AFFINE, SHAPE, write_case

MODEL = {"id": "unet3d", "name": "3D U-Net", "available": True, "reason": None, "description": "test"}
LABELS = {"0": "background", "1": "NETC", "2": "SNFH", "3": "ET", "4": "RC"}


def fake_inference(model_id, modalities, output_path, progress=None):
    """Writes the reference mask shifted by one voxel: partial overlap, so missed/extra volumes are non-zero."""
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[3:6, 3:7, 4:8] = 2
    mask[4:5, 4:6, 5:7] = 3
    nib.save(nib.Nifti1Image(mask, LPS_AFFINE), str(output_path))
    return {"model_id": model_id, "labels": LABELS, "device": "cpu"}


def wait(client, job_ids):
    for _ in range(200):
        jobs = [client.get(f"/api/jobs/{job_id}").json() for job_id in job_ids]
        if all(job["status"] in ("completed", "failed") for job in jobs):
            return jobs
        time.sleep(.05)
    raise AssertionError("jobs did not finish")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    root = tmp_path / "data"
    entries = []
    for index, (case_id, split, with_mask) in enumerate([("PatientID_0001_Timepoint_1", "val", True), ("PatientID_0002_Timepoint_1", "val", True),
                                                          ("PatientID_0003_Timepoint_1", "train", True), ("PatientID_0004_Timepoint_1", "val", False)]):
        entry, _ = write_case(root, case_id, with_mask=with_mask, seed=index)
        entry["split"] = split
        entries.append(entry)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"labels": LABELS, "cases": entries}), encoding="utf-8")
    monkeypatch.setattr(backend_main, "model_catalog", lambda: [MODEL])
    import ml.adapters
    monkeypatch.setattr(ml.adapters, "run_inference", fake_inference)
    with TestClient(create_app(tmp_path / "store", source_manifest=manifest)) as client:
        yield client


def test_reference_summary_is_computed_once_and_served(client):
    summary = client.get("/api/cases/PatientID_0001_Timepoint_1/reference-summary").json()
    assert summary["volumes_ml"]["2"] == pytest.approx(3 * 4 * 4 / 1000 - 1 * 2 * 2 / 1000)
    assert summary["total_volume_ml"] == pytest.approx(48 / 1000)
    assert client.get("/api/cases/PatientID_0004_Timepoint_1/reference-summary").status_code == 404
    stored = json.loads((client.app.state.store.root / "PatientID_0001_Timepoint_1" / "case.json").read_text(encoding="utf-8"))
    assert stored["reference_summary"] == summary
    # Relinking at the next server start keeps the summary instead of recomputing 571 of them.
    store = client.app.state.store
    again = CaseStore(store.root, source_manifest=store.root.parent / "data" / "manifest.json", demo="never")
    assert again.get("PatientID_0001_Timepoint_1")["reference_summary"] == summary


def test_batch_queues_only_eligible_unpredicted_cases_and_stores_metrics(client):
    ids = [case["id"] for case in client.get("/api/cases").json()["cases"]]
    result = client.post("/api/jobs/batch", json={"model_id": "unet3d", "case_ids": ids + ["nope"]})
    assert result.status_code == 202
    body = result.json()
    assert body["queued"] == 4 and {item["case_id"] for item in body["skipped"]} == {"nope"}
    jobs = wait(client, [job["id"] for job in body["jobs"]])
    assert all(job["status"] == "completed" for job in jobs), jobs

    case = client.get("/api/cases/PatientID_0001_Timepoint_1").json()
    prediction = next(segment for segment in case["segmentations"] if segment["kind"] == "prediction")
    metrics = prediction["metrics"]
    assert 0 < metrics["dice"] < 1 and metrics["missed_ml"] > 0 and metrics["extra_ml"] > 0
    assert metrics["reference_volume_ml"] == pytest.approx(48 / 1000) and metrics["prediction_volume_ml"] == pytest.approx(48 / 1000)
    assert set(metrics["labels"]) == {"1", "2", "3", "4"} and metrics["labels"]["2"]["dice"] is not None
    unlabeled = client.get("/api/cases/PatientID_0004_Timepoint_1").json()
    assert "metrics" not in next(segment for segment in unlabeled["segmentations"] if segment["kind"] == "prediction")

    # Re-running the batch skips cases this model already predicted.
    again = client.post("/api/jobs/batch", json={"model_id": "unet3d", "case_ids": ids}).json()
    assert again["queued"] == 0 and all("Already predicted" in item["reason"] for item in again["skipped"])
    forced = client.post("/api/jobs/batch", json={"model_id": "unet3d", "case_ids": ids[:1], "skip_predicted": False}).json()
    assert forced["queued"] == 1
    wait(client, [forced["jobs"][0]["id"]])


def test_overview_aggregates_stored_metrics_for_the_split(client):
    ids = [case["id"] for case in client.get("/api/cases").json()["cases"]]
    jobs = client.post("/api/jobs/batch", json={"model_id": "unet3d", "case_ids": ids}).json()["jobs"]
    wait(client, [job["id"] for job in jobs])

    overview = client.get("/api/overview", params={"model_id": "unet3d", "split": "val"}).json()
    assert overview["candidates"] == 2, "validation cases with a reference mask"
    assert overview["predicted"] == 3, "the unlabeled val case has a prediction but no score"
    scored = [row for row in overview["cases"] if row.get("dice") is not None]
    assert len(scored) == 2 and overview["cases"][-1]["case_id"] == "PatientID_0004_Timepoint_1", "unscored rows sort last"
    assert overview["summary"]["mean_dice"] == pytest.approx(np.mean([row["dice"] for row in scored]))
    assert sum(bucket["count"] for bucket in overview["summary"]["histogram"]) == 2
    assert overview["summary"]["labels"]["2"]["count"] == 2 and overview["summary"]["labels"]["1"]["count"] == 0
    assert overview["pending_jobs"] == 0
    everything = client.get("/api/overview", params={"model_id": "unet3d", "split": ""}).json()
    assert everything["predicted"] == 4


def test_diff_mesh_returns_missed_and_extra_surfaces(client):
    job = client.post("/api/jobs", json={"case_id": "PatientID_0001_Timepoint_1", "model_id": "unet3d"}).json()
    seg_id = wait(client, [job["id"]])[0]["segmentation_id"]
    response = client.get(f"/api/cases/PatientID_0001_Timepoint_1/diff-mesh", params={"prediction": seg_id})
    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"
    payload = response.json()  # the test client transparently decompresses
    assert payload["units"] == "mm" and len(payload["center"]) == 3
    for key in ("missed", "extra"):
        assert len(payload[key]["vertices"]) > 0 and len(payload[key]["faces"]) > 0, key
    assert client.get("/api/cases/PatientID_0001_Timepoint_1/diff-mesh", params={"prediction": "nope"}).status_code in (404, 422)
