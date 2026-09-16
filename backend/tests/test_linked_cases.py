"""Linked (by-reference) MU-Glioma-Post cases: no copies, RAS on read, demo hidden."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from fastapi.testclient import TestClient
import nibabel as nib
import numpy as np
import pytest

from backend.app.main import create_app, resolve_source_manifest
from backend.app.store import CaseStore

SHAPE = (12, 14, 10)
# LPS like the released files: x and y flipped relative to RAS.
LPS_AFFINE = np.array([[-1., 0, 0, 5], [0, -1., 0, 7], [0, 0, 1., -3], [0, 0, 0, 1]])
MODALITIES = {"t1": "t1n", "t1ce": "t1c", "t2": "t2w", "flair": "t2f"}


def write_case(root: Path, case_id: str, *, with_mask: bool = True, seed: int = 0) -> tuple[dict, np.ndarray | None]:
    folder = root / "MU-Glioma-Post" / case_id.split("_Timepoint_")[0] / f"Timepoint_{case_id.split('_Timepoint_')[1]}"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    entry = {"case_id": case_id, "patient_id": case_id.split("_Timepoint_")[0], "split": "train", "modalities": {}, "label": None}
    for key, suffix in MODALITIES.items():
        path = folder / f"{case_id}_brain_{suffix}.nii.gz"
        nib.save(nib.Nifti1Image(rng.normal(100, 10, SHAPE).astype(np.float32), LPS_AFFINE), str(path))
        entry["modalities"][key] = str(path.relative_to(root)).replace("\\", "/")
    mask = None
    if with_mask:
        mask = np.zeros(SHAPE, dtype=np.uint8)
        mask[2:5, 3:7, 4:8] = 2
        mask[3:4, 4:6, 5:7] = 3
        path = folder / f"{case_id}_tumorMask.nii.gz"
        nib.save(nib.Nifti1Image(mask, LPS_AFFINE), str(path))
        entry["label"] = str(path.relative_to(root)).replace("\\", "/")
    return entry, mask


@pytest.fixture()
def dataset(tmp_path):
    root = tmp_path / "data"
    labeled, mask = write_case(root, "PatientID_0003_Timepoint_1", seed=1)
    unlabeled, _ = write_case(root, "PatientID_0187_Timepoint_3", with_mask=False, seed=2)
    broken, _ = write_case(root, "PatientID_0999_Timepoint_1", seed=3)
    broken["modalities"]["t1"] = "MU-Glioma-Post/missing.nii.gz"
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"labels": {"0": "background", "1": "NETC", "2": "SNFH", "3": "ET", "4": "RC"},
                                    "cases": [labeled, unlabeled, broken, {"case_id": "../escape", "modalities": labeled["modalities"]}]}),
                        encoding="utf-8")
    return {"root": root, "manifest": manifest, "mask": mask, "store": tmp_path / "store"}


def test_manifest_cases_are_linked_without_copying_and_demo_is_hidden(dataset):
    with TestClient(create_app(dataset["store"], source_manifest=dataset["manifest"])) as client:
        health = client.get("/api/health").json()
        assert health["demo"] is False
        assert health["cases"] == {"total": 2, "linked": 2, "uploaded": 0}
        assert health["source_manifest"]["linked"] == 2 and health["source_manifest"]["skipped"] == 2
        assert any("missing" in error.lower() for error in health["source_manifest"]["errors"])
        cases = {case["id"]: case for case in client.get("/api/cases").json()["cases"]}
        assert set(cases) == {"PatientID_0003_Timepoint_1", "PatientID_0187_Timepoint_3"}
        case = cases["PatientID_0003_Timepoint_1"]
        assert case["source"] == "linked" and case["demo"] is False and case["label_preset"] == "mu_glioma_post"
        assert "files" not in case, "paths never leave the server"
        assert case["name"] == "Patient 0003 · Timepoint 1 · train"
        assert case["study"] == {"patient_id": "PatientID_0003", "split": "train", "manifest": "manifest.json"}
        assert case["modalities"] == ["t1n", "t1c", "t2w", "t2f"]
        assert [seg["id"] for seg in case["segmentations"]] == ["reference"]
        assert cases["PatientID_0187_Timepoint_3"]["segmentations"] == []
        # Geometry is the RAS grid of the LPS source: flipped axes, same world extent.
        expected = nib.as_closest_canonical(nib.Nifti1Image(np.zeros(SHAPE, dtype=np.uint8), LPS_AFFINE))
        assert case["shape"] == list(SHAPE)
        np.testing.assert_allclose(np.asarray(case["affine"]), expected.affine)
        assert case["spacing"] == [1, 1, 1]
        # Only case.json lives under the store; no MRI copies were written.
        stored = sorted(path.name for path in (dataset["store"] / "PatientID_0003_Timepoint_1").iterdir())
        assert stored == ["case.json"]
        assert "demo-brain-001" not in cases
        assert [model["id"] for model in client.get("/api/models").json()["models"]] == ["unet3d", "swinunetr", "nnunet"]


def test_linked_reads_are_canonical_and_export_matches_reoriented_original(dataset, tmp_path):
    with TestClient(create_app(dataset["store"], source_manifest=dataset["manifest"])) as client:
        base = "/api/cases/PatientID_0003_Timepoint_1"
        expected = nib.as_closest_canonical(nib.Nifti1Image(dataset["mask"], LPS_AFFINE))
        stats = client.get(base + "/stats").json()
        assert stats["regions"][1]["voxels"] == int((dataset["mask"] == 2).sum())
        assert stats["tumor_center_voxel"] == np.round(np.array(np.nonzero(expected.get_fdata())).mean(axis=1)).astype(int).tolist()
        assert client.get(base + "/slices/axial/5?modality=t2f").status_code == 200
        assert client.get(base + "/slices/axial/5?modality=t2f").headers["content-type"] == "image/png"
        mesh = client.get(base + "/mesh").json()
        assert mesh["regions"][1]["label"] == 2 and mesh["regions"][1]["vertices"]
        download = client.get(base + "/segmentations/reference/download")
        assert download.status_code == 200
        assert 'filename="PatientID_0003_Timepoint_1_reference.nii.gz"' in download.headers["content-disposition"]
        exported = nib.Nifti1Image.from_bytes(gzip.decompress(download.content))
        np.testing.assert_array_equal(exported.get_fdata(), expected.get_fdata())
        np.testing.assert_allclose(exported.affine, expected.affine)
        assert exported.header.get_xyzt_units()[0] == "mm"
        # Source files were not touched.
        original = nib.load(str(dataset["root"] / "MU-Glioma-Post/PatientID_0003/Timepoint_1/PatientID_0003_Timepoint_1_tumorMask.nii.gz"))
        np.testing.assert_allclose(original.affine, LPS_AFFINE)


def test_predictions_for_linked_cases_persist_and_reference_is_read_only(dataset):
    expected = nib.as_closest_canonical(nib.Nifti1Image(dataset["mask"], LPS_AFFINE))
    prediction = nib.Nifti1Image((expected.get_fdata() > 0).astype(np.uint8) * 3, expected.affine)
    store = CaseStore(dataset["store"], source_manifest=dataset["manifest"])
    store.add_segmentation("PatientID_0003_Timepoint_1", "pred-abc", "3D U-Net", "prediction", prediction, provenance={"model_id": "unet3d"})
    with pytest.raises(ValueError, match="read-only"):
        store.add_segmentation("PatientID_0003_Timepoint_1", "reference", "x", "reference", prediction)
    lps_prediction = nib.Nifti1Image(np.zeros(SHAPE, dtype=np.uint8), LPS_AFFINE)
    with pytest.raises(ValueError, match="same affine"):
        store.add_segmentation("PatientID_0003_Timepoint_1", "pred-lps", "x", "prediction", lps_prediction)
    assert (dataset["store"] / "PatientID_0003_Timepoint_1" / "seg-pred-abc.nii.gz").is_file()
    # Restart: the manifest is re-linked and the earlier prediction survives; the store stays RAS.
    reloaded = CaseStore(dataset["store"], source_manifest=dataset["manifest"])
    case = reloaded.get("PatientID_0003_Timepoint_1")
    assert [seg["id"] for seg in case["segmentations"]] == ["reference", "pred-abc"]
    assert case["segmentations"][1]["provenance"] == {"model_id": "unet3d"}
    mask = reloaded.read_segmentation("PatientID_0003_Timepoint_1", "pred-abc")
    np.testing.assert_array_equal(mask, (expected.get_fdata() > 0).astype(np.uint8) * 3)
    with TestClient(create_app(dataset["store"], source_manifest=dataset["manifest"])) as client:
        result = client.get("/api/cases/PatientID_0003_Timepoint_1/stats?segmentation=pred-abc&reference=reference").json()
        assert result["dice"] == pytest.approx(1.0)
        assert result["regions"][2]["dice"] < 1.0  # ET grew to the whole foreground


def test_demo_policy(dataset, tmp_path):
    with TestClient(create_app(tmp_path / "empty", source_manifest=None)) as client:
        ids = [case["id"] for case in client.get("/api/cases").json()["cases"]]
        assert ids == ["demo-brain-001"], "auto keeps the phantom when nothing real exists"
        assert client.get("/api/health").json()["demo"] is True
    with TestClient(create_app(dataset["store"], source_manifest=dataset["manifest"], demo="always")) as client:
        ids = {case["id"] for case in client.get("/api/cases").json()["cases"]}
        assert "demo-brain-001" in ids and len(ids) == 3
        assert client.get("/api/models").json()["models"][0]["id"] == "demo-phantom"
    with TestClient(create_app(dataset["store"], source_manifest=dataset["manifest"], demo="never")) as client:
        assert "demo-brain-001" not in {case["id"] for case in client.get("/api/cases").json()["cases"]}
        assert (dataset["store"] / "demo-brain-001" / "case.json").is_file(), "hidden, not deleted"
        response = client.post("/api/jobs", json={"case_id": "PatientID_0003_Timepoint_1", "model_id": "demo-phantom"})
        assert response.status_code == 404
    with pytest.raises(ValueError, match="MRI_DEMO_CASE"):
        with TestClient(create_app(tmp_path / "bad", source_manifest=None, demo="sometimes")):
            pass


def test_source_manifest_env_resolution(tmp_path, monkeypatch):
    assert resolve_source_manifest("") is None
    assert resolve_source_manifest("none") is None
    manifest = tmp_path / "m.json"
    manifest.write_text("{}", encoding="utf-8")
    assert resolve_source_manifest(str(manifest)) == manifest
    with pytest.raises(FileNotFoundError):
        resolve_source_manifest(str(tmp_path / "missing.json"))
    monkeypatch.setenv("MRI_SOURCE_MANIFEST", "none")
    with TestClient(create_app(tmp_path / "store")) as client:
        assert client.get("/api/health").json()["source_manifest"]["manifest"] is None


def test_unsafe_or_colliding_manifest_entries_are_skipped(dataset):
    store = CaseStore(dataset["store"], source_manifest=dataset["manifest"])
    assert "../escape" not in store.cases
    assert all("escape" not in path.name for path in dataset["store"].iterdir())
    # An uploaded case must not be silently replaced by a manifest entry with the same id.
    image = nib.Nifti1Image(np.ones(SHAPE, dtype=np.float32), np.eye(4))
    store.create_case("upload", [("t1n", image)], None, "mu_glioma_post", case_id="PatientID_0500_Timepoint_1")
    manifest = json.loads(dataset["manifest"].read_text(encoding="utf-8"))
    entry = dict(manifest["cases"][0], case_id="PatientID_0500_Timepoint_1")
    manifest["cases"] = [entry]
    dataset["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    report = store.link_manifest(dataset["manifest"])
    assert report["linked"] == 0 and "collides" in report["errors"][0]
    assert store.get("PatientID_0500_Timepoint_1")["source"] == "uploaded"
