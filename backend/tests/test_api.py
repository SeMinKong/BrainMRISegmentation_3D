from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import nibabel as nib
import numpy as np
from PIL import Image
import pytest

from backend.app.main import create_app
from backend.app.store import CaseStore
from backend.app.volumes import LABELS, _slice, load_nifti, mask_metrics, stats


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with TestClient(create_app(tmp_path_factory.mktemp("case-data"))) as browser:
        yield browser


def nifti_bytes(data=None, affine=None):
    if data is None:
        data = np.ones((12, 14, 10), dtype=np.float32) * 60
    image = nib.Nifti1Image(data, np.diag([1., 2., 3., 1.]) if affine is None else affine)
    image.header.set_xyzt_units("mm")
    return image.to_bytes()


def import_files(client, images, preset="mu_glioma_post"):
    return client.post("/api/cases/import", data={"name": "Test study", "label_preset": preset},
                       files=[("files", (name, data, "application/octet-stream")) for name, data in images])


def test_demo_is_explicit_and_preserves_physical_metadata(client):
    case = client.get("/api/cases/demo-brain-001").json()
    assert case["demo"] is True
    assert case["source"] == "synthetic"
    assert case["orientation"] == "RAS"
    assert case["shape"] == [96, 112, 96]
    assert case["spacing"] == [1.7, 1.7, 1.7]
    assert {seg["id"] for seg in case["segmentations"]} == {"reference", "demo_prediction"}
    assert "not learned" in case["description"]
    assert client.get("/api/health").json()["status"] == "ok"


def test_slices_are_real_png_and_selection_changes_overlay(client):
    base = "/api/cases/demo-brain-001/slices/axial/52"
    overlay = client.get(base)
    plain = client.get(base + "?overlay=false")
    assert overlay.status_code == 200
    assert overlay.headers["content-type"] == "image/png"
    assert overlay.content != plain.content
    image = Image.open(BytesIO(overlay.content))
    assert image.height == 512
    assert client.get(base + "?opacity=2").status_code == 422
    assert client.get("/api/cases/demo-brain-001/slices/axial/999").status_code == 422
    assert client.get(base + "?labels=99").status_code == 422


def test_radiological_orientation():
    grid = np.arange(3 * 4 * 5).reshape((3, 4, 5))
    axial, _ = _slice(grid, "axial", 2)
    coronal, _ = _slice(grid, "coronal", 1)
    sagittal, _ = _slice(grid, "sagittal", 0)
    assert axial[0, 0] == grid[2, 3, 2]  # anterior/right on top/left
    assert coronal[0, 0] == grid[2, 1, 4]  # superior/right on top/left
    assert sagittal[0, 0] == grid[0, 3, 4]  # superior/anterior on top/left


def test_volume_units_and_metrics_are_computed():
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 3
    affine = np.diag([2., 3., 4., 1.])
    result = stats(mask, affine)
    assert result["voxel_volume_mm3"] == pytest.approx(24)
    assert result["total_volume_ml"] == pytest.approx(8 * 24 / 1000)
    assert result["tumor_bbox_voxel"] == [[2, 2, 2], [3, 3, 3]]
    region = next(x for x in result["regions"] if x["label"] == 3)
    assert region["components"] == 1
    assert region["voxels"] == 8
    same = mask_metrics(mask > 0, mask > 0, affine)
    assert same["dice"] == 1
    assert same["hd95_mm"] == 0
    shifted = mask_metrics(mask > 0, np.roll(mask > 0, 1, axis=0), affine)
    assert shifted["dice"] == .5
    assert shifted["hd95_mm"] == pytest.approx(2)


def test_empty_mask_metrics_are_explicit():
    empty = np.zeros((3, 3, 3), dtype=bool)
    full = np.ones((3, 3, 3), dtype=bool)
    assert mask_metrics(empty, empty, np.eye(4))["hd95_mm"] is None
    assert mask_metrics(full, empty, np.eye(4))["dice"] == 0


def test_mesh_is_three_dimensional_and_uses_shared_physical_frame(client):
    result = client.get("/api/cases/demo-brain-001/mesh")
    assert result.status_code == 200
    mesh = result.json()
    assert mesh["units"] == "mm"
    assert mesh["center"] == pytest.approx([0, 0, 0], abs=1e-4)
    assert len(mesh["regions"]) == 4
    for part in [mesh["brain"], *mesh["regions"]]:
        points = np.asarray(part["vertices"]).reshape(-1, 3)
        faces = np.asarray(part["faces"]).reshape(-1, 3)
        assert points.shape[0] > 10
        assert np.ptp(points, axis=0).min() > 0
        assert faces.max() < points.shape[0]


def test_demo_scores_are_computed_without_self_comparison(client):
    result = client.get("/api/cases/demo-brain-001/stats?segmentation=demo_prediction").json()
    assert 0 < result["dice"] < 1
    assert result["hd95_mm"] > 0
    assert result["reference_id"] == "reference"
    own = client.get("/api/cases/demo-brain-001/stats").json()
    assert own["dice"] is None
    assert "Self-comparison" in own["metric_note"]


def test_import_rejects_mismatched_shape_and_affine(client):
    shape_result = import_files(client, [("subject_t1n.nii", nifti_bytes()),
        ("subject_t1c.nii", nifti_bytes(np.ones((10, 10, 10), dtype=np.float32)))])
    assert shape_result.status_code == 422
    assert "same shape" in shape_result.json()["detail"]
    affine_result = import_files(client, [("subject_t1n.nii", nifti_bytes()),
        ("subject_t1c.nii", nifti_bytes(affine=np.eye(4)))])
    assert affine_result.status_code == 422
    assert "affine" in affine_result.json()["detail"]


@pytest.mark.parametrize("bad_label", [1.5, 5, -1])
def test_invalid_segmentation_labels_rejected(client, bad_label):
    bad = np.ones((12, 14, 10), dtype=np.float32) * bad_label
    result = import_files(client, [("subject_t1n.nii", nifti_bytes()), ("subject_seg.nii", nifti_bytes(bad))])
    assert result.status_code == 422
    assert "integer labels" in result.json()["detail"]


def test_generic_semantics_are_explicit(client):
    mask = np.zeros((12, 14, 10), dtype=np.uint8)
    mask[4:7, 4:7, 4:7] = 4
    result = import_files(client, [("subject_t1n.nii", nifti_bytes()), ("subject_seg.nii", nifti_bytes(mask))], "generic")
    assert result.status_code == 201
    assert result.json()["label_preset"] == "generic"
    assert next(x for x in result.json()["labels"] if x["id"] == 4)["name"] == "Region 4"
    reject = import_files(client, [("subject_t1n.nii", nifti_bytes())], "unknown_preset")
    assert reject.status_code == 422
    assert "Unknown label preset" in reject.json()["detail"]


def test_mu_glioma_post_is_the_default_preset(client):
    presets = client.get("/api/label-presets").json()["presets"]
    assert {preset["id"] for preset in presets} == {"mu_glioma_post", "generic"}
    result = client.post("/api/cases/import", files=[("files", ("subject_t1n.nii", nifti_bytes()))])
    assert result.status_code == 201
    assert result.json()["label_preset"] == "mu_glioma_post"
    assert {label["id"]: label["name"] for label in result.json()["labels"]} == {
        1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"
    }


@pytest.mark.parametrize("preset,definitions,expected", [
    ("previous_preset", [dict(label, name="SNFH / edema") if label["id"] == 2 else dict(label) for label in LABELS], "mu_glioma_post"),
    ("generic", LABELS, "generic"),
    ("previous_preset", [dict(label, name="Other tissue") if label["id"] == 4 else dict(label) for label in LABELS], "generic"),
])
def test_saved_preset_normalization_uses_meaning_without_rewriting_files(client, tmp_path, preset, definitions, expected):
    case = client.get("/api/cases/demo-brain-001").json()
    case.update(label_preset=preset, labels=definitions)
    directory = tmp_path / case["id"]
    directory.mkdir()
    metadata = directory / "case.json"
    original = json.dumps(case).encode()
    metadata.write_bytes(original)
    volume = directory / "t1n.nii.gz"
    volume.write_bytes(b"preserve existing volume")

    restored = CaseStore(tmp_path).get(case["id"])

    assert restored["label_preset"] == expected
    assert restored["labels"] == (LABELS if expected == "mu_glioma_post" else definitions)
    assert metadata.read_bytes() == original
    assert volume.read_bytes() == b"preserve existing volume"
    assert {path.name for path in directory.iterdir()} == {"case.json", "t1n.nii.gz"}


def test_generic_case_cannot_run_a_configured_model(client, monkeypatch):
    import backend.app.main as backend_main

    images = [(f"subject_{modality}.nii", nifti_bytes()) for modality in ("t1n", "t1c", "t2w", "t2f")]
    case = import_files(client, images, "generic").json()
    monkeypatch.setattr(backend_main, "model_catalog", lambda: [{"id": "configured-test", "available": True}])
    result = client.post("/api/jobs", json={"case_id": case["id"], "model_id": "configured-test"})
    assert result.status_code == 422
    assert "MU-Glioma-Post labels" in result.json()["detail"]


def test_canonicalization_and_export_preserve_world_geometry(client, tmp_path):
    affine = np.diag([-1., 2., 3., 1.])
    affine[0, 3] = 11
    mask = np.zeros((12, 14, 10), dtype=np.uint8)
    mask[1:3, 4:6, 5:7] = 3
    result = import_files(client, [("subject_t1n.nii", nifti_bytes(affine=affine)),
                                   ("subject_seg.nii", nifti_bytes(mask, affine))])
    assert result.status_code == 201
    case = result.json()
    assert case["spacing"] == [1, 2, 3]
    download = client.get(f"/api/cases/{case['id']}/segmentations/reference/download")
    assert download.status_code == 200
    path = tmp_path / "result.nii.gz"
    path.write_bytes(download.content)
    exported = nib.load(path)
    expected = nib.as_closest_canonical(nib.Nifti1Image(mask, affine))
    np.testing.assert_allclose(exported.affine, expected.affine)
    np.testing.assert_array_equal(exported.get_fdata(), expected.get_fdata())
    assert exported.header.get_xyzt_units()[0] == "mm"


def test_no_mask_import_has_empty_stats_and_cannot_run_demo(client):
    result = import_files(client, [("scan.nii", nifti_bytes())])
    assert result.status_code == 201
    case = result.json()
    assert case["modalities"] == ["mri"]
    assert client.get(f"/api/cases/{case['id']}/stats?segmentation=").json()["total_volume_ml"] is None
    assert client.get(f"/api/cases/{case['id']}/mesh?segmentation=").json()["regions"] == []
    job = client.post("/api/jobs", json={"case_id": case["id"], "model_id": "demo-phantom"})
    assert job.status_code == 422
    assert "restricted" in job.json()["detail"]


def test_demo_job_produces_downloadable_mask(client):
    result = client.post("/api/jobs", json={"case_id": "demo-brain-001", "model_id": "demo-phantom"})
    assert result.status_code == 202
    job_id = result.json()["id"]
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(.05)
    assert job["status"] == "completed", job
    assert job["progress"] == 100
    assert job["segmentation_id"]
    assert client.get(f"/api/cases/demo-brain-001/segmentations/{job['segmentation_id']}/download").status_code == 200


def test_header_and_origin_guards(client, tmp_path):
    denied = client.post("/api/jobs", headers={"Origin": "https://malicious.example"},
                         json={"case_id": "demo-brain-001", "model_id": "demo-phantom"})
    assert denied.status_code == 403
    assert client.get("/api/not-real").status_code == 404
    oblique = np.eye(4)
    oblique[0, 1] = .2
    path = tmp_path / "oblique.nii"
    path.write_bytes(nifti_bytes(affine=oblique))
    with pytest.raises(ValueError, match="Oblique/sheared"):
        load_nifti(path)
    invalid = import_files(client, [("subject.nii", b"not a NIfTI volume")])
    assert invalid.status_code == 422


def test_size_checked_from_header_before_allocating_data(tmp_path):
    image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))
    image.header.set_data_shape((500, 500, 500))
    path = tmp_path / "large.nii"
    # Header advertises too much data but no data payload exists. Validation must
    # reject by shape before attempting to allocate or read the missing payload.
    path.write_bytes(image.header.binaryblock + b"\x00" * 4)
    with pytest.raises(ValueError, match="exceeds"):
        load_nifti(path)


def test_built_frontend_serves_module_scripts_as_javascript(client):
    import re
    page = client.get("/")
    if "text/html" not in page.headers.get("content-type", ""):
        pytest.skip("Frontend has not been built")
    script = re.search(r'src="(/assets/[^"]+\.js)"', page.text)
    assert script is not None
    asset = client.get(script.group(1))
    assert asset.status_code == 200
    assert asset.headers["content-type"].split(";")[0] == "text/javascript"
    assert asset.headers["x-content-type-options"] == "nosniff"
