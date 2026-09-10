import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ml.adapters import describe_models, run_inference
from ml.data import prepare_case, save_prediction
from ml.manifest import main as create_manifest
from ml.schema import DEFAULT_LABELS, MODALITIES, load_manifest, require_mu_glioma_post_labels
from ml.smoke import create_fixture


def test_native_axis_and_affine_survive_round_trip(tmp_path):
    manifest_path = create_fixture(tmp_path)
    manifest = load_manifest(manifest_path)
    case = manifest["cases"][0]
    image, label, affine, original = prepare_case(case["modalities"], (1, 1, 1), case["label"], DEFAULT_LABELS)
    assert image.shape[0] == 4
    assert nib.aff2axcodes(affine) == ("R", "A", "S")
    output = tmp_path / "restored.nii.gz"
    save_prediction(label, affine, original, output, DEFAULT_LABELS)
    result = nib.load(output)
    np.testing.assert_allclose(result.affine, original.affine)
    np.testing.assert_array_equal(result.get_fdata(), nib.load(case["label"]).get_fdata())
    assert result.get_data_dtype() == np.dtype("uint8")


def test_anisotropic_grid_and_labels_are_preserved(tmp_path):
    shape, affine = (9, 10, 11), np.diag([-2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [32, -12, 5]
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    modalities = {}
    for name in MODALITIES:
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(source, affine), path)
        modalities[name] = path
    expected = np.zeros(shape, dtype=np.uint8)
    expected[1:5, 2:6, 3:7] = 4
    label_path = tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(expected, affine), label_path)
    _, labels, prepared_affine, original = prepare_case(modalities, (2, 3, 4), label_path, DEFAULT_LABELS)
    output = tmp_path / "out.nii.gz"
    save_prediction(labels, prepared_affine, original, output, DEFAULT_LABELS)
    np.testing.assert_array_equal(nib.load(output).get_fdata(), expected)
    np.testing.assert_allclose(nib.load(output).affine, affine)


def test_patient_leakage_is_rejected(tmp_path):
    path = create_fixture(tmp_path)
    data = json.loads(path.read_text())
    data["cases"][1]["patient_id"] = data["cases"][0]["patient_id"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Patient leakage"):
        load_manifest(path)


def test_mismatched_grid_is_rejected(tmp_path):
    manifest = load_manifest(create_fixture(tmp_path))
    case = manifest["cases"][0]
    path = Path(case["modalities"]["flair"])
    volume = nib.load(path)
    affine = volume.affine.copy()
    affine[0, 3] += 20
    nib.save(nib.Nifti1Image(volume.get_fdata(), affine), path)
    with pytest.raises(ValueError, match="co-registered"):
        prepare_case(case["modalities"], (1, 1, 1))


def test_no_checkpoint_means_no_prediction(tmp_path, monkeypatch):
    for variable in ("MRI_UNET_CHECKPOINT", "MRI_SWIN_CHECKPOINT", "MRI_NNUNET_MODEL_DIR"):
        monkeypatch.delenv(variable, raising=False)
    assert all(not model["available"] for model in describe_models())
    with pytest.raises(RuntimeError, match="No checkpoint"):
        run_inference("unet3d", {}, tmp_path / "must-not-exist.nii.gz")
    assert not (tmp_path / "must-not-exist.nii.gz").exists()


def test_incorrect_label_meanings_cannot_be_used_for_mu_glioma_post():
    require_mu_glioma_post_labels(DEFAULT_LABELS)
    with pytest.raises(ValueError, match="MU-Glioma-Post"):
        require_mu_glioma_post_labels(dict(DEFAULT_LABELS, **{"3": "RC", "4": "ET"}))


def test_manifest_discovery_requires_an_explicit_patient_regex(tmp_path):
    with pytest.raises(SystemExit) as result:
        create_manifest(["--data-root", str(tmp_path), "--output", str(tmp_path / "manifest.json")])
    assert result.value.code == 2
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("pattern", [r"patient-\d+", r"(patient)-(\d+)"])
def test_manifest_patient_regex_requires_one_capture_group(tmp_path, pattern):
    with pytest.raises(ValueError, match="exactly one capture group"):
        create_manifest(["--data-root", str(tmp_path), "--output", str(tmp_path / "manifest.json"),
                         "--patient-regex", pattern])


def test_manifest_discovers_both_separators_and_keeps_sessions_together(tmp_path):
    for case_id, separator in (("patient01_visit1", "_"), ("patient01_visit2", "-"), ("patient02_visit1", "_")):
        for suffix in ("t1", "t1ce", "t2", "flair", "seg"):
            image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
            nib.save(image, tmp_path / f"{case_id}{separator}{suffix}.nii.gz")
    output = tmp_path / "manifest.json"
    create_manifest(["--data-root", str(tmp_path), "--output", str(output),
                     "--patient-regex", r"^(patient\d+)_visit\d+$"])
    manifest = load_manifest(output)
    assert manifest["labels"] == DEFAULT_LABELS
    assert len(manifest["cases"]) == 3
    patient01 = [case for case in manifest["cases"] if case["patient_id"] == "patient01"]
    patient02 = [case for case in manifest["cases"] if case["patient_id"] == "patient02"]
    assert len(patient01) == 2 and len(patient02) == 1
    assert len({case["split"] for case in patient01}) == 1
    assert patient01[0]["split"] != patient02[0]["split"]


def test_nnunet_export_preserves_explicit_train_val_split(tmp_path):
    from ml.export_nnunet import main as export

    manifest = create_fixture(tmp_path / "source")
    destination = tmp_path / "Dataset501_Test"
    export(["--manifest", str(manifest), "--output", str(destination)])
    dataset = json.loads((destination / "dataset.json").read_text())
    split = json.loads((destination / "splits_final.json").read_text())[0]
    assert dataset["numTraining"] == 2
    assert list(dataset["channel_names"].values()) == list(MODALITIES)
    assert split == {"train": ["synthetic-000"], "val": ["synthetic-001"]}
    image = nib.load(destination / "imagesTr/synthetic-000_0000.nii.gz")
    label = nib.load(destination / "labelsTr/synthetic-000.nii.gz")
    assert nib.aff2axcodes(image.affine) == ("R", "A", "S")
    np.testing.assert_allclose(image.affine, label.affine)
    assert set(np.unique(label.get_fdata())) == {0, 1, 2, 3, 4}
