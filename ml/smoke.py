"""Exercise training + registered-checkpoint inference on tiny synthetic volumes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def create_fixture(output: Path) -> Path:
    import nibabel as nib
    import numpy as np
    from .schema import DEFAULT_LABELS, MODALITIES

    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    grid = np.indices((16, 16, 16))
    radius = np.sqrt(sum((axis - 7.5) ** 2 for axis in grid))
    cases = []
    for index, split in enumerate(("train", "val")):
        case_id = f"synthetic-{index:03d}"
        folder = output / case_id
        folder.mkdir(parents=True, exist_ok=True)
        # A flipped native axis exercises orientation inversion during inference.
        affine = np.diag([-1.0, 1.0, 1.0, 1.0])
        affine[:3, 3] = [15.0, -8.0, -8.0]
        label = np.zeros(radius.shape, dtype=np.uint8)
        label[radius < 5] = 2
        label[radius < 3] = 1
        label[(radius < 2) & (grid[0] > 7)] = 3
        label[(radius < 1.5) & (grid[0] <= 7)] = 4
        modalities = {}
        for channel, name in enumerate(MODALITIES):
            image = (40 + 4 * label + 3 * grid[channel % 3] + rng.normal(0, 1, radius.shape)).astype(np.float32)
            image[radius > 7.5] = 0
            path = folder / f"{case_id}-{name}.nii.gz"
            nib.save(nib.Nifti1Image(image, affine), str(path))
            modalities[name] = str(path.resolve())
        label_path = folder / f"{case_id}-seg.nii.gz"
        nib.save(nib.Nifti1Image(label, affine), str(label_path))
        cases.append({"case_id": case_id, "patient_id": case_id, "split": split,
                      "modalities": modalities, "label": str(label_path.resolve())})
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"labels": DEFAULT_LABELS, "synthetic": True, "cases": cases}, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/smoke"))
    parser.add_argument("--fixture-only", action="store_true")
    args = parser.parse_args(argv)
    manifest = create_fixture(args.output / "fixture")
    if args.fixture_only:
        print(manifest)
        return
    from .train import main as train
    train(["--manifest", str(manifest), "--output", str(args.output), "--epochs", "1", "--max-steps", "1",
           "--patches-per-case", "1", "--patch-size", "16", "16", "16", "--channels", "4", "8", "16",
           "--threads", "2", "--device", "cpu"])
    from .adapters import run_inference
    import nibabel as nib
    import numpy as np
    content = json.loads(manifest.read_text(encoding="utf-8"))
    case = content["cases"][1]
    old_checkpoint, old_device = os.environ.get("MRI_UNET_CHECKPOINT"), os.environ.get("MRI_DEVICE")
    try:
        os.environ["MRI_UNET_CHECKPOINT"] = str((args.output / "best.pt").resolve())
        os.environ["MRI_DEVICE"] = "cpu"
        result = args.output / "synthetic-prediction.nii.gz"
        metadata = run_inference("unet3d", case["modalities"], result)
    finally:
        for key, previous in (("MRI_UNET_CHECKPOINT", old_checkpoint), ("MRI_DEVICE", old_device)):
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
    predicted, original = nib.load(str(result)), nib.load(case["modalities"]["t1"])
    assert predicted.shape == original.shape
    np.testing.assert_allclose(predicted.affine, original.affine)
    assert set(np.unique(predicted.get_fdata())).issubset({0, 1, 2, 3, 4})
    assert np.issubdtype(predicted.get_data_dtype(), np.integer)
    print(json.dumps({"smoke": "passed", "training_step": metadata["training_step"],
                      "output": str(result), "synthetic_only": True}), flush=True)


if __name__ == "__main__":
    main()
