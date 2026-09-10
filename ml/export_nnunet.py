"""Export a validated manifest to nnU-Net v2 raw dataset layout and fixed split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .schema import MODALITIES, load_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New DatasetXXX_Name folder under nnUNet_raw")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    if not re.fullmatch(r"Dataset\d{3}_[A-Za-z0-9_]+", args.output.name):
        raise ValueError("output name must follow DatasetXXX_Name")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty dataset folder")
    if list(map(int, manifest["labels"])) != list(range(len(manifest["labels"]))):
        raise ValueError("nnU-Net export requires consecutive labels starting at 0")
    import nibabel as nib
    import numpy as np
    from .data import load_volume

    splits = {"train": [], "val": []}
    allowed = set(map(int, manifest["labels"]))
    # Validate all case IDs and headers before creating outputs.
    for case in manifest["cases"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", case["case_id"]):
            raise ValueError("case_id must contain only letters, numbers, hyphens or underscores")
        reference = load_volume(Path(case["modalities"]["t1"]))
        for path in [*case["modalities"].values(), case["label"]]:
            volume = load_volume(Path(path))
            if volume.shape != reference.shape or not np.allclose(volume.affine, reference.affine, atol=1e-3):
                raise ValueError(f"Grid mismatch in {case['case_id']}")
        label_data = load_volume(Path(case["label"])).get_fdata()
        if not np.array_equal(label_data, np.rint(label_data)) or not set(np.unique(label_data).astype(int)).issubset(allowed):
            raise ValueError(f"Invalid labels in {case['case_id']}")
    for case in manifest["cases"]:
        suffix = "Ts" if case["split"] == "test" else "Tr"
        images_dir, labels_dir = args.output / f"images{suffix}", args.output / f"labels{suffix}"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(MODALITIES):
            image = nib.as_closest_canonical(nib.load(case["modalities"][name]))
            nib.save(image, images_dir / f"{case['case_id']}_{index:04d}.nii.gz")
        label = nib.as_closest_canonical(nib.load(case["label"]))
        nib.save(nib.Nifti1Image(label.get_fdata().astype(np.uint8), label.affine), labels_dir / f"{case['case_id']}.nii.gz")
        if case["split"] in splits:
            splits[case["split"]].append(case["case_id"])
    dataset = {"channel_names": {str(index): name for index, name in enumerate(MODALITIES)},
               "labels": {name: int(label) for label, name in manifest["labels"].items()},
               "numTraining": len(splits["train"]) + len(splits["val"]), "file_ending": ".nii.gz"}
    (args.output / "dataset.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    (args.output / "splits_final.json").write_text(json.dumps([splits], indent=2), encoding="utf-8")
    print(f"Exported {dataset['numTraining']} train/val volumes to {args.output}")
    print("After nnUNetv2_plan_and_preprocess, copy splits_final.json into the matching nnUNet_preprocessed dataset directory BEFORE training fold 0.")


if __name__ == "__main__":
    main()
