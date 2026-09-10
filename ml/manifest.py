"""Discover NIfTI cases using explicit filename and patient identity conventions.

This optional helper handles <case>_seg or <case>-seg naming. Use a manually
reviewed manifest for other layouts; no dataset release naming is assumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

from .schema import DEFAULT_LABELS, load_manifest, validate_labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patient-regex", required=True,
                        help="Explicit regex with exactly one capture group defining patient ID across sessions. Unmatched IDs are rejected.")
    parser.add_argument("--labels", type=Path, help="JSON map of integer label IDs to names; defaults to MU-Glioma-Post NETC/SNFH/ET/RC")
    args = parser.parse_args(argv)
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")
    if not args.data_root.is_dir():
        raise ValueError("data-root does not exist")
    labels = validate_labels(json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else DEFAULT_LABELS)
    pattern = re.compile(args.patient_regex)
    if pattern.groups != 1:
        raise ValueError("patient-regex must contain exactly one capture group for the patient ID")
    cases = []
    for label in sorted(args.data_root.rglob("*seg.nii*")):
        match = re.fullmatch(r"(.+)[_-]seg\.(nii|nii\.gz)", label.name)
        if not match:
            continue
        case_id = match.group(1)
        patient = pattern.fullmatch(case_id)
        if patient is None or not patient.group(1) or not patient.group(1).strip():
            raise ValueError(f"Cannot derive patient ID for {case_id}; supply --patient-regex with one capture group")
        modalities = {}
        for name, suffixes in {"t1": ("t1n", "t1"), "t1ce": ("t1c", "t1ce"), "t2": ("t2w", "t2"), "flair": ("t2f", "flair")}.items():
            found = [label.parent / f"{case_id}{separator}{suffix}.{extension}"
                     for separator in ("_", "-") for suffix in suffixes for extension in ("nii.gz", "nii")]
            found = [path for path in found if path.is_file()]
            if len(found) != 1:
                raise ValueError(f"Expected one {name} volume for {case_id}, found {len(found)}")
            modalities[name] = str(found[0].resolve())
        cases.append({"case_id": case_id, "patient_id": patient.group(1), "split": "train",
                      "modalities": modalities, "label": str(label.resolve())})
    patients = sorted({case["patient_id"] for case in cases})
    if len(patients) < 2:
        raise ValueError("At least two independently identified patients are needed for train/val splits")
    random.Random(args.seed).shuffle(patients)
    count = max(1, min(len(patients) - 1, round(len(patients) * args.val_fraction)))
    validation = set(patients[:count])
    for case in cases:
        case["split"] = "val" if case["patient_id"] in validation else "train"
    output = {"labels": labels, "split_seed": args.seed, "patient_regex": args.patient_regex, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    load_manifest(args.output)
    print(f"Saved {len(cases)} cases / {len(patients)} patients ({count} validation patients): {args.output}")
    print("Audit patient_id grouping against your dataset release before training; a filename convention is not a patient identity guarantee.")


if __name__ == "__main__":
    main()
