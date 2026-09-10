"""Read-only, full-volume audit of the observed MU-Glioma-Post release layout.

Only reports are written. Original NIfTI files are never renamed or modified.
Run from the project root with the project's Python environment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import nibabel as nib
import numpy as np


PATTERN = re.compile(r"(PatientID_\d+)_(Timepoint_\d+)_(brain_t1n|brain_t1c|brain_t2w|brain_t2f|tumorMask)\.nii\.gz")
ROLES = {"brain_t1n": "t1", "brain_t1c": "t1ce", "brain_t2w": "t2", "brain_t2f": "flair", "tumorMask": "label"}


def inspect_file(path: Path, root: Path) -> dict:
    before = path.stat()
    row = {"path": path.relative_to(root).as_posix(), "bytes": before.st_size,
           "mtime_ns": before.st_mtime_ns, "errors": [], "warnings": []}
    match = PATTERN.fullmatch(path.name)
    if not match:
        row["errors"].append("Unexpected release filename")
        return row
    patient, timepoint, token = match.groups()
    row.update(patient_id=patient, timepoint=timepoint, case_id=f"{patient}_{timepoint}", role=ROLES[token])
    if path.relative_to(root).parts != (patient, timepoint, path.name):
        row["errors"].append("Patient/timepoint directories disagree with filename")
    try:
        compressed = path.read_bytes()
        row["sha256"] = hashlib.sha256(compressed).hexdigest()
        payload = gzip.decompress(compressed)  # Reads entire stream and verifies gzip CRC/length.
        del compressed
        row["gzip_crc_ok"] = True
        image = nib.Nifti1Image.from_bytes(payload)
        row.update(shape=list(image.shape), dtype=str(image.get_data_dtype()),
                   spacing=nib.affines.voxel_sizes(image.affine).tolist(),
                   units=image.header.get_xyzt_units()[0],
                   orientation="".join(nib.aff2axcodes(image.affine)), affine=image.affine.tolist())
        if len(image.shape) != 3 or any(size < 2 for size in image.shape):
            row["errors"].append("Expected a three-dimensional volume")
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
            row["errors"].append("Invalid affine")
        if row["units"] not in {"mm", "unknown"}:
            row["errors"].append("Spatial units are not millimetres")
        elif row["units"] == "unknown":
            row["warnings"].append("Spatial unit unspecified; project assumes mm")
        row["voxels"] = int(np.prod(image.shape))
        canonical = nib.as_closest_canonical(image)
        spacing = nib.affines.voxel_sizes(canonical.affine)
        row["web_grid_compatible"] = bool(
            row["voxels"] <= 32_000_000 and np.all(spacing > 0) and np.all(spacing <= 50)
            and np.allclose(canonical.affine[:3, :3] / spacing, np.eye(3), atol=1e-3)
        )
        if not row["web_grid_compatible"]:
            row["warnings"].append("Outside web grid or voxel limits")
        data = image.get_fdata(dtype=np.float32, caching="unchanged")
        row["finite"] = bool(np.isfinite(data).all())
        if not row["finite"]:
            row["errors"].append("NaN or infinite voxel values")
        else:
            row["minimum"], row["maximum"] = float(data.min()), float(data.max())
            row["nonzero_voxels"] = int(np.count_nonzero(data))
            if row["role"] == "label":
                values, counts = np.unique(data, return_counts=True)
                row["label_counts"] = {str(float(v)): int(c) for v, c in zip(values, counts)}
                if not np.isin(values, [0, 1, 2, 3, 4]).all():
                    row["errors"].append("Mask contains values outside integer labels 0..4")
                if not row["nonzero_voxels"]:
                    row["warnings"].append("Empty foreground mask")
            else:
                nonzero = data[data != 0]
                row["nonzero_std"] = float(nonzero.std()) if nonzero.size else 0.0
                if not nonzero.size or row["nonzero_std"] < 1e-8:
                    row["errors"].append("MRI has empty or constant nonzero intensity")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            row["errors"].append("Source changed while being read; rerun after download completes")
    except Exception as exc:
        row["errors"].append(f"{type(exc).__name__}: {exc}")
    return row


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/MU-Glioma-Post"))
    parser.add_argument("--output", type=Path, default=Path("data/quality-check"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Keep reports outside the original dataset directory")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Dataset is empty or missing")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as pool:
        for index, row in enumerate(pool.map(lambda path: inspect_file(path, root), files), 1):
            rows.append(row)
            if index % 100 == 0 or index == len(files):
                print(f"Checked {index}/{len(files)} files; {time.monotonic() - started:.0f}s; "
                      f"files with errors: {sum(bool(r['errors']) for r in rows)}", flush=True)
    grouped = defaultdict(list)
    for row in rows:
        if "case_id" in row:
            grouped[row["case_id"]].append(row)
    cases = []
    for case_id, entries in sorted(grouped.items()):
        counts = Counter(row["role"] for row in entries)
        missing = [role for role in ROLES.values() if counts[role] == 0]
        duplicate = [role for role, count in counts.items() if count > 1]
        errors = [f"{row['role']}: {error}" for row in entries for error in row["errors"]]
        reference = next((row for row in entries if row["role"] == "t1" and "affine" in row), None)
        grid_ok = reference is not None and all(
            "affine" in row and row["shape"] == reference["shape"] and
            np.allclose(row["affine"], reference["affine"], atol=1e-3, rtol=1e-5) for row in entries)
        if not grid_ok:
            errors.append("Case volumes do not share the same valid shape/affine")
        complete_mri = all(counts[role] == 1 for role in ("t1", "t1ce", "t2", "flair"))
        web_ok = bool(not errors and not duplicate and all(row.get("web_grid_compatible") for row in entries)
                      and len(entries) <= 5 and sum(row.get("voxels", 0) for row in entries) <= 96_000_000
                      and sum(row["bytes"] for row in entries) <= 768 * 1024**2
                      and max(row["bytes"] for row in entries) <= 256 * 1024**2)
        cases.append({"case_id": case_id, "patient_id": entries[0]["patient_id"],
                      "timepoint": entries[0]["timepoint"], "file_count": len(entries),
                      "bytes": sum(row["bytes"] for row in entries), "missing": missing, "duplicate_roles": duplicate,
                      "grid_matches": grid_ok, "errors": errors, "web_compatible": web_ok,
                      "training_ready": complete_mri and not missing and not duplicate and not errors,
                      "mri_complete": complete_mri, "paths": {row["role"]: row["path"] for row in entries}})
    hash_groups = defaultdict(list)
    for row in rows:
        if "sha256" in row:
            hash_groups[row["sha256"]].append(row["path"])
    summary = {"patients": len({row["patient_id"] for row in cases}), "cases": len(cases),
               "files": len(rows), "bytes": sum(row["bytes"] for row in rows),
               "role_counts": dict(Counter(row.get("role", "unknown") for row in rows)),
               "files_with_errors": sum(bool(row["errors"]) for row in rows),
               "files_with_warnings": sum(bool(row["warnings"]) for row in rows),
               "gzip_verified": sum(row.get("gzip_crc_ok", False) for row in rows),
               "cases_with_errors": sum(bool(row["errors"]) for row in cases),
               "complete_mri_cases": sum(row["mri_complete"] for row in cases),
               "training_ready_cases": sum(row["training_ready"] for row in cases),
               "web_compatible_cases": sum(row["web_compatible"] for row in cases),
               "shapes": dict(Counter(str(row.get("shape")) for row in rows)),
               "orientations": dict(Counter(row.get("orientation") for row in rows)),
               "spacings": dict(Counter(str(row.get("spacing")) for row in rows)),
               "units": dict(Counter(row.get("units") for row in rows))}
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "dataset_root": str(root),
              "elapsed_seconds": round(time.monotonic() - started, 2), "summary": summary,
              "incomplete_cases": [row for row in cases if row["missing"] or row["duplicate_roles"]],
              "identical_compressed_files": [group for group in hash_groups.values() if len(group) > 1],
              "cases": cases, "files": rows,
              "limits": ["No publisher-provided per-file checksum list was available for comparison.",
                         "Geometry and value validation do not establish anatomical registration or annotation accuracy.",
                         "Longitudinal timepoints are not independent patients; keep patient IDs disjoint across splits."]}
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output / "inventory.csv", rows, ["path", "patient_id", "case_id", "role", "bytes", "sha256", "shape",
              "spacing", "orientation", "units", "dtype", "gzip_crc_ok", "finite", "minimum", "maximum",
              "nonzero_voxels", "label_counts", "errors", "warnings"])
    write_csv(output / "cases.csv", cases, ["case_id", "patient_id", "timepoint", "file_count", "bytes", "missing",
              "duplicate_roles", "grid_matches", "mri_complete", "training_ready", "web_compatible", "errors"])
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if summary["files_with_errors"] or summary["cases_with_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
