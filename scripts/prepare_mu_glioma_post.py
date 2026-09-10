"""Build conservative local manifests from the completed full-file audit.

Preserve source files and patient IDs. Quarantine exact content duplicates,
and keep linked patient IDs in the same split without asserting their identity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.schema import DEFAULT_LABELS, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("data/quality-check/audit.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between zero and one")
    audit_bytes = args.audit.read_bytes()
    audit = json.loads(audit_bytes)
    root, output = Path(audit["dataset_root"]).resolve(), args.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError("Write manifests outside the original dataset")
    if audit["summary"]["files_with_errors"] or audit["summary"]["cases_with_errors"]:
        raise ValueError("Resolve file/geometry errors and rerun the audit first")
    original_paths = {row["path"] for row in audit["files"]}
    current_paths = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if current_paths != original_paths:
        raise ValueError("Dataset inventory changed; rerun the audit")
    for row in audit["files"]:
        stat = (root / row["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
            raise ValueError(f"Source changed since audit: {row['path']}")
    patients = sorted({case["patient_id"] for case in audit["cases"]})
    parents = {patient: patient for patient in patients}

    def find(patient):
        while parents[patient] != patient:
            parents[patient] = parents[parents[patient]]
            patient = parents[patient]
        return patient

    by_path = {row["path"]: row for row in audit["files"]}
    reasons, duplicate_details = defaultdict(set), []
    for paths in audit["identical_compressed_files"]:
        rows = [by_path[path] for path in paths]
        group_patients = sorted({row["patient_id"] for row in rows})
        group_cases = sorted({row["case_id"] for row in rows})
        reason = "exact_content_duplicate_across_cases" if len(group_cases) > 1 else "identical_content_across_modalities"
        for case_id in group_cases:
            reasons[case_id].add(reason)
        for patient in group_patients[1:]:
            left, right = sorted((find(group_patients[0]), find(patient)))
            parents[right] = left
        duplicate_details.append({"sha256": rows[0]["sha256"], "paths": paths,
                                  "case_ids": group_cases, "patient_ids": group_patients,
                                  "roles": sorted({row["role"] for row in rows}), "reason": reason})
    eligible = [case for case in audit["cases"] if case["training_ready"] and case["case_id"] not in reasons]
    groups = sorted({find(case["patient_id"]) for case in eligible})
    if len(groups) < 2:
        raise ValueError("At least two independent split groups are required")
    random.Random(args.seed).shuffle(groups)
    val_count = max(1, min(len(groups) - 1, round(len(groups) * args.val_fraction)))
    val_groups = set(groups[:val_count])
    output.mkdir(parents=True, exist_ok=True)

    def relative(path):
        import os
        return Path(os.path.relpath(root / path, output)).as_posix()

    all_cases, selected, unlabeled = [], [], []
    for case in audit["cases"]:
        group = find(case["patient_id"])
        split = "val" if group in val_groups else "train"
        entry = {"case_id": case["case_id"], "patient_id": case["patient_id"],
                 "split": split, "split_group": group,
                 "modalities": {key: relative(path) for key, path in case["paths"].items() if key != "label"},
                 "label": relative(case["paths"]["label"]) if "label" in case["paths"] else None}
        if not case["training_ready"]:
            entry.update(status="reference_mask_not_present", excluded_from_curated=True)
            unlabeled.append(entry)
        else:
            all_cases.append(dict(entry, excluded_from_curated=bool(reasons[case["case_id"]])))
            if not reasons[case["case_id"]]:
                selected.append(entry)
    metadata = {"labels": DEFAULT_LABELS, "split_seed": args.seed,
                "val_fraction_of_split_groups": args.val_fraction,
                "patient_regex": r"^(PatientID_\d+)_Timepoint_\d+$",
                "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
                "split_policy": "Patient IDs sharing exact file content are kept in one split group; this does not establish patient identity.",
                "curation_policy": "Exclude every case with duplicated content across cases or modalities; preserve all source files."}
    target = output / "mu-glioma-post-manifest.json"

    def save(path, content):
        path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")

    save(target, dict(metadata, cases=selected))
    save(output / "mu-glioma-post-all-labeled.json", dict(metadata,
         purpose="Inventory of all labeled cases, including quarantined cases. Use the curated manifest for training.", cases=all_cases))
    save(output / "mu-glioma-post-unlabeled.json", {
         "purpose": "Inventory only; not a supervised training manifest. No masks have been fabricated.", "cases": unlabeled})
    load_manifest(target)
    split_by_patient = {case["patient_id"]: case["split"] for case in all_cases + unlabeled}
    for duplicate in duplicate_details:
        if len({split_by_patient[patient] for patient in duplicate["patient_ids"]}) > 1:
            raise AssertionError("A content-linked patient group crossed splits")
    selected_ids = {case["case_id"] for case in selected}
    if any(reasons[case_id] for case_id in selected_ids):
        raise AssertionError("Quarantined cases entered curated manifest")
    member_groups = defaultdict(list)
    for patient in patients:
        member_groups[find(patient)].append(patient)
    excluded = [{"case_id": case["case_id"], "patient_id": case["patient_id"],
                 "reasons": sorted(reasons[case["case_id"]])} for case in audit["cases"] if reasons[case["case_id"]]]
    report = {"audit_sha256": metadata["audit_sha256"], "source_files_preserved": True,
              "labeled_cases": len(all_cases), "unlabeled_cases": len(unlabeled),
              "quarantined_labeled_cases": len(excluded), "curated_cases": len(selected),
              "curated_patients": len({case["patient_id"] for case in selected}),
              "curated_split_groups": len(groups),
              "split_case_counts": dict(Counter(case["split"] for case in selected)),
              "split_patient_counts": {split: len({case["patient_id"] for case in selected if case["split"] == split}) for split in ("train", "val")},
              "split_group_counts": {"train": len(groups) - val_count, "val": val_count},
              "linked_patient_groups": [members for members in member_groups.values() if len(members) > 1],
              "exact_file_duplicate_groups": duplicate_details, "excluded_cases": excluded,
              "curated_duplicate_files_crossing_splits": 0,
              "notes": ["Exclusion is reversible: restore reviewed cases to a future manifest; originals are untouched.",
                        "Identical compressed-file checks do not detect near-duplicates or recompressed equivalents.",
                        "Source of duplicate content and missing masks is unconfirmed; no relabeling or identity correction was attempted."]}
    quality = output / "quality-check"
    quality.mkdir(parents=True, exist_ok=True)
    save(quality / "curation.json", report)
    with (quality / "excluded-cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, ["case_id", "patient_id", "reasons"])
        writer.writeheader()
        writer.writerows(dict(row, reasons="; ".join(row["reasons"])) for row in excluded)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"exact_file_duplicate_groups", "excluded_cases", "notes"}}, indent=2))


if __name__ == "__main__":
    main()
