"""Shared, dependency-free model and dataset contracts."""
from __future__ import annotations

import json
from pathlib import Path

MODALITIES = ("t1", "t1ce", "t2", "flair")
ALIASES = {"t1n": "t1", "t1c": "t1ce", "t2w": "t2", "t2f": "flair"}
DEFAULT_LABELS = {"0": "background", "1": "NETC", "2": "SNFH", "3": "ET", "4": "RC"}


def canonical_modalities(modalities: dict) -> dict:
    result = {}
    for key, value in modalities.items():
        name = ALIASES.get(key.lower(), key.lower())
        if name in result:
            raise ValueError(f"Duplicate modality: {name}")
        result[name] = value
    if set(result) != set(MODALITIES):
        raise ValueError(f"Exactly four modalities required: {', '.join(MODALITIES)}")
    return {name: result[name] for name in MODALITIES}


def validate_labels(labels: dict) -> dict[str, str]:
    if not isinstance(labels, dict) or not labels:
        raise ValueError("labels must map integer label IDs to names")
    cleaned = {str(int(key)): str(value) for key, value in labels.items()}
    if len(cleaned) != len(labels) or "0" not in cleaned:
        raise ValueError("Unique label IDs including background 0 are required")
    if not set(map(int, cleaned)).issubset(range(5)) or len(cleaned) < 2:
        raise ValueError("This MVP supports background 0 and configurable labels within 1..4")
    names = [name.strip().lower() for name in cleaned.values()]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Each label needs a distinct nonempty semantic name")
    return dict(sorted(cleaned.items(), key=lambda item: int(item[0])))


def require_mu_glioma_post_labels(labels: dict) -> None:
    """Web MVP labels have fixed semantics; custom training tasks stay CLI-only."""
    if {str(key): str(value).strip().upper() for key, value in labels.items()} != {
        key: value.upper() for key, value in DEFAULT_LABELS.items()
    }:
        raise ValueError("Web inference requires the MU-Glioma-Post mapping 0 background, 1 NETC, 2 SNFH, 3 ET, 4 RC. Custom label semantics are CLI-only.")


def load_manifest(path: Path, *, require_validation: bool = True) -> dict:
    path = path.resolve()
    content = json.loads(path.read_text(encoding="utf-8"))
    labels = validate_labels(content.get("labels", DEFAULT_LABELS))
    if not isinstance(content.get("cases"), list) or not content["cases"]:
        raise ValueError("Manifest must contain a nonempty cases list")
    patients, identifiers, cases = {}, set(), []
    for raw in content["cases"]:
        case = dict(raw)
        case_id, patient_id = case.get("case_id"), case.get("patient_id")
        split = case.get("split")
        if not isinstance(case_id, str) or not isinstance(patient_id, str) or not case_id.strip() or not patient_id.strip() or case_id in identifiers:
            raise ValueError("Each case needs a unique case_id and a patient_id")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for {case_id}: {split}")
        if patient_id in patients and patients[patient_id] != split:
            raise ValueError(f"Patient leakage: {patient_id} occurs across splits")
        patients[patient_id], identifiers = split, identifiers | {case_id}
        paths = canonical_modalities(case.get("modalities", {}))
        paths["label"] = case.get("label")
        for name, value in paths.items():
            if not value:
                raise ValueError(f"Missing {name} for {case_id}")
            resolved = Path(value)
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            if not resolved.is_file():
                raise ValueError(f"Missing file for {case_id}/{name}: {resolved}")
            paths[name] = str(resolved.resolve())
        case["modalities"] = {name: paths[name] for name in MODALITIES}
        case["label"] = paths["label"]
        cases.append(case)
    splits = {case["split"] for case in cases}
    if "train" not in splits or (require_validation and "val" not in splits):
        raise ValueError("Patient-disjoint train and val cases are required")
    return {"labels": labels, "cases": cases, "source": str(path)}
