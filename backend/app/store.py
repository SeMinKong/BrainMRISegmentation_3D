"""Filesystem-backed local case store. Public API values never expose paths."""
from __future__ import annotations

from datetime import datetime, timezone
from collections import OrderedDict
import json
from pathlib import Path
import threading
from typing import Iterable
from uuid import uuid4

import nibabel as nib
import numpy as np

from .volumes import LABELS, LABEL_PRESETS, make_phantom, validate_matching_grid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_case_metadata(case: dict) -> dict:
    """Resolve saved preset names by label meaning without rewriting source files."""
    definitions = case.get("labels", [])
    semantics = {}
    for label in definitions:
        name = str(label.get("name", "")).strip().upper()
        if label.get("id") == 2 and name == "SNFH / EDEMA":
            name = "SNFH"
        semantics[label.get("id")] = name
    expected = {label["id"]: label["name"] for label in LABELS}
    if case.get("label_preset") != "generic" and len(definitions) == len(expected) and semantics == expected:
        case["label_preset"] = "mu_glioma_post"
        case["labels"] = [dict(label) for label in LABELS]
    else:
        # Preserve unfamiliar label meanings for viewing; never enable model inference by guessing.
        case["label_preset"] = "generic"
    return case


class CaseStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = 192 * 1024 * 1024
        self.cases: dict[str, dict] = {}
        for metadata in self.root.glob("*/case.json"):
            try:
                item = json.loads(metadata.read_text(encoding="utf-8"))
                if item["id"] == metadata.parent.name:
                    self.cases[item["id"]] = _normalize_case_metadata(item)
            except (ValueError, KeyError, OSError):
                continue
        self.ensure_demo()

    def ensure_demo(self) -> None:
        if "demo-brain-001" in self.cases:
            return
        volumes, mask, prediction, affine = make_phantom()
        prepared = [(key, nib.Nifti1Image(data, affine)) for key, data in volumes.items()]
        case = self.create_case("Synthetic brain study", prepared, nib.Nifti1Image(mask, affine),
                                "mu_glioma_post", case_id="demo-brain-001", demo=True)
        self.add_segmentation(case["id"], "demo_prediction", "Demo · perturbed reference", "demo",
                              nib.Nifti1Image(prediction, affine))

    def directory(self, case_id: str) -> Path:
        # IDs are generated locally; dictionary membership also prevents traversal.
        self.get(case_id)
        return self.root / case_id

    def get(self, case_id: str) -> dict:
        with self.lock:
            if case_id not in self.cases:
                raise KeyError("Case not found")
            return json.loads(json.dumps(self.cases[case_id]))

    def list(self) -> list[dict]:
        with self.lock:
            return [self.get(case_id) for case_id in self.cases]

    def _persist(self, case: dict) -> None:
        directory = self.root / case["id"]
        temporary = directory / "case.json.tmp"
        temporary.write_text(json.dumps(case, indent=2), encoding="utf-8")
        temporary.replace(directory / "case.json")
        self.cases[case["id"]] = case

    def create_case(self, name: str, modalities: Iterable[tuple[str, nib.Nifti1Image]],
                    mask: nib.Nifti1Image | None, label_preset: str, *,
                    case_id: str | None = None, demo: bool = False) -> dict:
        images = dict(modalities)
        if not images:
            raise ValueError("At least one MRI volume is required.")
        reference = next(iter(images.values()))
        for image in images.values():
            validate_matching_grid(reference, image)
        if mask is not None:
            validate_matching_grid(reference, mask)
            allowed = [0] + [item["id"] for item in LABEL_PRESETS[label_preset]["labels"]]
            if not np.isin(np.asarray(mask.dataobj), allowed).all():
                raise ValueError(f"Mask labels do not match selected preset '{label_preset}'. No automatic label conversion is performed.")
        case_id = case_id or f"case-{uuid4().hex[:16]}"
        directory = self.root / case_id
        directory.mkdir(parents=True, exist_ok=False)
        case = {"id": case_id, "name": name.strip()[:120] or "Imported MRI", "source": "synthetic" if demo else "uploaded",
                "demo": demo, "shape": list(reference.shape), "spacing": nib.affines.voxel_sizes(reference.affine).tolist(),
                "affine": reference.affine.tolist(), "orientation": "RAS", "modalities": list(images),
                "segmentations": [], "labels": LABEL_PRESETS[label_preset]["labels"], "label_preset": label_preset,
                "created_at": utc_now(), "spatial_units": "mm", "original_geometry_note": "Canonical RAS; full affine retained on export.",
                "description": "Mathematical phantom for UI exploration. Not patient data; demo masks are not learned predictions." if demo else "Local NIfTI import. Label meanings follow the explicitly selected preset."}
        for key, image in images.items():
            image.header.set_xyzt_units("mm")
            nib.save(image, str(directory / f"{key}.nii.gz"))
        with self.lock:
            self._persist(case)
        if mask is not None:
            self.add_segmentation(case_id, "reference", "Reference mask", "reference", mask)
        return self.get(case_id)

    def modality_path(self, case_id: str, modality: str) -> Path:
        case = self.get(case_id)
        if modality not in case["modalities"]:
            raise KeyError(f"Modality '{modality}' not available")
        return self.directory(case_id) / f"{modality}.nii.gz"

    def segmentation_path(self, case_id: str, segmentation: str) -> Path:
        case = self.get(case_id)
        if segmentation not in [item["id"] for item in case["segmentations"]]:
            raise KeyError("Segmentation not found")
        return self.directory(case_id) / f"seg-{segmentation}.nii.gz"

    def read_modality(self, case_id: str, modality: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        case = self.get(case_id)
        modality = modality or ("t1c" if "t1c" in case["modalities"] else case["modalities"][0])
        return self._cached_image(self.modality_path(case_id, modality), mask=False)

    def read_segmentation(self, case_id: str, segmentation: str) -> np.ndarray:
        return self._cached_image(self.segmentation_path(case_id, segmentation), mask=True)[0]

    def _cached_image(self, path: Path, *, mask: bool) -> tuple[np.ndarray, np.ndarray]:
        """Bounded shared cache keeps linked slice navigation responsive."""
        key = (str(path), path.stat().st_mtime_ns, mask)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        image = nib.load(str(path), mmap=False)
        array = np.asarray(image.dataobj, dtype=np.uint8) if mask else image.get_fdata(dtype=np.float32)
        array.setflags(write=False)
        value = (array, image.affine)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            if array.nbytes <= self.cache_limit:
                while self.cache and self.cache_bytes + array.nbytes > self.cache_limit:
                    _, old = self.cache.popitem(last=False)
                    self.cache_bytes -= old[0].nbytes
                self.cache[key] = value
                self.cache_bytes += array.nbytes
        return value

    def add_segmentation(self, case_id: str, seg_id: str, name: str, kind: str, image: nib.Nifti1Image,
                         provenance: dict | None = None) -> dict:
        case = self.get(case_id)
        if not seg_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Invalid segmentation identifier")
        reference = nib.load(str(self.modality_path(case_id, case["modalities"][0])))
        validate_matching_grid(reference, image)
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all() or not np.allclose(data, np.round(data)) or not np.isin(data, [0] + [label["id"] for label in case["labels"]]).all():
            raise ValueError("Output segmentation contains labels incompatible with this case preset.")
        output = nib.Nifti1Image(data.astype(np.uint8), reference.affine)
        output.header.set_xyzt_units("mm")
        nib.save(output, str(self.directory(case_id) / f"seg-{seg_id}.nii.gz"))
        item = {"id": seg_id, "name": name, "kind": kind, "created_at": utc_now()}
        if provenance:
            item["provenance"] = provenance
        with self.lock:
            case = self.get(case_id)
            case["segmentations"] = [segment for segment in case["segmentations"] if segment["id"] != seg_id] + [item]
            self._persist(case)
        return item
