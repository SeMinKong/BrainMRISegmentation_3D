"""Filesystem-backed local case store. Public API values never expose paths.

Cases come from three sources:

* ``uploaded`` – NIfTI files posted through the web import; copies are saved as canonical RAS
  under ``<root>/<case_id>/``.
* ``linked`` – MU-Glioma-Post studies registered from a training manifest. The original files stay
  where they are (LPS, untouched); only ``case.json`` and prediction masks live under the store root
  and volumes are reoriented to RAS when read.
* ``synthetic`` – the mathematical phantom, created only when no real case is available or when
  ``MRI_DEMO_CASE=always``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from collections import OrderedDict
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Iterable
from uuid import uuid4

import nibabel as nib
import numpy as np

from .volumes import LABELS, LABEL_PRESETS, canonical_geometry, is_safe_case_id, label_volumes, make_phantom, read_canonical, validate_matching_grid

log = logging.getLogger(__name__)

DEMO_CASE_ID = "demo-brain-001"
DEMO_POLICIES = ("auto", "always", "never")
# Training manifests use ml.schema names; the web store uses release suffixes.
MANIFEST_MODALITY_KEYS = {"t1": "t1n", "t1n": "t1n", "t1ce": "t1c", "t1c": "t1c", "t2": "t2w", "t2w": "t2w", "flair": "t2f", "t2f": "t2f"}
MODALITY_ORDER = ("t1n", "t1c", "t2w", "t2f")


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


def _display_name(case_id: str, patient_id: str | None, split: str | None) -> str:
    name = case_id
    parts = case_id.split("_Timepoint_")
    if patient_id and len(parts) == 2 and parts[0] == patient_id:
        name = f"{patient_id.replace('PatientID_', 'Patient ')} · Timepoint {parts[1]}"
    return f"{name} · {split}" if split else name


class CaseStore:
    def __init__(self, root: Path, *, source_manifest: Path | None = None, demo: str = "auto"):
        if demo not in DEMO_POLICIES:
            raise ValueError(f"demo policy must be one of {', '.join(DEMO_POLICIES)}")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = 192 * 1024 * 1024
        self.cases: dict[str, dict] = {}
        self.source_manifest = source_manifest.resolve() if source_manifest else None
        self.link_report: dict = {"manifest": str(self.source_manifest) if self.source_manifest else None,
                                  "linked": 0, "skipped": 0, "errors": []}
        for metadata in self.root.glob("*/case.json"):
            try:
                item = json.loads(metadata.read_text(encoding="utf-8"))
                if item["id"] == metadata.parent.name:
                    self.cases[item["id"]] = _normalize_case_metadata(item)
            except (ValueError, KeyError, OSError):
                continue
        if self.source_manifest is not None:
            self.link_manifest(self.source_manifest)
        has_real = any(not case.get("demo") for case in self.cases.values())
        if demo == "always" or (demo == "auto" and not has_real):
            self.ensure_demo()
        else:
            # Files stay on disk for MRI_DEMO_CASE=always; the phantom is simply not listed.
            for case_id in [key for key, case in self.cases.items() if case.get("demo")]:
                del self.cases[case_id]

    # ----------------------------------------------------------------- demo
    def ensure_demo(self) -> None:
        if DEMO_CASE_ID in self.cases:
            return
        volumes, mask, prediction, affine = make_phantom()
        prepared = [(key, nib.Nifti1Image(data, affine)) for key, data in volumes.items()]
        case = self.create_case("Synthetic brain study", prepared, nib.Nifti1Image(mask, affine),
                                "mu_glioma_post", case_id=DEMO_CASE_ID, demo=True)
        self.add_segmentation(case["id"], "demo_prediction", "Demo · perturbed reference", "demo",
                              nib.Nifti1Image(prediction, affine))

    def has_demo(self) -> bool:
        with self.lock:
            return any(case.get("demo") for case in self.cases.values())

    # -------------------------------------------------------------- linking
    def link_manifest(self, manifest_path: Path) -> dict:
        """Register every case of a training manifest by reference. Originals are never copied or rewritten."""
        started = time.monotonic()
        manifest_path = manifest_path.resolve()
        content = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = content.get("cases")
        if not isinstance(cases, list):
            raise ValueError("Manifest must contain a cases list")
        base = manifest_path.parent
        linked = skipped = 0
        errors: list[str] = []
        for raw in cases:
            case_id = raw.get("case_id") if isinstance(raw, dict) else None
            try:
                if not isinstance(case_id, str) or not is_safe_case_id(case_id):
                    raise ValueError("case_id must be a safe identifier (letters, digits, '_', '-', '.')")
                files = {}
                for key, value in (raw.get("modalities") or {}).items():
                    modality = MANIFEST_MODALITY_KEYS.get(str(key).lower())
                    if modality is None or modality in files:
                        raise ValueError(f"Unknown or duplicate modality '{key}'")
                    files[modality] = self._resolve(base, value)
                if not files:
                    raise ValueError("At least one MRI modality is required")
                if raw.get("label"):
                    files["reference"] = self._resolve(base, raw["label"])
                self._link_case(case_id, files, patient_id=raw.get("patient_id"), split=raw.get("split"),
                                manifest=manifest_path.name)
                linked += 1
            except (ValueError, OSError, KeyError, nib.filebasedimages.ImageFileError) as exc:
                skipped += 1
                errors.append(f"{case_id}: {exc}")
        self.link_report = {"manifest": str(manifest_path), "linked": linked, "skipped": skipped, "errors": errors[:50],
                            "seconds": round(time.monotonic() - started, 2)}
        if errors:
            log.warning("Linked %d cases from %s; skipped %d (first: %s)", linked, manifest_path.name, skipped, errors[0])
        else:
            log.info("Linked %d cases from %s in %.1fs", linked, manifest_path.name, self.link_report["seconds"])
        return dict(self.link_report)

    @staticmethod
    def _resolve(base: Path, value) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Missing file path")
        path = Path(value)
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path.name}")
        if not path.name.lower().endswith((".nii", ".nii.gz")):
            raise ValueError(f"Not a NIfTI file: {path.name}")
        return path

    def _link_case(self, case_id: str, files: dict[str, Path], *, patient_id: str | None, split: str | None,
                   manifest: str) -> dict:
        modalities = [key for key in MODALITY_ORDER if key in files]
        geometry = {}
        for key, path in files.items():
            image = nib.load(str(path), mmap=False)  # header only; voxels stay compressed on disk
            geometry[key] = canonical_geometry(image)
            unit = image.header.get_xyzt_units()[0]
            if unit not in ("mm", "unknown"):
                raise ValueError(f"{path.name}: spatial units must be millimetres")
        shape, affine = geometry[modalities[0]]
        for key, (other_shape, other_affine) in geometry.items():
            if other_shape != shape or not np.allclose(other_affine, affine, atol=1e-3, rtol=1e-5):
                raise ValueError(f"{key} grid differs from {modalities[0]} after RAS canonicalization")
        previous = self.cases.get(case_id)
        if previous is not None and previous.get("source") not in ("linked",):
            raise ValueError("case_id collides with an uploaded or synthetic case")
        directory = self.root / case_id
        directory.mkdir(parents=True, exist_ok=True)
        # Keep predictions produced earlier for this study; the reference always follows the manifest.
        kept = [seg for seg in (previous or {}).get("segmentations", [])
                if seg["id"] != "reference" and (directory / f"seg-{seg['id']}.nii.gz").is_file()]
        segmentations = ([{"id": "reference", "name": "Reference mask", "kind": "reference",
                           "created_at": (previous or {}).get("created_at") or utc_now()}] if "reference" in files else []) + kept
        case = {"id": case_id, "name": _display_name(case_id, patient_id, split)[:120], "source": "linked", "demo": False,
                "shape": list(shape), "spacing": nib.affines.voxel_sizes(affine).tolist(), "affine": affine.tolist(),
                "orientation": "RAS", "modalities": modalities, "segmentations": segmentations,
                "labels": LABEL_PRESETS["mu_glioma_post"]["labels"], "label_preset": "mu_glioma_post",
                "created_at": (previous or {}).get("created_at") or utc_now(), "spatial_units": "mm",
                "original_geometry_note": "Linked source file; reoriented to canonical RAS on read, originals untouched.",
                "description": "MU-Glioma-Post study linked from the local dataset folder. The reference mask is the released tumorMask, not a model prediction.",
                "study": {"patient_id": patient_id, "split": split, "manifest": manifest},
                "files": {key: str(path) for key, path in files.items()}}
        if (previous or {}).get("reference_summary") and "reference" in files:
            case["reference_summary"] = previous["reference_summary"]  # computed once; survives restarts
        with self.lock:
            self._persist(case)
        return self.get(case_id)

    # --------------------------------------------------------------- access
    def directory(self, case_id: str) -> Path:
        # IDs are generated locally or validated on link; dictionary membership also prevents traversal.
        self.get(case_id)
        return self.root / case_id

    def _internal(self, case_id: str) -> dict:
        with self.lock:
            if case_id not in self.cases:
                raise KeyError("Case not found")
            return self.cases[case_id]

    def get(self, case_id: str) -> dict:
        with self.lock:
            case = json.loads(json.dumps(self._internal(case_id)))
        case.pop("files", None)
        return case

    def list(self) -> list[dict]:
        with self.lock:
            return [self.get(case_id) for case_id in self.cases]

    def _persist(self, case: dict) -> None:
        """Atomic write of case.json; safe when a second server process shares the same store directory.

        The temp name carries pid and thread so two processes never write the same file, and the final
        replace retries because Windows refuses to swap a file another process is reading at that instant.
        """
        directory = self.root / case["id"]
        target = directory / "case.json"
        payload = json.dumps(case, indent=2)
        self.cases[case["id"]] = case
        try:
            if target.is_file() and target.read_text(encoding="utf-8") == payload:
                return  # unchanged (typical for relinking at startup): no write, no collision
        except OSError:
            pass
        temporary = directory / f"case.json.{os.getpid()}.{threading.get_ident()}.tmp"
        temporary.write_text(payload, encoding="utf-8")
        for attempt in range(6):
            try:
                temporary.replace(target)
                return
            except PermissionError:
                if attempt == 5:
                    temporary.unlink(missing_ok=True)
                    raise
                time.sleep(0.05 * (attempt + 1))

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

    def is_linked_path(self, path: Path) -> bool:
        return not path.resolve().is_relative_to(self.root)

    def modality_path(self, case_id: str, modality: str) -> Path:
        case = self._internal(case_id)
        if modality not in case["modalities"]:
            raise KeyError(f"Modality '{modality}' not available")
        linked = case.get("files", {}).get(modality)
        return Path(linked) if linked else self.directory(case_id) / f"{modality}.nii.gz"

    def segmentation_path(self, case_id: str, segmentation: str) -> Path:
        case = self._internal(case_id)
        if segmentation not in [item["id"] for item in case["segmentations"]]:
            raise KeyError("Segmentation not found")
        linked = case.get("files", {}).get(segmentation)
        return Path(linked) if linked else self.directory(case_id) / f"seg-{segmentation}.nii.gz"

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
        image = read_canonical(path)
        if mask:
            raw = np.asarray(image.dataobj)
            if not np.isin(raw, [0, 1, 2, 3, 4]).all():
                raise ValueError("Segmentation must contain integer labels 0, 1, 2, 3, 4.")
            array = raw.astype(np.uint8)
        else:
            array = image.get_fdata(dtype=np.float32)
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
        if seg_id in self._internal(case_id).get("files", {}):
            raise ValueError("The linked reference mask is read-only.")
        shape, affine = canonical_geometry(nib.load(str(self.modality_path(case_id, case["modalities"][0])), mmap=False))
        if tuple(image.shape) != shape or not np.allclose(image.affine, affine, atol=1e-3, rtol=1e-5):
            raise ValueError("All modalities and segmentation must share the same affine/spacing. Register and resample them first.")
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all() or not np.allclose(data, np.round(data)) or not np.isin(data, [0] + [label["id"] for label in case["labels"]]).all():
            raise ValueError("Output segmentation contains labels incompatible with this case preset.")
        output = nib.Nifti1Image(data.astype(np.uint8), affine)
        output.header.set_xyzt_units("mm")
        nib.save(output, str(self.directory(case_id) / f"seg-{seg_id}.nii.gz"))
        item = {"id": seg_id, "name": name, "kind": kind, "created_at": utc_now()}
        if provenance:
            item["provenance"] = provenance
        with self.lock:
            case = self._internal(case_id)
            case["segmentations"] = [segment for segment in case["segmentations"] if segment["id"] != seg_id] + [item]
            self._persist(case)
        return item

    def record_metrics(self, case_id: str, seg_id: str, metrics: dict) -> None:
        """Attach precomputed comparison metrics (vs the reference) to a segmentation so overviews need no recompute."""
        with self.lock:
            case = self._internal(case_id)
            for segment in case["segmentations"]:
                if segment["id"] == seg_id:
                    segment["metrics"] = metrics
                    break
            else:
                raise KeyError("Segmentation not found")
            self._persist(case)

    def reference_summary(self, case_id: str) -> dict | None:
        """Per-label volumes of the reference mask, computed once and stored in case.json."""
        case = self._internal(case_id)
        if not any(segment["id"] == "reference" for segment in case["segmentations"]):
            return None
        summary = case.get("reference_summary")
        if summary is None:
            mask = self.read_segmentation(case_id, "reference")
            summary = label_volumes(mask, np.asarray(case["affine"]))
            with self.lock:
                case = self._internal(case_id)
                case["reference_summary"] = summary
                self._persist(case)
        return summary

    def warm_reference_summaries(self, stop: threading.Event | None = None) -> int:
        """Background pass that fills reference summaries for every case (about 0.15 s each), skipping work on repeat runs."""
        done = 0
        for case_id in list(self.cases):
            if stop is not None and stop.is_set():
                break
            try:
                if self._internal(case_id).get("reference_summary") is None and self.reference_summary(case_id) is not None:
                    done += 1
            except (KeyError, OSError, ValueError) as exc:
                log.warning("reference summary skipped for %s: %s", case_id, exc)
        return done
