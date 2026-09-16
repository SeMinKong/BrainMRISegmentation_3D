"""Preprocessed-volume cache so training epochs are not bound by gzip decoding.

Each case is stored once as ``<cache-dir>/<case_id>.npz`` holding the exact output of
``data.prepare_case`` (canonical RAS, target spacing, nonzero z-score): the four-channel
image as float16, the class-index label as uint8, the target affine, and a key derived
from the source files' path/size/mtime plus spacing and labels. A changed source file,
spacing or label set makes the entry stale and it is rebuilt. Source NIfTI files are never
modified.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

from .schema import MODALITIES, canonical_modalities

CACHE_FORMAT = 1
IMAGE_DTYPE = "float16"


def cache_key(case: dict, spacing, labels: dict) -> str:
    modalities = canonical_modalities(case["modalities"])
    parts = {"format": CACHE_FORMAT, "image_dtype": IMAGE_DTYPE, "spacing": [float(value) for value in spacing],
             "labels": {str(key): str(value) for key, value in labels.items()}, "files": []}
    sources = [(name, modalities[name]) for name in MODALITIES] + [("label", case.get("label"))]
    for name, value in sources:
        if not value:
            parts["files"].append([name, None])
            continue
        path = Path(value)
        stat = path.stat()
        parts["files"].append([name, str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:24]


def cache_path(cache_dir: Path, case: dict) -> Path:
    return Path(cache_dir) / f"{case['case_id']}.npz"


def is_fresh(cache_dir: Path, case: dict, spacing, labels: dict) -> bool:
    """Read only the small key member; never decompress the volume to check validity."""
    import numpy as np

    path = cache_path(cache_dir, case)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["key"]) == cache_key(case, spacing, labels)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def load_cached(cache_dir: Path, case: dict, spacing, labels: dict):
    """Return (image float32 [4,X,Y,Z], label int64 or None, affine) or None when missing/stale."""
    import numpy as np

    path = cache_path(cache_dir, case)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["key"]) != cache_key(case, spacing, labels):
                return None
            image = np.ascontiguousarray(data["image"].astype(np.float32))
            label = np.ascontiguousarray(data["label"].astype(np.int64)) if "label" in data.files else None
            affine = data["affine"]
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    return image, label, affine


def build_one(cache_dir: Path, case: dict, spacing, labels: dict) -> str:
    """Preprocess one case and write it atomically. Safe to run in a worker process."""
    import numpy as np
    from .data import prepare_case

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(case, spacing, labels)
    image, label, affine, _ = prepare_case(case["modalities"], tuple(spacing), case.get("label"), labels)
    arrays = {"key": np.array(key), "image": image.astype(np.float16), "affine": np.asarray(affine, dtype=np.float64)}
    if label is not None:
        if label.min() < 0 or label.max() > 255:
            raise ValueError("Class indices must fit uint8")
        arrays["label"] = label.astype(np.uint8)
    target = cache_path(cache_dir, case)
    temporary = target.with_name(f"{target.stem}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, target)
    return case["case_id"]


def default_workers() -> int:
    return max(1, min(6, (os.cpu_count() or 2) - 2))


def ensure_cache(cache_dir: Path, cases: list[dict], spacing, labels: dict, *, workers: int | None = None,
                 report=None) -> dict:
    """Build missing/stale entries in parallel. Raises if any case fails, so no case is silently dropped."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pending = [case for case in cases if not is_fresh(cache_dir, case, spacing, labels)]
    summary = {"cache_dir": str(cache_dir.resolve()), "cases": len(cases), "fresh": len(cases) - len(pending),
               "built": 0, "workers": workers or default_workers(), "seconds": 0.0}
    if report:
        report({"event": "cache_check", **summary})
    if pending:
        failures = []
        with ProcessPoolExecutor(max_workers=summary["workers"]) as pool:
            futures = {pool.submit(build_one, cache_dir, case, tuple(spacing), labels): case["case_id"] for case in pending}
            for future in as_completed(futures):
                case_id = futures[future]
                try:
                    future.result()
                    summary["built"] += 1
                    if report and summary["built"] % 25 == 0:
                        report({"event": "cache_progress", "built": summary["built"], "pending": len(pending),
                                "seconds": round(time.monotonic() - started, 1)})
                except Exception as exc:  # noqa: BLE001 - collected and re-raised below
                    failures.append(f"{case_id}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError(f"{len(failures)} case(s) could not be cached; fix or exclude them: " + "; ".join(failures[:5]))
    summary["seconds"] = round(time.monotonic() - started, 1)
    if report:
        report({"event": "cache_ready", **summary})
    return summary
