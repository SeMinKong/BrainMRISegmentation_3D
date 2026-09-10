"""NIfTI geometry, demonstrator volumes, and reproducible display functions.

All internal volumes are closest-canonical RAS. Mesh vertices are transformed
through the full NIfTI affine into millimetres; slices preserve physical aspect.
The demo is a mathematical phantom, not patient data or a model prediction.
"""
from __future__ import annotations

import io
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

MAX_VOXELS = 32_000_000
LABELS = [
    {"id": 1, "name": "NETC", "color": "#fb7185"},
    {"id": 2, "name": "SNFH", "color": "#fbbf24"},
    {"id": 3, "name": "ET", "color": "#2dd4bf"},
    {"id": 4, "name": "RC", "color": "#a78bfa"},
]
LABEL_PRESETS = {
    "mu_glioma_post": {"id": "mu_glioma_post", "name": "MU-Glioma-Post (NETC / SNFH / ET / RC)", "labels": LABELS},
    "generic": {"id": "generic", "name": "Generic labels 1–4", "labels": [dict(label, name=f"Region {label['id']}") for label in LABELS]},
}
MODALITY_ALIASES = {
    "t1": "t1n", "t1n": "t1n", "t1ce": "t1c", "t1gd": "t1c", "t1c": "t1c",
    "t2": "t2w", "t2w": "t2w", "flair": "t2f", "t2f": "t2f",
    "seg": "seg", "mask": "seg", "segmentation": "seg", "tumormask": "seg",
}


def modality_from_name(filename: str) -> str:
    stem = filename.lower().removesuffix(".gz").removesuffix(".nii")
    token = stem.replace("-", "_").split("_")[-1]
    return MODALITY_ALIASES.get(token, "mri")


def load_nifti(path: Path, *, mask: bool = False) -> nib.Nifti1Image:
    """Inspect header before decompressing voxel data; refuse unsupported grids."""
    # No memory map: Windows must release temporary uploads even if validation raises.
    image = nib.load(str(path), mmap=False)
    if len(image.shape) != 3 or any(n < 2 for n in image.shape):
        raise ValueError("A 3D NIfTI volume with at least 2 voxels per axis is required.")
    if int(np.prod(image.shape, dtype=np.int64)) > MAX_VOXELS:
        raise ValueError(f"Volume exceeds {MAX_VOXELS:,} voxels; crop or downsample it first.")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError("NIfTI affine must be finite and invertible.")
    unit = image.header.get_xyzt_units()[0]
    if unit not in ("mm", "unknown"):
        raise ValueError("Spatial units must be millimetres (mm); unspecified units are interpreted as mm.")
    canonical = nib.as_closest_canonical(image)
    spacing = nib.affines.voxel_sizes(canonical.affine)
    if np.any(spacing <= 0) or np.any(spacing > 50):
        raise ValueError("Voxel spacing must be positive and at most 50 mm.")
    # A slice renderer without reslicing cannot truthfully label oblique planes.
    normalized = canonical.affine[:3, :3] / spacing
    if not np.allclose(normalized, np.eye(3), atol=1e-3):
        raise ValueError("Oblique/sheared grids are not supported in this MVP. Resample to an axis-aligned RAS grid first.")
    data = canonical.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("Volume contains NaN or infinite values.")
    if mask:
        if not np.allclose(data, np.round(data)) or not np.isin(data, [0, 1, 2, 3, 4]).all():
            raise ValueError("Segmentation must contain integer labels 0, 1, 2, 3, 4. Map other label conventions explicitly.")
        data = data.astype(np.uint8)
    else:
        data = data.astype(np.float32)
    output = nib.Nifti1Image(data, canonical.affine)
    output.header.set_xyzt_units("mm")
    return output


def validate_matching_grid(reference: nib.Nifti1Image, other: nib.Nifti1Image) -> None:
    if reference.shape != other.shape:
        raise ValueError("All modalities and segmentation must have the same shape.")
    if not np.allclose(reference.affine, other.affine, atol=1e-3, rtol=1e-5):
        raise ValueError("All modalities and segmentation must share the same affine/spacing. Register and resample them first.")


def make_phantom() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic smooth brain-like mathematical phantom in a 1.7 mm grid."""
    shape = (96, 112, 96)
    x, y, z = np.meshgrid(*(np.linspace(-1, 1, n) for n in shape), indexing="ij")
    ell = (x / .82) ** 2 + (y / .91) ** 2 + (z / .86) ** 2
    brain = ell < 1
    angle = np.arctan2(y, x)
    folding = np.sin(28 * np.sqrt(ell) + 3 * np.sin(angle * 8) + 2 * np.sin(z * 17))
    cortical = (ell > .70) & brain
    tissue = np.where(brain, 115 + 14 * np.cos(9 * x) * np.sin(11 * y + 4 * z), 0)
    tissue += np.where(cortical, 26 + 17 * folding, 0)
    ventricle = (((np.abs(x) - .10) / .075) ** 2 + ((y + .03) / .26) ** 2 + ((z + .02) / .13) ** 2) < 1
    tissue[ventricle] = 28
    tissue[(np.abs(x) < .014) & brain & (z > -.36)] *= .62
    lesion = ((x - .27) / .25) ** 2 + ((y + .08) / .26) ** 2 + ((z - .10) / .24) ** 2
    core = ((x - .29) / .14) ** 2 + ((y + .07) / .15) ** 2 + ((z - .10) / .14) ** 2
    mask = np.zeros(shape, dtype=np.uint8)
    mask[(lesion < 1) & brain] = 2
    mask[(core < 1) & brain] = 3
    mask[(core < .39) & brain] = 1
    cavity = ((x - .33) / .065) ** 2 + ((y + .095) / .075) ** 2 + ((z - .09) / .06) ** 2
    mask[(cavity < 1) & brain] = 4
    noise = np.random.default_rng(2025).normal(0, 2.3, shape) * brain
    t1n = tissue.copy() + noise
    t1n[mask == 2] *= .91
    t1n[mask == 1] = 55
    t1n[mask == 4] = 20
    t1c = t1n.copy()
    t1c[mask == 3] += 105
    t2w = np.where(brain, 175 - tissue * .45, 0) + noise
    t2w[ventricle] = 205
    t2w[mask == 2] = 180 + noise[mask == 2]
    t2w[mask == 3] = 165
    t2w[mask == 1] = 135
    t2w[mask == 4] = 205
    t2f = t2w.copy()
    t2f[ventricle | (mask == 4)] = 18
    modalities = {key: val.astype(np.float32) for key, val in zip(("t1n", "t1c", "t2w", "t2f"), (t1n, t1c, t2w, t2f))}
    prediction = np.roll(mask, (1, -1, 0), axis=(0, 1, 2))
    prediction[(prediction == 2) & (lesion > .94)] = 0
    affine = np.diag([1.7, 1.7, 1.7, 1.0])
    affine[:3, 3] = -(np.asarray(shape) - 1) * 1.7 / 2
    return modalities, mask, prediction, affine


def _slice(data: np.ndarray, plane: str, index: int) -> tuple[np.ndarray, tuple[int, int]]:
    # Radiological display: patient right on screen left; superior at the top.
    if plane == "axial":
        return np.flip(data[:, :, index].T, axis=(0, 1)), (0, 1)
    if plane == "coronal":
        return np.flip(data[:, index, :].T, axis=(0, 1)), (0, 2)
    return np.flip(data[index, :, :].T, axis=(0, 1)), (1, 2)


def slice_png(volume: np.ndarray, mask: np.ndarray | None, spacing: tuple[float, ...], plane: str, index: int,
              opacity: float, labels: set[int], window: float, tumor_only: bool = False,
              definitions: list[dict] = LABELS) -> bytes:
    axis = {"axial": 2, "coronal": 1, "sagittal": 0}.get(plane)
    if axis is None:
        raise ValueError("Plane must be axial, coronal, or sagittal.")
    if index < 0 or index >= volume.shape[axis]:
        raise ValueError(f"Slice index must be 0–{volume.shape[axis] - 1}.")
    positive = volume[volume > 0]
    if positive.size:
        low, high = np.percentile(positive[::max(1, positive.size // 200_000)], (1, 99))
    else:
        low, high = float(volume.min()), float(volume.max())
    center = (float(low) + float(high)) / 2
    width = max(float(high - low), 1e-6) * window / 100
    raw, axes = _slice(volume, plane, index)
    gray = np.clip((raw - (center - width / 2)) / width, 0, 1)
    rgb = np.repeat((gray * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    if tumor_only:
        rgb[:] = 0
    if mask is not None:
        seg, _ = _slice(mask, plane, index)
        for label in definitions:
            if label["id"] not in labels:
                continue
            color = np.asarray(tuple(bytes.fromhex(label["color"].lstrip("#"))))
            selected = seg == label["id"]
            effective_opacity = 1.0 if tumor_only else opacity
            rgb[selected] = rgb[selected] * (1 - effective_opacity) + color * effective_opacity
    pil = Image.fromarray(rgb)
    physical_width, physical_height = pil.width * spacing[axes[0]], pil.height * spacing[axes[1]]
    factor = 512 / max(physical_width, physical_height)
    pil = pil.resize((max(1, round(physical_width * factor)), max(1, round(physical_height * factor))), Image.Resampling.NEAREST)
    output = io.BytesIO()
    pil.save(output, format="PNG")
    return output.getvalue()


def _mesh(binary: np.ndarray, affine: np.ndarray, step: int = 1) -> dict:
    if not binary.any():
        return {"vertices": [], "faces": []}
    crossings = sum(int(np.count_nonzero(np.diff(binary.astype(np.int8), axis=axis))) for axis in range(3))
    if crossings > 250_000:
        return {"vertices": [], "faces": [], "note": "Surface is too complex for the bounded MVP viewer; use slice view."}
    padded = np.pad(binary.astype(np.uint8), 1)
    vertices, faces, _, _ = marching_cubes(padded, level=.5, step_size=step, allow_degenerate=False)
    vertices -= 1
    vertices = nib.affines.apply_affine(affine, vertices)
    return {"vertices": np.round(vertices, 3).ravel().tolist(), "faces": faces.ravel().tolist()}


def make_mesh(volume: np.ndarray, mask: np.ndarray | None, affine: np.ndarray) -> dict:
    # Downsampling caps mesh size while returned vertices retain mm coordinates.
    stride = max(1, int(np.ceil(max(volume.shape) / 128)))
    small = volume[::stride, ::stride, ::stride]
    small_affine = affine.copy()
    small_affine[:3, :3] *= stride
    nonzero = small[small > 0]
    threshold = np.percentile(nonzero, 8) if nonzero.size else 0
    brain = ndimage.binary_fill_holes(small > threshold)
    components, count = ndimage.label(brain)
    if count:
        counts = np.bincount(components.ravel())
        counts[0] = 0
        brain = components == counts.argmax()
    brain = ndimage.binary_closing(brain, iterations=2)
    regions = []
    if mask is not None:
        for label in LABELS:
            regions.append({"label": label["id"], **_mesh(mask[::stride, ::stride, ::stride] == label["id"], small_affine)})
    center = nib.affines.apply_affine(affine, (np.array(volume.shape) - 1) / 2)
    return {"brain": _mesh(brain, small_affine, 2), "regions": regions, "center": center.tolist(),
            "units": "mm", "brain_surface": "intensity-derived contextual surface, not anatomical segmentation"}


def _surface_points(binary: np.ndarray, affine: np.ndarray) -> np.ndarray:
    surface = binary & ~ndimage.binary_erosion(binary)
    points = np.argwhere(surface)
    return nib.affines.apply_affine(affine, points)


def mask_metrics(pred: np.ndarray, reference: np.ndarray, affine: np.ndarray) -> dict:
    if not pred.any() and not reference.any():
        return {"dice": 1.0, "hd95_mm": None, "metric_note": "Both masks are empty; HD95 is undefined."}
    denominator = int(pred.sum()) + int(reference.sum())
    dice = 2 * int(np.count_nonzero(pred & reference)) / denominator
    if not pred.any() or not reference.any():
        return {"dice": dice, "hd95_mm": None, "metric_note": "One mask is empty; HD95 is undefined."}
    if np.count_nonzero(pred & ~ndimage.binary_erosion(pred)) + np.count_nonzero(reference & ~ndimage.binary_erosion(reference)) > 1_000_000:
        return {"dice": dice, "hd95_mm": None, "metric_note": "HD95 omitted: surfaces exceed the one-million-point memory limit."}
    a, b = _surface_points(pred, affine), _surface_points(reference, affine)
    # KD trees avoid allocating a volume-sized distance transform twice.
    distances = np.concatenate((cKDTree(a).query(b)[0], cKDTree(b).query(a)[0]))
    return {"dice": dice, "hd95_mm": float(np.percentile(distances, 95))}


def stats(mask: np.ndarray, affine: np.ndarray, reference: np.ndarray | None = None,
          definitions: list[dict] = LABELS) -> dict:
    voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000
    regions = []
    for label in definitions:
        binary = mask == label["id"]
        entry = {"label": label["id"], "name": label["name"], "color": label["color"],
                 "voxels": int(binary.sum()), "volume_ml": float(binary.sum() * voxel_ml),
                 "components": int(ndimage.label(binary)[1])}
        if reference is not None:
            entry.update(mask_metrics(binary, reference == label["id"], affine))
        regions.append(entry)
    output = {"regions": regions, "total_volume_ml": float(np.count_nonzero(mask) * voxel_ml),
              "voxel_volume_mm3": voxel_ml * 1000, "dice": None, "hd95_mm": None,
              "metric_definition": "Whole foreground (labels 1–4); per-label scores listed separately. HD95 uses surface voxel centres in mm."}
    foreground = mask > 0
    if foreground.any():
        output["tumor_center_voxel"] = np.round(ndimage.center_of_mass(foreground)).astype(int).tolist()
        occupied_axes = [np.flatnonzero(foreground.any(axis=tuple(a for a in range(3) if a != axis))) for axis in range(3)]
        output["tumor_bbox_voxel"] = [[int(a[0]) for a in occupied_axes], [int(a[-1]) for a in occupied_axes]]
    else:
        output["tumor_center_voxel"] = None
        output["tumor_bbox_voxel"] = None
    if reference is not None:
        output.update(mask_metrics(mask > 0, reference > 0, affine))
    return output
