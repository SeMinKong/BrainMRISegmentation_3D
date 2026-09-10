"""Spatial preprocessing shared by training and inference.

MRI interpolation is linear; segmentation interpolation is nearest-neighbour.
No registration is inferred: input modalities and labels must share a grid.
"""
from __future__ import annotations

from pathlib import Path

from .schema import MODALITIES, canonical_modalities


def load_volume(path: Path):
    import nibabel as nib
    import numpy as np

    image = nib.load(str(path))
    if image.header.get_xyzt_units()[0] not in {"mm", "unknown"}:
        raise ValueError(f"MRI spatial units must be millimetres (unspecified units are interpreted as mm): {path.name}")
    if len(image.shape) != 3 or any(size < 2 for size in image.shape):
        raise ValueError(f"Expected a 3D MRI volume: {path.name}")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError(f"Invalid spatial affine: {path.name}")
    if not np.isfinite(image.get_fdata(dtype=np.float32)).all():
        raise ValueError(f"Nonfinite voxel intensity: {path.name}")
    return image


def prepare_case(modalities: dict, spacing: tuple[float, float, float], label_path=None, labels=None):
    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to, resample_to_output

    if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("spacing must be three positive finite millimetre values")
    modalities = canonical_modalities(modalities)
    images = {name: load_volume(Path(modalities[name])) for name in MODALITIES}
    original = images["t1"]
    for name, img in images.items():
        if img.shape != original.shape or not np.allclose(img.affine, original.affine, atol=1e-3):
            raise ValueError(f"{name} is not co-registered to T1; register/resample modalities before import")
    canonical = nib.as_closest_canonical(original)
    target = resample_to_output(canonical, voxel_sizes=spacing, order=1)
    target_grid = (target.shape, target.affine)
    channels = []
    for name in MODALITIES:
        volume = resample_from_to(images[name], target_grid, order=1).get_fdata(dtype=np.float32)
        mask = volume != 0
        if not mask.any():
            raise ValueError(f"{name} contains no nonzero brain voxels")
        values = volume[mask]
        std = float(values.std())
        if std < 1e-8:
            raise ValueError(f"{name} has constant nonzero intensity")
        volume[mask] = (values - float(values.mean())) / std
        volume[~mask] = 0
        channels.append(volume)
    segmentation = None
    if label_path is not None:
        label_img = load_volume(Path(label_path))
        if label_img.shape != original.shape or not np.allclose(label_img.affine, original.affine, atol=1e-3):
            raise ValueError("Label grid does not match the MRI grid")
        raw = label_img.get_fdata(dtype=np.float32)
        if not np.array_equal(raw, np.rint(raw)):
            raise ValueError("Segmentation must contain integer label IDs")
        ids = sorted(int(key) for key in labels)
        unexpected = set(np.unique(raw).astype(int)) - set(ids)
        if unexpected:
            raise ValueError(f"Unexpected segmentation labels: {sorted(unexpected)}")
        resampled = resample_from_to(label_img, target_grid, order=0).get_fdata(dtype=np.float32)
        segmentation = np.zeros(resampled.shape, dtype=np.int64)
        for channel_id, label_id in enumerate(ids):
            segmentation[resampled == label_id] = channel_id
    return np.stack(channels).astype(np.float32), segmentation, target.affine, original


def save_prediction(class_indices, affine, original, output_path: Path, labels: dict):
    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to

    lookup = np.array(sorted(int(key) for key in labels), dtype=np.uint8)
    indices = np.asarray(class_indices)
    if indices.min() < 0 or indices.max() >= len(lookup):
        raise ValueError("Model returned an invalid class index")
    segmentation = lookup[indices.astype(np.int64)]
    predicted = nib.Nifti1Image(segmentation, affine)
    restored = resample_from_to(predicted, (original.shape, original.affine), order=0)
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    result = nib.Nifti1Image(np.rint(restored.get_fdata()).astype(np.uint8), original.affine, header)
    result.set_qform(original.affine, code=1)
    result.set_sform(original.affine, code=1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(output_path))


def sample_patch(image, label, patch_size, rng):
    """Foreground-centred sampling on half of patches; random spatial flips."""
    import numpy as np

    padding = [(0, max(0, p - d)) for d, p in zip(label.shape, patch_size)]
    image = np.pad(image, [(0, 0), *padding])
    label = np.pad(label, padding)
    foreground = np.argwhere(label > 0)
    if len(foreground) and rng.random() < 0.5:
        center = foreground[rng.integers(len(foreground))]
    else:
        center = [rng.integers(size) for size in label.shape]
    starts = [int(np.clip(c - p // 2, 0, d - p)) for c, p, d in zip(center, patch_size, label.shape)]
    slices = tuple(slice(start, start + size) for start, size in zip(starts, patch_size))
    image, label = image[(slice(None), *slices)], label[slices]
    for axis in range(3):
        if rng.random() < 0.5:
            image, label = np.flip(image, axis + 1), np.flip(label, axis)
    return np.ascontiguousarray(image), np.ascontiguousarray(label[None])
