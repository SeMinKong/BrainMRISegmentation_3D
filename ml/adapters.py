"""Web model registry and spatially reversible inference.

Only administrator-configured environment paths are used to load checkpoints.
No weights are bundled; an unconfigured model cannot produce predictions.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Callable, Mapping

from .schema import MODALITIES, canonical_modalities, require_mu_glioma_post_labels, validate_labels

MODEL_INFO = {
    "unet3d": ("3D U-Net", "MRI_UNET_CHECKPOINT", "MONAI 3D CNN baseline; four MRI channels, patch training and sliding-window inference."),
    "swinunetr": ("Swin UNETR", "MRI_SWIN_CHECKPOINT", "MONAI shifted-window Transformer encoder with a 3D segmentation decoder."),
    "nnunet": ("nnU-Net v2", "MRI_NNUNET_MODEL_DIR", "Adapter for a trained nnU-Net v2 model; uses its saved plans and preprocessing."),
}


def _has_package(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def describe_models() -> list[dict]:
    models = []
    for model_id, (name, variable, description) in MODEL_INFO.items():
        path = Path(os.environ[variable]) if os.environ.get(variable) else None
        requirements = ["torch", "nibabel", "scipy", "nnunetv2" if model_id == "nnunet" else "monai"]
        missing = [package for package in requirements if not _has_package(package)]
        if path is None:
            reason = f"Set {variable} to locally trained weights. No checkpoint is bundled."
        elif model_id == "nnunet" and (not path.is_dir() or not (path / "plans.json").is_file() or not (path / "dataset.json").is_file() or not list(path.glob("fold_*/checkpoint_final.pth"))):
            reason = "nnU-Net folder needs plans.json, dataset.json and fold_*/checkpoint_final.pth."
        elif model_id != "nnunet" and not path.is_file():
            reason = f"Configured checkpoint does not exist: {variable}"
        elif missing:
            reason = f"Optional packages missing: {', '.join(missing)}"
        else:
            reason = "Configured locally; checkpoint compatibility is validated when inference starts."
        available = path is not None and not missing and (
            model_id != "nnunet" and path.is_file() or
            model_id == "nnunet" and path.is_dir() and (path / "plans.json").is_file()
            and (path / "dataset.json").is_file() and bool(list(path.glob("fold_*/checkpoint_final.pth")))
        )
        models.append({"id": model_id, "name": name, "available": bool(available),
                       "reason": reason, "description": description})
    return models


def _device(torch):
    requested = os.environ.get("MRI_DEVICE", "auto").lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("MRI_DEVICE must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MRI_DEVICE=cuda but CUDA is unavailable")
    return torch.device(requested)


def validate_metadata(metadata: dict, model_id: str) -> dict:
    import numpy as np
    from .models import validate_patch_size

    if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
        raise ValueError("Checkpoint lacks supported format_version=1 metadata; use python -m ml.train")
    if metadata.get("model_id") != model_id:
        raise ValueError("Checkpoint architecture does not match the selected model")
    if metadata.get("modalities") != list(MODALITIES):
        raise ValueError("Checkpoint modality order differs from t1,t1ce,t2,flair")
    if metadata.get("orientation") != "RAS" or metadata.get("normalization") != "nonzero_zscore":
        raise ValueError("Unsupported checkpoint preprocessing contract")
    labels = validate_labels(metadata.get("labels", {}))
    spacing = metadata.get("spacing_mm", [])
    if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Checkpoint has invalid target spacing")
    validate_patch_size(model_id, metadata.get("patch_size", []), metadata.get("model_settings"))
    if int(metadata.get("global_step", 0)) < 1:
        raise ValueError("Checkpoint contains no completed optimizer steps")
    return labels


def run_inference(model_id: str, modalities: Mapping[str, Path], output_path: Path,
                  progress: Callable[[float, str], None] | None = None) -> dict:
    report = progress or (lambda percent, message: None)
    status = next((model for model in describe_models() if model["id"] == model_id), None)
    if status is None:
        raise ValueError(f"Unknown model: {model_id}")
    if not status["available"]:
        raise RuntimeError(status["reason"])
    paths = canonical_modalities(dict(modalities))
    if model_id == "nnunet":
        return _run_nnunet(paths, Path(output_path), report)
    import torch
    from monai.inferers import sliding_window_inference
    from .data import prepare_case, save_prediction
    from .models import build_model

    checkpoint_path = Path(os.environ[MODEL_INFO[model_id][1]])
    report(0.05, "Loading registered checkpoint")
    # Tensor-only deserialization; never enable arbitrary-pickle fallback here.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata", {})
    labels = validate_metadata(metadata, model_id)
    require_mu_glioma_post_labels(labels)
    model = build_model(model_id, len(labels), metadata.get("model_settings"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = _device(torch)
    model.to(device).eval()
    report(0.18, "Canonical RAS orientation, spacing and intensity normalization")
    image, _, affine, original = prepare_case(paths, tuple(metadata["spacing_mm"]))
    tensor = torch.from_numpy(image[None])
    report(0.35, "Sliding-window volume inference")
    with torch.inference_mode():
        # Stitch on CPU to avoid keeping an entire 3D logits volume on the GPU.
        logits = sliding_window_inference(tensor, tuple(metadata["patch_size"]), 1, model,
                                          overlap=0.5, mode="gaussian", sw_device=device,
                                          device=torch.device("cpu"))
        prediction = logits.argmax(1)[0].numpy()
    report(0.9, "Restoring original MRI grid and writing integer segmentation")
    save_prediction(prediction, affine, original, Path(output_path), labels)
    report(1.0, "Inference complete")
    return {"model_id": model_id, "device": str(device), "labels": labels,
            "spacing_mm": metadata["spacing_mm"], "checkpoint": checkpoint_path.name,
            "training_step": metadata["global_step"], "output_grid": "input_t1",
            "validation_mean_dice": metadata.get("validation_mean_dice")}


def _run_nnunet(modalities: dict, output_path: Path, report) -> dict:
    import tempfile
    import nibabel as nib
    import numpy as np
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from .data import load_volume, save_prediction

    folder = Path(os.environ["MRI_NNUNET_MODEL_DIR"])
    dataset = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    channels = dataset.get("channel_names", {})
    try:
        names = [channels[str(index)] for index in range(4)]
        from .schema import ALIASES
        order = [ALIASES.get(name.lower(), name.lower()) for name in names]
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError("nnU-Net dataset.json requires channel_names 0..3 naming t1/t1ce/t2/flair (or t1n/t1c/t2w/t2f aliases)") from exc
    if len(channels) != 4 or set(order) != set(MODALITIES):
        raise ValueError("nnU-Net channels must explicitly identify the four MRI modalities")
    raw_labels = dataset.get("labels", {})
    if not raw_labels or any(not isinstance(value, int) for value in raw_labels.values()):
        raise ValueError("MVP nnU-Net adapter supports atomic integer labels, not region-based labels")
    labels = validate_labels({str(value): key for key, value in raw_labels.items()})
    require_mu_glioma_post_labels(labels)
    reference = load_volume(Path(modalities["t1"]))
    for path in modalities.values():
        volume = load_volume(Path(path))
        if volume.shape != reference.shape or not np.allclose(volume.affine, reference.affine, atol=1e-3):
            raise ValueError("nnU-Net input modalities must be co-registered")
    device = _device(torch)
    report(0.1, "Loading nnU-Net plans and trained folds")
    predictor = nnUNetPredictor(device=device, perform_everything_on_device=device.type == "cuda", allow_tqdm=False)
    folds_value = os.environ.get("MRI_NNUNET_FOLDS", "0")
    folds = tuple("all" if value.strip() == "all" else int(value) for value in folds_value.split(","))
    predictor.initialize_from_trained_model_folder(str(folder), use_folds=folds, checkpoint_name="checkpoint_final.pth")
    if predictor.configuration_manager.previous_stage_name is not None:
        raise ValueError("Cascaded nnU-Net models require a previous stage and are not supported in this MVP")
    # Use the trained plan's reader/writer so axis ordering and spacing follow nnU-Net's contract.
    reader = predictor.plans_manager.image_reader_writer_class()
    report(0.25, "Reading modalities with nnU-Net's trained image reader")
    image, properties = reader.read_images([str(modalities[name]) for name in order])
    report(0.4, "nnU-Net preprocessing and sliding-window inference")
    prediction = predictor.predict_single_npy_array(image, properties)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nnunet-", dir=output_path.parent) as temporary:
        predicted_path = Path(temporary) / "prediction.nii.gz"
        reader.write_seg(prediction, str(predicted_path), properties)
        predicted = nib.load(str(predicted_path))
        values = predicted.get_fdata()
        if not np.isfinite(values).all() or not np.array_equal(values, np.rint(values)):
            raise ValueError("nnU-Net returned a non-integer segmentation")
        ids = sorted(map(int, labels))
        if not set(np.unique(values).astype(int)).issubset(ids):
            raise ValueError("nnU-Net prediction contains unknown label IDs")
        indices = np.zeros(values.shape, dtype=np.uint8)
        for index, label in enumerate(ids):
            indices[values == label] = index
        save_prediction(indices, predicted.affine, reference, output_path, labels)
    report(1.0, "nnU-Net inference complete")
    return {"model_id": "nnunet", "device": str(device), "labels": labels,
            "checkpoint": folder.name, "folds": list(folds), "output_grid": "input_t1"}
