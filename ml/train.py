"""Train a real four-channel 3D segmentation model on a patient-split manifest.

Run from the project root: python -m ml.train --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .schema import MODALITIES, load_manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--output", type=Path, default=Path("runs/unet3d"))
    result.add_argument("--model", choices=["unet3d", "swinunetr"], default="unet3d")
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--max-steps", type=int, default=0, help="Stop after N optimizer steps; 0 means no limit. Smoke checks only.")
    result.add_argument("--patches-per-case", type=int, default=4)
    result.add_argument("--patch-size", type=int, nargs=3, default=[64, 64, 64])
    result.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    result.add_argument("--channels", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    result.add_argument("--feature-size", type=int, default=24)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    result.add_argument("--threads", type=int, default=4, help="CPU intra-operation threads")
    return result


def dice_scores(prediction, target, labels: dict) -> dict:
    import numpy as np

    metrics = {}
    for class_id, label_id in enumerate(sorted(map(int, labels))):
        if label_id == 0:
            continue
        predicted, expected = prediction == class_id, target == class_id
        count = int(predicted.sum() + expected.sum())
        metrics[str(label_id)] = None if count == 0 else float(2 * np.logical_and(predicted, expected).sum() / count)
    return metrics


def main(argv=None):
    args = parser().parse_args(argv)
    from .models import build_model, validate_patch_size

    settings = {"channels": args.channels, "num_res_units": 2, "feature_size": args.feature_size}
    validate_patch_size(args.model, args.patch_size, settings)
    if min(args.epochs, args.patches_per_case, args.threads) <= 0 or args.max_steps < 0 or args.lr <= 0:
        raise ValueError("epochs, patches-per-case, threads and lr must be positive; max-steps must be nonnegative")
    manifest = load_manifest(args.manifest)
    train_cases = [case for case in manifest["cases"] if case["split"] == "train"]
    val_cases = [case for case in manifest["cases"] if case["split"] == "val"]
    import numpy as np
    import torch
    import monai
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceCELoss
    from monai.utils import set_determinism
    from .data import prepare_case, sample_patch

    set_determinism(seed=args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    model = build_model(args.model, len(manifest["labels"]), settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(manifest, sort_keys=True)
    (args.output / "resolved-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metadata = {"format_version": 1, "model_id": args.model, "modalities": list(MODALITIES),
                "labels": manifest["labels"], "orientation": "RAS", "normalization": "nonzero_zscore",
                "spacing_mm": args.spacing, "patch_size": args.patch_size, "model_settings": settings,
                "seed": args.seed, "learning_rate": args.lr, "patches_per_case": args.patches_per_case,
                "manifest_sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
                "train_patients": sorted({case["patient_id"] for case in train_cases}),
                "val_patients": sorted({case["patient_id"] for case in val_cases}),
                "torch_version": str(torch.__version__), "monai_version": str(monai.__version__),
                "validation_grid": "canonical_resampled", "loss": "DiceCE", "batch_size": 1}
    global_step, best, history, started = 0, -1.0, [], time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for index in rng.permutation(len(train_cases)):
            case = train_cases[int(index)]
            image, label, _, _ = prepare_case(case["modalities"], tuple(args.spacing), case["label"], manifest["labels"])
            for _ in range(args.patches_per_case):
                patch, target = sample_patch(image, label, args.patch_size, rng)
                inputs = torch.from_numpy(patch[None]).to(device)
                targets = torch.from_numpy(target[None]).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(inputs), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Training loss became nonfinite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
                optimizer.step()
                global_step += 1
                losses.append(float(loss.detach().cpu()))
                print(json.dumps({"event": "train_step", "epoch": epoch, "step": global_step, "loss": losses[-1]}), flush=True)
                if args.max_steps and global_step >= args.max_steps:
                    break
            if args.max_steps and global_step >= args.max_steps:
                break
        model.eval()
        case_metrics = []
        with torch.inference_mode():
            for case in val_cases:
                image, target, _, _ = prepare_case(case["modalities"], tuple(args.spacing), case["label"], manifest["labels"])
                logits = sliding_window_inference(torch.from_numpy(image[None]), tuple(args.patch_size), 1,
                                                  model, overlap=0.5, mode="gaussian", sw_device=device,
                                                  device=torch.device("cpu"))
                scores = dice_scores(logits.argmax(1)[0].numpy(), target, manifest["labels"])
                case_metrics.append({"case_id": case["case_id"], "dice": scores})
        valid = [score for case in case_metrics for score in case["dice"].values() if score is not None]
        mean_dice = float(np.mean(valid)) if valid else None
        event = {"event": "epoch_complete", "epoch": epoch, "step": global_step,
                 "loss": float(np.mean(losses)), "validation_mean_dice": mean_dice,
                 "validation": case_metrics, "elapsed_seconds": round(time.monotonic() - started, 2)}
        history.append(event)
        metadata.update({"epoch": epoch, "global_step": global_step, "validation_mean_dice": mean_dice})
        checkpoint = {"metadata": dict(metadata), "state_dict": model.state_dict()}
        temporary = args.output / "checkpoint.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(args.output / "last.pt")
        if epoch == 1 or (mean_dice if mean_dice is not None else -1.0) > best:
            best = mean_dice if mean_dice is not None else best
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "best.pt")
        (args.output / "metrics.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(event), flush=True)
        if args.max_steps and global_step >= args.max_steps:
            break
    print(f"Saved {args.output / 'best.pt'}; optimizer steps={global_step}. Smoke-run weights are not clinically meaningful.", flush=True)


if __name__ == "__main__":
    main()
