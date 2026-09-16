"""Train a real four-channel 3D segmentation model on a patient-split manifest.

Run from the project root: python -m ml.train --help
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import threading
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
    result.add_argument("--batch-size", type=int, default=1,
                        help="Patches per optimizer step. Patches are pooled across consecutive cases and drawn at random.")
    result.add_argument("--cosine", action="store_true", help="Cosine-anneal the learning rate per epoch to 1%% of --lr")
    result.add_argument("--sw-batch-size", type=int, default=4, help="Windows per forward pass in validation sliding-window inference")
    result.add_argument("--patch-size", type=int, nargs=3, default=[64, 64, 64])
    result.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    result.add_argument("--channels", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    result.add_argument("--feature-size", type=int, default=24)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    result.add_argument("--threads", type=int, default=4, help="CPU intra-operation threads")
    result.add_argument("--val-every", type=int, default=1,
                        help="Run full-volume validation every N epochs (always on the final epoch). last.pt is still saved every epoch.")
    result.add_argument("--amp", action="store_true",
                        help="bfloat16 autocast for the training forward/backward pass on CUDA. Validation and inference stay float32.")
    result.add_argument("--cache-dir", type=Path, default=None,
                        help="Store preprocessed volumes (float16 .npz) here and train from them. Built/refreshed before epoch 1.")
    result.add_argument("--cache-workers", type=int, default=0, help="Processes used to build the cache; 0 = CPU count - 2 (max 6)")
    result.add_argument("--prefetch", type=int, default=3, help="Cases loaded (and patch-sampled) ahead of the GPU")
    result.add_argument("--loader-threads", type=int, default=2, help="Threads that load cases and sample patches in the background")
    result.add_argument("--no-validation", action="store_true", help="Skip every validation pass (throughput benchmarks only; best.pt is never written)")
    return result


def batch_stream(order, loader, prefetch, threads, drain, pin_memory, stop):
    """Background producer: load cases, pool patches, emit stacked CPU tensors ready for a non-blocking device copy.

    Keeps np.stack/pin_memory (~15-40 ms per 160^3 batch) off the thread that launches CUDA kernels, so the GPU
    sees back-to-back steps instead of idling while the next batch is assembled.
    """
    import queue
    import threading

    import numpy as np
    import torch

    ready: queue.Queue = queue.Queue(maxsize=3)
    sentinel = object()

    def tensors(batch):
        inputs = torch.from_numpy(np.stack([patch for patch, _ in batch]))
        targets = torch.from_numpy(np.stack([target for _, target in batch]))
        return (inputs.pin_memory(), targets.pin_memory()) if pin_memory else (inputs, targets)

    def offer(item):
        while not stop.is_set():
            try:
                ready.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def run():
        try:
            pool = []
            for _, patches in iterate_cases(order, loader, prefetch, threads):
                pool.extend(patches)
                for batch in drain(pool, False):
                    if not offer(tensors(batch)):
                        return
                if stop.is_set():
                    return
            for batch in drain(pool, True):
                if not offer(tensors(batch)):
                    return
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer side
            offer((sentinel, exc))
        finally:
            try:
                ready.put_nowait((sentinel, None))
            except queue.Full:
                pass  # consumer is stopping or will notice the dead thread

    thread = threading.Thread(target=run, name="batch-producer", daemon=True)
    thread.start()
    try:
        while True:
            try:
                item = ready.get(timeout=1.0)
            except queue.Empty:
                if not thread.is_alive():
                    return
                continue
            if item[0] is sentinel:
                if item[1] is not None:
                    raise item[1]
                return
            yield item
    finally:
        stop.set()
        thread.join(timeout=5)


def iterate_cases(items, loader, prefetch: int, threads: int = 1):
    """Yield (item, loader(item)) in input order while up to `prefetch` items load ahead on `threads` threads.

    Loader exceptions surface at the item's position; the GPU thread never blocks on disk when the loader
    keeps up. Order is preserved so runs stay reproducible for a given seed.
    """
    if prefetch <= 0 or len(items) <= 1:
        for item in items:
            yield item, loader(item)
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    import itertools

    with ThreadPoolExecutor(max_workers=max(1, min(threads, prefetch)), thread_name_prefix="case-loader") as pool:
        queue_ = deque()
        remaining = iter(items)
        for item in itertools.islice(remaining, prefetch):
            queue_.append((item, pool.submit(loader, item)))
        while queue_:
            item, future = queue_.popleft()
            result = future.result()
            following = next(remaining, None)
            if following is not None:
                queue_.append((following, pool.submit(loader, following)))
            yield item, result


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
    if min(args.epochs, args.patches_per_case, args.threads, args.val_every, args.batch_size, args.sw_batch_size, args.loader_threads) <= 0 \
            or min(args.max_steps, args.cache_workers, args.prefetch) < 0 or args.lr <= 0:
        raise ValueError("epochs, patches-per-case, threads, val-every, batch-size, sw-batch-size, loader-threads and lr must be positive; "
                         "max-steps, cache-workers and prefetch must be nonnegative")
    manifest = load_manifest(args.manifest)
    train_cases = [case for case in manifest["cases"] if case["split"] == "train"]
    val_cases = [case for case in manifest["cases"] if case["split"] == "val"]
    import numpy as np
    import torch
    import monai
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceCELoss
    from monai.utils import set_determinism
    from .data import prepare_case, sample_patches

    set_determinism(seed=args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    spacing = tuple(args.spacing)
    cache_summary = None
    if args.cache_dir is not None:
        from .cache import ensure_cache, load_cached
        cache_summary = ensure_cache(args.cache_dir, train_cases + val_cases, spacing, manifest["labels"],
                                     workers=args.cache_workers or None, report=lambda event: print(json.dumps(event), flush=True))

    def load_case(case):
        if args.cache_dir is not None:
            cached = load_cached(args.cache_dir, case, spacing, manifest["labels"])
            if cached is not None:
                return cached[0], cached[1]
            print(json.dumps({"event": "cache_miss", "case_id": case["case_id"], "note": "source changed during training; preprocessing directly"}), flush=True)
        image, label, _, _ = prepare_case(case["modalities"], spacing, case["label"], manifest["labels"])
        return image, label

    def load_train_patches(item):
        # Patch sampling runs on the loader thread with a per-case seed drawn on the main thread,
        # so it is deterministic per seed and thread-safe with several loader threads.
        case, seed = item
        image, label = load_case(case)
        return sample_patches(image, label, args.patch_size, np.random.default_rng(seed), args.patches_per_case)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed patch size: let cuDNN pick the fastest kernels
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = build_model(args.model, len(manifest["labels"]), settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01) if args.cosine else None
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
                "validation_grid": "canonical_resampled", "loss": "DiceCE", "batch_size": args.batch_size,
                "lr_schedule": "cosine_to_1pct" if scheduler else "constant", "weight_decay": 1e-5, "grad_clip_norm": 12.0,
                "device": device.type, "amp": "bfloat16_autocast" if use_amp else None, "val_every": args.val_every,
                "train_cases": len(train_cases), "val_cases": len(val_cases), "patches_per_case": args.patches_per_case,
                "preprocessing_cache": None if cache_summary is None else {"dir": cache_summary["cache_dir"], "image_dtype": "float16"}}
    steps_per_epoch = -(-len(train_cases) * args.patches_per_case // args.batch_size)
    print(json.dumps({"event": "train_start", "device": device.type, "amp": use_amp, "train_cases": len(train_cases),
                      "val_cases": len(val_cases), "batch_size": args.batch_size, "steps_per_epoch": steps_per_epoch,
                      "lr_schedule": metadata["lr_schedule"]}), flush=True)
    # Patches from consecutive cases are pooled and drawn at random so a batch mixes cases.
    pool_threshold = max(2 * args.batch_size, args.patches_per_case + args.batch_size)

    def train_batch(batch, epoch):
        nonlocal global_step
        inputs = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(inputs)
        loss = loss_function(logits.float(), targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
        optimizer.step()
        global_step += 1
        value = float(loss.detach().cpu())
        print(json.dumps({"event": "train_step", "epoch": epoch, "step": global_step, "loss": value,
                          "lr": optimizer.param_groups[0]["lr"], "batch": int(inputs.shape[0])}), flush=True)
        return value

    def drain(pool, final):
        while pool and (final or len(pool) >= pool_threshold):
            picks = sorted(rng.choice(len(pool), size=min(args.batch_size, len(pool)), replace=False), reverse=True)
            yield [pool.pop(int(index)) for index in picks]

    global_step, best, history, started = 0, -1.0, [], time.monotonic()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        losses = []
        order = [(train_cases[int(index)], int(rng.integers(2**31 - 1))) for index in rng.permutation(len(train_cases))]
        stop = threading.Event()
        for batch in batch_stream(order, load_train_patches, args.prefetch, args.loader_threads, drain,
                                  pin_memory=device.type == "cuda", stop=stop):
            losses.append(train_batch(batch, epoch))
            if args.max_steps and global_step >= args.max_steps:
                stop.set()
                break
        if scheduler is not None:
            scheduler.step()
        finished = bool(args.max_steps and global_step >= args.max_steps)
        validate = (epoch % args.val_every == 0 or epoch == args.epochs or finished) and not args.no_validation
        case_metrics, mean_dice = [], None
        train_seconds = round(time.monotonic() - epoch_started, 1)
        validation_started = time.monotonic()
        if validate:
            model.eval()
            with torch.inference_mode():
                for case, (image, target) in iterate_cases(val_cases, load_case, args.prefetch, args.loader_threads):
                    logits = sliding_window_inference(torch.from_numpy(image[None]), tuple(args.patch_size), args.sw_batch_size,
                                                      model, overlap=0.5, mode="gaussian", sw_device=device,
                                                      device=torch.device("cpu"))
                    scores = dice_scores(logits.argmax(1)[0].numpy(), target, manifest["labels"])
                    case_metrics.append({"case_id": case["case_id"], "dice": scores})
            valid = [score for case in case_metrics for score in case["dice"].values() if score is not None]
            mean_dice = float(np.mean(valid)) if valid else None
        event = {"event": "epoch_complete", "epoch": epoch, "step": global_step,
                 "loss": float(np.mean(losses)), "validated": validate, "validation_mean_dice": mean_dice,
                 "validation": case_metrics, "train_seconds": train_seconds,
                 "validation_seconds": round(time.monotonic() - validation_started, 1) if validate else 0.0,
                 "elapsed_seconds": round(time.monotonic() - started, 2)}
        history.append(event)
        metadata.update({"epoch": epoch, "global_step": global_step})
        if validate:
            metadata["validation_mean_dice"] = mean_dice
            metadata["validated_epoch"] = epoch
        checkpoint = {"metadata": dict(metadata), "state_dict": model.state_dict()}
        temporary = args.output / "checkpoint.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(args.output / "last.pt")
        # best.pt only ever holds validated weights; the first validated epoch seeds it.
        if validate and (not (args.output / "best.pt").exists() or (mean_dice if mean_dice is not None else -1.0) > best):
            best = mean_dice if mean_dice is not None else best
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "best.pt")
        (args.output / "metrics.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({**event, "validation": f"{len(case_metrics)} cases (see metrics.json)"}), flush=True)
        if finished:
            break
    print(f"Saved {args.output / 'best.pt'}; optimizer steps={global_step}. Smoke-run weights are not clinically meaningful.", flush=True)


if __name__ == "__main__":
    main()
