import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from ml.cache import build_one, cache_path, ensure_cache, is_fresh, load_cached
from ml.data import prepare_case
from ml.schema import DEFAULT_LABELS, load_manifest
from ml.smoke import create_fixture
from ml.train import iterate_cases

SPACING = (1.0, 1.0, 1.0)


def test_cache_round_trips_prepare_case_output(tmp_path):
    manifest = load_manifest(create_fixture(tmp_path / "src"))
    case = manifest["cases"][0]
    cache_dir = tmp_path / "cache"
    summary = ensure_cache(cache_dir, manifest["cases"], SPACING, manifest["labels"], workers=1)
    assert summary["built"] == 2 and summary["fresh"] == 0
    assert sorted(path.name for path in cache_dir.iterdir()) == ["synthetic-000.npz", "synthetic-001.npz"]
    image, label, affine, _ = prepare_case(case["modalities"], SPACING, case["label"], manifest["labels"])
    cached_image, cached_label, cached_affine = load_cached(cache_dir, case, SPACING, manifest["labels"])
    assert cached_image.dtype == np.float32 and cached_label.dtype == np.int64
    np.testing.assert_allclose(cached_image, image, rtol=2e-3, atol=2e-3)  # float16 storage
    np.testing.assert_array_equal(cached_label, label)
    np.testing.assert_allclose(cached_affine, affine)
    # A second pass is a no-op.
    again = ensure_cache(cache_dir, manifest["cases"], SPACING, manifest["labels"], workers=1)
    assert again["built"] == 0 and again["fresh"] == 2


def test_changed_source_spacing_or_labels_invalidate_entries(tmp_path):
    manifest = load_manifest(create_fixture(tmp_path / "src"))
    cache_dir = tmp_path / "cache"
    ensure_cache(cache_dir, manifest["cases"], SPACING, manifest["labels"], workers=1)
    changed, untouched = manifest["cases"]
    assert is_fresh(cache_dir, changed, SPACING, manifest["labels"])
    # Different spacing or label semantics never reuse an entry.
    assert not is_fresh(cache_dir, changed, (2.0, 2.0, 2.0), manifest["labels"])
    assert not is_fresh(cache_dir, changed, SPACING, dict(manifest["labels"], **{"3": "RC", "4": "ET"}))
    # Rewriting a modality (new mtime/size) makes only that case stale.
    path = Path(changed["modalities"]["t1"])
    volume = nib.load(path)
    nib.save(nib.Nifti1Image(volume.get_fdata() * 2, volume.affine), path)
    os.utime(path, ns=(os.stat(path).st_atime_ns, os.stat(path).st_mtime_ns + 10_000_000))
    assert not is_fresh(cache_dir, changed, SPACING, manifest["labels"])
    assert load_cached(cache_dir, changed, SPACING, manifest["labels"]) is None
    assert is_fresh(cache_dir, untouched, SPACING, manifest["labels"])
    summary = ensure_cache(cache_dir, manifest["cases"], SPACING, manifest["labels"], workers=1)
    assert summary["built"] == 1 and summary["fresh"] == 1
    # Source files are read only: the rewritten file is exactly what we wrote.
    np.testing.assert_allclose(nib.load(path).get_fdata(), volume.get_fdata() * 2)


def test_corrupt_entry_is_treated_as_stale(tmp_path):
    manifest = load_manifest(create_fixture(tmp_path / "src"))
    case = manifest["cases"][0]
    cache_dir = tmp_path / "cache"
    build_one(cache_dir, case, SPACING, manifest["labels"])
    cache_path(cache_dir, case).write_bytes(b"not a zip")
    assert not is_fresh(cache_dir, case, SPACING, manifest["labels"])
    assert load_cached(cache_dir, case, SPACING, manifest["labels"]) is None


def test_failed_case_stops_the_build_with_its_id(tmp_path):
    manifest = load_manifest(create_fixture(tmp_path / "src"))
    broken = dict(manifest["cases"][0], modalities=dict(manifest["cases"][0]["modalities"]))
    Path(broken["modalities"]["flair"]).write_bytes(b"")
    with pytest.raises(RuntimeError, match="synthetic-000"):
        ensure_cache(tmp_path / "cache", [broken, manifest["cases"][1]], SPACING, manifest["labels"], workers=1)


@pytest.mark.parametrize("threads", [1, 3])
def test_prefetch_iterator_preserves_order_and_propagates_errors(threads):
    import time

    cases = [{"case_id": f"c{i}"} for i in range(7)]

    def slow_upper(case):
        time.sleep(0.02 if case["case_id"] in ("c0", "c3") else 0.001)  # uneven loads must not reorder
        return case["case_id"].upper()

    seen = list(iterate_cases(cases, slow_upper, prefetch=3, threads=threads))
    assert seen == [(case, case["case_id"].upper()) for case in cases]
    assert list(iterate_cases(cases, lambda case: 1, prefetch=0)) == [(case, 1) for case in cases]

    def loader(case):
        if case["case_id"] == "c2":
            raise ValueError("boom")
        return 0

    with pytest.raises(ValueError, match="boom"):
        list(iterate_cases(cases, loader, prefetch=3, threads=threads))


def test_batches_mix_cases_and_are_seed_reproducible(tmp_path, capsys):
    from ml.train import main as train

    manifest_path = create_fixture(tmp_path / "fixture")
    cache_dir = tmp_path / "cache"
    common = ["--manifest", str(manifest_path), "--epochs", "2", "--patches-per-case", "6", "--batch-size", "4",
              "--patch-size", "16", "16", "16", "--channels", "4", "8", "16", "--threads", "2", "--device", "cpu",
              "--cache-dir", str(cache_dir), "--cache-workers", "1", "--loader-threads", "2", "--cosine"]
    train([*common, "--output", str(tmp_path / "a")])
    first = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    train([*common, "--output", str(tmp_path / "b")])
    second = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    steps_a = [event for event in first if event["event"] == "train_step"]
    steps_b = [event for event in second if event["event"] == "train_step"]
    # 1 train case x 6 patches / batch 4 -> 2 steps per epoch (4 + 2), two epochs.
    assert [step["batch"] for step in steps_a] == [4, 2, 4, 2]
    assert [round(step["loss"], 6) for step in steps_a] == [round(step["loss"], 6) for step in steps_b]
    start = next(event for event in first if event["event"] == "train_start")
    assert start["batch_size"] == 4 and start["steps_per_epoch"] == 2 and start["lr_schedule"] == "cosine_to_1pct"
    assert steps_a[-1]["lr"] < steps_a[0]["lr"]
    metadata = json.loads((tmp_path / "a" / "metadata.json").read_text())
    assert metadata["batch_size"] == 4 and metadata["lr_schedule"] == "cosine_to_1pct" and metadata["global_step"] == 4


def test_available_model_reports_training_summary_from_checkpoint(tmp_path, monkeypatch, capsys):
    from ml.adapters import checkpoint_summary, describe_models
    from ml.train import main as train

    manifest_path = create_fixture(tmp_path / "fixture")
    train(["--manifest", str(manifest_path), "--epochs", "1", "--max-steps", "1", "--patches-per-case", "1",
           "--patch-size", "16", "16", "16", "--channels", "4", "8", "16", "--threads", "2", "--device", "cpu",
           "--output", str(tmp_path / "run")])
    capsys.readouterr()
    checkpoint = tmp_path / "run" / "best.pt"
    monkeypatch.setenv("MRI_UNET_CHECKPOINT", str(checkpoint))
    unet = next(model for model in describe_models() if model["id"] == "unet3d")
    assert unet["available"] is True
    summary = unet["training"]
    assert summary["validated_epoch"] == 1 and summary["global_step"] == 1
    assert summary["train_cases"] == 1 and summary["val_cases"] == 1
    assert 0 <= summary["validation_mean_dice"] <= 1
    assert checkpoint_summary(checkpoint) is summary, "second read is served from the cache"
    assert checkpoint_summary(tmp_path / "missing.pt") == {}
    (tmp_path / "junk.pt").write_bytes(b"not a checkpoint")
    assert checkpoint_summary(tmp_path / "junk.pt") == {}


def test_max_steps_stops_the_batch_producer_and_no_validation_skips_best(tmp_path, capsys):
    import threading

    from ml.train import main as train

    manifest_path = create_fixture(tmp_path / "fixture")
    before = threading.active_count()
    train(["--manifest", str(manifest_path), "--epochs", "3", "--max-steps", "2", "--patches-per-case", "8", "--batch-size", "2",
           "--patch-size", "16", "16", "16", "--channels", "4", "8", "16", "--threads", "2", "--device", "cpu",
           "--cache-dir", str(tmp_path / "cache"), "--cache-workers", "1", "--loader-threads", "3", "--prefetch", "4",
           "--no-validation", "--output", str(tmp_path / "run")])
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    steps = [event for event in events if event["event"] == "train_step"]
    assert len(steps) == 2
    epoch = next(event for event in events if event["event"] == "epoch_complete")
    assert epoch["validated"] is False and epoch["validation_seconds"] == 0.0 and epoch["train_seconds"] >= 0
    assert (tmp_path / "run" / "last.pt").is_file() and not (tmp_path / "run" / "best.pt").exists()
    # Producer/loader threads must not linger after the early stop.
    for _ in range(50):
        if threading.active_count() <= before:
            break
        import time
        time.sleep(0.1)
    assert threading.active_count() <= before


def test_training_from_cache_matches_metadata_and_reuses_entries(tmp_path, capsys):
    from ml.train import main as train

    manifest_path = create_fixture(tmp_path / "fixture")
    cache_dir = tmp_path / "cache"
    common = ["--manifest", str(manifest_path), "--epochs", "1", "--max-steps", "1", "--patches-per-case", "1",
              "--patch-size", "16", "16", "16", "--channels", "4", "8", "16", "--threads", "2", "--device", "cpu",
              "--cache-dir", str(cache_dir), "--cache-workers", "1"]
    train([*common, "--output", str(tmp_path / "run1")])
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    ready = next(event for event in events if event["event"] == "cache_ready")
    assert ready["built"] == 2 and ready["fresh"] == 0
    metadata = json.loads((tmp_path / "run1" / "metadata.json").read_text())
    assert metadata["preprocessing_cache"]["image_dtype"] == "float16"
    assert Path(metadata["preprocessing_cache"]["dir"]) == cache_dir.resolve()
    assert metadata["global_step"] == 1 and metadata["validation_mean_dice"] is not None
    train([*common, "--output", str(tmp_path / "run2")])
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    ready = next(event for event in events if event["event"] == "cache_ready")
    assert ready["built"] == 0 and ready["fresh"] == 2
    assert not any(event["event"] == "cache_miss" for event in events)
