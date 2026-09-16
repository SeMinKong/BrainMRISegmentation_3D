import numpy as np
import pytest

from ml.data import sample_patches

torch = pytest.importorskip("torch")
from ml.augment import augment_batch  # noqa: E402


def test_augment_keeps_shapes_labels_and_is_seed_reproducible():
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(2, 4, 16, 16, 16)
    targets = torch.randint(0, 5, (2, 1, 16, 16, 16))
    out_a, lab_a = augment_batch(inputs.clone(), targets.clone(), generator)
    assert out_a.shape == inputs.shape and lab_a.shape == targets.shape
    assert lab_a.dtype == torch.long
    assert set(lab_a.unique().tolist()).issubset({0, 1, 2, 3, 4}), "nearest resampling must not invent labels"
    assert torch.isfinite(out_a).all()
    assert not torch.equal(out_a, inputs), "augmentation should change the images"
    out_b, lab_b = augment_batch(inputs.clone(), targets.clone(), torch.Generator().manual_seed(7))
    assert torch.equal(out_a, out_b) and torch.equal(lab_a, lab_b)


def test_augment_leaves_background_zero_where_no_content_moved_in():
    generator = torch.Generator().manual_seed(1)
    inputs = torch.zeros(1, 4, 16, 16, 16)
    targets = torch.zeros(1, 1, 16, 16, 16, dtype=torch.long)
    out, lab = augment_batch(inputs, targets, generator)
    assert lab.sum() == 0
    # noise is only added where the intensity path applies; an all-zero volume stays near zero everywhere
    assert out.abs().max() < 0.5


def test_balanced_sampling_centres_small_labels_far_more_often():
    rng = np.random.default_rng(3)
    label = np.zeros((40, 40, 40), dtype=np.int64)
    label[4:36, 4:36, 4:36] = 2          # big class 2
    label[30:33, 30:33, 30:33] = 1       # tiny class 1: 27 voxels vs 32768
    image = np.random.default_rng(0).normal(size=(4, *label.shape)).astype(np.float32)
    def centre_labels(balanced):
        hits = 0
        for _ in range(200):
            patch, target = sample_patches(image, label, (8, 8, 8), rng, 1, balanced=balanced)[0]
            assert patch.shape == (4, 8, 8, 8) and target.shape == (1, 8, 8, 8)
            if (target == 1).any():
                hits += 1
        return hits
    proportional = centre_labels(False)
    balanced = centre_labels(True)
    assert balanced > proportional * 3, (proportional, balanced)
