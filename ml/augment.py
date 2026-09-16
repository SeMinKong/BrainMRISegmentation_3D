"""GPU batch augmentation for training patches. Nothing here touches validation or inference.

Spatial: one random rotation (about a random axis) and isotropic scale per sample, applied with
`grid_sample` — trilinear for the four MRI channels, nearest for the integer label map, so no new
label values appear. Intensity: per-sample brightness/contrast/gamma, Gaussian noise and a light blur
on a random subset of channels. All parameters are drawn from a torch Generator seeded by the caller,
so a run is reproducible for a given seed.
"""
from __future__ import annotations

import math


def augment_batch(inputs, targets, generator, *, max_rotation_deg: float = 15.0, scale_range=(0.9, 1.1),
                  p_spatial: float = 0.8, p_intensity: float = 0.8):
    """inputs [B,C,X,Y,Z] float, targets [B,1,X,Y,Z] integer. Returns augmented copies on the same device."""
    import torch
    import torch.nn.functional as F

    batch, channels = inputs.shape[0], inputs.shape[1]
    device = inputs.device

    # Draw on the generator's own device: a CUDA generator keeps the 32M-element noise tensor on the GPU
    # instead of generating it on the CPU and copying it every step (which halved throughput).
    def rand(*shape):
        return torch.rand(*shape, generator=generator, device=generator.device).to(device)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device=generator.device).to(device)

    # ---- spatial: rotation about a random unit axis + isotropic scale, as a 3x4 affine in normalised coords
    spatial = rand(batch) < p_spatial
    if bool(spatial.any()):
        angle = (rand(batch) * 2 - 1) * math.radians(max_rotation_deg)
        axis = randn(batch, 3)
        axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-6)
        scale = scale_range[0] + rand(batch) * (scale_range[1] - scale_range[0])
        cos, sin = torch.cos(angle), torch.sin(angle)
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        one_minus = 1 - cos
        rotation = torch.stack([
            torch.stack([cos + x * x * one_minus, x * y * one_minus - z * sin, x * z * one_minus + y * sin], dim=1),
            torch.stack([y * x * one_minus + z * sin, cos + y * y * one_minus, y * z * one_minus - x * sin], dim=1),
            torch.stack([z * x * one_minus - y * sin, z * y * one_minus + x * sin, cos + z * z * one_minus], dim=1),
        ], dim=1)  # [B,3,3]
        # grid_sample maps output coords -> input coords, so use the inverse scale (sampling a bigger
        # region shrinks the content) and the transposed rotation.
        theta = rotation.transpose(1, 2) / scale.view(batch, 1, 1)
        theta = torch.cat([theta, torch.zeros(batch, 3, 1, device=device)], dim=2)
        identity = torch.eye(3, 4, device=device).expand(batch, 3, 4).clone()
        theta = torch.where(spatial.view(batch, 1, 1), theta, identity)
        grid = F.affine_grid(theta, list(inputs.shape), align_corners=False)
        inputs = F.grid_sample(inputs, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        targets = F.grid_sample(targets.to(inputs.dtype), grid, mode="nearest", padding_mode="zeros", align_corners=False).round().to(torch.long)

    # ---- intensity: per sample & channel; z-scored inputs so offsets are in standard deviations
    apply = (rand(batch, channels, 1, 1, 1) < p_intensity).to(inputs.dtype)
    contrast = 1 + (rand(batch, channels, 1, 1, 1) * 2 - 1) * 0.15
    brightness = (rand(batch, channels, 1, 1, 1) * 2 - 1) * 0.15
    inputs = inputs * (1 + (contrast - 1) * apply) + brightness * apply
    # gamma on the positive part only (background stays 0); map to [0,1] per channel first
    gamma = torch.exp((rand(batch, channels, 1, 1, 1) * 2 - 1) * math.log(1.4))
    positive = inputs.clamp_min(0)
    peak = positive.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    gammaed = (positive / peak).pow(gamma) * peak
    inputs = torch.where(inputs > 0, gammaed * apply + positive * (1 - apply), inputs)
    noise_sigma = rand(batch, channels, 1, 1, 1) * 0.08
    inputs = inputs + randn(*inputs.shape) * noise_sigma * apply
    # light blur on ~a third of the augmented channels: 3-tap separable box via avg_pool3d
    blur = (rand(batch, channels, 1, 1, 1) < 0.3).to(inputs.dtype) * apply
    if bool(blur.any()):
        blurred = F.avg_pool3d(inputs, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        inputs = inputs * (1 - blur) + blurred * blur
    return inputs, targets
