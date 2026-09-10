"""Network factory; optional dependencies are imported only when requested."""
from __future__ import annotations

DEFAULT_SETTINGS = {"channels": [16, 32, 64, 128, 256], "num_res_units": 2, "feature_size": 24}


def build_model(model_id: str, num_classes: int, settings: dict | None = None):
    from monai.networks.nets import UNet, SwinUNETR

    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    if model_id == "unet3d":
        channels = tuple(settings["channels"])
        return UNet(spatial_dims=3, in_channels=4, out_channels=num_classes,
                    channels=channels, strides=(2,) * (len(channels) - 1),
                    num_res_units=int(settings["num_res_units"]), norm="INSTANCE")
    if model_id == "swinunetr":
        return SwinUNETR(in_channels=4, out_channels=num_classes,
                        feature_size=int(settings["feature_size"]),
                        spatial_dims=3, use_checkpoint=True)
    raise ValueError(f"Unsupported network: {model_id}")


def validate_patch_size(model_id: str, patch_size, settings=None):
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    divisor = 32 if model_id == "swinunetr" else 2 ** (len(settings["channels"]) - 1)
    if len(patch_size) != 3 or any(size % divisor != 0 or size <= divisor for size in patch_size):
        raise ValueError(f"{model_id} patch dimensions must be multiples of {divisor} and greater than {divisor}")
