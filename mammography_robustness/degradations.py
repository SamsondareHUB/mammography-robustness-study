"""Test-time input corruptions for the CDD-CESM robustness protocol.

CPU float32 operations; clean preprocessing remains delegated to preprocessing.py.
"""
from __future__ import annotations

import hashlib
import math
from types import MappingProxyType

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn.functional import interpolate
from torchvision.transforms.functional import normalize, to_tensor
from scipy.ndimage import gaussian_filter

from .preprocessing import IMAGENET_MEAN, IMAGENET_STD, PadToSize, make_transform

SEVERITIES = ("low", "medium", "high")
PARAMETERS = MappingProxyType({
    "noise": (0.02, 0.05, 0.10),
    "blur": (0.5, 1.0, 2.0),
    "contrast": (0.8, 0.6, 0.4),
    "resolution": (0.75, 0.50, 0.25),
})
CONDITIONS = (("clean", "none"),) + tuple(
    (name, severity) for name in PARAMETERS for severity in SEVERITIES
)


def noise_seed(case_id: str) -> int:
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a nonempty manifest string")
    return int.from_bytes(hashlib.sha256(("314159:" + case_id).encode("utf-8")).digest(), "big", signed=False)


def content_box(image: Image.Image, size: int = 224) -> tuple[int, int, int, int]:
    """Get exact Pillow padding geometry using a nonzero constant image.

    This avoids inferring padding from black source pixels or duplicating
    Pillow's version-dependent rounding rules. Returns left, top, right, bottom.
    """
    mask = Image.new("L", image.size, 255)
    padded = ImageOps.pad(mask, (size, size), method=Image.Resampling.BILINEAR, color=0)
    box = padded.getbbox()
    if box is None:
        raise ValueError("empty resized image content")
    return box


def input_hash(tensor: torch.Tensor) -> str:
    """SHA-256 of contiguous little-endian float32 CHW bytes (shape separate)."""
    if tensor.ndim != 3 or tensor.shape[0] != 3 or tensor.dtype != torch.float32:
        raise ValueError("expected float32 CHW RGB tensor")
    array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def corrupted_input(image: Image.Image, case_id: str, degradation: str,
                    severity: str, *, size: int = 224) -> torch.Tensor:
    """Return CPU float32 CHW input before normalization; preserve black padding."""
    if (degradation, severity) not in CONDITIONS:
        raise ValueError(f"unknown condition: {degradation}/{severity}")
    if type(size) is not int or size != 224:
        raise ValueError("frozen baseline requires image size 224")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a nonempty manifest string")
    if degradation == "clean":
        return make_transform(size, training=False, normalization="none")(image)
    clean = to_tensor(PadToSize(size)(image)).to(torch.float32)
    left, top, right, bottom = content_box(image, size)
    content = clean[:, top:bottom, left:right].contiguous()
    _, height, width = content.shape
    parameter = PARAMETERS[degradation][SEVERITIES.index(severity)]
    if degradation == "noise":
        rng = np.random.Generator(np.random.PCG64(noise_seed(case_id)))
        field = rng.standard_normal(size=(height, width), dtype=np.float32)
        result = content + torch.from_numpy(field * np.float32(parameter)).unsqueeze(0)
    elif degradation == "blur":
        result = torch.from_numpy(gaussian_filter(content.numpy(),
            sigma=(0.0, parameter, parameter), mode="reflect", truncate=4.0))
    elif degradation == "contrast":
        mean = content.mean(dim=(1, 2), keepdim=True)
        result = mean + parameter * (content - mean)
    else:
        target = (max(1, math.floor(height * parameter + 0.5)),
                  max(1, math.floor(width * parameter + 0.5)))
        down = interpolate(content.unsqueeze(0), size=target, mode="bilinear",
                           align_corners=False, antialias=True)
        result = interpolate(down, size=(height, width), mode="bilinear",
                             align_corners=False, antialias=False).squeeze(0)
    output = clean.clone()
    output[:, top:bottom, left:right] = result.clamp(0, 1)
    return output.contiguous()


def model_input(image: Image.Image, case_id: str, degradation: str,
                severity: str, *, size: int = 224) -> tuple[torch.Tensor, str]:
    """Return normalized input and hash of its pre-normalization tensor.

    The clean branch calls the baseline transform unchanged; this is not a
    replacement for real checkpoint-based clean prediction verification.
    """
    raw = corrupted_input(image, case_id, degradation, severity, size=size)
    digest = input_hash(raw)
    if degradation == "clean":
        return make_transform(size, training=False, normalization="imagenet")(image), digest
    return normalize(raw, IMAGENET_MEAN, IMAGENET_STD), digest
