"""Shared, explicitly versioned CDD-CESM image preprocessing."""

from __future__ import annotations

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PadToSize:
    """Aspect-preserving RGB padding; defined at module level for worker portability."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, image):
        from PIL import Image, ImageOps

        return ImageOps.pad(
            image.convert("RGB"), (self.size, self.size),
            method=Image.Resampling.BILINEAR, color=(0, 0, 0),
        )


def make_transform(
    size: int, *, training: bool = False, normalization: str = "imagenet",
    flip_probability: float = 0.5, rotation_degrees: float = 10.0,
):
    """Pad full RGB image without cropping, then augment only training inputs."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if type(size) is not int or size < 64:
        raise ValueError("image size must be an integer >= 64")
    if normalization not in {"imagenet", "none"}:
        raise ValueError("normalization must be imagenet or none")
    if not 0 <= flip_probability <= 1 or not 0 <= rotation_degrees <= 180:
        raise ValueError("invalid augmentation parameters")

    steps = [PadToSize(size)]
    if training:
        steps += [
            transforms.RandomHorizontalFlip(p=flip_probability),
            transforms.RandomRotation(
                degrees=rotation_degrees, interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
        ]
    steps.append(transforms.ToTensor())
    if normalization == "imagenet":
        steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)
