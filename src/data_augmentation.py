"""
Class Data Augmentation operations to images.
"""

import random
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance


# TODO (1): Implement logging


class RandAugment:
    """Apply RandAugment to the image.

    Args:
        num_operations (int): Number of operations to apply.
        magnitude (int): Magnitude of the operations.
        img_size (int, optional): Size of the image. Defaults to 32.

    Returns:
        Image: Augmented image.

    Example:
        augmenter = RandAugment(num_operations=3, magnitude=5)
        augmented_image = augmenter(image)
    """

    def __init__(self, num_operations: int, magnitude: int, img_size: int = 32):
        assert num_operations >= 1
        assert 1 <= magnitude <= 10
        self.num_operations = num_operations
        self.magnitude = magnitude
        self.img_size = img_size
        self.augment_pool = self._augment_pool()

    def __call__(self, img: Image.Image, label: Image.Image) -> Tuple[Image.Image, Image.Image]:
        operations = random.choices(self.augment_pool, k=self.num_operations)
        for operation, max_value, bias in operations:
            value = np.random.randint(1, self.magnitude)
            if random.random() < 0.5:
                if operation in [RandAugment._brightness, RandAugment._color, RandAugment._contrast,
                                 RandAugment._sharpness]:
                    img = operation(img, v=value, max_v=max_value, bias=bias)
                else:
                    img, label = operation(img, label, v=value, max_v=max_value, bias=bias)
        img, label = self._cutout_abs(img, label, int(self.img_size * 0.08))
        return img, label

    @staticmethod
    def _augment_pool() -> List[Tuple[Callable, float, float]]:
        """Return a pool of augmentations."""
        return [
            (RandAugment._brightness, 0.9, 0.05),
            (RandAugment._color, 0.1, 0.05),
            (RandAugment._contrast, 0.1, 0.05),
            (RandAugment._identity, None, None),
            (RandAugment._rotate, 10, 0),
            (RandAugment._sharpness, 0.1, 0.05),
            (RandAugment._shear_x, 0.1, 0),
            (RandAugment._shear_y, 0.1, 0),
            (RandAugment._translate_x, 0.1, 0),
            (RandAugment._translate_y, 0.1, 0),
        ]

    # Define the image operations here as static methods, e.g.,
    @staticmethod
    def _brightness(img: Image, v: float, max_v: float, bias: float = 0) -> Image:
        """Change image brightness."""
        v = 1 - RandAugment._float_parameter(v, max_v) + bias
        return ImageEnhance.Brightness(img).enhance(v)

    @staticmethod
    def _color(img: Image, v: float, max_v: float, bias: float = 0) -> Image:
        """Change image color."""
        v = 1 - RandAugment._float_parameter(v, max_v) + bias
        return ImageEnhance.Color(img).enhance(v)

    @staticmethod
    def _contrast(img: Image, v: float, max_v: float, bias: float = 0) -> Image:
        """Change image contrast."""
        v = 1 - RandAugment._float_parameter(v, max_v) + bias
        return ImageEnhance.Contrast(img).enhance(v)

    @staticmethod
    def _identity(img: Image) -> Image:
        """Return the original image."""
        return img

    @staticmethod
    def _rotate(img: Image, v: float, max_v: float, bias: float = 0) -> Image:
        """Rotate the image."""
        v = RandAugment._int_parameter(v, max_v) + bias
        if random.random() < 0.5:
            v = -v
        return img.rotate(v)

    @staticmethod
    def _sharpness(img: Image, v: float, max_v: float, bias: float = 0) -> Image:
        """Change image sharpness."""
        v = 1 - RandAugment._float_parameter(v, max_v) + bias
        return ImageEnhance.Sharpness(img).enhance(v)

    @staticmethod
    def _shear_x(img: Image, v: float, max_v: float, bias: float = 0) -> Image:
        """Shear the image along the X-axis."""
        v = RandAugment._float_parameter(v, max_v) + bias
        if random.random() < 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))

    @staticmethod
    def _shear_y(img: Image, v: float, max_v: float, bias: float = 0) -> Image:
        """Shear the image along the Y-axis."""
        v = RandAugment._float_parameter(v, max_v) + bias
        if random.random() < 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))

    @staticmethod
    def _translate_x(img: Image, v: float, max_v: float, bias: float = 0) -> Image:
        """Translate the image along the X-axis."""
        v = RandAugment._float_parameter(v, max_v) + bias
        if random.random() < 0.5:
            v = -v
        v = int(v * img.size[0])
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

    @staticmethod
    def _translate_y(img: Image, v: float, max_v: float, bias: float = 0) -> Image:
        """Translate the image along the Y-axis."""
        v = RandAugment._float_parameter(v, max_v) + bias
        if random.random() < 0.5:
            v = -v
        v = int(v * img.size[1])
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))

    @staticmethod
    def _cutout_abs(img: Image, size: int) -> Image:
        """Apply cutout augmentation to the image with absolute size."""
        img = img.copy()
        width, height = img.size
        left = np.random.randint(0, width - size)
        upper = np.random.randint(0, height - size)
        img.paste((128, 128, 128), (left, upper, left + size, upper + size))
        return img

    @staticmethod
    def _float_parameter(v: float, max_v: float) -> float:
        """Convert parameter to float."""
        return float(v) * max_v / 10

    @staticmethod
    def _int_parameter(v: int, max_v: int) -> int:
        """Convert parameter to integer."""
        return int(v * max_v / 10)
