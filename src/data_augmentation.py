"""
Module providing data augmentation operations for images using RandAugment.

This module implements the RandAugment algorithm, which applies a series
of image transformations to enhance the diversity of training data. The main class provided is:

- `RandAugment`: Applies a sequence of randomly chosen augmentation operations to images.
The operations include brightness, color, contrast adjustments, rotation,
sharpness enhancement, shearing, translation, and cutout.

Classes:
- `RandAugment`:
    - **Args**:
        - `num_operations` (int): Number of augmentation operations to apply to each image.
        - `magnitude` (int): Magnitude or strength of the augmentation operations.
        - `img_size` (int, optional): Size of the image used for cutout augmentation. Defaults to 32.
    - **Returns**:
        - Tuple[Image, Image]: Augmented image and corresponding label.

    - **Methods**:
        - `__init__(num_operations, magnitude, img_size)`: Initializes the RandAugment with specified operations and magnitude.
        - `__call__(img, label)`: Applies the augmentations to the input image and label.
        - `_augment_pool()`: Returns a list of possible augmentation operations.
        - Augmentation operations such as `_brightness`, `_color`, `_contrast`, `_rotate`,
        `_sharpness`, `_shear_x`, `_shear_y`, `_translate_x`, `_translate_y`, and `_cutout_abs` perform various image transformations.

Usage:
    >>> augmenter = RandAugment(num_operations=3, magnitude=5)
    >>> augmented_image, augmented_label = augmenter(image, label)

Notes:
    - Ensure to use `PIL` and `numpy` libraries for image and numerical operations.
    - The `cutout` operation masks a random area of the image with a solid color.

Not implemented:
    - Logging functionality is not implemented in this module.
    - It's not being included in the current training pipeline !

"""

import random
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance


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
