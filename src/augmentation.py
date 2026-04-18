"""Data augmentation for digit images.

Geometric transformations (rotation, translation, scaling) applied to
training images to improve model generalization.
"""

import numpy as np
from typing import Literal

try:
    from skimage.transform import rotate, AffineTransform, warp
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def augment_digit(
    img_vector: np.ndarray,
    n_augment: int = 3,
    strength: Literal["mild", "medium", "strong"] = "mild",
    seed: int | None = None,
) -> np.ndarray:
    """Apply geometric augmentation to a single digit image.

    Args:
        img_vector: Flattened 784-dim image vector.
        n_augment: Number of augmented copies to generate.
        strength: Augmentation intensity.
            - mild:   rotation ±5°,  translation ±1px, scale 0.95–1.05
            - medium: rotation ±10°, translation ±2px, scale 0.90–1.10
            - strong: rotation ±15°, translation ±3px, scale 0.85–1.15
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_augment, 784) with augmented image vectors.
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image is required for augmentation. Install with: uv add scikit-image"
        )

    if seed is not None:
        np.random.seed(seed)

    img = img_vector.reshape((28, 28), order="F")
    augmented = []

    if strength == "mild":
        rot_range, shift_range, scale_range = 5, 1, 0.05
    elif strength == "medium":
        rot_range, shift_range, scale_range = 10, 2, 0.1
    else:
        rot_range, shift_range, scale_range = 15, 3, 0.15

    for _ in range(n_augment):
        aug_img = img.copy().astype(np.float64)

        # 1. Random rotation
        angle = np.random.uniform(-rot_range, rot_range)
        aug_img = rotate(aug_img, angle, mode="edge", preserve_range=True)

        # 2. Random translation
        tx = np.random.uniform(-shift_range, shift_range)
        ty = np.random.uniform(-shift_range, shift_range)
        transform = AffineTransform(translation=(tx, ty))
        aug_img = warp(aug_img, transform, mode="edge", preserve_range=True)

        # 3. Random scaling
        scale = 1 + np.random.uniform(-scale_range, scale_range)
        transform = AffineTransform(scale=(scale, scale))
        aug_img = warp(aug_img, transform, mode="edge", preserve_range=True)

        # 4. Strong: noise + brightness
        if strength == "strong":
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 3, aug_img.shape)
                aug_img = aug_img + noise
                aug_img = np.clip(aug_img, 0, 255)
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.85, 1.15)
                aug_img = aug_img * brightness
                aug_img = np.clip(aug_img, 0, 255)

        augmented.append(aug_img.flatten(order="F"))

    return np.array(augmented)


def augment_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_augment: int = 3,
    strength: Literal["mild", "medium", "strong"] = "mild",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment an entire training set.

    Args:
        X_train: Training features, shape (n_samples, 784).
        y_train: Training labels, shape (n_samples,).
        n_augment: Number of augmented copies per sample.
        strength: Augmentation intensity.
        seed: Random seed.

    Returns:
        Tuple (X_aug, y_aug) where the original samples are prepended
        followed by n_augment copies per sample.
    """
    n_samples = len(X_train)
    total = n_samples * (1 + n_augment)

    X_aug = np.zeros((total, X_train.shape[1]), dtype=X_train.dtype)
    y_aug = np.zeros(total, dtype=y_train.dtype)

    X_aug[:n_samples] = X_train
    y_aug[:n_samples] = y_train

    for i in range(n_samples):
        aug = augment_digit(
            X_train[i], n_augment=n_augment, strength=strength, seed=seed + i
        )
        start = n_samples + i * n_augment
        end = start + n_augment
        X_aug[start:end] = aug
        y_aug[start:end] = y_train[i]

    return X_aug, y_aug
