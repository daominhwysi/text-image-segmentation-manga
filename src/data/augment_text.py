import numpy as np
from PIL import Image
import albumentations as A
import cv2


def augment_text_image(image):
    """Applies common text augmentations: rotation, noise, and blur."""
    # Convert PIL Image to NumPy array (Albumentations requirement)
    img_array = np.array(image)

    # Define the augmentation pipeline

    transform = A.Compose(
        [
            A.Affine(
                scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                translate_percent=[-0.05, 0.05],
                rotate=[-30, 30],
                shear=[-10, 10],
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                fit_output=True,
                keep_ratio=False,
                rotate_method="ellipse",
                balanced_scale=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.Perspective(
                scale=[0.05, 0.1],
                keep_size=True,
                fit_output=True,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            # Replacement for PiecewiseAffine
            # alpha: Scaling factor for the displacement (higher = more distortion)
            # sigma: Smoothness of the displacement (higher = smoother waves)
            # alpha_affine: Scaling factor for the internal affine transform
            A.ElasticTransform(
                alpha=250,
                sigma=10,
                interpolation=cv2.INTER_LANCZOS4,
                approximate=False,
                same_dxdy=True,
                mask_interpolation=cv2.INTER_NEAREST,
                noise_distribution="gaussian",
                keypoint_remapping_method="mask",
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
        ]
    )

    augmented = transform(image=img_array)["image"]

    # Convert back to PIL for easy viewing/saving
    return Image.fromarray(augmented)


def augment_output_image(image):
    img_array = np.array(image)

    transform = A.Compose(
        [
            A.ColorJitter(
                brightness=[0.8, 1.2],
                contrast=[0.8, 1.2],
                saturation=[0.8, 1.2],
                hue=[-0.5, 0.5],
            )
        ]
    )

    augmented = transform(image=img_array)["image"]
    return Image.fromarray(augmented)
