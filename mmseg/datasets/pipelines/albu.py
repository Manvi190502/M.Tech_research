import albumentations as A
import numpy as np
from mmseg.datasets.builder import PIPELINES

@PIPELINES.register_module()
class Albu:
    def __init__(self):
        self.aug = A.Compose([
            # 🔥 UNDERWATER-SPECIFIC AUGMENTATIONS

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),

            A.RandomRotate90(p=0.5),

            # Blur (VERY IMPORTANT)
            A.GaussianBlur(blur_limit=(3,7), p=0.5),

            # Noise (simulates particles)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),

            # Color shift (underwater effect)
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.5
            ),

            # Contrast enhancement
            A.CLAHE(clip_limit=2.0, p=0.5),

            # Random brightness
            A.RandomBrightnessContrast(p=0.5),
        ])

    def __call__(self, results):
        img = results['img']
        mask = results['gt_semantic_seg']

        augmented = self.aug(image=img, mask=mask)

        results['img'] = augmented['image']
        results['gt_semantic_seg'] = augmented['mask']

        return results